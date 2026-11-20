# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Adapted from https://github.com/divelab/AIRS (OpenMat/HIENet)

import math
import random
from typing import Optional, Tuple, Union

from .. import _keys as KEY
import numpy as np
import paddle
from ..e3nn import o3
from ..e3nn.o3 import Irreps, SphericalHarmonics
from ..e3nn.util.jit import compile_mode
from .._const import AtomGraphDataType
from .. import paddle_compat
from ppmat.models.common.message_passing.message_passing import MessagePassing
from ..paddle_compat import Adj, OptTensor, PairTensor
from ppmat.utils.scatter import scatter


@compile_mode("script")
class EdgePreprocess(paddle.nn.Module):
    """
    preprocessing pos to edge vectors and edge lengths
    """

    def __init__(self, is_stress):
        super().__init__()
        self.is_stress = is_stress
        self._is_batch_data = True

    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:
        if self._is_batch_data:
            cell = data[KEY.CELL].reshape(-1, 3, 3)
        else:
            cell = data[KEY.CELL].reshape(3, 3)
        cell_shift = data[KEY.CELL_SHIFT]
        # ASE pbc shifts arrive as int64; align to the cell's float dtype
        # before the matmul.
        cell_shift = cell_shift.to(cell.dtype)
        pos = data[KEY.POS]
        batch = data[KEY.BATCH]
        if self.is_stress:
            # Reuse an externally provided strain leaf (HIENet main class);
            # only create one when absent (legacy single-structure path).
            prev_strain = data["_strain"] if "_strain" in data else None
            if self._is_batch_data:
                if prev_strain is None:
                    # Graph-safe num_graphs from a static shape: no
                    # data-dependent host read on the batch vector.
                    num_batch = data[KEY.NUM_ATOMS].shape[0]
                    strain = paddle.zeros(
                        (num_batch, 3, 3), dtype=pos.dtype, device=pos.device
                    )
                    strain.requires_grad_(True)
                    data["_strain"] = strain
                else:
                    strain = prev_strain
                sym_strain = 0.5 * (strain + strain.transpose(-1, -2))
                pos = pos + paddle.bmm(pos.unsqueeze(-2), sym_strain[batch]).squeeze(-2)
                cell = cell + paddle.bmm(cell, sym_strain)
            else:
                if prev_strain is None:
                    strain = paddle.zeros((3, 3), dtype=pos.dtype, device=pos.device)
                    strain.requires_grad_(True)
                    data["_strain"] = strain
                else:
                    strain = prev_strain
                sym_strain = 0.5 * (strain + strain.transpose(-1, -2))
                pos = pos + paddle.mm(pos, sym_strain)
                cell = cell + paddle.mm(cell, sym_strain)
        idx_src = data[KEY.EDGE_IDX][0]
        idx_dst = data[KEY.EDGE_IDX][1]
        edge_vec = pos[idx_dst] - pos[idx_src]
        if self._is_batch_data:
            # einsum "ni,nij->nj" -> bmm (ppmat utils/crystal.py precedent)
            edge_vec = edge_vec + paddle.bmm(
                cell_shift.unsqueeze(1), cell[batch[idx_src]]
            ).squeeze(1)
        else:
            # einsum "ni,ij->nj" -> mm
            edge_vec = edge_vec + paddle.mm(cell_shift, cell.squeeze(0))
        data[KEY.EDGE_VEC] = edge_vec
        data[KEY.EDGE_LENGTH] = paddle.sqrt((edge_vec * edge_vec).sum(-1))
        return data


class BesselBasis(paddle.nn.Module):
    """
    f : (*, 1) -> (*, bessel_basis_num)
    """

    def __init__(
        self,
        cutoff_length: float,
        bessel_basis_num: int = 8,
        trainable_coeff: bool = True,
    ):
        super().__init__()
        self.num_basis = bessel_basis_num
        self.prefactor = 2.0 / cutoff_length
        self.coeffs = paddle.FloatTensor(
            [(n * math.pi / cutoff_length) for n in range(1, bessel_basis_num + 1)]
        )
        if trainable_coeff:
            self.coeffs = paddle.nn.Parameter(self.coeffs)

    def forward(self, r: paddle.Tensor) -> paddle.Tensor:
        ur = r.unsqueeze(-1)
        return self.prefactor * paddle.sin(self.coeffs * ur) / ur


class PolynomialCutoff(paddle.nn.Module):
    """
    f : (*, 1) -> (*, 1)
    https://arxiv.org/pdf/2003.03123.pdf
    """

    def __init__(self, cutoff_length: float, poly_cut_p_value: int = 6):
        super().__init__()
        p = poly_cut_p_value
        self.cutoff_length = cutoff_length
        self.p = p
        self.coeff_p0 = (p + 1.0) * (p + 2.0) / 2.0
        self.coeff_p1 = p * (p + 2.0)
        self.coeff_p2 = p * (p + 1.0) / 2.0

    def forward(self, r: paddle.Tensor) -> paddle.Tensor:
        r = r / self.cutoff_length
        return (
            1
            - self.coeff_p0 * paddle.pow(r, self.p)
            + self.coeff_p1 * paddle.pow(r, self.p + 1.0)
            - self.coeff_p2 * paddle.pow(r, self.p + 2.0)
        )


class XPLORCutoff(paddle.nn.Module):
    """
    https://hoomd-blue.readthedocs.io/en/latest/module-md-pair.html
    """

    def __init__(self, cutoff_length: float, cutoff_on: float):
        super().__init__()
        self.r_on = cutoff_on
        self.r_cut = cutoff_length
        assert self.r_on < self.r_cut

    def forward(self, r: paddle.Tensor) -> paddle.Tensor:
        r_sq = r * r
        r_on_sq = self.r_on * self.r_on
        r_cut_sq = self.r_cut * self.r_cut
        return paddle.where(
            r < self.r_on,
            1.0,
            (r_cut_sq - r_sq) ** 2
            * (r_cut_sq + 2 * r_sq - 3 * r_on_sq)
            / (r_cut_sq - r_on_sq) ** 3,
        )


class CosineCutoff(paddle.nn.Module):
    """Cosine cutoff function."""

    def __init__(self, cutoff_lower=0.0, cutoff_upper=5.0):
        super(CosineCutoff, self).__init__()
        self.cutoff_lower = cutoff_lower
        self.cutoff_upper = cutoff_upper

    def forward(self, distances):
        """Compute the cutoff function."""
        if self.cutoff_lower > 0:
            cutoffs = 0.5 * (
                paddle.cos(
                    math.pi
                    * (
                        2
                        * (distances - self.cutoff_lower)
                        / (self.cutoff_upper - self.cutoff_lower)
                        + 1.0
                    )
                )
                + 1.0
            )
            cutoffs = cutoffs * (distances < self.cutoff_upper).float()
            cutoffs = cutoffs * (distances > self.cutoff_lower).float()
            return cutoffs
        else:
            cutoffs = 0.5 * (paddle.cos(distances * math.pi / self.cutoff_upper) + 1.0)
            cutoffs = cutoffs * (distances < self.cutoff_upper).float()
            return cutoffs


class ExpNormalSmearing(paddle.nn.Module):
    """Exponential normal smearing function."""

    def __init__(self, cutoff_lower=0.0, cutoff_upper=10.0, num_rbf=50, trainable=True):
        """Exponential normal smearing function.

        Distances are expanded into exponential radial basis functions.
        Basis function parameters are initialised as proposed by Unke & Mewly 2019 Physnet,
        https://arxiv.org/pdf/1902.08408.pdf.
        A cosine cutoff function is used to ensure smooth transition to 0.

        Args:
            cutoff_lower (float): Lower cutoff radius.
            cutoff_upper (float): Upper cutoff radius.
            num_rbf (int): Number of radial basis functions.
            trainable (bool): Whether the parameters are trainable.
        """
        super(ExpNormalSmearing, self).__init__()
        self.cutoff_lower = cutoff_lower
        self.cutoff_upper = cutoff_upper
        self.num_rbf = num_rbf
        self.trainable = trainable
        self.alpha = cutoff_upper / (cutoff_upper - cutoff_lower)
        means, betas = self._initial_params()
        if trainable:
            self.register_parameter("means", paddle.nn.Parameter(means))
            self.register_parameter("betas", paddle.nn.Parameter(betas))
        else:
            self.register_buffer("means", means)
            self.register_buffer("betas", betas)

    def _initial_params(self):
        start_value = paddle.exp(
            paddle.to_tensor(
                data=-self.cutoff_upper + self.cutoff_lower, dtype=paddle.float32
            )
        )
        means = paddle.linspace(float(start_value), 1.0, self.num_rbf)
        betas = paddle.tensor(
            [(2 / self.num_rbf * (1 - start_value)) ** -2] * self.num_rbf
        )
        return means, betas

    def reset_parameters(self):
        """Reset the parameters to their default values."""
        means, betas = self._initial_params()
        self.means.data.copy_(means)
        self.betas.data.copy_(betas)

    def forward(self, dist):
        """Expand incoming distances into basis functions."""
        dist = dist.unsqueeze(-1)
        assert isinstance(self.betas, paddle.Tensor)
        return paddle.exp(
            -self.betas
            * (paddle.exp(self.alpha * (-dist + self.cutoff_lower)) - self.means) ** 2
        )


@compile_mode("script")
class SphericalEncoding(paddle.nn.Module):
    """
    Calculate spherical harmonics from 0 to lmax
    taking displacement vector (EDGE_VEC) as input.

    lmax: maximum angular momentum quantum number used in model
    normalization : {'integral', 'component', 'norm'}
        normalization of the output tensors
        Valid options:
        * *component*: :math:`\\|Y^l(x)\\|^2 = 2l+1, x \\in S^2`
        * *norm*: :math:`\\|Y^l(x)\\| = 1, x \\in S^2`, ``component / sqrt(2l+1)``
        * *integral*: :math:`\\int_{S^2} Y^l_m(x)^2 dx = 1`, ``component / sqrt(4pi)``

    Returns
    -------
    `torch.Tensor`
        a tensor of shape ``(..., (lmax+1)^2)``
    """

    def __init__(
        self,
        lmax: int,
        parity: int = -1,
        normalization: str = "component",
        normalize=True,
    ):
        super().__init__()
        self.lmax = lmax
        self.normalization = normalization
        self.irreps_in = Irreps("1x1o") if parity == -1 else Irreps("1x1e")
        self.irreps_out = Irreps.spherical_harmonics(lmax, parity)
        self.sph = SphericalHarmonics(
            self.irreps_out,
            normalize=normalize,
            normalization=normalization,
            irreps_in=self.irreps_in,
        )

    def forward(self, r: paddle.Tensor) -> paddle.Tensor:
        return self.sph(r)


def bond_cosine(r1, r2):
    bond_cosine = paddle.sum(r1 * r2, dim=-1) / (
        paddle.norm(r1, dim=-1) * paddle.norm(r2, dim=-1)
    )
    bond_cosine = paddle.clamp(bond_cosine, -1, 1)
    return bond_cosine


def bond_sine(r1, r2):
    bond_sine = paddle.norm(paddle.cross(r1, r2, dim=-1), dim=-1) / (
        paddle.norm(r1, dim=-1) * paddle.norm(r2, dim=-1)
    )
    bond_sine = paddle.clamp(bond_sine, -1, 1)
    return bond_sine


@compile_mode("script")
class EdgeEmbedding(paddle.nn.Module):
    """
    embedding layer of |r| by
    RadialBasis(|r|)*CutOff(|r|)
    f : (N_edge) -> (N_edge, basis_num)
    """

    def __init__(
        self,
        basis_module: paddle.nn.Module,
        cutoff_module: paddle.nn.Module,
        spherical_module: paddle.nn.Module,
        use_edge_conv: bool = False,
        angle_module: paddle.nn.Module = None,
        use_sine: bool = True,
    ):
        super().__init__()
        self.basis_module = basis_module
        self.cutoff_function = cutoff_module
        self.spherical = spherical_module
        self.use_edge_conv = use_edge_conv
        if self.use_edge_conv:
            self.angle_module = angle_module
            if use_sine:
                self.angle_embedding = bond_sine
            else:
                self.angle_embedding = bond_cosine

    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:
        rvec = data[KEY.EDGE_VEC]
        r = paddle.sqrt((data[KEY.EDGE_VEC] * data[KEY.EDGE_VEC]).sum(-1))
        data[KEY.EDGE_LENGTH] = r
        data[KEY.EDGE_EMBEDDING] = self.basis_module(r) * self.cutoff_function(
            r
        ).unsqueeze(-1)
        data[KEY.EDGE_ATTR] = self.spherical(rvec)
        if self.use_edge_conv:
            rvec = data[KEY.EDGE_VEC]
            edge_features = -0.75 / r
            r2 = rvec.unsqueeze(1).repeat(1, 3, 1)
            cells = data[KEY.CELL].reshape(-1, 3, 3)
            edge_neighbors = cells[data[KEY.BATCH][data[KEY.EDGE_IDX][0]]]
            edge_angles = self.angle_embedding(edge_neighbors, r2)
            edge_angles = edge_angles.reshape(-1)
            edge_angle_embeddings = self.angle_module(edge_angles)
            edge_angle_embeddings = edge_angle_embeddings.reshape(
                edge_features.shape[0], 3, -1
            )
            edge_lattice_lengths = -0.75 / paddle.norm(edge_neighbors, dim=-1)
            edge_lattice_lengths = edge_lattice_lengths.reshape(-1)
            edge_lattice_embeddings = self.basis_module(edge_lattice_lengths)
            edge_lattice_embeddings = edge_lattice_embeddings.reshape(
                edge_features.shape[0], 3, -1
            )
            data[KEY.ANGLE_EMBEDDING] = edge_angle_embeddings
            data[KEY.LATTICE_EMBEDDING] = edge_lattice_embeddings
        return data


from typing import List, Optional, Tuple


@compile_mode("script")
class ComformerEdgeEmbedding(paddle.nn.Module):
    def __init__(
        self,
        basis_module: paddle.nn.Module,
        radial_basis_num: int,
        out_dim: int,
        spherical_module: paddle.nn.Module,
        use_sine: bool = True,
    ):
        super().__init__()
        self.radial_module = paddle.nn.Sequential(
            basis_module,
            paddle.compat.nn.Linear(radial_basis_num, out_dim),
            paddle.nn.Softplus(),
        )
        self.spherical = spherical_module
        use_edge_conv = False
        if use_edge_conv:
            self.rbf_angle = paddle.nn.Sequential(
                RBFExpansion(r_min=-1.0, r_max=1.0, n_bins=triplet_features),
                paddle.compat.nn.Linear(triplet_features, out_dim),
                paddle.nn.Softplus(),
            )
            if use_sine:
                self.angle_embedding = bond_sine
            else:
                self.angle_embedding = bond_cosine

    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:
        rvec = data[KEY.EDGE_VEC]
        r = paddle.linalg.norm(data[KEY.EDGE_VEC], dim=-1)
        data[KEY.EDGE_LENGTH] = r
        edge_features = -0.75 / r
        data[KEY.EDGE_EMBEDDING] = self.radial_module(edge_features)
        use_edge_conv = False
        data[KEY.EDGE_ATTR] = self.spherical(rvec)
        if use_edge_conv:
            r2 = rvec.unsqueeze(1).repeat(1, 3, 1)
            cells = data[KEY.CELL].reshape(-1, 3, 3)
            edge_neighbors = cells[data[KEY.BATCH][data[KEY.EDGE_IDX][0]]]
            edge_angles = self.angle_embedding(edge_neighbors, r2)
            edge_angles = edge_angles.reshape(-1)
            edge_angle_embeddings = self.rbf_angle(edge_angles)
            edge_angle_embeddings = edge_angle_embeddings.reshape(
                edge_features.shape[0], 3, -1
            )
            edge_lattice_lengths = -0.75 / paddle.norm(edge_neighbors, dim=-1)
            edge_lattice_lengths = edge_lattice_lengths.reshape(-1)
            edge_lattice_embeddings = self.rbf(edge_lattice_lengths)
            edge_lattice_embeddings = edge_lattice_embeddings.reshape(
                edge_features.shape[0], 3, -1
            )
            data[KEY.ANGLE_EMBEDDING] = edge_angle_embeddings
            data[KEY.LATTICE_EMBEDDING] = edge_lattice_embeddings
        return data


@compile_mode("script")
class RBFExpansion(paddle.nn.Module):
    """Expand interatomic distances with radial basis functions."""

    def __init__(
        self,
        r_min: float = 0,
        r_max: float = 8,
        n_bins: int = 40,
        lengthscale: Optional[float] = None,
    ):
        """Register torch parameters for RBF expansion."""
        super().__init__()
        self.r_min = r_min
        self.r_max = r_max
        self.n_bins = n_bins
        self.register_buffer(
            "centers", paddle.linspace(self.r_min, self.r_max, self.n_bins)
        )
        if lengthscale is None:
            self.lengthscale = np.diff(self.centers).mean()
            self.gamma = 1 / self.lengthscale
        else:
            self.lengthscale = lengthscale
            self.gamma = 1 / lengthscale**2

    def forward(self, distance: paddle.Tensor) -> paddle.Tensor:
        """Apply RBF expansion to interatomic distance tensor."""
        return paddle.exp(-self.gamma * (distance.unsqueeze(1) - self.centers) ** 2)

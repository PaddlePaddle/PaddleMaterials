# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# This code is adapted from https://github.com/CederGroupHub/chgnet


from __future__ import annotations

import collections
import itertools
from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import paddle
from pymatgen.core import Structure

from ppmat.models.chgnet.prefitted_weights import MPF_prefitted_data
from ppmat.models.chgnet.prefitted_weights import MPTrj_prefitted_data
from ppmat.models.common.runtime import RuntimeMixin
from ppmat.models.common.runtime import runtime_boundary
from ppmat.utils import logger
from ppmat.utils.crystal import frac_to_cart_coords
from ppmat.utils.scatter import scatter

if TYPE_CHECKING:
    from pathlib import Path


class MLP(paddle.nn.Layer):
    """Multi-Layer Perceptron used for non-linear regression."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 1,
        hidden_dim: int | Sequence[int] | None = (64, 64),
        dropout: float = 0,
        bias: bool = True,
    ) -> None:
        """Initialize the MLP.

        Args:
            input_dim (int): the input dimension
            output_dim (int): the output dimension
            hidden_dim (list[int] | int]): a list of integers or a single integer
                representing the number of hidden units in each layer of the MLP.
                Default = [64, 64]
            dropout (float): the dropout rate before each linear layer. Default: 0
            bias (bool): whether to use bias in each Linear layers.
                Default = True
        """
        super().__init__()
        if hidden_dim is None or hidden_dim == 0:
            layers = [
                paddle.nn.Dropout(p=dropout),
                paddle.nn.Linear(
                    in_features=input_dim, out_features=output_dim, bias_attr=bias
                ),
            ]
        elif isinstance(hidden_dim, int):
            layers = [
                paddle.nn.Linear(
                    in_features=input_dim, out_features=hidden_dim, bias_attr=bias
                ),
                paddle.nn.Silu(),
                paddle.nn.Dropout(p=dropout),
                paddle.nn.Linear(
                    in_features=hidden_dim, out_features=output_dim, bias_attr=bias
                ),
            ]
        elif isinstance(hidden_dim, Sequence):
            layers = [
                paddle.nn.Linear(
                    in_features=input_dim, out_features=hidden_dim[0], bias_attr=bias
                ),
                paddle.nn.Silu(),
            ]
            if len(hidden_dim) != 1:
                for h_in, h_out in itertools.pairwise(hidden_dim):
                    layers.append(
                        paddle.nn.Linear(
                            in_features=h_in, out_features=h_out, bias_attr=bias
                        )
                    )
                    layers.append(paddle.nn.Silu())
            layers.append(paddle.nn.Dropout(p=dropout))
            layers.append(
                paddle.nn.Linear(
                    in_features=hidden_dim[-1], out_features=output_dim, bias_attr=bias
                )
            )
        else:
            raise TypeError(
                f"hidden_dim={hidden_dim!r} must be an integer, a list of integers, "
                "or None."
            )
        self.layers = paddle.nn.Sequential(*layers)

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        """Performs a forward pass through the MLP.

        Args:
            x (Tensor): a tensor of shape (batch_size, input_dim)

        Returns:
            Tensor: a tensor of shape (batch_size, output_dim)
        """
        return self.layers(x)


class GatedMLP(paddle.nn.Layer):
    """Gated MLP,  similar model structure is used in CGCNN and M3GNet."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int | list[int] | None = None,
        dropout: float = 0,
        bias: bool = True,
    ) -> None:
        """Initialize a gated MLP.

        Args:
            input_dim (int): the input dimension
            output_dim (int): the output dimension
            hidden_dim (list[int] | int]): a list of integers or a single integer
                representing the number of hidden units in each layer of the MLP.
                Default = None
            dropout (float): the dropout rate before each linear layer.
                Default: 0
            bias (bool): whether to use bias in each Linear layers.
                Default = True
        """
        super().__init__()
        self.mlp_core = MLP(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            bias=bias,
        )
        self.mlp_gate = MLP(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            bias=bias,
        )
        self.activation = paddle.nn.Silu()
        self.sigmoid = paddle.nn.Sigmoid()

        self.bn1 = paddle.nn.LayerNorm(output_dim)
        self.bn2 = paddle.nn.LayerNorm(output_dim)

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        """Performs a forward pass through the MLP.

        Args:
            x (Tensor): a tensor of shape (batch_size, input_dim)

        Returns:
            Tensor: a tensor of shape (batch_size, output_dim)
        """
        core = self.activation(self.bn1(self.mlp_core(x)))
        gate = self.sigmoid(self.bn2(self.mlp_gate(x)))
        return core * gate


class AtomRef(paddle.nn.Layer):
    """A linear regression for elemental energy.
    From: https://github.com/materialsvirtuallab/m3gnet/.
    """

    def __init__(self, is_intensive: bool = True, max_num_elements: int = 94) -> None:
        """Initialize an AtomRef model."""
        super().__init__()
        self.is_intensive = is_intensive
        self.max_num_elements = max_num_elements
        self.fc = paddle.nn.Linear(
            in_features=max_num_elements, out_features=1, bias_attr=False
        )
        self.fitted = False

    def forward(self, graphs) -> paddle.Tensor:
        """Get the energy of a list of graphs.

        Args:
            graphs: a list of Crystal Graph to compute

        Returns:
            energy (tensor)
        """
        if not self.fitted:
            raise ValueError("composition model needs to be fitted first!")
        composition_feas = graphs.node_feat["composition_fea"]
        return self._get_energy(composition_feas)

    def _get_energy(self, composition_feas: paddle.Tensor) -> paddle.Tensor:
        """Predict the energy given composition encoding.

        Args:
            composition_feas: batched atom feature matrix of shape
                [batch_size, total_num_elements].

        Returns:
            prediction associated with each composition [batchsize].
        """
        return self.fc(composition_feas)  # .flatten()

    def fit(
        self,
        structures_or_graphs,
        energies: Sequence[float],
    ) -> None:
        """Fit the model to a list of crystals and energies.

        Args:
            structures_or_graphs: Any iterable of pymatgen structures and/or graphs.
            energies (list[float]): Target energies.
        """
        num_data = len(energies)
        composition_feas = paddle.zeros(shape=[num_data, self.max_num_elements])
        e = paddle.zeros(shape=[num_data])
        for index, (structure, energy) in enumerate(
            zip(structures_or_graphs, energies, strict=True)
        ):
            if isinstance(structure, Structure):
                atomic_number = paddle.to_tensor(
                    [site.specie.Z for site in structure], dtype="int32"
                )
            else:
                atomic_number = structure.node_feat["atom_types"]
            composition_fea = paddle.bincount(
                atomic_number - 1, minlength=self.max_num_elements
            )
            if self.is_intensive:
                composition_fea = composition_fea / atomic_number.shape[0]
            composition_feas[index, :] = composition_fea
            e[index] = energy

        # Use numpy for pinv
        self.feature_matrix = composition_feas.detach().numpy()
        self.energies = e.detach().numpy()
        state_dict = collections.OrderedDict()
        weight = (
            np.linalg.pinv(self.feature_matrix.T @ self.feature_matrix)
            @ self.feature_matrix.T
            @ self.energies
        )
        state_dict["weight"] = paddle.to_tensor(data=weight).view(94, 1)
        self.fc.set_state_dict(state_dict)
        self.fitted = True

    def get_site_energies(self, graphs) -> list[paddle.Tensor]:
        """Predict the site energies given a list of CrystalGraphs.

        Args:
            graphs: a list of Crystal Graph to compute

        Returns:
            a list of tensors corresponding to site energies of each graph [batchsize].
        """
        return [
            self.fc.state_dict()["weight"][0, graph.node_feat["atom_types"] - 1]
            for graph in graphs
        ]

    def initialize_from(self, dataset: str) -> None:
        """Initialize pre-fitted weights from a dataset."""
        if dataset in {"MPtrj", "MPtrj_e"}:
            self.initialize_from_MPtrj()
        elif dataset == "MPF":
            self.initialize_from_MPF()
        else:
            raise NotImplementedError(f"dataset={dataset!r} not supported yet")

    def initialize_from_MPtrj(self) -> None:
        """Initialize pre-fitted weights from MPtrj dataset."""
        state_dict = collections.OrderedDict()
        state_dict["weight"] = paddle.to_tensor(data=MPTrj_prefitted_data).view([94, 1])
        self.fc.set_state_dict(state_dict=state_dict)
        self.is_intensive = True
        self.fitted = True

    def initialize_from_MPF(self) -> None:
        """Initialize pre-fitted weights from MPF dataset."""
        state_dict = collections.OrderedDict()
        state_dict["weight"] = paddle.to_tensor(data=MPF_prefitted_data).view([94, 1])
        self.fc.set_state_dict(state_dict=state_dict)
        self.is_intensive = False
        self.fitted = True

    def initialize_from_numpy(self, file_name: str | Path) -> None:
        """Initialize pre-fitted weights from numpy file."""
        atom_ref_np = np.load(file_name)
        state_dict = collections.OrderedDict()
        state_dict["weight"] = paddle.to_tensor(data=atom_ref_np).view([1, 94])
        self.fc.set_state_dict(state_dict=state_dict)
        self.is_intensive = False
        self.fitted = True


class Fourier(paddle.nn.Layer):
    """Fourier Expansion for angle features."""

    def __init__(self, order: int = 5, learnable: bool = False) -> None:
        """Initialize the Fourier expansion.

        Args:
            order (int): the maximum order, refer to the N in eq 1 in CHGNet paper
                Default = 5
            learnable (bool): whether to set the frequencies as learnable parameters
                Default = False
        """
        super().__init__()
        self.order = order
        if learnable:
            self.frequencies = paddle.base.framework.EagerParamBase.from_tensor(
                tensor=paddle.arange(start=1, end=order + 1, dtype="float32"),
                trainable=True,
            )
        else:
            self.register_buffer(
                name="frequencies",
                tensor=paddle.arange(start=1, end=order + 1, dtype="float32"),
            )

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        """Apply Fourier expansion to a feature Tensor."""
        # The following is the original implementation. As index does not currently
        # support high-order gradients, an alternative implementation was used
        # result = paddle.zeros(shape=[tuple(x.shape)[0], 1 + 2 * self.order],
        #     dtype=x.dtype)
        result = paddle.ones(shape=[x.shape[0], 1], dtype=x.dtype)
        result = result / paddle.sqrt(x=paddle.to_tensor(data=[2.0]))

        tmp = paddle.outer(x=x, y=self.frequencies)
        # The following is the original implementation. As index does not currently
        # support high-order gradients, an alternative implementation was used
        # result[:, 1:self.order + 1] = paddle.sin(x=tmp)
        # result[:, self.order + 1:] = paddle.cos(x=tmp)
        result = paddle.concat([result, paddle.sin(tmp), paddle.cos(tmp)], axis=1)

        return result / np.sqrt(np.pi)


class RadialBessel(paddle.nn.Layer):
    """1D Bessel Basis
    from: https://github.com/TUM-DAML/gemnet_pytorch/.
    """

    def __init__(
        self,
        num_radial: int = 9,
        cutoff: float = 5,
        learnable: bool = False,
        smooth_cutoff: int = 5,
    ) -> None:
        """Initialize the SmoothRBF function.

        Args:
            num_radial (int): Controls maximum frequency
                Default = 9
            cutoff (float):  Cutoff distance in Angstrom.
                Default = 5
            learnable (bool): whether to set the frequencies learnable
                Default = False
            smooth_cutoff (int): smooth cutoff strength
                Default = 5
        """
        super().__init__()
        self.num_radial = num_radial
        self.inv_cutoff = 1 / cutoff
        self.norm_const = (2 * self.inv_cutoff) ** 0.5
        if learnable:
            self.frequencies = paddle.base.framework.EagerParamBase.from_tensor(
                tensor=paddle.to_tensor(
                    data=np.pi * np.arange(1, self.num_radial + 1, dtype=np.float32),
                    dtype="float32",
                ),
                trainable=True,
            )
        else:
            self.register_buffer(
                name="frequencies",
                tensor=np.pi
                * paddle.arange(start=1, end=self.num_radial + 1, dtype="float32"),
            )
        if smooth_cutoff is not None:
            self.smooth_cutoff = CutoffPolynomial(
                cutoff=cutoff, cutoff_coeff=smooth_cutoff
            )
        else:
            self.smooth_cutoff = None

    def forward(
        self, dist: paddle.Tensor, return_smooth_factor: bool = False
    ) -> paddle.Tensor | tuple[paddle.Tensor, paddle.Tensor]:
        """Apply Bessel expansion to a feature Tensor.

        Args:
            dist (Tensor): tensor of distances [n, 1]
            return_smooth_factor (bool): whether to return the smooth factor
                Default = False

        Returns:
            out (Tensor): tensor of Bessel distances [n, dim]
            where the expanded dimension will be num_radial
            smooth_factor (Tensor): tensor of smooth factors [n, 1]
        """
        dist = dist[:, None]
        d_scaled = dist * self.inv_cutoff
        out = self.norm_const * paddle.sin(x=self.frequencies * d_scaled) / dist
        if self.smooth_cutoff is not None:
            smooth_factor = self.smooth_cutoff(dist)
            out = smooth_factor * out
            if return_smooth_factor:
                return out, smooth_factor
        return out


class CutoffPolynomial(paddle.nn.Layer):
    """Polynomial soft-cutoff function for atom graph
    ref: https://github.com/TUM-DAML/gemnet_pytorch/blob/-/gemnet/model/layers/envelope.py.
    """

    def __init__(self, cutoff: float = 5, cutoff_coeff: float = 5) -> None:
        """Initialize the polynomial cutoff function.

        Args:
            cutoff (float): cutoff radius (A) in atom graph construction
            Default = 5
            cutoff_coeff (float): the strength of soft-Cutoff
            0 will disable the cutoff, returning 1 at every r
            for positive numbers > 0, the smaller cutoff_coeff is, the faster this
                function decays. Default = 5.
        """
        super().__init__()
        self.cutoff = cutoff
        self.p = cutoff_coeff
        self.a = -(self.p + 1) * (self.p + 2) / 2
        self.b = self.p * (self.p + 2)
        self.c = -self.p * (self.p + 1) / 2

    def forward(self, r: paddle.Tensor) -> paddle.Tensor:
        """Polynomial cutoff function.

        Args:
            r (Tensor): radius distance tensor

        Returns:
            polynomial cutoff functions: decaying from 1 at r=0 to 0 at r=cutoff
        """
        if self.p != 0:
            r_scaled = r / self.cutoff
            env_val = (
                1
                + self.a * r_scaled**self.p
                + self.b * r_scaled ** (self.p + 1)
                + self.c * r_scaled ** (self.p + 2)
            )
            return paddle.where(
                condition=r_scaled < 1, x=env_val, y=paddle.zeros_like(x=r_scaled)
            )
        return paddle.ones(shape=tuple(r.shape), dtype=r.dtype)


class AtomEmbedding(paddle.nn.Layer):
    """Encode an atom by its atomic number using a learnable embedding layer."""

    def __init__(self, atom_feature_dim: int, max_num_elements: int = 94) -> None:
        """Initialize the Atom featurizer.

        Args:
            atom_feature_dim (int): dimension of atomic embedding.
            max_num_elements (int): maximum number of elements in the dataset.
                Default = 94
        """
        super().__init__()
        # The original implementation is using paddle.nn.Embedding
        # self.embedding = paddle.nn.Embedding(num_embeddings=
        #     max_num_elements, embedding_dim=atom_feature_dim)
        self.max_num_elements = max_num_elements
        self.embedding = paddle.nn.Linear(
            max_num_elements, atom_feature_dim, bias_attr=False
        )

    def forward(self, atomic_numbers: paddle.Tensor) -> paddle.Tensor:
        """Convert the structure to a atom embedding tensor.

        Args:
            atomic_numbers (Tensor): [n_atom, 1].

        Returns:
            atom_fea (Tensor): atom embeddings [n_atom, atom_feature_dim].
        """
        atomic_numbers = paddle.nn.functional.one_hot(
            atomic_numbers, self.max_num_elements
        )
        return self.embedding(atomic_numbers)


class BondEncoder(paddle.nn.Layer):
    """Encode a chemical bond given the positions of two atoms using Gaussian
    distance.
    """

    def __init__(
        self,
        atom_graph_cutoff: float = 5,
        bond_graph_cutoff: float = 3,
        num_radial: int = 9,
        cutoff_coeff: int = 5,
        learnable: bool = False,
    ) -> None:
        """Initialize the bond encoder.

        Args:
            atom_graph_cutoff (float): The cutoff for constructing AtomGraph default = 5
            bond_graph_cutoff (float): The cutoff for constructing BondGraph default = 3
            num_radial (int): The number of radial component. Default = 9
            cutoff_coeff (int): Strength for graph cutoff smoothness. Default = 5
            learnable(bool): Whether the frequency in rbf expansion is learnable.
                Default = False
        """
        super().__init__()
        self.rbf_expansion_ag = RadialBessel(
            num_radial=num_radial,
            cutoff=atom_graph_cutoff,
            smooth_cutoff=cutoff_coeff,
            learnable=learnable,
        )
        self.rbf_expansion_bg = RadialBessel(
            num_radial=num_radial,
            cutoff=bond_graph_cutoff,
            smooth_cutoff=cutoff_coeff,
            learnable=learnable,
        )

    def forward(
        self,
        center: paddle.Tensor,
        neighbor: paddle.Tensor,
        undirected2directed: paddle.Tensor,
        image: paddle.Tensor,
        lattice: paddle.Tensor,
    ) -> tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]:
        """Compute the pairwise distance between 2 3d coordinates.

        Args:
            center (Tensor): 3d cartesian coordinates of center atoms [n_bond, 3]
            neighbor (Tensor): 3d cartesian coordinates of neighbor atoms [n_bond, 3]
            undirected2directed (Tensor): mapping from undirected bond to one of its
                directed bond [n_bond]
            image (Tensor): the periodic image specifying the location of neighboring
                atom [n_bond, 3]
            lattice (Tensor): the lattice of this structure [3, 3]

        Returns:
            bond_basis_ag (Tensor): the bond basis in AtomGraph [n_bond, num_radial]
            bond_basis_ag (Tensor): the bond basis in BondGraph [n_bond, num_radial]
            bond_vectors (Tensor): normalized bond vectors, for tracking the bond
                directions [n_bond, 3]
        """
        neighbor = neighbor + image @ lattice
        bond_vectors = center - neighbor
        bond_lengths = paddle.linalg.norm(x=bond_vectors, axis=1)
        bond_vectors = bond_vectors / bond_lengths[:, None]
        undirected_bond_lengths = paddle.gather(
            x=bond_lengths, axis=0, index=undirected2directed
        )
        bond_basis_ag = self.rbf_expansion_ag(undirected_bond_lengths)
        bond_basis_bg = self.rbf_expansion_bg(undirected_bond_lengths)
        return bond_basis_ag, bond_basis_bg, bond_vectors


class AngleEncoder(paddle.nn.Layer):
    """Encode an angle given the two bond vectors using Fourier Expansion."""

    def __init__(self, num_angular: int = 9, learnable: bool = True) -> None:
        """Initialize the angle encoder.

        Args:
            num_angular (int): number of angular basis to use. Must be an odd integer.
            learnable (bool): whether to set the frequencies of the Fourier expansion
                as learnable parameters. Default = False
        """
        super().__init__()
        if num_angular % 2 != 1:
            raise ValueError(f"num_angular={num_angular!r} must be an odd integer")
        circular_harmonics_order = (num_angular - 1) // 2
        self.fourier_expansion = Fourier(
            order=circular_harmonics_order, learnable=learnable
        )

    def forward(self, bond_i: paddle.Tensor, bond_j: paddle.Tensor) -> paddle.Tensor:
        """Compute the angles between normalized vectors.

        Args:
            bond_i (Tensor): normalized left bond vector [n_angle, 3]
            bond_j (Tensor): normalized right bond vector [n_angle, 3]

        Returns:
            angle_fea (Tensor):  expanded cos_ij [n_angle, angle_feature_dim]
        """
        cosine_ij = paddle.sum(x=bond_i * bond_j, axis=1) * (1 - 1e-06)
        angle = paddle.acos(x=cosine_ij)
        return self.fourier_expansion(angle)


class AtomConv(paddle.nn.Layer):
    """A convolution Layer to update atom features."""

    def __init__(
        self,
        atom_fea_dim: int,
        bond_fea_dim: int,
        hidden_dim: int = 64,
        dropout: float = 0,
        use_mlp_out: bool = True,
        mlp_out_bias: bool = False,
        resnet: bool = True,
    ) -> None:
        """Initialize the AtomConv layer.

        Args:
            atom_fea_dim (int): The dimensionality of the input atom features.
            bond_fea_dim (int): The dimensionality of the input bond features.
            hidden_dim (int, optional): The dimensionality of the hidden layers in the
                gated MLP.
                Default = 64
            dropout (float, optional): The dropout rate to apply to the gated MLP.
                Default = 0.
            use_mlp_out (bool, optional): Whether to apply an MLP output layer to the
                updated atom features.
                Default = True
            mlp_out_bias (bool): whether to use bias in the output MLP Linear layer.
                Default = False
            resnet (bool, optional): Whether to apply a residual connection to the
                updated atom features.
                Default = True
            gMLP_norm (str, optional): The name of the normalization layer to use on the
                gated MLP. Must be one of "batch", "layer", or None.
                Default = None
        """
        super().__init__()
        self.use_mlp_out = use_mlp_out
        self.resnet = resnet
        self.activation = paddle.nn.Silu()
        self.twoBody_atom = GatedMLP(
            input_dim=2 * atom_fea_dim + bond_fea_dim,
            output_dim=atom_fea_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
        if self.use_mlp_out:
            self.mlp_out = MLP(
                input_dim=atom_fea_dim,
                output_dim=atom_fea_dim,
                hidden_dim=0,
                bias=mlp_out_bias,
            )

    def forward(
        self,
        atom_feas: paddle.Tensor,
        bond_feas: paddle.Tensor,
        bond_weights: paddle.Tensor,
        atom_graph: paddle.Tensor,
        directed2undirected: paddle.Tensor,
    ) -> paddle.Tensor:
        """Forward pass of AtomConv module that updates the atom features and
            optionally bond features.

        Args:
            atom_feas (Tensor): Input tensor with shape
                [num_batch_atoms, atom_fea_dim]
            bond_feas (Tensor): Input tensor with shape
                [num_undirected_bonds, bond_fea_dim]
            bond_weights (Tensor): AtomGraph bond weights with shape
                [num_undirected_bonds, bond_fea_dim]
            atom_graph (Tensor): Directed AtomGraph adjacency list with shape
                [num_directed_bonds, 2]
            directed2undirected (Tensor): Index tensor that maps directed bonds to
                undirected bonds.with shape
                [num_undirected_bonds]

        Returns:
            Tensor: the updated atom features tensor with shape
            [num_batch_atom, atom_fea_dim]

        Notes:
            - num_batch_atoms = sum(num_atoms) in batch
        """
        center_atoms = paddle.gather(x=atom_feas, axis=0, index=atom_graph[:, 0])
        nbr_atoms = paddle.gather(x=atom_feas, axis=0, index=atom_graph[:, 1])
        bonds = paddle.gather(x=bond_feas, axis=0, index=directed2undirected)
        messages = paddle.concat(x=[center_atoms, bonds, nbr_atoms], axis=1)
        messages = self.twoBody_atom(messages)
        bond_weight = paddle.gather(x=bond_weights, axis=0, index=directed2undirected)
        messages *= bond_weight
        new_atom_feas = scatter(
            messages,
            atom_graph[:, 0].cast("int64"),
            dim=0,
            dim_size=len(atom_feas),
            reduce="sum",
        )
        if self.use_mlp_out:
            new_atom_feas = self.mlp_out(new_atom_feas)
        if self.resnet:
            new_atom_feas += atom_feas
        return new_atom_feas


class BondConv(paddle.nn.Layer):
    """A convolution Layer to update bond features."""

    def __init__(
        self,
        atom_fea_dim: int,
        bond_fea_dim: int,
        angle_fea_dim: int,
        hidden_dim: int = 64,
        dropout: float = 0,
        use_mlp_out: bool = True,
        mlp_out_bias: bool = False,
        resnet=True,
    ) -> None:
        """Initialize the BondConv layer.

        Args:
            atom_fea_dim (int): The dimensionality of the input atom features.
            bond_fea_dim (int): The dimensionality of the input bond features.
            angle_fea_dim (int): The dimensionality of the input angle features.
            hidden_dim (int, optional): The dimensionality of the hidden layers
                in the gated MLP.
                Default = 64
            dropout (float, optional): The dropout rate to apply to the gated MLP.
                Default = 0.
            use_mlp_out (bool, optional): Whether to apply an MLP output layer to the
                updated atom features.
                Default = True
            mlp_out_bias (bool): whether to use bias in the output MLP Linear layer.
                Default = False
            resnet (bool, optional): Whether to apply a residual connection to the
                updated atom features.
                Default = True
            gMLP_norm (str, optional): The name of the normalization layer to use on the
                gated MLP. Must be one of "batch", "layer", or None.
                Default = None
        """
        super().__init__()
        self.use_mlp_out = use_mlp_out
        self.resnet = resnet
        self.activation = paddle.nn.Silu()
        self.twoBody_bond = GatedMLP(
            input_dim=atom_fea_dim + 2 * bond_fea_dim + angle_fea_dim,
            output_dim=bond_fea_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
        if self.use_mlp_out:
            self.mlp_out = MLP(
                input_dim=bond_fea_dim,
                output_dim=bond_fea_dim,
                hidden_dim=0,
                bias=mlp_out_bias,
            )

    def forward(
        self,
        atom_feas: paddle.Tensor,
        bond_feas: paddle.Tensor,
        bond_weights: paddle.Tensor,
        angle_feas: paddle.Tensor,
        bond_graph: paddle.Tensor,
    ) -> paddle.Tensor:
        """Update the bond features.

        Args:
            atom_feas (Tensor): atom features tensor with shape
                [num_batch_atoms, atom_fea_dim]
            bond_feas (Tensor): bond features tensor with shape
                [num_undirected_bonds, bond_fea_dim]
            bond_weights (Tensor): BondGraph bond weights with shape
                [num_undirected_bonds, bond_fea_dim]
            angle_feas (Tensor): angle features tensor with shape
                [num_batch_angles, angle_fea_dim]
            bond_graph (Tensor): Directed BondGraph tensor with shape
                [num_batched_angles, 3]

        Returns:
            new_bond_feas (Tensor): bond feature tensor with shape
                [num_undirected_bonds, bond_fea_dim]

        Notes:
            - num_batch_atoms = sum(num_atoms) in batch
        """
        center_atoms = paddle.gather(
            x=atom_feas, axis=0, index=bond_graph[:, 0].cast("int64")
        )
        bond_feas_i = paddle.gather(
            x=bond_feas, axis=0, index=bond_graph[:, 1].cast("int64")
        )
        bond_feas_j = paddle.gather(
            x=bond_feas, axis=0, index=bond_graph[:, 2].cast("int64")
        )
        total_fea = paddle.concat(
            x=[bond_feas_i, bond_feas_j, angle_feas, center_atoms], axis=1
        )
        bond_update = self.twoBody_bond(total_fea)
        bond_weights_i = paddle.gather(
            x=bond_weights, axis=0, index=bond_graph[:, 1].cast("int64")
        )
        bond_weights_j = paddle.gather(
            x=bond_weights, axis=0, index=bond_graph[:, 2].cast("int64")
        )
        bond_update = bond_update * bond_weights_i * bond_weights_j
        new_bond_feas = scatter(
            bond_update,
            bond_graph[:, 1].cast("int64"),
            dim=0,
            dim_size=len(bond_feas),
            reduce="sum",
        )
        if self.use_mlp_out:
            new_bond_feas = self.mlp_out(new_bond_feas)
        if self.resnet:
            new_bond_feas += bond_feas
        return new_bond_feas


class AngleUpdate(paddle.nn.Layer):
    """Update angle features."""

    def __init__(
        self,
        atom_fea_dim: int,
        bond_fea_dim: int,
        angle_fea_dim: int,
        hidden_dim: int = 0,
        dropout: float = 0,
        resnet: bool = True,
    ) -> None:
        """Initialize the AngleUpdate layer.

        Args:
            atom_fea_dim (int): The dimensionality of the input atom features.
            bond_fea_dim (int): The dimensionality of the input bond features.
            angle_fea_dim (int): The dimensionality of the input angle features.
            hidden_dim (int, optional): The dimensionality of the hidden layers
                in the gated MLP.
                Default = 0
            dropout (float, optional): The dropout rate to apply to the gated MLP.
                Default = 0.
            resnet (bool, optional): Whether to apply a residual connection to the
                updated atom features.
                Default = True
        """
        super().__init__()
        self.resnet = resnet
        self.activation = paddle.nn.Silu()
        self.twoBody_bond = GatedMLP(
            input_dim=atom_fea_dim + 2 * bond_fea_dim + angle_fea_dim,
            output_dim=angle_fea_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

    def forward(
        self,
        atom_feas: paddle.Tensor,
        bond_feas: paddle.Tensor,
        angle_feas: paddle.Tensor,
        bond_graph: paddle.Tensor,
    ) -> paddle.Tensor:
        """Update the angle features using bond graph.

        Args:
            atom_feas (Tensor): atom features tensor with shape
                [num_batch_atoms, atom_fea_dim]
            bond_feas (Tensor): bond features tensor with shape
                [num_undirected_bonds, bond_fea_dim]
            angle_feas (Tensor): angle features tensor with shape
                [num_batch_angles, angle_fea_dim]
            bond_graph (Tensor): Directed BondGraph tensor with shape
                [num_batched_angles, 3]

        Returns:
            new_angle_feas (Tensor): angle features tensor with shape
                [num_batch_angles, angle_fea_dim]

        Notes:
            - num_batch_atoms = sum(num_atoms) in batch
        """
        bond_graph = bond_graph.astype("int64")
        center_atoms = paddle.gather(x=atom_feas, axis=0, index=bond_graph[:, 0])
        bond_feas_i = paddle.gather(x=bond_feas, axis=0, index=bond_graph[:, 1])
        bond_feas_j = paddle.gather(x=bond_feas, axis=0, index=bond_graph[:, 2])
        total_fea = paddle.concat(
            x=[bond_feas_i, bond_feas_j, angle_feas, center_atoms], axis=1
        )
        new_angle_feas = self.twoBody_bond(total_fea)

        if self.resnet:
            new_angle_feas += angle_feas
        return new_angle_feas


class GraphPooling(paddle.nn.Layer):
    """Pooling the sub-graphs in the batched graph."""

    def __init__(self, *, average: bool = False) -> None:
        """Args:
        average (bool): whether to average the features.
        """
        super().__init__()
        self.average = average

    def forward(
        self,
        atom_feas: paddle.Tensor,
        atom_owner: paddle.Tensor,
        num_owner: int | None = None,
    ) -> paddle.Tensor:
        """Merge the atom features that belong to same graph in a batched graph.

        Args:
            atom_feas (Tensor): batched atom features after convolution layers.
                [num_batch_atoms, atom_fea_dim or 1]
            atom_owner (Tensor): graph indices for each atom.
                [num_batch_atoms]
            num_owner (int, optional): number of graphs in the batch.

        Returns:
            crystal_feas (Tensor): crystal feature matrix.
                [n_crystals, atom_fea_dim or 1]
        """
        return scatter(
            atom_feas,
            atom_owner.cast("int64"),
            dim=0,
            dim_size=num_owner,
            reduce="mean" if self.average else "sum",
        )


class GraphAttentionReadOut(paddle.nn.Layer):
    """Multi Head Attention Read Out Layer
    merge the information from atom_feas to crystal_fea.
    """

    def __init__(
        self,
        atom_fea_dim: int,
        num_head: int = 3,
        hidden_dim: int = 32,
        average=False,
    ) -> None:
        """Initialize the layer.

        Args:
            atom_fea_dim (int): atom feature dimension
            num_head (int): number of attention heads used
            hidden_dim (int): dimension of hidden layer
            average (bool): whether to average the features
        """
        super().__init__()
        self.key = MLP(
            input_dim=atom_fea_dim, output_dim=num_head, hidden_dim=hidden_dim
        )
        self.softmax = paddle.nn.Softmax(axis=0)
        self.average = average

    def forward(
        self,
        atom_feas: paddle.Tensor,
        atom_owner: paddle.Tensor,
        num_owner: int | None = None,
    ) -> paddle.Tensor:
        """Merge the atom features that belong to same graph in a batched graph.

        Args:
            atom_feas (Tensor): batched atom features after convolution layers.
                [num_batch_atoms, atom_fea_dim]
            atom_owner (Tensor): graph indices for each atom.
                [num_batch_atoms]
            num_owner (int, optional): number of graphs in the batch.

        Returns:
            crystal_feas (Tensor): crystal feature matrix.
                [n_crystals, atom_fea_dim]
        """
        crystal_feas = []
        weights = self.key(atom_feas)
        bin_count = paddle.bincount(x=atom_owner)
        start_index = 0
        for n_atom in bin_count:
            atom_fea = atom_feas[start_index : start_index + n_atom, :]
            weight = self.softmax(weights[start_index : start_index + n_atom, :])
            crystal_fea = (atom_fea.T @ weight).reshape([-1])
            if self.average:
                crystal_fea /= n_atom
            crystal_feas.append(crystal_fea)
            start_index += n_atom
        return paddle.stack(x=crystal_feas, axis=0)


class CHGNet(RuntimeMixin, paddle.nn.Layer):
    """Crystal Hamiltonian Graph neural Network. A model that takes in a crystal graph
    and output energy, force, magmom, stress.

    https://www.nature.com/articles/s42256-023-00716-3

    Args:
        atom_fea_dim (int): atom feature vector embedding dimension. Default = 64
        bond_fea_dim (int): bond feature vector embedding dimension. Default = 64
        angle_fea_dim (int): angle feature vector embedding dimension. Default = 64
        bond_fea_dim (int): angle feature vector embedding dimension. Default = 64
        composition_model (nn.Layer, str): attach a composition model to
            predict energy or initialize a pretrained linear regression (AtomRef).
            The default 'MPtrj' is the atom reference energy linear regression
            trained on all Materials Project relaxation trajectories. Default = 'MPtrj'
        num_radial (int): number of radial basis used in bond basis expansion. Default
            to 31.
        num_angular (int): number of angular basis used in angle basis expansion.
            Default = 31.
        n_conv (int): number of interaction blocks. Default = 4
            Note: last interaction block contain only an atom_conv layer
        atom_conv_hidden_dim (List or int): hidden dimensions of atom convolution
            layers. Default = 64
        update_bond (bool): whether to use bond_conv_layer in bond graph to update bond
            embeddings. Default = True.
        bond_conv_hidden_dim (List or int): hidden dimensions of bond convolution
            layers. Default = 64
        update_angle (bool): whether to use angle_update_layer to update angle
            embeddings. Default = True
        angle_layer_hidden_dim (List or int): hidden dimensions of angle layers.
            Default = 0
        conv_dropout (float): dropout rate in all conv_layers.
            Default = 0
        read_out (str): method for pooling layer, 'ave' for standard
            average pooling, 'attn' for multi-head attention.
            Default = "ave"
        mlp_hidden_dims (int or list): readout multilayer perceptron
            hidden dimensions.
            Default = [64, 64]
        mlp_dropout (float): dropout rate in readout MLP.
            Default = 0.
        is_intensive (bool): whether the energy training label is intensive
            i.e. energy per atom.
            Default = True
        mlp_first (bool): whether to apply mlp first then pooling.
            if set to True, then CHGNet is essentially calculating energy for each
            atom, them sum them up, this is used for the pretrained model
            Default = True
        atom_graph_cutoff (float): cutoff radius (A) in creating atom_graph,
            this need to be consistent with the value in training dataloader
            Default = 5
        bond_graph_cutoff (float): cutoff radius (A) in creating bond_graph,
            this need to be consistent with value in training dataloader
            Default = 3
        cutoff_coeff (float): cutoff strength used in graph smooth cutoff function.
            the smaller this coeff is, the smoother the basis is
            Default = 5
        learnable_rbf (bool): whether to set the frequencies in rbf and Fourier
            basis functions learnable.
            Default = True
        return_site_energies (bool): whether to return per-site energies,
            only available if mlp_first == True. Default = False
        return_atom_feas (bool): whether to return the atom features before last
            conv layer. Default = False
        return_crystal_feas (bool): whether to return crystal feature. Default = False
        **kwargs: Additional keyword arguments

    """

    def __init__(
        self,
        atom_fea_dim: int = 64,
        bond_fea_dim: int = 64,
        angle_fea_dim: int = 64,
        composition_model: str | paddle.nn.Layer = "MPtrj",
        num_radial: int = 31,
        num_angular: int = 31,
        n_conv: int = 4,
        atom_conv_hidden_dim: Sequence[int] | int = 64,
        update_bond: bool = True,
        bond_conv_hidden_dim: Sequence[int] | int = 64,
        update_angle: bool = True,
        angle_layer_hidden_dim: Sequence[int] | int = 0,
        conv_dropout: float = 0,
        read_out: str = "ave",
        mlp_hidden_dims: Sequence[int] | int = (64, 64, 64),
        mlp_dropout: float = 0,
        mlp_first: bool = True,
        is_intensive: bool = True,
        atom_graph_cutoff: float = 6,
        bond_graph_cutoff: float = 3,
        cutoff_coeff: int = 8,
        learnable_rbf: bool = True,
        is_freeze: bool = False,
        property_names: Sequence[str] | None = None,
        return_site_energies: bool = False,
        return_atom_feas: bool = False,
        return_crystal_feas: bool = False,
        loss_type: str = "mse_loss",
        huber_loss_delta: float = 0.1,
        loss_weights_dict: dict | None = None,
        execution_backend: str = "eager",
        runtime_options: dict | None = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self._init_runtime(execution_backend, runtime_options)
        self.atom_fea_dim = atom_fea_dim
        self.bond_fea_dim = bond_fea_dim
        self.is_intensive = is_intensive
        self.n_conv = n_conv
        self.is_freeze = is_freeze

        support_property_names = ["energy_per_atom", "force", "stress", "magmom"]
        support_loss_weights_dict = {
            "energy_per_atom": 1.0,
            "force": 1.0,
            "stress": 0.1,
            "magmom": 0.1,
        }
        if property_names is None:
            property_names = support_property_names
        else:
            for property_name in property_names:
                assert (
                    property_name in support_property_names
                ), f"{property_name} is not supported, please choose from "
                f"{support_property_names}"
        self.property_names = property_names
        if loss_weights_dict is None:
            loss_weights_dict = {
                key: support_loss_weights_dict[key] for key in property_names
            }
        self.loss_weights_dict = loss_weights_dict

        self.return_site_energies = return_site_energies
        self.return_atom_feas = return_atom_feas
        self.return_crystal_feas = return_crystal_feas

        if isinstance(composition_model, paddle.nn.Layer):
            self.composition_model = composition_model
        elif isinstance(composition_model, str):
            self.composition_model = AtomRef(is_intensive=is_intensive)
            self.composition_model.initialize_from(composition_model)
        else:
            self.composition_model = None
        if self.composition_model is not None:
            for param in self.composition_model.parameters():
                param.stop_gradient = True
        self.atom_embedding = AtomEmbedding(atom_feature_dim=atom_fea_dim)
        self.bond_basis_expansion = BondEncoder(
            atom_graph_cutoff=atom_graph_cutoff,
            bond_graph_cutoff=bond_graph_cutoff,
            num_radial=num_radial,
            cutoff_coeff=cutoff_coeff,
            learnable=learnable_rbf,
        )
        self.bond_embedding = paddle.nn.Linear(
            in_features=num_radial, out_features=bond_fea_dim, bias_attr=False
        )
        self.bond_weights_ag = paddle.nn.Linear(
            in_features=num_radial, out_features=atom_fea_dim, bias_attr=False
        )
        self.bond_weights_bg = paddle.nn.Linear(
            in_features=num_radial, out_features=bond_fea_dim, bias_attr=False
        )
        self.angle_basis_expansion = AngleEncoder(
            num_angular=num_angular, learnable=learnable_rbf
        )
        self.angle_embedding = paddle.nn.Linear(
            in_features=num_angular, out_features=angle_fea_dim, bias_attr=False
        )

        mlp_out_bias = kwargs.pop("mlp_out_bias", False)
        atom_graph_layers = [
            AtomConv(
                atom_fea_dim=atom_fea_dim,
                bond_fea_dim=bond_fea_dim,
                hidden_dim=atom_conv_hidden_dim,
                dropout=conv_dropout,
                use_mlp_out=True,
                mlp_out_bias=mlp_out_bias,
                resnet=True,
            )
            for _ in range(n_conv)
        ]
        self.atom_conv_layers = paddle.nn.LayerList(sublayers=atom_graph_layers)
        if update_bond:
            bond_graph_layers = [
                BondConv(
                    atom_fea_dim=atom_fea_dim,
                    bond_fea_dim=bond_fea_dim,
                    angle_fea_dim=angle_fea_dim,
                    hidden_dim=bond_conv_hidden_dim,
                    dropout=conv_dropout,
                    use_mlp_out=True,
                    mlp_out_bias=mlp_out_bias,
                    resnet=True,
                )
                for _ in range(n_conv - 1)
            ]
            self.bond_conv_layers = paddle.nn.LayerList(sublayers=bond_graph_layers)
        else:
            self.bond_conv_layers = [None for _ in range(n_conv - 1)]
        if update_angle:
            angle_layers = [
                AngleUpdate(
                    atom_fea_dim=atom_fea_dim,
                    bond_fea_dim=bond_fea_dim,
                    angle_fea_dim=angle_fea_dim,
                    hidden_dim=angle_layer_hidden_dim,
                    dropout=conv_dropout,
                    resnet=True,
                )
                for _ in range(n_conv - 1)
            ]
            self.angle_layers = paddle.nn.LayerList(sublayers=angle_layers)
        else:
            self.angle_layers = [None for _ in range(n_conv - 1)]
        self.site_wise = paddle.nn.Linear(in_features=atom_fea_dim, out_features=1)
        self.readout_norm = paddle.nn.LayerNorm(atom_fea_dim)
        self.mlp_first = mlp_first
        if mlp_first:
            self.read_out_type = "sum"
            input_dim = atom_fea_dim
            self.pooling = GraphPooling(average=False)
        elif read_out in {"attn", "weighted"}:
            self.read_out_type = "attn"
            num_heads = kwargs.pop("num_heads", 3)
            self.pooling = GraphAttentionReadOut(
                atom_fea_dim, num_head=num_heads, average=True
            )
            input_dim = atom_fea_dim * num_heads
        else:
            self.read_out_type = "ave"
            input_dim = atom_fea_dim
            self.pooling = GraphPooling(average=True)
        if kwargs.pop("final_mlp", "MLP") in {"normal", "MLP"}:
            self.mlp = MLP(
                input_dim=input_dim,
                hidden_dim=mlp_hidden_dims,
                output_dim=1,
                dropout=mlp_dropout,
            )
        else:
            self.mlp = paddle.nn.Sequential(
                GatedMLP(
                    input_dim=input_dim,
                    hidden_dim=mlp_hidden_dims,
                    output_dim=mlp_hidden_dims[-1],
                    dropout=mlp_dropout,
                ),
                paddle.nn.Linear(in_features=mlp_hidden_dims[-1], out_features=1),
            )
        if is_freeze:
            self.freeze_weights()
            logger.info("Some weights are frozen.")

        self.loss_type = loss_type
        if loss_type == "mse_loss":
            self.loss_fn = paddle.nn.MSELoss()
        elif loss_type == "smooth_l1_loss" or loss_type == "huber_loss":
            self.loss_fn = paddle.nn.SmoothL1Loss(delta=huber_loss_delta)
            self.huber_loss_delta = huber_loss_delta
        elif loss_type == "l1_loss":
            self.loss_fn = paddle.nn.L1Loss()
        else:
            raise ValueError(f"Unknown loss type {loss_type}.")

    def freeze_weights(self) -> None:
        for layer in [
            self.atom_embedding,
            self.bond_embedding,
            self.angle_embedding,
            self.bond_basis_expansion,
            self.angle_basis_expansion,
            self.atom_conv_layers[:3],
            self.bond_conv_layers,
            self.angle_layers,
        ]:
            for param in layer.parameters():
                param.stop_gradient = True

    def forward(self, data, return_loss=True, return_prediction=True):
        assert (
            return_loss or return_prediction
        ), "At least one of return_loss or return_prediction must be True."
        (
            energy,
            force,
            stress,
            magmom,
            site_energies,
            atom_feas,
            crystal_feas,
        ) = self._forward(data)

        pred_dict = {
            "energy_per_atom": energy,
            "force": force,
            "stress": stress,
            "magmom": magmom,
        }

        loss_dict = {}
        if return_loss:
            loss = 0.0
            for property_name in self.property_names:
                label = data[property_name]
                pred = pred_dict[property_name]

                valid_value_indices = ~paddle.isnan(label)
                valid_label = label[valid_value_indices]
                valid_pred = pred[valid_value_indices]

                if valid_label.numel() > 0:
                    loss_property = self.loss_fn(
                        input=valid_pred,
                        label=valid_label,
                    )

                    loss_dict[property_name] = loss_property
                    loss += loss_property * self.loss_weights_dict[property_name]
                # else:
                #     loss_dict[property_name] = 0.0
            loss_dict["loss"] = loss

        prediction = {}
        if return_prediction:
            for property_name in self.property_names:
                prediction[property_name] = pred_dict[property_name]
            if self.return_site_energies:
                prediction["site_energies"] = site_energies
            if self.return_atom_feas:
                prediction["atom_feas"] = atom_feas
            if self.return_crystal_feas:
                prediction["crystal_feas"] = crystal_feas

        return {"loss_dict": loss_dict, "pred_dict": prediction}

    def _forward(
        self,
        batch_data,
    ) -> dict[str, paddle.Tensor]:
        """Prepare graph topology and execute one differentiable tensor graph."""
        batch_data["graph"] = batch_data["graph"].tensor()
        graphs = batch_data["graph"]

        num_atoms = graphs.node_feat["num_atoms"].astype("int64")
        num_edges = graphs.edge_feat["num_edges"].astype("int64")
        atom_owners = graphs.graph_node_id.astype("int64")
        directed2undirected = graphs.edge_feat["directed2undirected"].astype("int64")
        undirected2directed = graphs.edge_feat["undirected2directed"].astype("int64")
        atomic_numbers = graphs.node_feat["atom_types"].astype("int64")
        # Make the geometry inputs differentiable leaves before the boundary.
        # Forces differentiate the energy w.r.t. cartesian positions, which are
        # derived from frac_coords; stress differentiates w.r.t. strains. Each
        # requested derivative needs its own leaf, otherwise paddle.grad finds
        # no path from the energy back to it.
        frac_coords = graphs.node_feat["frac_coords"].detach()
        lattice = graphs.node_feat["lattice"].detach()
        strains = paddle.zeros_like(lattice)
        if "force" in self.property_names:
            frac_coords.stop_gradient = False
        if "stress" in self.property_names:
            strains.stop_gradient = False
        if self.composition_model is None or isinstance(
            self.composition_model, AtomRef
        ):
            composition_energy = paddle.zeros(
                [num_atoms.shape[0], 1], dtype=frac_coords.dtype
            )
        else:
            composition_energy = self.composition_model(graphs)

        atom_graph = graphs.edge_feat["atom_graph"].astype("int64")
        num_atom_graph = graphs.edge_feat["num_atom_graph"].astype("int64")
        num_atoms_cumsum = paddle.cumsum(num_atoms)
        num_atoms_cumsum = paddle.concat(
            [paddle.zeros(1, dtype=num_atoms_cumsum.dtype), num_atoms_cumsum]
        )
        num_atoms_cumsum = num_atoms_cumsum[:-1]
        # Convert the index to the index of the overall graph
        atom_graph_offset = paddle.repeat_interleave(num_atoms_cumsum, num_atom_graph)
        atom_graph = atom_graph + atom_graph_offset[:, None]

        num_edges_cumsum = paddle.cumsum(num_edges)
        num_edges_cumsum = paddle.concat(
            [paddle.zeros(1, dtype=num_edges_cumsum.dtype), num_edges_cumsum]
        )
        num_edges_cumsum = num_edges_cumsum[:-1]

        undirected2directed_offset = paddle.repeat_interleave(
            num_edges_cumsum,
            graphs.edge_feat["undirected2directed_len"].astype("int64"),
        )
        undirected2directed = undirected2directed + undirected2directed_offset

        num_bond_graph = graphs.edge_feat["num_bond_graph"].astype("int64")
        bond_vec_index_offset = paddle.repeat_interleave(
            num_edges_cumsum, num_bond_graph
        )

        undirected2directed_len_cumsum = paddle.cumsum(
            graphs.edge_feat["undirected2directed_len"].astype("int64")
        )
        undirected2directed_len_cumsum = paddle.concat(
            [
                paddle.zeros(1, dtype=undirected2directed_len_cumsum.dtype),
                undirected2directed_len_cumsum,
            ]
        )
        undirected2directed_len_cumsum = undirected2directed_len_cumsum[:-1]

        bond_graph = graphs.edge_feat["bond_graph"].astype("int64")
        bond_vecs_i_index = bond_graph[:, 2] + bond_vec_index_offset
        bond_vecs_j_index = bond_graph[:, 4] + bond_vec_index_offset
        atom_offset = paddle.repeat_interleave(num_atoms_cumsum, num_bond_graph)
        bond_offset = paddle.repeat_interleave(
            undirected2directed_len_cumsum, num_bond_graph
        )
        bond_graph = paddle.stack(
            [
                bond_graph[:, 0] + atom_offset,
                bond_graph[:, 1] + bond_offset,
                bond_graph[:, 3] + bond_offset,
            ],
            axis=1,
        )

        offset_tmp = paddle.repeat_interleave(undirected2directed_len_cumsum, num_edges)
        directed2undirected = directed2undirected + offset_tmp

        (
            energy,
            force,
            stress,
            magmom,
            site_energies,
            atom_feas,
            crystal_feas,
        ) = self._runtime_forward(
            atomic_numbers=atomic_numbers,
            composition_fea=graphs.node_feat["composition_fea"],
            composition_energy=composition_energy,
            frac_coords=frac_coords,
            lattice=lattice,
            strains=strains,
            num_atoms=num_atoms,
            num_edges=num_edges,
            image=graphs.edge_feat["image"],
            batched_atom_graph=atom_graph,
            batched_bond_graph=bond_graph,
            bond_vecs_i_index=bond_vecs_i_index,
            bond_vecs_j_index=bond_vecs_j_index,
            undirected2directed=undirected2directed,
            atom_owners=atom_owners,
            directed2undirected=directed2undirected,
        )

        atoms_per_graph = paddle.bincount(x=atom_owners)
        if self.return_site_energies:
            site_energies = paddle.split(
                x=site_energies, num_or_sections=atoms_per_graph.tolist()
            )
        else:
            site_energies = None
        if self.return_atom_feas:
            atom_feas = paddle.split(
                x=atom_feas, num_or_sections=atoms_per_graph.tolist()
            )

        if self.return_site_energies and self.composition_model is not None:
            site_energy_shifts = self.composition_model.get_site_energies(graphs)
            site_energies = [
                (i + j) for i, j in zip(site_energies, site_energy_shifts, strict=True)
            ]

        return energy, force, stress, magmom, site_energies, atom_feas, crystal_feas

    @runtime_boundary("forward")
    def _runtime_forward(
        self,
        atomic_numbers,
        composition_fea,
        composition_energy,
        frac_coords,
        lattice,
        strains,
        num_atoms,
        num_edges,
        image,
        batched_atom_graph,
        batched_bond_graph,
        bond_vecs_i_index,
        bond_vecs_j_index,
        undirected2directed,
        atom_owners,
        directed2undirected,
    ):
        """Run geometry, message passing, readout, force and stress as one graph."""
        if "stress" in self.property_names:
            lattice = paddle.matmul(
                lattice, paddle.eye(3, dtype=lattice.dtype)[None, :, :] + strains
            )
            volumes = paddle.sum(
                lattice[:, 0] * paddle.cross(x=lattice[:, 1], y=lattice[:, 2], axis=-1),
                axis=-1,
            ).detach()
        else:
            volumes = paddle.ones([lattice.shape[0]], dtype=lattice.dtype)

        atom_positions = frac_to_cart_coords(
            frac_coords,
            num_atoms=num_atoms,
            lattices=lattice,
        )
        center = atom_positions[batched_atom_graph[:, 0]]
        neighbor = atom_positions[batched_atom_graph[:, 1]]
        lattice_edges = paddle.repeat_interleave(x=lattice, repeats=num_edges, axis=0)
        offset = paddle.bmm(image.unsqueeze(1), lattice_edges).squeeze(1)
        bond_vectors = center - neighbor - offset
        bond_lengths = paddle.linalg.norm(x=bond_vectors, axis=1)
        bond_vectors = bond_vectors / bond_lengths[:, None]

        undirected_bond_lengths = paddle.gather(
            x=bond_lengths, axis=0, index=undirected2directed
        )
        bond_bases_ag = self.bond_basis_expansion.rbf_expansion_ag(
            undirected_bond_lengths
        )
        bond_bases_bg = self.bond_basis_expansion.rbf_expansion_bg(
            undirected_bond_lengths
        )
        bond_vecs_i = paddle.gather(x=bond_vectors, axis=0, index=bond_vecs_i_index)
        bond_vecs_j = paddle.gather(x=bond_vectors, axis=0, index=bond_vecs_j_index)
        angle_bases = self.angle_basis_expansion(bond_vecs_i, bond_vecs_j)

        return self._compute(
            atomic_numbers=atomic_numbers,
            composition_fea=composition_fea,
            composition_energy=composition_energy,
            bond_bases_ag=bond_bases_ag,
            bond_bases_bg=bond_bases_bg,
            angle_bases=angle_bases,
            batched_atom_graph=batched_atom_graph,
            batched_bond_graph=batched_bond_graph,
            atom_owners=atom_owners,
            directed2undirected=directed2undirected,
            atom_positions=atom_positions,
            strains=strains,
            volumes=volumes,
        )

    def _compute(
        self,
        atomic_numbers,
        composition_fea,
        composition_energy,
        bond_bases_ag,
        bond_bases_bg,
        angle_bases,
        batched_atom_graph,
        batched_bond_graph,
        atom_owners,
        directed2undirected,
        atom_positions,  # atom_positions_list,
        strains,
        volumes,
    ) -> dict:
        """Get Energy, Force, Stress, Magmom associated with input graphs
        force = - d(Energy)/d(atom_positions)
        stress = 1/V * d(Energy)/d(strain).

        Returns:
            prediction (dict): containing the fields:
                e (Tensor) : energy of structures [batch_size, 1]
                f (Tensor) : force on atoms [num_batch_atoms, 3]
                s (Tensor) : stress of structure [3 * batch_size, 3]
                m (Tensor) : magnetic moments of sites [num_batch_atoms, 3]
        """

        atoms_per_graph = paddle.bincount(x=atom_owners)

        atom_feas = self.atom_embedding(atomic_numbers - 1)
        atom_feas_out = atom_feas
        magmom = paddle.zeros([atomic_numbers.shape[0], 1], dtype=atom_feas.dtype)
        bond_feas = self.bond_embedding(bond_bases_ag)
        bond_weights_ag = self.bond_weights_ag(bond_bases_ag)
        bond_weights_bg = self.bond_weights_bg(bond_bases_bg)
        angle_feas = self.angle_embedding(angle_bases)
        for idx, (atom_layer, bond_layer, angle_layer) in enumerate(
            zip(
                self.atom_conv_layers[:-1],
                self.bond_conv_layers,
                self.angle_layers,
            )
        ):
            atom_feas = atom_layer(
                atom_feas=atom_feas,
                bond_feas=bond_feas,
                bond_weights=bond_weights_ag,
                atom_graph=batched_atom_graph,
                directed2undirected=directed2undirected,
            )
            if bond_layer is not None:
                bond_feas = bond_layer(
                    atom_feas=atom_feas,
                    bond_feas=bond_feas,
                    bond_weights=bond_weights_bg,
                    angle_feas=angle_feas,
                    bond_graph=batched_bond_graph,
                )
                if angle_layer is not None:
                    angle_feas = angle_layer(
                        atom_feas=atom_feas,
                        bond_feas=bond_feas,
                        angle_feas=angle_feas,
                        bond_graph=batched_bond_graph,
                    )
            if idx == self.n_conv - 2:
                atom_feas_out = atom_feas
                if "magmom" in self.property_names:
                    magmom = paddle.abs(x=self.site_wise(atom_feas))
                else:
                    magmom = paddle.zeros(
                        [atomic_numbers.shape[0], 1], dtype=atom_feas.dtype
                    )
        atom_feas = self.atom_conv_layers[-1](
            atom_feas=atom_feas,
            bond_feas=bond_feas,
            bond_weights=bond_weights_ag,
            atom_graph=batched_atom_graph,
            directed2undirected=directed2undirected,
        )
        if self.readout_norm is not None:
            atom_feas = self.readout_norm(atom_feas)

        if self.mlp_first:
            energies = self.mlp(atom_feas)
            energy = self.pooling(
                energies, atom_owners, num_owner=atoms_per_graph.shape[0]
            )  # .reshape([-1])
            site_energies = energies.squeeze(axis=1)
            if self.return_crystal_feas:
                crystal_feas = self.pooling(
                    atom_feas, atom_owners, num_owner=atoms_per_graph.shape[0]
                )
            else:
                crystal_feas = paddle.zeros(
                    [0, self.atom_fea_dim], dtype=atom_feas.dtype
                )
        else:
            crystal_feas = self.pooling(
                atom_feas, atom_owners, num_owner=atoms_per_graph.shape[0]
            )
            energy = self.mlp(crystal_feas) * atoms_per_graph
            site_energies = paddle.zeros([0], dtype=energy.dtype)

        if "force" in self.property_names:
            # Stress differentiates the same energy afterwards, so the graph has
            # to survive this call. Without it the second paddle.grad reads freed
            # buffers and eager and compiled execution disagree on stress.
            force = -paddle.grad(
                outputs=energy.sum(),
                inputs=atom_positions,
                create_graph=self.training,
                retain_graph=self.training or "stress" in self.property_names,
            )[0]
        else:
            force = paddle.zeros_like(atom_positions)

        if "stress" in self.property_names:
            stress = paddle.grad(
                outputs=energy.sum(),
                inputs=strains,
                create_graph=self.training,
                retain_graph=self.training,
            )[0]
            scale = 1 / volumes * 160.21766208
            stress = stress * scale.reshape([-1, 1, 1])
        else:
            stress = paddle.zeros_like(strains)

        if self.is_intensive:
            energy /= atoms_per_graph.unsqueeze(-1).cast("float32")
        if isinstance(self.composition_model, AtomRef):
            energy += self.composition_model._get_energy(composition_fea)
        else:
            energy += composition_energy
        return (
            energy,
            force,
            stress,
            magmom,
            site_energies,
            atom_feas_out,
            crystal_feas,
        )

    def _prediction_to_numpy(self, prediction):
        for key in prediction:
            if isinstance(prediction[key], list):
                prediction[key] = [
                    prediction[key][i].numpy() for i in range(len(prediction[key]))
                ]
            else:
                prediction[key] = prediction[key].numpy()
            if key == "stress" and len(prediction["stress"].shape) == 3:
                prediction[key] = prediction[key][0]
            if key == "magmom" and isinstance(prediction[key], list):
                prediction[key] = prediction[key][0]
            if key == "energy_pre_atom" and isinstance(prediction[key], np.ndarray):
                prediction[key] = prediction[key][0]
        return prediction

    def predict(self, samples):
        is_list = isinstance(samples, list)
        samples = samples if is_list else [samples]

        results = []
        for sample in samples:
            result = self.forward(
                {"graph": sample},
                return_loss=False,
                return_prediction=True,
            )
            prediction = result["pred_dict"]
            prediction = self._prediction_to_numpy(prediction)
            results.append(prediction)
        return results if is_list else results[0]

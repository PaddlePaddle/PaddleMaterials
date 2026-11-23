import math

import paddle
from ppmat.models.sevennet import _keys as KEY
from ppmat.models.common.e3nn.o3 import Irreps, SphericalHarmonics
from ppmat.models.common.e3nn.util.jit import compile_mode
from ppmat.models.sevennet._const import AtomGraphDataType


@compile_mode("script")
class EdgePreprocess(paddle.nn.Layer):
    """
    preprocessing pos to edge vectors and edge lengths
    currently used in sevenn/scripts/deploy for lammps serial model
    """

    def __init__(self, is_stress: bool) -> None:
        super().__init__()
        self.is_stress = is_stress
        self._is_batch_data = True

    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:
        if self._is_batch_data:
            cell = data[KEY.CELL].view(-1, 3, 3)
        else:
            cell = data[KEY.CELL].view(3, 3)
        cell_shift = data[KEY.CELL_SHIFT]
        pos = data[KEY.POS]
        batch = data[KEY.BATCH]
        if self.is_stress:
            if self._is_batch_data:
                num_batch = int(paddle.max(batch).item()) + 1
                strain = paddle.zeros([num_batch, 3, 3], dtype=pos.dtype)
                strain.stop_gradient = False
                data["_strain"] = strain
                sym_strain = 0.5 * (strain + strain.transpose(-1, -2))
                pos = pos + paddle.bmm(pos.unsqueeze(-2), sym_strain[batch]).squeeze(-2)
                cell = cell + paddle.bmm(cell, sym_strain)
            else:
                strain = paddle.zeros([3, 3], dtype=pos.dtype)
                strain.stop_gradient = False
                data["_strain"] = strain
                sym_strain = 0.5 * (strain + strain.transpose(-1, -2))
                pos = pos + paddle.mm(pos, sym_strain)
                cell = cell + paddle.mm(cell, sym_strain)
        idx_src = data[KEY.EDGE_IDX][0]
        idx_dst = data[KEY.EDGE_IDX][1]
        edge_vec = pos[idx_dst] - pos[idx_src]
        if self._is_batch_data:
            edge_vec = edge_vec + paddle.einsum(
                "ni,nij->nj", cell_shift, cell[batch[idx_src]]
            )
        else:
            edge_vec = edge_vec + paddle.einsum(
                "ni,ij->nj", cell_shift, cell.squeeze(0)
            )
        data[KEY.EDGE_VEC] = edge_vec
        data[KEY.EDGE_LENGTH] = paddle.linalg.norm(edge_vec, axis=-1)
        return data


class BesselBasis(paddle.nn.Layer):
    """
    f : (*, 1) -> (*, bessel_basis_num)
    """

    def __init__(
        self,
        cutoff_length: float,
        bessel_basis_num: int = 8,
        trainable_coeff: bool = True,
    ) -> None:
        super().__init__()
        self.num_basis = bessel_basis_num
        self.prefactor = 2.0 / cutoff_length
        self.coeffs = paddle.to_tensor(
            [(n * math.pi / cutoff_length) for n in range(1, bessel_basis_num + 1)]
        )
        if trainable_coeff:
            self.coeffs = self.create_parameter(shape=self.coeffs.shape, default_initializer=paddle.nn.initializer.Assign(self.coeffs))

    def forward(self, r: paddle.Tensor) -> paddle.Tensor:
        ur = r.unsqueeze(-1)
        return self.prefactor * paddle.sin(self.coeffs * ur) / ur


class PolynomialCutoff(paddle.nn.Layer):
    """
    f : (*, 1) -> (*, 1)
    https://arxiv.org/pdf/2003.03123.pdf
    """

    def __init__(self, cutoff_length: float, poly_cut_p_value: int = 6) -> None:
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


class XPLORCutoff(paddle.nn.Layer):
    """
    https://hoomd-blue.readthedocs.io/en/latest/module-md-pair.html
    """

    def __init__(self, cutoff_length: float, cutoff_on: float) -> None:
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


@compile_mode("script")
class SphericalEncoding(paddle.nn.Layer):
    def __init__(
        self,
        lmax: int,
        parity: int = -1,
        normalization: str = "component",
        normalize: bool = True,
    ) -> None:
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


# @compile_mode("script")
class EdgeEmbedding(paddle.nn.Layer):
    """
    embedding layer of |r| by
    RadialBasis(|r|)*CutOff(|r|)
    f : (N_edge) -> (N_edge, basis_num)
    """

    def __init__(
        self,
        basis_module: paddle.nn.Layer,
        cutoff_module: paddle.nn.Layer,
        spherical_module: paddle.nn.Layer,
    ) -> None:
        super().__init__()
        self.basis_function = basis_module
        self.cutoff_function = cutoff_module
        self.spherical = spherical_module

    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:
        # SevenNet-0 uses float32 parameters.
        # Keep all edge features in the same dtype to avoid
        # float64 / float32 promotion inside Paddle e3nn TensorProduct.

        rvec = paddle.cast(
            data[KEY.EDGE_VEC],
            dtype="float32",
        )

        data[KEY.EDGE_VEC] = rvec

        r = paddle.linalg.norm(
            rvec,
            axis=-1,
        )

        r = paddle.cast(
            r,
            dtype="float32",
        )

        data[KEY.EDGE_LENGTH] = r

        basis = self.basis_function(r)
        basis = paddle.cast(
            basis,
            dtype="float32",
        )

        cutoff = self.cutoff_function(r)
        cutoff = paddle.cast(
            cutoff,
            dtype="float32",
        )

        if len(cutoff.shape) == 1:
            cutoff = cutoff.reshape(
                [-1, 1]
            )

        data[KEY.EDGE_EMBEDDING] = (
            basis * cutoff
        )

        spherical = self.spherical(
            rvec
        )

        spherical = paddle.cast(
            spherical,
            dtype="float32",
        )

        data[KEY.EDGE_ATTR] = spherical

        return data

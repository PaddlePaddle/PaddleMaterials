"""SevenNet radial features and energy derivatives."""
import math
import numpy as np
import paddle
from ppmat.models.common.e3nn.util.jit import compile_mode
from ppmat.models.common.graph_converter import FindPointsInSpheres
from . import config as KEY
from ppmat.utils.scatter import scatter_sum
from .layers import AtomGraphDataType


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


@compile_mode("script")
class ForceStressOutputFromEdge(paddle.nn.Layer):
    """
    Compute stress and force from edge.
    """

    def __init__(
        self,
        data_key_edge: str = KEY.EDGE_VEC,
        data_key_edge_idx: str = KEY.EDGE_IDX,
        data_key_energy: str = KEY.PRED_TOTAL_ENERGY,
        data_key_force: str = KEY.PRED_FORCE,
        data_key_stress: str = KEY.PRED_STRESS,
        data_key_cell_volume: str = KEY.CELL_VOLUME,
    ) -> None:
        super().__init__()
        self.key_edge = data_key_edge
        self.key_edge_idx = data_key_edge_idx
        self.key_energy = data_key_energy
        self.key_force = data_key_force
        self.key_stress = data_key_stress
        self.key_cell_volume = data_key_cell_volume
        self._is_batch_data = True

    def get_grad_key(self) -> str:
        return self.key_edge

    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:
        tot_num = paddle.sum(data[KEY.NUM_ATOMS])
        rij = data[self.key_edge]
        energy = [data[self.key_energy].sum()]
        edge_idx = data[self.key_edge_idx]
        grad = paddle.grad(
            outputs=energy, inputs=[rij], create_graph=self.training, allow_unused=True
        )
        fij = grad[0]
        if fij is None:
            return data

        source, destination = edge_idx[0], edge_idx[1]
        data[self.key_force] = scatter_sum(
            fij, source, dim=0, dim_size=tot_num
        ) - scatter_sum(fij, destination, dim=0, dim_size=tot_num)
        diagonal = rij * fij
        shear = paddle.stack(
            [rij[..., 0] * fij[..., 1], rij[..., 1] * fij[..., 2], rij[..., 2] * fij[..., 0]],
            axis=-1,
        )
        virial = paddle.concat([diagonal, shear], axis=-1)
        atomic_virial = scatter_sum(virial, destination, dim=0, dim_size=tot_num)
        if self._is_batch_data:
            virial = scatter_sum(atomic_virial, data[KEY.BATCH], dim=0)
        else:
            virial = atomic_virial.sum(axis=0)
        data[self.key_stress] = -virial / data[self.key_cell_volume].unsqueeze(-1)
        return data



class SevenNetGraphConverter(FindPointsInSpheres):
    """Reuse the repository's periodic graph builder with a fixed cutoff."""

    def __init__(self, cutoff=5.0, num_cpus=1):
        super().__init__(cutoff=cutoff, num_cpus=num_cpus)

    def get_graph_by_find_points_in_spheres(self, structure):
        graph = super().get_graph_by_find_points_in_spheres(structure)
        if graph is None:
            return self.build_pgl_graph(structure, [], [])

        # The shared converter may retry an isolated structure at a larger
        # radius. SevenNet's neighbor topology must use the model cutoff.
        keep = graph.edge_feat["bond_dist"] <= self.cutoff
        if not np.all(keep):
            return self.build_pgl_graph(
                structure, graph.edges[keep], graph.edge_feat["pbc_offset"][keep]
            )
        return graph

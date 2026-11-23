import paddle

from ppmat.utils.scatter import scatter
from ppmat.models.sevennet import _keys as KEY
from ppmat.models.common.e3nn.util.jit import compile_mode
from ppmat.models.sevennet._const import AtomGraphDataType

from .util import broadcast


@compile_mode("script")
class ForceOutput(paddle.nn.Layer):
    """
    works when pos.requires_grad_ is True
    """

    def __init__(
        self,
        data_key_pos: str = KEY.POS,
        data_key_energy: str = KEY.PRED_TOTAL_ENERGY,
        data_key_force: str = KEY.PRED_FORCE,
    ) -> None:
        super().__init__()
        self.key_pos = data_key_pos
        self.key_energy = data_key_energy
        self.key_force = data_key_force

    def get_grad_key(self) -> str:
        return self.key_pos

    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:
        pos_tensor = [data[self.key_pos]]
        energy = [data[self.key_energy].sum()]
        grad = paddle.grad(
            outputs=energy,
            inputs=pos_tensor,
            create_graph=self.training,
            allow_unused=True,
        )[0]
        if grad is not None:
            data[self.key_force] = paddle.neg(grad)
        return data


@compile_mode("script")
class ForceStressOutput(paddle.nn.Layer):
    """
    Compute stress and force from positions.
    Used in serial torchscipt models
    """

    def __init__(
        self,
        data_key_pos: str = KEY.POS,
        data_key_energy: str = KEY.PRED_TOTAL_ENERGY,
        data_key_force: str = KEY.PRED_FORCE,
        data_key_stress: str = KEY.PRED_STRESS,
        data_key_cell_volume: str = KEY.CELL_VOLUME,
    ) -> None:
        super().__init__()
        self.key_pos = data_key_pos
        self.key_energy = data_key_energy
        self.key_force = data_key_force
        self.key_stress = data_key_stress
        self.key_cell_volume = data_key_cell_volume
        self._is_batch_data = True

    def get_grad_key(self) -> str:
        return self.key_pos

    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:
        pos_tensor = data[self.key_pos]
        energy = [data[self.key_energy].sum()]
        grad = paddle.grad(
            outputs=energy,
            inputs=[pos_tensor, data["_strain"]],
            create_graph=self.training,
            allow_unused=True,
        )
        fgrad = grad[0]
        if fgrad is not None:
            data[self.key_force] = paddle.neg(fgrad)
        sgrad = grad[1]
        volume = data[self.key_cell_volume]
        vlim = 0.001
        if self._is_batch_data:
            volume[volume < vlim] = vlim
        elif volume < vlim:
            volume = paddle.tensor(vlim)
        if sgrad is not None:
            if self._is_batch_data:
                stress = sgrad / volume.view(-1, 1, 1)
                stress = paddle.neg(stress)
                virial_stress = paddle.vstack(
                    x=(
                        stress[:, 0, 0],
                        stress[:, 1, 1],
                        stress[:, 2, 2],
                        stress[:, 0, 1],
                        stress[:, 1, 2],
                        stress[:, 0, 2],
                    )
                )
                data[self.key_stress] = virial_stress.transpose(0, 1)
            else:
                stress = sgrad / volume
                stress = paddle.neg(stress)
                virial_stress = paddle.stack(
                    (
                        stress[0, 0],
                        stress[1, 1],
                        stress[2, 2],
                        stress[0, 1],
                        stress[1, 2],
                        stress[0, 2],
                    )
                )
                data[self.key_stress] = virial_stress
        return data


@compile_mode("script")
class ForceStressOutputFromEdge(paddle.nn.Layer):
    """
    Compute stress and force from edge.
    Used in parallel torchscipt models, and training
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
        if fij is not None:
            pf = paddle.zeros([tot_num, 3], dtype=fij.dtype)
            nf = paddle.zeros([tot_num, 3], dtype=fij.dtype)
            _edge_src = broadcast(edge_idx[0], fij, 0)
            _edge_dst = broadcast(edge_idx[1], fij, 0)
            """Not Support auto convert *.scatter_reduce_, please judge whether it is Pytorch API and convert by yourself"""
            pf = scatter(
                fij,
                edge_idx[0],
                dim=0,
                dim_size=tot_num,
                reduce="sum",
            )
            """Not Support auto convert *.scatter_reduce_, please judge whether it is Pytorch API and convert by yourself"""
            nf = scatter(
                fij,
                edge_idx[1],
                dim=0,
                dim_size=tot_num,
                reduce="sum",
            )
            data[self.key_force] = pf - nf
            diag = rij * fij
            s12 = rij[..., 0] * fij[..., 1]
            s23 = rij[..., 1] * fij[..., 2]
            s31 = rij[..., 2] * fij[..., 0]
            _virial = paddle.concat(
                [diag, s12.unsqueeze(-1), s23.unsqueeze(-1), s31.unsqueeze(-1)], axis=-1
            )
            _s = paddle.zeros([tot_num, 6], dtype=fij.dtype)
            _edge_dst6 = broadcast(edge_idx[1], _virial, 0)
            """Not Support auto convert *.scatter_reduce_, please judge whether it is Pytorch API and convert by yourself"""
            _s = scatter(
                _virial,
                edge_idx[1],
                dim=0,
                dim_size=tot_num,
                reduce="sum",
            )
            if self._is_batch_data:
                batch = data[KEY.BATCH]
                nbatch = int(paddle.max(batch).item()) + 1
                sout = paddle.zeros([nbatch, 6], dtype=_virial.dtype)
                _batch = broadcast(batch, _s, 0)
                """Not Support auto convert *.scatter_reduce_, please judge whether it is Pytorch API and convert by yourself"""
                sout = scatter(
                    _s,
                    batch,
                    dim=0,
                    dim_size=nbatch,
                    reduce="sum",
                )
            else:
                sout = paddle.sum(_s, axis=0)
            data[self.key_stress] = paddle.neg(sout) / data[
                self.key_cell_volume
            ].unsqueeze(-1)
        return data

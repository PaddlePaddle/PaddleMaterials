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

from .. import _keys as KEY
import paddle
from ..e3nn.util.jit import compile_mode
from .._const import AtomGraphDataType
from ppmat.utils.scatter import scatter


@compile_mode("script")
class ForceOutputFromEdge(paddle.nn.Module):
    """
    works when edge_vec.requires_grad_ is True
    """

    def __init__(
        self,
        data_key_edge_vec: str = KEY.EDGE_VEC,
        data_key_edge_idx: str = KEY.EDGE_IDX,
        data_key_energy: str = KEY.SCALED_ENERGY,
        data_key_force: str = KEY.SCALED_FORCE,
        data_key_num_atoms: str = KEY.NUM_ATOMS,
    ):
        super().__init__()
        self.key_edge_vec = data_key_edge_vec
        self.key_energy = data_key_energy
        self.key_force = data_key_force
        self.key_edge_idx = data_key_edge_idx
        self.key_num_atoms = data_key_num_atoms

    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:
        tot_num = paddle.sum(data[self.key_num_atoms])
        edge_idx = data[self.key_edge_idx]
        edge_vec_tensor = [data[self.key_edge_vec]]
        energy = [data[self.key_energy].sum()]
        dE_dr = paddle.grad(
            outputs=energy, inputs=edge_vec_tensor, create_graph=self.training
        )[0]
        if dE_dr is not None:
            force = paddle.zeros(tot_num, 3)
            force = scatter(dE_dr, edge_idx[0], dim=0, dim_size=tot_num)
            force -= scatter(dE_dr, edge_idx[1], dim=0, dim_size=tot_num)
            data[self.key_force] = force
        return data


@compile_mode("script")
class ForceOutput(paddle.nn.Module):
    """
    works when pos.requires_grad_ is True
    """

    def __init__(
        self,
        data_key_pos: str = KEY.POS,
        data_key_energy: str = KEY.SCALED_ENERGY,
        data_key_force: str = KEY.SCALED_FORCE,
    ):
        super().__init__()
        self.key_pos = data_key_pos
        self.key_energy = data_key_energy
        self.key_force = data_key_force

    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:
        pos_tensor = [data[self.key_pos]]
        energy = [data[self.key_energy].sum()]
        grad = paddle.grad(
            outputs=energy, inputs=pos_tensor, create_graph=self.training
        )[0]
        if grad is not None:
            data[self.key_force] = paddle.neg(grad)
        return data


@compile_mode("script")
class ForceStressOutput(paddle.nn.Module):
    def __init__(
        self,
        data_key_pos: str = KEY.POS,
        data_key_energy: str = KEY.SCALED_ENERGY,
        data_key_force: str = KEY.SCALED_FORCE,
        data_key_stress: str = KEY.SCALED_STRESS,
        data_key_cell_volume: str = KEY.CELL_VOLUME,
    ):
        super().__init__()
        self.key_pos = data_key_pos
        self.key_energy = data_key_energy
        self.key_force = data_key_force
        self.key_stress = data_key_stress
        self.key_cell_volume = data_key_cell_volume
        self._is_batch_data = True

    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:
        pos_tensor = data[self.key_pos]
        energy = data[self.key_energy].sum()
        volume = data[self.key_cell_volume]
        if not self.training:
            # Eval inference: accumulate gradients on all leaves (LayerNorm
            # weights included) via backward(), then read .grad.
            paddle.autograd.backward(energy)
            fgrad = pos_tensor.grad
            sgrad = data["_strain"].grad
        else:
            # Training path: paddle.grad with create_graph (second-order
            # graph).
            grad = paddle.grad(
                outputs=[energy],
                inputs=[pos_tensor, data["_strain"]],
                create_graph=True,
            )
            fgrad, sgrad = grad[0], grad[1]
        if fgrad is not None:
            data[self.key_force] = paddle.neg(fgrad)
        if sgrad is not None:
            if self._is_batch_data:
                stress = sgrad / volume.reshape(-1, 1, 1)
                stress = paddle.neg(stress)
                voigt_stress = paddle.vstack(
                    x=(
                        stress[:, 0, 0],
                        stress[:, 1, 1],
                        stress[:, 2, 2],
                        stress[:, 0, 1],
                        stress[:, 1, 2],
                        stress[:, 0, 2],
                    )
                )
                data[self.key_stress] = voigt_stress.transpose(0, 1)
            else:
                stress = sgrad / volume
                stress = paddle.neg(stress)
                voigt_stress = paddle.stack(
                    (
                        stress[0, 0],
                        stress[1, 1],
                        stress[2, 2],
                        stress[0, 1],
                        stress[1, 2],
                        stress[0, 2],
                    )
                )
                data[self.key_stress] = voigt_stress
        return data

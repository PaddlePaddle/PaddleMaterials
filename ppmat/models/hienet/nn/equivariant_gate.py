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

from typing import Callable, Dict

from .. import _keys as KEY
import paddle
from ..e3nn.nn import Gate
from ..e3nn.o3 import Irreps
from ..e3nn.util.jit import compile_mode
from .._const import AtomGraphDataType


@compile_mode("script")
class EquivariantGate(paddle.nn.Module):
    """
    wrapper of e3nn.nn Gate (equivariant-nonlinear gate for irreps)
    required irreps_in for Gate forward is computed after instantiation
    of this class
    see
    https://docs.e3nn.org/en/stable/api/nn/nn_gate.html
    in nequip, result of convolution and self-interaction linear2
    is directly used for irreps_gates

    Usage in NequIP
    irreps_x: Representation of lmax, fixed multiplicity applied irreps
    act_scalar/gate_dict: dictionary of parity and activation function
        depends on parity, the activation function is regulated (odd or even function)
    """

    def __init__(
        self,
        irreps_x: Irreps,
        act_scalar_dict: Dict[int, Callable],
        act_gate_dict: Dict[int, Callable],
        data_key_x: str = KEY.NODE_FEATURE,
    ):
        super().__init__()
        self.key_x = data_key_x
        parity_mapper = {"e": 1, "o": -1}
        act_scalar_dict = {parity_mapper[k]: v for k, v in act_scalar_dict.items()}
        act_gate_dict = {parity_mapper[k]: v for k, v in act_gate_dict.items()}
        irreps_gated_elem = []
        irreps_scalars_elem = []
        for mul, irreps in irreps_x:
            if irreps.l > 0:
                irreps_gated_elem.append((mul, irreps))
            else:
                irreps_scalars_elem.append((mul, irreps))
        irreps_scalars = Irreps(irreps_scalars_elem)
        irreps_gated = Irreps(irreps_gated_elem)
        irreps_gates_parity = 1 if "0e" in irreps_scalars else -1
        irreps_gates = Irreps(
            [(mul, (0, irreps_gates_parity)) for mul, _ in irreps_gated]
        )
        act_scalars = [act_scalar_dict[p] for _, (_, p) in irreps_scalars]
        act_gates = [act_gate_dict[p] for _, (_, p) in irreps_gates]
        self.gate = Gate(
            irreps_scalars, act_scalars, irreps_gates, act_gates, irreps_gated
        )

    def get_gate_irreps_in(self):
        """
        user must call this function to get proper irreps in for forward
        """
        return self.gate.irreps_in

    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:
        data[self.key_x] = self.gate(data[self.key_x])
        return data


@compile_mode("script")
class eComfEquivariantGate(paddle.nn.Module):
    def __init__(
        self,
        irreps_x: Irreps,
        act_scalar_dict: Dict[int, Callable],
        act_gate_dict: Dict[int, Callable],
        data_key_x: str = KEY.NODE_FEATURE,
    ):
        super().__init__()
        self.key_x = data_key_x
        parity_mapper = {"e": 1, "o": -1}
        act_scalar_dict = {parity_mapper[k]: v for k, v in act_scalar_dict.items()}
        act_gate_dict = {parity_mapper[k]: v for k, v in act_gate_dict.items()}
        irreps_gated_elem = []
        irreps_scalars_elem = []
        for mul, irreps in irreps_x:
            if irreps.l > 0:
                irreps_gated_elem.append((mul, irreps))
            else:
                irreps_scalars_elem.append((mul, irreps))
        irreps_scalars = Irreps(irreps_scalars_elem)
        irreps_gated = Irreps(irreps_gated_elem)
        irreps_gates_parity = 1 if "0e" in irreps_scalars else -1
        irreps_gates = Irreps(
            [(mul, (0, irreps_gates_parity)) for mul, _ in irreps_gated]
        )
        act_scalars = [act_scalar_dict[p] for _, (_, p) in irreps_scalars]
        act_gates = [act_gate_dict[p] for _, (_, p) in irreps_gates]
        self.gate = Gate(
            irreps_scalars, act_scalars, irreps_gates, act_gates, irreps_gated
        )

    def get_gate_irreps_in(self):
        """
        user must call this function to get proper irreps in for forward
        """
        return self.gate.irreps_in

    def forward(self, node_feature):
        return self.gate(node_feature)

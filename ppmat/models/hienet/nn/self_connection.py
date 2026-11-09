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
from ..e3nn.o3 import FullyConnectedTensorProduct, Irreps, Linear
from ..e3nn.util.jit import compile_mode
from .._const import AtomGraphDataType


@compile_mode("script")
class SelfConnectionIntro(paddle.nn.Module):
    """
    do TensorProduct of x and some data(here attribute of x)
    and save it (to concatenate updated x at SelfConnectionOutro)
    """

    def __init__(
        self,
        irreps_x: Irreps,
        irreps_operand: Irreps,
        irreps_out: Irreps,
        data_key_x: str = KEY.NODE_FEATURE,
        data_key_operand: str = KEY.NODE_ATTR,
        **kwargs
    ):
        super().__init__()
        self.fc_tensor_product = FullyConnectedTensorProduct(
            irreps_x, irreps_operand, irreps_out
        )
        self.key_x = data_key_x
        self.key_operand = data_key_operand

    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:
        data[KEY.SELF_CONNECTION_TEMP] = self.fc_tensor_product(
            data[self.key_x], data[self.key_operand]
        )
        return data


@compile_mode("script")
class SelfConnectionLinearIntro(paddle.nn.Module):
    """
    Linear style self connection update
    """

    def __init__(
        self,
        irreps_x: Irreps,
        irreps_out: Irreps,
        data_key_x: str = KEY.NODE_FEATURE,
        **kwargs
    ):
        super().__init__()
        self.linear = Linear(irreps_x, irreps_out)
        self.key_x = data_key_x

    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:
        data[KEY.SELF_CONNECTION_TEMP] = self.linear(data[self.key_x])
        return data


@compile_mode("script")
class SelfConnectionOutro(paddle.nn.Module):
    """
    do TensorProduct of x and some data(here attribute of x)
    and save it (to concatenate updated x at SelfConnectionOutro)
    """

    def __init__(self, data_key_x: str = KEY.NODE_FEATURE):
        super().__init__()
        self.key_x = data_key_x

    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:
        data[self.key_x] = data[self.key_x] + data[KEY.SELF_CONNECTION_TEMP]
        del data[KEY.SELF_CONNECTION_TEMP]
        return data

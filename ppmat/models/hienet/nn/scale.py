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

from typing import List

from .. import _keys as KEY
import paddle
from ..e3nn.util.jit import compile_mode
from .._const import AtomGraphDataType


@compile_mode("script")
class Rescale(paddle.nn.Module):
    """
    Scaling and shifting energy (and automatically force and stress)
    """

    def __init__(
        self,
        shift: float,
        scale: float,
        data_key_in=KEY.SCALED_ATOMIC_ENERGY,
        data_key_out=KEY.ATOMIC_ENERGY,
        train_shift_scale: bool = False,
    ):
        super().__init__()
        self.shift = paddle.nn.Parameter(
            paddle.FloatTensor([shift]), requires_grad=train_shift_scale
        )
        self.scale = paddle.nn.Parameter(
            paddle.FloatTensor([scale]), requires_grad=train_shift_scale
        )
        self.key_input = data_key_in
        self.key_output = data_key_out

    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:
        data[self.key_output] = data[self.key_input] * self.scale + self.shift
        return data


@compile_mode("script")
class SpeciesWiseRescale(paddle.nn.Module):
    """
    Scaling and shifting energy (and automatically force and stress)
    """

    def __init__(
        self,
        shift: List[float],
        scale: List[float],
        data_key_in=KEY.SCALED_ATOMIC_ENERGY,
        data_key_out=KEY.ATOMIC_ENERGY,
        data_key_indicies=KEY.ATOM_TYPE,
        train_shift_scale: bool = False,
    ):
        super().__init__()
        self.shift = paddle.nn.Parameter(
            paddle.FloatTensor(shift), requires_grad=train_shift_scale
        )
        self.scale = paddle.nn.Parameter(
            paddle.FloatTensor(scale), requires_grad=train_shift_scale
        )
        self.key_input = data_key_in
        self.key_output = data_key_out
        self.key_indicies = data_key_indicies

    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:
        indicies = data[self.key_indicies]
        data[self.key_output] = data[self.key_input] * self.scale[indicies].reshape(
            -1, 1
        ) + self.shift[indicies].reshape(-1, 1)
        return data

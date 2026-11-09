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

import warnings
from collections import OrderedDict
from typing import Dict

from .. import _keys as KEY
import paddle
from ..e3nn.util.jit import compile_mode
from .._const import AtomGraphDataType


@compile_mode("script")
class AtomGraphSequential(paddle.nn.Sequential):
    """
    same as nn.Sequential but with type notation
    see
    https://github.com/pytorch/pytorch/issues/52588
    """

    def __init__(
        self,
        modules: Dict[str, paddle.nn.Module],
        cutoff: float = 0.0,
        type_map: Dict[int, int] = {(-1): -1},
    ):
        if type(modules) != OrderedDict:
            modules = OrderedDict(modules)
        self.cutoff = cutoff
        self.type_map = type_map
        if cutoff == 0.0:
            warnings.warn("cutoff is 0.0 or not given", UserWarning)
        if type_map == {(-1): -1}:
            warnings.warn("type_map is not given", UserWarning)
        super().__init__(modules)

    def set_is_batch_data(self, flag: bool):
        for module in self:
            try:
                module._is_batch_data = flag
            except AttributeError:
                pass

    def get_irreps_in(self, modlue_name: str, attr_key: str = "irreps_in"):
        tg_module = self._modules[modlue_name]
        for m in tg_module.modules():
            try:
                return repr(m.__getattribute__(attr_key))
            except AttributeError:
                pass
        return None

    def prepand_module(self, key: str, module: paddle.nn.Module):
        self._modules.update({key: module})
        self._modules.move_to_end(key, last=False)

    def replace_module(self, key: str, module: paddle.nn.Module):
        self._modules.update({key: module})

    def delete_module_by_key(self, key: str):
        if key in self._modules.keys():
            del self._modules[key]

    def to_onehot_idx(self, data: AtomGraphDataType) -> AtomGraphDataType:
        """
        User must call this function first before the forward
        if the data is not one-hot encoded
        """
        if self.type_map is {(-1): -1}:
            raise ValueError("type_map is not set")
        zt = data[KEY.NODE_FEATURE].cast("int64")
        # Pure-tensor lookup; no per-element host reads.
        lut = [0] * (max(self.type_map) + 1)
        for z, idx in self.type_map.items():
            lut[z] = idx
        data[KEY.NODE_FEATURE] = paddle.gather(
            paddle.to_tensor(lut, dtype="int64", place=zt.place), zt
        )
        return data

    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:
        """
        type_map is a dict of {atomic_number: one_hot_idx}
        """
        for module in self:
            data = module(data)
        return data

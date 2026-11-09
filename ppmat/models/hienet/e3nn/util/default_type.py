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

from typing import Optional, Tuple

import paddle


def torch_get_default_tensor_type() -> str:
    return str(paddle.empty(0).dtype)


def _torch_get_default_dtype() -> paddle.dtype:
    """A torchscript-compatible version of torch.get_default_dtype()"""
    return paddle.empty(0).dtype


def torch_get_default_device() -> paddle.device:
    return paddle.empty(0).device


def explicit_default_types(
    dtype: Optional[paddle.dtype], device: Optional[paddle.device]
) -> Tuple[paddle.dtype, paddle.device]:
    """A torchscript-compatible type resolver"""
    if dtype is None:
        dtype = _torch_get_default_dtype()
    if device is None:
        device = torch_get_default_device()
    return dtype, device

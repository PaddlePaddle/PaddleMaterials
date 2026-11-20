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

from typing import NamedTuple, Optional


class Chunk(NamedTuple):
    mul: int
    dim: int
    slice: Optional[slice] = None


class Path(NamedTuple):
    input_1_slice: Chunk
    input_2_slice: Chunk
    output_slice: Chunk

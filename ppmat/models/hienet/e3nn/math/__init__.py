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

from ._linalg import complete_basis, direct_sum, orthonormalize
from ._normalize_activation import moment, normalize2mom
from ._soft_unit_step import soft_unit_step
from ._soft_one_hot_linspace import soft_one_hot_linspace
from ._reduce import germinate_formulas, reduce_permutation
from ._bessel import bessel

__all__ = [
    "complete_basis",
    "direct_sum",
    "orthonormalize",
    "moment",
    "normalize2mom",
    "soft_unit_step",
    "bessel",
    "soft_one_hot_linspace",
    "germinate_formulas",
    "reduce_permutation",
]

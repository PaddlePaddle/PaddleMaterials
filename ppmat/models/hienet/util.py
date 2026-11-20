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

from typing import Union

from ppmat.models.common.e3nn.o3 import FullTensorProduct
from ppmat.models.common.e3nn.o3 import Irreps


def infer_irreps_out(
    irreps_x: Irreps,
    irreps_operand: Irreps,
    drop_l: Union[bool, int] = False,
    parity_mode: str = "full",
    fix_multiplicity: Union[bool, int] = False,
) -> Irreps:
    assert parity_mode in ["full", "even", "sph"]
    irreps_out = FullTensorProduct(irreps_x, irreps_operand).irreps_out.simplify()
    new_irreps_elem = []
    for mul, (l, p) in irreps_out:
        elem = mul, (l, p)
        if drop_l is not False and l > drop_l:
            continue
        if parity_mode == "even" and p == -1:
            continue
        elif parity_mode == "sph" and p != (-1) ** l:
            continue
        if fix_multiplicity:
            elem = fix_multiplicity, (l, p)
        new_irreps_elem.append(elem)
    return Irreps(new_irreps_elem)

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

"""Allows for clean lookup of Irreducible representations of :math:`O(3)`

Examples
--------
Create a scalar representation (:math:`l=0`) of even parity.

>>> from ppmat.models.hienet.e3nn.o3 import irrep
>>> irrep.l0e == Irrep("0e")
True

>>> from ppmat.models.hienet.e3nn.o3.irrep import l1o, l2o
>>> l1o + l2o == Irrep("1o") + Irrep("2o")
True
"""
from .._irreps import Irrep


def __getattr__(name: str) -> Irrep:
    """Creates an Irreps obeject by reflection

    Parameters
    ----------
    name : string
        the o3 object name prefixed by l. Example: l1o == Irrep("1o")

    Returns
    -------
    `e3nn.o3.Irrep`
        irreducible representation of :math:`O(3)`
    """
    prefix, *ir = name
    if prefix != "l" or not ir:
        raise AttributeError(f"'e3nn.o3.irrep' module has no attribute '{name}'")
    try:
        return Irrep("".join(ir))
    except (ValueError, AssertionError):
        raise AttributeError(f"'e3nn.o3.irrep' module has no attribute '{name}'")

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

from typing import Dict, List

from .. import _keys as KEY
import numpy as np
import paddle
from ase.symbols import symbols2numbers
from ..e3nn.util.jit import compile_mode
from .._const import AtomGraphDataType


@compile_mode("script")
class CGCNNEmbedding(paddle.nn.Module):
    """
    x : tensor of shape (N, 1)
    x_after : tensor of shape (N, num_classes)
    It overwrite data_key_x
    and saves input to data_key_save and output to data_key_additional
    I know this is strange but it is for compatibility with previous version
    and to specie wise shift scale work
    ex) [0 1 1 0] -> [[1, 0] [0, 1] [0, 1] [1, 0]] (num_classes = 2)
    """

    def __init__(
        self,
        data_key_x: str = KEY.NODE_FEATURE,
        data_key_save: str = KEY.ATOM_TYPE,
        data_key_additional: str = KEY.NODE_ATTR,
        features="cgcnn",
    ):
        super().__init__()
        self.key_x = data_key_x
        self.key_save = data_key_save
        self.key_additional_output = data_key_additional
        element_atomic_numbers = {
            (0): 89,
            (1): 47,
            (2): 13,
            (3): 18,
            (4): 33,
            (5): 79,
            (6): 5,
            (7): 56,
            (8): 4,
            (9): 83,
            (10): 35,
            (11): 6,
            (12): 20,
            (13): 48,
            (14): 58,
            (15): 17,
            (16): 27,
            (17): 24,
            (18): 55,
            (19): 29,
            (20): 66,
            (21): 68,
            (22): 63,
            (23): 9,
            (24): 26,
            (25): 31,
            (26): 64,
            (27): 32,
            (28): 1,
            (29): 2,
            (30): 72,
            (31): 80,
            (32): 67,
            (33): 53,
            (34): 49,
            (35): 77,
            (36): 19,
            (37): 36,
            (38): 57,
            (39): 3,
            (40): 71,
            (41): 12,
            (42): 25,
            (43): 42,
            (44): 7,
            (45): 11,
            (46): 41,
            (47): 60,
            (48): 10,
            (49): 28,
            (50): 93,
            (51): 8,
            (52): 76,
            (53): 15,
            (54): 91,
            (55): 82,
            (56): 46,
            (57): 61,
            (58): 59,
            (59): 78,
            (60): 94,
            (61): 37,
            (62): 75,
            (63): 45,
            (64): 44,
            (65): 16,
            (66): 51,
            (67): 21,
            (68): 34,
            (69): 14,
            (70): 62,
            (71): 50,
            (72): 38,
            (73): 73,
            (74): 65,
            (75): 43,
            (76): 52,
            (77): 90,
            (78): 22,
            (79): 81,
            (80): 69,
            (81): 92,
            (82): 23,
            (83): 74,
            (84): 54,
            (85): 39,
            (86): 70,
            (87): 30,
            (88): 40,
        }
        element_atomic_numbers = paddle.tensor(list(element_atomic_numbers.values()))
        self.register_buffer("element_atomic_numbers", element_atomic_numbers)
        self.features = features

    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:
        inp = data[self.key_x]
        atomic_nums = self.element_atomic_numbers[inp]
        embd = paddle.tensor(
            np.array(
                [
                    get_node_attributes(species, atom_features=self.features)
                    for species in atomic_numbers_to_symbols(atomic_nums.tolist())
                ]
            ),
            device=inp.device,
        )
        embd = embd.float()
        data[self.key_x] = embd
        if self.key_additional_output is not None:
            data[self.key_additional_output] = embd
        if self.key_save is not None:
            data[self.key_save] = inp
        return data


@compile_mode("script")
class OnehotEmbedding(paddle.nn.Module):
    """
    x : tensor of shape (N, 1)
    x_after : tensor of shape (N, num_classes)
    It overwrite data_key_x
    and saves input to data_key_save and output to data_key_additional
    I know this is strange but it is for compatibility with previous version
    and to specie wise shift scale work
    ex) [0 1 1 0] -> [[1, 0] [0, 1] [0, 1] [1, 0]] (num_classes = 2)
    """

    def __init__(
        self,
        num_classes: int,
        data_key_x: str = KEY.NODE_FEATURE,
        data_key_save: str = KEY.ATOM_TYPE,
        data_key_additional: str = KEY.NODE_ATTR,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.key_x = data_key_x
        self.key_save = data_key_save
        self.key_additional_output = data_key_additional

    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:
        inp = data[self.key_x]
        embd = paddle.nn.functional.one_hot(inp, self.num_classes)
        embd = embd.float()
        data[self.key_x] = embd
        if self.key_additional_output is not None:
            data[self.key_additional_output] = embd
        if self.key_save is not None:
            data[self.key_save] = inp
        return data


def get_type_mapper_from_specie(specie_list: List[str]):
    """
    from ['Hf', 'O']
    return {72: 0, 16: 1}
    """
    specie_list = sorted(specie_list)
    type_map = {}
    unique_counter = 0
    for specie in specie_list:
        atomic_num = symbols2numbers(specie)[0]
        if atomic_num in type_map:
            continue
        type_map[atomic_num] = unique_counter
        unique_counter += 1
    return type_map


def one_hot_atom_embedding(atomic_numbers: List[int], type_map: Dict[int, int]):
    """
    atomic numbers from ase.get_atomic_numbers
    type_map from get_type_mapper_from_specie()
    """
    num_classes = len(type_map)
    try:
        type_numbers = paddle.LongTensor([type_map[num] for num in atomic_numbers])
    except KeyError as e:
        raise ValueError(f"Atomic number {e.args[0]} is not expected")
    embd = paddle.nn.functional.one_hot(type_numbers, num_classes)
    embd = embd.to(paddle.get_default_dtype())
    return embd

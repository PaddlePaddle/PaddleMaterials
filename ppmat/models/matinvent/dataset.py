# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""RL-side data pipeline for MatInvent.

Generated structures carry the reward of the round that produced them, so the
samples follow the ``structure_array`` contract of the framework datasets and
keep ``reward`` at the top level of the batch.
"""

from typing import Callable
from typing import List

import numpy as np
import paddle
import paddle.io as io
from pymatgen.core.structure import Structure

from ppmat.datasets.custom_data_type import ConcatData
from ppmat.utils import logger


def to_tensor_batch(batch):
    """Move the numpy leaves produced by ``DefaultCollator`` onto the device."""
    return paddle.utils.map_structure(
        lambda value: paddle.to_tensor(value)
        if isinstance(value, np.ndarray)
        else value,
        batch,
    )


class RLDataset(io.Dataset):
    """Generated structures labelled with the reward of their round."""

    def __init__(
        self,
        structures: List[Structure],
        rewards: np.ndarray,
        transforms: Callable = None,
    ):
        if len(structures) != len(rewards):
            raise ValueError(
                f"structures and rewards must have the same length, but got "
                f"{len(structures)} and {len(rewards)}"
            )
        self.structures = structures
        self.rewards = rewards
        self.transforms = transforms
        logger.info(f"Created RLDataset with {len(structures)} samples")

    def __len__(self):
        return len(self.structures)

    def __getitem__(self, idx):
        structure = self.structures[idx]
        lattice_parameters = structure.lattice.parameters
        data = {
            "structure_array": {
                "frac_coords": ConcatData(
                    np.array(structure.frac_coords, dtype=np.float32)
                ),
                "atom_types": ConcatData(
                    np.array(structure.atomic_numbers, dtype=np.int64)
                ),
                "lattice": ConcatData(
                    structure.lattice.matrix.astype(np.float32).reshape(1, 3, 3)
                ),
                "lengths": ConcatData(
                    np.array(lattice_parameters[:3], dtype=np.float32).reshape(1, 3)
                ),
                "angles": ConcatData(
                    np.array(lattice_parameters[3:], dtype=np.float32).reshape(1, 3)
                ),
                "num_atoms": len(structure),
            },
            "reward": np.float32(self.rewards[idx]),
        }
        if self.transforms is not None:
            data = self.transforms(data)
        return data

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

from typing import List

import paddle
from pymatgen.core.structure import Structure

from ppmat.datasets.build_structure import BuildStructure
from ppmat.datasets.num_atom_crystal_dataset import NumAtomsCrystalDataset
from ppmat.models.matinvent.dataset import _sample_mp20_num_atoms
from ppmat.utils import logger

# Reference DiffCSP samples atom types over atomic numbers 1..100.
_MAX_ATOMIC_NUM = 100


class MatInventSampler:
    """Prepare RL inputs and parse samples from the current in-memory agent."""

    def __init__(
        self,
        model_type: str,
        batch_size: int,
        num_batches: int,
        num_inference_steps: int,
        build_structure_cfg: dict,
        atom_count_distribution: str = "ALEX_MP_20",
    ):
        if model_type not in ("diffcsp", "mattergen"):
            raise ValueError(f"Unsupported MatInvent model type: {model_type}")
        self.model_type = model_type
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.num_inference_steps = num_inference_steps
        self.atom_count_distribution = atom_count_distribution
        self.structure_options = dict(build_structure_cfg)
        self.structure_format = self.structure_options.pop("format")

    @paddle.no_grad()
    def generate(self, model) -> List[Structure]:
        structures = []
        for _ in range(self.num_batches):
            counts = (
                _sample_mp20_num_atoms(self.batch_size)
                if self.atom_count_distribution == "MP_20"
                else NumAtomsCrystalDataset(
                    total_num=self.batch_size, dist_name=self.atom_count_distribution
                ).num_atoms
            )
            num_atoms = paddle.to_tensor(counts, dtype="int64")
            inputs = {"num_atoms": num_atoms}
            if self.model_type == "diffcsp":
                inputs["atom_types"] = paddle.randint(
                    low=1,
                    high=_MAX_ATOMIC_NUM + 1,
                    shape=[int(num_atoms.sum().item())],
                    dtype="int64",
                )
            sampled = model.sample(
                {"structure_array": inputs},
                num_inference_steps=self.num_inference_steps,
            )
            for result in sampled["result"]:
                try:
                    structure = BuildStructure.build_one(
                        result, self.structure_format, **self.structure_options
                    )
                except ValueError as error:
                    logger.warning(f"Failed to parse a sampled structure: {error}")
                    continue
                if structure is not None:
                    structures.append(structure)
        return structures

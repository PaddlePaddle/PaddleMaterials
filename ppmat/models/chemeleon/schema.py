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

# Adapted from https://github.com/hspark1212/chemeleon

"""Trajectory schema for sampling runs (pydantic-free).

TrajectoryStep holds the per-timestep tensors; TrajectoryContainer maps
t -> step in an OrderedDict and get_atoms() converts one timestep to a
list of sorted ASE Atoms with atom types above 103 clamped to the dummy
type 0.
"""

from collections import OrderedDict
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

import paddle
from ase import Atoms
from ase.build.tools import sort


class TrajectoryStep:
    def __init__(
        self,
        num_atoms: paddle.Tensor,
        atom_types: paddle.Tensor,
        frac_coords: paddle.Tensor,
        lattices: paddle.Tensor,
        batch_idx: paddle.Tensor,
        atom_types_probs: Optional[paddle.Tensor] = None,
    ):
        self.num_atoms = num_atoms
        self.atom_types = atom_types
        self.frac_coords = frac_coords
        self.lattices = lattices
        self.batch_idx = batch_idx
        self.atom_types_probs = atom_types_probs


class TrajectoryContainer:
    """
    key is the time step, value is a TrajectoryStep
    """

    def __init__(self, total_steps: int):
        self.trajectory_continaer: Dict[int, TrajectoryStep] = OrderedDict()
        self.total_steps = total_steps
        for t in range(self.total_steps):
            self.trajectory_continaer[t] = None

    def __setitem__(self, t: int, step: TrajectoryStep) -> None:
        self.trajectory_continaer[t] = step

    def __getitem__(self, t: int) -> TrajectoryStep:
        if t == -1:
            t = self.total_steps
        return self.trajectory_continaer[t]

    def __len__(self) -> int:
        return len(self.trajectory_continaer)

    def __iter__(self):
        return iter(self.trajectory_continaer)

    def get_atoms(self, t: int = 0, idx: int = None) -> Union[Atoms, List[Atoms]]:
        trajectory_step = self[t]
        # if atom type is greater than 103, set it to 0
        trajectory_step.atom_types = paddle.where(
            trajectory_step.atom_types <= 103,
            trajectory_step.atom_types,
            paddle.zeros_like(trajectory_step.atom_types),
        )
        split_atom_types = paddle.split(
            trajectory_step.atom_types,
            num_or_sections=trajectory_step.num_atoms.tolist(),
        )
        split_frac_coords = paddle.split(
            trajectory_step.frac_coords,
            num_or_sections=trajectory_step.num_atoms.tolist(),
        )
        atoms_list = []
        for i, (frac_coords, atom_types) in enumerate(
            zip(split_frac_coords, split_atom_types)
        ):
            atoms = Atoms(
                numbers=atom_types.detach().cpu().numpy(),
                cell=trajectory_step.lattices[i].detach().cpu().numpy(),
                pbc=True,
            )
            positions = frac_coords.detach().cpu().numpy()
            atoms.set_scaled_positions(positions)
            atoms_list.append(sort(atoms))
        if idx is None:
            return atoms_list
        return atoms_list[idx]

    def get_trajectory(self, idx: int = None) -> Union[List[Atoms], List[List[Atoms]]]:
        if idx is None:
            return [self.get_atoms(t) for t in range(self.total_steps + 1)]
        return [self.get_atoms(t, idx) for t in range(self.total_steps + 1)]

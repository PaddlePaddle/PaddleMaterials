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

"""HIENetGraphConverter: ase Atoms -> ppmat Data with the HIENet contract keys.

Graph construction: ase ``primitive_neighbor_list("ijDS", ...)`` followed by
trivial-self-loop removal.

Label boundary: stress is stored as the RAW 3x3 (eV/A^3, no negation,
no Voigt reorder). Negation / Voigt reorder are model loss semantics and
belong to the HIENet main class; exactly one side transforms.
"""

import ase
import numpy as np
from ase.neighborlist import primitive_neighbor_list
from p_tqdm import p_map
from pymatgen.core.structure import Structure

from ppmat.datasets.geometric_data_type import Data


class HIENetGraphConverter:
    """Convert a structure into a ppmat Data (HIENet contract keys).

    Accepts a single ase Atoms (the original HIENet entry type), a single
    pymatgen Structure (converted to ase Atoms first, as in the m3gnet
    converter), or a list of either. Attributes: atom_types [N] int64,
    cart_coords [N,3] float32, lattice [1,3,3] float32, num_atoms [1]
    int64, edge_index [2,E] int64, pbc_offset [E,3] int64; plus energy
    [1] float64, force [N,3] float64, stress [3,3] float64 when calc
    results are available.
    """

    def __init__(self, cutoff: float = 5.0, num_cpus: int | None = None):
        self.cutoff = cutoff
        self.num_cpus = num_cpus

    def __call__(self, atoms):
        if isinstance(atoms, Structure):
            atoms = atoms.to_ase_atoms()
            # The pymatgen round-trip may restore a partial
            # SinglePointCalculator from site properties; Structure inputs
            # yield structure-only Data, labels attach dataset-side.
            atoms.calc = None
        elif isinstance(atoms, (list, tuple)):
            return p_map(self.__call__, list(atoms), num_cpus=self.num_cpus)
        if not isinstance(atoms, ase.Atoms):
            raise TypeError(
                "The input must be an ase.Atoms, a pymatgen.Structure, "
                "or a list of them."
            )
        pos = atoms.get_positions()
        cell = np.array(atoms.get_cell())
        edge_src, edge_dst, edge_vec, shifts = primitive_neighbor_list(
            "ijDS", atoms.get_pbc(), cell, pos, self.cutoff, self_interaction=True
        )
        is_zero_idx = np.all(edge_vec == 0, axis=1)
        is_self_idx = edge_src == edge_dst
        non_trivials = ~(is_zero_idx & is_self_idx)
        cell_shift = np.array(shifts[non_trivials])
        edge_src = edge_src[non_trivials]
        edge_dst = edge_dst[non_trivials]
        edge_idx = np.array([edge_src, edge_dst])
        atomic_numbers = np.asarray(atoms.get_atomic_numbers(), dtype=np.int64)
        num_atoms = len(atomic_numbers)
        data = Data(
            atom_types=atomic_numbers,
            cart_coords=np.asarray(pos, dtype=np.float32),
            lattice=np.asarray(cell, dtype=np.float32).reshape(1, 3, 3),
            num_atoms=np.array([num_atoms], dtype=np.int64),
            edge_index=edge_idx.astype(np.int64),
            pbc_offset=cell_shift.astype(np.int64),
            num_nodes=num_atoms,
        )
        if atoms.calc is not None:
            try:
                y_energy = atoms.get_potential_energy(force_consistent=True)
            except NotImplementedError:
                y_energy = atoms.get_potential_energy()
            y_force = atoms.get_forces(apply_constraint=False)
            try:
                # Raw 3x3 in eV/A^3; negation/Voigt reorder live in the model loss.
                y_stress = atoms.get_stress(voigt=False)
            except RuntimeError:
                y_stress = None
            data["energy"] = np.array([y_energy], dtype=np.float64)
            data["force"] = np.asarray(y_force, dtype=np.float64)
            if y_stress is not None:
                data["stress"] = np.asarray(y_stress, dtype=np.float64)
        return data

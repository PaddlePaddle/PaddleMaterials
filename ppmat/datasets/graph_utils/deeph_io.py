# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Readers for DeepH processed structures and matrix blocks."""

import json
import os
import os.path as osp

import numpy as np


def load_deeph_structure(folder: str | os.PathLike[str]):
    """Load a DeepH processed structure without changing its lattice or coordinates."""
    from ase import Atoms
    from pymatgen.io.ase import AseAtomsAdaptor

    folder = os.fspath(folder)
    lattice = np.loadtxt(osp.join(folder, "lat.dat")).T
    atomic_numbers = np.loadtxt(osp.join(folder, "element.dat")).astype(np.int64)
    cart_coords = np.loadtxt(osp.join(folder, "site_positions.dat")).T
    atoms = Atoms(
        numbers=atomic_numbers.reshape(-1),
        positions=cart_coords,
        cell=lattice,
        pbc=True,
    )
    return AseAtomsAdaptor.get_structure(atoms)


def load_deeph_orbital_counts(path: str | os.PathLike[str]):
    """Return the number of atomic orbitals described on each input line."""
    with open(path) as file:
        orbital_types = [list(map(int, line.split())) for line in file if line.strip()]
    return [
        sum(2 * orbital + 1 for orbital in atom_orbitals)
        for atom_orbitals in orbital_types
    ]


def _load_deeph_npz_matrices(path, dtype):
    matrices = {}
    with np.load(path) as matrix_file:
        for key_string, value in matrix_file.items():
            translation_x, translation_y, translation_z, atom_i, atom_j = json.loads(
                key_string
            )
            matrices[
                (translation_x, translation_y, translation_z, atom_i - 1, atom_j - 1)
            ] = np.asarray(value, dtype=dtype)
    return matrices


def load_deeph_matrices(folder, dtype, include_hamiltonian=True):
    """Load orbital counts, local rotations, and optional Hamiltonian blocks."""
    orbital_counts = load_deeph_orbital_counts(osp.join(folder, "orbital_types.dat"))
    rotations = _load_deeph_npz_matrices(osp.join(folder, "rc.npz"), dtype)
    hamiltonians = None
    if include_hamiltonian:
        hamiltonians = _load_deeph_npz_matrices(osp.join(folder, "rh.npz"), dtype)
    return orbital_counts, rotations, hamiltonians

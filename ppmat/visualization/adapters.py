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

from __future__ import annotations

from typing import Any

from ase import Atoms
from cvve import Structure as CVVEStructure
from pymatgen.core import Molecule
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from rdkit import Chem

from ppmat.utils.ext_rdkit import mol_from_graphs


def molecular_graph_to_rdkit(
    node_types,
    edge_types,
    *,
    atom_decoder,
    bond_decoder,
) -> Chem.Mol | None:
    """Convert a discrete molecular graph to an RDKit molecule."""

    return mol_from_graphs(
        atom_decoder,
        node_types,
        edge_types,
        bond_decoder=bond_decoder,
    )


def to_ase_atoms(structure: Any) -> Atoms:
    """Normalize supported atomic-structure objects to ASE ``Atoms``."""

    if isinstance(structure, Atoms):
        return structure
    if isinstance(structure, CVVEStructure):
        return structure.to_atoms()
    if isinstance(structure, (Structure, Molecule)):
        return AseAtomsAdaptor.get_atoms(structure)
    raise TypeError(
        "structure must be ase.Atoms, cvve.Structure, pymatgen.Structure, "
        f"or pymatgen.Molecule, but got {type(structure).__name__}."
    )

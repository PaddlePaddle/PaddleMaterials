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

from __future__ import annotations

from typing import Dict

from rdkit import Chem

from ppmat.datasets.build_molecule import BuildMolecule
from ppmat.utils.ext_rdkit import bond_types_from_vocab


def build_rdkit_mol(molecule: Dict) -> Chem.Mol:
    """Build an RDKit molecule from a sampled molecule dictionary.

    The dictionary carries ``atom_types`` (symbols, including the ``Se`` mask
    placeholder), ``positions``, ``charges`` and ``bonds`` (source, destination,
    bond type index).
    """
    atomic_numbers = [
        Chem.GetPeriodicTable().GetAtomicNumber(symbol)
        for symbol in molecule["atom_types"]
    ]
    mol = Chem.RWMol(
        BuildMolecule.build_one(
            {"atomic_numbers": atomic_numbers, "positions": molecule["positions"]},
            "dict",
            sanitize=False,
            add_hs=False,
            remove_hs=False,
            kekulize=False,
        )
    )
    for atom, charge in zip(mol.GetAtoms(), molecule["charges"]):
        atom.SetFormalCharge(int(charge))

    bond_types = bond_types_from_vocab()
    for src, dst, bond_type in molecule["bonds"]:
        bond_type = int(bond_type)
        if bond_type not in bond_types:
            raise ValueError(f"unsupported bond type index: {bond_type}")
        mol.AddBond(int(src), int(dst), bond_types[bond_type])
    return mol.GetMol()

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

"""Convert RDKit molecules into the raw arrays consumed by FlowMol."""

from __future__ import annotations

from typing import List
from typing import Optional
from typing import Sequence
from typing import Union

import numpy as np
from p_tqdm import p_map
from rdkit import Chem
from rdkit import rdBase
from rdkit.Chem import rdDetermineBonds

# Class zero denotes absent bonds in the complete graph.
_BOND_TYPE_TO_CLASS = {
    Chem.BondType.SINGLE: 1,
    Chem.BondType.DOUBLE: 2,
    Chem.BondType.TRIPLE: 3,
}
_DEFAULT_COVALENT_FACTORS = (1.1, 1.2, 1.3)


class FlowMolMolConverter:
    """Convert RDKit Mol into the raw per-molecule arrays used by FlowMol.

    The output is a picklable dict of numpy arrays (not a ``pgl.Graph``) so that
    ``QM9Dataset`` can cache it and a downstream transform can assemble the
    ``structure_array`` batch fields, including the dense upper-triangle bonds.

    Args:
        num_cpus (Optional[int]): Number of CPUs for parallel conversion.
            Defaults to 1 (no parallelism).
        covalent_factors (Sequence[float]): Distance factors attempted in order
            when assigning bonds to geometry-only molecules.
            Defaults to (1.1, 1.2, 1.3).
    """

    def __init__(
        self,
        num_cpus: Optional[int] = None,
        covalent_factors: Sequence[float] = _DEFAULT_COVALENT_FACTORS,
    ) -> None:
        self.num_cpus = 1 if num_cpus is None else int(num_cpus)
        if not covalent_factors:
            raise ValueError("covalent_factors must not be empty.")
        self.covalent_factors = tuple(covalent_factors)

    @staticmethod
    def build_one(
        mol: Chem.Mol, covalent_factors: Sequence[float] = _DEFAULT_COVALENT_FACTORS
    ) -> dict:
        if mol is None:
            raise ValueError("FlowMol requires a valid RDKit molecule.")
        n_atoms = mol.GetNumAtoms()
        if n_atoms == 0:
            raise ValueError("FlowMol requires at least one atom.")
        if mol.GetNumConformers() == 0:
            raise ValueError("FlowMol requires a molecule with 3D coordinates.")

        if mol.GetNumBonds() == 0 and n_atoms > 1:
            for index, factor in enumerate(covalent_factors):
                bonded = Chem.Mol(mol)
                try:
                    with rdBase.BlockLogs():
                        rdDetermineBonds.DetermineBonds(
                            bonded, charge=0, covFactor=factor, useVdw=True
                        )
                except (ValueError, RuntimeError):
                    if index == len(covalent_factors) - 1:
                        raise
                else:
                    mol = bonded
                    break
        else:
            mol = Chem.Mol(mol)
        Chem.Kekulize(mol, clearAromaticFlags=True)

        positions = mol.GetConformer().GetPositions().astype("float32")
        atomic_numbers = np.asarray(
            [atom.GetAtomicNum() for atom in mol.GetAtoms()], dtype="int64"
        )
        formal_charges = np.asarray(
            [atom.GetFormalCharge() for atom in mol.GetAtoms()], dtype="int64"
        )

        bond_index = []
        bond_class = []
        for bond in mol.GetBonds():
            cls = _BOND_TYPE_TO_CLASS[bond.GetBondType()]
            bond_index.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
            bond_class.append(cls)
        if bond_index:
            bond_index = np.asarray(bond_index, dtype="int64")
            bond_class = np.asarray(bond_class, dtype="int64")
        else:
            bond_index = np.empty((0, 2), dtype="int64")
            bond_class = np.empty((0,), dtype="int64")

        return {
            "positions": positions,
            "atomic_numbers": atomic_numbers,
            "formal_charges": formal_charges,
            "bond_index": bond_index,
            "bond_class": bond_class,
        }

    def __call__(
        self, mols: Union[Sequence[Chem.Mol], Chem.Mol]
    ) -> Union[List[dict], dict]:
        if isinstance(mols, (list, tuple)):
            return p_map(
                self.build_one,
                mols,
                [self.covalent_factors] * len(mols),
                num_cpus=self.num_cpus,
                desc="Converting molecules",
                dynamic_ncols=True,
                mininterval=0.2,
            )
        return self.build_one(mols, self.covalent_factors)

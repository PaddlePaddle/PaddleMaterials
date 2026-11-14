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

from typing import Any
from typing import Dict
from typing import List

from ppmat.metrics.streaming_base import StreamingMetricBase

VALENCE_ELECTRONS = {
    "H": 1,
    "He": 2,
    "Li": 1,
    "Be": 2,
    "B": 3,
    "C": 4,
    "N": 5,
    "O": 6,
    "F": 7,
    "Ne": 8,
    "Na": 1,
    "Mg": 2,
    "Al": 3,
    "Si": 4,
    "P": 5,
    "S": 6,
    "Cl": 7,
    "Ar": 8,
    "K": 1,
    "Ca": 2,
    "Ga": 3,
    "Ge": 4,
    "As": 5,
    "Se": 6,
    "Br": 7,
    "Kr": 8,
    "Rb": 1,
    "Sr": 2,
    "In": 3,
    "Sn": 4,
    "Sb": 5,
    "Te": 6,
    "I": 7,
    "Xe": 8,
    "Cs": 1,
    "Ba": 2,
    "Tl": 3,
    "Pb": 4,
    "Bi": 5,
    "Po": 6,
    "At": 7,
    "Rn": 8,
    "Sc": 3,
    "Ti": 4,
    "V": 5,
    "Cr": 6,
    "Mn": 7,
    "Fe": 8,
    "Co": 9,
    "Ni": 10,
    "Cu": 11,
    "Zn": 12,
}

EXPECTED_VALENCES = {
    "H": 1,
    "C": 4,
    "N": 3,
    "O": 2,
    "F": 1,
    "P": 3,
    "S": 2,
    "Cl": 1,
    "Br": 1,
    "I": 1,
    "Al": 3,
    "Si": 4,
    "As": 3,
    "Se": 2,
    "B": 3,
}

ALTERNATIVE_VALENCES = {"S": 6, "Si": 2, "As": 5, "Se": 6}


class PropMolFlowMoleculeMetric(StreamingMetricBase):
    """Validity and valence-stability metrics of generated molecules.

    Computes the standard PropMolFlow metrics over sampled molecule
    dictionaries: ``rdkit_valid``, ``valid_and_unique``, ``close_shell_ratio``,
    ``atom_stable`` and ``mol_stable``. The streaming ``update_step`` consumes
    the ``MolecularSampler`` result contract (``result["samples"]["pred"]``);
    the one-shot ``__call__`` shares the same ingest implementation.
    ``close_shell_ratio`` is the even-electron-count proxy, corrected for
    formal charge; it does not establish the absence of unpaired electrons.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self._n_molecules = 0
        self._n_valid = 0
        self._n_close_shell = 0
        self._n_stable_atoms = 0
        self._n_atoms = 0
        self._n_stable_molecules = 0
        self._unique_smiles = set()

    def _ingest(self, molecules: List[Dict]) -> None:
        from rdkit import Chem
        from rdkit import rdBase

        from ppmat.models.propmolflow.molecule_builder import build_rdkit_mol

        with rdBase.BlockLogs():
            for molecule in molecules:
                self._n_molecules += 1
                self._n_atoms += len(molecule["atom_types"])
                try:
                    rdkit_mol = build_rdkit_mol(molecule)
                    rdkit_mol.UpdatePropertyCache(strict=False)
                except (IndexError, RuntimeError, ValueError):
                    continue

                atoms = rdkit_mol.GetAtoms()
                total_charge = sum(atom.GetFormalCharge() for atom in atoms)
                valence_electrons = sum(
                    VALENCE_ELECTRONS.get(atom.GetSymbol(), 0) for atom in atoms
                )
                electron_count = valence_electrons - total_charge
                if electron_count % 2 == 0:
                    self._n_close_shell += 1

                molecule_stable = total_charge == 0
                for atom in atoms:
                    symbol = atom.GetSymbol()
                    effective_valence = (
                        atom.GetExplicitValence() - atom.GetFormalCharge()
                    )
                    valid = False
                    if symbol in EXPECTED_VALENCES:
                        valid = effective_valence == EXPECTED_VALENCES[symbol]
                        if not valid and symbol in ALTERNATIVE_VALENCES:
                            valid = effective_valence == ALTERNATIVE_VALENCES[symbol]
                        if not valid and symbol == "Se":
                            valid = effective_valence == 4
                    self._n_stable_atoms += int(valid)
                    molecule_stable = molecule_stable and valid
                self._n_stable_molecules += int(molecule_stable)

                try:
                    Chem.SanitizeMol(rdkit_mol)
                except (RuntimeError, ValueError):
                    continue
                self._n_valid += 1
                self._unique_smiles.add(Chem.MolToSmiles(rdkit_mol))

    def update_step(self, *, result: Dict[str, Any], batch: Any, stage: str):
        if stage != "sample":
            return
        samples = result.get("samples") or {}
        molecules = samples.get("pred")
        if molecules is None:
            raise KeyError(
                "PropMolFlowMoleculeMetric expects samples['pred'] in the "
                "MolecularSampler result"
            )
        self._ingest(molecules)

    def compute_epoch(self, *, stage: str) -> Dict[str, float]:
        if stage != "sample":
            return {}
        if self._n_molecules == 0:
            raise ValueError("No molecules were accumulated")
        return {
            "rdkit_valid": self._n_valid / self._n_molecules,
            "valid_and_unique": len(self._unique_smiles) / self._n_molecules,
            "close_shell_ratio": self._n_close_shell / self._n_molecules,
            "atom_stable": (
                self._n_stable_atoms / self._n_atoms if self._n_atoms else 0.0
            ),
            "mol_stable": self._n_stable_molecules / self._n_molecules,
        }

    def __call__(self, molecules: List[Dict]) -> Dict[str, float]:
        self.reset()
        self._ingest(molecules)
        return self.compute_epoch(stage="sample")

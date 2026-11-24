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

ELECTRON_COUNT_ELEMENTS = frozenset(
    "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Ga Ge As Se Br Kr Rb "
    "Sr In Sn Sb Te I Xe Cs Ba Tl Pb Bi Po At Rn Sc Ti V Cr Mn Fe Co Ni Cu Zn".split()
)
STABILITY_ELEMENTS = frozenset("H C N O F P S Cl Br I Al Si As Se B".split())

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

        periodic_table = Chem.GetPeriodicTable()
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
                    periodic_table.GetNOuterElecs(atom.GetAtomicNum())
                    for atom in atoms
                    if atom.GetSymbol() in ELECTRON_COUNT_ELEMENTS
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
                    if symbol in STABILITY_ELEMENTS:
                        valid = effective_valence == periodic_table.GetDefaultValence(
                            atom.GetAtomicNum()
                        )
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

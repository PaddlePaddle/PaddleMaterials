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

"""Perceive bonds and extract raw arrays for PropMolFlow datasets."""

from __future__ import annotations

import os.path as osp
import subprocess
import sys
import tempfile
from datetime import datetime
from functools import partial

import numpy as np
from rdkit import Chem
from rdkit import rdBase
from rdkit.Chem import rdDetermineBonds

from ppmat.utils.ext_rdkit import bond_types_from_vocab

N_CHARGES = 6
CHARGE_OFFSET = 2
N_BOND_TYPES = 5
BOND_TYPE_INDEX = {bond: index for index, bond in bond_types_from_vocab().items()}
_PERCEPTION_OPTIONS = (
    {"charge": 0},
    {"charge": 0, "allowChargedFragments": False},
)
_HUECKEL_SCRIPT = (
    "import sys\n"
    "from pathlib import Path\n"
    "from rdkit import Chem\n"
    "from rdkit.Chem import rdDetermineBonds\n"
    "mol = Chem.Mol(sys.stdin.buffer.read())\n"
    "rdDetermineBonds.DetermineBonds(mol, charge=0, useHueckel=True)\n"
    "Path(sys.argv[1]).write_bytes("
    "mol.ToBinary(Chem.PropertyPickleOptions.AllProps))\n"
)


def _perceive_molecule(molecule, work_dir):
    if molecule is None:
        return None
    for kwargs in _PERCEPTION_OPTIONS:
        mol = Chem.Mol(molecule)
        try:
            with rdBase.BlockLogs():
                rdDetermineBonds.DetermineBonds(mol, **kwargs)
            return mol
        except (RuntimeError, ValueError):
            continue
    # Extended Hueckel writes fixed filenames; isolate its working directory.
    stamp = datetime.now().strftime("%Y%m%d%H%M%S")
    with tempfile.TemporaryDirectory(
        prefix=f"perception_{stamp}_", dir=work_dir
    ) as tmp:
        filename = f"molecule_{stamp}.bin"
        result = subprocess.run(
            [sys.executable, "-c", _HUECKEL_SCRIPT, filename],
            input=molecule.ToBinary(),
            cwd=tmp,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            check=False,
        )
        if result.returncode != 0:
            return None
        with open(osp.join(tmp, filename), "rb") as stream:
            return Chem.Mol(stream.read())


def _featurize_molecule(mol, atom_map: list):
    if mol is None:
        return None
    atom_map_dict = {symbol: index for index, symbol in enumerate(atom_map)}
    atom_types, charges = [], []
    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol()
        if symbol not in atom_map_dict:
            return None
        atom_types.append(atom_map_dict[symbol])
        charge = atom.GetFormalCharge()
        if not -CHARGE_OFFSET <= charge < N_CHARGES - CHARGE_OFFSET:
            return None
        charges.append(charge)
    num_atoms = mol.GetNumAtoms()
    if num_atoms == 0:
        return None
    adjacency = np.zeros((num_atoms, num_atoms), dtype="int64")
    for bond in mol.GetBonds():
        bond_type = bond.GetBondType()
        if bond_type not in BOND_TYPE_INDEX:
            return None
        begin, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        adjacency[begin, end] = BOND_TYPE_INDEX[bond_type]
        adjacency[end, begin] = BOND_TYPE_INDEX[bond_type]
    upper_i, upper_j = np.triu_indices(num_atoms, k=1)
    return {
        "positions": np.asarray(mol.GetConformer().GetPositions(), dtype="float32"),
        "atom_types": np.asarray(atom_types, dtype="int64"),
        "charges": np.asarray(charges, dtype="int64"),
        "bond_types": adjacency[upper_i, upper_j],
        "num_atoms": num_atoms,
    }


def _featurize_one(molecule, atom_map: list, work_dir: str) -> dict:
    """Module-level worker so the featurization stays process-picklable."""
    return _featurize_molecule(_perceive_molecule(molecule, work_dir), atom_map)


class PropMolFlowMolConverter:
    """Convert RDKit molecules into raw complete-graph features.

    Failed perception or unsupported atom/charge/bond classes return None so
    the dataset can filter molecules together with their property rows.
    ``work_dir`` isolates temporary files created by extended Hueckel.
    """

    def __init__(self, atom_map: list, work_dir: str, num_cpus: int = 1):
        self.atom_map = list(atom_map)
        self.work_dir = work_dir
        self.num_cpus = num_cpus

    def __call__(self, molecules):
        worker = partial(_featurize_one, atom_map=self.atom_map, work_dir=self.work_dir)
        with rdBase.BlockLogs():
            if isinstance(molecules, (list, tuple)):
                if self.num_cpus and self.num_cpus > 1:
                    from p_tqdm import p_map

                    return p_map(worker, molecules, num_cpus=self.num_cpus)
                return [worker(molecule) for molecule in molecules]
            return worker(molecules)

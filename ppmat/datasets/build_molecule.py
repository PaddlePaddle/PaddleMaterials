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
from typing import List
from typing import Literal
from typing import Optional
from typing import Sequence
from typing import Union

from p_tqdm import p_map
from rdkit import Chem
from rdkit.Geometry import Point3D


class BuildMolecule:
    """Build RDKit Mol from different formats.

    Args:
        format (Literal["smiles","mol_block","mol_file","sdf_file","xyz_block",
            "xyz_file","inchi","dict","rdmol"]): format of input molecules data
            used by convertion of RDKit. The ``dict`` format accepts
            ``atomic_numbers`` and ``positions`` arrays.
        sanitize (bool): Whether to sanitize the molecule using RDKit after construction
            (e.g., validate valence, adjust bond orders). Defaults to True.
        add_hs (bool): Whether to add explicit hydrogen atoms to the molecule.
            Defaults to False.
        remove_hs (bool): Whether to remove existing explicit hydrogen atoms from the
            molecule. Defaults to False.
        kekulize (bool): Whether to attempt Kekulization of the molecule (convert
            aromatic bonds to explicit single/double bonds). Defaults to False.
        num_cpus (Optional[int]): Number of CPUs for parallel processing during
            molecule construction. Defaults to 1 (no parallelism).
            Set to None for automatic CPU detection.
    """

    def __init__(
        self,
        format: Literal[
            "smiles",
            "mol_block",
            "mol_file",
            "sdf_file",
            "xyz_block",
            "xyz_file",
            "inchi",
            "dict",
            "rdmol",
        ],
        sanitize: bool = True,
        add_hs: bool = False,
        remove_hs: bool = False,
        kekulize: bool = False,
        num_cpus: Optional[int] = None,
    ) -> None:
        self.format = format
        self.sanitize = sanitize
        self.add_hs = add_hs
        self.remove_hs = remove_hs
        self.kekulize = kekulize
        self.num_cpus = 1 if num_cpus is None else int(num_cpus)

    @staticmethod
    def _post_process(
        mol: Chem.Mol, sanitize: bool, add_hs: bool, remove_hs: bool, kekulize: bool
    ) -> Chem.Mol:
        if mol is None:
            return None
        if sanitize:
            Chem.SanitizeMol(mol)
        if add_hs:
            mol = Chem.AddHs(mol)
        if kekulize:
            try:
                Chem.Kekulize(mol, clearAromaticFlags=True)
            except Exception:
                pass
        if remove_hs:
            mol = Chem.RemoveHs(mol)
        return mol

    @staticmethod
    def build_one(
        mol_data: Any,
        format: str,
        sanitize: bool,
        add_hs: bool,
        remove_hs: bool,
        kekulize: bool,
    ) -> Optional[Chem.Mol]:
        if format == "smiles":
            mol = Chem.MolFromSmiles(str(mol_data), sanitize=sanitize)
        elif format == "mol_block":
            mol = Chem.MolFromMolBlock(
                str(mol_data),
                sanitize=sanitize,
                removeHs=False,
            )
        elif format == "mol_file":
            with open(str(mol_data), "r") as f:
                mol_block = f.read()
            mol = Chem.MolFromMolBlock(
                mol_block,
                sanitize=sanitize,
                removeHs=False,
            )
        elif format == "sdf_file":
            suppl = Chem.SDMolSupplier(str(mol_data), sanitize=sanitize, removeHs=False)
            mol = next((m for m in suppl if m is not None), None)
        elif format == "xyz_block":
            mol = Chem.MolFromXYZBlock(str(mol_data))
        elif format == "xyz_file":
            mol = Chem.MolFromXYZFile(str(mol_data))
        elif format == "inchi":
            mol = Chem.MolFromInchi(str(mol_data))
        elif format == "dict":
            atomic_numbers = mol_data["atomic_numbers"]
            positions = mol_data["positions"]
            mol = Chem.RWMol()
            for atomic_number in atomic_numbers:
                mol.AddAtom(Chem.Atom(int(atomic_number)))
            mol = mol.GetMol()
            conformer = Chem.Conformer(len(atomic_numbers))
            for atom_idx, position in enumerate(positions):
                conformer.SetAtomPosition(
                    atom_idx, Point3D(*(float(value) for value in position))
                )
            mol.AddConformer(conformer)
        elif format == "rdmol":
            mol = mol_data
        else:
            raise ValueError(f"Invalid format specified: {format}")

        return BuildMolecule._post_process(mol, sanitize, add_hs, remove_hs, kekulize)

    def __call__(
        self, molecules_data: Union[Sequence[Any], Any]
    ) -> Union[List[Chem.Mol], Chem.Mol, None]:
        if isinstance(molecules_data, (list, tuple)):
            return p_map(
                BuildMolecule.build_one,
                molecules_data,
                [self.format] * len(molecules_data),
                [self.sanitize] * len(molecules_data),
                [self.add_hs] * len(molecules_data),
                [self.remove_hs] * len(molecules_data),
                [self.kekulize] * len(molecules_data),
                num_cpus=self.num_cpus,
                desc="Building molecules",
                dynamic_ncols=True,
                mininterval=0.2,
            )
        else:
            return BuildMolecule.build_one(
                molecules_data,
                self.format,
                self.sanitize,
                self.add_hs,
                self.remove_hs,
                self.kekulize,
            )

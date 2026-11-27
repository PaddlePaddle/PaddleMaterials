# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

from pathlib import Path
from tempfile import NamedTemporaryFile

from rdkit import Chem

GEODIFF_SOURCE_SMILES = "GEODIFF_SOURCE_SMILES"


def molecular_identity(molecule):
    """Return canonical isomeric SMILES without hydrogens, save isotopic ones.

    A molecule RDKit cannot sanitise without them is named with them.
    """
    try:
        molecule = Chem.RemoveHs(molecule)
    except (RuntimeError, ValueError):
        pass
    return Chem.MolToSmiles(molecule, isomericSmiles=True)


def _molecule_from_record(record, bond_types):
    if record["rdmol"] is not None:
        molecule = Chem.Mol(record["rdmol"])
        molecule.RemoveAllConformers()
        return molecule

    molecule = Chem.RWMol()
    for atomic_num in record["atom_types"]:
        molecule.AddAtom(Chem.Atom(int(atomic_num)))
    for (left, right), bond_type in zip(
        record["edge_index"].T.tolist(), record["edge_type"].tolist()
    ):
        if molecule.GetBondBetweenAtoms(left, right) is None:
            molecule.AddBond(left, right, bond_types[bond_type])
    Chem.SanitizeMol(molecule)
    return molecule


def write_molecular_conformers(path, records):
    """Write an SDF from records with RDKit topology or explicit GEOM bonds."""
    path = Path(path)
    bond_types = tuple(Chem.BondType.names.values())
    with NamedTemporaryFile(dir=path.parent, suffix=".sdf", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        with Chem.SDWriter(str(temporary)) as writer:
            writer.SetKekulize(False)
            for record in records:
                molecule = _molecule_from_record(record, bond_types)
                # Preserve the conditioning graph even if the sampled pose inverts it.
                molecule.SetProp(GEODIFF_SOURCE_SMILES, molecular_identity(molecule))
                molecule.AddConformer(Chem.Conformer(molecule.GetNumAtoms()))
                conformer = molecule.GetConformer()
                for coordinates in record["cart_coords"]:
                    for index, point in enumerate(coordinates.tolist()):
                        conformer.SetAtomPosition(index, point)
                    writer.write(molecule)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)

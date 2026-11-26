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

"""Pair GEOM reference coordinates with source-tagged generated conformers."""

import numpy as np
from rdkit import Chem

from ppmat.utils.molecular_conformers import GEODIFF_SOURCE_SMILES
from ppmat.utils.molecular_conformers import molecular_identity


def _ordered_topology(mol):
    # A canonical SMILES alone cannot detect a permutation of coordinate rows.
    atoms = tuple(
        (
            a.GetAtomicNum(),
            a.GetIsotope(),
            a.GetFormalCharge(),
            a.GetNumRadicalElectrons(),
            a.GetTotalNumHs(),
            a.GetIsAromatic(),
            a.GetAtomMapNum(),
        )
        for a in mol.GetAtoms()
    )
    bonds = {}
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bonds[min(i, j), max(i, j)] = int(bond.GetBondType())
    return atoms, bonds


def _check_offsets(array, name, rows, total):
    if (
        array.shape != (rows + 1,)
        or array.dtype.kind not in "iu"
        or array[0] != 0
        or array[-1] != total
        or np.any(array[1:] < array[:-1])
    ):
        raise ValueError(f"Invalid GEOM reference {name} offsets.")


_CIS_TRANS = {
    Chem.BondStereo.STEREOCIS: Chem.BondStereo.STEREOZ,
    Chem.BondStereo.STEREOTRANS: Chem.BondStereo.STEREOE,
}


def _specified_stereo(mol):
    """Return the R/S centres and E/Z bonds a molecule specifies, by atom index.

    Tags on atoms that are not stereocentres describe nothing a 3D pose can
    reproduce, so they are dropped before the centres are labelled.
    """
    mol = Chem.Mol(mol)
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    centres = {
        index: label
        for index, label in Chem.FindMolChiralCenters(
            mol, includeUnassigned=True, useLegacyImplementation=False
        )
        if label != "?"
    }
    bonds = set()
    for bond in mol.GetBonds():
        stereo = _CIS_TRANS.get(bond.GetStereo(), bond.GetStereo())
        if stereo in (Chem.BondStereo.STEREOE, Chem.BondStereo.STEREOZ):
            ends = (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
            bonds.add((frozenset(zip(ends, bond.GetStereoAtoms())), stereo))
    return centres, bonds


def _stereo_changes(specified, mol):
    """Tell whether a 3D pose changes stereochemistry its source specifies."""
    centres, bonds = specified
    if not centres and not bonds:
        return False
    perceived = _specified_stereo(mol)
    return any(
        perceived[0].get(index) != label for index, label in centres.items()
    ) or any(bond not in perceived[1] for bond in bonds)


def _reference_groups(reference):
    fields = ("smiles", "pos", "node_ptr", "rdmol_ptr", "rdmol_data")
    with np.load(reference, allow_pickle=False) as archive:
        missing = set(fields).difference(archive.files)
        if missing:
            raise ValueError(f"GEOM reference is missing fields: {sorted(missing)}.")
        smiles, xyz, nodes, offsets, binary = (archive[key] for key in fields)
    if smiles.ndim != 1 or smiles.dtype.kind != "U" or not len(smiles):
        raise ValueError("GEOM reference requires an author SMILES for each row.")
    if (
        xyz.ndim != 2
        or xyz.shape[1] != 3
        or xyz.dtype.kind not in "fiu"
        or not np.isfinite(xyz).all()
    ):
        raise ValueError(
            "GEOM reference positions must contain finite xyz coordinates."
        )
    if binary.ndim != 1 or binary.dtype != np.uint8:
        raise ValueError("Invalid GEOM reference RDKit binary data.")
    _check_offsets(nodes, "node_ptr", len(smiles), len(xyz))
    _check_offsets(offsets, "rdmol_ptr", len(smiles), len(binary))

    # As in the author's packed test set, a molecule is one entry: the graph of
    # its first row and the coordinates of all its rows.
    groups, sources, topologies, stereo = {}, {}, {}, {}
    for i, key in enumerate(smiles.tolist()):
        label = f"GEOM reference row {i + 1} ({key!r})"
        if not key.strip():
            raise ValueError(f"{label}: missing author SMILES.")
        if key in groups:
            groups[key][2].append(xyz[nodes[i] : nodes[i + 1]])
            continue
        mol = Chem.Mol(binary[offsets[i] : offsets[i + 1]].tobytes())
        if mol.GetNumAtoms() == 0:
            raise ValueError(f"{label}: empty molecule.")
        source = molecular_identity(mol)
        sources[source] = None if source in sources else key
        groups[key] = (key, mol, [xyz[nodes[i] : nodes[i + 1]]], [])
        topologies[key] = _ordered_topology(mol)
        stereo[source] = _specified_stereo(mol)
    return groups, sources, topologies, stereo


def read_covmat_inputs(reference, generated):
    """Return per-molecule conformer groups and stereochemistry diagnostics.

    Each group contains (author SMILES, source Mol, reference xyzs, generated
    xyzs), in first-seen group order and input order within each sequence. The
    source Mol is the first reference row of its molecule. Pairing uses only the
    source tag.
    """
    groups, sources, topologies, stereo = _reference_groups(reference)
    mismatches = dict.fromkeys(sources, 0)
    matched = 0
    with open(generated, "rb") as handle:
        for i, mol in enumerate(Chem.ForwardSDMolSupplier(handle, removeHs=False), 1):
            label = f"Generated SDF record {i}"
            if mol is None or mol.GetNumAtoms() == 0 or mol.GetNumConformers() != 1:
                raise ValueError(f"{label}: invalid molecule or coordinates.")
            source = (
                mol.GetProp(GEODIFF_SOURCE_SMILES)
                if mol.HasProp(GEODIFF_SOURCE_SMILES)
                else ""
            )
            if not source.strip():
                raise ValueError(
                    f"{label}: missing {GEODIFF_SOURCE_SMILES}; "
                    "re-sample to record source identities."
                )
            if source not in sources:
                raise ValueError(f"{label}: unknown source {source!r}.")
            key = sources[source]
            if key is None:
                raise ValueError(
                    f"{label}: ambiguous source {source!r}, the first row of "
                    "two author groups."
                )
            if _ordered_topology(mol) != topologies[key]:
                raise ValueError(
                    f"{label}: ordered topology differs from source {source!r}."
                )
            xyz = mol.GetConformer().GetPositions()
            if not np.isfinite(xyz).all():
                raise ValueError(f"{label}: non-finite coordinates.")
            groups[key][3].append(xyz)
            matched += 1
            if _stereo_changes(stereo[source], mol):
                mismatches[source] += 1
    return (
        groups.values(),
        {
            "matched_records": matched,
            "mismatch_count": sum(mismatches.values()),
            "mismatches_by_source": mismatches,
        },
    )

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

# Adapted from GeoDiff (https://github.com/MinkaiXu/GeoDiff). See
# ppmat/models/geodiff/LICENSE.

from numbers import Integral

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolAlign


def _coordinates(values, num_atoms, label):
    xyz = np.asarray(values, dtype=np.float64)
    if xyz.shape == (0,):
        xyz = xyz.reshape(0, num_atoms, 3)
    if xyz.ndim != 3 or xyz.shape[1:] != (num_atoms, 3):
        raise ValueError(f"{label}: expected conformers of shape ({num_atoms}, 3)")
    if not np.isfinite(xyz).all():
        raise ValueError(f"{label}: nonfinite coordinates")
    return xyz


def _heavy_conformer(template, xyz):
    mol = Chem.Mol(template)
    conf = Chem.Conformer(mol.GetNumAtoms())
    for index, position in enumerate(xyz):
        conf.SetAtomPosition(index, position)
    mol.AddConformer(conf)
    return Chem.RemoveHs(mol)


def _nearest_distances(mol, references, generated):
    template = Chem.Mol(mol)
    template.RemoveAllConformers()
    refs = [_heavy_conformer(template, xyz) for xyz in references]
    if not refs[0].GetNumHeavyAtoms():
        raise ValueError("RMSD requires at least one heavy atom")
    ref_min = np.full(len(refs), np.inf)
    gen_min = np.full(len(generated), np.inf)
    for i, xyz in enumerate(generated):
        probe = _heavy_conformer(template, xyz)
        for j, ref in enumerate(refs):
            # GetBestRMS aligns its probe in place. Start each pair unaligned.
            rmsd = rdMolAlign.GetBestRMS(Chem.Mol(probe), ref)
            if not np.isfinite(rmsd) or rmsd < 0:
                raise ValueError(f"Invalid RMSD for generated {i}, reference {j}")
            ref_min[j] = min(ref_min[j], rmsd)
            gen_min[i] = min(gen_min[i], rmsd)
    return ref_min, gen_min


def compute_covmat(groups, *, ratio=2, thresholds=None):
    """Evaluate GeoDiff coverage (fractions) and matching (angstroms).

    Each group contains the author SMILES, an RDKit molecule and two coordinate
    collections (reference, generated), both in that molecule's atom order.
    The first ``ratio * num_reference`` generated conformers enter the score.
    Summary statistics give each eligible molecule equal weight.
    """
    if isinstance(ratio, bool) or not isinstance(ratio, Integral) or ratio < 1:
        raise ValueError("ratio must be a positive integer")
    ratio = int(ratio)
    if thresholds is None:
        thresholds = np.arange(0.05, 3.05, 0.05)
    thresholds = np.asarray(thresholds, dtype=np.float64)
    if (
        thresholds.ndim != 1
        or not thresholds.size
        or not np.isfinite(thresholds).all()
        or (thresholds < 0).any()
    ):
        raise ValueError(
            "thresholds must be a nonempty vector of finite nonnegative distances"
        )

    counts = dict(
        input_molecules=0,
        eligible_molecules=0,
        disconnected=0,
        missing_generated=0,
        insufficient_generated=0,
    )
    records = []
    for smiles, mol, references, generated in groups:
        counts["input_molecules"] += 1
        if not isinstance(smiles, str) or not smiles or not isinstance(mol, Chem.Mol):
            raise ValueError(
                "Each group requires an author SMILES and an RDKit molecule"
            )
        if "." in smiles:
            counts["disconnected"] += 1
            continue
        if not len(references):
            raise ValueError(f"{smiles}: no reference conformers")
        budget = ratio * len(references)
        num_generated = len(generated)
        if not num_generated:
            counts["missing_generated"] += 1
            continue
        if num_generated < budget:
            counts["insufficient_generated"] += 1
            continue
        references = _coordinates(references, mol.GetNumAtoms(), f"{smiles} reference")
        used = _coordinates(
            generated[:budget], mol.GetNumAtoms(), f"{smiles} generated"
        )
        ref_min, gen_min = _nearest_distances(mol, references, used)
        records.append(
            {
                "smiles": smiles,
                "num_reference": len(references),
                "num_generated": num_generated,
                "num_generated_used": budget,
                "CoverageR": (ref_min[:, None] <= thresholds).mean(axis=0).tolist(),
                "CoverageP": (gen_min[:, None] <= thresholds).mean(axis=0).tolist(),
                "MatchingR": float(ref_min.mean()),
                "MatchingP": float(gen_min.mean()),
            }
        )
    if not records:
        raise ValueError(
            f"No eligible molecules for COV/MAT (ratio={ratio}, "
            f"disconnected={counts['disconnected']}, "
            f"missing_generated={counts['missing_generated']}, "
            f"insufficient_generated={counts['insufficient_generated']})."
        )
    counts["eligible_molecules"] = len(records)
    summary = {}
    for name in ("CoverageR", "CoverageP", "MatchingR", "MatchingP"):
        values = np.asarray([record[name] for record in records])
        summary[name] = {
            "mean": values.mean(axis=0).tolist(),
            "median": np.median(values, axis=0).tolist(),
            "std": values.std(axis=0).tolist(),
        }
    return {
        "counts": counts,
        "units": {
            "coverage": "fraction",
            "matching": "angstrom",
            "thresholds": "angstrom",
        },
        "ratio": ratio,
        "thresholds": thresholds.tolist(),
        "per_molecule": records,
        "summary": summary,
    }

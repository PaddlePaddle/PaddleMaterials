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

import itertools
import warnings
from collections import Counter

import numpy as np
import paddle
import smact
from matminer.featurizers.composition.composite import ElementProperty
from matminer.featurizers.site.fingerprint import CrystalNNFingerprint
from pymatgen.core import Element
from pymatgen.core.composition import Composition
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure
from scipy.linalg import polar
from scipy.spatial.distance import cdist
from smact.screening import pauling_test

from ppmat.utils.crystal import lattices_to_params_shape_numpy

# Ignore warnings
warnings.filterwarnings("ignore", category=UserWarning)


# Warning: the smact package version is 2.5.5,
# different version may cause slight differences in accuracy.


CrystalNNFP = CrystalNNFingerprint.from_preset("ops")
CompFP = ElementProperty.from_preset("magpie")


def smact_validity(comp, count, use_pauling_test=True, include_alloys=True):
    elem_symbols = tuple([Element.from_Z(elem).symbol for elem in comp])
    space = smact.element_dictionary(elem_symbols)
    smact_elems = [e[1] for e in space.items()]
    electronegs = [e.pauling_eneg for e in smact_elems]
    ox_combos = [e.oxidation_states for e in smact_elems]
    if len(set(elem_symbols)) == 1:
        return True
    if include_alloys:
        is_metal_list = [(elem_s in smact.metals) for elem_s in elem_symbols]
        if all(is_metal_list):
            return True
    threshold = np.max(count)
    oxn = 1
    for oxc in ox_combos:
        oxn *= len(oxc)
    if oxn > 10000000.0:
        return False
    for ox_states in itertools.product(*ox_combos):
        stoichs = [(c,) for c in count]
        cn_e, cn_r = smact.neutral_ratios(
            ox_states, stoichs=stoichs, threshold=threshold
        )
        if cn_e:
            if use_pauling_test:
                try:
                    electroneg_OK = pauling_test(ox_states, electronegs)
                except TypeError:
                    electroneg_OK = True
            else:
                electroneg_OK = True
            if electroneg_OK:
                return True
    return False


def structure_validity(crystal, cutoff=0.5):
    dist_mat = crystal.distance_matrix
    dist_mat = dist_mat + np.diag(np.ones(tuple(dist_mat.shape)[0]) * (cutoff + 10.0))
    if dist_mat.min() < cutoff or crystal.volume < 0.1:
        return False
    else:
        return True


class Crystal(object):
    def __init__(self, crys_array_dict):
        if isinstance(crys_array_dict["frac_coords"], paddle.Tensor):
            self.frac_coords = crys_array_dict["frac_coords"].cpu().numpy()
        else:
            self.frac_coords = np.array(crys_array_dict["frac_coords"])
        if isinstance(crys_array_dict["atom_types"], paddle.Tensor):
            self.atom_types = crys_array_dict["atom_types"].cpu().numpy()
        else:
            self.atom_types = np.array(crys_array_dict["atom_types"])

        if "lengths" in crys_array_dict and "angles" in crys_array_dict:
            if isinstance(crys_array_dict["lengths"], paddle.Tensor):
                self.lengths = crys_array_dict["lengths"].cpu().numpy()
            else:
                self.lengths = np.array(crys_array_dict["lengths"])
            if isinstance(crys_array_dict["angles"], paddle.Tensor):
                self.angles = crys_array_dict["angles"].cpu().numpy()
            else:
                self.angles = np.array(crys_array_dict["angles"])
        else:
            if isinstance(crys_array_dict["lattice"], paddle.Tensor):
                lattice = crys_array_dict["lattice"].cpu().numpy()
            else:
                lattice = np.array([crys_array_dict["lattice"]])
            self.lengths, self.angles = lattices_to_params_shape_numpy(lattice)
            self.lengths, self.angles = self.lengths[0], self.angles[0]
        self.dict = {
            "frac_coords": self.frac_coords,
            "atom_types": self.atom_types,
            "lengths": self.lengths,
            "angles": self.angles,
        }
        if len(tuple(self.atom_types.shape)) > 1:
            self.dict["atom_types"] = np.argmax(self.atom_types, axis=-1) + 1
            self.atom_types = np.argmax(self.atom_types, axis=-1) + 1
        self.get_structure()
        self.get_composition()
        self.get_validity()
        self.get_fingerprints()

    def get_structure(self):
        if min(self.lengths.tolist()) < 0:
            self.constructed = False
            self.invalid_reason = "non_positive_lattice"
        if (
            np.isnan(self.lengths).any()
            or np.isnan(self.angles).any()
            or np.isnan(self.frac_coords).any()
        ):
            self.constructed = False
            self.invalid_reason = "nan_value"
        else:
            try:
                self.structure = Structure(
                    lattice=Lattice.from_parameters(
                        *(self.lengths.tolist() + self.angles.tolist())
                    ),
                    species=self.atom_types,
                    coords=self.frac_coords,
                    coords_are_cartesian=False,
                )
                self.constructed = True
                if self.structure.volume < 0.1:
                    self.constructed = False
                    self.invalid_reason = "unrealistically_small_lattice"
            except Exception:
                self.constructed = False
                self.invalid_reason = "construction_raises_exception"

    def get_composition(self):
        elem_counter = Counter(self.atom_types)
        composition = [
            (elem, elem_counter[elem]) for elem in sorted(elem_counter.keys())
        ]
        elems, counts = list(zip(*composition))
        counts = np.array(counts)
        counts = counts / np.gcd.reduce(counts)
        self.elems = elems
        self.comps = tuple(counts.astype("int").tolist())

    def get_validity(self):
        self.comp_valid = smact_validity(self.elems, self.comps)
        if self.constructed:
            self.struct_valid = structure_validity(self.structure)
        else:
            self.struct_valid = False
        self.valid = self.comp_valid and self.struct_valid

    def get_fingerprints(self):
        elem_counter = Counter(self.atom_types)
        comp = Composition(elem_counter)
        self.comp_fp = CompFP.featurize(comp)
        try:
            site_fps = [
                CrystalNNFP.featurize(self.structure, i)
                for i in range(len(self.structure))
            ]
        except Exception:
            self.valid = False
            self.comp_fp = None
            self.struct_fp = None
            return
        self.struct_fp = np.array(site_fps).mean(axis=0)


def get_crys_from_cif(cif, polar_decompose=False):
    structure = Structure.from_str(cif, fmt="cif")
    lattice = structure.lattice

    atom_types = np.array([site.specie.Z for site in structure])

    if polar_decompose:
        lattice_m = lattice.matrix
        _, lattice_m = polar(lattice.matrix)
        lengths, angles = lattices_to_params_shape_numpy(lattice_m)
        crys_array_dict = {
            "frac_coords": structure.frac_coords,
            "atom_types": atom_types,
            "lengths": lengths,
            "angles": angles,
        }
    else:
        crys_array_dict = {
            "frac_coords": structure.frac_coords,
            "atom_types": atom_types,
            "lengths": np.array(lattice.abc),
            "angles": np.array(lattice.angles),
        }
    return Crystal(crys_array_dict)


class FingerprintScaler:
    """Z-score scaler for fingerprint features used by metric computation.

    Pure numpy implementation, separate from the training-side paddle.nn.Layer
    scalers: metric computation runs under ``paddle.no_grad()``-free numpy
    post-processing and must not create paddle graph nodes. Handles NaN
    means/stds and zero-std columns: missing values fall back to 0 / 1 after
    standardization, and constant columns keep the raw value.
    """

    def __init__(self, means=None, stds=None, nan_replacement=None):
        self.means = means
        self.stds = stds
        self.nan_replacement = nan_replacement

    def fit(self, X):
        X = np.array(X).astype(float)
        self.means = np.nanmean(X, axis=0)
        self.stds = np.nanstd(X, axis=0)
        self.means = np.where(np.isnan(self.means), np.zeros(self.means.shape), self.means)
        self.stds = np.where(np.isnan(self.stds), np.ones(self.stds.shape), self.stds)
        self.stds = np.where(self.stds == 0, np.ones(self.stds.shape), self.stds)
        return self

    def transform(self, X):
        X = np.array(X).astype(float)
        transformed = (X - self.means) / self.stds
        if self.nan_replacement is not None:
            transformed = np.where(
                np.isnan(transformed), self.nan_replacement, transformed
            )
        return transformed

    def inverse_transform(self, X):
        X = np.array(X).astype(float)
        transformed = X * self.stds + self.means
        if self.nan_replacement is not None:
            transformed = np.where(
                np.isnan(transformed), self.nan_replacement, transformed
            )
        return transformed


def get_matches(structure, alternatives, matcher):
    """Indices and RMS distances of alternative structures matching ``structure``."""
    matches, rms_dists = [], []
    for idx, alt in enumerate(alternatives):
        rms_dist = matcher.get_rms_dist(structure, alt)
        if rms_dist is not None:
            rms_dists.append(rms_dist[0])
            matches.append(idx)
    return matches, rms_dists


def get_unique_structures(structures, matcher):
    """Return structurally unique structures (and their first-match indices)."""
    unique_structures = []
    unique_structure_idxs = []
    for i, structure in enumerate(structures):
        matches, _ = get_matches(structure, structures, matcher)
        first_match = sorted(matches)[0] if matches else i
        if first_match not in unique_structure_idxs:
            unique_structure_idxs.append(first_match)
            unique_structures.append(structures[first_match])
    return unique_structures, unique_structure_idxs


def get_chemsys(structure):
    return str(sorted(set(e.name for e in structure.composition.elements)))


def get_novel_structures(structures, reference_structures, matcher):
    """Return structures not present (within tolerance) in the reference set."""
    generated_chemsys = np.array(list(map(get_chemsys, structures)))
    reference_chemsys = np.array(list(map(get_chemsys, reference_structures)))

    intersection = np.intersect1d(generated_chemsys, reference_chemsys)
    gen_to_compare = np.isin(generated_chemsys, intersection)
    ref_to_compare = np.isin(reference_chemsys, intersection)
    filtered_reference = [
        s for compare, s in zip(ref_to_compare, reference_structures) if compare
    ]

    novel_structures = []
    novel_structure_idxs = []
    for i, (compare, structure) in enumerate(zip(gen_to_compare, structures)):
        if not compare:
            novel_structures.append(structure)
            novel_structure_idxs.append(i)

    for gen_idx, (compare, structure) in enumerate(zip(gen_to_compare, structures)):
        if not compare:
            continue
        matches, _ = get_matches(structure, filtered_reference, matcher)
        if len(matches) == 0:
            novel_structures.append(structure)
            novel_structure_idxs.append(gen_idx)

    return novel_structures, novel_structure_idxs


def compute_cov(crys, gt_crys, struc_cutoff, comp_cutoff, num_gen_crystals=None):
    """Coverage / matching metrics between generated and ground-truth crystals.

    Crystals lacking a valid structural OR composition fingerprint are
    dropped as a pair so the structural and composition fingerprints stay
    row-aligned. The composition fingerprints are standardized on the
    ground-truth set only (reference distribution), avoiding leakage from
    generated structures.
    """
    valid_crys = [
        c for c in crys if c.struct_fp is not None and c.comp_fp is not None
    ]
    valid_gt_crys = [
        c for c in gt_crys if c.struct_fp is not None and c.comp_fp is not None
    ]
    struc_fps = [c.struct_fp for c in valid_crys]
    comp_fps = [c.comp_fp for c in valid_crys]
    gt_struc_fps = [c.struct_fp for c in valid_gt_crys]
    gt_comp_fps = [c.comp_fp for c in valid_gt_crys]

    if len(struc_fps) == 0 or len(gt_struc_fps) == 0:
        raise ValueError(
            "compute_cov requires at least one valid fingerprint on each "
            f"side, got {len(struc_fps)} generated and "
            f"{len(gt_struc_fps)} ground-truth crystals with valid "
            "structural/composition fingerprints"
        )

    if num_gen_crystals is None:
        num_gen_crystals = len(struc_fps)

    scaler = FingerprintScaler(nan_replacement=0.0).fit(gt_comp_fps)
    comp_fps = scaler.transform(comp_fps)
    gt_comp_fps = scaler.transform(gt_comp_fps)

    struc_pdist = cdist(np.array(struc_fps), np.array(gt_struc_fps))
    comp_pdist = cdist(np.array(comp_fps), np.array(gt_comp_fps))

    struc_recall_dist = struc_pdist.min(axis=0)
    struc_precision_dist = struc_pdist.min(axis=1)
    comp_recall_dist = comp_pdist.min(axis=0)
    comp_precision_dist = comp_pdist.min(axis=1)

    cov_recall = np.mean(
        np.logical_and(struc_recall_dist <= struc_cutoff, comp_recall_dist <= comp_cutoff)
    )
    cov_precision = (
        np.sum(
            np.logical_and(
                struc_precision_dist <= struc_cutoff,
                comp_precision_dist <= comp_cutoff,
            )
        )
        / num_gen_crystals
    )

    return {
        "cov_recall": cov_recall,
        "cov_precision": cov_precision,
        "amsd_recall": np.mean(struc_recall_dist),
        "amsd_precision": np.mean(struc_precision_dist),
        "amcd_recall": np.mean(comp_recall_dist),
        "amcd_precision": np.mean(comp_precision_dist),
    }

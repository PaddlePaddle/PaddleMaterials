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

"""De novo generation (DNG) metrics for crystal structure generation."""

import json
import os
from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple

import numpy as np

from ppmat.datasets.build_structure import BuildStructure
from ppmat.metrics.utils import Crystal
from ppmat.utils import logger

_COMP_FP_STATS_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "comp_fp_stats.json",
)
with open(_COMP_FP_STATS_PATH) as _stats_file:
    _COMP_FP_STATS = json.load(_stats_file)

COMP_FP_MEANS = np.asarray(_COMP_FP_STATS["means"], dtype=np.float64)
COMP_FP_STDS = np.asarray(_COMP_FP_STATS["stds"], dtype=np.float64)

if COMP_FP_MEANS.shape != COMP_FP_STDS.shape:
    raise ValueError(
        "comp_fp_stats.json means and stds must have the same length; got "
        f"{COMP_FP_MEANS.shape} and {COMP_FP_STDS.shape}."
    )

_COV_CUTOFFS = {
    "mp_20": {"struc": 0.4, "comp": 10.0},
    "carbon_24": {"struc": 0.2, "comp": 4.0},
    "perov_5": {"struc": 0.2, "comp": 4.0},
}


def _normalize_comp_fps(fps) -> np.ndarray:
    """Z-score composition fingerprints and replace NaNs with zero."""
    normalized = (np.asarray(fps, dtype=np.float64) - COMP_FP_MEANS) / (
        COMP_FP_STDS + 1.0e-30
    )
    normalized[np.isnan(normalized)] = 0.0
    return normalized


def _mean_coordination_number(structure) -> float:
    """Mean coordination number via covalent-radii bonds (upstream)."""
    from ase.data import covalent_radii
    from pymatgen.io.ase import AseAtomsAdaptor

    atoms = AseAtomsAdaptor.get_atoms(structure)
    distances = atoms.get_all_distances(mic=True)
    radii = np.array([covalent_radii[number] for number in atoms.numbers])
    radii_sum = radii[:, None] + radii[None, :]
    np.fill_diagonal(distances, np.inf)
    coordination_numbers = (distances <= radii_sum * 1.25).sum(axis=1)
    return float(np.mean(coordination_numbers))


def _element_check(
    crystal_one: Crystal, crystal_two: Crystal, check_reduced: bool
) -> bool:
    """Check whether two crystals are of the same composition."""
    if check_reduced:
        # Crystal.comps stores the gcd-reduced counts of each element.
        return crystal_one.comps == crystal_two.comps
    counts_one = np.bincount(
        np.asarray(crystal_one.atom_types, dtype=np.int64), minlength=1
    )
    counts_two = np.bincount(
        np.asarray(crystal_two.atom_types, dtype=np.int64), minlength=1
    )
    length = max(len(counts_one), len(counts_two))
    counts_one = np.pad(counts_one, (0, length - len(counts_one)))
    counts_two = np.pad(counts_two, (0, length - len(counts_two)))
    return np.array_equal(counts_one, counts_two)


def _structure_matcher_rmsd(
    crystal_one: Crystal,
    crystal_two: Crystal,
    ltol: float,
    stol: float,
    angle_tol: float,
) -> Optional[float]:
    """StructureMatcher RMSD between two crystals, None if they do not match."""
    from pymatgen.analysis.structure_matcher import StructureMatcher

    matcher = StructureMatcher(ltol=ltol, stol=stol, angle_tol=angle_tol)
    result = matcher.get_rms_dist(crystal_one.structure, crystal_two.structure)
    return result[0] if result is not None else None


def _get_match_and_rmsd(
    crystal_one: Crystal,
    crystal_two: Crystal,
    ltol: float,
    stol: float,
    angle_tol: float,
    check_reduced: bool,
) -> Optional[float]:
    """RMSD if the crystals match (same composition + matcher), else None."""
    if not crystal_one.constructed or not crystal_two.constructed:
        return None
    if _element_check(crystal_one, crystal_two, check_reduced):
        return _structure_matcher_rmsd(crystal_one, crystal_two, ltol, stol, angle_tol)
    return None


def _get_match_and_rmsd_sequence(
    crystal_one: Crystal,
    sequence_two: Sequence[Crystal],
    ltol: float,
    stol: float,
    angle_tol: float,
    check_reduced: bool,
) -> Optional[List[Tuple[float, int]]]:
    """All (rmsd, index) matches of crystal_one against a reference sequence."""
    rmsds = []
    for index, crystal_two in enumerate(sequence_two):
        result = _get_match_and_rmsd(
            crystal_one, crystal_two, ltol, stol, angle_tol, check_reduced
        )
        if result is not None:
            rmsds.append((result, index))
    return rmsds if len(rmsds) > 0 else None


def metre_rmsds(
    crystal_list: Sequence[Crystal],
    ref_list: Sequence[Crystal],
    ltol: float = 0.3,
    stol: float = 0.5,
    angle_tol: float = 10.0,
    check_reduced: bool = True,
) -> Tuple[float, float, float, float, float, float]:
    """Match-everyone-to-reference rate: best RMSD per reference crystal kept."""

    rmsds_ref: List[Optional[float]] = [None for _ in range(len(ref_list))]
    valid_rmsds_ref: List[Optional[float]] = [None for _ in range(len(ref_list))]
    for gen in crystal_list:
        result = _get_match_and_rmsd_sequence(
            gen, ref_list, ltol, stol, angle_tol, check_reduced
        )
        if result is None:
            continue
        for rmsd, index in result:
            if rmsds_ref[index] is None or rmsd < rmsds_ref[index]:
                rmsds_ref[index] = rmsd
            if gen.valid and ref_list[index].valid:
                if valid_rmsds_ref[index] is None or rmsd < valid_rmsds_ref[index]:
                    valid_rmsds_ref[index] = rmsd

    def summarize(rmsds):
        match_count = sum(r is not None for r in rmsds)
        matched = [r for r in rmsds if r is not None]
        mean_rmsd = float(np.mean(matched)) if matched else 0.0
        corr_rmsds = [r if r is not None else stol for r in rmsds]
        corr_rmsd = float(np.mean(corr_rmsds))
        return match_count / len(ref_list), mean_rmsd, corr_rmsd

    rate, mean_rmsd, corr_rmsd = summarize(rmsds_ref)
    valid_rate, valid_mean_rmsd, valid_corr_rmsd = summarize(valid_rmsds_ref)
    return rate, mean_rmsd, corr_rmsd, valid_rate, valid_mean_rmsd, valid_corr_rmsd


def get_cov(
    gen_crystals: Sequence[Crystal],
    ref_crystals: Sequence[Crystal],
    struc_cutoff: float,
    comp_cutoff: float,
) -> Tuple[float, float]:
    """Coverage recall / precision on fingerprint distances.

    Mirrors the upstream ``get_cov``: composition fingerprints are z-scored
    with the FlowMM statistics and combined with CrystalNN structure
    fingerprints via pairwise distances. Generated crystals whose fingerprints
    could not be computed are dropped from the distances, but still counted in
    the precision denominator (upstream behaviour).

    Returns:
        (cov_recall, cov_precision).
    """
    if any(c.comp_fp is None or c.struct_fp is None for c in ref_crystals):
        raise ValueError(
            "Reference crystals with missing composition or structure fingerprints "
            "cannot be scored by COV; check the ground-truth CIFs."
        )
    ref_comp_fps = np.asarray([c.comp_fp for c in ref_crystals])
    ref_struc_fps = np.asarray([c.struct_fp for c in ref_crystals])

    # Drop generated crystals with missing fingerprints (upstream filtering).
    filtered_gen_struc_fps, filtered_gen_comp_fps = [], []
    for crystal in gen_crystals:
        if crystal.struct_fp is not None and crystal.comp_fp is not None:
            filtered_gen_struc_fps.append(crystal.struct_fp)
            filtered_gen_comp_fps.append(crystal.comp_fp)

    if not filtered_gen_comp_fps or not gen_crystals:
        return 0.0, 0.0

    ref_comp_fps = _normalize_comp_fps(ref_comp_fps)
    gen_comp_fps = _normalize_comp_fps(filtered_gen_comp_fps)
    gen_struc_fps = np.asarray(filtered_gen_struc_fps)

    from scipy.spatial.distance import cdist

    struc_pdist = cdist(gen_struc_fps, ref_struc_fps)
    comp_pdist = cdist(gen_comp_fps, ref_comp_fps)

    struc_recall_dist = struc_pdist.min(axis=0)
    struc_precision_dist = struc_pdist.min(axis=1)
    comp_recall_dist = comp_pdist.min(axis=0)
    comp_precision_dist = comp_pdist.min(axis=1)

    cov_recall = float(
        np.mean(
            np.logical_and(
                struc_recall_dist <= struc_cutoff, comp_recall_dist <= comp_cutoff
            )
        )
    )
    cov_precision = float(
        np.sum(
            np.logical_and(
                struc_precision_dist <= struc_cutoff, comp_precision_dist <= comp_cutoff
            )
        )
        / len(gen_crystals)
    )
    return cov_recall, cov_precision


class _OMatGCrystal(Crystal):
    """Crystal with the upstream OMatG validity checks.

    ``Crystal.valid`` is the conjunction of composition and structural validity.
    The upstream ``ValidAtoms`` class additionally requires the polar sine of the
    lattice to be non-degenerate and the fingerprints to be computable, so the
    same checks are applied here.
    """

    def __init__(self, data):
        self.constructed = False
        self.valid = False
        self.comp_valid = False
        self.struct_valid = False
        self.polar_sine_valid = False
        self.fingerprint_valid = False
        self.comp_fp = None
        self.struct_fp = None
        try:
            if isinstance(data, str):
                structure = BuildStructure.build_one(
                    data, "cif_str", niggli=False, canocial=False
                )
                data = {
                    "frac_coords": structure.frac_coords,
                    "atom_types": structure.atomic_numbers,
                    "lengths": list(structure.lattice.abc),
                    "angles": list(structure.lattice.angles),
                }
            super().__init__(data)
            self._apply_upstream_validity()
        except (KeyError, TypeError, ValueError, IndexError, RuntimeError) as error:
            self.constructed = False
            self.valid = False
            self.comp_fp = None
            self.struct_fp = None
            logger.warning(f"[DNGMetric] Invalid structure: {error}")

    def _apply_upstream_validity(self):
        # Upstream only evaluates SMACT when the structure itself is valid.
        self.comp_valid = bool(self.comp_valid and self.struct_valid)
        lengths = np.asarray(self.lengths, dtype=np.float64)
        if lengths.size == 3 and np.all(np.isfinite(lengths)):
            denominator = float(np.prod(lengths))
            volume = float(self.structure.lattice.volume) if self.constructed else 0.0
            self.polar_sine_valid = (
                denominator > 0.0 and volume / denominator > 1.0e-3
            )
        self.fingerprint_valid = self.comp_fp is not None and self.struct_fp is not None
        self.valid = bool(
            self.comp_valid
            and self.struct_valid
            and self.polar_sine_valid
            and self.fingerprint_valid
        )

    def get_structure(self):
        if (
            not np.isfinite(self.lengths).all()
            or not np.isfinite(self.angles).all()
            or not np.isfinite(self.frac_coords).all()
            or np.any(self.lengths <= 0)
        ):
            return
        self.structure = BuildStructure.build_one(
            self.dict, "array", niggli=False, canocial=False
        )
        self.constructed = self.structure.volume >= 0.1

    def get_fingerprints(self):
        if self.constructed:
            super().get_fingerprints()


def _parse_crystals(data: Sequence) -> List[Crystal]:
    """Parse every input, preserving failed entries in metric denominators."""
    return [_OMatGCrystal(item) for item in data]


def _constructible(crystals: Sequence[Crystal], label: str) -> List[Crystal]:
    """Return the crystals holding a parsed structure, warning about the rest.

    ``Crystal.get_structure`` leaves ``structure`` unset when the lattice cannot be
    built (``constructed`` is then False), so those crystals cannot contribute to
    the distribution metrics.
    """
    constructible = [crystal for crystal in crystals if crystal.constructed]
    dropped = len(crystals) - len(constructible)
    if dropped:
        logger.warning(
            f"[DNGMetric] {dropped} of {len(crystals)} {label} structures could not "
            f"be constructed; they are excluded from the distribution metrics and "
            f"counted as invalid."
        )
    return constructible


class DNGMetric:
    """De novo generation (DNG) metrics for crystal structure generation.

    Reports the validity rate, match-everyone-to-reference (METRe) statistics,
    Wasserstein distances over density / n-arity / coordination number and, for
    datasets with published cutoffs, composition-structure coverage. The metric
    only consumes generated structure dictionaries and a reference set, so it is
    independent of the model that produced them.

    Args:
        gt_file_path (str): CSV file with a ``cif`` column holding the reference
            structures. Used when ``gt_data`` is not passed to ``__call__``.
        dataset_name (str, optional): Dataset key selecting the coverage
            cutoffs. Defaults to ``"mp_20"``.
        ltol, stol, angle_tol: ``StructureMatcher`` tolerances.
        check_reduced (bool): Compare gcd-reduced compositions instead of counts.
    """

    def __init__(
        self,
        gt_file_path: Optional[str] = None,
        dataset_name: Optional[str] = "mp_20",
        ltol: float = 0.3,
        stol: float = 0.5,
        angle_tol: float = 10.0,
        check_reduced: bool = True,
    ) -> None:
        self._dataset_name = dataset_name
        self._gt_file_path = gt_file_path
        self._ltol = ltol
        self._stol = stol
        self._angle_tol = angle_tol
        self._check_reduced = check_reduced
        self._gt_crystals: Optional[List[Crystal]] = None

    def _load_reference(self, gt_data: Optional[Sequence]) -> List[Crystal]:
        if gt_data is not None:
            return _parse_crystals(gt_data)
        if self._gt_crystals is None:
            if self._gt_file_path is None:
                raise ValueError(
                    "gt_data is None, but no gt_file_path was provided to DNGMetric."
                )
            import pandas as pd

            csv = pd.read_csv(self._gt_file_path)
            self._gt_crystals = _parse_crystals(csv["cif"])
        return self._gt_crystals

    def _dng_metrics(
        self, gen_crystals: List[Crystal], ref_crystals: List[Crystal]
    ) -> Dict[str, float]:
        from scipy.stats import wasserstein_distance

        valid_rate = float(sum(c.valid for c in gen_crystals) / len(gen_crystals))
        composition_valid_rate = float(
            sum(c.comp_valid for c in gen_crystals) / len(gen_crystals)
        )
        structural_valid_rate = float(
            sum(c.struct_valid for c in gen_crystals) / len(gen_crystals)
        )

        gen_measurable = _constructible(gen_crystals, "generated")
        ref_measurable = _constructible(ref_crystals, "reference")
        if not gen_measurable or not ref_measurable:
            logger.warning(
                "[DNGMetric] Distribution metrics need at least one constructible "
                "generated and reference structure; reporting the validity rate only."
            )
            return {
                "valid_rate": valid_rate,
                "composition_valid_rate": composition_valid_rate,
                "structural_valid_rate": structural_valid_rate,
            }

        def wdist(key):
            gen_values = [getattr(c.structure, key) for c in gen_measurable]
            ref_values = [getattr(c.structure, key) for c in ref_measurable]
            return float(wasserstein_distance(gen_values, ref_values))

        wdist_density = wdist("density")
        gen_narity = [len(set(c.structure.species)) for c in gen_measurable]
        ref_narity = [len(set(c.structure.species)) for c in ref_measurable]
        wdist_narity = float(wasserstein_distance(gen_narity, ref_narity))
        gen_cn = [_mean_coordination_number(c.structure) for c in gen_measurable]
        ref_cn = [_mean_coordination_number(c.structure) for c in ref_measurable]
        wdist_coordination_numbers = float(wasserstein_distance(gen_cn, ref_cn))
        wdist_avg = float(
            np.average([wdist_density, wdist_narity, wdist_coordination_numbers])
        )

        (
            metre_rate,
            metre_mean_rmsd,
            metre_corr_rmsd,
            metre_valid_rate,
            metre_valid_mean_rmsd,
            metre_valid_corr_rmsd,
        ) = metre_rmsds(
            gen_crystals,
            ref_crystals,
            ltol=self._ltol,
            stol=self._stol,
            angle_tol=self._angle_tol,
            check_reduced=self._check_reduced,
        )

        metrics = {
            "valid_rate": valid_rate,
            "composition_valid_rate": composition_valid_rate,
            "structural_valid_rate": structural_valid_rate,
            "wdist_density": wdist_density,
            "wdist_narity": wdist_narity,
            "wdist_coordination_numbers": wdist_coordination_numbers,
            "metre_rate": metre_rate,
            "metre_mean_rmsd": metre_mean_rmsd,
            "metre_corr_rmsd": metre_corr_rmsd,
            "metre_valid_rate": metre_valid_rate,
            "metre_valid_mean_rmsd": metre_valid_mean_rmsd,
            "metre_valid_corr_rmsd": metre_valid_corr_rmsd,
        }
        if self._dataset_name in _COV_CUTOFFS:
            cutoffs = _COV_CUTOFFS[self._dataset_name]
            cov_recall, cov_precision = get_cov(
                gen_crystals, ref_crystals, cutoffs["struc"], cutoffs["comp"]
            )
            cov_avg = float(np.average([1.0 - cov_precision, 1.0 - cov_recall]))
            dng_eval = float(np.average([cov_avg, wdist_avg, 1.0 - valid_rate]))
            metrics.update(
                {
                    "cov_precision": cov_precision,
                    "cov_recall": cov_recall,
                    "dng_eval": dng_eval,
                }
            )
        else:
            dng_eval = float(np.average([wdist_avg, 1.0 - valid_rate]))
            metrics["dng_eval"] = dng_eval
        return metrics

    def __call__(
        self, pred_data: Sequence[dict], gt_data: Optional[Sequence] = None
    ) -> Dict[str, float]:
        """Score generated structures against the reference set."""
        if len(pred_data) == 0:
            logger.warning("[DNGMetric] No generated structures to evaluate.")
            return {}
        gen_crystals = _parse_crystals(pred_data)
        ref_crystals = self._load_reference(gt_data)
        if len(ref_crystals) == 0:
            logger.warning(
                "[DNGMetric] The reference set is empty; nothing to score."
            )
            return {}
        return self._dng_metrics(gen_crystals, ref_crystals)

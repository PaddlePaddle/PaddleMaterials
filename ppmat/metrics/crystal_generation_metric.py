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

"""Model-independent validity, diversity and coverage of generated crystals."""

import json
from importlib import resources
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from ppmat.metrics.streaming_base import StreamingMetricBase


class CrystalGenerationMetric(StreamingMetricBase):
    """Evaluate crystal arrays with a shared batch and streaming implementation.

    ``input_format`` accepts array, cif_str or cif_file via BuildStructure.
    ``metrics`` selects validity, uniqueness, novelty and/or coverage. Validity
    uses the existing structure/SMACT checks; fingerprint failures affect only
    coverage. Uniqueness and novelty use valid generated structures as their
    denominator. Coverage uses paired available fingerprints on each side.

    ``gt_file_path`` supplies reference CIFs for novelty/coverage. An optional
    ``novelty_gt_file_path`` selects a separate novelty pool, e.g. training CIFs
    for novelty and test CIFs for coverage. Explicit ``gt_data`` overrides the
    common reference pool in a batch call.

    ``matcher_cfg`` configures StructureMatcher. ``coverage_cfg`` must specify
    struc_cutoff, comp_cutoff, distance_batch_size, and either statistics_path
    or statistics_resource (package/name). Statistics contain mean/std and
    ordered Magpie feature_labels; no normalization is fitted to generated or
    evaluation data.
    No generated fingerprints means zero coverage and undefined (NaN) distances;
    a reference pool without fingerprints is rejected.

    Streaming state is process-local. Comparisons span all accumulated batches;
    reset preserves reference caches. Only the sample stage is evaluated.
    """

    def __init__(
        self,
        metrics=("validity", "uniqueness"),
        input_format: str = "array",
        gt_file_path: Optional[str] = None,
        novelty_gt_file_path: Optional[str] = None,
        matcher_cfg: Optional[dict] = None,
        coverage_cfg: Optional[dict] = None,
        structure_validity_cfg: Optional[dict] = None,
        composition_validity_cfg: Optional[dict] = None,
        n_structures: Optional[int] = None,
    ):
        from pymatgen.analysis.structure_matcher import StructureMatcher

        self.metrics = tuple(metrics)
        unknown = set(self.metrics) - {"validity", "uniqueness", "novelty", "coverage"}
        if not self.metrics or unknown:
            raise ValueError(f"Unsupported crystal generation metrics: {self.metrics}")
        if input_format not in ("array", "cif_str", "cif_file"):
            raise ValueError(f"Unsupported evaluation input format: {input_format}")
        self.input_format = input_format
        if n_structures is not None and n_structures <= 0:
            raise ValueError("n_structures must be positive.")
        self.gt_file_path = gt_file_path
        self.novelty_gt_file_path = novelty_gt_file_path
        self.matcher = StructureMatcher(**(matcher_cfg or {}))
        self.structure_validity_cfg = structure_validity_cfg or {}
        self.composition_validity_cfg = composition_validity_cfg or {}
        self.n_structures = n_structures
        self._reference_cache = {}
        if "coverage" in self.metrics:
            from ppmat.metrics.utils import CompFP

            if coverage_cfg is None:
                raise ValueError("coverage_cfg is required for coverage.")
            self.struc_cutoff = coverage_cfg["struc_cutoff"]
            self.comp_cutoff = coverage_cfg["comp_cutoff"]
            self.distance_batch_size = coverage_cfg["distance_batch_size"]
            if self.distance_batch_size <= 0:
                raise ValueError("distance_batch_size must be positive.")
            if self.struc_cutoff < 0 or self.comp_cutoff < 0:
                raise ValueError("Coverage cutoffs must be nonnegative.")
            has_path = "statistics_path" in coverage_cfg
            has_resource = "statistics_resource" in coverage_cfg
            if has_path == has_resource:
                raise ValueError("Specify one statistics_path or statistics_resource.")
            if has_path:
                statistics_path = Path(coverage_cfg["statistics_path"]).expanduser()
            else:
                resource = coverage_cfg["statistics_resource"]
                statistics_path = resources.files(resource["package"]).joinpath(
                    resource["name"]
                )
            with statistics_path.open(encoding="utf-8") as file_obj:
                statistics = json.load(file_obj)
            labels = CompFP.feature_labels()
            self.comp_mean = np.asarray(statistics["mean"], dtype=float)
            self.comp_std = np.asarray(statistics["std"], dtype=float)
            if (
                statistics["feature_labels"] != labels
                or self.comp_mean.shape != (len(labels),)
                or self.comp_std.shape != (len(labels),)
                or not np.isfinite(self.comp_mean).all()
                or not np.isfinite(self.comp_std).all()
                or (self.comp_std <= 0).any()
            ):
                raise ValueError(
                    "Statistics must match finite ordered Magpie features."
                )
        self.reset()

    def reset(self):
        self._pred_data = []

    def update_step(self, *, result, batch, stage):
        if stage != "sample" or result is None:
            return
        samples = result.get("samples", result) if isinstance(result, dict) else result
        if isinstance(samples, dict):
            samples = samples["result"]
        remaining = (
            None
            if self.n_structures is None
            else self.n_structures - len(self._pred_data)
        )
        self._pred_data.extend(samples[:remaining])

    def compute_epoch(self, *, stage):
        if stage != "sample" or not self._pred_data:
            return {}
        return self(self._pred_data)

    def _prepare_samples(self, samples, format, fingerprints):
        from paddle import Tensor

        from ppmat.datasets.build_structure import BuildStructure
        from ppmat.metrics.utils import CompFP
        from ppmat.metrics.utils import CrystalNNFP
        from ppmat.metrics.utils import smact_validity
        from ppmat.metrics.utils import structure_validity

        crystals = []
        for sample in samples:
            crystal = {
                "structure": None,
                "valid": False,
                "struct_fp": None,
                "comp_fp": None,
            }
            crystals.append(crystal)
            if format == "array":
                sample = {
                    key: value.tolist()
                    if isinstance(value, (np.ndarray, Tensor))
                    else value
                    for key, value in sample.items()
                }
            try:
                structure = BuildStructure.build_one(
                    sample, format, niggli=False, canocial=False
                )
                if (
                    not len(structure)
                    or not np.isfinite(structure.lattice.matrix).all()
                    or not np.isfinite(structure.frac_coords).all()
                ):
                    continue
            except (ValueError, TypeError, KeyError, IndexError, OverflowError):
                continue
            elements, counts = np.unique(structure.atomic_numbers, return_counts=True)
            counts = counts // np.gcd.reduce(counts)
            comp_valid = smact_validity(
                elements, counts, **self.composition_validity_cfg
            )
            struct_valid = structure_validity(structure, **self.structure_validity_cfg)
            crystal["structure"] = structure
            crystal["valid"] = comp_valid and struct_valid
            if fingerprints:
                try:
                    comp_fp = CompFP.featurize(structure.composition)
                    site_fps = [
                        CrystalNNFP.featurize(structure, index)
                        for index in range(len(structure))
                    ]
                except (ValueError, TypeError, KeyError, IndexError, RuntimeError):
                    continue
                crystal["comp_fp"] = comp_fp
                crystal["struct_fp"] = np.asarray(site_fps).mean(axis=0)
        return crystals

    def _load_reference(self, path, fingerprints):
        if path is None:
            raise ValueError("A reference CSV or explicit gt_data is required.")
        key = (path, fingerprints)
        if key not in self._reference_cache:
            samples = pd.read_csv(path)["cif"].tolist()
            self._reference_cache[key] = self._prepare_samples(
                samples, "cif_str", fingerprints
            )
        return self._reference_cache[key]

    def _coverage(self, crystals, references):
        generated = [c for c in crystals if c["struct_fp"] is not None]
        reference = [c for c in references if c["struct_fp"] is not None]
        if not reference:
            raise ValueError("Coverage requires a reference with valid fingerprints.")
        if not generated:
            return {
                "cov_recall": 0.0,
                "cov_precision": 0.0,
                "amsd_recall": float("nan"),
                "amsd_precision": float("nan"),
                "amcd_recall": float("nan"),
                "amcd_precision": float("nan"),
            }
        reference_struct = np.asarray([c["struct_fp"] for c in reference])
        reference_comp = np.asarray([c["comp_fp"] for c in reference])
        reference_comp = (reference_comp - self.comp_mean) / self.comp_std
        reference_comp = np.where(np.isnan(reference_comp), 0.0, reference_comp)
        struct_recall = np.full(len(reference), np.inf)
        comp_recall = np.full(len(reference), np.inf)
        struct_precision = np.empty(len(generated))
        comp_precision = np.empty(len(generated))
        for start in range(0, len(generated), self.distance_batch_size):
            stop = start + self.distance_batch_size
            block = generated[start:stop]
            struct_fp = np.asarray([c["struct_fp"] for c in block])
            comp_fp = np.asarray([c["comp_fp"] for c in block])
            comp_fp = (comp_fp - self.comp_mean) / self.comp_std
            comp_fp = np.where(np.isnan(comp_fp), 0.0, comp_fp)
            struct_dist = cdist(struct_fp, reference_struct)
            comp_dist = cdist(comp_fp, reference_comp)
            struct_recall = np.minimum(struct_recall, struct_dist.min(axis=0))
            comp_recall = np.minimum(comp_recall, comp_dist.min(axis=0))
            struct_precision[start:stop] = struct_dist.min(axis=1)
            comp_precision[start:stop] = comp_dist.min(axis=1)
        return {
            "cov_recall": float(
                np.mean(
                    (struct_recall <= self.struc_cutoff)
                    & (comp_recall <= self.comp_cutoff)
                )
            ),
            "cov_precision": float(
                np.mean(
                    (struct_precision <= self.struc_cutoff)
                    & (comp_precision <= self.comp_cutoff)
                )
            ),
            "amsd_recall": float(struct_recall.mean()),
            "amsd_precision": float(struct_precision.mean()),
            "amcd_recall": float(comp_recall.mean()),
            "amcd_precision": float(comp_precision.mean()),
        }

    def __call__(self, pred_data, gt_data=None):
        if not pred_data:
            raise ValueError("Evaluation requires at least one generated structure.")
        fingerprints = "coverage" in self.metrics
        crystals = self._prepare_samples(
            pred_data[: self.n_structures], self.input_format, fingerprints
        )
        valid = [c["structure"] for c in crystals if c["valid"]]
        result = {}
        if "validity" in self.metrics:
            result["validity"] = len(valid) / len(crystals)
        if "uniqueness" in self.metrics:
            representatives = set()
            for index, structure in enumerate(valid):
                first = next(
                    (
                        i
                        for i, other in enumerate(valid)
                        if self.matcher.get_rms_dist(structure, other) is not None
                    ),
                    index,
                )
                representatives.add(first)
            result["uniqueness"] = len(representatives) / len(valid) if valid else 0.0
        if fingerprints or (
            "novelty" in self.metrics and self.novelty_gt_file_path is None
        ):
            references = (
                self._prepare_samples(gt_data, self.input_format, fingerprints)
                if gt_data is not None
                else self._load_reference(self.gt_file_path, fingerprints)
            )
        if "novelty" in self.metrics:
            novelty_reference = (
                self._load_reference(self.novelty_gt_file_path, False)
                if self.novelty_gt_file_path is not None
                else references
            )
            reference_by_chemistry = {}
            for crystal in novelty_reference:
                if crystal["valid"]:
                    structure = crystal["structure"]
                    key = frozenset(structure.atomic_numbers)
                    reference_by_chemistry.setdefault(key, []).append(structure)
            if not reference_by_chemistry:
                raise ValueError(
                    "Novelty requires at least one valid reference structure."
                )
            novel = sum(
                not any(
                    self.matcher.get_rms_dist(structure, other) is not None
                    for other in reference_by_chemistry.get(
                        frozenset(structure.atomic_numbers), []
                    )
                )
                for structure in valid
            )
            result["novelty"] = novel / len(valid) if valid else 0.0
        if fingerprints:
            result.update(self._coverage(crystals, references))
        return result

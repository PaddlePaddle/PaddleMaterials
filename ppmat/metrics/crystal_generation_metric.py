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
"""Distribution, validity and structural diversity metrics for generated crystals."""

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

from ppmat.metrics.streaming_base import StreamingMetricBase
from ppmat.utils import download


class CrystalGenerationMetric(StreamingMetricBase):
    """Evaluate generated structures against explicit MP20 reference CSV files.

    Rates use all generated samples as their denominator, including parse
    failures. Unique and UN count the first representative of each structural
    cluster. Novelty compares every constructed sample against training data.
    Distribution distances use jointly chemically and structurally valid samples.
    State is local to one process; global comparison spans every update call.
    """

    def __init__(
        self,
        train_file_path: str,
        test_file_path: str,
        stol: float = 0.3,
        angle_tol: float = 5.0,
        ltol: float = 0.2,
        num_cpus: int = 4,
        reference_archive_cfg: dict | None = None,
    ):
        from pymatgen.analysis.structure_matcher import StructureMatcher

        from ppmat.datasets.build_structure import BuildStructure

        self.matcher = StructureMatcher(stol=stol, angle_tol=angle_tol, ltol=ltol)
        converter = BuildStructure(
            format="cif_str", niggli=False, canocial=False, num_cpus=num_cpus
        )
        reference_paths = []
        for source in (train_file_path, test_file_path):
            path = Path(source)
            if not path.is_file() and reference_archive_cfg is not None:
                root = download.get_datasets_path_from_url(
                    reference_archive_cfg["url"], reference_archive_cfg["md5"]
                )
                path = Path(root) / reference_archive_cfg["name"] / path.name
            reference_paths.append(path)
        self.reference = defaultdict(list)
        for structure in converter(pd.read_csv(reference_paths[0])["cif"].tolist()):
            self.reference[structure.composition.fractional_composition].append(
                structure
            )
        test_structures = converter(pd.read_csv(reference_paths[1])["cif"].tolist())
        self.test_densities = [float(s.density) for s in test_structures]
        self.test_elements = [len(s.composition) for s in test_structures]
        self.reset()

    def reset(self):
        self.total = 0
        self.chemical_valid = 0
        self.structural_valid = 0
        self.overall_valid = 0
        self.unique = 0
        self.novel = 0
        self.unique_novel = 0
        self.seen = defaultdict(list)
        self.densities = []
        self.elements = []

    def update_step(self, *, result, batch=None, stage="sample"):
        if stage != "sample":
            return
        from ppmat.datasets.build_structure import BuildStructure
        from ppmat.metrics.utils import smact_validity
        from ppmat.metrics.utils import structure_validity

        for item in result["result"]:
            self.total += 1
            try:
                structure = BuildStructure.build_one(
                    item, "array", niggli=False, canocial=False
                )
            except (ValueError, TypeError, KeyError, IndexError):
                continue
            if (
                len(structure) == 0
                or not np.isfinite(structure.lattice.matrix).all()
                or not np.isfinite(structure.frac_coords).all()
                or structure.volume <= 0
            ):
                continue
            numbers, counts = np.unique(structure.atomic_numbers, return_counts=True)
            counts = counts // np.gcd.reduce(counts)
            chemical_valid = smact_validity(numbers, counts)
            structural_valid = structure_validity(structure)
            self.chemical_valid += int(chemical_valid)
            self.structural_valid += int(structural_valid)
            if chemical_valid and structural_valid:
                self.overall_valid += 1
                self.densities.append(float(structure.density))
                self.elements.append(len(numbers))

            composition = structure.composition.fractional_composition
            is_unique = not any(
                self.matcher.fit(structure, previous)
                for previous in self.seen[composition]
            )
            is_novel = not any(
                self.matcher.fit(structure, reference)
                for reference in self.reference.get(composition, [])
            )
            self.novel += int(is_novel)
            if is_unique:
                self.seen[composition].append(structure)
                self.unique += 1
                self.unique_novel += int(is_novel)

    def compute_epoch(self, *, stage="sample"):
        if stage != "sample":
            return {}
        if not self.total:
            raise ValueError("No generated structures were provided for evaluation.")
        metrics = {
            "structural_validity": self.structural_valid / self.total,
            "chemical_validity": self.chemical_valid / self.total,
            "overall_validity": self.overall_valid / self.total,
            "unique_rate": self.unique / self.total,
            "novel_rate": self.novel / self.total,
            "un_rate": self.unique_novel / self.total,
        }
        if not self.densities:
            raise ValueError(
                "No valid generated structures for distribution distances."
            )
        metrics["density_wasserstein"] = float(
            wasserstein_distance(self.densities, self.test_densities)
        )
        metrics["elements_wasserstein"] = float(
            wasserstein_distance(self.elements, self.test_elements)
        )
        return metrics

    def __call__(self, pred_data):
        self.reset()
        self.update_step(result={"result": pred_data})
        return self.compute_epoch()

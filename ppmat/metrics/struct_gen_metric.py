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

from __future__ import annotations

import pickle
from typing import TYPE_CHECKING
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence

import numpy as np

from ppmat.datasets.build_structure import BuildStructure
from ppmat.metrics.streaming_base import StreamingMetricBase

if TYPE_CHECKING:
    from pymatgen.core.composition import Composition
    from pymatgen.core.structure import Structure


class StructGenMetric(StreamingMetricBase):
    """Metrics for unconditional crystal structure generation.

    Evaluates a list of generated structures (the ``array`` dicts produced by
    structure generation models) with three standard indicators:

    - ``validity``: fraction of generated structures that pass the shared
      ``Crystal`` validity checks (smact composition validity + distances).
    - ``uniqueness``: fraction of unique structures among valid ones, judged
      by ``StructureMatcher.group_structures``.
    - ``novelty``: fraction of unique structures that do not match any
      structure in an optional reference set, passed either as a pickle of
      pymatgen structures (``reference_file_path``) or directly as a list
      (``reference_structures``).

    One-shot and streaming evaluation share the same global structure pool.
    """

    def __init__(
        self,
        reference_file_path: Optional[str] = None,
        reference_structures: Optional[Sequence[Structure]] = None,
        stol: float = 0.5,
        angle_tol: float = 10,
        ltol: float = 0.3,
    ) -> None:
        from pymatgen.analysis.structure_matcher import StructureMatcher

        if reference_file_path is not None and reference_structures is not None:
            raise ValueError(
                "Pass either reference_file_path or reference_structures, not both."
            )
        if reference_file_path is not None and not reference_file_path.endswith(".pkl"):
            raise ValueError(
                "reference_file_path should be a pickle file with "
                "pymatgen structures, got: " + reference_file_path
            )
        self.matcher = StructureMatcher(stol=stol, angle_tol=angle_tol, ltol=ltol)
        if reference_file_path is not None:
            with open(reference_file_path, "rb") as f:
                reference_structures = pickle.load(f)
        if reference_structures is not None and not isinstance(
            reference_structures, (list, tuple)
        ):
            raise ValueError(
                "reference structures must be a list of pymatgen structures"
            )
        self.reference_structures = reference_structures
        self._reference_index: Optional[Dict[Composition, List[Structure]]] = None
        if reference_structures is not None:
            self._reference_index = {}
            for ref_structure in reference_structures:
                self._reference_index.setdefault(
                    self._composition_key(ref_structure), []
                ).append(ref_structure)
        self.reset()

    def update_step(self, *, result, batch, stage):
        if "result" in result:
            self._samples.setdefault(stage, []).extend(result["result"])

    def compute_epoch(self, *, stage):
        return self(self._samples.get(stage, []))

    def reset(self):
        self._samples = {}

    @staticmethod
    def _composition_key(structure: Structure) -> Composition:
        return structure.composition.element_composition.fractional_composition

    def _novelty_one(
        self,
        structure: Structure,
        reference_index: Dict[Composition, List[Structure]],
    ) -> bool:
        candidates = reference_index.get(self._composition_key(structure), [])
        # Novel: matches none of the reference structures
        return all(
            not self.matcher.fit(structure, ref_structure)
            for ref_structure in candidates
        )

    def __call__(self, pred_data: Sequence[Dict[str, Any]]) -> Dict[str, float]:
        from ppmat.metrics.utils import smact_validity
        from ppmat.metrics.utils import structure_validity

        valid_structures = []
        for data in pred_data:
            try:
                data = dict(data)
                if "lattice" in data:
                    data["lattice"] = np.asarray(data["lattice"]).tolist()
                structure = BuildStructure.build_one(
                    data, "array", niggli=False, canocial=False
                )
                elements, counts = np.unique(
                    structure.atomic_numbers, return_counts=True
                )
                counts = counts // np.gcd.reduce(counts)
                if smact_validity(
                    tuple(elements), tuple(counts)
                ) and structure_validity(structure):
                    valid_structures.append(structure)
            except (
                KeyError,
                TypeError,
                ValueError,
                RuntimeError,
                np.linalg.LinAlgError,
            ):
                continue
        total = len(pred_data)
        n_valid = len(valid_structures)

        results = {
            "validity": n_valid / total if total > 0 else 0.0,
            "uniqueness": 0.0,
        }
        if self.reference_structures is not None:
            results["novelty"] = 0.0
        if n_valid == 0:
            return results

        groups = self.matcher.group_structures(valid_structures)
        n_unique = len(groups)
        results["uniqueness"] = n_unique / n_valid

        if self.reference_structures is not None:
            unique_structures = [group[0] for group in groups]
            flags = [
                self._novelty_one(structure, self._reference_index)
                for structure in unique_structures
            ]
            results["novelty"] = sum(flags) / max(n_unique, 1)
        return results

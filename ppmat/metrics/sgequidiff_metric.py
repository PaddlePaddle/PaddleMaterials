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

"""SGEQuiDiff generation quality metrics (validity / uniqueness /
novelty / coverage)."""

from __future__ import annotations

from typing import Any
from typing import Dict
from typing import List
from typing import Optional

import numpy as np
import pandas as pd
from pymatgen.analysis.structure_matcher import StructureMatcher

from ppmat.metrics.streaming_base import StreamingMetricBase
from ppmat.metrics.utils import Crystal
from ppmat.metrics.utils import compute_cov
from ppmat.metrics.utils import get_crys_from_cif
from ppmat.metrics.utils import get_novel_structures
from ppmat.metrics.utils import get_unique_structures


def _extract_samples(result: Any) -> Optional[List[Dict[str, Any]]]:
    """Extract a list of crystal dicts from a sampling result container.

    Accepted shapes:
      - a plain list of crystal dicts,
      - {"result": [...]},
      - {"samples": [...]} / {"samples": {"result": [...]}}.
    """
    samples = result
    if isinstance(samples, dict):
        samples = samples.get("samples")
        if samples is None:
            samples = result.get("result")
    if isinstance(samples, dict) and "result" in samples:
        samples = samples["result"]
    if isinstance(samples, list) and len(samples) > 0:
        return samples
    return None


class SGEQuiDiffMetric(StreamingMetricBase):
    """Generation-quality metrics for SGEQuiDiff (streaming compatible).

    Implements the ``StreamingMetricBase`` contract: generated structures are
    collected per ``update_step`` (``stage == "sample"``) and the full metric
    set (validity / uniqueness / novelty / coverage) is computed in
    ``compute_epoch``. The legacy batch interface ``__call__(pred_data,
    gt_data)`` remains the entry used by
    ``StructureSampler.compute_metric`` today; the streaming interface is
    wired the same way as molecular samplers and is ready for
    chunked/streaming generation loops.

    Args:
        gt_file_path: Optional CSV with a ``"cif"`` column used as the
            evaluation reference set (validity baseline, and the reference
            pool for novelty / coverage; pass the **train** split to measure
            novelty against the training distribution).
        stol / angle_tol / ltol: ``StructureMatcher`` tolerance parameters
            (structure distance, angle tolerance in degrees, and fractional
            length tolerance) used for uniqueness and novelty matching.
        struc_cutoff: Structural fingerprint distance cutoff for coverage.
        comp_cutoff: Composition fingerprint distance cutoff for coverage.
        n_structures: Optional cap on the number of generated structures scored.
    """

    def __init__(
        self,
        gt_file_path: Optional[str] = None,
        stol: float = 0.5,
        angle_tol: float = 10,
        ltol: float = 0.3,
        struc_cutoff: float = 0.4,
        comp_cutoff: float = 10.0,
        n_structures: Optional[int] = None,
    ):
        super().__init__()
        self.gt_file_path = gt_file_path
        self.matcher = StructureMatcher(stol=stol, angle_tol=angle_tol, ltol=ltol)
        self.struc_cutoff = struc_cutoff
        self.comp_cutoff = comp_cutoff
        self.n_structures = n_structures
        self._gt_crys: Optional[List[Crystal]] = None
        self.reset()

    def _load_gt_crys(self) -> List[Crystal]:
        if self._gt_crys is None:
            if not self.gt_file_path:
                raise ValueError("gt_file_path is required when gt_data is None")
            csv = pd.read_csv(self.gt_file_path)
            self._gt_crys = [get_crys_from_cif(cif) for cif in csv["cif"].tolist()]
        return self._gt_crys

    # ---- streaming interface ----
    def reset(self):
        self._pred_data: List[Dict[str, Any]] = []

    def update_step(self, *, result: Any, batch: Any, stage: str):
        if stage != "sample" or result is None:
            return
        samples = _extract_samples(result)
        if samples is not None:
            self._pred_data.extend(samples)

    def compute_epoch(self, *, stage: str) -> Dict[str, float]:
        if stage != "sample" or not self._pred_data:
            return {}
        return self._compute(self._pred_data)

    # ---- batch interface ----
    def __call__(self, pred_data: Any, gt_data: Any = None) -> dict:
        """Score generated structures against ground truth (batch interface).

        Args:
            pred_data: list of dicts (``num_atoms`` / ``atom_types`` /
                ``frac_coords`` / ``lengths`` / ``angles``).
            gt_data: optional list of dicts; falls back to ``gt_file_path``.
        """
        return self._compute(pred_data, gt_data)

    def _compute(self, pred_data: Any, gt_data: Any = None) -> dict:
        if not pred_data:
            raise ValueError(
                "pred_data is empty; generation-quality metrics require at "
                "least one generated structure"
            )
        pred_crys = [Crystal(d) for d in pred_data]
        if self.n_structures is not None:
            pred_crys = pred_crys[: self.n_structures]

        if gt_data is not None:
            gt_crys = [Crystal(d) for d in gt_data]
        else:
            gt_crys = self._load_gt_crys()

        validity = float(np.mean([c.valid for c in pred_crys]))

        valid_structs = [c.structure for c in pred_crys if c.valid]
        n_valid = len(valid_structs)

        uniqueness = 0.0
        novelty = 0.0
        if n_valid > 0:
            unique, _ = get_unique_structures(valid_structs, self.matcher)
            uniqueness = len(unique) / n_valid

            gt_valid_structs = [c.structure for c in gt_crys if c.valid]
            novel, _ = get_novel_structures(
                valid_structs, gt_valid_structs, self.matcher
            )
            novelty = len(novel) / n_valid

        cov_metrics = compute_cov(
            pred_crys,
            gt_crys,
            self.struc_cutoff,
            self.comp_cutoff,
        )

        return {
            "validity": float(validity),
            "uniqueness": float(uniqueness),
            "novelty": float(novelty),
            **{k: float(v) for k, v in cov_metrics.items()},
        }

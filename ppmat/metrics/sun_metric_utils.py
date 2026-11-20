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

import hashlib
import io
import os
import pickle
import tempfile
from collections import defaultdict
from datetime import datetime
from importlib.metadata import version
from typing import TYPE_CHECKING
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

import numpy as np

from ppmat.datasets.build_structure import BuildStructure
from ppmat.metrics.streaming_base import StreamingMetricBase
from ppmat.utils import download
from ppmat.utils import logger

if TYPE_CHECKING:
    from pymatgen.core import Structure


def _parse_structures(raw_list):
    results = []
    for item in raw_list:
        try:
            fmt = "cif_str" if isinstance(item, str) else "array"
            results.append(
                BuildStructure.build_one(item, fmt, niggli=False, canocial=False)
            )
        except Exception as exc:
            logger.debug("Failed to parse structure: %s", exc)
            results.append(None)
    return results


def _match_against(
    structures: List[Optional["Structure"]],
    reference_by_comp: dict,
    matcher,
    symmetric: bool = False,
    dynamic: bool = False,
) -> List[bool]:
    from tqdm import tqdm

    results = []
    for struct in tqdm(structures, desc="Matching structures"):
        if struct is None:
            results.append(False)
            continue
        h = str(sorted(struct.atomic_numbers))
        refs = reference_by_comp.get(h)
        if refs is None:
            results.append(True)
        else:
            for ref in refs:
                if matcher.fit(struct, ref, symmetric=symmetric):
                    results.append(False)
                    break
            else:
                results.append(True)
        if dynamic and results[-1]:
            reference_by_comp[h].append(struct)
    return results


def compute_novelty(
    structures: List[Optional["Structure"]],
    reference_structures: List["Structure"],
    stol: float = 0.5,
    angle_tol: float = 10.0,
    ltol: float = 0.3,
    attempt_supercell: bool = False,
    symmetric: bool = False,
) -> List[bool]:
    from pymatgen.analysis.structure_matcher import StructureMatcher

    matcher = StructureMatcher(
        stol=stol,
        angle_tol=angle_tol,
        ltol=ltol,
        attempt_supercell=attempt_supercell,
    )
    reference_by_comp = defaultdict(list)
    for ref in reference_structures:
        reference_by_comp[str(sorted(ref.atomic_numbers))].append(ref)
    return _match_against(structures, reference_by_comp, matcher, symmetric=symmetric)


def compute_stability(
    energy_above_hull: List[Optional[float]],
    threshold: float = 0.0,
) -> List[bool]:
    return [
        (eah is not None and np.isfinite(eah) and eah < threshold)
        for eah in energy_above_hull
    ]


def compute_sun(
    stability: List[bool],
    uniqueness: List[bool],
    novelty: List[bool],
    structures: List[Optional["Structure"]],
    min_elements: int = 2,
) -> Dict[str, float]:
    n = len(structures)
    if any(len(flags) != n for flags in (stability, uniqueness, novelty)):
        raise ValueError("S.U.N. flags must align with every generated structure")
    non_trivial = []
    for s in structures:
        if s is not None and len(set(s.composition)) >= min_elements:
            non_trivial.append(True)
        else:
            non_trivial.append(False)

    sun = [
        s and u and nv and nt
        for s, u, nv, nt in zip(stability, uniqueness, novelty, non_trivial)
    ]

    def rate(flags):
        return round(100 * sum(flags) / max(len(flags), 1), 2)

    return {
        "total": n,
        "valid": sum(1 for s in structures if s is not None),
        "non_trivial": sum(non_trivial),
        "stability_rate": rate(stability),
        "uniqueness_rate": rate(uniqueness),
        "novelty_rate": rate(novelty),
        "sun_rate": rate(sun),
        "sun_count": sum(sun),
    }


class SUNMetric(StreamingMetricBase):
    """S.U.N. (Stability / Uniqueness / Novelty) metric for crystal generation.

    Novelty references the training split (``reference_file_path``). Stability
    requires ``energy_above_hull`` (hull reference not yet migrated); otherwise
    ``stability_rate`` reports 0.0 with a warning. One-shot (``__call__``) and
    streaming evaluation share the same cross-batch uniqueness pool.
    """

    _REFERENCE_CACHE_VERSION = "buildstructure-cif-v1"

    def __init__(
        self,
        gt_file_path: Optional[str] = None,
        reference_file_path: Optional[str] = None,
        stol: float = 0.5,
        angle_tol: float = 10.0,
        ltol: float = 0.3,
        stability_threshold: float = 0.0,
        attempt_supercell: bool = False,
        cache_dir: Optional[str] = None,
    ):
        self.reference_file_path = reference_file_path or gt_file_path
        self.stol = stol
        self.angle_tol = angle_tol
        self.ltol = ltol
        self.stability_threshold = stability_threshold
        self.attempt_supercell = attempt_supercell
        self.cache_dir = cache_dir
        self._reference_structures = None
        self._reference_cache_key = None
        self.reset()

    def reset(self):
        """Clear the per-evaluation uniqueness pool and accumulated flags."""
        self._unique_pool: Dict[str, List] = defaultdict(list)
        self._structures: List[Optional["Structure"]] = []
        self._unique: List[bool] = []
        self._novelty: List[bool] = []
        self._stability: List[bool] = []
        self._energy_above_hull: List[Optional[float]] = []
        self._stability_given = False
        self._warned_no_eah = False

    def update_step(self, *, result: Dict, batch, stage: str):
        """Accumulate one step's generated structures from a dict ``result``."""
        if stage not in ("eval", "sample"):
            return
        generated = (
            result.get("result") or result.get("samples") or result.get("structures")
        )
        if not generated:
            return
        self._ingest(generated, result.get("energy_above_hull"))

    def compute_epoch(self, *, stage: str) -> Dict[str, float]:
        """Finalize S.U.N. rates over everything accumulated so far."""
        if not self._structures:
            return {}
        return self._finalize_dict()

    def __call__(
        self,
        generated: Union[List[str], List[dict]],
        energy_above_hull: Optional[List[Optional[float]]] = None,
    ) -> Dict[str, float]:
        self.reset()
        self._ingest(generated, energy_above_hull)
        return self._finalize_dict()

    def _ingest(
        self,
        generated: Union[List[str], List[dict]],
        energy_above_hull: Optional[List[Optional[float]]],
    ):
        if not isinstance(generated, list) or not generated:
            raise ValueError(
                "generated must be a non-empty list of cif strings or array dicts"
            )
        n = len(generated)
        energies = [None] * n if energy_above_hull is None else list(energy_above_hull)
        if len(energies) != n:
            raise ValueError("energy_above_hull must have one entry per structure")
        stability = compute_stability(energies, threshold=self.stability_threshold)
        if not self._structures:
            self._load_reference()
        structures = _parse_structures(generated)
        novelty = compute_novelty(
            structures,
            self._reference_structures,
            stol=self.stol,
            angle_tol=self.angle_tol,
            ltol=self.ltol,
            attempt_supercell=self.attempt_supercell,
        )
        uniqueness = self._match_unique(structures)
        logger.info(f"Evaluating {n} generated structures")
        self._structures.extend(structures)
        self._unique.extend(uniqueness)
        self._novelty.extend(novelty)
        self._stability.extend(stability)
        self._energy_above_hull.extend(energies)
        self._stability_given |= energy_above_hull is not None

    def _match_unique(self, structures: List[Optional["Structure"]]) -> List[bool]:
        """Incremental uniqueness against the shared (cross-batch) pool."""
        from pymatgen.analysis.structure_matcher import StructureMatcher

        matcher = StructureMatcher(
            stol=self.stol,
            angle_tol=self.angle_tol,
            ltol=self.ltol,
            attempt_supercell=self.attempt_supercell,
        )
        return _match_against(structures, self._unique_pool, matcher, dynamic=True)

    def _finalize_dict(self) -> Dict[str, float]:
        if not self._stability_given and not self._warned_no_eah:
            logger.warning(
                "No energy_above_hull provided; "
                "stability_rate will be 0.0 (prerelaxation + hull not migrated yet)."
            )
            self._warned_no_eah = True

        results = compute_sun(
            self._stability, self._unique, self._novelty, self._structures
        )

        if self._stability_given:
            metastable = compute_stability(self._energy_above_hull, threshold=0.1)
            msun = compute_sun(
                metastable, self._unique, self._novelty, self._structures
            )
            results["metastable_rate"] = msun["stability_rate"]
            results["msun_rate"] = msun["sun_rate"]
            results["msun_count"] = msun["sun_count"]

        logger.info(
            f"S.U.N.: Stability={results['stability_rate']:.2f}%, "
            f"Uniqueness={results['uniqueness_rate']:.2f}%, "
            f"Novelty={results['novelty_rate']:.2f}%, "
            f"S.U.N.={results['sun_rate']:.2f}%"
        )
        return results

    def _load_reference(self):
        """Snapshot and cache the reference content at evaluation start."""
        import pandas as pd
        from pymatgen.core import Structure

        if self.reference_file_path is None:
            raise ValueError("reference_file_path is required for novelty evaluation")
        with open(self.reference_file_path, "rb") as source:
            content = source.read()
        digest = hashlib.sha256()
        for value in (
            self._REFERENCE_CACHE_VERSION,
            version("pymatgen"),
            pd.__version__,
            np.__version__,
        ):
            digest.update(value.encode("utf-8") + b"\0")
        digest.update(content)
        cache_key = digest.hexdigest()
        if cache_key == self._reference_cache_key:
            return

        cache_dir = self.cache_dir or os.path.join(download.DATASETS_HOME, "miad")
        cache_path = os.path.join(cache_dir, f"novelty_reference_{cache_key}.pkl")
        try:
            with open(cache_path, "rb") as cached:
                structures = pickle.load(cached)
            if (
                not isinstance(structures, list)
                or not structures
                or not all(isinstance(struct, Structure) for struct in structures)
            ):
                raise ValueError("Invalid reference cache")
        except FileNotFoundError:
            structures = None
        except (OSError, EOFError, pickle.UnpicklingError, ValueError) as exc:
            logger.warning("Rebuilding invalid reference cache: %s", exc)
            structures = None
        if structures is None:
            frame = pd.read_csv(io.BytesIO(content))
            if "cif" not in frame.columns or frame.empty:
                raise ValueError("Reference CSV must contain a non-empty 'cif' column")
            structures = _parse_structures(frame["cif"].tolist())
            if any(struct is None for struct in structures):
                raise ValueError("Reference CSV contains unparseable structures")
            self._write_reference_cache(cache_dir, cache_path, structures)
        self._reference_structures = structures
        self._reference_cache_key = cache_key

    @staticmethod
    def _write_reference_cache(cache_dir, cache_path, structures):
        temporary_path = None
        try:
            os.makedirs(cache_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            with tempfile.NamedTemporaryFile(
                mode="wb",
                prefix=f"novelty_reference_{timestamp}_",
                suffix=".tmp",
                dir=cache_dir,
                delete=False,
            ) as temporary:
                temporary_path = temporary.name
                pickle.dump(structures, temporary, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(temporary_path, cache_path)
        except OSError as exc:
            logger.warning("Unable to cache reference structures: %s", exc)
        finally:
            if temporary_path is not None and os.path.exists(temporary_path):
                os.unlink(temporary_path)

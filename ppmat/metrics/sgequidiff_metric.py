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
from scipy.spatial.distance import cdist

from ppmat.metrics.streaming_base import StreamingMetricBase
from ppmat.metrics.utils import Crystal
from ppmat.metrics.utils import get_crys_from_cif

# Fixed composition statistics from SGEquiDiff's src/utils/eval_utils.py.
_COMPOSITION_MEANS = np.array(
    [
        21.194441759304013,
        58.20212663122281,
        37.0076848719188,
        36.52738520455582,
        13.350626389725019,
        29.468922184630255,
        28.71735137747704,
        78.8868535524408,
        50.16950217496375,
        59.56764743604155,
        19.020429484306277,
        61.335572740454325,
        47.14515893344343,
        141.75135923307818,
        94.60620029962553,
        85.95794070476977,
        34.07300576173523,
        68.06189371516912,
        637.9862061297893,
        1817.2394155466848,
        1179.2532094169414,
        1127.2743149568837,
        431.51034284549826,
        909.1060025135899,
        3.7744320927984534,
        13.673707104881585,
        9.899275012083132,
        9.620186927095652,
        3.8426065581251856,
        9.96950217496375,
        3.305461575640406,
        5.483035282745288,
        2.1775737071048815,
        4.215114560306594,
        0.8206087101824266,
        3.732092798453359,
        109.16732721121315,
        179.5570323827936,
        70.38970517158047,
        136.0978305229613,
        27.027545809538527,
        119.16713388110198,
        1.2721433060967857,
        2.4614001837260617,
        1.1892568776289631,
        1.9844483610247092,
        0.4691462290494881,
        2.100143582306204,
        1.4829869502174964,
        1.9899951667472209,
        0.5070082165297245,
        1.7956250375970633,
        0.2056251946617602,
        1.745867568873852,
        0.05650072498791687,
        2.3618656355727405,
        2.3053649105848235,
        1.2829636137262992,
        0.9995555685850794,
        1.5150314161430642,
        0.7731271145480909,
        7.4648139197680035,
        6.691686805219913,
        4.010677272036105,
        2.612307566507693,
        3.303528274528758,
        0.2739487675205413,
        5.889753504108265,
        5.615804736587724,
        2.3244356612494683,
        2.1426251769710905,
        1.4464475592073465,
        4.739246012566457,
        14.578395360077332,
        9.839149347510874,
        9.413701584608935,
        3.537059747455868,
        8.550410826486225,
        0.008119864668922184,
        0.43286611889801835,
        0.4247462542290962,
        0.16687837041055423,
        0.17139889490813626,
        0.10898985016916385,
        0.06283228612856452,
        2.6573707104881583,
        2.594538424359594,
        1.219602938224228,
        1.0596390454742999,
        1.1120831319478008,
        0.14842919284678588,
        3.8473658772353794,
        3.6989366843885936,
        1.4541605082183982,
        1.3862277372859781,
        0.8018849685838569,
        0.03542774287095215,
        2.4474625422909617,
        2.4120347994200095,
        0.7745217539010397,
        0.9145812330586208,
        0.3198646689221846,
        1.552730787820203,
        6.910681488641856,
        5.357950700821653,
        3.615163570754227,
        1.9072256165179793,
        2.6702271628806185,
        14.608536589568727,
        34.83222477045747,
        20.223688180890715,
        22.47901710732293,
        7.17674504190757,
        18.641837024143584,
        0.009066988883518605,
        0.9185191396809959,
        0.9094521507974755,
        0.4368550481994018,
        0.38905942883427047,
        0.48375558240695804,
        0.0012985909686158003,
        0.21708593995837092,
        0.21578734898975546,
        0.08167977375391729,
        0.08155386250705281,
        0.06036340747305611,
        116.32010633156113,
        217.5905751570807,
        101.27046882551957,
        162.87154200548844,
        41.920624308665566,
        136.4664572257129,
    ]
)

_COMPOSITION_STDS = np.array(
    [
        16.35781741152948,
        20.189540126474725,
        20.516298414514758,
        16.816765336550194,
        7.966591328222124,
        22.270791076753067,
        21.802116630115243,
        12.804546460581966,
        24.756629388687983,
        13.930306216047477,
        10.214535652334533,
        27.801612936980938,
        39.74031558353379,
        54.269739685575814,
        53.70466607591569,
        42.852342044453444,
        20.78341194242935,
        56.28783510219931,
        563.8004405882157,
        732.0722574247563,
        736.2122907972664,
        606.351603075103,
        272.62646060896407,
        810.6156779688841,
        3.0362262146833428,
        3.2075174256751606,
        4.0633818989245665,
        2.9738244769894764,
        1.7805586029644034,
        5.643243225066782,
        1.1994336274579853,
        0.8939013979423364,
        1.2297581799896975,
        1.0066021334519983,
        0.49129747526397105,
        1.4159553146070951,
        31.754756468836774,
        28.054241463256226,
        38.16336054795611,
        25.83485338379922,
        15.388376641904662,
        39.67137484594156,
        0.31988340032011076,
        0.6833658037760536,
        0.7464197945553585,
        0.4881349085029781,
        0.3176591553643101,
        0.8601748146737138,
        0.5864801661863596,
        0.10048913710210677,
        0.5836289120986499,
        0.2811748167435902,
        0.2468696279341553,
        0.5007375747433073,
        0.37237566669029587,
        1.7235989187720187,
        1.7058836077743305,
        1.1558859351244697,
        0.7677842566598179,
        1.9203550253462733,
        2.1289400248865182,
        3.5326064169848332,
        3.708508303762512,
        2.8709941136664567,
        1.6110681295257014,
        4.310192504023775,
        1.6644182118209292,
        6.228287671164213,
        6.1200848808512305,
        3.1986202996110302,
        2.4492978142248867,
        4.030497343977163,
        3.662028270049814,
        6.8192125550358345,
        6.614243783887738,
        4.334987449618594,
        2.568319610320196,
        5.9494890200106925,
        0.08974370432893491,
        0.4954725441517777,
        0.494304434278516,
        0.2309340434963803,
        0.2072873961103969,
        0.31162647950590266,
        0.39805702757060923,
        1.8111691089355726,
        1.7973395144505941,
        0.9486995373104102,
        0.7538753151875139,
        1.5233177017753785,
        0.7952606701778913,
        3.711190225170556,
        3.638721437232604,
        1.7171165424006831,
        1.4307904413917036,
        2.1047820817622904,
        0.49193748323158065,
        4.064840532426175,
        4.035286619587313,
        1.4858577214526643,
        1.5799117659864677,
        1.6130080156145745,
        1.555249156140194,
        4.776932951077492,
        4.569790780459629,
        2.224617778217326,
        1.7217507416156546,
        2.5969733650703763,
        7.215001918238936,
        19.252513469778584,
        18.775394044177858,
        9.447222764774764,
        6.7467931836261235,
        11.106825644766616,
        0.27206794253092115,
        1.6449321034573106,
        1.6236282792648686,
        0.8506917026741503,
        0.7020945355184042,
        1.2281895279350408,
        0.04134438177238229,
        0.5508855867341717,
        0.5486095551438679,
        0.24239297524046477,
        0.2127779137935831,
        0.3036750942874694,
        80.06063945615361,
        21.345794811194104,
        80.16475677581042,
        52.58533928558554,
        35.40836791039412,
        85.980205895116,
    ]
)


def _scale_composition_fingerprints(values):
    scaled = (np.asarray(values, dtype=float) - _COMPOSITION_MEANS) / _COMPOSITION_STDS
    return np.where(np.isnan(scaled), 0.0, scaled)


def get_matches(structure, alternatives, matcher):
    """Indices of alternative structures matching ``structure``."""
    matches = []
    for idx, alt in enumerate(alternatives):
        if matcher.get_rms_dist(structure, alt) is not None:
            matches.append(idx)
    return matches


def get_unique_structures(structures, matcher):
    """Return structurally unique structures."""
    unique_structures = []
    unique_structure_idxs = []
    for i, structure in enumerate(structures):
        matches = get_matches(structure, structures, matcher)
        first_match = sorted(matches)[0] if matches else i
        if first_match not in unique_structure_idxs:
            unique_structure_idxs.append(first_match)
            unique_structures.append(structures[first_match])
    return unique_structures


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
    for compare, structure in zip(gen_to_compare, structures):
        if not compare:
            novel_structures.append(structure)

    for compare, structure in zip(gen_to_compare, structures):
        if not compare:
            continue
        matches = get_matches(structure, filtered_reference, matcher)
        if len(matches) == 0:
            novel_structures.append(structure)

    return novel_structures


def compute_cov(crys, gt_crys, struc_cutoff, comp_cutoff):
    """Coverage / matching metrics between generated and ground-truth crystals.

    Crystals lacking a valid structural OR composition fingerprint are
    dropped as a pair so the structural and composition fingerprints stay
    row-aligned. Composition fingerprints use the fixed reference statistics
    from the original evaluation implementation.
    """
    valid_crys = [c for c in crys if c.struct_fp is not None and c.comp_fp is not None]
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

    comp_fps = _scale_composition_fingerprints(comp_fps)
    gt_comp_fps = _scale_composition_fingerprints(gt_comp_fps)

    struc_pdist = cdist(np.array(struc_fps), np.array(gt_struc_fps))
    comp_pdist = cdist(np.array(comp_fps), np.array(gt_comp_fps))

    struc_recall_dist = struc_pdist.min(axis=0)
    struc_precision_dist = struc_pdist.min(axis=1)
    comp_recall_dist = comp_pdist.min(axis=0)
    comp_precision_dist = comp_pdist.min(axis=1)

    cov_recall = np.mean(
        np.logical_and(
            struc_recall_dist <= struc_cutoff, comp_recall_dist <= comp_cutoff
        )
    )
    cov_precision = np.sum(
        np.logical_and(
            struc_precision_dist <= struc_cutoff,
            comp_precision_dist <= comp_cutoff,
        )
    ) / len(struc_fps)

    return {
        "cov_recall": cov_recall,
        "cov_precision": cov_precision,
        "amsd_recall": np.mean(struc_recall_dist),
        "amsd_precision": np.mean(struc_precision_dist),
        "amcd_recall": np.mean(comp_recall_dist),
        "amcd_precision": np.mean(comp_precision_dist),
    }


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
            unique = get_unique_structures(valid_structs, self.matcher)
            uniqueness = len(unique) / n_valid

            gt_valid_structs = [c.structure for c in gt_crys if c.valid]
            novel = get_novel_structures(valid_structs, gt_valid_structs, self.matcher)
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

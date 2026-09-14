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

"""OMatG evaluation metrics: match (CSP) and dng (de novo) modes."""

from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple

import numpy as np

from ppmat.metrics.streaming_base import StreamingMetricBase
from ppmat.metrics.utils import Crystal
from ppmat.metrics.utils import get_crys_from_cif
from ppmat.utils import logger

_COV_CUTOFFS = {
    "mp_20": {"struc": 0.4, "comp": 10.0},
    "carbon_24": {"struc": 0.2, "comp": 4.0},
    "perov_5": {"struc": 0.2, "comp": 4.0},
}

# Normalized Magpie composition fingerprint statistics (upstream FlowMM table).
_COMP_FP_MEANS = np.array(
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
_COMP_FP_STDS = np.array(
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


class _StandardScaler:
    """Z-score normalization with NaN replacement, matching the upstream utility."""

    def __init__(
        self, means: np.ndarray, stds: np.ndarray, replace_nan_token: float = 0.0
    ) -> None:
        self._means = means
        self._stds = stds
        self._replace_nan_token = replace_nan_token

    def transform(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        out = (x - self._means) / (self._stds + 1.0e-30)
        out[np.isnan(out)] = self._replace_nan_token
        return out


def _mean_coordination_number(structure) -> float:
    """Mean coordination number via covalent-radii bonds (upstream)."""
    from ase.data import covalent_radii
    from pymatgen.io.ase import AseAtomsAdaptor

    atoms = AseAtomsAdaptor.get_atoms(structure)
    distances = atoms.get_all_distances(mic=True)
    radii = np.array([covalent_radii[number] for number in atoms.numbers])
    radii_sum = radii[:, None] + radii[None, :]
    np.fill_diagonal(radii_sum, np.inf)
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
        np.asarray(crystal_one.atom_types, dtype=np.int64).reshape(-1), minlength=1
    )
    counts_two = np.bincount(
        np.asarray(crystal_two.atom_types, dtype=np.int64).reshape(-1), minlength=1
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


def match_rmsds(
    crystal_list: Sequence[Crystal],
    ref_list: Sequence[Crystal],
    ltol: float = 0.3,
    stol: float = 0.5,
    angle_tol: float = 10.0,
    check_reduced: bool = True,
    enable_progress_bar: bool = False,
) -> Tuple[float, float, float, float, float, float]:
    """Per-index match rate and normalized RMSD (non-matching pairs use stol)."""

    if len(crystal_list) != len(ref_list):
        logger.warning(
            "[OMatGMetric] The number of generated crystals differs from the "
            "number of reference crystals."
        )
    if len(crystal_list) > len(ref_list):
        raise ValueError(
            "The number of generated crystals is greater than the number of "
            "reference crystals."
        )

    results = [
        _get_match_and_rmsd(gen, ref, ltol, stol, angle_tol, check_reduced)
        for gen, ref in zip(crystal_list, ref_list)
    ]
    valid_results = [
        r if gen.valid and ref.valid else None
        for r, gen, ref in zip(results, crystal_list, ref_list)
    ]

    def summarize(res):
        match_count = sum(r is not None for r in res)
        rmsds = [r for r in res if r is not None]
        mean_rmsd = float(np.mean(rmsds)) if rmsds else 0.0
        corr_rmsds = [r if r is not None else stol for r in res]
        corr_rmsd = float(np.mean(corr_rmsds))
        return match_count / len(crystal_list), mean_rmsd, corr_rmsd

    fmr, frmsd, corr_rmsd = summarize(results)
    vmr, vrmsd, vcorr_rmsd = summarize(valid_results)
    return fmr, frmsd, corr_rmsd, vmr, vrmsd, vcorr_rmsd


def metre_rmsds(
    crystal_list: Sequence[Crystal],
    ref_list: Sequence[Crystal],
    ltol: float = 0.3,
    stol: float = 0.5,
    angle_tol: float = 10.0,
    check_reduced: bool = True,
) -> Tuple[float, float, float, float, float, float]:
    """Match-everyone-to-reference rate: best RMSD per reference crystal kept."""

    if len(crystal_list) > len(ref_list):
        raise ValueError(
            "The number of generated crystals is greater than the number of "
            "reference crystals."
        )

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
    num_gen_crystals: Optional[int] = None,
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

    if num_gen_crystals is None:
        num_gen_crystals = len(gen_crystals)

    scaler = _StandardScaler(_COMP_FP_MEANS, _COMP_FP_STDS)
    ref_comp_fps = scaler.transform(ref_comp_fps)
    gen_comp_fps = scaler.transform(np.asarray(filtered_gen_comp_fps))
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
        / num_gen_crystals
    )
    return cov_recall, cov_precision


def _parse_crystals(data: Sequence[dict]) -> List[Crystal]:
    """Parse result dicts into crystals."""
    return [Crystal(item) for item in data]


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
            f"[OMatGMetric] {dropped} of {len(crystals)} {label} structures could not "
            f"be constructed; they are excluded from the distribution metrics and "
            f"counted as invalid."
        )
    return constructible


class OMatGMetric(StreamingMetricBase):
    """OMatG metrics: match (per-index) or dng (validity/METRe/Wasserstein/COV).

    Supports both one-shot ``__call__`` and streaming ``update_step`` /
    ``compute_epoch`` / ``reset``.
    """

    _STREAM_STAGES = ("sample", "eval")

    def __init__(
        self,
        metric_type: str = "match",
        dataset_name: Optional[str] = None,
        gt_file_path: Optional[str] = None,
        ltol: float = 0.3,
        stol: float = 0.5,
        angle_tol: float = 10.0,
        check_reduced: bool = True,
    ) -> None:
        if metric_type not in ("match", "dng"):
            raise ValueError(
                f"metric_type must be 'match' or 'dng', got {metric_type!r}."
            )
        self._metric_type = metric_type
        self._dataset_name = dataset_name
        self._gt_file_path = gt_file_path
        self._ltol = ltol
        self._stol = stol
        self._angle_tol = angle_tol
        self._check_reduced = check_reduced
        self._gt_crystals: Optional[List[Crystal]] = None
        self.reset()

    def _load_reference(self, gt_data: Optional[Sequence]) -> List[Crystal]:
        if gt_data is not None:
            return [
                Crystal(item) if isinstance(item, dict) else get_crys_from_cif(item)
                for item in gt_data
            ]
        if self._gt_crystals is None:
            if self._gt_file_path is None:
                raise ValueError(
                    "gt_data is None, but no gt_file_path was provided to OMatGMetric."
                )
            import pandas as pd
            from p_tqdm import p_map

            csv = pd.read_csv(self._gt_file_path)
            self._gt_crystals = p_map(
                get_crys_from_cif, csv["cif"], desc="Loading ground truth"
            )
        return self._gt_crystals

    def _match_metrics(
        self, gen_crystals: List[Crystal], ref_crystals: List[Crystal]
    ) -> Dict[str, float]:
        fmr, frmsd, corr_rmsd, vmr, vrmsd, vcorr_rmsd = match_rmsds(
            gen_crystals,
            ref_crystals,
            ltol=self._ltol,
            stol=self._stol,
            angle_tol=self._angle_tol,
            check_reduced=self._check_reduced,
        )
        return {
            "match_rate": fmr,
            "mean_rmsd": frmsd,
            "corr_rmsd": corr_rmsd,
            "valid_match_rate": vmr,
            "valid_mean_rmsd": vrmsd,
            "valid_mean_corr_rmsd": vcorr_rmsd,
        }

    def _dng_metrics(
        self, gen_crystals: List[Crystal], ref_crystals: List[Crystal]
    ) -> Dict[str, float]:
        from scipy.stats import wasserstein_distance

        valid_rate = float(sum(c.valid for c in gen_crystals) / len(gen_crystals))

        gen_measurable = _constructible(gen_crystals, "generated")
        ref_measurable = _constructible(ref_crystals, "reference")
        if not gen_measurable or not ref_measurable:
            logger.warning(
                "[OMatGMetric] Distribution metrics need at least one constructible "
                "generated and reference structure; reporting the validity rate only."
            )
            return {"valid_rate": valid_rate}

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
        """One-shot metric computation (sample.py --mode compute_metric)."""
        return self._compute(pred_data, gt_data)

    def reset(self):
        """Clear accumulated predictions; keep the loaded gt reference cache."""
        self._pred_buffer: List[dict] = []

    def update_step(self, *, result: Dict[str, Any], batch: Any, stage: str):
        """Accumulate generated structures from a sampling/eval step."""
        if stage not in self._STREAM_STAGES:
            return
        payload = result.get("result") if isinstance(result, dict) else result
        if payload is None:
            return
        if isinstance(payload, dict):
            payload = [payload]
        for item in payload:
            if isinstance(item, dict):
                self._pred_buffer.append(item)

    def compute_epoch(self, *, stage: str) -> Dict[str, float]:
        """Evaluate accumulated structures against the reference set; {} if none."""
        if stage not in self._STREAM_STAGES:
            return {}
        if not self._pred_buffer:
            return {}
        if self._gt_file_path is None:
            logger.warning(
                "[OMatGMetric] compute_epoch requires gt_file_path; skipping."
            )
            return {}
        return self._compute(self._pred_buffer, None)

    def _compute(
        self, pred_data: Sequence[dict], gt_data: Optional[Sequence]
    ) -> Dict[str, float]:
        if len(pred_data) == 0:
            logger.warning("[OMatGMetric] No generated structures to evaluate.")
            return {}
        gen_crystals = _parse_crystals(pred_data)
        ref_crystals = self._load_reference(gt_data)
        if len(ref_crystals) == 0:
            logger.warning(
                "[OMatGMetric] The reference set is empty; nothing to score."
            )
            return {}
        if self._metric_type == "match":
            return self._match_metrics(gen_crystals, ref_crystals)
        return self._dng_metrics(gen_crystals, ref_crystals)

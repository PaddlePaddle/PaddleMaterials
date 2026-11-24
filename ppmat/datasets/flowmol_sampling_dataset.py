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

"""FlowMol sampling datasets that only sample molecule sizes (atom counts)."""

from __future__ import annotations

import os.path as osp
from typing import Callable
from typing import Optional

import numpy as np
import pandas as pd
from paddle.io import Dataset

from ppmat.datasets.qm9_dataset import QM9Dataset
from ppmat.utils import download
from ppmat.utils import logger

# Atom-count histogram of the GEOM training split used by the released FlowMol3
# model, inlined from the former ``flowmol3_geom_hist`` package (157 pairs,
# 5730535 molecules in total). Counts are normalized at sampling time.
_FLOWMOL3_GEOM_ATOM_COUNTS = {
    3: 1,
    5: 2,
    6: 2,
    7: 2,
    8: 7,
    9: 16,
    10: 48,
    11: 27,
    12: 88,
    13: 206,
    14: 281,
    15: 339,
    16: 415,
    17: 625,
    18: 1113,
    19: 1513,
    20: 2131,
    21: 3150,
    22: 5335,
    23: 7249,
    24: 11532,
    25: 14954,
    26: 21473,
    27: 26176,
    28: 36328,
    29: 46466,
    30: 57773,
    31: 68552,
    32: 84293,
    33: 100392,
    34: 114255,
    35: 135026,
    36: 151115,
    37: 158970,
    38: 178103,
    39: 189910,
    40: 194809,
    41: 209013,
    42: 217942,
    43: 222340,
    44: 226071,
    45: 225086,
    46: 228137,
    47: 224394,
    48: 214241,
    49: 206132,
    50: 201565,
    51: 190578,
    52: 180107,
    53: 172223,
    54: 161016,
    55: 149982,
    56: 135637,
    57: 128518,
    58: 115781,
    59: 99745,
    60: 90912,
    61: 79283,
    62: 70086,
    63: 56682,
    64: 52604,
    65: 42724,
    66: 32963,
    67: 28635,
    68: 23341,
    69: 17373,
    70: 16802,
    71: 13531,
    72: 10628,
    73: 8708,
    74: 8013,
    75: 5968,
    76: 5838,
    77: 3935,
    78: 3714,
    79: 3501,
    80: 3006,
    81: 2445,
    82: 2127,
    83: 1680,
    84: 1394,
    85: 1273,
    86: 1393,
    87: 1186,
    88: 1192,
    89: 637,
    90: 480,
    91: 750,
    92: 709,
    93: 461,
    94: 751,
    95: 600,
    96: 419,
    97: 570,
    98: 150,
    99: 600,
    100: 359,
    101: 180,
    102: 150,
    103: 180,
    104: 176,
    105: 180,
    106: 300,
    107: 570,
    108: 270,
    109: 120,
    110: 240,
    111: 390,
    112: 180,
    113: 210,
    114: 60,
    115: 60,
    116: 180,
    117: 390,
    118: 720,
    119: 360,
    120: 240,
    121: 180,
    122: 150,
    123: 480,
    124: 330,
    125: 180,
    126: 242,
    127: 210,
    128: 60,
    129: 30,
    130: 150,
    131: 30,
    132: 180,
    133: 180,
    134: 330,
    135: 229,
    136: 270,
    138: 300,
    139: 120,
    140: 630,
    141: 120,
    142: 210,
    143: 120,
    144: 120,
    145: 420,
    146: 120,
    147: 120,
    148: 330,
    150: 300,
    151: 60,
    152: 60,
    153: 60,
    155: 90,
    156: 60,
    158: 60,
    159: 30,
    160: 30,
    162: 30,
    169: 30,
    176: 90,
    181: 30,
}


class _FlowMolAtomCountDataset(Dataset):
    """Atom-count sampling shared by the FlowMol sampling datasets.

    Args:
        n_atoms (np.ndarray): Atom counts to sample from.
        n_atom_counts (np.ndarray): Histogram counts aligned with ``n_atoms``.
        total_num (int): Number of sampled molecules. Must be provided.
        seed (int): Random seed. Must be provided.
        transforms (Optional[Callable], optional): Preprocess transforms for each
            sample. Defaults to None.
    """

    def __init__(
        self,
        n_atoms: np.ndarray,
        n_atom_counts: np.ndarray,
        total_num: Optional[int] = None,
        seed: Optional[int] = None,
        transforms: Optional[Callable] = None,
    ):
        super().__init__()

        if total_num is None:
            raise ValueError("total_num must be provided")
        if seed is None:
            raise ValueError("seed must be provided")

        self.total_num = total_num
        self.transforms = transforms
        probs = n_atom_counts / n_atom_counts.sum()
        rng = np.random.RandomState(seed)
        self.sampled_num_atoms = rng.choice(n_atoms, size=total_num, p=probs).astype(
            "int64"
        )

    def __len__(self):
        return self.total_num

    def __getitem__(self, idx):
        """Get item at index idx."""
        data = {
            "structure_array": {
                "num_atoms": int(self.sampled_num_atoms[idx]),
            },
            "id": idx,
        }
        data = self.transforms(data) if self.transforms is not None else data
        return data


class FlowMolSamplingDataset(_FlowMolAtomCountDataset):
    """FlowMol sampling dataset from the QM9 atom-count histogram.

    Sampling ignores training molecules and draws only atom counts, from which
    the model generates new molecules of those sizes. The histogram is built
    dynamically from the ``num_atoms`` column of the original QM9 split CSV
    (the same source used by ``QM9Dataset``); no pre-baked archive is required.

    **Dataset Overview**
    - **Source**: ``num_atoms`` column of the QM9 train split CSV.

    Args:
        path (str, optional): Path to a QM9 split CSV. If the file does not
            exist, the QM9 archive is downloaded automatically. Defaults to
            "./data/qm9/train.csv".
        total_num (int): Number of sampled molecules. Must be provided.
        seed (int): Random seed. Must be provided.
        transforms (Optional[Callable], optional): Preprocess transforms for each
            sample. Defaults to None.
    """

    name = QM9Dataset.name
    url = QM9Dataset.url
    md5 = QM9Dataset.md5

    def __init__(
        self,
        path: str = "./data/qm9/train.csv",
        total_num: Optional[int] = None,
        seed: Optional[int] = None,
        transforms: Optional[Callable] = None,
    ):
        if not osp.exists(path):
            logger.message("The dataset is not found. Will download it now.")
            root_path = download.get_datasets_path_from_url(self.url, self.md5)
            path = osp.join(root_path, self.name, osp.basename(path))
        self.path = path

        num_atoms = pd.read_csv(path, usecols=["num_atoms"])["num_atoms"].to_numpy()
        n_atoms, n_atom_counts = np.unique(num_atoms, return_counts=True)
        n_atoms = n_atoms.astype("int64")
        n_atom_counts = n_atom_counts.astype("int64")

        logger.info(f"Load {n_atoms.shape[0]} atom-count bins from {path}")
        super().__init__(n_atoms, n_atom_counts, total_num, seed, transforms)


class FlowMol3GeomSamplingDataset(_FlowMolAtomCountDataset):
    """FlowMol3 sampling dataset from the inlined GEOM atom-count histogram.

    The released FlowMol3 model is trained on GEOM-Drugs; its atom-count
    histogram is inlined, so this dataset needs no download.

    **Dataset Overview**
    - **Source**: GEOM training split used by the released FlowMol3 model.
    - **Histogram**: 157 atom counts covering 5,730,535 molecules.

    Args:
        total_num (int): Number of sampled molecules. Must be provided.
        seed (int): Random seed. Must be provided.
        transforms (Optional[Callable], optional): Preprocess transforms for each
            sample. Defaults to None.
    """

    def __init__(
        self,
        total_num: Optional[int] = None,
        seed: Optional[int] = None,
        transforms: Optional[Callable] = None,
    ):
        n_atoms = np.array(sorted(_FLOWMOL3_GEOM_ATOM_COUNTS), dtype="int64")
        n_atom_counts = np.array(
            [_FLOWMOL3_GEOM_ATOM_COUNTS[n] for n in n_atoms], dtype="int64"
        )
        super().__init__(n_atoms, n_atom_counts, total_num, seed, transforms)

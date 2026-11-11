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

"""FlowMol molecule datasets backed by preprocessed npz splits.

Every sample provides the endpoint features (COM-free positions, one-hot atom
types, one-hot charges and upper-triangle one-hot bond types) plus the aligned
position prior consumed by the flow matching training path.

The property-prediction QM9Dataset (``ppmat/datasets/qm9_dataset.py``) is not
reused: generation needs process_qm9.py tensors, not property labels.
"""

from __future__ import annotations

import os.path as osp

import numpy as np
from paddle.io import Dataset
from scipy.optimize import linear_sum_assignment

from ppmat.datasets.custom_data_type import ConcatData
from ppmat.utils import download
from ppmat.utils import logger

DATASET_NAME = "flowmol_qm9"
DATASET_URL = "https://paddle-org.bj.bcebos.com/paddlematerials/datasets/qm9/flowmol_qm9.zip"
DATASET_MD5 = "91a65706ed27c5cd1ebf92faa9dab201"

N_CHARGES = 6
N_BOND_TYPES = 4
CHARGE_OFFSET = 2

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


def _one_hot(idx: np.ndarray, num_classes: int) -> np.ndarray:
    one_hot = np.zeros((idx.shape[0], num_classes), dtype=np.float32)
    one_hot[np.arange(idx.shape[0]), idx] = 1.0
    return one_hot


def _align_prior(prior: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Align a position prior to the endpoint (Kabsch, after permutation)."""
    cost = np.linalg.norm(dst[:, None, :] - prior[None, :, :], axis=-1)
    _, prior_idx = linear_sum_assignment(cost)
    prior = prior[prior_idx]

    prior_mean = prior.mean(axis=0, keepdims=True)
    dst_mean = dst.mean(axis=0, keepdims=True)
    prior_c = prior - prior_mean
    dst_c = dst - dst_mean

    h = prior_c.T @ dst_c
    u, _, vh = np.linalg.svd(h)
    r = vh.T @ u.T
    aligned = prior_c @ r.T
    return (aligned + dst_mean).astype("float32")


def _resolve_split_path(path: str, name: str, url: str, md5: str) -> str:
    """Resolve a split path, downloading the package when it is missing.

    The download source comes from the caller class attributes so that every
    dataset subclass resolves against its own package (``name/url/md5``).
    """
    if osp.exists(path):
        return path
    logger.message("The dataset is not found. Will download it now.")
    root_path = download.get_datasets_path_from_url(url, md5)
    return osp.join(root_path, name, osp.basename(path))


class FlowMolDataset(Dataset):
    """FlowMol training dataset over a preprocessed npz split."""

    name = DATASET_NAME
    url = DATASET_URL
    md5 = DATASET_MD5

    def __init__(
        self,
        path: str = "./data/flowmol_qm9/train_data_processed.npz",
        atom_map: list = None,
        fake_atom_p: float = 0.0,
        fake_atom_std: float = 1.0,
        prior_std: float = 1.0,
        prior_align: bool = True,
        seed: int = None,
    ):
        super().__init__()

        if atom_map is None:
            raise ValueError("atom_map must be provided")

        self.atom_map = list(atom_map)
        self.n_atom_types = len(self.atom_map)
        self.fake_atom_p = fake_atom_p
        self.fake_atom_std = fake_atom_std
        self.use_fake_atoms = fake_atom_p > 0
        self.prior_std = prior_std
        self.prior_align = prior_align
        self.rng = np.random.RandomState(seed)

        path = _resolve_split_path(path, self.name, self.url, self.md5)
        self.path = path

        with np.load(path) as data:
            self.positions = data["positions"]
            self.atom_types = data["atom_types"]
            self.atom_charges = data["atom_charges"]
            self.bond_types = data["bond_types"]
            self.bond_idxs = data["bond_idxs"]
            self.node_idx_array = data["node_idx_array"]
            self.edge_idx_array = data["edge_idx_array"]

    def __len__(self):
        return self.node_idx_array.shape[0]

    def __getitem__(self, idx):
        node_start, node_end = self.node_idx_array[idx]
        edge_start, edge_end = self.edge_idx_array[idx]

        positions = self.positions[node_start:node_end].astype("float32")
        atom_types_idx = self.atom_types[node_start:node_end].argmax(-1)
        atom_charges_idx = self.atom_charges[node_start:node_end].astype("int64")

        if self.use_fake_atoms:
            n_real_atoms = positions.shape[0]
            max_num_fake_atoms = int(np.ceil(n_real_atoms * self.fake_atom_p))
            num_fake_atoms = int(self.rng.randint(0, max_num_fake_atoms + 1))
            anchor_atom_idxs = self.rng.randint(0, n_real_atoms, num_fake_atoms)
            fake_positions = positions[anchor_atom_idxs]
            fake_positions = fake_positions + (
                self.rng.randn(*fake_positions.shape) * self.fake_atom_std
            ).astype("float32")

            positions = np.concatenate((positions, fake_positions), axis=0)
            atom_types_idx = np.concatenate(
                (
                    atom_types_idx,
                    np.full(num_fake_atoms, self.n_atom_types, dtype="int64"),
                ),
                axis=0,
            )
            atom_charges_idx = np.concatenate(
                (
                    atom_charges_idx,
                    np.zeros(num_fake_atoms, dtype="int64"),
                ),
                axis=0,
            )

        positions = positions - positions.mean(axis=0, keepdims=True)

        n_atoms = positions.shape[0]
        atom_types = _one_hot(
            atom_types_idx, self.n_atom_types + int(self.use_fake_atoms)
        )
        atom_charges = _one_hot(atom_charges_idx + CHARGE_OFFSET, N_CHARGES)

        adjacency = np.zeros((n_atoms, n_atoms), dtype=np.int64)
        bond_idxs = self.bond_idxs[edge_start:edge_end]
        adjacency[bond_idxs[:, 0], bond_idxs[:, 1]] = self.bond_types[
            edge_start:edge_end
        ]
        upper_i, upper_j = np.triu_indices(n_atoms, k=1)
        bond_types = _one_hot(adjacency[upper_i, upper_j], N_BOND_TYPES)

        prior_positions = self.rng.randn(n_atoms, 3).astype("float32") * self.prior_std
        prior_positions = prior_positions - prior_positions.mean(axis=0, keepdims=True)
        if self.prior_align:
            prior_positions = _align_prior(prior_positions, positions)

        structure_array = {
            "num_atoms": ConcatData(np.array([n_atoms], dtype="int64")),
            "positions": ConcatData(positions),
            "atom_types": ConcatData(atom_types),
            "atom_charges": ConcatData(atom_charges),
            # upper and lower triangle edges of a pair share the bond one-hot
            "bond_types": ConcatData(np.concatenate((bond_types, bond_types), axis=0)),
        }
        return {
            "structure_array": structure_array,
            "prior_positions": ConcatData(prior_positions),
        }


class _FlowMolAtomCountDataset(Dataset):
    """Atom-count sampling shared by the FlowMol sampling datasets."""

    def __init__(
        self,
        n_atoms: np.ndarray,
        n_atom_counts: np.ndarray,
        total_num: int = None,
        seed: int = None,
    ):
        super().__init__()

        if total_num is None:
            raise ValueError("total_num must be provided")
        if seed is None:
            raise ValueError("seed must be provided")

        self.total_num = total_num
        probs = n_atom_counts / n_atom_counts.sum()
        rng = np.random.RandomState(seed)
        self.sampled_num_atoms = rng.choice(n_atoms, size=total_num, p=probs).astype(
            "int64"
        )

    def __len__(self):
        return self.total_num

    def __getitem__(self, idx):
        return {
            "structure_array": {
                "num_atoms": int(self.sampled_num_atoms[idx]),
            }
        }


class FlowMolSamplingDataset(_FlowMolAtomCountDataset):
    """Sampling inputs for FlowMol: atom counts drawn from the train histogram.

    ``path`` points at the atom-count histogram npz of the training split;
    the histogram package is downloaded through the shared pipeline when the
    file is missing.
    """

    name = DATASET_NAME
    url = DATASET_URL
    md5 = DATASET_MD5

    def __init__(
        self,
        path: str = "./data/flowmol_qm9/train_data_n_atoms_histogram.npz",
        total_num: int = None,
        seed: int = None,
    ):
        path = _resolve_split_path(path, self.name, self.url, self.md5)
        self.path = path

        with np.load(path) as hist:
            n_atoms = hist["n_atoms"]
            n_atom_counts = hist["n_atom_counts"]

        super().__init__(n_atoms, n_atom_counts, total_num, seed)


class FlowMol3GeomSamplingDataset(_FlowMolAtomCountDataset):
    """Sampling inputs of the released FlowMol3 model (GEOM atom histogram).

    The histogram is inlined from the released package, so this dataset needs
    no download.
    """

    def __init__(self, total_num: int = None, seed: int = None):
        n_atoms = np.array(sorted(_FLOWMOL3_GEOM_ATOM_COUNTS), dtype="int64")
        n_atom_counts = np.array(
            [_FLOWMOL3_GEOM_ATOM_COUNTS[n] for n in n_atoms], dtype="int64"
        )
        super().__init__(n_atoms, n_atom_counts, total_num, seed)

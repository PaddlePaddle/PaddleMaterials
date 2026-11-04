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

import os.path as osp
from pathlib import Path

import numpy as np
import paddle
import pandas as pd
from ase import data as ase_data
from paddle.io import Dataset

from ppmat.datasets.geometric_data_type.data import Data
from ppmat.models.liflow.geometry import get_neighbor_list_batch
from ppmat.models.liflow.prior import AdaptiveMaxwellBoltzmannPrior
from ppmat.models.liflow.prior import Prior
from ppmat.utils import download
from ppmat.utils import logger

__all__ = ["LiFlowDataset", "LiFlowTrajectoryDataset", "VALIDATION_TIMES"]

_INDEX_COLUMNS = {
    "name",
    "temp",
    "t_start",
    "t_end",
    "prior_Li",
    "prior_frame",
    "comp",
}

# Fixed flow-time grid used by the reference ``validation_step``
# (``torch.linspace(0, 1, 11)``).
VALIDATION_TIMES = np.linspace(0.0, 1.0, 11, dtype=np.float32)


class LiFlowTrajectoryDataset(Dataset):
    """Shared file layout, caching and sampling policy for LiFlow datasets.

    Both LiFlow training datasets read the same trajectory archive:

    ``{split}.csv``
        One row per ``(name, temp, t_start, t_end, prior_Li, prior_frame,
        comp)``; ``name`` indexes ``atomic_numbers.npy`` / ``lattice.npy`` and
        the trajectory arrays.
    ``element_index.npy``
        Atomic number -> element embedding index table used by ``DualPaiNN``.
    ``atomic_numbers.npy`` / ``lattice.npy``
        Pickled ``dict`` keyed by ``name``.
    ``positions_{T}K.npz``
        ``dict`` of ``name -> float32 [n_frames, n_atoms, 3]`` trajectories per
        temperature ``T``.

    Only the raw arrays are cached (optionally in memory).  Time windows,
    flow times, priors and noise are sampled on **every** ``__getitem__`` for
    the train split; freezing them in a per-sample cache would turn the
    stochastic flow-matching objective into a fixed one and change the trained
    model.  The validation split reuses one deterministic draw per row for
    ``num_validation_times`` evenly spaced flow times, following the reference
    ``FlowModule.validation_step``.

    Subclasses implement :meth:`_build_pair` and provide a default prior.

    Args:
        path: Directory with the files listed above.  When it does not exist
            and ``url`` is given, the archive is downloaded with
            :func:`ppmat.utils.download`.
        split: One of ``"train"``, ``"val"``, ``"test"``.
        time_delay_steps: Frame gap used by the propagator window and by the
            corrector's ``skip_steps``.
        cutoff: Neighbour-list cutoff in Angstrom.
        periodic: Whether to use periodic boundary conditions.
        seed: Seed for the deterministic validation draw.
        in_memory: Load every ``positions_{T}K.npz`` into memory.
        num_validation_times: Flow times evaluated per validation sample
            (reference default: 11).
        prior: Prior override; subclass dependent.
        url: Dataset archive URL.  ``md5`` is required together with it.  No
            default URL is provided: the LiFlow training archive is not hosted
            by PaddleMaterials, so a missing ``path`` is an error rather than a
            silent download attempt.
        md5: MD5 checksum of the archive.
    """

    def __init__(
        self,
        path: str,
        split: str = "train",
        *,
        time_delay_steps: int = 100,
        cutoff: float = 5.0,
        periodic: bool = True,
        seed: int = 42,
        in_memory: bool = False,
        num_validation_times: int = 11,
        prior: Prior | None = None,
        url: str | None = None,
        md5: str | None = None,
    ):
        super().__init__()
        if split not in {"train", "val", "test"}:
            raise ValueError(f"split must be 'train', 'val' or 'test', got {split!r}.")
        if time_delay_steps <= 0:
            raise ValueError("time_delay_steps must be positive.")
        if num_validation_times <= 0:
            raise ValueError("num_validation_times must be positive.")

        self.split = split
        self.time_delay_steps = int(time_delay_steps)
        self.cutoff = float(cutoff)
        self.periodic = bool(periodic)
        self.seed = int(seed)
        self.in_memory = bool(in_memory)
        self.num_validation_times = int(num_validation_times)
        self.validation_times = np.linspace(
            0.0, 1.0, self.num_validation_times, dtype=np.float32
        )
        self.atomic_masses = ase_data.atomic_masses
        self.prior = prior if prior is not None else self._default_prior()

        self.path = self._resolve_path(path, url, md5)
        self._load_index()
        self._load_static_data()
        self._load_position_files()

    def _default_prior(self) -> Prior:
        raise NotImplementedError

    def _build_pair(
        self,
        *,
        name: str,
        row,
        trajectory,
        atomic_numbers: np.ndarray,
        masses: np.ndarray,
        temperature: float,
        rng: np.random.Generator,
    ) -> dict:
        """Return ``start``/``end``/``prior``/``neighbor_positions``."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # data loading / serialization
    # ------------------------------------------------------------------
    @staticmethod
    def _resolve_path(path: str, url: str | None, md5: str | None) -> Path:
        if osp.exists(path):
            return Path(path)
        if url is None:
            raise FileNotFoundError(
                f"LiFlow data directory {path!r} does not exist and no download "
                "URL was provided.  Download the trajectory archive from the "
                "original LiFlow release and point `path` at the extracted "
                "directory, or pass `url` and `md5` explicitly."
            )
        if md5 is None:
            raise ValueError("`md5` is required when `url` is provided.")
        logger.message(f"LiFlow data not found at {path!r}; downloading from {url}.")
        root = download.get_datasets_path_from_url(url, md5)
        return Path(root)

    def _load_index(self) -> None:
        csv_path = self.path / f"{self.split}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing split index: {csv_path}")
        df = pd.read_csv(csv_path)
        missing = _INDEX_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(f"{csv_path} is missing index columns: {sorted(missing)}")
        self._df = df.reset_index(drop=True)
        self.sample_ids = [str(name) for name in self._df["name"]]

    def _load_static_data(self) -> None:
        self.element_index = np.load(self.path / "element_index.npy")
        with open(self.path / "atomic_numbers.npy", "rb") as handle:
            self._atomic_numbers = np.load(handle, allow_pickle=True).item()
        with open(self.path / "lattice.npy", "rb") as handle:
            self._lattice = np.load(handle, allow_pickle=True).item()

    def _load_position_files(self) -> None:
        self._position_files: dict[float, Path] = {}
        self._positions: dict[float, object] = {}
        for temp in sorted({float(t) for t in self._df["temp"]}):
            position_path = self.path / f"positions_{int(temp)}K.npz"
            if not position_path.exists():
                raise FileNotFoundError(f"Missing trajectory file: {position_path}")
            self._position_files[temp] = position_path
            if self.in_memory:
                self._positions[temp] = dict(np.load(position_path))

    def _get_positions(self, temp: float):
        temp = float(temp)
        if temp not in self._positions:
            # ``np.load`` on ``.npz`` returns a lazy ``NpzFile``; it is opened
            # once per process and reused across samples.
            self._positions[temp] = np.load(self._position_files[temp])
        return self._positions[temp]

    def __getstate__(self) -> dict:
        # ``NpzFile`` handles cannot be pickled into DataLoader workers.  Drop
        # them and reopen lazily in the worker process.
        state = self.__dict__.copy()
        if not self.in_memory:
            state["_positions"] = {}
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)

    # ------------------------------------------------------------------
    # dataset protocol
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        if self.split == "val":
            return len(self._df) * self.num_validation_times
        return len(self._df)

    def __getitem__(self, idx: int) -> Data:
        idx = int(idx)
        if idx < 0:
            idx += len(self)

        if self.split == "val":
            row_idx = idx // self.num_validation_times
            time_idx = idx % self.num_validation_times
            # Deterministic per-row RNG: all flow times of one row share the
            # same (start, end, prior) pair, as in the reference validation.
            rng = np.random.default_rng(self.seed + row_idx)
            flow_time = float(self.validation_times[time_idx])
        else:
            row_idx = idx
            # Draw from the global NumPy RNG so that
            # ``ppmat.datasets.worker_init_fn`` gives every DataLoader worker a
            # different stream (a per-dataset ``default_rng`` would be cloned
            # identically into all workers).
            rng = np.random.default_rng(np.random.randint(0, 2**31 - 1))
            flow_time = float(rng.uniform(0.0, 1.0))

        row = self._df.iloc[row_idx]
        name = str(row["name"])
        temp = float(row["temp"])
        atomic_numbers = np.asarray(self._atomic_numbers[name], dtype=np.int64)
        elements = np.asarray(self.element_index[atomic_numbers], dtype=np.int64)
        masses = self.atomic_masses[atomic_numbers]
        trajectory = self._get_positions(temp)[name]
        lattice = np.asarray(self._lattice[name], dtype=np.float64)

        pair = self._build_pair(
            name=name,
            row=row,
            trajectory=trajectory,
            atomic_numbers=atomic_numbers,
            masses=masses,
            temperature=temp,
            rng=rng,
        )

        start_positions = pair["start_positions"]
        end_positions = pair["end_positions"]
        prior = pair["prior"]
        neighbor_positions = pair["neighbor_positions"]

        edge_index, shifts = get_neighbor_list_batch(
            positions_batch=neighbor_positions,
            lattice=lattice,
            cutoff=self.cutoff,
            periodic=self.periodic,
        )
        velocity = (end_positions - (start_positions + prior)).astype(np.float32)

        return Data(
            num_nodes=int(elements.shape[0]),
            start_positions=paddle.to_tensor(start_positions, dtype="float32"),
            end_positions=paddle.to_tensor(end_positions, dtype="float32"),
            prior=paddle.to_tensor(prior, dtype="float32"),
            velocity=paddle.to_tensor(velocity, dtype="float32"),
            elements=paddle.to_tensor(elements, dtype="int64"),
            atomic_numbers=paddle.to_tensor(atomic_numbers, dtype="int64"),
            edge_index=paddle.to_tensor(edge_index, dtype="int64"),
            shifts=paddle.to_tensor(shifts, dtype="float32"),
            temperature=paddle.to_tensor(temp, dtype="float32"),
            flow_time=paddle.to_tensor(flow_time, dtype="float32"),
            name=name,
        )


class LiFlowDataset(LiFlowTrajectoryDataset):
    """Propagator dataset (reference ``TimeDelayedPairDataset``).

    ``start``/``end`` are two frames separated by ``time_delay_steps`` inside
    ``[t_start, t_end)`` and the prior is an
    :class:`AdaptiveMaxwellBoltzmannPrior` driven by the ``prior_Li`` /
    ``prior_frame`` columns of the split index.  The flow-matching target is::

        x_0 = start + prior
        x_t = (1 - t) * x_0 + t * end
        velocity = end - x_0

    Args:
        path: Dataset directory (see :class:`LiFlowTrajectoryDataset`).
        split: ``"train"``, ``"val"`` or ``"test"``.
        time_delay_steps: Frame gap between ``start`` and ``end``.
        cutoff: Neighbour-list cutoff in Angstrom.
        periodic: Whether to use periodic boundary conditions.
        neighbor_list_both_ends: If True the neighbour list is the union of the
            start and end frames (reference default: False).
        seed: Seed for the deterministic validation draw.
        in_memory: Cache the raw trajectory arrays in memory.
        num_validation_times: Flow times per validation sample.
        prior: Optional prior override (default ``AdaptiveMaxwellBoltzmannPrior``).
        url: Optional archive URL (``md5`` required).  No default URL.
        md5: Archive checksum.
    """

    def __init__(
        self,
        path: str,
        split: str = "train",
        *,
        time_delay_steps: int = 100,
        cutoff: float = 5.0,
        periodic: bool = True,
        neighbor_list_both_ends: bool = False,
        seed: int = 42,
        in_memory: bool = False,
        num_validation_times: int = 11,
        prior: Prior | None = None,
        url: str | None = None,
        md5: str | None = None,
    ):
        self.neighbor_list_both_ends = bool(neighbor_list_both_ends)
        super().__init__(
            path=path,
            split=split,
            time_delay_steps=time_delay_steps,
            cutoff=cutoff,
            periodic=periodic,
            seed=seed,
            in_memory=in_memory,
            num_validation_times=num_validation_times,
            prior=prior,
            url=url,
            md5=md5,
        )

    def _default_prior(self) -> Prior:
        return AdaptiveMaxwellBoltzmannPrior()

    def _build_pair(
        self,
        *,
        name: str,
        row,
        trajectory,
        atomic_numbers: np.ndarray,
        masses: np.ndarray,
        temperature: float,
        rng: np.random.Generator,
    ) -> dict:
        t_start = int(row["t_start"])
        t_end = int(row["t_end"])
        if t_end - t_start <= self.time_delay_steps:
            raise ValueError(
                f"{name}: trajectory window [{t_start}, {t_end}) is shorter "
                f"than time_delay_steps={self.time_delay_steps}."
            )
        start_time = int(rng.integers(t_start, t_end - self.time_delay_steps))
        end_time = start_time + self.time_delay_steps
        start_positions = np.asarray(trajectory[start_time], dtype=np.float32)
        end_positions = np.asarray(trajectory[end_time], dtype=np.float32)
        neighbor_positions = (
            [start_positions, end_positions]
            if self.neighbor_list_both_ends
            else [start_positions]
        )
        prior = self.prior.sample(
            temperature=temperature,
            atomic_numbers=atomic_numbers,
            masses=masses,
            scale_Li_index=int(row["prior_Li"]),
            scale_frame_index=int(row["prior_frame"]),
            shape=start_positions.shape,
            rng=rng,
        )
        return {
            "start_positions": start_positions,
            "end_positions": end_positions,
            "prior": np.asarray(prior, dtype=np.float32),
            "neighbor_positions": neighbor_positions,
        }

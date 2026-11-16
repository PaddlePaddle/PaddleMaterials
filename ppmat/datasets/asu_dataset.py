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

"""Asymmetric Unit (ASU) dataset. """

import hashlib
import json
import os.path as osp
import pickle
from pathlib import Path
from typing import Dict
from typing import Optional

import numpy as np
import paddle.distributed as dist
from paddle.io import Dataset

from ppmat.datasets.build_asu import ELEMENT_ENCODING_SIZE
from ppmat.datasets.build_asu import BuildAsuCrystal
from ppmat.datasets.build_asu import save_asu_cache
from ppmat.datasets.custom_data_type import ConcatData
from ppmat.utils import download
from ppmat.utils import logger


class AsymmetricUnitDataset(Dataset):
    """Base ASU-representation dataset.

    Concrete datasets (e.g. ``MP20ASUDataset`` and ``MPTS52ASUDataset``)
    declare the class-level ``name`` / ``url`` / ``md5`` download
    information and their own ``path`` default; the base class itself holds
    no download source.

    Crystals are stored as flat packed arrays in NPZ archives
    (``<split>.npz``). The packed arrays are parsed into per-crystal field
    dicts by :class:`~ppmat.datasets.build_asu.BuildAsuCrystal` once and
    serialized to cache files, so building the dataset (and its DataLoader)
    stays cheap regardless of dataset size.

    **Data Format**

    Each crystal is a 1-D float array in the packed layout (NE = element
    encoding size):

    - ``[0]``: num_atoms (n)
    - ``[1]``: space group number (1-indexed)
    - ``[2:2+NE]``: composition (one-hot over NE elements)
    - ``[2+NE:5+NE]``: conventional lattice lengths (a, b, c)
    - ``[5+NE:8+NE]``: conventional lattice angles (alpha, beta, gamma)
    - ``[8+NE:8+NE+n]``: element indices
    - ``[8+NE+n:8+NE+2n]``: wyckoff indices
    - ``[8+NE+2n:8+NE+5n]``: fractional coords (n*3)
    - ``[8+NE+5n:8+NE+6n]``: wyckoff shape indices (optional)

    Args:
        path (str): The path of the dataset npz file, defaulted by the
            concrete subclass. If the path does not exist, the dataset is
            downloaded via the class-level ``url`` and ``md5``.
        build_crystal_cfg (Dict, optional): The configs for building the
            per-crystal field dicts from packed arrays. Defaults to None.
        cache_path (Optional[str], optional): Directory for derived crystal
            caches. Defaults to the unified SGEquiDiff dataset cache.
        overwrite (bool, optional): Overwrite the existing cache file at the
            given path if it already exists. Defaults to False.
    """

    name = None
    url = None
    md5 = None

    def __init__(
        self,
        path: str,
        build_crystal_cfg: Dict = None,
        cache_path: Optional[str] = None,
        overwrite: bool = False,
    ):
        super().__init__()

        if path is None:
            raise ValueError("path must be a str, got None.")

        if not osp.exists(path):
            if self.url is None:
                raise ValueError(
                    f"Dataset path {path} does not exist, and "
                    f"{type(self).__name__} does not define url/md5 for "
                    "download. Use a concrete dataset subclass or pass an "
                    "existing path."
                )
            logger.message("The dataset is not found. Will download it now.")
            root_path = download.get_datasets_path_from_url(self.url, self.md5)
            path = osp.join(root_path, self.name, osp.basename(path))

        self.path = path

        build_crystal_cfg = {
            "element_encoding_size": ELEMENT_ENCODING_SIZE,
            **(build_crystal_cfg or {}),
        }
        self.build_crystal_cfg = build_crystal_cfg
        self.overwrite = overwrite
        flat_crystals, self.num_samples = self.read_data(path)
        logger.info(f"Load {self.num_samples} samples from {path}")
        source_digest = hashlib.sha256()
        for crystal in flat_crystals:
            source_digest.update(str((crystal.shape, crystal.dtype.str)).encode())
            source_digest.update(crystal.tobytes())
        identity = {
            "source": source_digest.hexdigest(),
            "preprocessing": BuildAsuCrystal.cache_version,
            "build": build_crystal_cfg,
            "numpy": np.__version__,
        }
        digest = hashlib.sha256(
            json.dumps(identity, sort_keys=True).encode()
        ).hexdigest()
        cache_root = cache_path or osp.join(download.DATASETS_HOME, "sgequidiff", "asu")
        self.cache_path = osp.join(cache_root, f"{Path(path).stem}-{digest}.pkl")
        builder = BuildAsuCrystal(**build_crystal_cfg)
        distributed = dist.is_initialized()
        status = [None]
        if not distributed or dist.get_rank() == 0:
            try:
                self._prepare_cache(builder, flat_crystals, identity, overwrite)
            except Exception as error:
                if not distributed:
                    raise
                status[0] = f"{type(error).__name__}: {error}"
        if distributed:
            dist.broadcast_object_list(status, src=0)
        if status[0] is not None:
            raise RuntimeError(
                f"ASU cache construction failed on rank zero: {status[0]}"
            )
        self.crystals = self.load_from_cache(self.cache_path)["crystals"]

    def _prepare_cache(self, builder, flat_crystals, identity, overwrite):
        if not overwrite and osp.isfile(self.cache_path):
            try:
                payload = self.load_from_cache(self.cache_path)
                if (
                    payload["identity"] == identity
                    and len(payload["crystals"]) == self.num_samples
                ):
                    return
            except (EOFError, KeyError, OSError, pickle.UnpicklingError, TypeError):
                logger.warning(f"Rebuilding invalid ASU cache: {self.cache_path}")
        self.save_to_cache(
            self.cache_path,
            {"identity": identity, "crystals": builder(flat_crystals)},
        )

    def read_data(self, path: str):
        """Read the packed NPZ archive and split it into per-crystal arrays."""
        with np.load(path) as npz:
            flat_crystals = np.split(npz["packed"], npz["indices"])
        return flat_crystals, len(flat_crystals)

    def save_to_cache(self, cache_path: str, data):
        save_asu_cache(cache_path, data)

    def load_from_cache(self, cache_path: str):
        with open(cache_path, "rb") as file_obj:
            return pickle.load(file_obj)

    def __getitem__(self, index: int) -> dict:
        crystal = {
            key: value.copy() if isinstance(value, np.ndarray) else value
            for key, value in self.crystals[index].items()
        }
        # Variable-length fields are wrapped for the default collator.
        for key in (
            "element_indices",
            "wyckoff_indices",
            "wyckoff_shape_indices",
            "frac_coords",
        ):
            crystal[key] = ConcatData(crystal[key])
        return crystal

    def __len__(self) -> int:
        return self.num_samples


class MP20ASUDataset(AsymmetricUnitDataset):
    """ASU-representation dataset (MP-20)."""

    name = "mp_20"
    url = "https://paddle-org.bj.bcebos.com/paddlematerials/datasets/asu/mp_20_asu.zip"
    md5 = "c8dc162555808bf8dc0183b840209f6a"

    def __init__(self, path: str = "./data/mp_20/train.npz", **kwargs):
        super().__init__(path=path, **kwargs)


class MPTS52ASUDataset(AsymmetricUnitDataset):
    """ASU-representation dataset (MPTS-52)."""

    name = "mpts_52"
    url = "https://paddle-org.bj.bcebos.com/paddlematerials/datasets/asu/mpts_52_asu.zip"  # noqa
    md5 = "bdbfdad0352bbf32afb1ee6561cea97b"

    def __init__(self, path: str = "./data/mpts_52/train.npz", **kwargs):
        super().__init__(path=path, **kwargs)

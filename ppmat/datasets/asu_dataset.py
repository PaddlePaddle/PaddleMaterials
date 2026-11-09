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

import os
import os.path as osp
import pickle
from typing import Dict
from typing import Optional

import numpy as np
import paddle.distributed as dist
from paddle.io import Dataset

from ppmat.datasets.build_asu import BuildAsuCrystal
from ppmat.datasets.custom_data_type import ConcatData
from ppmat.models.sgequidiff.sgequidiff_meta import ELEMENT_ENCODING_SIZE
from ppmat.utils import download
from ppmat.utils import logger
from ppmat.utils.misc import is_equal


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
        cache_path (Optional[str], optional): If a cache_path is set, the
            built crystals will be read directly from this path; if the cache
            does not exist, the built crystals will be saved to this path.
            Defaults to None.
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
        **kwargs,  # for compatibility
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

        if build_crystal_cfg is None:
            build_crystal_cfg = {
                "element_encoding_size": ELEMENT_ENCODING_SIZE,
                "num_cpus": 1,
            }
            logger.message(
                "The build_crystal_cfg is not set, will use the default "
                f"configs: {build_crystal_cfg}"
            )
        self.build_crystal_cfg = build_crystal_cfg

        if cache_path is not None:
            self.cache_path = cache_path
        else:
            # for example:
            # path = ./data/mp_20/train.npz
            # cache_path = ./data/mp_20_cache/train
            self.cache_path = osp.join(
                osp.split(path)[0] + "_cache", osp.splitext(osp.basename(path))[0]
            )
        logger.info(f"Cache path: {self.cache_path}")

        self.overwrite = overwrite
        self.cache_exists = True if osp.exists(self.cache_path) else False

        flat_crystals, self.num_samples = self.read_data(path)
        logger.info(f"Load {self.num_samples} samples from {path}")

        if self.cache_exists and not overwrite:
            try:
                build_crystal_cfg_cache = self.load_from_cache(
                    osp.join(self.cache_path, "build_crystal_cfg.pkl")
                )
                if is_equal(build_crystal_cfg_cache, build_crystal_cfg):
                    logger.info(
                        "The cached build_crystal_cfg configuration matches "
                        "the current settings. Reusing previously generated "
                        "crystal data to optimize performance."
                    )
                else:
                    logger.warning(
                        "build_crystal_cfg is different from "
                        "build_crystal_cfg_cache. Will rebuild the crystals."
                    )
                    logger.warning(
                        "If you want to use the cached crystals, please "
                        "ensure that the settings used in match your "
                        "current settings."
                    )
                    overwrite = True
            except Exception as e:
                logger.warning(e)
                logger.warning(
                    "Failed to load build_crystal_cfg.pkl from cache. "
                    "Will rebuild the crystals."
                )
                overwrite = True

        crystal_cache_path = osp.join(self.cache_path, "crystals")
        if overwrite or not self.cache_exists:
            # convert crystals
            # only rank 0 process do the conversion
            if dist.get_rank() == 0:
                # save build_crystal_cfg to cache file
                os.makedirs(self.cache_path, exist_ok=True)
                self.save_to_cache(
                    osp.join(self.cache_path, "build_crystal_cfg.pkl"),
                    build_crystal_cfg,
                )
                # convert crystals
                crystals = BuildAsuCrystal(**build_crystal_cfg)(flat_crystals)
                # save crystals to cache file
                os.makedirs(crystal_cache_path, exist_ok=True)
                for i in range(self.num_samples):
                    self.save_to_cache(
                        osp.join(crystal_cache_path, f"{i:010d}.pkl"), crystals[i]
                    )
                logger.info(f"Save {self.num_samples} crystals to {crystal_cache_path}")

            # sync all processes
            if dist.is_initialized():
                dist.barrier()
        self.crystals = [
            osp.join(crystal_cache_path, f"{i:010d}.pkl")
            for i in range(self.num_samples)
        ]

    def read_data(self, path: str):
        """Read the packed NPZ archive and split it into per-crystal arrays."""
        npz = np.load(path)
        flat_crystals = np.split(npz["packed"], npz["indices"])
        return flat_crystals, len(flat_crystals)

    def save_to_cache(self, cache_path: str, data):
        with open(cache_path, "wb") as f:
            pickle.dump(data, f)

    def load_from_cache(self, cache_path: str):
        if osp.exists(cache_path):
            with open(cache_path, "rb") as f:
                data = pickle.load(f)
            return data
        else:
            raise FileNotFoundError(f"No such file or directory: {cache_path}")

    def __getitem__(self, index: int) -> dict:
        crystal = self.load_from_cache(self.crystals[index])
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

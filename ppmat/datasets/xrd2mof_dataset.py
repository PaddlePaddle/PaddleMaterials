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
import os.path as osp
from typing import Callable
from typing import Optional

import numpy as np
from paddle.io import Dataset

from ppmat.datasets.custom_data_type import ConcatData
from ppmat.utils import download
from ppmat.utils import logger

XRD2MOF_DATA_NAME = "xrd2mof_cgmof50"
XRD2MOF_DATA_URL = "https://paddle-org.bj.bcebos.com/paddlematerials/datasets/MOF_PXRD/xrd2mof_cgmof50_data.zip"  # noqa
XRD2MOF_DATA_MD5 = "e14457945570dd577e4fcaef82cc5232"


class _Xrd2MofBaseDataset(Dataset):
    """Shared loading logic for the coarse-grained Xrd2Mof npz archives.

    The archive stores variable-length fields (`atom_types`, `frac_coords`)
    concatenated over samples; per-sample slices are recovered from `num_atoms`.
    Subclasses only declare which fields to keep and how to build a sample.
    """

    name = XRD2MOF_DATA_NAME
    url = XRD2MOF_DATA_URL
    md5 = XRD2MOF_DATA_MD5

    def __init__(
        self,
        path: str,
        transforms: Optional[Callable] = None,
        **kwargs,
    ):
        super().__init__()
        self.path = self._resolve_path(path)
        data = np.load(self.path, allow_pickle=True)
        self.num_atoms = data["num_atoms"].astype("int64")
        self.offsets = np.concatenate([[0], np.cumsum(self.num_atoms)]).astype("int64")
        self._load_fields(data)
        self.transforms = transforms

    def _resolve_path(self, path: str) -> str:
        if osp.exists(path):
            return path
        logger.message("The dataset is not found. Will download it now.")
        root_path = download.get_datasets_path_from_url(self.url, self.md5)
        return osp.join(root_path, self.name, osp.basename(path))

    def _load_fields(self, data):
        raise NotImplementedError

    def _get_structure_array(self, idx: int):
        raise NotImplementedError

    def _slice_bounds(self, idx: int):
        return int(self.offsets[idx]), int(self.offsets[idx + 1])

    def __getitem__(self, idx: int):
        data = {"structure_array": self._get_structure_array(idx)}
        data = self.transforms(data) if self.transforms is not None else data
        return data

    def __len__(self):
        return int(self.num_atoms.shape[0])


class Xrd2MofDataset(_Xrd2MofBaseDataset):
    """Coarse-grained MOF dataset with precomputed XRD feature embeddings.

    Each sample is a coarse-grained structure whose atoms are either metal nodes
    (type 0) or organic linkers (type 1), paired with the 256-dim embedding
    extracted from the PXRD pattern and the metal/linker priors.

    The npz archive stores variable-length fields (`atom_types`, `frac_coords`)
    concatenated over samples; per-sample slices are recovered from
    `num_atoms`.

    Args:
        path (str, optional): Path of the dataset. If it does not exist, the
            packaged archive is downloaded and extracted. Defaults to
            "./data/xrd2mof_cgmof50/train.npz".
        transforms (Optional[Callable], optional): Transforms applied to each
            sample. Defaults to None.
    """

    def __init__(
        self,
        path: str = "./data/xrd2mof_cgmof50/train.npz",
        transforms: Optional[Callable] = None,
        **kwargs,
    ):
        super().__init__(path, transforms, **kwargs)

    def _load_fields(self, data):
        self.feature_embedding = data["feature_embedding"].astype("float32")
        self.atom_types = data["atom_types"].astype("int64")
        self.frac_coords = data["frac_coords"].astype("float32")
        self.lengths = data["lengths"].astype("float32")
        self.angles = data["angles"].astype("float32")

    def _get_structure_array(self, idx: int):
        start, end = self._slice_bounds(idx)
        return {
            "feature_embedding": self.feature_embedding[idx],
            "num_atoms": ConcatData(np.array([self.num_atoms[idx]], dtype="int64")),
            "atom_types": ConcatData(self.atom_types[start:end]),
            "frac_coords": ConcatData(self.frac_coords[start:end]),
            "lengths": self.lengths[idx],
            "angles": self.angles[idx],
        }


class Xrd2MofTestDataset(_Xrd2MofBaseDataset):
    """Sampling dataset for Xrd2Mof structure generation.

    Only the generation inputs are provided: the XRD feature embedding, the
    coarse-grained atom types and the sample identifier.

    Args:
        path (str, optional): Path of the dataset. If it does not exist, the
            packaged archive is downloaded and extracted. Defaults to
            "./data/xrd2mof_cgmof50/test.npz".
        transforms (Optional[Callable], optional): Transforms applied to each
            sample. Defaults to None.
    """

    def __init__(
        self,
        path: str = "./data/xrd2mof_cgmof50/test.npz",
        transforms: Optional[Callable] = None,
        **kwargs,
    ):
        super().__init__(path, transforms, **kwargs)

    def _load_fields(self, data):
        self.feature_embedding = data["feature_embedding"].astype("float32")
        self.atom_types = data["atom_types"].astype("int64")
        self.mof_id = data["mof_id"]

    def _get_structure_array(self, idx: int):
        start, end = self._slice_bounds(idx)
        return {
            "feature_embedding": self.feature_embedding[idx],
            "num_atoms": ConcatData(np.array([self.num_atoms[idx]], dtype="int64")),
            "atom_types": ConcatData(self.atom_types[start:end]),
            "mof_id": str(self.mof_id[idx]),
        }

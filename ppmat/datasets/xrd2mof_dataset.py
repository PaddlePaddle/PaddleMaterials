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
from ppmat.utils.crystal import lattice_params_to_matrix


class Xrd2MofDataset(Dataset):
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

    name = "xrd2mof_cgmof50"
    url = "https://paddle-org.bj.bcebos.com/paddlematerials/datasets/MOF_PXRD/xrd2mof_cgmof50_data.zip"  # noqa
    md5 = "e14457945570dd577e4fcaef82cc5232"

    def __init__(
        self,
        path: str = "./data/xrd2mof_cgmof50/train.npz",
        transforms: Optional[Callable] = None,
    ):
        super().__init__()
        if not osp.exists(path):
            logger.message("The dataset is not found. Will download it now.")
            root_path = download.get_datasets_path_from_url(self.url, self.md5)
            path = osp.join(root_path, self.name, osp.basename(path))
        self.path = path
        with np.load(self.path, allow_pickle=True) as data:
            self.num_atoms = data["num_atoms"].astype("int64")
            self.feature_embedding = data["feature_embedding"].astype("float32")
            self.atom_types = data["atom_types"].astype("int64")
            self._load_fields(data)
        self.offsets = np.concatenate([[0], np.cumsum(self.num_atoms)]).astype("int64")
        self.transforms = transforms

    def __getitem__(self, idx: int):
        data = self._get_sample(idx)
        return self.transforms(data) if self.transforms is not None else data

    def __len__(self):
        return int(self.num_atoms.shape[0])

    def _load_fields(self, data):
        self.frac_coords = data["frac_coords"].astype("float32")
        self.lengths = data["lengths"].astype("float32")
        self.angles = data["angles"].astype("float32")
        self.lattice = np.stack(
            [
                lattice_params_to_matrix(*lengths, *angles)
                for lengths, angles in zip(self.lengths, self.angles)
            ]
        ).astype("float32")

    def _get_sample(self, idx: int):
        start, end = self.offsets[idx : idx + 2]
        return {
            "structure_array": {
                "num_atoms": ConcatData(np.array([self.num_atoms[idx]], dtype="int64")),
                "atom_types": ConcatData(self.atom_types[start:end]),
                "frac_coords": ConcatData(self.frac_coords[start:end]),
                "lattice": self.lattice[idx],
                "lengths": self.lengths[idx],
                "angles": self.angles[idx],
            },
            "feature_embedding": self.feature_embedding[idx],
        }


class Xrd2MofTestDataset(Xrd2MofDataset):
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
    ):
        super().__init__(path, transforms)

    def _load_fields(self, data):
        self.mof_id = data["mof_id"]

    def _get_sample(self, idx: int):
        start, end = self.offsets[idx : idx + 2]
        return {
            "structure_array": {
                "num_atoms": ConcatData(np.array([self.num_atoms[idx]], dtype="int64")),
                "atom_types": ConcatData(self.atom_types[start:end]),
            },
            "feature_embedding": self.feature_embedding[idx],
            "mof_id": str(self.mof_id[idx]),
        }

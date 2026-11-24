# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

import json
import os
import os.path as osp
import pickle
from typing import Any
from typing import Dict

import paddle.distributed as dist
from paddle.io import Dataset

from ppmat.datasets.build_structure import BuildStructure
from ppmat.datasets.geometric_data_type.data import Data
from ppmat.models import build_graph_converter
from ppmat.utils import download
from ppmat.utils import logger
from ppmat.utils.misc import is_equal


class DeepHData(Data):
    """Graph sample with DeepH-specific batching increments."""

    def __cat_dim__(self, key, value):
        if key in {"sub_atom_idx", "sub_edge_idx", "sub_edge_ang", "sub_index"}:
            return 0
        return super().__cat_dim__(key, value)

    def __inc__(self, key, value):
        if key == "sub_atom_idx":
            return self.num_nodes
        if key == "sub_edge_idx":
            return self.num_edges
        if key == "sub_index":
            return self.num_edges * 2
        return super().__inc__(key, value)


class DeepHDataset(Dataset):
    """DeepH processed Hamiltonian structures.

    Each sample directory contains lat.dat, site_positions.dat, element.dat,
    orbital_types.dat, rc.npz (local rotations), and rh.npz (Hamiltonian blocks).
    The graphene benchmark contains 450 carbon structures with 13 orbitals per
    atom. Lattice and Cartesian coordinates must be kept unchanged so that the
    periodic matrix keys and local frames remain aligned.

    Structure parsing uses BuildStructure; Hamiltonian labels and LCMP features
    are produced by the configured graph converter. Train/val/test splitting is
    defined by an explicit split manifest, independently of model initialization.

    Args:
        path (str): Directory containing processed sample folders.
        build_graph_cfg (Dict): Graph converter factory configuration.
        build_structure_cfg (Dict, optional): BuildStructure configuration.
            Cell and coordinate normalization must remain disabled.
        nums (int, optional): Limit the number of discovered samples.
        cache_path (str, optional): Per-sample graph cache directory.
        overwrite (bool): Rebuild cached graphs even if the configuration matches.
        url (str, optional): Dataset archive URL used when path does not exist.
        md5 (str, optional): Checksum for the supplied dataset URL.
        split (str, optional): Select train, val or test from split_file.
        split_file (str, optional): JSON manifest of disjoint sample folder IDs.
            Relative paths are resolved from the working directory, then the
            parent of path. The manifest must cover every discovered sample.
    """

    name = "deeph_graphene"
    url = None
    md5 = "0f8a77098cb9334214c6111f967e9afb"

    def __init__(
        self,
        path: str,
        build_graph_cfg: Dict,
        build_structure_cfg: Dict | None = None,
        nums: int | None = None,
        cache_path: str | None = None,
        overwrite: bool = False,
        url: str | None = None,
        md5: str | None = None,
        split: str | None = None,
        split_file: str | None = None,
    ):
        super().__init__()
        if not osp.exists(path):
            dataset_url = url or self.url
            if dataset_url is None:
                raise FileNotFoundError(
                    f"DeepH dataset directory does not exist: {path}. "
                    "Set path to the processed dataset or provide a dataset URL."
                )
            root_path = download.get_datasets_path_from_url(
                dataset_url, md5 or self.md5
            )
            path = osp.join(root_path, "processed")

        self.path = path
        self.build_structure_cfg = build_structure_cfg or {
            "format": "deeph",
            "primitive": False,
            "niggli": False,
            "canocial": False,
            "num_cpus": 1,
        }
        if any(
            self.build_structure_cfg.get(key, default)
            for key, default in (
                ("primitive", False),
                ("niggli", True),
                ("canocial", True),
            )
        ):
            raise ValueError(
                "DeepH matrix keys require primitive=False, niggli=False "
                "and canocial=False to preserve the original coordinates."
            )
        self.build_graph_cfg = build_graph_cfg
        self.folder_list, self.num_samples = self.read_data(path, nums)
        self.sample_ids = list(range(self.num_samples))
        if split is not None:
            if split not in {"train", "val", "test"}:
                raise ValueError("split must be train, val or test.")
            if split_file is None:
                raise ValueError("split_file is required when split is specified.")
            if not osp.isabs(split_file) and not osp.isfile(split_file):
                split_file = osp.join(osp.dirname(path), split_file)
            with open(split_file) as file:
                split_data = json.load(file)
            if set(split_data) != {"train", "val", "test"}:
                raise ValueError("The split manifest must contain train, val and test.")
            entries = [str(entry) for group in split_data.values() for entry in group]
            if len(entries) != len(set(entries)):
                raise ValueError("Duplicate sample IDs in the split manifest.")
            folder_ids = {
                osp.relpath(folder, path): index
                for index, folder in enumerate(self.folder_list)
            }
            unknown = set(entries) - set(folder_ids)
            if unknown:
                raise ValueError(
                    f"Split manifest contains missing samples: {sorted(unknown)}"
                )
            if set(entries) != set(folder_ids):
                raise ValueError("The split manifest must cover all dataset samples.")
            self.sample_ids = [folder_ids[str(entry)] for entry in split_data[split]]
        logger.info(f"Load {self.num_samples} DeepH samples from {path}")
        self.cache_path = cache_path or path.rstrip(os.sep) + "_cache"
        graph_cache_path = osp.join(self.cache_path, "graphs")
        self.graphs = [
            osp.join(graph_cache_path, f"{index:010d}.pkl")
            for index in range(self.num_samples)
        ]
        cfg_cache_path = osp.join(self.cache_path, "dataset_cfg.pkl")
        cache_cfg = {
            "cache_version": 5,
            "folders": self.folder_list,
            "build_structure_cfg": self.build_structure_cfg,
            "build_graph_cfg": self.build_graph_cfg,
        }
        if all(osp.isfile(path) for path in self.graphs) and not overwrite:
            try:
                overwrite = not is_equal(
                    self.load_from_cache(cfg_cache_path), cache_cfg
                )
            except (OSError, EOFError, pickle.UnpicklingError):
                overwrite = True
        else:
            overwrite = True

        if overwrite and dist.get_rank() == 0:
            structures = BuildStructure(**self.build_structure_cfg)(self.folder_list)
            converter = build_graph_converter(self.build_graph_cfg)
            for index, (structure, folder) in enumerate(
                zip(structures, self.folder_list)
            ):
                graph = converter(structure, folder)
                model_input = converter.to_model_input(graph, structure, folder)
                self.save_to_cache(
                    self.graphs[index],
                    DeepHData(num_nodes=len(structure), **model_input),
                )
                if (index + 1) % 25 == 0 or index + 1 == self.num_samples:
                    logger.info(f"Cached {index + 1}/{self.num_samples} DeepH graphs")
            self.save_to_cache(cfg_cache_path, cache_cfg)
        if dist.is_initialized():
            dist.barrier()

    def read_data(self, path: str, nums: int | None = None):
        folders = sorted(
            osp.abspath(root) for root, _, files in os.walk(path) if "rc.npz" in files
        )
        if nums is not None:
            if nums <= 0:
                raise ValueError("nums must be positive.")
            folders = folders[:nums]
        if not folders:
            raise ValueError(
                f"No DeepH samples containing rc.npz were found in {path}."
            )
        return folders, len(folders)

    def save_to_cache(self, cache_path: str, data: Any):
        os.makedirs(osp.dirname(cache_path), exist_ok=True)
        with open(cache_path, "wb") as file:
            pickle.dump(data, file)

    def load_from_cache(self, cache_path: str):
        with open(cache_path, "rb") as file:
            return pickle.load(file)

    def __getitem__(self, index: int):
        return self.load_from_cache(self.graphs[self.sample_ids[index]])

    def __len__(self):
        return len(self.sample_ids)

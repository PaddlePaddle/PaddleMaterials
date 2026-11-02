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

from __future__ import annotations

import math
import os
import os.path as osp
import pickle
from typing import Any
from typing import Callable
from typing import Dict
from typing import Optional

import numpy as np
import paddle.distributed as dist
import pandas as pd
from paddle.io import Dataset

from ppmat.datasets.build_molecule import BuildMolecule
from ppmat.utils import download
from ppmat.utils import logger
from ppmat.utils.misc import is_equal


class TransPolymerDataset(Dataset):
    """Dataset for TransPolymer downstream property prediction.

    The official TransPolymer archive is downloaded automatically when the
    requested CSV is absent. The dataset returns raw polymer SMILES and labels;
    tokenizer loading, token caching, and encoding are handled by the model.
    """

    name = "transpolymer_data"
    url = (
        "https://paddle-org.bj.bcebos.com/paddlematerials/datasets/"
        "TransPolymer_data/transpolymer_data.tar.gz"
    )
    md5 = "d483359a1607aecd4453ba286a66b6c9"

    def __init__(
        self,
        path: Optional[str] = None,
        property_names: Optional[list[str]] = None,
        smiles_key: str = "smiles",
        build_molecule_cfg: Optional[Dict[str, Any]] = None,
        build_sequence_cfg: Optional[Dict[str, Any]] = None,
        transforms: Optional[Callable] = None,
        cache_path: Optional[str] = None,
        overwrite: bool = False,
        filter_unvalid: bool = True,
        return_molecule: bool = False,
        file_path: Optional[str] = None,
        label_name: Optional[str] = None,
        **kwargs,
    ):
        super().__init__()
        path = path or file_path
        if path is None:
            raise ValueError("`path` must be specified for TransPolymerDataset.")

        self.path = self.resolve_path(path)
        self.smiles_key = smiles_key
        self.transforms = transforms
        self.build_molecule_cfg = build_molecule_cfg
        self.build_sequence_cfg = build_sequence_cfg or {}
        self.return_molecule = return_molecule
        self.overwrite = overwrite

        if isinstance(property_names, str):
            property_names = [property_names]
        if property_names is not None and len(property_names) != 1:
            raise NotImplementedError(
                "TransPolymerDataset currently supports single-target regression."
            )

        self.row_data, self.num_samples = self.read_data(self.path)
        self.sequence_column = self.build_sequence_cfg.get("column", smiles_key)
        if self.sequence_column not in self.row_data:
            raise KeyError(
                f"SMILES column {self.sequence_column!r} is not found in {self.path}."
            )
        self.property_names = property_names or [
            label_name or self._infer_property_name(self.row_data)
        ]
        self.property_data = self.read_property_data(self.row_data, self.property_names)

        if filter_unvalid:
            self.filter_unvalid_by_property()

        self.cache_path = cache_path or osp.join(
            osp.split(self.path)[0] + "_cache",
            osp.splitext(osp.basename(self.path))[0],
        )
        self.sequences = self._prepare_sequence_cache()
        self.molecules = self._prepare_molecule_cache()
        logger.info(f"Load {self.num_samples} samples from {self.path}")

    @classmethod
    def resolve_path(cls, path: str) -> str:
        """Resolve a local CSV, downloading the official archive if necessary."""
        if osp.exists(path):
            return path
        logger.message("TransPolymer dataset is not found. Will download it now.")
        root_path = download.get_datasets_path_from_url(cls.url, cls.md5)
        downloaded_path = osp.join(root_path, osp.basename(path))
        if not osp.exists(downloaded_path):
            raise FileNotFoundError(
                f"{osp.basename(path)} is missing from downloaded archive at {root_path}."
            )
        return downloaded_path

    @staticmethod
    def _infer_property_name(row_data: Dict[str, list]) -> str:
        for name in row_data:
            if name.lower() not in {"smiles", "csmiles", "id"}:
                return name
        raise ValueError("Could not infer a property column from TransPolymer CSV.")

    @staticmethod
    def _cache_config_matches(cache_file: str, config: Dict[str, Any]) -> bool:
        if not osp.exists(cache_file):
            return False
        with open(cache_file, "rb") as f:
            return is_equal(pickle.load(f), config)

    @staticmethod
    def _save_cache(cache_file: str, data: Any) -> None:
        os.makedirs(osp.dirname(cache_file), exist_ok=True)
        with open(cache_file, "wb") as f:
            pickle.dump(data, f)

    @staticmethod
    def _load_cache(cache_file: str) -> Any:
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    def _prepare_sequence_cache(self) -> list[str]:
        config = {
            "column": self.sequence_column,
            "strip": bool(self.build_sequence_cfg.get("strip", True)),
        }
        cache_file = osp.join(self.cache_path, "sequences.pkl")
        config_file = osp.join(self.cache_path, "build_sequence_cfg.pkl")
        rebuild = self.overwrite or not (
            osp.exists(cache_file) and self._cache_config_matches(config_file, config)
        )

        if rebuild and dist.get_rank() == 0:
            sequences = [str(value) for value in self.row_data[self.sequence_column]]
            if config["strip"]:
                sequences = [value.strip() for value in sequences]
            self._save_cache(cache_file, sequences)
            self._save_cache(config_file, config)
        if dist.is_initialized():
            dist.barrier()
        return self._load_cache(cache_file)

    def _prepare_molecule_cache(self) -> Optional[list[Any]]:
        if self.build_molecule_cfg is None:
            return None

        cache_file = osp.join(self.cache_path, "molecules.pkl")
        config_file = osp.join(self.cache_path, "build_molecule_cfg.pkl")
        rebuild = self.overwrite or not (
            osp.exists(cache_file)
            and self._cache_config_matches(config_file, self.build_molecule_cfg)
        )
        if rebuild and dist.get_rank() == 0:
            molecules = BuildMolecule(**self.build_molecule_cfg)(self.sequences)
            self._save_cache(cache_file, molecules)
            self._save_cache(config_file, self.build_molecule_cfg)
        if dist.is_initialized():
            dist.barrier()
        return self._load_cache(cache_file)

    @staticmethod
    def read_data(path: str) -> tuple[Dict[str, list], int]:
        data_frame = pd.read_csv(path)
        data = {
            column: data_frame[column].tolist()
            for column in data_frame.columns
            if not column.startswith("Unnamed")
        }
        return data, len(data_frame)

    @staticmethod
    def read_property_data(
        row_data: Dict[str, list], property_names: list[str]
    ) -> Dict[str, list]:
        missing = [name for name in property_names if name not in row_data]
        if missing:
            raise KeyError(f"Property columns not found: {missing}")
        return {name: row_data[name] for name in property_names}

    def filter_unvalid_by_property(self) -> None:
        property_name = self.property_names[0]
        valid_indices = []
        for index, value in enumerate(self.property_data[property_name]):
            try:
                is_valid = value is not None and math.isfinite(float(value))
            except (TypeError, ValueError):
                is_valid = False
            if is_valid:
                valid_indices.append(index)

        if len(valid_indices) == self.num_samples:
            return
        self.row_data = {
            name: [values[index] for index in valid_indices]
            for name, values in self.row_data.items()
        }
        self.property_data = {
            name: [values[index] for index in valid_indices]
            for name, values in self.property_data.items()
        }
        self.num_samples = len(valid_indices)
        logger.warning(f"Remaining {self.num_samples} samples after filtering.")

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        property_name = self.property_names[0]
        sample = {
            self.smiles_key: self.sequences[idx],
            property_name: np.asarray([self.property_data[property_name][idx]], dtype="float32"),
            "id": self.row_data.get("id", list(range(self.num_samples)))[idx],
        }
        if self.return_molecule and self.molecules is not None:
            sample["molecule"] = self.molecules[idx]
        if self.transforms is not None:
            sample = self.transforms(sample)
        return sample

    def __len__(self) -> int:
        return self.num_samples


TransPolymerCsvDataset = TransPolymerDataset

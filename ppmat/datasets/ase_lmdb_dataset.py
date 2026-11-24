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

import glob
import hashlib
import json
import os
import os.path as osp
import pickle
import tempfile
import zlib
from typing import Any
from typing import Callable

import lmdb
import numpy as np
from ase import Atoms
from ase.io.jsonio import decode
from paddle.io import Dataset

from ppmat.datasets.build_structure import BuildStructure
from ppmat.datasets.custom_data_type import ConcatData
from ppmat.datasets.custom_data_type import ConcatNumpyWarper
from ppmat.models import build_graph_converter
from ppmat.utils import download
from ppmat.utils import logger


class ASELMDBDataset(Dataset):
    """Read structures and labels from compressed ASE-LMDB files.

    Args:
        path: Local ASE-LMDB file or directory of shards. Registered datasets
            are downloaded when this path is missing.
        property_names: Labels to return, selected from energy and forces.
        build_structure_cfg: Configuration for converting ASE atoms.
        build_graph_cfg: Configuration for the registered graph converter.
        transforms: Optional transform applied to each sample.
        cache_path: Optional directory for on-demand structure and graph caches.
        overwrite: Rebuild cached samples instead of reading existing files.
        filter_unvalid: Remove records with nonfinite property values.
        pattern: Glob pattern for shards within a local directory.
        offset: Number of initial records to skip.
        limit: Maximum number of records to load.
        energy_key: Energy label key in the decoded record.
        forces_key: Force label key in the decoded record.
        split: Dataset partition used to resolve the download URL and paths.
    """

    name: str | None = None
    url: str | None = None
    md5: str | None = None
    splits: tuple[str, ...] | None = None
    download_subdir: str = ""
    dataset_subdir: str = ""

    def __init__(
        self,
        path: str | os.PathLike[str] | None = None,
        property_names: str | list[str] | tuple[str, ...] = ("energy", "forces"),
        build_structure_cfg: dict[str, Any] | None = None,
        build_graph_cfg: dict[str, Any] | None = None,
        transforms: Callable | None = None,
        cache_path: str | os.PathLike[str] | None = None,
        overwrite: bool = False,
        filter_unvalid: bool = False,
        split: str = "train",
        pattern: str = "*.aselmdb",
        offset: int = 0,
        limit: int | None = None,
        energy_key: str = "energy",
        forces_key: str = "forces",
    ) -> None:
        super().__init__()
        if self.splits is not None and split not in self.splits:
            supported = ", ".join(self.splits)
            raise ValueError(
                f"{type(self).__name__} supports only {supported} splits, "
                f"got {split!r}."
            )
        self.split = split
        if path is None or not osp.exists(path):
            if self.url is None or self.name is None:
                raise FileNotFoundError(
                    f"No local ASE-LMDB data at {path!r} "
                    "and no download URL configured."
                )
            logger.message("The dataset is not found. Will download it now.")
            cache_dir = osp.join(
                download.DATASETS_HOME,
                self.download_subdir.format(name=self.name, split=split),
            )
            url = self.url.format(split=split)
            if self.download_subdir:
                root_path = download.get_path_from_url(url, cache_dir, self.md5)
            else:
                root_path = download.get_datasets_path_from_url(url, self.md5)
            package_path = osp.join(root_path, self.name)
            if osp.isdir(package_path):
                root_path = package_path
            relative_path = self.dataset_subdir.format(split=split)
            path = osp.join(root_path, relative_path) if relative_path else root_path
        self.path = os.fspath(path)
        if offset < 0 or (limit is not None and limit < 0):
            raise ValueError("offset and limit must be non-negative.")
        if isinstance(property_names, str):
            property_names = [property_names]
        self.property_names = tuple(property_names)
        unknown_properties = set(self.property_names) - {"energy", "forces"}
        if unknown_properties:
            raise ValueError(f"Unsupported properties: {sorted(unknown_properties)}.")

        if build_structure_cfg is None:
            build_structure_cfg = {
                "format": "ase_atoms",
                "primitive": False,
                "niggli": False,
                "canocial": False,
                "num_cpus": 1,
            }
        self.build_structure_cfg = build_structure_cfg
        self.build_graph_cfg = build_graph_cfg
        self.energy_key = energy_key
        self.forces_key = forces_key
        self.transforms = transforms
        self.pattern = pattern
        self.offset = offset
        self.limit = limit
        self.overwrite = overwrite
        self.filter_unvalid = filter_unvalid
        self.indices, self.num_samples = self.read_data(self.path)
        if self.filter_unvalid:
            self.filter_unvalid_by_property()
        self.num_samples = len(self.indices)
        if not self.num_samples:
            logger.error(f"No valid samples found in {self.path}.")
            raise ValueError(f"No valid samples found in {self.path}.")
        self.build_structure = BuildStructure(**self.build_structure_cfg)
        self.graph_converter = build_graph_converter(self.build_graph_cfg)
        logger.info(f"Load {self.num_samples} samples from {self.path}")

        self.cache_path = None
        if cache_path is not None:
            cache_config = {
                "sources": [
                    (path, os.stat(path).st_size, os.stat(path).st_mtime_ns)
                    for path in self.paths
                ],
                "build_structure_cfg": self.build_structure_cfg,
                "build_graph_cfg": self.build_graph_cfg,
                "property_names": self.property_names,
                "energy_key": self.energy_key,
                "forces_key": self.forces_key,
            }
            cache_key = hashlib.sha256(
                json.dumps(cache_config, sort_keys=True).encode()
            ).hexdigest()[:16]
            self.cache_path = osp.join(cache_path, cache_key)
            logger.info(f"Cache path: {self.cache_path}")

    def read_data(self, path: str, index: int | None = None):
        """Index the LMDB split, or read one indexed sample without preloading it."""
        try:
            if index is not None:
                database_index, row_id = self.indices[index]
                with self.databases[database_index].begin() as transaction:
                    raw = transaction.get(str(row_id).encode())
                if raw is None:
                    raise KeyError(
                        f"Missing row {row_id} in {self.paths[database_index]}."
                    )
                try:
                    raw = zlib.decompress(raw)
                except zlib.error:
                    pass
                return decode(raw.decode(), always_array=False)

            self.paths = (
                [osp.abspath(path)]
                if osp.isfile(path)
                else sorted(glob.glob(osp.join(osp.abspath(path), self.pattern)))
            )
            if not self.paths:
                raise FileNotFoundError(f"No ASE-LMDB file found in {path}.")
            self.databases = []
            indices = []
            offset = self.offset
            for database_index, database_path in enumerate(self.paths):
                database = lmdb.open(
                    database_path,
                    subdir=False,
                    readonly=True,
                    lock=False,
                    readahead=False,
                    meminit=False,
                )
                self.databases.append(database)
                with database.begin() as transaction:
                    length = transaction.get(b"length")
                    if length is not None:
                        row_ids = range(1, int(length.decode()) + 1)
                    else:
                        row_ids = sorted(
                            int(key.decode())
                            for key in transaction.cursor().iternext(values=False)
                            if key.isdigit()
                        )
                start = min(offset, len(row_ids))
                offset -= start
                stop = (
                    start + self.limit - len(indices)
                    if self.limit is not None
                    else len(row_ids)
                )
                indices.extend(
                    (database_index, row_id) for row_id in row_ids[start:stop]
                )
                if self.limit is not None and len(indices) == self.limit:
                    break
            if not indices:
                raise ValueError(f"No records found in {path}.")
            return indices, len(indices)
        except (OSError, lmdb.Error, ValueError, KeyError) as error:
            logger.error(f"Failed to read ASE-LMDB data from {path}: {error}")
            raise

    def read_property_data(self, record: dict, property_names) -> dict:
        property_data = {}
        for property_name in property_names:
            key = self.energy_key if property_name == "energy" else self.forces_key
            value = record.get(key)
            if value is None:
                value = (record.get("data") or {}).get(key)
            if property_name == "energy":
                value = [np.nan if value is None else value]
            elif value is None:
                value = np.full((len(record["numbers"]), 3), np.nan)
            property_data[property_name] = np.asarray(value, dtype="float32")
        return property_data

    def filter_unvalid_by_property(self) -> None:
        reserved = []
        for index, row in enumerate(self.indices):
            record = self.read_data(self.path, index=index)
            properties = self.read_property_data(record, self.property_names)
            if all(np.isfinite(value).all() for value in properties.values()):
                reserved.append(row)
        logger.info(f"Filter out {len(self.indices) - len(reserved)} invalid samples.")
        self.indices = reserved

    @staticmethod
    def save_to_cache(cache_path: str, data: Any) -> None:
        os.makedirs(osp.dirname(cache_path), exist_ok=True)
        with tempfile.NamedTemporaryFile(
            dir=osp.dirname(cache_path), delete=False
        ) as file_obj:
            temporary_path = file_obj.name
            try:
                pickle.dump(data, file_obj)
                file_obj.close()
                os.replace(temporary_path, cache_path)
            finally:
                if osp.exists(temporary_path):
                    os.unlink(temporary_path)

    @staticmethod
    def load_from_cache(cache_path: str) -> Any:
        with open(cache_path, "rb") as file_obj:
            return pickle.load(file_obj)

    @staticmethod
    def get_structure_array(structure) -> dict:
        return {
            "frac_coords": ConcatData(structure.frac_coords.astype("float32")),
            "cart_coords": ConcatData(structure.cart_coords.astype("float32")),
            "atom_types": ConcatData(np.asarray(structure.atomic_numbers)),
            "lattice": ConcatData(structure.lattice.matrix.astype("float32")[None]),
            "lengths": ConcatData(np.asarray([structure.lattice.abc], dtype="float32")),
            "angles": ConcatData(
                np.asarray([structure.lattice.angles], dtype="float32")
            ),
            "num_atoms": ConcatData(np.asarray([len(structure)])),
        }

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> dict[str, Any]:
        database_index, row_id = self.indices[index]
        cache_path = (
            osp.join(self.cache_path, f"{database_index:04d}_{row_id:010d}.pkl")
            if self.cache_path is not None
            else None
        )
        if cache_path is not None and osp.isfile(cache_path) and not self.overwrite:
            data = self.load_from_cache(cache_path)
            return self.transforms(data) if self.transforms is not None else data

        record = self.read_data(self.path, index=index)
        atoms = Atoms(
            numbers=record["numbers"],
            positions=record["positions"],
            cell=record.get("cell"),
            pbc=record.get("pbc", True),
            tags=record.get("tags"),
        )
        structure = self.build_structure(atoms)
        if self.graph_converter is not None:
            graph = self.graph_converter(structure)
            if graph is None:
                raise ValueError(f"Failed to build graph for sample {index}.")
            data = {"graph": graph}
        else:
            data = {"structure_array": self.get_structure_array(structure)}
        data.update(self.read_property_data(record, self.property_names))
        if "forces" in data:
            data["forces"] = ConcatNumpyWarper(data["forces"])
        if cache_path is not None:
            self.save_to_cache(cache_path, data)
        return self.transforms(data) if self.transforms is not None else data

    def __del__(self):
        for database in getattr(self, "databases", []):
            database.close()


class OMat24Dataset(ASELMDBDataset):
    """OMat24 rattled-500 ASE-LMDB release with train and val partitions."""

    name = "OMat24"
    url = (
        "https://paddle-org.bj.bcebos.com/paddlematerials/datasets/OMat24/"
        "{split}/rattled-500.tar.gz"
    )
    splits = ("train", "val")
    download_subdir = osp.join("{name}", "{split}")


class UMAOC20Dataset(ASELMDBDataset):
    """Prepared OC20 S2EF data used by the UMA configuration."""

    name = "uma_datasets"
    url = (
        "https://paddle-org.bj.bcebos.com/paddlematerials/datasets/UMA/uma_datasets.zip"
    )
    splits = ("train", "val", "test")
    dataset_subdir = osp.join("oc20", "uma_aselmdb", "{split}")

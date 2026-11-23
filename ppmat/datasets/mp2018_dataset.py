# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import annotations

import math
import os
import os.path as osp
import pickle
from collections import defaultdict
from typing import Any
from typing import Callable
from typing import Dict
from typing import Optional

import numpy as np
import paddle.distributed as dist
from paddle.io import Dataset

from ppmat.datasets.build_structure import BuildStructure
from ppmat.datasets.custom_data_type import ConcatData
from ppmat.models import build_graph_converter
from ppmat.utils import download
from ppmat.utils import logger
from ppmat.utils.io import read_json
from ppmat.utils.misc import is_equal


class MP2018Dataset(Dataset):
    """MP2018.6.1 Dataset Handler

    This class provides utilities for loading and processing the MP2018.6.1 materials
    science dataset. The dataset contains computed properties of inorganic materials.
    The implementation supports both standard dataset loading and custom data
    processing when adhering to the MP2018.6.1 schema.

    **Dataset Overview**
    - **Source**: Original data available at https://figshare.com/ndownloader/files/15087992
    - **Preprocessed Version**:
    ```
    ┌───────────────────┬─────────┬─────────┬─────────┐
    │ Dataset Partition │ Train   │ Val     │ Test    │
    ├───────────────────┼─────────┼─────────┼─────────┤
    │ Sample Count      │ 60,000  │ 5,000   │ 4,239   │
    └───────────────────┴─────────┴─────────┴─────────┘
    ```
    Download preprocessed data: https://paddle-org.bj.bcebos.com/paddlematerial/datasets/mp2018/mp2018_train_60k.zip

    **Data Format**
    The dataset is stored in JSON format with the following structure:

    ```json
    {
    "structure": {
        "0": "<CIF string>",    // Crystal structure in CIF format
        "1": "<CIF string>",
        // ...
    },
    "material_id": {
        "0": "mvc-8139",        // Unique material identifier
        "1": "mvc-600",
        // ...
    },
    "formation_energy_per_atom": {  // Formation energy (eV/atom)
        "0": -1.8169,
        "1": -1.8948,
        // ...
    },
    "band_gap": {              // Electronic band gap (eV)
        "0": 0.0149,
        "1": 0.0,
        // ...
    },
    "G": {                     // Shear modulus (GPa)
        "0": 45.0,
        "1": null,               // Missing value indicator
        // ...
    },
    "K": {                     // Bulk modulus (GPa)
        "0": 91.0,
        "1": null,               // Missing value indicator
        // ...
    }
    }
    ```

    **Notes**
    - Missing values are represented as `null` in JSON and converted to `NaN` during
        loading
    - CIF parsing requires additional dependencies (e.g., pymatgen)
    - For custom data, ensure index consistency across all fields


    Args:
        path (str, optional): The path of the dataset, if path is not exists, it will
            be downloaded. Defaults to "./data/mp18/mp.2018.6.1.json".
        property_names (Optional[list[str]], optional): Property names you want to use,
            for mp2018.6.1, the property_names should be selected from
            ["formation_energy_per_atom", "band_gap", "G", "K"]. Defaults to None.
        build_structure_cfg (Dict, optional): The configs for building the pymatgen
            structure from cif string, if not specified, the default setting will be
            used. Defaults to None.
        build_graph_cfg (Dict, optional): The configs for building the graph from
            structure. Defaults to None.
        transforms (Optional[Callable], optional): The preprocess transforms for each
            sample. Defaults to None.
        cache_path (Optional[str], optional): If a cache_path is set, structures and
            graph will be read directly from this path; if the cache does not exist,
            the converted structures and graph will be saved to this path. Defaults
            to None.
        overwrite (bool, optional): Overwrite the existing cache file at the given cache
            path if it already exists. Defaults to False.
        filter_unvalid (bool, optional): Whether to filter out unvalid samples. Defaults
            to True.
    """

    name = "mp2018_train_60k"
    url = "https://paddle-org.bj.bcebos.com/paddlematerial/datasets/mp2018/mp2018_train_60k.zip"
    md5 = "216202f16a5081358798e15c060facee"

    def __init__(
        self,
        path: str = "./data/mp18/mp.2018.6.1.json",
        property_names: Optional[list[str]] = None,
        build_structure_cfg: Dict = None,
        build_graph_cfg: Dict = None,
        transforms: Optional[Callable] = None,
        cache_path: Optional[str] = None,
        overwrite: bool = False,
        filter_unvalid: bool = True,
        **kwargs,  # for compatibility
    ):
        super().__init__()

        if not osp.exists(path):
            logger.message("The dataset is not found. Will download it now.")
            root_path = download.get_datasets_path_from_url(self.url, self.md5)
            path = osp.join(root_path, self.name, osp.basename(path))

        self.path = path
        if isinstance(property_names, str):
            property_names = [property_names]

        if build_structure_cfg is None:
            build_structure_cfg = {
                "format": "cif_str",
                "primitive": False,
                "niggli": True,
                "num_cpus": 1,
            }
            logger.message(
                "The build_structure_cfg is not set, will use the default "
                f"configs: {build_structure_cfg}"
            )

        self.property_names = property_names if property_names is not None else []
        self.build_structure_cfg = build_structure_cfg
        self.build_graph_cfg = build_graph_cfg
        self.transforms = transforms

        if cache_path is not None:
            self.cache_path = cache_path
        else:
            # for example:
            # path = ./data/mp2018_train_60k/mp2018_train_60k_train.json
            # cache_path = ./data/mp2018_train_60k_cache/mp2018_train_60k_train
            self.cache_path = osp.join(
                osp.split(path)[0] + "_cache", osp.splitext(osp.basename(path))[0]
            )
        logger.info(f"Cache path: {self.cache_path}")

        self.overwrite = overwrite
        self.filter_unvalid = filter_unvalid

        self.cache_exists = True if osp.exists(self.cache_path) else False
        self.raw_data, self.num_samples = self.read_data(path)
        logger.info(f"Load {self.num_samples} samples from {path}")
        self.property_data = self.read_property_data(self.raw_data, self.property_names)

        if self.cache_exists and not overwrite:
            logger.warning(
                "Cache enabled. If a cache file exists, it will be automatically "
                "read and current settings will be ignored. Please ensure that the "
                "settings used in match your current settings."
            )
            try:
                build_structure_cfg_cache = self.load_from_cache(
                    osp.join(self.cache_path, "build_structure_cfg.pkl")
                )
                if is_equal(build_structure_cfg_cache, build_structure_cfg):
                    logger.info(
                        "The cached build_structure_cfg configuration matches "
                        "the current settings. Reusing previously generated"
                        " structural data to optimize performance."
                    )
                else:
                    logger.warning(
                        "build_structure_cfg is different from "
                        "build_structure_cfg_cache. Will rebuild the structures and "
                        "graphs."
                    )
                    logger.warning(
                        "If you want to use the cached structures and graphs, please "
                        "ensure that the settings used in match your current settings."
                    )
                    overwrite = True
            except Exception as e:
                logger.warning(e)
                logger.warning(
                    "Failed to load builded_structure_cfg.pkl from cache. "
                    "Will rebuild the structures and graphs(if need)."
                )
                overwrite = True

            if build_graph_cfg is not None and not overwrite:
                try:
                    build_graph_cfg_cache = self.load_from_cache(
                        osp.join(self.cache_path, "build_graph_cfg.pkl")
                    )
                    if is_equal(build_graph_cfg_cache, build_graph_cfg):
                        logger.info(
                            "The cached build_structure_cfg configuration "
                            "matches the current settings. Reusing previously "
                            "generated structural data to optimize performance."
                        )
                    else:
                        logger.warning(
                            "build_graph_cfg is different from build_graph_cfg_cache"
                            ". Will rebuild the graphs."
                        )
                        logger.warning(
                            "If you want to use the cached structures and graphs, "
                            "please ensure that the settings used in match your "
                            "current settings."
                        )
                        overwrite = True

                except Exception as e:
                    logger.warning(e)
                    logger.warning(
                        "Failed to load builded_graph_cfg.pkl from cache. "
                        "Will rebuild the graphs."
                    )
                    overwrite = True

        structure_cache_path = osp.join(self.cache_path, "structures")
        graph_cache_path = osp.join(self.cache_path, "graphs")
        if overwrite or not self.cache_exists:
            # convert strucutes and graphs
            # only rank 0 process do the conversion
            if dist.get_rank() == 0:
                # save build_structure_cfg and build_graph_cfg to cache file
                os.makedirs(self.cache_path, exist_ok=True)
                self.save_to_cache(
                    osp.join(self.cache_path, "build_structure_cfg.pkl"),
                    build_structure_cfg,
                )
                self.save_to_cache(
                    osp.join(self.cache_path, "build_graph_cfg.pkl"), build_graph_cfg
                )
                # convert strucutes
                structures = BuildStructure(**build_structure_cfg)(
                    self.raw_data["structure"]
                )
                # save structures to cache file
                os.makedirs(structure_cache_path, exist_ok=True)
                for i in range(self.num_samples):
                    self.save_to_cache(
                        osp.join(structure_cache_path, f"{i:010d}.pkl"),
                        structures[i],
                    )
                logger.info(
                    f"Save {self.num_samples} structures to {structure_cache_path}"
                )

                if build_graph_cfg is not None:
                    converter = build_graph_converter(build_graph_cfg)
                    graphs = converter(structures)
                    # save graphs to cache file
                    os.makedirs(graph_cache_path, exist_ok=True)
                    for i in range(self.num_samples):
                        self.save_to_cache(
                            osp.join(graph_cache_path, f"{i:010d}.pkl"), graphs[i]
                        )
                    logger.info(f"Save {self.num_samples} graphs to {graph_cache_path}")

            # sync all processes
            if dist.is_initialized():
                dist.barrier()
        self.structures = [
            osp.join(structure_cache_path, f"{i:010d}.pkl")
            for i in range(self.num_samples)
        ]
        if build_graph_cfg is not None:
            self.graphs = [
                osp.join(graph_cache_path, f"{i:010d}.pkl")
                for i in range(self.num_samples)
            ]
        else:
            self.graphs = None

        assert (
            len(self.structures) == self.num_samples
        ), "The number of structures must be equal to the number of samples."
        assert (
            self.graphs is None or len(self.graphs) == self.num_samples
        ), "The number of graphs must be equal to the number of samples."

        # filter by property data, since some samples may have no valid properties
        if filter_unvalid:
            self.filter_unvalid_by_property()

    def read_data(self, path: str):
        """Read the data from the given json path.

        Args:
            path (str): Path to the data.
        """
        json_data = read_json(path)
        num_samples = len(json_data["structure"])

        idxs = list(json_data["structure"])
        data = defaultdict(list)
        for key in json_data.keys():
            for idx in idxs:
                data[key].append(json_data[key][idx])
        return data, num_samples

    def filter_unvalid_by_property(self):
        for property_name in self.property_names:
            data = self.property_data[property_name]
            reserve_idx = []
            for i, data_item in enumerate(data):
                if isinstance(data_item, str) or (
                    data_item is not None and not math.isnan(data_item)
                ):
                    reserve_idx.append(i)
            for key in self.property_data.keys():
                self.property_data[key] = [
                    self.property_data[key][i] for i in reserve_idx
                ]

            self.raw_data = [self.raw_data[i] for i in reserve_idx]
            self.structures = [self.structures[i] for i in reserve_idx]
            if self.graphs is not None:
                self.graphs = [self.graphs[i] for i in reserve_idx]
            logger.warning(
                f"Filter out {len(reserve_idx)} samples with valid properties: "
                f"{property_name}"
            )
        self.num_samples = len(self.raw_data)
        logger.warning(f"Remaining {self.num_samples} samples after filtering.")

    def read_property_data(self, data: Dict, property_names: list[str]):
        """Read the property data from the given data and property names.

        Args:
            data (Dict): Data that contains the property data.
            property_names (list[str]): Property names.
        """
        property_data = {}
        for property_name in property_names:
            if property_name not in data:
                raise ValueError(f"{property_name} not found in the data")
            property_data[property_name] = [
                data[property_name][i] for i in range(self.num_samples)
            ]
        return property_data

    def save_to_cache(self, cache_path: str, data: Any):
        with open(cache_path, "wb") as f:
            pickle.dump(data, f)

    def load_from_cache(self, cache_path: str):
        if osp.exists(cache_path):
            with open(cache_path, "rb") as f:
                data = pickle.load(f)
            return data
        else:
            raise FileNotFoundError(f"No such file or directory: {cache_path}")

    def get_structure_array(self, structure):
        atom_types = np.array([site.specie.Z for site in structure])
        # get lattice parameters and matrix
        lattice_parameters = structure.lattice.parameters
        lengths = np.array(lattice_parameters[:3], dtype="float32").reshape(1, 3)
        angles = np.array(lattice_parameters[3:], dtype="float32").reshape(1, 3)
        lattice = structure.lattice.matrix.astype("float32")

        structure_array = {
            "frac_coords": ConcatData(structure.frac_coords.astype("float32")),
            "cart_coords": ConcatData(structure.cart_coords.astype("float32")),
            "atom_types": ConcatData(atom_types),
            "lattice": ConcatData(lattice.reshape(1, 3, 3)),
            "lengths": ConcatData(lengths),
            "angles": ConcatData(angles),
            "num_atoms": ConcatData(np.array([tuple(atom_types.shape)[0]])),
        }
        return structure_array

    def __getitem__(self, idx: int):
        """Get item at index idx."""
        data = {}
        # get graph
        if self.graphs is not None:
            graph = self.graphs[idx]
            if isinstance(graph, str):
                graph = self.load_from_cache(graph)
            data["graph"] = graph
        else:
            structure = self.structures[idx]
            if isinstance(structure, str):
                structure = self.load_from_cache(structure)
            data["structure_array"] = self.get_structure_array(structure)
        for property_name in self.property_names:
            if property_name in self.property_data:
                data[property_name] = np.array(
                    [self.property_data[property_name][idx]]
                ).astype("float32")
            else:
                raise KeyError(f"Property {property_name} not found.")

        data["id"] = (
            self.property_data["id"][idx] if "id" in self.property_data else idx
        )
        data = self.transforms(data) if self.transforms is not None else data

        return data

    def __len__(self):
        return self.num_samples

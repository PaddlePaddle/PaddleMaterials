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
from typing import Any
from typing import Callable
from typing import Dict
from typing import Optional

import numpy as np
import paddle.distributed as dist
import pandas as pd
from p_tqdm import p_map
from paddle.io import Dataset
from pymatgen.symmetry.groups import SpaceGroup

from ppmat.datasets.build_structure import BuildStructure
from ppmat.datasets.custom_data_type import ConcatData
from ppmat.models import build_graph_converter
from ppmat.utils import download
from ppmat.utils import logger
from ppmat.utils.misc import is_equal


class MP20Dataset(Dataset):
    """MP20 Dataset Handler

    This class provides utilities for loading and processing the MP20 materials
    science dataset, which is utilized in both CDVAE and DiffCSP models. The
    implementation supports both standard dataset loading and custom data processing
    when adhering to the MP20 data schema.

    **Dataset Overview**
    - **Source**: Original data available at https://github.com/txie-93/cdvae/tree/main/data/mp_20
    ```
    ┌───────────────────┬─────────┬─────────┬─────────┐
    │ Dataset Partition │ Train   │ Val     │ Test    │
    ├───────────────────┼─────────┼─────────┼─────────┤
    │ Sample Count      │ 27,136  │ 9,047   │ 9,046   │
    └───────────────────┴─────────┴─────────┴─────────┘
    ```
    The dataset can also be downloaded from the following source: https://paddle-org.bj.bcebos.com/paddlematerial/datasets/mp_20/mp_20.zip

    **Data Format**
    The dataset is structured as a comma-separated values (CSV) file with the following
    columns:

    | Column Name               | Description                         | Example Value |
    |---------------------------|-------------------------------------|---------------|
    | `material_id`             | Unique material identifier          | mp-10009      |
    | `formation_energy_per_atom`| Formation energy per atom (eV/atom)| -0.57         |
    | `band_gap`                | Electronic band gap (eV)            | 0.89          |
    | `pretty_formula`          | Chemical formula in standard notation| GaTe         |
    | `e_above_hull`            | Energy above convex hull (eV/atom)  | 0.0           |
    | `cif`                     | CIF string                          | cif_str       |
    | ...                       | Other properties...                 | -             |

    **Example Row:**
    ```csv
    material_id,formation_energy_per_atom,band_gap,pretty_formula,e_above_hull,cif
    mp-10009,-0.57,0.89,GaTe,0.0,cif_str
    ```

    Args:
        path (str, optional): The path of the dataset, if path is not exists, it will
            be downloaded. Defaults to "./data/mp_20/train.csv".
        property_names (Optional[list[str]], optional): Property names you want to use
            for mp_20, the property_names should be selected from
            ["formation_energy_per_atom", "band_gap", "e_above_hull",
            "spacegroup.number"]. Defaults to None.
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
        overwrite (bool, optional): Overwrite the existing cache file at the given
            path if it already exists. Defaults to False.
        filter_unvalid (bool, optional): Whether to filter out unvalid samples. Defaults
            to True.
    """

    name = "mp_20"
    url = "https://paddle-org.bj.bcebos.com/paddlematerial/datasets/mp_20/mp_20.zip"
    md5 = "73371948155aa9da609436e291142d7e"

    def __init__(
        self,
        path: str = "./data/mp_20/train.csv",
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
        self.property_names = property_names if property_names is not None else []

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

        self.build_structure_cfg = build_structure_cfg
        self.build_graph_cfg = build_graph_cfg
        self.transforms = transforms

        if cache_path is not None:
            self.cache_path = cache_path
        else:
            # for example:
            # path = ./data/mp_20/train.csv
            # cache_path = ./data/mp_20_cache/train
            self.cache_path = osp.join(
                osp.split(path)[0] + "_cache", osp.splitext(osp.basename(path))[0]
            )
        logger.info(f"Cache path: {self.cache_path}")

        self.overwrite = overwrite
        self.filter_unvalid = filter_unvalid

        self.cache_exists = True if osp.exists(self.cache_path) else False
        self.row_data, self.num_samples = self.read_data(path)
        logger.info(f"Load {self.num_samples} samples from {path}")
        self.property_data = self.read_property_data(self.row_data, self.property_names)

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
                structures = BuildStructure(**build_structure_cfg)(self.row_data["cif"])
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
        data = pd.read_csv(path)
        logger.info(f"Read {len(data)} structures from {path}")
        data = {key: data[key].tolist() for key in data if "Unnamed" not in key}
        num_samples = 0
        for key in data:
            num_samples = max(num_samples, len(data[key]))
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
            for key in self.row_data.keys():
                self.row_data[key] = [self.row_data[key][i] for i in reserve_idx]
            self.structures = [self.structures[i] for i in reserve_idx]
            if self.graphs is not None:
                self.graphs = [self.graphs[i] for i in reserve_idx]
            logger.warning(
                f"Filter out {len(reserve_idx)} samples with valid properties: "
                f"{property_name}"
            )
        self.num_samples = len(self.structures)
        logger.warning(f"Remaining {self.num_samples} samples after filtering.")

    def read_property_data(self, data: Dict, property_names: list[str]):
        """Read the property data from the given data and property names.

        Args:
            data (Dict): Data that contains the property data.
            property_names (list[str]): Property names.
        """
        property_data = {
            property_name: data[property_name] for property_name in property_names
        }
        if "space_group" in property_names:
            property_data["space_group"] = self.space_group_to_int(
                property_data["space_group"]
            )
        return property_data

    def space_group_to_int(self, space_groups):
        def space_group_to_int_single(space_group):
            return SpaceGroup(space_group).int_number

        space_groups_int = p_map(space_group_to_int_single, space_groups)

        return space_groups_int

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
                if isinstance(self.property_data[property_name][idx], str):
                    data[property_name] = self.property_data[property_name][idx]
                else:
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


class MP20MatterGenDataset(MP20Dataset):
    """This class is a subclass of MP20Dataset that is specifically designed for
    the mp20 dataset used for mattergen.
    """

    name = "mp_20_mattergen"
    url = "https://paddle-org.bj.bcebos.com/paddlematerial/datasets/mp_20/mp_20_chemical_system.zip"
    md5 = "9ce0c42c5696d07b73a0bf4d4ceadb71"


class AlexMP20MatterGenDataset(MP20Dataset):
    """This class is a subclass of MP20Dataset that is specifically designed for
    the mp20 dataset used for mattergen.
    """

    name = "alex_mp_20"
    url = "https://paddle-org.bj.bcebos.com/paddlematerial/datasets/alex_mp_20/alex_mp_20.zip"
    md5 = "624361c17259cc3af63a00b29fffe9cd"

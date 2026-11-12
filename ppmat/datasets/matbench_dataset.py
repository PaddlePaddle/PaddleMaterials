# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

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

import json
import math
import os
import os.path as osp
import pickle
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional

import numpy as np
import paddle.distributed as dist
from paddle.io import Dataset
from pymatgen.core import Structure

from ppmat.datasets.build_structure import BuildStructure
from ppmat.datasets.cache_fingerprint import cache_matches_data
from ppmat.datasets.cache_fingerprint import compute_data_fingerprint
from ppmat.datasets.cache_fingerprint import mark_cache_rebuilding
from ppmat.datasets.cache_fingerprint import save_data_fingerprint
from ppmat.datasets.custom_data_type import ConcatData
from ppmat.models import build_graph_converter
from ppmat.utils import download
from ppmat.utils import logger
from ppmat.utils.misc import is_equal


class MatbenchDataset(Dataset):
    """Matbench Dataset Handler

    This class provides utilities for loading and processing the Matbench materials
    science benchmark datasets.
    The implementation supports loading multiple properties from different
    matbench JSON files and processing them for materials property prediction.

    **Dataset Overview**
    Matbench is a benchmark suite for materials property prediction containing multiple
    datasets with different properties:
    - Formation Energy (mp_e_form): ~132k samples
    - Band Gap (mp_gap): ~106k samples
    - Shear Modulus G (elasticity_log10(G_VRH)): ~11k samples
    - Bulk Modulus K (elasticity_log10(K_VRH)): ~11k samples

    **Automatic Download**
    If the data directory doesn't exist, the dataset will be
    automatically downloaded from:
    https://paddle-org.bj.bcebos.com/paddlematerial/datasets/matbench/matbench.zip

    **Data Format**
    Each matbench JSON file has the following structure:
    ```json
    {
        "index": [0, 1, 2, ...],
        "columns": ["structure", "property_name"],
        "data": [
            [structure_dict, property_value],
            [structure_dict, property_value],
            ...
        ]
    }
    ```

    **Property Mapping**
    - "e_form": Formation energy per atom (eV/atom) from mp_e_form.json
    - "gap pbe": Band gap (eV) from mp_gap.json
    - "log10(G_VRH)": Log10 of shear modulus (GPa) from elasticity_log10(G_VRH).json
    - "log10(K_VRH)": Log10 of bulk modulus (GPa) from elasticity_log10(K_VRH).json

    Args:
        path (str): Directory containing matbench JSON files.
            Defaults to "./data/matbench".
        property_names (Optional[List[str]]): Property names to load.
            Should be selected from
                ["e_form", "gap pbe", "log10(G_VRH)", "log10(K_VRH)"].
            Defaults to None (loads all available properties).
        build_structure_cfg (Dict, optional): Configs for building pymatgen structures.
            Defaults to None.
        build_graph_cfg (Dict, optional): Configs for building graphs from structures.
            Defaults to None.
        transforms (Optional[Callable], optional):
            Preprocessing transforms for each sample.
            Defaults to None.
        cache_path (Optional[str], optional):
            Path for caching processed structures and graphs.
            Defaults to None.
        overwrite (bool, optional): Whether to overwrite existing cache files.
            Defaults to False.
        filter_unvalid (bool, optional): Whether to filter out invalid samples.
            Defaults to True.
        max_samples (Optional[int], optional): Maximum number of samples to load.
            Defaults to None (load all).
    """

    # Dataset download information
    name = "matbench"
    url = (
        "https://paddle-org.bj.bcebos.com/paddlematerial/datasets/matbench/matbench.zip"
    )
    # TODO: Replace with actual MD5 hash when available
    md5 = "71e85300825604c2e228cbbf75574906"

    # Property file mapping
    PROPERTY_FILES = {
        "e_form": "mp_e_form.json",
        "gap pbe": "mp_gap.json",
        "log10(G_VRH)": "elasticity_log10(G_VRH).json",
        "log10(K_VRH)": "elasticity_log10(K_VRH).json",
    }

    def __init__(
        self,
        path: str = "./data/matbench",
        property_names: Optional[List[str]] = None,
        build_structure_cfg: Dict = None,
        build_graph_cfg: Dict = None,
        transforms: Optional[Callable] = None,
        cache_path: Optional[str] = None,
        overwrite: bool = False,
        filter_unvalid: bool = True,
        max_samples: Optional[int] = None,
        **kwargs,  # for compatibility
    ):
        super().__init__()

        # Check if data directory and required files exist, if not download the dataset
        # This follows the same pattern as MP2018Dataset
        if not osp.exists(path):
            logger.message("The matbench dataset is not found. Will download it now.")
            root_path = download.get_datasets_path_from_url(self.url, self.md5)
            path = osp.join(root_path, self.name)
        else:
            # Check if required files exist in the directory
            required_files = list(self.PROPERTY_FILES.values())
            files_exist = all(osp.exists(osp.join(path, f)) for f in required_files)

            if not files_exist:
                logger.message(
                    "Some matbench data files are missing. "
                    "Will download the dataset now."
                )
                root_path = download.get_datasets_path_from_url(self.url, self.md5)
                path = osp.join(root_path, self.name)

        self.path = path
        if isinstance(property_names, str):
            property_names = [property_names]

        # Default to all available properties if none specified
        if property_names is None:
            property_names = list(self.PROPERTY_FILES.keys())

        # Validate property names
        for prop in property_names:
            if prop not in self.PROPERTY_FILES:
                raise ValueError(
                    f"Unknown property '{prop}'. "
                    f"Available properties: {list(self.PROPERTY_FILES.keys())}"
                )

        self.property_names = property_names
        self.max_samples = max_samples

        if build_structure_cfg is None:
            build_structure_cfg = {
                "format": "structure",  # Already Structure objects
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
            # Generate cache path based on data directory and properties
            prop_str = "_".join(sorted(property_names))
            self.cache_path = osp.join(path + "_cache", f"matbench_{prop_str}")
        logger.info(f"Cache path: {self.cache_path}")

        self.overwrite = overwrite
        self.filter_unvalid = filter_unvalid

        self.cache_exists = True if osp.exists(self.cache_path) else False

        # Load data from matbench JSON files
        self.raw_data, self.num_samples = self.load_matbench_data()
        logger.info(f"Load {self.num_samples} samples from matbench datasets")

        # Extract property data
        self.property_data = self.extract_property_data(
            self.raw_data, self.property_names
        )

        # Handle cache and structure/graph processing (similar to MP2018Dataset)
        self._setup_cache_and_processing()

    def load_matbench_data(self):
        """Load data from matbench JSON files."""
        if len(self.property_names) == 1:
            # Single property case - load from one file
            return self._load_single_property_data()
        else:
            # Multiple properties case - need to handle differently
            raise NotImplementedError(
                "Loading multiple properties from different files is not implemented. "
                "Please specify only one property at a time."
            )

    def _load_single_property_data(self):
        """Load data for a single property from its matbench JSON file."""
        prop_name = self.property_names[0]
        file_path = osp.join(self.path, self.PROPERTY_FILES[prop_name])

        if not osp.exists(file_path):
            raise FileNotFoundError(f"Matbench file not found: {file_path}")

        logger.info(f"Loading {prop_name} from {file_path}")

        with open(file_path, "r") as f:
            data = json.load(f)

        # If max_samples is not None, keep only the specified number of samples
        if self.max_samples is not None:
            data_samples = {
                "index": data["index"][: self.max_samples],
                "columns": data["columns"],
                "data": data["data"][: self.max_samples],
            }
            data = data_samples

        # Validate data format
        if not all(key in data for key in ["index", "columns", "data"]):
            raise ValueError(f"Invalid matbench file format: {file_path}")

        # Verify columns
        expected_columns = [
            "structure",
            data["columns"][1],
        ]  # Second column is property name
        if data["columns"] != expected_columns:
            logger.warning(
                f"Expected columns {expected_columns}, got {data['columns']}"
            )

        # Extract structures and properties
        structures = []
        properties = []

        for structure_dict, prop_value in data["data"]:
            # Convert structure dict to pymatgen Structure
            try:
                structure = Structure.from_dict(structure_dict)
                structures.append(structure)
                properties.append(prop_value)
            except Exception as e:
                logger.warning(f"Failed to parse structure in {file_path}: {e}")
                if not self.filter_unvalid:
                    structures.append(None)
                    properties.append(None)

        logger.info(f"Loaded {len(structures)} samples for {prop_name}")

        raw_data = {"structures": structures, "properties": {prop_name: properties}}

        num_samples = len(structures)
        return raw_data, num_samples

    def extract_property_data(self, raw_data, property_names):
        """Extract property data from raw data."""
        property_data = {}
        for prop_name in property_names:
            if prop_name in raw_data["properties"]:
                # Take only the first N samples to match structure count
                n_samples = len(raw_data["structures"])
                property_data[prop_name] = raw_data["properties"][prop_name][:n_samples]
            else:
                raise KeyError(f"Property {prop_name} not found in raw data")
        return property_data

    def _setup_cache_and_processing(self):
        """Setup cache and handle structure/graph processing."""
        # Check cache configuration consistency (similar to MP2018Dataset)
        if self.cache_exists and not self.overwrite:
            logger.warning(
                "Cache enabled. If a cache file exists, it will be automatically "
                "read and current settings will be ignored. Please ensure that the "
                "settings used match your current settings."
            )
            try:
                build_structure_cfg_cache = self.load_from_cache(
                    osp.join(self.cache_path, "build_structure_cfg.pkl")
                )
                if is_equal(build_structure_cfg_cache, self.build_structure_cfg):
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
                    self.overwrite = True
            except Exception as e:
                logger.warning(e)
                logger.warning(
                    "Failed to load build_structure_cfg.pkl from cache. "
                    "Will rebuild the structures and graphs(if need)."
                )
                self.overwrite = True

            if self.build_graph_cfg is not None and not self.overwrite:
                try:
                    build_graph_cfg_cache = self.load_from_cache(
                        osp.join(self.cache_path, "build_graph_cfg.pkl")
                    )
                    if is_equal(build_graph_cfg_cache, self.build_graph_cfg):
                        logger.info(
                            "The cached build_graph_cfg configuration "
                            "matches the current settings. Reusing previously "
                            "generated graph data to optimize performance."
                        )
                    else:
                        logger.warning(
                            "build_graph_cfg is different from build_graph_cfg_cache"
                            ". Will rebuild the graphs."
                        )
                        self.overwrite = True
                except Exception as e:
                    logger.warning(e)
                    logger.warning(
                        "Failed to load build_graph_cfg.pkl from cache. "
                        "Will rebuild the graphs."
                    )
                    self.overwrite = True

        structure_cache_path = osp.join(self.cache_path, "structures")
        graph_cache_path = osp.join(self.cache_path, "graphs")

        # The build settings alone do not identify the data: also check that the
        # cache was built from the current input before reusing it.
        data_fingerprint = compute_data_fingerprint(self.raw_data["structures"])
        if self.cache_exists and not self.overwrite:
            if not cache_matches_data(
                self.cache_path,
                data_fingerprint,
                ["structures"]
                + (["graphs"] if self.build_graph_cfg is not None else []),
            ):
                self.overwrite = True
        if self.overwrite or not self.cache_exists:
            # Convert structures and graphs (only rank 0 process)
            if dist.get_rank() == 0:
                # Save build configs to cache
                os.makedirs(self.cache_path, exist_ok=True)
                mark_cache_rebuilding(self.cache_path)
                self.save_to_cache(
                    osp.join(self.cache_path, "build_structure_cfg.pkl"),
                    self.build_structure_cfg,
                )
                self.save_to_cache(
                    osp.join(self.cache_path, "build_graph_cfg.pkl"),
                    self.build_graph_cfg,
                )

                # Process structures
                if self.build_structure_cfg["format"] == "structure":
                    # Structures are already pymatgen Structure objects
                    structures = self.raw_data["structures"]
                else:
                    # Convert structures using BuildStructure
                    structures = BuildStructure(**self.build_structure_cfg)(
                        self.raw_data["structures"]
                    )

                # Save structures to cache
                os.makedirs(structure_cache_path, exist_ok=True)
                for i in range(self.num_samples):
                    self.save_to_cache(
                        osp.join(structure_cache_path, f"{i:010d}.pkl"),
                        structures[i],
                    )
                logger.info(
                    f"Save {self.num_samples} structures to {structure_cache_path}"
                )

                # Process graphs if needed
                if self.build_graph_cfg is not None:
                    converter = build_graph_converter(self.build_graph_cfg)
                    graphs = converter(structures)
                    # Save graphs to cache
                    os.makedirs(graph_cache_path, exist_ok=True)
                    for i in range(self.num_samples):
                        self.save_to_cache(
                            osp.join(graph_cache_path, f"{i:010d}.pkl"), graphs[i]
                        )
                    logger.info(f"Save {self.num_samples} graphs to {graph_cache_path}")

            # Sync all processes
            if dist.is_initialized():
                dist.barrier()
        if dist.get_rank() == 0:
            save_data_fingerprint(self.cache_path, data_fingerprint)

        # Set up structure and graph paths
        self.structures = [
            osp.join(structure_cache_path, f"{i:010d}.pkl")
            for i in range(self.num_samples)
        ]
        if self.build_graph_cfg is not None:
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

        # Filter by property data if needed
        if self.filter_unvalid:
            self.filter_unvalid_by_property()

    def filter_unvalid_by_property(self):
        """Filter out samples with invalid property values."""
        for property_name in self.property_names:
            data = self.property_data[property_name]
            reserve_idx = []
            for i, data_item in enumerate(data):
                if isinstance(data_item, str) or (
                    data_item is not None and not math.isnan(data_item)
                ):
                    reserve_idx.append(i)

            # Update all data structures to keep only valid samples
            for key in self.property_data.keys():
                self.property_data[key] = [
                    self.property_data[key][i] for i in reserve_idx
                ]

            self.raw_data["structures"] = [
                self.raw_data["structures"][i] for i in reserve_idx
            ]
            self.structures = [self.structures[i] for i in reserve_idx]
            if self.graphs is not None:
                self.graphs = [self.graphs[i] for i in reserve_idx]
            logger.warning(
                f"Filter out {len(reserve_idx)} samples with valid properties: "
                f"{property_name}"
            )
        self.num_samples = len(self.raw_data["structures"])
        logger.warning(f"Remaining {self.num_samples} samples after filtering.")

    def save_to_cache(self, cache_path: str, data: Any):
        """Save data to cache file."""
        with open(cache_path, "wb") as f:
            pickle.dump(data, f)

    def load_from_cache(self, cache_path: str):
        """Load data from cache file."""
        if osp.exists(cache_path):
            with open(cache_path, "rb") as f:
                data = pickle.load(f)
            return data
        else:
            raise FileNotFoundError(f"No such file or directory: {cache_path}")

    def get_structure_array(self, structure):
        """Convert pymatgen Structure to array format."""
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

        # Add property data
        for property_name in self.property_names:
            if property_name in self.property_data:
                data[property_name] = np.array(
                    [self.property_data[property_name][idx]]
                ).astype("float32")
            else:
                raise KeyError(f"Property {property_name} not found.")

        data["id"] = idx
        data = self.transforms(data) if self.transforms is not None else data

        return data

    def __len__(self):
        return self.num_samples

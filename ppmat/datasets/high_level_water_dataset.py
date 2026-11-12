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

from __future__ import annotations

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
from ase import Atoms
from ase.io import read as ase_read
from ase.units import GPa
from paddle.io import Dataset
from pymatgen.io.ase import AseAtomsAdaptor

from ppmat.datasets.build_structure import BuildStructure
from ppmat.datasets.cache_fingerprint import cache_matches_data
from ppmat.datasets.cache_fingerprint import compute_data_fingerprint
from ppmat.datasets.cache_fingerprint import mark_cache_rebuilding
from ppmat.datasets.cache_fingerprint import save_data_fingerprint
from ppmat.datasets.custom_data_type import ConcatData
from ppmat.datasets.custom_data_type import ConcatNumpyWarper
from ppmat.models import build_graph_converter
from ppmat.utils import download
from ppmat.utils import logger
from ppmat.utils.misc import is_equal


class HighLevelWaterDataset(Dataset):
    """The high level water dataset, this is toy dataset from mattersim: https://github.com/microsoft/mattersim

    Args:
        path (str): File path to the dataset file.
        force_key (Optional[str], optional): Key used to retrieve force values when
            returning properties. If None, force values won't be included in results.
            Defaults to 'force'.
        stress_key (Optional[str], optional): Key used to retrieve stress values when
            returning properties. If None, stress values won't be included in results.
            Defaults to None.
        build_structure_cfg (Dict, optional): The configs for building the structure.
            Defaults to None.
        build_graph_cfg (Dict, optional): The configs for building the graph. Defaults
            to None.
        transforms (Optional[Callable], optional): The preprocess transforms for each
            sample. Defaults to None.
        cache_path (Optional[str], optional): If a cache_path is set, structures and
            graph will be read directly from this path; if the cache does not exist,
            the converted structures and graph will be saved to this path. Defaults
            to None.
        overwrite (bool, optional): Overwrite the existing cache file at the given
            path if it already exists. Defaults to False.
        filter_unvalid (bool, optional): Whether to filter out invalid samples. Defaults
            to True.
    """

    name = "high_level_water"
    url = "https://paddle-org.bj.bcebos.com/paddlematerial/datasets/high_level_water/high_level_water.zip"
    md5 = "4bf53db054e6e8fb55a38f7cbdf6fea6"

    def __init__(
        self,
        path: str,  # ./data/high_level_water/high_level_water.xyz
        energy_key: Optional[str] = "energy",
        force_key: Optional[str] = "force",
        stress_key: Optional[str] = None,
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
        self.energy_key = energy_key
        self.force_key = force_key
        self.stress_key = stress_key

        self.property_names = []
        if energy_key is not None:
            self.property_names.append(energy_key)
        if force_key is not None:
            self.property_names.append(force_key)
        if stress_key is not None:
            self.property_names.append(stress_key)

        if build_structure_cfg is None:
            build_structure_cfg = {
                "format": "ase_atoms",
                "primitive": False,
                "niggli": False,
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
            # path = ./data/high_level_water_dataset/high_level_water_dataset.xyz
            # cache_path= ./data/high_level_water_dataset_cache/high_level_water_dataset
            self.cache_path = osp.join(
                osp.split(path)[0] + "_cache", osp.splitext(osp.basename(path))[0]
            )
        logger.info(f"Cache path: {self.cache_path}")

        self.overwrite = overwrite
        self.filter_unvalid = filter_unvalid

        self.cache_exists = True if osp.exists(self.cache_path) else False
        self.row_data, self.num_samples = self.read_data(path)
        logger.info(f"Load {self.num_samples} samples from {path}")
        self.property_data = self.read_property_data(self.row_data)

        structure_cache_path = osp.join(self.cache_path, "structures")
        graph_cache_path = osp.join(self.cache_path, "graphs")

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

        # The build settings alone do not identify the data: also check that the
        # cache was built from the current input before reusing it.
        data_fingerprint = compute_data_fingerprint(self.row_data)
        if self.cache_exists and not overwrite:
            if not cache_matches_data(
                self.cache_path,
                data_fingerprint,
                ["structures"] + (["graphs"] if build_graph_cfg is not None else []),
            ):
                overwrite = True
        if overwrite or not self.cache_exists:
            # convert strucutes and graphs
            # only rank 0 process do the conversion
            if dist.get_rank() == 0:
                # save build_structure_cfg and build_graph_cfg to cache file
                os.makedirs(self.cache_path, exist_ok=True)
                mark_cache_rebuilding(self.cache_path)
                self.save_to_cache(
                    osp.join(self.cache_path, "build_structure_cfg.pkl"),
                    build_structure_cfg,
                )
                self.save_to_cache(
                    osp.join(self.cache_path, "build_graph_cfg.pkl"), build_graph_cfg
                )
                # convert strucutes
                structures = BuildStructure(**build_structure_cfg)(self.row_data)
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
        if dist.get_rank() == 0:
            save_data_fingerprint(self.cache_path, data_fingerprint)
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

    def read_data(self, path: str, format: str = None):
        """Read the data from the given json path.

        Args:
            path (str): Path to the data.
        """
        if format:
            atoms_list = ase_read(path, index=":", format=format)
        else:
            try:
                atoms_list = ase_read(path, index=":")
            except Exception as e:
                raise ValueError(f"Can not automately guess the file format: {e}")
        num_samples = len(atoms_list)
        return atoms_list, num_samples

    def atoms_to_structure(self, atoms: Atoms):
        return AseAtomsAdaptor().get_structure(atoms)

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

            self.row_data = [self.row_data[i] for i in reserve_idx]
            self.structures = [self.structures[i] for i in reserve_idx]
            if self.graphs is not None:
                self.graphs = [self.graphs[i] for i in reserve_idx]
            logger.warning(
                f"Filter out {len(reserve_idx)} samples with valid properties: "
                f"{property_name}"
            )
        self.num_samples = len(self.row_data)
        logger.warning(f"Remaining {self.num_samples} samples after filtering.")

    def read_property_data(self, data: List[Atoms]):
        """Read the property data from the given data and property names.

        Args:
            data (Dict): Data that contains the property data.
        """
        property_data = {}
        if self.energy_key is not None:
            property_data[self.energy_key] = [
                data[i].get_potential_energy() for i in range(self.num_samples)
            ]
        if self.force_key is not None:
            property_data[self.force_key] = [
                data[i].get_forces() for i in range(self.num_samples)
            ]
        if self.stress_key is not None:
            property_data[self.stress_key] = [
                data[i].get_stress(voigt=False) / GPa for i in range(self.num_samples)
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
            if property_name == self.force_key:
                data[property_name] = ConcatNumpyWarper(
                    self.property_data[property_name][idx]
                ).astype("float32")
            elif property_name in self.property_data:
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

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
from ppmat.datasets.custom_data_type import ConcatNumpyWarper
from ppmat.models import build_graph_converter
from ppmat.utils import download
from ppmat.utils import logger
from ppmat.utils.io import read_json
from ppmat.utils.misc import is_equal


class MPTrjDataset(Dataset):
    """Dataset class for loading and processing the Materials Project Trajectory (MPtrj)
    dataset used in CHGNet.

    https://figshare.com/articles/dataset/Materials_Project_Trjectory_MPtrj_Dataset/23713842?file=41619375

    This dataset contains 1,580,395 structures with corresponding:
    - 1,580,395 energies
    - 7,944,833 magnetic moments
    - 49,295,660 forces
    - 14,223,555 stresses

    All data originates from GGA/GGA+U static/relaxation trajectories in the 2022.9
    Materials Project release. The dataset employs a selection protocol that excludes
    incompatible calculations and duplicate structures.

    Data Structure:
    The JSON file follows this hierarchical structure:
    {
        "mp-id-0": {
            "frame-id-0": {
                "structure": <pymatgen.Structure dict>,
                "uncorrected_total_energy": float,  # Raw VASP output [eV]
                "corrected_total_energy": float,    # VASP total energy after MP2020
                                                    # compatibility [eV]
                "energy_per_atom": float,           # Corrected energy per atom, this
                                                    # is the energy label used to train
                                                    # CHGNet [eV/atom]
                "ef_per_atom": float,               # Formation energy [eV/atom]
                "e_per_atom_relaxed": float,        # Corrected energy per atom of the
                                                    # relaxed structure, this is the
                                                    # energy you can find for the mp-id
                                                    # on materials project website
                                                    #[eV/atom]
                "ef_per_atom_relaxed": float,       # Relaxed formation energy [eV/atom]
                "force": List[float],               # Atomic forces [eV/Å]
                "stress": List[float],              # Stress tensor [kBar]
                "magmom": List[float] or None,      # Magmom on the atoms [μB]
                "bandgap": float                    # Bandgap [eV]
            },
            "frame-id-1": {...},
            ...
        },
        "mp-id-1": {...},
        ...

    }

    Notes:
    1. Frame ID Format:
       'task_id-calc_id-ionic_step' where:
       - calc_id: 0 (second relaxation) or 1 (first relaxation) in double relaxation
       workflows

    2. Energy Compatibility:
       MP2020 corrections are applied to unify GGA/GGA+U energy scales.
       The 'energy_per_atom' field contains these corrected values used for CHGNet
       training. See pymatgen compatibility documentation:
       https://pymatgen.org/pymatgen.entries.html#pymatgen.entries.compatibility.Compatibility

    3. Magnetic Moment Handling:
       Missing MAGMOM values are represented as None (not zero).
       CHGNet uses absolute DFT magmom values directly from this dataset.
       Unit conversion is handled automatically when using the provided dataset loader.
       Reference implementation:
       https://github.com/CederGroupHub/chgnet/blob/main/chgnet/data/dataset.py

    4. Stress Units:
       VASP raw stress values (kBar) are converted to GPa for CHGNet using:
       stress_gpa = -0.1 * vasp_stress_kbar
       This conversion is automatically applied when loading the dataset.


    Args:
        path (str, optional): The path of the dataset, if path is not exists, it will
            be downloaded.
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

    name = "MPtrj_2022.9_full"
    url = "https://paddle-org.bj.bcebos.com/paddlematerial/datasets/mptrj/MPtrj_2022.9_full.zip"
    md5 = "949069910f4ce1f1ed8c49a8d6ae5c5e"

    def __init__(
        self,
        path: str,
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
                "format": "dict",
                "primitive": False,
                "niggli": False,
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
            # path = ./data/MPtrj_2022.9_full/train.json
            # cache_path = ./data/MPtrj_2022.9_full/train
            self.cache_path = osp.join(
                osp.split(path)[0] + "_cache", osp.splitext(osp.basename(path))[0]
            )
        logger.info(f"Cache path: {self.cache_path}")

        self.overwrite = overwrite
        self.filter_unvalid = filter_unvalid

        self.cache_exists = True if osp.exists(self.cache_path) else False
        self.row_data = self.read_data(path)
        self.keys = [
            (mp_id, graph_id)
            for mp_id, dct in self.row_data.items()
            for graph_id in dct
        ]
        self.num_samples = len(self.keys)
        logger.info(f"Load {self.num_samples} samples from {path}")

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
                structure_str = []
                for mp_id, graph_ids in self.keys:
                    structure_str.append(self.row_data[mp_id][graph_ids]["structure"])
                structures = BuildStructure(**build_structure_cfg)(structure_str)
                # save structures to cache file
                os.makedirs(structure_cache_path, exist_ok=True)
                for i, (mp_id, graph_ids) in enumerate(self.keys):
                    self.save_to_cache(
                        osp.join(structure_cache_path, f"{mp_id}_{graph_ids}.pkl"),
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
                    for i, (mp_id, graph_ids) in enumerate(self.keys):
                        self.save_to_cache(
                            osp.join(graph_cache_path, f"{mp_id}_{graph_ids}.pkl"),
                            graphs[i],
                        )
                    logger.info(f"Save {self.num_samples} graphs to {graph_cache_path}")

            # sync all processes
            if dist.is_initialized():
                dist.barrier()
        self.structures = defaultdict(dict)
        for i, (mp_id, graph_ids) in enumerate(self.keys):
            self.structures[mp_id][graph_ids] = osp.join(
                structure_cache_path, f"{mp_id}_{graph_ids}.pkl"
            )
        if build_graph_cfg is not None:
            self.graphs = defaultdict(dict)
            for mp_id, graph_ids in self.keys:
                self.graphs[mp_id][graph_ids] = osp.join(
                    graph_cache_path, f"{mp_id}_{graph_ids}.pkl"
                )
        else:
            self.graphs = None

        # filter by property data, since some samples may have no valid properties
        if filter_unvalid:
            self.filter_unvalid_by_property()

    def read_data(self, path: str):
        """Read the data from the given json path.

        Args:
            path (str): Path to the data.
        """
        json_data = read_json(path)
        return json_data

    def filter_unvalid_by_property(self):
        for property_name in self.property_names:
            reserve_idx = []
            delete_id = []
            for i, (mp_id, graph_ids) in enumerate(self.keys):
                property_value = self.row_data[mp_id][graph_ids][property_name]
                if isinstance(property_value, str) or (
                    property_value is not None and not math.isnan(property_value)
                ):
                    reserve_idx.append(i)
                else:
                    delete_id.append([mp_id, graph_ids])

            self.keys = [self.keys[i] for i in reserve_idx]
            for mp_id, graph_ids in delete_id:
                del self.row_data[mp_id][graph_ids]

                del self.structures[mp_id][graph_ids]
                if self.graphs is not None:
                    del self.graphs[mp_id][graph_ids]
            logger.warning(
                f"Filter out {len(reserve_idx)} samples with valid properties: "
                f"{property_name}"
            )
        self.num_samples = len(self.keys)
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
        mp_id, graph_id = self.keys[idx]
        data = {}
        # get graph
        if self.graphs is not None:
            graph = self.graphs[mp_id][graph_id]
            if isinstance(graph, str):
                graph = self.load_from_cache(graph)
            data["graph"] = graph
        else:
            structure = self.structures[mp_id][graph_id]
            if isinstance(structure, str):
                structure = self.load_from_cache(structure)
            data["structure_array"] = self.get_structure_array(structure)

        row_data = self.row_data[mp_id][graph_id]
        for property_name in self.property_names:
            if property_name in row_data.keys():
                value = row_data[property_name]
                if isinstance(value, str):
                    data[property_name] = value
                elif property_name in ["force", "stress", "magmom"]:
                    num_atoms = (
                        data["graph"].node_feat["num_atoms"]
                        if "graph" in data.keys()
                        else data["structure_array"]["num_atoms"].data
                    )
                    num_atoms = num_atoms[0]
                    if value is None:
                        if property_name == "force":
                            value = np.full((num_atoms, 3), np.nan)
                        elif property_name == "stress":
                            value = np.full((3, 3), np.nan)
                        elif property_name == "magmom":
                            value = np.full((num_atoms,), np.nan)

                    if property_name == "stress":
                        value = [value]
                    elif property_name == "magmom":
                        value = np.abs(np.array(value).reshape([-1, 1]))
                    data[property_name] = ConcatNumpyWarper(value).astype("float32")
                else:
                    data[property_name] = np.array([value]).astype("float32")
            else:
                raise KeyError(f"Property {property_name} not found.")

        data["id"] = idx
        data = self.transforms(data) if self.transforms is not None else data

        return data

    def __len__(self):
        return self.num_samples

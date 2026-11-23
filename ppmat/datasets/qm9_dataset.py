# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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
import pandas as pd
import paddle.distributed as dist
from paddle.io import Dataset
from ppmat.datasets.build_molecule import BuildMolecule
from ppmat.models import build_graph_converter
from ppmat.utils import download
from ppmat.utils import logger
from ppmat.utils.misc import is_equal


class QM9Dataset(Dataset):
    """QM9 Dataset Handler

    This class provides utilities for loading and processing the QM9 molecular
    quantum chemistry dataset. The implementation supports both standard dataset
    loading and custom data processing when adhering to the QM9 data schema.

    **Dataset Overview**
    - **Source**: Original data from "Quantum chemistry structures and properties
      of 134 kilo molecules" and the QM9/GDB9 dataset.
    - **Filtering**: The 3,054 uncharacterized molecules listed by the official
      QM9 consistency check are removed.
    ```
    ┌───────────────────┬─────────┬─────────┬─────────┐
    │ Dataset Partition │ Train   │ Val     │ Test    │
    ├───────────────────┼─────────┼─────────┼─────────┤
    │ Sample Count      │ 110,000 │ 10,000  │ 10,831  │
    └───────────────────┴─────────┴─────────┴─────────┘
    ```
    The dataset can also be downloaded from the following source:
    https://paddle-org.bj.bcebos.com/paddlematerials/datasets/qm9/qm9.zip

    **Data Format**
    The dataset is structured as comma-separated values (CSV) files with one
    molecule per row. The split files are `train.csv`, `val.csv`, and `test.csv`.

    | Column Name                | Description                                      | Example Value  |
    |----------------------------|--------------------------------------------------|----------------|
    | `file_name`                | Original QM9 xyz file name                       | qm9_000001.xyz |
    | `standard_xyz`             | Standard XYZ string without Mulliken charges     | xyz_str        |
    | `mulliken_xyz`             | XYZ string with Mulliken partial charges         | xyz_charge_str |
    | `molecule_id`              | 1-based QM9 molecule identifier                  | 1              |
    | `num_atoms`                | Number of atoms in the molecule                  | 5              |
    | `A`, `B`, `C`              | Rotational constants                             | 157.7118       |
    | `mu`                       | Dipole moment                                    | 0.0            |
    | `alpha`                    | Isotropic polarizability                         | 13.21          |
    | `homo`, `lumo`, `gap`      | HOMO, LUMO, and HOMO-LUMO gap                    | -0.3877        |
    | `r2`                       | Electronic spatial extent                        | 35.3641        |
    | `zpve`                     | Zero point vibrational energy                    | 0.044749       |
    | `U0`, `U`, `H`, `G`        | Thermochemical energies                          | -40.47893      |
    | `Cv`                       | Heat capacity                                    | 6.469          |
    | `vibrational_frequencies`  | Space-separated vibrational frequencies          | 1341.307 ...   |
    | `canonical_smiles`         | Canonical SMILES string                          | C              |
    | `isomeric_smiles`          | Isomeric SMILES string                           | C              |
    | `canonical_inchi`          | Canonical InChI string                           | InChI=1S/CH4/h1H4 |
    | `isomeric_inchi`           | Isomeric InChI string                            | InChI=1S/CH4/h1H4 |

    **Example Row:**
    ```csv
    file_name,molecule_id,num_atoms,A,B,C,mu,alpha,homo,lumo,gap,...
    qm9_000001.xyz,1,5,157.7118,157.70997,157.70699,0.0,13.21,-0.3877,0.1171,0.5048,...
    ```

    Args:
        path (str, optional): The path of the dataset. If the path does not exist,
            it will be downloaded. Defaults to "./data/qm9/train.csv".
        property_names (Optional[list[str]], optional): Property names to use for
            QM9. The property_names should be selected from
            ["A", "B", "C", "mu", "alpha", "homo", "lumo", "gap", "r2",
            "zpve", "U0", "U", "H", "G", "Cv"]. Defaults to None.
        build_molecule_cfg (Dict, optional): Configs for building molecular
            structures from xyz strings. If not specified, the default setting
            will be used. Defaults to None.
        build_graph_cfg (Dict, optional): Configs for building molecular graphs
            from structures. Defaults to None.
        transforms (Optional[Callable], optional): Preprocess transforms for each
            sample. Defaults to None.
        cache_path (Optional[str], optional): If a cache_path is set, parsed
            molecules and graphs will be read directly from this path; if the
            cache does not exist, the converted molecules and graphs will be
            saved to this path. Defaults to None.
        overwrite (bool, optional): Overwrite the existing cache file at the given
            path if it already exists. Defaults to False.
        filter_unvalid (bool, optional): Whether to filter out invalid samples.
            Defaults to True.
    """

    name = "qm9"
    url = "https://paddle-org.bj.bcebos.com/paddlematerials/datasets/qm9/qm9.zip"
    md5 = "a70eb6cc913427db1fc1f8d3631fe00a"

    def __init__(
        self,
        path: str = "./data/qm9/train.csv",
        property_names: Optional[list[str]] = None,
        build_molecule_cfg: Dict = None,
        build_graph_cfg: Dict = None,
        transforms: Optional[Callable] = None,
        cache_path: Optional[str] = None,
        overwrite: bool = False,
        filter_unvalid: bool = True,
        **kwargs,
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

        if build_molecule_cfg is None:
            build_molecule_cfg = {
                "format": "xyz_block",
                "sanitize": False,
                "add_hs": False,
                "remove_hs": False,
                "kekulize": False,
                "num_cpus": 1,
            }
            logger.message(
                "The build_molecule_cfg is not set, will use the default "
                f"configs: {build_molecule_cfg}"
            )

        self.build_molecule_cfg = build_molecule_cfg
        self.build_graph_cfg = build_graph_cfg
        self.transforms = transforms

        if cache_path is not None:
            self.cache_path = cache_path
        else:
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
                build_molecule_cfg_cache = self.load_from_cache(
                    osp.join(self.cache_path, "build_molecule_cfg.pkl")
                )
                if is_equal(build_molecule_cfg_cache, build_molecule_cfg):
                    logger.info(
                        "The cached build_molecule_cfg configuration matches "
                        "the current settings. Reusing previously generated"
                        " molecular data to optimize performance."
                    )
                else:
                    logger.warning(
                        "build_molecule_cfg is different from "
                        "build_molecule_cfg_cache. Will rebuild the molecules and "
                        "graphs."
                    )
                    logger.warning(
                        "If you want to use the cached molecules and graphs, please "
                        "ensure that the settings used in match your current settings."
                    )
                    overwrite = True
            except Exception as e:
                logger.warning(e)
                logger.warning(
                    "Failed to load builded_molecules_cfg.pkl from cache. "
                    "Will rebuild the molecules and graphs(if need)."
                )
                overwrite = True

            if build_graph_cfg is not None and not overwrite:
                try:
                    build_graph_cfg_cache = self.load_from_cache(
                        osp.join(self.cache_path, "build_graph_cfg.pkl")
                    )
                    if is_equal(build_graph_cfg_cache, build_graph_cfg):
                        logger.info(
                            "The cached build_molecule_cfg configuration "
                            "matches the current settings. Reusing previously "
                            "generated molecular data to optimize performance."
                        )
                    else:
                        logger.warning(
                            "build_graph_cfg is different from build_graph_cfg_cache"
                            ". Will rebuild the graphs."
                        )
                        logger.warning(
                            "If you want to use the cached molecules and graphs, "
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

        molecule_cache_path = osp.join(self.cache_path, "molecules")
        graph_cache_path = osp.join(self.cache_path, "graphs")
        if overwrite or not self.cache_exists:
            if dist.get_rank() == 0:
                os.makedirs(self.cache_path, exist_ok=True)
                self.save_to_cache(
                    osp.join(self.cache_path, "build_molecule_cfg.pkl"),
                    build_molecule_cfg,
                )
                self.save_to_cache(
                    osp.join(self.cache_path, "build_graph_cfg.pkl"), build_graph_cfg
                )

                molecules = BuildMolecule(**build_molecule_cfg)(
                    self.raw_data["molecule"]
                )
                os.makedirs(molecule_cache_path, exist_ok=True)
                for i in range(self.num_samples):
                    self.save_to_cache(
                        osp.join(molecule_cache_path, f"{i:010d}.pkl"), molecules[i]
                    )
                logger.info(
                    f"Save {self.num_samples} molecules to {molecule_cache_path}"
                )

                if build_graph_cfg is not None:
                    converter = build_graph_converter(build_graph_cfg)
                    graphs = converter(molecules)
                    os.makedirs(graph_cache_path, exist_ok=True)
                    for i in range(self.num_samples):
                        self.save_to_cache(
                            osp.join(graph_cache_path, f"{i:010d}.pkl"), graphs[i]
                        )
                    logger.info(f"Save {self.num_samples} graphs to {graph_cache_path}")

            if dist.is_initialized():
                dist.barrier()

        self.molecules = [
            osp.join(molecule_cache_path, f"{i:010d}.pkl")
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
            len(self.molecules) == self.num_samples
        ), "The number of molecules must be equal to the number of samples."
        assert (
            self.graphs is None or len(self.graphs) == self.num_samples
        ), "The number of graphs must be equal to the number of samples."

        if filter_unvalid:
            self.filter_unvalid_by_property()

    def read_data(self, path: str):
        """Read the data from the given csv path."""
        data = pd.read_csv(path)
        logger.info(f"Read {len(data)} molecules from {path}")

        data = {key: data[key].tolist() for key in data if "Unnamed" not in key}
        data["molecule"] = data.pop("standard_xyz") # adapted for this qm9 split file
        data["id"] = data["molecule_id"] # adapted for this qm9 split file
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
            for key in self.raw_data.keys():
                self.raw_data[key] = [self.raw_data[key][i] for i in reserve_idx]
            self.molecules = [self.molecules[i] for i in reserve_idx]
            if self.graphs is not None:
                self.graphs = [self.graphs[i] for i in reserve_idx]
            logger.warning(
                f"Filter out {len(reserve_idx)} samples with valid properties: "
                f"{property_name}"
            )
        self.num_samples = len(self.molecules)
        logger.warning(f"Remaining {self.num_samples} samples after filtering.")

    def read_property_data(self, data: Dict, property_names: list[str]):
        property_data = {
            property_name: data[property_name] for property_name in property_names
        }
        return property_data

    def save_to_cache(self, cache_path: str, data: Any):
        with open(cache_path, "wb") as f:
            pickle.dump(data, f)

    def load_from_cache(self, cache_path: str):
        if osp.exists(cache_path):
            with open(cache_path, "rb") as f:
                data = pickle.load(f)
            return data
        raise FileNotFoundError(f"No such file or directory: {cache_path}")

    def __getitem__(self, idx: int):
        data = {}

        if self.graphs is None:
            raise ValueError(
                "QM9Dataset requires build_graph_cfg to return model-ready samples."
            )

        graph = self.graphs[idx]
        if isinstance(graph, str):
            graph = self.load_from_cache(graph)
        data["graph"] = graph

        for property_name in self.property_names:
            if property_name in self.property_data:
                data[property_name] = np.array(
                    [self.property_data[property_name][idx]], dtype=np.float32
                )
            else:
                raise KeyError(f"Property {property_name} not found.")
        data["id"] = self.raw_data["id"][idx]
        data = self.transforms(data) if self.transforms is not None else data

        return data

    def __len__(self):
        return self.num_samples

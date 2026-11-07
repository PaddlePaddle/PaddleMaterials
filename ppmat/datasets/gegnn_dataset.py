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

import copy
import os
import os.path as osp
import pickle
from typing import Any
from typing import Dict
from typing import Optional

import paddle
import paddle.distributed as dist
import pandas as pd
from paddle.io import Dataset

from ppmat.datasets.build_solvent import BuildBinaryMixture
from ppmat.datasets.build_solvent import BuildSolvent
from ppmat.models import build_graph_converter
from ppmat.utils import download
from ppmat.utils import logger
from ppmat.utils.download import DATASETS_HOME
from ppmat.utils.misc import is_equal
from ppmat.vocab import build_vocab

__all__ = ["BinaryActivityDataset"]


class BinaryActivityDataset(Dataset):
    """Binary-mixture activity-coefficient dataset.

    Molecular graphs are built once and cached as indexed pickle files.

    Args:
        path: Path to the binary activity CSV. If it does not exist, the
            dataset will be downloaded.
        solvent_list_path: Path to the solvent metadata CSV. Defaults to a
            ``solvent_list.csv`` sibling of ``path``.
        vocab: Vocabulary package provided by the shared vocabulary builder.
        build_molecule_cfg: Options for BuildMolecule. Defaults to SMILES.
        build_graph_cfg: Config for building molecular graphs.
        cache_path: Directory for cached molecular graphs. Defaults to a
            ``_cache`` sibling of the data directory.
        overwrite: Whether to rebuild the cache even if it exists.
    """

    name = "binary_activity"
    url = "https://paddle-org.bj.bcebos.com/paddlematerials/datasets/thermodynamic_data_of_binary_mixtures/"  # noqa
    md5 = {
        "output_binary_with_inf_all.csv": "67c7b4112cb248c7284004bab14398a4",
        "solvent_list.csv": "5fdb2e2295b327cd111cea5e19db9fcf",
    }

    def __init__(
        self,
        path: str = "./data/binary_activity/output_binary_with_inf_all.csv",
        solvent_list_path: Optional[str] = None,
        vocab: Optional[Dict] = None,
        build_graph_cfg: Optional[Dict] = None,
        build_molecule_cfg: Optional[Dict] = None,
        cache_path: Optional[str] = None,
        overwrite: bool = False,
    ):
        super().__init__()
        vocab = build_vocab(vocab)

        if not osp.exists(path):
            if osp.basename(path) not in self.md5:
                raise FileNotFoundError(
                    f"Dataset file {path} does not exist. Prepare the train/validation "
                    "CSV files with ppmat.datasets.split_binary_activity_data first."
                )
            logger.message("The dataset is not found. Will download it now.")
            root_path = osp.join(DATASETS_HOME, self.name)
            os.makedirs(root_path, exist_ok=True)
            path = download.get_path_from_url(
                self.url + osp.basename(path),
                root_path,
                md5sum=self.md5.get(osp.basename(path)),
                decompress=False,
            )
        if solvent_list_path is None:
            solvent_list_path = osp.join(osp.dirname(path), "solvent_list.csv")
        if not osp.exists(solvent_list_path):
            root_path = osp.join(DATASETS_HOME, self.name)
            os.makedirs(root_path, exist_ok=True)
            solvent_list_path = download.get_path_from_url(
                self.url + "solvent_list.csv",
                root_path,
                md5sum=self.md5.get("solvent_list.csv"),
                decompress=False,
            )

        self.path = path
        self.solvent_list_path = solvent_list_path
        self.build_graph_cfg = (
            copy.deepcopy(build_graph_cfg)
            if build_graph_cfg is not None
            else {
                "__class_name__": "MolecularGraphConverter",
                "__init_params__": {
                    "vocab": vocab,
                    "remove_h": False,
                    "add_self_loops": True,
                    "edge_mode": "bidirectional",
                    "node_feature_key": "h",
                },
            }
        )
        if vocab is not None and "__init_params__" in self.build_graph_cfg:
            self.build_graph_cfg["__init_params__"]["vocab"] = vocab
        self.build_molecule_cfg = build_molecule_cfg or {"format": "smiles"}
        self.build_mixture = BuildBinaryMixture()

        if cache_path is not None:
            self.cache_path = cache_path
        else:
            self.cache_path = osp.join(
                osp.split(path)[0] + "_cache",
                osp.splitext(osp.basename(path))[0],
            )
        logger.info(f"Cache path: {self.cache_path}")

        self.cache_exists = osp.exists(self.cache_path)
        self.overwrite = overwrite

        self.dataset = self.read_data(path)
        self.num_samples = len(self.dataset)
        logger.info(f"Load {self.num_samples} binary-mixture samples from {path}")

        self.solvent_smiles = self.read_solvent_smiles(solvent_list_path)
        self.solvent_ids = list(self.solvent_smiles)
        self.solvent_index = {
            solvent_id: index for index, solvent_id in enumerate(self.solvent_ids)
        }
        self.num_solvents = len(self.solvent_ids)

        graph_cache_path = osp.join(self.cache_path, "graphs")
        config_cache_path = osp.join(self.cache_path, "build_graph_cfg.pkl")

        cache_config = {
            "build_graph_cfg": self.build_graph_cfg,
            "build_molecule_cfg": self.build_molecule_cfg,
            "solvent_ids": self.solvent_ids,
            "solvent_smiles": [
                self.solvent_smiles[solvent_id] for solvent_id in self.solvent_ids
            ],
        }

        need_rebuild = overwrite or not self.cache_exists
        if self.cache_exists and not overwrite:
            try:
                graph_paths = [
                    osp.join(graph_cache_path, f"{index:010d}.pkl")
                    for index in range(self.num_solvents)
                ]
                if not all(osp.exists(p) for p in graph_paths):
                    raise FileNotFoundError("The cached graph files are incomplete.")
                cached_config = self.load_from_cache(config_cache_path)
                if is_equal(cached_config, cache_config):
                    logger.info(
                        "The cached graph configuration matches the current "
                        "settings. Reusing previously generated molecular graphs."
                    )
                else:
                    logger.warning(
                        "build_graph_cfg or solvent metadata differs from cache. "
                        "Will rebuild the graphs."
                    )
                    need_rebuild = True
            except Exception as error:
                logger.warning(error)
                logger.warning(
                    "Failed to load graph cache metadata. Will rebuild the graphs."
                )
                need_rebuild = True

        if need_rebuild:
            if not dist.is_initialized() or dist.get_rank() == 0:
                os.makedirs(graph_cache_path, exist_ok=True)
                build_solvent = BuildSolvent(
                    build_graph_converter(self.build_graph_cfg),
                    self.build_molecule_cfg,
                )
                for index, solvent_id in enumerate(self.solvent_ids):
                    self.save_to_cache(
                        osp.join(graph_cache_path, f"{index:010d}.pkl"),
                        build_solvent(self.solvent_smiles[solvent_id]),
                    )
                self.save_to_cache(config_cache_path, cache_config)
                logger.info(f"Save {self.num_solvents} graphs to {graph_cache_path}")
            if dist.is_initialized():
                dist.barrier()

        self.graphs = [
            osp.join(graph_cache_path, f"{index:010d}.pkl")
            for index in range(self.num_solvents)
        ]

    def read_data(self, path):
        """Read the binary-mixture records."""
        dataset = pd.read_csv(path, low_memory=False)
        return dataset.reset_index(drop=True)

    def read_solvent_smiles(self, solvent_list_path):
        """Read SMILES for solvent IDs used by the dataset."""
        solvents = pd.read_csv(solvent_list_path, index_col="solvent_id")
        if "smiles_can" not in solvents:
            raise ValueError("Solvent metadata must contain 'smiles_can'.")
        solvent_ids = pd.unique(
            self.dataset[["solv1", "solv2"]].to_numpy().ravel()
        ).tolist()
        return {
            solvent_id: solvents.loc[solvent_id, "smiles_can"]
            for solvent_id in solvent_ids
        }

    def save_to_cache(self, cache_path: str, data: Any):
        with open(cache_path, "wb") as file:
            pickle.dump(data, file)

    def load_from_cache(self, cache_path: str):
        if osp.exists(cache_path):
            with open(cache_path, "rb") as file:
                return pickle.load(file)
        raise FileNotFoundError(f"No such file or directory: {cache_path}")

    def __getitem__(self, idx: int):
        """Get one binary-mixture sample."""
        if isinstance(idx, paddle.Tensor):
            idx = idx.item()
        row = self.dataset.iloc[idx]
        solvent1 = self.load_from_cache(self.graphs[self.solvent_index[row.solv1]])
        solvent2 = self.load_from_cache(self.graphs[self.solvent_index[row.solv2]])
        return self.build_mixture(
            solvent1,
            solvent2,
            row.solv1_x,
            gamma=[row.solv1_gamma, row.solv2_gamma],
        )

    def __len__(self):
        return self.num_samples

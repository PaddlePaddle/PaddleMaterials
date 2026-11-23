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

from __future__ import absolute_import
from __future__ import annotations

import json
import os
import os.path as osp
import pickle
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
import paddle
import paddle.distributed as dist
import pandas as pd
import pgl
from paddle.io import Dataset
from rdkit import Chem
from rdkit import RDLogger

from ppmat.datasets.build_molecule import BuildMolecule
from ppmat.datasets.build_spectrum import BuildSpectrumNMR
from ppmat.datasets.custom_data_type import ConcatData
from ppmat.models import build_graph_converter
from ppmat.models.diffnmr.utils import diffgraphformer_utils as utils
from ppmat.utils import download
from ppmat.utils import logger
from ppmat.utils.ext_rdkit import build_molecule_with_partial_charges
from ppmat.utils.ext_rdkit import compute_molecular_metrics
from ppmat.utils.ext_rdkit import mol2smiles
from ppmat.utils.misc import is_equal

# Molecular weight used to normalize the weight feature, per subset and per
# ``remove_h``. Unlike the histograms these are release constants rather than
# data statistics, so they stay in code.
MSD_NMR_MAX_WEIGHT = {
    "n<15": {False: 390, True: 564},
    "n<20": {False: 390, True: 631},
    "n<25": {False: 390, True: 998},
    "n<35": {False: 390, True: 1094},
}

TRAIN_SMILES_REGISTRY = {
    ("n<15", True): {
        "name": "msd_nmr_nless15_train_smiles_no_h",
        "url": (
            "https://paddle-org.bj.bcebos.com/paddlematerials/datasets/"
            "MSD_nmr/msd_nmr_nless15_train_smiles_no_h.npy"
        ),
        "md5": "a0047cd89ed46b98a79a607641231d04",
    },
}


class MSDnmrDataset(Dataset):
    """Multimodal Spectrum Dataset‑Nuclear Magnetic Resonance subset handler.

    This class provides utilities for loading and processing the MSD NMR dataset.
    The dataset contains preprocess dataset includes SMILES of molecules, tokenized
    input of NMR and atom counts of molecules. Tokenized input includs chemical shift,
    multiplicity, intensity etc.
    The total dataset is divided into three parts: training, validation, and testing
    and devided into 4 parts by number of atoms per molecules.

    **Dataset Overview**
    - **Source**: Original data available at
    https://github.com/rxn4chemistry/multimodal-spectroscopic-dataset
    - **Preprocessed Version**:
    ```
    ┌───────────────────┬─────────┬─────────┬─────────┬──────────┐
    │ Dataset Partition │ Train   │ Val     │ Test    │ Total    │
    ├───────────────────┼─────────┼─────────┼─────────┼──────────┤
    │       n<15        │ 109,358 │ 6,076   │ 6,075   │ 121509   │
    ├───────────────────┼─────────┼─────────┼─────────┼──────────┤
    │       n<20        │ 235,512 │ 13,085  │ 13,084  │ 261681   │
    ├───────────────────┼─────────┼─────────┼─────────┼──────────┤
    │       n<25        │ 351,273 │ 19,516  │ 19,515  │ 390,304  │
    ├───────────────────┼─────────┼─────────┼─────────┼──────────┤
    │       n<35        │ 517,319 │ 28,741  │ 28,739  │ 574,799  │
    └───────────────────┴─────────┴─────────┴─────────┴──────────┘
    ```
    Download preprocessed data:
    https://paddle-org.bj.bcebos.com/paddlematerial/datasets/msd/msd_nmr.zip

    **Data Format**
    The dataset is stored in CSV format with the following structure:
    ```CSV
    smiles,tokenized_input,atom_count

    The tokenized_input is stored as a JSON-style dictionary with two top-level keys:
    "1HNMR"  : a list of proton (^1H) NMR signals
    "13CNMR" : a list of carbon (^13C) NMR chemical shifts

    Each element in the "1HNMR" list represents a single proton signal and is itself
    a five-element array in the form:
    [chemical_shift_ppm, line_width_ppm, multiplicity, integration, coupling_constants]

    - chemical_shift_ppm: float – the chemical shift δ value in parts per million.
    - line_width_ppm   : float – the peak width (or half-height width) in ppm.
    - multiplicity     : str   – the splitting pattern, e.g.:
        * "t"   : triplet
        * "dd"  : doublet of doublets
        * "td"  : triplet of doublet
        * "ddt" : doublet of doublet of triplet
        * "qt"  : quartet of triplet
        (other patterns may appear depending on the spectrum).
    - integration      : str   – the number of protons represented by the signal,
        expressed as a string like "1H", "2H", "3H", etc.
    - coupling_constants: list of floats – a list of J couplings (Hz) associated
        with this signal; an empty list means no couplings were reported.

    The "13CNMR" entry is simply a list of float values, where each value is a carbon
    chemical shift in ppm. No additional information (such as line widths or couplings)
    is provided for the carbon spectra.
    ```

    Args:
        path: CSV file or list of CSV files containing ``smiles``,
            ``tokenized_input``, and ``atom_count`` columns.
        data_flag: Dataset subset identifier.
        max_atoms: Maximum number of atoms represented by the model.
        vocab: Registered atom, bond, peak-width, split, and integral
            vocabularies.
        build_molecule_cfg: Keyword arguments passed to ``BuildMolecule``.
        build_graph_cfg: Registered graph converter configuration.
        build_spectrum_cfg: Keyword arguments passed to ``BuildSpectrumNMR``.
            The registered vocabulary is injected at runtime.
        transforms: Optional per-sample transform.
        cache_path: Directory containing converted molecule, graph, and
            spectrum caches.
        statistics_cache_path: Optional statistics cache override. Relative
            paths are resolved from the current working directory.
        overwrite: Whether to rebuild an existing cache.
        filter_unvalid: Whether to remove invalid molecular samples.
        **kwargs: Additional dataset options.

    """

    name = "msd_nmr"
    url = "https://paddle-org.bj.bcebos.com/paddlematerial/datasets/msd/msd_nmr.zip"
    md5 = "bcd731f6d4075a93c11641fdebd1d6bd"

    def __init__(
        self,
        path: str,
        data_flag: str,
        max_atoms: int,
        vocab: Dict[str, Dict[str, Any]],
        build_molecule_cfg: Optional[Dict[str, Any]] = None,
        build_graph_cfg: Optional[Dict[str, Any]] = None,
        build_spectrum_cfg: Optional[Dict[str, Any]] = None,
        transforms: Optional[Any] = None,
        cache_path: Optional[str] = None,
        statistics_cache_path: Optional[str] = None,
        overwrite: bool = False,
        filter_unvalid: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__()

        if not osp.exists(path):
            logger.message("The dataset is not found. Will download it now.")
            root_path = download.get_datasets_path_from_url(self.url, self.md5)
            path = osp.join(
                root_path,
                self.name,
                _get_msd_nmr_subdataset_name(data_flag),
                osp.basename(path),
            )

        self.path = path
        self.data_flag = data_flag
        self.max_atoms = max_atoms
        self.filter_unvalid = filter_unvalid
        self.vocab = vocab

        # Config dicts controlling molecule and graph construction
        if build_molecule_cfg is None:
            build_molecule_cfg = {
                "format": "smiles",
                "sanitize": False,
                "add_hs": False,
                "remove_hs": False,
                "kekulize": False,
            }
            logger.message(
                "The build_molecule_cfg is not set, will use the default "
                f"configs: {build_molecule_cfg}"
            )
        self.build_molecule_cfg = build_molecule_cfg

        if build_graph_cfg is None:
            build_graph_cfg = {
                "__class_name__": "MolecularGraphConverter",
                "__init_params__": {
                    "remove_h": False,
                    "add_self_loops": False,
                    "edge_mode": "bidirectional",
                    "num_cpus": 1,
                },
            }
            logger.message(
                "The build_graph_cfg is not set, will use the default "
                f"configs: {build_graph_cfg}"
            )
        self.build_graph_cfg = build_graph_cfg

        if build_spectrum_cfg is None:
            build_spectrum_cfg = {
                "seq_len_H1": 32,
                "seq_len_C13": 32,
                "j_len": 6,
                "num_cpus": 1,
            }
            logger.message(
                "The build_spectrum_cfg is not set, will use the default "
                f"configs: {build_spectrum_cfg}"
            )
        self.build_spectrum_cfg = build_spectrum_cfg
        graph_vocab = {
            "atom": vocab["atom"],
            "bond": vocab["bond"],
        }
        spectrum_vocab = {
            "peakwidth": vocab["peakwidth"],
            "split": vocab["split"],
            "integral": vocab["integral"],
        }

        self.transforms = transforms

        if cache_path is not None:
            self.cache_path = cache_path
        else:
            self.cache_path = osp.join(
                osp.dirname(path) + "_cache", osp.splitext(osp.basename(path))[0]
            )
        logger.info(f"Cache path: {self.cache_path}")

        suffix = "no_h" if build_graph_cfg["__init_params__"]["remove_h"] else "h"
        self.statistics_cache_path = (
            osp.abspath(os.fspath(statistics_cache_path))
            if statistics_cache_path is not None
            else osp.join(osp.dirname(self.cache_path), f"statistics_{suffix}.pdparams")
        )

        self.overwrite = overwrite

        self.cache_exists = True if osp.exists(self.cache_path) else False
        self.raw_data, self.num_samples = self.read_data(path)
        logger.info(f"Load {self.num_samples} samples from {path}")

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
                        " moelcular data to optimize performance."
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
                    graph_vocab_cache = self.load_from_cache(
                        osp.join(self.cache_path, "graph_vocab.pkl")
                    )
                    if is_equal(build_graph_cfg_cache, build_graph_cfg) and is_equal(
                        graph_vocab_cache, graph_vocab
                    ):
                        logger.info(
                            "The cached graph configuration and vocabulary "
                            "match the current settings. Reusing previously "
                            "generated molecular data to optimize performance."
                        )
                    else:
                        logger.warning(
                            "Graph configuration or vocabulary differs from the "
                            "cache. Will rebuild the graphs."
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

            if build_spectrum_cfg is not None and not overwrite:
                try:
                    build_spectrum_cfg_cache = self.load_from_cache(
                        osp.join(self.cache_path, "build_spectrum_cfg.pkl")
                    )
                    spectrum_vocab_cache = self.load_from_cache(
                        osp.join(self.cache_path, "spectrum_vocab.pkl")
                    )
                    if is_equal(
                        build_spectrum_cfg_cache, build_spectrum_cfg
                    ) and is_equal(spectrum_vocab_cache, spectrum_vocab):
                        logger.info(
                            "The cached spectrum configuration and vocabulary "
                            "match the current settings. Reusing previously "
                            "generated spectrum data to optimize performance."
                        )
                    else:
                        logger.warning(
                            "Spectrum configuration or vocabulary differs from the "
                            "cache. Will rebuild the spectrums."
                        )
                        logger.warning(
                            "If you want to use the cached spectrums, "
                            "please ensure that the settings used in match your "
                            "current settings."
                        )
                        overwrite = True

                except Exception as e:
                    logger.warning(e)
                    logger.warning(
                        "Failed to load build_spectrum_cfg.pkl from cache. "
                        "Will rebuild the spectrums."
                    )
                    overwrite = True

        molecule_cache_path = osp.join(self.cache_path, "molecules")
        graph_cache_path = osp.join(self.cache_path, "graphs")
        spectrum_cache_path = osp.join(self.cache_path, "spectrums")

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
                if build_graph_cfg is not None:
                    self.save_to_cache(
                        osp.join(self.cache_path, "graph_vocab.pkl"), graph_vocab
                    )
                self.save_to_cache(
                    osp.join(self.cache_path, "build_spectrum_cfg.pkl"),
                    build_spectrum_cfg,
                )
                if build_spectrum_cfg is not None:
                    self.save_to_cache(
                        osp.join(self.cache_path, "spectrum_vocab.pkl"),
                        spectrum_vocab,
                    )

                molecules = BuildMolecule(**build_molecule_cfg)(self.raw_data["smiles"])
                os.makedirs(molecule_cache_path, exist_ok=True)
                for index, molecule in enumerate(molecules):
                    self.save_to_cache(
                        osp.join(molecule_cache_path, f"{index:010d}.pkl"), molecule
                    )
                logger.info(
                    f"Save {self.num_samples} molecules to {molecule_cache_path}"
                )

                if build_graph_cfg is not None:
                    converter = build_graph_converter(build_graph_cfg, vocab=vocab)
                    graphs = converter(molecules)
                    os.makedirs(graph_cache_path, exist_ok=True)
                    for index, graph in enumerate(graphs):
                        self.save_to_cache(
                            osp.join(graph_cache_path, f"{index:010d}.pkl"), graph
                        )
                    logger.info(f"Save {self.num_samples} graphs to {graph_cache_path}")

                if build_spectrum_cfg is not None:
                    converter = BuildSpectrumNMR(
                        vocab=vocab,
                        **build_spectrum_cfg,
                    )
                    spectrums = converter(self.raw_data["tokenized_nmr"])
                    os.makedirs(spectrum_cache_path, exist_ok=True)
                    for index, spectrum in enumerate(spectrums):
                        self.save_to_cache(
                            osp.join(spectrum_cache_path, f"{index:010d}.pkl"),
                            spectrum,
                        )
                    logger.info(
                        f"Save {self.num_samples} spectrums to {spectrum_cache_path}"
                    )

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
        if build_spectrum_cfg is not None:
            self.spectrums = [
                osp.join(spectrum_cache_path, f"{i:010d}.pkl")
                for i in range(self.num_samples)
            ]
        else:
            self.spectrums = None
        self.properties = self.get_property_data(self.raw_data)  # represent "y"

        assert (
            len(self.molecules) == self.num_samples
        ), "The number of molecules must be equal to the number of samples."
        assert (
            self.graphs is None or len(self.graphs) == self.num_samples
        ), "The number of graphs must be equal to the number of samples."

        # filter data by specific requirement such as max atom number
        if filter_unvalid:
            self.filter_by_atom_count_raw(max_atoms=max_atoms)

    def __getitem__(self, idx: int) -> Tuple[pgl.Graph, Dict[str, Any]]:
        """Get item at index idx."""
        data = {}

        # get graph
        if self.graphs is not None:
            graph = self.graphs[idx]
            if isinstance(graph, str):
                graph = self.load_from_cache(graph)
            data["graph"] = graph
        else:
            molecule = self.molecules[idx]
            if isinstance(molecule, str):
                molecule = self.load_from_cache(molecule)
            data["molecule_array"] = self.get_molecule_array(molecule)

        # Keep the public sample key expected by DiffNMR.
        if self.spectrums is not None:
            spectrum = self.spectrums[idx]
            if isinstance(spectrum, str):
                spectrum = self.load_from_cache(spectrum)
            data["spectrum"] = spectrum

        # get property-like data "y"
        if self.properties is not None:
            property = self.properties[idx]
            atom_count = self.raw_data["atom_count"][idx]
            if isinstance(property, str):
                property = self.load_from_cache(property)
            data["property"] = {
                "y": property,
                "atom_count": atom_count,
            }

        # data = self.transforms(data) if self.transforms is not None else data

        return data

    def __len__(self) -> int:
        return self.num_samples

    def read_data(self, csv_path: str) -> Tuple[Dict[str, List[Any]], int]:
        """Read the prepared MSD-NMR CSV schema."""

        frame = pd.read_csv(csv_path)
        required = {"smiles", "tokenized_input", "atom_count"}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"Missing MSD-NMR columns: {sorted(missing)}")
        raw_data = {
            "smiles": frame["smiles"].astype(str).tolist(),
            "tokenized_nmr": [
                json.loads(value) for value in frame["tokenized_input"].tolist()
            ],
            "atom_count": frame["atom_count"].astype(int).tolist(),
        }
        return raw_data, len(frame)

    def save_to_cache(self, cache_path: str, data: Any):
        with open(cache_path, "wb") as handle:
            pickle.dump(data, handle)

    def load_from_cache(self, cache_path: str):
        if osp.exists(cache_path):
            with open(cache_path, "rb") as f:
                data = pickle.load(f)
            return data
        else:
            raise FileNotFoundError(f"No such file or directory: {cache_path}")

    def filter_by_atom_count_raw(
        self,
        min_atoms: int | None = None,
        max_atoms: int | None = None,
        allowed: set[int] | list[int] | tuple[int, ...] | None = None,
        inplace: bool = True,
    ):
        """
        Filter samples based on raw_data['atom_count'].

        Criteria (AND):
        - min_atoms: keep if count >= min_atoms (if provided)
        - max_atoms: keep if count <= max_atoms (if provided)
        - allowed:   keep if count ∈ allowed (if provided)

        Returns:
            reserve_idx (List[int]): kept indices (when inplace=True, also mutates
                datasets).

        Usage:
            # filter by atom count range [5, 20]
            dataset.filter_by_atom_count_raw(min_atoms=5, max_atoms=20)

            # filter by specific atom counts ∈ {10, 12, 14}
            dataset.filter_by_atom_count_raw(allowed={10, 12, 14})

            # return filtered indices without modifying data
            idx = dataset.filter_by_atom_count_raw(min_atoms=6, inplace=False)
        """
        # 1) Read atom counts from raw data (required source) raw_data['atom_count']
        if not hasattr(self, "raw_data") or "atom_count" not in self.raw_data:
            raise ValueError("raw_data['atom_count'] is required for filtering.")

        counts = np.asarray(self.raw_data["atom_count"], dtype=np.int64).reshape(-1)
        n = counts.shape[0]
        keep = np.ones(n, dtype=bool)

        # 2) Apply filtering criteria (combined with AND)
        if min_atoms is not None:
            keep &= counts >= int(min_atoms)
        if max_atoms is not None:
            keep &= counts <= int(max_atoms)
        if allowed is not None:
            allowed_arr = np.asarray(list(allowed), dtype=np.int64)
            keep &= np.isin(counts, allowed_arr)

        reserve_idx = np.nonzero(keep)[0].tolist()
        filtered_out = n - len(reserve_idx)

        # If not mutating the dataset, return the indices only
        if not inplace:
            return reserve_idx

        # 3) Slice all containers that are aligned to raw_data length
        def _slice_aligned(values):
            if isinstance(values, np.ndarray):
                return values[np.asarray(reserve_idx, dtype=np.int64)]
            if isinstance(values, tuple):
                return tuple(values[i] for i in reserve_idx)
            return [values[i] for i in reserve_idx]

        # 3.1 raw_data: slice list-like values with length n
        for k, v in list(self.raw_data.items()):
            if isinstance(v, (list, tuple, np.ndarray)) and len(v) == n:
                self.raw_data[k] = _slice_aligned(v)

        # 3.2 property_data: slice when aligned
        if hasattr(self, "property_data") and isinstance(self.property_data, dict):
            for k, v in list(self.property_data.items()):
                if isinstance(v, (list, tuple, np.ndarray)) and len(v) == n:
                    self.property_data[k] = _slice_aligned(v)

        # 3.3 other parallel containers (if present and aligned)
        for attr in (
            "molecules",
            "graphs",
            "spectrums",
            "properties",
            "metas",
            "sample_indices",
        ):
            if hasattr(self, attr):
                seq = getattr(self, attr)
                if isinstance(seq, (list, tuple, np.ndarray)) and len(seq) == n:
                    setattr(self, attr, _slice_aligned(seq))

        # 4) Update dataset size and log
        if hasattr(self, "num_samples"):
            self.num_samples = len(reserve_idx)

        try:
            from ppmat.utils import logger

            logger.warning(
                f"[filter_by_atom_count_raw] filtered_out={filtered_out}, "
                f"remaining={self.num_samples}, criteria: "
                f"min={min_atoms}, max={max_atoms}, "
                f"allowed={set(allowed) if allowed is not None else None}"
            )
        except Exception:
            pass

        return reserve_idx

    def get_property_data(self, data):
        property = np.zeros([len(data["smiles"]), 0], dtype=np.float32)
        return property

    def get_molecule_array(self, molecule):
        """
        Return graph-ready arrays (not a pgl.Graph):
        - num_nodes: [1] int64
        - edges:     [E, 2] int64 (sorted)
        - node_feat: [N, A] float32
        - edge_feat: [E, K] float32
        """
        atom_vocab = self.vocab["atom"]
        atom_token_to_id = atom_vocab["token_to_id"]
        num_atom_embeddings = int(atom_vocab["num_embeddings"])
        bond_vocab = self.vocab["bond"]
        bond_token_to_id = bond_vocab["token_to_id"]
        num_bond_embeddings = int(bond_vocab["num_embeddings"])
        no_bond_id = bond_token_to_id["NO_BOND"]

        graph_cfg = self.build_graph_cfg["__init_params__"]
        remove_h = bool(graph_cfg.get("remove_h", True))
        add_self = bool(graph_cfg.get("add_self_loops", False))
        edge_mode = graph_cfg.get("edge_mode", "bidirectional")

        # ---- RDKit Mol ----
        mol = Chem.MolFromSmiles(molecule) if isinstance(molecule, str) else molecule
        if mol is None:
            raise ValueError(f"Invalid molecule/SMILES: {molecule}")
        if remove_h:
            mol = Chem.RemoveHs(mol)
        N = mol.GetNumAtoms()
        if N == 0:
            # empty arrays for consistency
            return {
                "num_nodes": ConcatData(np.asarray([0], dtype=np.int64)),
                "edges": ConcatData(np.zeros((0, 2), dtype=np.int64)),
                "node_feat": ConcatData(
                    np.zeros((0, num_atom_embeddings), dtype=np.float32)
                ),
                "edge_feat": ConcatData(
                    np.zeros((0, num_bond_embeddings), dtype=np.float32)
                ),
            }

        # ---- node_feat (one-hot over the registered atom vocabulary) ----
        idxs = []
        for atom in mol.GetAtoms():
            sym = atom.GetSymbol()
            if sym not in atom_token_to_id:
                raise ValueError(f"Unknown atom symbol '{sym}' in atom vocabulary")
            idxs.append(atom_token_to_id[sym])
        idxs = np.asarray(idxs, dtype=np.int64)  # [N]
        node_feat = np.eye(num_atom_embeddings, dtype=np.float32)[idxs]

        # ---- edges & edge_feat (one-hot over the registered bond vocabulary) ----
        rows, cols, etypes = [], [], []

        def push(u, v, et):
            rows.append(u)
            cols.append(v)
            etypes.append(et)

        for b in mol.GetBonds():
            u, v = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
            bond_type = str(b.GetBondType()).split(".")[-1]
            et = bond_token_to_id.get(bond_type, no_bond_id)

            if edge_mode == "directed":
                push(u, v, et)
            elif edge_mode == "undirected":
                uu, vv = (u, v) if u < v else (v, u)
                push(uu, vv, et)
            elif edge_mode == "bidirectional":
                push(u, v, et)
                push(v, u, et)
            else:
                raise ValueError(f"Unknown edge_mode: {edge_mode}")

        # dedup for undirected
        if edge_mode == "undirected" and rows:
            pair = np.stack([np.asarray(rows), np.asarray(cols)], axis=1)  # [E,2]
            ety = np.asarray(etypes)
            view = pair.view([("r", pair.dtype), ("c", pair.dtype)])[:, 0]
            _, keep = np.unique(view, return_index=True)
            pair, ety = pair[keep], ety[keep]
            rows, cols, etypes = pair[:, 0].tolist(), pair[:, 1].tolist(), ety.tolist()

        if add_self:
            for i in range(N):
                rows.append(i)
                cols.append(i)
                etypes.append(no_bond_id)

        if rows:
            row_np = np.asarray(rows, dtype=np.int64)
            col_np = np.asarray(cols, dtype=np.int64)
            et_np = np.asarray(etypes, dtype=np.int64)
            # deterministic order
            order = np.argsort(row_np * N + col_np, kind="mergesort")
            row_np, col_np, et_np = row_np[order], col_np[order], et_np[order]
            edges = np.stack([row_np, col_np], axis=1).astype(np.int64)  # [E,2]
            edge_feat = np.eye(num_bond_embeddings, dtype=np.float32)[et_np]
        else:
            edges = np.zeros((0, 2), dtype=np.int64)
            edge_feat = np.zeros((0, num_bond_embeddings), dtype=np.float32)

        # ---- pack arrays (mirrors the 4 graph arguments) ----
        molecule_array = {
            "num_nodes": ConcatData(np.asarray([N], dtype=np.int64)),
            "edges": ConcatData(edges),  # [E, 2] int64
            "node_feat": ConcatData(node_feat),  # [N, A] float32
            "edge_feat": ConcatData(edge_feat),  # [E, K] float32
        }
        return molecule_array


class MSDnmrinfos:
    def __init__(self, dataloaders, cfg, vocab, recompute_statistics=False):
        if isinstance(dataloaders, dict):
            dataloaders = DataLoaderCollection(
                dataloaders.get("train"),
                dataloaders.get("val"),
                dataloaders.get("test"),
            )

        self._cfg = cfg
        self._train_dataloader = (
            dataloaders.train_dataloader if dataloaders is not None else None
        )
        self.remove_h = cfg["build_graph_cfg"]["__init_params__"]["remove_h"]
        self.dataflag = cfg["data_flag"]
        self.vocab = vocab
        spectrum_cfg = cfg["build_spectrum_cfg"]
        self.seq_len_H1 = int(spectrum_cfg["seq_len_H1"])
        self.seq_len_C13 = int(spectrum_cfg["seq_len_C13"])
        self.need_to_strip = False

        atom_vocab = vocab["atom"]
        self.atom_encoder = atom_vocab["token_to_id"]
        self.atom_decoder = [
            atom_vocab["id_to_token"][index]
            for index in range(atom_vocab["num_embeddings"])
        ]
        self.num_atom_types = atom_vocab["num_embeddings"]
        self.valencies = (
            [1, 4, 3, 2, 1] if not self.remove_h else [4, 3, 2, 1, 3, 2, 1, 1, 1]
        )
        self.atom_weights = (
            {0: 1, 1: 12, 2: 14, 3: 16, 4: 19}
            if not self.remove_h
            else {
                0: 12,
                1: 14,
                2: 16,
                3: 19,
                4: 30.97,
                5: 32.07,
                6: 35.45,
                7: 79.9,
                8: 126.9,
            }
        )
        self.max_weight = MSD_NMR_MAX_WEIGHT[self.dataflag][self.remove_h]

        train_dataset = getattr(self._train_dataloader, "dataset", None)
        cache_path = cfg.get("statistics_cache_path") or getattr(
            train_dataset, "statistics_cache_path", None
        )
        if cache_path is not None:
            cache_path = osp.abspath(os.fspath(cache_path))

        statistics = None
        if not recompute_statistics and cache_path and osp.exists(cache_path):
            statistics = paddle.load(cache_path)

        distributed = dist.is_initialized() and dist.get_world_size() > 1
        rank = dist.get_rank() if distributed else 0
        if statistics is None and self._train_dataloader is not None:
            if rank == 0 or cache_path is None:
                bond_vocab = vocab["bond"]
                bond_orders = np.zeros(bond_vocab["num_embeddings"], dtype=np.float64)
                order_by_token = {
                    "NO_BOND": 0.0,
                    "SINGLE": 1.0,
                    "DOUBLE": 2.0,
                    "TRIPLE": 3.0,
                    "AROMATIC": 1.5,
                }
                for token, index in bond_vocab["token_to_id"].items():
                    bond_orders[index] = order_by_token[token]

                logger.message(
                    f"Computing training statistics for MSD-NMR {self.dataflag} ..."
                )
                statistics = dataloaders.statistics(
                    num_node_types=self.num_atom_types,
                    num_edge_types=bond_vocab["num_embeddings"],
                    max_nodes_hint=cfg.get("max_atoms"),
                    edge_mode=cfg["build_graph_cfg"]["__init_params__"].get(
                        "edge_mode", "bidirectional"
                    ),
                    bond_orders=bond_orders,
                    no_bond_id=bond_vocab["token_to_id"]["NO_BOND"],
                )
                if cache_path:
                    os.makedirs(osp.dirname(cache_path) or ".", exist_ok=True)
                    paddle.save(statistics, cache_path)
                    logger.message(f"Cached dataset statistics to {cache_path}")
            if distributed and cache_path:
                dist.barrier()
                if rank != 0:
                    statistics = paddle.load(cache_path)

        if statistics is None:
            raise FileNotFoundError(
                "MSD-NMR statistics are not cached; provide the training loader "
                "to compute them on first use."
            )

        self.n_nodes = statistics["n_nodes"]
        self.node_types = statistics["node_types"]
        self.edge_types = statistics["edge_types"]
        self.valency_distribution = statistics["valency_distribution"]
        self.num_classes = len(self.node_types)
        self.max_n_nodes = len(self.n_nodes) - 1
        self.nodes_dist = DistributionNodes(self.n_nodes)

    def load_train_smiles(self):
        return get_train_smiles(
            self._cfg,
            self._train_dataloader,
            self,
            evaluate_dataset=False,
        )


def _get_msd_nmr_subdataset_name(data_flag: str):
    if data_flag == "n<15":
        return "msd_nmr_nless15"
    if data_flag == "n<20":
        return "msd_nmr_nless20"
    if data_flag == "n<25":
        return "msd_nmr_nless25"
    if data_flag == "n<35":
        return "msd_nmr_nless35"
    raise ValueError(
        f"Unknown data_flag: {data_flag}. Expected one of "
        f"{'n<15', 'n<20', 'n<25', 'n<35'}."
    )


def get_train_smiles(cfg, dataloader, dataset_infos, evaluate_dataset=False):
    if evaluate_dataset:
        assert dataset_infos is not None

    remove_h = cfg["build_graph_cfg"]["__init_params__"]["remove_h"]
    file_name = "train_smiles_no_h.npy" if remove_h else "train_smiles_h.npy"
    dataset = getattr(dataloader, "dataset", None)
    cache_dir = getattr(dataset, "cache_path", None)

    if cache_dir is None and cfg.get("datadir") and osp.isdir(cfg["datadir"]):
        subset_dir = cfg["datadir"]
        if not osp.isfile(osp.join(subset_dir, "train.csv")):
            subset_dir = osp.join(
                subset_dir, _get_msd_nmr_subdataset_name(cfg["data_flag"])
            )
        cache_dir = osp.join(subset_dir + "_cache", "train")

    smiles_path = osp.join(cache_dir, file_name) if cache_dir else None
    if smiles_path and osp.exists(smiles_path):
        train_smiles = np.load(smiles_path)
    elif dataloader is None:
        resource = TRAIN_SMILES_REGISTRY[(cfg["data_flag"], remove_h)]
        smiles_path = download.get_datasets_path_from_url(
            resource["url"], resource["md5"]
        )
        train_smiles = np.load(smiles_path)
    else:
        if smiles_path is None:
            raise ValueError("MSD-NMR train SMILES require a dataset cache path.")
        if dist.get_rank() == 0:
            train_smiles = compute_MSDnmr_smiles(
                dataset_infos.atom_decoder,
                dataloader,
                remove_h,
                bond_decoder=dataset_infos.vocab["bond"],
            )
            os.makedirs(osp.dirname(smiles_path), exist_ok=True)
            np.save(smiles_path, np.asarray(train_smiles))
        if dist.is_initialized():
            dist.barrier()
        train_smiles = np.load(smiles_path)

    if evaluate_dataset:
        all_molecules = []
        for data in dataloader:
            dense_data, node_mask = utils.to_dense(
                data.x, data.edge_index, data.edge_attr, data.graph_node_id
            )
            dense_data = dense_data.mask(node_mask, collapse=True)
            X, E = dense_data.X, dense_data.E
            for index in range(X.shape[0]):
                num_nodes = int(paddle.sum((X != -1)[index]))
                all_molecules.append(
                    [X[index, :num_nodes].cpu(), E[index, :num_nodes, :num_nodes].cpu()]
                )
        metrics = compute_molecular_metrics(
            molecule_list=all_molecules,
            train_smiles=train_smiles,
            dataset_info=dataset_infos,
        )
        logger.info(metrics[0])
    return train_smiles


def compute_MSDnmr_smiles(atom_decoder, dataloader, remove_h, bond_decoder=None):
    logger.message(f"Converting MSDnmr dataset to SMILES for remove_h={remove_h}...")
    mols_smiles = []
    dataset = getattr(dataloader, "dataset", None)
    batches = (
        (dataset[index] for index in range(len(dataset)))
        if dataset is not None
        else iter(dataloader)
    )
    len_train = len(dataset) if dataset is not None else len(dataloader)
    invalid = 0
    disconnected = 0
    for i, batch in enumerate(batches):
        RDLogger.DisableLog("rdApp.*")
        if i % 1000 == 0:
            logger.message(
                f"Converting MSDnmr dataset to SMILES "
                f"{float(i)/max(len_train, 1):.2%}"
            )

        logger.info(f"compute_MSDnmr_smiles i: {i:d}")
        dense_data, node_mask = utils.to_dense(
            paddle.to_tensor(batch["graph"].node_feat["feat"], dtype="float32"),
            paddle.to_tensor(batch["graph"].edges.T, dtype="int64"),
            paddle.to_tensor(batch["graph"].edge_feat["feat"], dtype="float32"),
            paddle.to_tensor(batch["graph"].graph_node_id, dtype="int64"),
        )
        dense_data = dense_data.mask(node_mask, collapse=True)
        X, E = dense_data.X, dense_data.E
        n_nodes = [int(paddle.sum((X != -1)[j, :])) for j in range(X.shape[0])]
        molecule_list = []
        for k in range(X.shape[0]):
            n = n_nodes[k]
            atom_types = X[k, :n].cpu()
            edge_types = E[k, :n, :n].cpu()
            molecule_list.append([atom_types, edge_types])
        for _, molecule in enumerate(molecule_list):
            mol = build_molecule_with_partial_charges(
                molecule[0],
                molecule[1],
                atom_decoder,
                bond_decoder=bond_decoder,
            )
            smile = mol2smiles(mol)
            if smile is not None:
                mols_smiles.append(smile)
                mol_frags = Chem.rdmolops.GetMolFrags(
                    mol, asMols=True, sanitizeFrags=True
                )
                if len(mol_frags) > 1:
                    logger.info(f"Disconnected molecule {mol}, {mol_frags}")
                    disconnected += 1
            else:
                logger.info("Invalid molecule obtained.")
                invalid += 1

    logger.info(f"Number of invalid molecules {invalid}")
    logger.info(f"Number of disconnected molecules {disconnected}")

    return mols_smiles


class DataLoaderCollection:
    def __init__(self, train_dataloader, val_dataloader=None, test_dataloader=None):
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader

    def statistics(
        self,
        num_node_types,
        num_edge_types,
        max_nodes_hint=None,
        *,
        edge_mode="bidirectional",
        bond_orders=None,
        no_bond_id=0,
    ):
        """Compute graph histograms from each training sample exactly once."""

        dataset = self.train_dataloader.dataset
        node_counts = np.zeros(int(max_nodes_hint or 0) + 1, dtype=np.float64)
        node_types = np.zeros(num_node_types, dtype=np.float64)
        edge_types = np.zeros(num_edge_types, dtype=np.float64)
        valencies = []

        for index in range(len(dataset)):
            graph = dataset[index]["graph"]
            node_feat = np.asarray(graph.node_feat["feat"])
            edges = np.asarray(graph.edges)
            edge_feat = np.asarray(graph.edge_feat["feat"])
            num_nodes = node_feat.shape[0]

            if num_nodes >= len(node_counts):
                node_counts = np.pad(node_counts, (0, num_nodes + 1 - len(node_counts)))
            node_counts[num_nodes] += 1
            node_types += node_feat.sum(axis=0)

            non_self = edges[:, 0] != edges[:, 1]
            edges = edges[non_self]
            edge_feat = edge_feat[non_self]
            all_pairs = num_nodes * (num_nodes - 1)
            if edge_mode == "undirected":
                all_pairs //= 2
            edge_types += edge_feat.sum(axis=0)
            edge_types[no_bond_id] += all_pairs - len(edges)

            node_valencies = np.zeros(num_nodes, dtype=np.float64)
            seen_pairs = set()
            for edge, edge_type in zip(edges, edge_feat):
                source, target = map(int, edge)
                pair = (min(source, target), max(source, target))
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)
                order = float(edge_type @ bond_orders)
                node_valencies[source] += order
                node_valencies[target] += order
            valencies.extend(node_valencies.astype(np.int64).tolist())

        max_nodes = np.flatnonzero(node_counts)[-1]
        node_counts = node_counts[: max_nodes + 1]
        valency_counts = np.bincount(
            valencies, minlength=max(1, 3 * max_nodes - 2)
        ).astype(np.float64)
        return {
            "n_nodes": paddle.to_tensor(
                node_counts / node_counts.sum(), dtype="float32"
            ),
            "node_types": paddle.to_tensor(
                node_types / node_types.sum(), dtype="float32"
            ),
            "edge_types": paddle.to_tensor(
                edge_types / edge_types.sum(), dtype="float32"
            ),
            "valency_distribution": paddle.to_tensor(
                valency_counts / valency_counts.sum(), dtype="float32"
            ),
        }


class DistributionNodes(object):
    def __init__(self, histogram):
        """Compute the distribution of the number of nodes in the dataset,
            and sample from this distribution.
        historgram: dict. The keys are num_nodes, the values are counts
        """
        if type(histogram) == dict:
            max_n_nodes = max(histogram.keys())
            prob = paddle.zeros(shape=max_n_nodes + 1)
            for num_nodes, count in histogram.items():
                prob[num_nodes] = count
        else:
            prob = histogram
        self.prob = prob / prob.sum()
        self.m = paddle.distribution.Categorical(prob)

    def sample_n(self, n_samples):
        idx = self.m.sample((n_samples,))
        return idx

    def log_prob(self, batch_n_nodes):
        assert len(tuple(batch_n_nodes.shape)) == 1
        p = self.prob.to(batch_n_nodes.place)
        probas = p[batch_n_nodes]
        log_p = paddle.log(x=probas + 1e-30)
        return log_p

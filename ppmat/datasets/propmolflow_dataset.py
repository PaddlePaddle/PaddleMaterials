# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""QM9 adapters for PropMolFlow training and conditional sampling."""

from __future__ import annotations

import hashlib
import json
import os
import os.path as osp
import subprocess
import sys
import tempfile
import zipfile
from datetime import datetime
from functools import partial
from importlib import resources
from typing import Dict

import numpy as np
import paddle.distributed as dist
import pandas as pd
from paddle.io import Dataset
from rdkit import Chem
from rdkit import rdBase
from rdkit.Chem import rdDetermineBonds

from ppmat.datasets.custom_data_type import ConcatData
from ppmat.datasets.qm9_dataset import QM9Dataset
from ppmat.utils import download
from ppmat.utils import logger
from ppmat.utils.ext_rdkit import bond_types_from_vocab

CACHE_VERSION = "v3"
_HASH_CHUNK_SIZE = 1024 * 1024
N_CHARGES = 6
CHARGE_OFFSET = 2
N_BOND_TYPES = 5
BOND_TYPE_INDEX = {bond: index for index, bond in bond_types_from_vocab().items()}
_PERCEPTION_OPTIONS = (
    {"charge": 0},
    {"charge": 0, "allowChargedFragments": False},
)
_HUECKEL_SCRIPT = (
    "import sys\n"
    "from pathlib import Path\n"
    "from rdkit import Chem\n"
    "from rdkit.Chem import rdDetermineBonds\n"
    "mol = Chem.Mol(sys.stdin.buffer.read())\n"
    "rdDetermineBonds.DetermineBonds(mol, charge=0, useHueckel=True)\n"
    "Path(sys.argv[1]).write_bytes("
    "mol.ToBinary(Chem.PropertyPickleOptions.AllProps))\n"
)


def _perceive_molecule(molecule, work_dir):
    if molecule is None:
        return None
    for kwargs in _PERCEPTION_OPTIONS:
        mol = Chem.Mol(molecule)
        try:
            with rdBase.BlockLogs():
                rdDetermineBonds.DetermineBonds(mol, **kwargs)
            return mol
        except (RuntimeError, ValueError):
            continue
    # Extended Hueckel writes fixed filenames; isolate its working directory.
    stamp = datetime.now().strftime("%Y%m%d%H%M%S")
    with tempfile.TemporaryDirectory(
        prefix=f"perception_{stamp}_", dir=work_dir
    ) as tmp:
        filename = f"molecule_{stamp}.bin"
        result = subprocess.run(
            [sys.executable, "-c", _HUECKEL_SCRIPT, filename],
            input=molecule.ToBinary(),
            cwd=tmp,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            check=False,
        )
        if result.returncode != 0:
            return None
        with open(osp.join(tmp, filename), "rb") as stream:
            return Chem.Mol(stream.read())


def _featurize_molecule(mol, atom_map: list):
    if mol is None:
        return None
    atom_map_dict = {symbol: index for index, symbol in enumerate(atom_map)}
    atom_types, charges = [], []
    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol()
        if symbol not in atom_map_dict:
            return None
        atom_types.append(atom_map_dict[symbol])
        charge = atom.GetFormalCharge()
        if not -CHARGE_OFFSET <= charge < N_CHARGES - CHARGE_OFFSET:
            return None
        charges.append(charge)
    num_atoms = mol.GetNumAtoms()
    if num_atoms == 0:
        return None
    adjacency = np.zeros((num_atoms, num_atoms), dtype="int64")
    for bond in mol.GetBonds():
        bond_type = bond.GetBondType()
        if bond_type not in BOND_TYPE_INDEX:
            return None
        begin, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        adjacency[begin, end] = BOND_TYPE_INDEX[bond_type]
        adjacency[end, begin] = BOND_TYPE_INDEX[bond_type]
    upper_i, upper_j = np.triu_indices(num_atoms, k=1)
    return {
        "positions": np.asarray(mol.GetConformer().GetPositions(), dtype="float32"),
        "atom_types": np.asarray(atom_types, dtype="int64"),
        "charges": np.asarray(charges, dtype="int64"),
        "bond_types": adjacency[upper_i, upper_j],
        "num_atoms": num_atoms,
    }


def _one_hot(indices: np.ndarray, num_classes: int) -> np.ndarray:
    indices = np.asarray(indices, dtype="int64")
    out = np.zeros(indices.shape + (num_classes,), dtype="float32")
    np.put_along_axis(out, indices[..., None], 1.0, axis=-1)
    return out


def _featurize_one(molecule, atom_map: list, work_dir: str) -> dict:
    """Module-level worker so the featurization stays process-picklable."""
    return _featurize_molecule(_perceive_molecule(molecule, work_dir), atom_map)


class _QM9Properties:
    """Shared split resolution and conditioning statistics."""

    name = QM9Dataset.name
    url = QM9Dataset.url
    md5 = QM9Dataset.md5

    def _resolve_split_path(self, path: str) -> str:
        """Resolve a split path, downloading the official package when missing."""
        if osp.exists(path):
            return path
        logger.message("The QM9 split is not found. Will download it now.")
        root_path = download.get_datasets_path_from_url(self.url, self.md5)
        return osp.join(root_path, self.name, osp.basename(path))

    def _trim_raw_data(self) -> None:
        """Release CSV geometries and unused columns after feature extraction."""
        self.raw_data = {}

    def _normalization_statistics(self, normalization_source, property_name):
        if normalization_source is None:
            resource = resources.files("ppmat.models.propmolflow").joinpath(
                "qm9_normalization.json"
            )
            with resource.open(encoding="utf-8") as stream:
                reference = json.load(stream)["properties"]
            if property_name not in reference:
                raise ValueError(
                    f"No checkpoint statistics for property {property_name}; "
                    "provide normalization_source for a custom training split"
                )
            mean = reference[property_name]["mean"]
            std = reference[property_name]["std"]
        else:
            if not osp.isfile(normalization_source):
                raise FileNotFoundError(
                    f"normalization_source does not exist: {normalization_source}"
                )
            frame = pd.read_csv(normalization_source, usecols=[property_name])
            values = frame[property_name].to_numpy(dtype="float64")
            if values.size == 0 or not np.isfinite(values).all():
                raise ValueError("Normalization properties must be nonempty and finite")
            mean = float(values.mean())
            std = float(values.std())
        if not np.isfinite([mean, std]).all() or std <= 0:
            raise ValueError("Normalization statistics require positive variance")
        return mean, std


class _QM9Split(_QM9Properties, Dataset):
    """Read conditioning data without constructing training molecules."""

    def __init__(self, path: str):
        super().__init__()
        self.path = self._resolve_split_path(path)
        self.raw_data, self.num_samples = QM9Dataset.read_data(self, self.path)
        if "num_atoms" not in self.raw_data:
            self.raw_data["num_atoms"] = [
                int(xyz.splitlines()[0]) for xyz in self.raw_data["molecule"]
            ]


class PropMolFlowDataset(_QM9Properties, QM9Dataset):
    """Adapt cached QM9 molecules to complete-graph flow-matching features.

    ``normalization_source=None`` preserves the released checkpoint statistics.
    An explicit CSV supplies custom training statistics. ``property_range``
    optionally retains the half-open interval [lower, upper) before molecule
    construction; it never changes the normalization reference implicitly.
    """

    def __init__(
        self,
        path: str = "./data/qm9/train.csv",
        atom_map: list = None,
        conditioning_cfg: Dict = None,
        normalization_source: str = None,
        cache_dir: str = None,
        num_cpus: int = 1,
        rebuild: bool = False,
        property_range: list = None,
    ):
        if atom_map is None:
            raise ValueError("atom_map must be provided")
        if conditioning_cfg is None:
            raise ValueError("conditioning_cfg must be provided")
        self.atom_map = list(atom_map)
        self.conditioning = conditioning_cfg
        self.property_name = conditioning_cfg.get("property")
        if conditioning_cfg.get("enabled", False) and self.property_name is None:
            raise ValueError("property must be provided when conditioning is enabled")
        self.property_range = property_range
        if property_range is not None:
            if (
                self.property_name is None
                or len(property_range) != 2
                or not np.isfinite(property_range).all()
                or property_range[0] >= property_range[1]
            ):
                raise ValueError(
                    "property_range requires a property and finite lower < upper bounds"
                )
        if self.conditioning.get("enabled", False) and self.conditioning.get(
            "normalize", False
        ):
            self.property_mean, self.property_std = self._normalization_statistics(
                normalization_source, self.property_name
            )

        path = self._resolve_split_path(path)
        if cache_dir is None:
            cache_dir = osp.join(download.DATASETS_HOME, "propmolflow")
        source_hash = hashlib.sha256()
        with open(path, "rb") as stream:
            for chunk in iter(lambda: stream.read(_HASH_CHUNK_SIZE), b""):
                source_hash.update(chunk)
        self.cache_metadata = {
            "version": CACHE_VERSION,
            "source_sha256": source_hash.hexdigest(),
            "atom_map": self.atom_map,
            "rdkit_version": rdBase.rdkitVersion,
            "property_range": property_range,
            "selection_property": self.property_name if property_range else None,
        }
        cache_key = hashlib.sha256(
            json.dumps(self.cache_metadata, sort_keys=True).encode("utf-8")
        ).hexdigest()
        cache_stem = osp.join(cache_dir, f"{osp.basename(path)}.{cache_key}")
        molecule_config = {
            "format": "xyz_block",
            "sanitize": False,
            "add_hs": False,
            "remove_hs": False,
            "kekulize": False,
            "num_cpus": num_cpus,
        }
        self.cache_path = cache_stem + ".npz"
        self._validate_distributed_cache(cache_stem, molecule_config, rebuild)
        super().__init__(
            path=path,
            property_names=[self.property_name] if self.property_name else [],
            build_molecule_cfg=molecule_config,
            build_graph_cfg=None,
            cache_path=cache_stem,
            overwrite=rebuild,
            filter_unvalid=False,
        )
        self.cache_path = cache_stem + ".npz"
        self._load_or_build_cache(num_cpus, rebuild)

        if self.conditioning.get("enabled", False):
            self.property_values = np.asarray(
                self.raw_data[self.property_name], dtype="float64"
            )[self.valid_indices]
            if not np.isfinite(self.property_values).all():
                raise ValueError("Conditioning properties must be finite")
        self._trim_raw_data()

    def read_data(self, path: str):
        data, num_samples = super().read_data(path)
        if self.property_range is None:
            return data, num_samples
        values = np.asarray(data[self.property_name], dtype="float64")
        lower, upper = self.property_range
        indices = np.flatnonzero((values >= lower) & (values < upper))
        if not indices.size:
            raise ValueError("No QM9 samples fall within the requested property_range")
        return (
            {key: [value[index] for index in indices] for key, value in data.items()},
            len(indices),
        )

    def _validate_distributed_cache(self, cache_stem, molecule_config, rebuild):
        if not dist.is_initialized():
            return
        status = [None]
        if dist.get_rank() == 0:
            try:
                if rebuild or not self._cache_is_valid():
                    raise ValueError("Prewarm this dataset in a single process first")
                cached_config = self.load_from_cache(
                    osp.join(cache_stem, "build_molecule_cfg.pkl")
                )
                if cached_config != molecule_config:
                    raise ValueError("Prewarm with the same num_cpus setting")
                if not osp.isdir(osp.join(cache_stem, "molecules")):
                    raise FileNotFoundError("The QM9 molecule cache is missing")
            except Exception as error:
                status[0] = f"{type(error).__name__}: {error}"
        dist.broadcast_object_list(status, src=0)
        if status[0] is not None:
            raise RuntimeError(
                f"PropMolFlow distributed cache validation failed: {status[0]}"
            )

    def _cache_is_valid(self) -> bool:
        if not osp.exists(self.cache_path):
            return False
        try:
            with np.load(self.cache_path) as cache:
                meta = json.loads(str(cache["meta"]))
        except (OSError, ValueError, KeyError, zipfile.BadZipFile):
            return False
        return meta == self.cache_metadata

    def _load_or_build_cache(self, num_cpus: int, rebuild: bool) -> None:
        distributed = dist.is_initialized()
        build_error = None
        status = [None]
        if not distributed or dist.get_rank() == 0:
            try:
                if rebuild or not self._cache_is_valid():
                    self._build_cache(num_cpus)
            except Exception as error:
                if not distributed:
                    raise
                build_error = error
                status[0] = f"{type(error).__name__}: {error}"
        if distributed:
            dist.broadcast_object_list(status, src=0)
            if status[0] is not None:
                raise RuntimeError(
                    f"PropMolFlow cache construction failed on rank 0: {status[0]}"
                ) from build_error
        with np.load(self.cache_path) as cache:
            self.positions = cache["positions"]
            self.atom_types = cache["atom_types"]
            self.charges = cache["charges"]
            self.bond_types = cache["bond_types"]
            self.node_idx_array = cache["node_idx_array"]
            self.edge_idx_array = cache["edge_idx_array"]
            self.valid_indices = cache["valid_indices"]

    def _build_cache(self, num_cpus: int) -> None:
        molecules = [self.load_from_cache(path) for path in self.molecules]
        logger.info(f"Featurizing {len(molecules)} molecules from {self.path}")
        worker = partial(
            _featurize_one,
            atom_map=self.atom_map,
            work_dir=osp.dirname(self.cache_path),
        )
        with rdBase.BlockLogs():
            if num_cpus and num_cpus > 1:
                from p_tqdm import p_map

                features = p_map(worker, molecules, num_cpus=num_cpus)
            else:
                features = [worker(mol) for mol in molecules]

        positions, atom_types, charges, bond_types = [], [], [], []
        node_idx_array, edge_idx_array, valid_indices = [], [], []
        node_offset = 0
        edge_offset = 0
        for index, feature in enumerate(features):
            if feature is None:
                continue
            num_atoms = int(feature["num_atoms"])
            positions.append(feature["positions"])
            atom_types.append(feature["atom_types"])
            charges.append(feature["charges"])
            bond_types.append(feature["bond_types"])
            node_idx_array.append((node_offset, node_offset + num_atoms))
            node_offset += num_atoms
            num_edges = int(feature["bond_types"].shape[0])
            edge_idx_array.append((edge_offset, edge_offset + num_edges))
            edge_offset += num_edges
            valid_indices.append(index)

        if not valid_indices:
            raise ValueError(f"No molecule could be featurized from {self.path}")
        skipped = len(molecules) - len(valid_indices)
        if skipped:
            logger.warning(f"Skipped {skipped} molecules during bond perception")

        os.makedirs(osp.dirname(self.cache_path), exist_ok=True)
        cache_file = tempfile.NamedTemporaryFile(
            dir=osp.dirname(self.cache_path),
            prefix=f"{osp.basename(self.path)}.{datetime.now():%Y%m%d%H%M%S}.",
            suffix=".npz",
            delete=False,
        )
        try:
            with cache_file:
                np.savez(
                    cache_file,
                    meta=np.array(json.dumps(self.cache_metadata)),
                    positions=np.concatenate(positions, axis=0),
                    atom_types=np.concatenate(atom_types, axis=0),
                    charges=np.concatenate(charges, axis=0),
                    bond_types=np.concatenate(bond_types, axis=0),
                    node_idx_array=np.asarray(node_idx_array, dtype="int64"),
                    edge_idx_array=np.asarray(edge_idx_array, dtype="int64"),
                    valid_indices=np.asarray(valid_indices, dtype="int64"),
                )
            os.replace(cache_file.name, self.cache_path)
        finally:
            if osp.exists(cache_file.name):
                os.unlink(cache_file.name)
        logger.info(
            f"Cached {len(valid_indices)} featurized molecules to {self.cache_path}"
        )

    def __len__(self):
        return self.node_idx_array.shape[0]

    def __getitem__(self, idx: int):
        node_start, node_end = self.node_idx_array[idx]
        edge_start, edge_end = self.edge_idx_array[idx]

        positions = self.positions[node_start:node_end].astype("float32")
        positions = positions - positions.mean(axis=0, keepdims=True)
        num_atoms = int(node_end - node_start)

        structure_array = {
            "num_atoms": np.array([num_atoms], dtype="int64"),
            "positions": positions,
            "atom_types": _one_hot(
                self.atom_types[node_start:node_end], len(self.atom_map)
            ),
            "atom_charges": _one_hot(
                self.charges[node_start:node_end] + CHARGE_OFFSET, N_CHARGES
            ),
            "bond_types": _one_hot(self.bond_types[edge_start:edge_end], N_BOND_TYPES),
        }

        sample = {
            "structure_array": {
                key: ConcatData(value) for key, value in structure_array.items()
            }
        }
        if self.conditioning.get("enabled", False):
            property_value = float(self.property_values[idx])
            if self.conditioning.get("normalize", False):
                property_value = (
                    property_value - self.property_mean
                ) / self.property_std
            sample["prop"] = property_value
        return sample


class PropMolFlowSamplingDataset(_QM9Split):
    """Sampling inputs for PropMolFlow: number of atoms and property values.

    ``n_atoms`` and ``property_value`` are orthogonal pins: an explicit value
    fixes that side for every sample, while ``mode`` selects the training
    distribution of the unpinned side. ``in_distribution`` draws atom counts
    from the split atom-count distribution and property values from the
    per-atom-count property histograms. ``out_of_distribution`` requires an
    explicit ``property_value`` and draws atom counts from the molecules whose
    property lies in the top quantile range.
    """

    def __init__(
        self,
        path: str = "./data/qm9/train.csv",
        property_name: str = None,
        mode: str = None,
        total_num: int = None,
        seed: int = None,
        normalization_source: str = None,
        property_value: float = None,
        n_atoms: int = None,
        num_bins: int = 1000,
    ):
        if property_name is None:
            raise ValueError("property_name must be provided")
        if mode is None:
            raise ValueError("mode must be provided")
        if total_num is None:
            raise ValueError("total_num must be provided")
        if seed is None:
            raise ValueError("seed must be provided")
        if mode not in ["in_distribution", "out_of_distribution"]:
            raise ValueError(
                "mode must be either 'in_distribution' or 'out_of_distribution', "
                f"got {mode}"
            )
        if mode == "out_of_distribution" and property_value is None:
            raise ValueError(
                "property_value must be provided for out_of_distribution sampling"
            )

        self.property_name = property_name
        self.mode = mode
        self.total_num = total_num

        super().__init__(path=path)

        self.property_mean, self.property_std = self._normalization_statistics(
            normalization_source, self.property_name
        )

        num_atoms_all = np.asarray(self.raw_data["num_atoms"], dtype="int64")
        properties_all = np.asarray(self.raw_data[self.property_name], dtype="float64")

        if n_atoms is not None:
            self.num_atoms = np.full(total_num, int(n_atoms), dtype="int64")
        elif mode == "in_distribution":
            self.num_atoms = self._sample_num_atoms(num_atoms_all, seed)
        else:
            self.num_atoms = self._sample_ood_num_atoms(
                properties_all, num_atoms_all, seed
            )

        if property_value is not None:
            self.property_values = np.full(total_num, float(property_value))
        else:
            self.property_values = self._sample_properties_given_atoms(
                self.num_atoms, properties_all, num_atoms_all, num_bins, seed
            )

        self._trim_raw_data()

    def _sample_num_atoms(self, num_atoms, seed):
        rng = np.random.RandomState(seed)
        unique_num_atoms, counts = np.unique(num_atoms, return_counts=True)
        num_atoms_probs = counts / counts.sum()
        return rng.choice(unique_num_atoms, size=self.total_num, p=num_atoms_probs)

    def _sample_ood_num_atoms(self, properties, num_atoms, seed):
        rng = np.random.RandomState(seed)
        _, low, high = np.quantile(properties, [0.5, 0.97, 1.0])
        mask = (properties >= low) & (properties <= high)
        candidates = num_atoms[mask]
        if candidates.size == 0:
            raise ValueError(
                "No training molecules found in the requested property range"
            )
        return rng.choice(candidates, size=self.total_num, replace=True)

    def _sample_properties_given_atoms(
        self, num_atoms_seq, properties_all, num_atoms_all, num_bins, seed
    ):
        rng = np.random.RandomState(seed)
        histograms = {}
        for n in np.unique(num_atoms_all):
            values = properties_all[num_atoms_all == n]
            prop_min, prop_max = values.min(), values.max()
            prop_range = prop_max - prop_min + 1e-12
            hist, _ = np.histogram(values, bins=num_bins, range=(prop_min, prop_max))
            hist = hist.astype("float64")
            hist = hist / hist.sum()
            histograms[int(n)] = (hist, prop_min, prop_range)

        sampled = np.empty(self.total_num, dtype="float64")
        for i, n in enumerate(num_atoms_seq):
            if int(n) not in histograms:
                raise ValueError(f"No training molecules with {int(n)} atoms")
            hist, prop_min, prop_range = histograms[int(n)]
            bin_idx = rng.choice(num_bins, p=hist)
            left = bin_idx / num_bins * prop_range + prop_min
            right = (bin_idx + 1) / num_bins * prop_range + prop_min
            sampled[i] = rng.uniform(left, right)
        return sampled

    def __getitem__(self, idx: int):
        normalized = (
            self.property_values[idx] - self.property_mean
        ) / self.property_std
        return {
            "structure_array": {
                "num_atoms": ConcatData(np.array([self.num_atoms[idx]], dtype="int64")),
            },
            "properties": ConcatData(np.array([normalized], dtype="float32")),
        }

    def __len__(self):
        return self.total_num

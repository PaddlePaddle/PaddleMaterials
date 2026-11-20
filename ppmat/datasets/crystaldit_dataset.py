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
"""CrystalDiT dataset for the MP-20 crystal structure generation task.

The dataset reads the MP-20 CSV files (``train.csv`` / ``val.csv`` /
``test.csv``) and converts every CIF string into the fixed-length CrystalDiT
representation:

- ``lattice_vectors``: ``[3, 3]`` lattice matrix divided by ``max_length``;
- ``atom_features``: ``[max_atoms, 5]`` features ``(period, group, x, y, z)``
  where ``(period, group)`` is the normalized 2D periodic-table position,
  ``(x, y, z)`` are fractional coordinates, and invalid padding atoms are
  ``[-1, -1, -1, -1, -1]``.

Parsed samples are cached as a pickle file under
``<DATASETS_HOME>/crystaldit/`` so that the CIF parsing runs only once.
"""

import os
import os.path as osp
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from p_tqdm import p_map
from paddle.io import Dataset

from ppmat.datasets.build_structure import BuildStructure
from ppmat.utils import download
from ppmat.utils import logger
from ppmat.utils.periodic_table import ATOMIC_NUMBER_TO_POSITION
from ppmat.utils.periodic_table import normalize_column
from ppmat.utils.periodic_table import normalize_row

INVALID_ATOM_FEATURE = [-1.0, -1.0, -1.0, -1.0, -1.0]


def _convert_single_sample(args):
    """Convert one CIF string into the fixed-length CrystalDiT arrays.

    Returns an ``error`` entry instead of a sample when parsing fails, so
    that the caller can log and skip it rather than cache a fake sample.
    """
    cif_string, max_atoms, max_length = args
    try:
        structure = BuildStructure.build_one(
            cif_string, "cif_str", niggli=False, canocial=False
        )

        lattice_vectors = np.array(structure.lattice.matrix, dtype=np.float64)
        normalized_lattice = lattice_vectors / max_length

        atomic_numbers = []
        atom_coords = []
        for site in structure.sites:
            if len(atomic_numbers) >= max_atoms:
                break
            atomic_numbers.append(site.specie.Z)
            atom_coords.append(site.frac_coords)

        while len(atomic_numbers) < max_atoms:
            atomic_numbers.append(0)
            atom_coords.append([-1.0, -1.0, -1.0])

        atom_features = []
        for atomic_number, coords in zip(atomic_numbers, atom_coords):
            if atomic_number > 0:
                row, column = ATOMIC_NUMBER_TO_POSITION[atomic_number]
                atom_features.append(
                    [
                        normalize_row(row),
                        normalize_column(column),
                        coords[0],
                        coords[1],
                        coords[2],
                    ]
                )
            else:
                atom_features.append(INVALID_ATOM_FEATURE)

        return {
            "lattice_vectors": normalized_lattice.astype(np.float32),
            "atom_features": np.array(atom_features, dtype=np.float32),
        }
    except Exception as error:
        return {"error": f"{type(error).__name__}: {error}"}


class CrystalDiTDataset(Dataset):
    """Fixed-length crystal dataset for CrystalDiT training and sampling.

    Args:
        path (str, optional): Path to the MP-20 CSV file. If the path does not
            exist, the dataset is downloaded through the unified download
            pipeline. Defaults to ``"./data/mp_20/train.csv"``.
        max_atoms (int, optional): Maximum number of atoms per crystal; every
            sample is padded to this fixed length. Defaults to 20.
        max_length (float, optional): Maximum lattice vector length used to
            normalize the lattice matrix. Defaults to 46.7425.
        num_cpus (int, optional): Number of worker processes used for CIF
            parsing. Defaults to 1.
        cache_path (Optional[str], optional): Explicit cache file path for the
            preprocessed samples. When ``None``, the cache is stored under
            ``<DATASETS_HOME>/crystaldit/``. Defaults to None.
        overwrite (bool, optional): Rebuild the cache even if it exists.
            Defaults to False.
    """

    name = "mp_20"
    url = "https://paddle-org.bj.bcebos.com/paddlematerial/datasets/mp_20/mp_20.zip"
    md5 = "73371948155aa9da609436e291142d7e"

    def __init__(
        self,
        path: str = "./data/mp_20/train.csv",
        max_atoms: int = 20,
        max_length: float = 46.7425,
        num_cpus: int = 1,
        cache_path: str = None,
        overwrite: bool = False,
        **kwargs,  # for compatibility
    ):
        super().__init__()

        if not osp.exists(path):
            logger.message("The dataset is not found. Will download it now.")
            root_path = download.get_datasets_path_from_url(self.url, self.md5)
            path = osp.join(root_path, self.name, osp.basename(path))

        self.path = path
        self.max_atoms = max_atoms
        self.max_length = max_length
        self.num_cpus = num_cpus

        if cache_path is not None:
            self.cache_path = cache_path
        else:
            csv_stem = osp.splitext(osp.basename(path))[0]
            self.cache_path = str(
                Path(download.DATASETS_HOME)
                / "crystaldit"
                / f"{csv_stem}_ma{max_atoms}_ml{max_length}.pkl"
            )

        if overwrite or not osp.exists(self.cache_path):
            self._build_cache(path)

        with open(self.cache_path, "rb") as f:
            self.data = pickle.load(f)

        logger.info(
            f"Loaded {len(self.data)} samples from {path} "
            f"(cache: {self.cache_path})"
        )

    def _build_cache(self, path):
        cache_dir = osp.dirname(self.cache_path)
        os.makedirs(cache_dir, exist_ok=True)

        data = pd.read_csv(path)
        logger.info(f"Parsing {len(data)} CIF strings from {path} ...")

        tasks = [
            (cif_string, self.max_atoms, self.max_length) for cif_string in data["cif"]
        ]
        results = p_map(_convert_single_sample, tasks, num_cpus=self.num_cpus)

        samples = []
        for index, result in enumerate(results):
            if "error" in result:
                logger.warning(f"Skip CIF sample {index}: {result['error']}")
            else:
                samples.append(result)
        if not samples:
            raise RuntimeError(f"No CIF sample parsed from {path}.")

        with open(self.cache_path, "wb") as f:
            pickle.dump(samples, f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        return {
            "lattice_vectors": sample["lattice_vectors"],
            "atom_features": sample["atom_features"],
        }

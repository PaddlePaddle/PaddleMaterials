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

The dataset extends :class:`MP20Dataset` for downloading, CIF parsing and
pymatgen structure caching, and converts every structure into the fixed-length
CrystalDiT representation on access:

- ``lattice_vectors``: ``[3, 3]`` lattice matrix divided by ``max_length``;
- ``atom_features``: ``[max_atoms, 5]`` features ``(period, group, x, y, z)``
  where ``(period, group)`` is the normalized 2D periodic-table position,
  ``(x, y, z)`` are fractional coordinates, and invalid padding atoms are
  ``-1`` in every channel.

Structures are cached as pymatgen pickles under a CrystalDiT-specific cache
directory, so the shared MP-20 cache used by other models stays untouched.
"""

import os.path as osp

import numpy as np

from ppmat.datasets.mp20_dataset import MP20Dataset
from ppmat.utils import download
from ppmat.utils.periodic_table import ATOMIC_NUMBER_TO_POSITION
from ppmat.utils.periodic_table import normalize_column
from ppmat.utils.periodic_table import normalize_row


def _structure_to_sample(structure, max_atoms, max_length):
    """Convert one pymatgen structure into the fixed-length arrays."""
    lattice_vectors = (
        np.array(structure.lattice.matrix, dtype=np.float64) / max_length
    ).astype(np.float32)

    atom_features = np.full((max_atoms, 5), -1.0, dtype=np.float32)
    for atom_index, site in enumerate(structure.sites[:max_atoms]):
        row, column = ATOMIC_NUMBER_TO_POSITION[site.specie.Z]
        atom_features[atom_index] = (
            normalize_row(row),
            normalize_column(column),
            *site.frac_coords,
        )

    return {"lattice_vectors": lattice_vectors, "atom_features": atom_features}


class CrystalDiTDataset(MP20Dataset):
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
        cache_path (Optional[str], optional): Explicit cache directory for the
            parsed structures. When ``None``, the cache is stored under
            ``<DATASETS_HOME>/crystaldit/<split>/``. Defaults to None.
        overwrite (bool, optional): Rebuild the cache even if it exists.
            Defaults to False.
    """

    def __init__(
        self,
        path: str = "./data/mp_20/train.csv",
        max_atoms: int = 20,
        max_length: float = 46.7425,
        num_cpus: int = 1,
        cache_path: str = None,
        overwrite: bool = False,
    ):
        if cache_path is None:
            split = osp.splitext(osp.basename(path))[0]
            cache_path = osp.join(download.DATASETS_HOME, "crystaldit", split)

        super().__init__(
            path=path,
            build_structure_cfg={
                "format": "cif_str",
                "primitive": False,
                "niggli": False,
                "canocial": False,
                "num_cpus": num_cpus,
            },
            cache_path=cache_path,
            overwrite=overwrite,
            filter_unvalid=False,
        )

        self.max_atoms = max_atoms
        self.max_length = max_length

    def __getitem__(self, index):
        structure = self.structures[index]
        if isinstance(structure, str):
            structure = self.load_from_cache(structure)
        return _structure_to_sample(structure, self.max_atoms, self.max_length)

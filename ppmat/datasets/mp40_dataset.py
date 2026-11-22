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

# Adapted from https://github.com/hspark1212/chemeleon

from __future__ import absolute_import
from __future__ import annotations

import os.path as osp
import warnings
from typing import List
from typing import Optional
from typing import Sequence

import numpy as np
import pandas as pd

from ppmat.datasets.custom_data_type import ConcatData
from ppmat.datasets.mp20_dataset import MP20Dataset
from ppmat.utils import download
from ppmat.utils import logger

warnings.filterwarnings("ignore", message="Issues encountered while parsing CIF:")


class MP40Dataset(MP20Dataset):
    """MP-40 Dataset Handler.

    This class provides the mp-40 dataset used by the Chemeleon model within
    the MP20 data pipeline. Download, structure parsing and caching are
    inherited from MP20Dataset; the structure fields are read from an ASE
    Atoms object reached through the pymatgen -> ASE conversion, which is the
    data chain of the Chemeleon reference implementation, and the item keys
    follow the Chemeleon naming (``lattices``/``natoms``).

    **Dataset Overview**
    - **Source**: Original data available at
      https://github.com/hspark1212/chemeleon/tree/main/data/mp-40
    ```
    ┌───────────────────┬────────┬──────┬──────┐
    │ Dataset Partition │ Train  │ Val  │ Test │
    ├───────────────────┼────────┼──────┼──────┤
    │ Sample Count      │ 28,636 │ 3,181│ 708  │
    └───────────────────┴────────┴──────┴──────┘
    ```
    The dataset can also be downloaded from the following source:
    https://paddle-org.bj.bcebos.com/paddlematerials/datasets/mp_40/mp_40.zip

    **Data Format**
    The dataset is a comma-separated values (CSV) file with a ``cif`` column
    plus text-target columns such as ``composition``.

    Each item is a dict with:
      - structure_array: frac_coords (N, 3) float32, atom_types (N,) int64,
        lattices (1, 3, 3) float32 and natoms (1,) int64, each wrapped in
        ConcatData so the ppmat DefaultCollator concatenates them flat
        without padding.
      - text: string joined from the text_targets columns, when given.

    Args:
        path (str, optional): The path of the dataset csv file, if path is
            not exists, it will be downloaded. Defaults to None, in which
            case the path is composed from ``data_dir`` and ``split``.
        data_dir (str, optional): The directory holding the ``{split}.csv``
            files, used to compose the path when ``path`` is not given.
            Defaults to "./data/mp_40".
        split (str, optional): The csv file stem, e.g. train / val / test.
            Defaults to "train". Ignored when ``path`` is given.
        indices (Optional[Sequence[int]], optional): Row indices selecting a
            subset of the csv. The default cache directory is bound to the
            split only, so pass ``cache_path`` to isolate a subset from other
            settings of the same split. Defaults to None.
        text_targets (Optional[List[str]], optional): The csv columns joined
            into the text field. With a single column the text is its value
            as str, with several columns the format is "col: value" joined
            by ", ". Defaults to None.
        cache_path (Optional[str], optional): The directory caching the
            parsed structures in the MP20 per-sample layout. If a cache_path
            is set, the parsed structures are read directly from this path;
            if the cache does not exist, the parsed structures are saved to
            this path. Defaults to None, which follows the MP20 default
            directory of the split.
        **kwargs: Further arguments forwarded to MP20Dataset
            (property_names, build_structure_cfg, build_graph_cfg,
            transforms, overwrite, filter_unvalid).
    """

    name = "mp_40"
    url = "https://paddle-org.bj.bcebos.com/paddlematerials/datasets/mp_40/mp_40.zip"
    md5 = "b6b7d4a888f8b101e640728ea0edc12b"

    def __init__(
        self,
        path: Optional[str] = None,
        data_dir: str = "./data/mp_40",
        split: str = "train",
        indices: Optional[Sequence[int]] = None,
        text_targets: Optional[List[str]] = None,
        cache_path: Optional[str] = None,
        **kwargs,
    ):
        # assigned before super().__init__ so read_data can cut the subset
        # while the parent parses and caches the structures
        self.indices = list(indices) if indices is not None else None
        self.text_targets = text_targets
        if path is None:
            path = osp.join(data_dir, f"{split}.csv")
        if not osp.exists(path):
            path = self._locate_downloaded_csv(path)
        super().__init__(path=path, cache_path=cache_path, **kwargs)

    def _locate_downloaded_csv(self, path: str) -> str:
        """Download the dataset package and locate the requested csv file.

        The package stores the csv files below a top-level ``mp_40/`` folder
        while ``get_path_from_url`` returns the directory derived from the
        first archive entry, so the csv file is searched below the extraction
        directory instead of using the parent class join.

        Args:
            path (str): Path of the missing csv file, only its basename is
                used for the lookup.

        Returns:
            str: The path of the csv file inside the downloaded package.
        """
        root_path = download.get_datasets_path_from_url(self.url, self.md5)
        for base in (root_path, osp.dirname(root_path)):
            candidate = osp.join(base, self.name, osp.basename(path))
            if osp.exists(candidate):
                return candidate
        raise FileNotFoundError(
            f"{osp.basename(path)} not found below {root_path} after "
            f"downloading {self.url}"
        )

    def read_data(self, path: str):
        """Read the csv file and keep only the rows selected by indices.

        Args:
            path (str): Path to the csv file.
        """
        data = pd.read_csv(path)
        if self.indices is not None:
            data = data.iloc[self.indices]
        logger.info(f"Read {len(data)} structures from {path}")
        data = {key: data[key].tolist() for key in data if "Unnamed" not in key}
        num_samples = 0
        for key in data:
            num_samples = max(num_samples, len(data[key]))
        return data, num_samples

    def get_structure_array(self, structure):
        """Get the Chemeleon structure array from a pymatgen Structure.

        The fields are extracted from the ASE Atoms object of the structure
        without normalization: lattices stay in angstroms, fractional
        coordinates stay in [0, 1), atom types stay as raw atomic numbers Z
        (vocabulary size 104 including the dummy 0).

        Args:
            structure (pymatgen.core.Structure): The parsed structure.
        """
        atoms = structure.to_ase_atoms()
        structure_array = {
            "frac_coords": ConcatData(
                np.asarray(atoms.get_scaled_positions(), dtype=np.float32)
            ),
            "atom_types": ConcatData(np.asarray(atoms.numbers, dtype=np.int64)),
            "lattices": ConcatData(
                np.asarray(atoms.cell, dtype=np.float32).reshape(1, 3, 3)
            ),
            "natoms": ConcatData(np.asarray([len(atoms)], dtype=np.int64)),
        }
        return structure_array

    def _build_text(self, idx: int) -> str:
        """Build the text condition of the sample at index idx.

        Args:
            idx (int): Sample index within the selected rows.
        """
        properties = [self.row_data[col][idx] for col in self.text_targets]
        if len(self.text_targets) == 1:
            return str(properties[0])
        return ", ".join(
            f"{self.text_targets[i]}: {properties[i]}"
            for i in range(len(self.text_targets))
        )

    def __getitem__(self, idx: int) -> dict:
        """Get item at index idx.

        Args:
            idx (int): Sample index.
        """
        item = super().__getitem__(idx)
        # MP20Dataset tags every item with "id"; the Chemeleon contract
        # carries only structure_array and text
        item.pop("id")
        if self.text_targets is not None:
            item["text"] = self._build_text(idx)
        return item

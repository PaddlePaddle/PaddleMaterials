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

"""MP-40 dataset producing ppmat-style structure_array records."""

from __future__ import annotations

import pickle
import warnings
from pathlib import Path
from typing import List
from typing import Optional
from typing import Sequence

import numpy as np
import pandas as pd
from ase import Atoms
from paddle.io import Dataset
from pymatgen.core import Structure
from pymatgen.core.lattice import Lattice

from ppmat.datasets.custom_data_type import ConcatData

warnings.filterwarnings("ignore", message="Issues encountered while parsing CIF:")


def cif_to_canonical_atoms(cif_str: str) -> Atoms:
    """Parse a CIF string into a canonical ASE atoms object.

    Processing chain: pymatgen parse, Niggli reduction, lattice rebuild from
    lattice parameters, conversion to ASE.
    """
    st = Structure.from_str(cif_str, fmt="cif")
    reduced_st = st.get_reduced_structure()
    canonical_st = Structure(
        lattice=Lattice.from_parameters(*reduced_st.lattice.parameters),
        species=reduced_st.species,
        coords=reduced_st.frac_coords,
        coords_are_cartesian=False,
    )
    return canonical_st.to_ase_atoms()


def atoms_to_structure_array(atoms: Atoms) -> dict:
    """Extract plain numpy fields from an ASE atoms object.

    No normalization is applied: lattices stay in angstroms, fractional
    coordinates stay in [0, 1), atom types stay as raw atomic numbers Z
    (vocabulary size 104 including the dummy 0).
    """
    return {
        "frac_coords": np.asarray(atoms.get_scaled_positions(), dtype=np.float32),
        "atom_types": np.asarray(atoms.numbers, dtype=np.int64),
        "lattices": np.asarray(atoms.cell, dtype=np.float32).reshape(1, 3, 3),
        "natoms": np.asarray([len(atoms)], dtype=np.int64),
    }


class MP40Dataset(Dataset):
    """MP-40 CSV dataset returning ppmat-compatible item dicts.

    **Dataset Overview**
    - **Source**: Original data available at https://github.com/hspark1212/chemeleon/tree/main/data/mp-40

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
        data_dir: directory holding {split}.csv files.
        split: csv file stem, e.g. train / val / test.
        indices: optional row indices selecting a subset of the csv.
        text_targets: csv columns joined into the text field. With a single
            column the text is its value as str, with several columns the
            format is "col: value" joined by ", ".
        cache_path: optional pkl path caching the parsed structure arrays
            of the selected rows. Built on first construction when absent.
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        indices: Optional[Sequence[int]] = None,
        text_targets: Optional[List[str]] = None,
        cache_path: Optional[str] = None,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.split = split
        self.indices = list(indices) if indices is not None else None
        self.text_targets = text_targets
        self.cache_path = Path(cache_path) if cache_path is not None else None

        path_data = self.data_dir / f"{split}.csv"
        data = pd.read_csv(path_data)
        if self.indices is not None:
            data = data.iloc[self.indices]
        self.data = data.reset_index(drop=True)

        self.structure_arrays = None
        if self.cache_path is not None:
            if self.cache_path.exists():
                with open(self.cache_path, "rb") as f:
                    cache = pickle.load(f)
                if cache.get("indices") == self.indices:
                    self.structure_arrays = cache["structure_arrays"]
            if self.structure_arrays is None:
                self.structure_arrays = [
                    atoms_to_structure_array(
                        cif_to_canonical_atoms(self.data.iloc[i]["cif"])
                    )
                    for i in range(len(self.data))
                ]
                self.cache_path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.cache_path, "wb") as f:
                    pickle.dump(
                        {
                            "indices": self.indices,
                            "structure_arrays": self.structure_arrays,
                        },
                        f,
                    )

    def __len__(self) -> int:
        return len(self.data)

    def _build_text(self, idx: int) -> str:
        row = self.data.iloc[idx]
        properties = row[self.text_targets].values
        if len(self.text_targets) == 1:
            return str(properties[0])
        return ", ".join(
            f"{self.text_targets[i]}: {properties[i]}" for i in range(len(properties))
        )

    def __getitem__(self, idx: int) -> dict:
        if self.structure_arrays is not None:
            structure_array = self.structure_arrays[idx]
        else:
            structure_array = atoms_to_structure_array(
                cif_to_canonical_atoms(self.data.iloc[idx]["cif"])
            )
        item = {
            "structure_array": {
                key: ConcatData(value) for key, value in structure_array.items()
            }
        }
        if self.text_targets is not None:
            item["text"] = self._build_text(idx)
        return item

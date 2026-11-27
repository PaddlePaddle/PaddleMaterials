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

from pathlib import Path

import numpy as np
from paddle.io import Dataset

from ppmat.datasets.geometric_data_type.data import Data


class GeomConformerDataset(Dataset):
    """Read GEOM conformer graphs with explicit, molecule-local bonds.

    ``pack_by_molecule`` gives one graph per molecule, taken from its first
    row as in the author's packed test set, with ``num_reference`` set to the
    number of the molecule's rows.
    """

    _SPLIT_FILES = {
        "train": "train_data_40k.npz",
        "val": "val_data_5k.npz",
        "test": "test_data_1k.npz",
    }

    def __init__(
        self,
        path: str,
        split: str = "train",
        include_rdkit: bool = False,
        pack_by_molecule: bool = False,
    ):
        super().__init__()
        if split not in self._SPLIT_FILES:
            raise ValueError(f"Unknown GEOM split: {split!r}.")
        path = Path(path)
        if path.is_dir():
            path = path / self._SPLIT_FILES[split]
        self.first_rows = None
        with np.load(path, allow_pickle=False) as archive:
            self.rdmol_ptr = (
                archive["rdmol_ptr"]
                if include_rdkit and "rdmol_ptr" in archive
                else None
            )
            if self.rdmol_ptr is not None:
                self.rdmol_data = archive["rdmol_data"]
            self.node_ptr = archive["node_ptr"]
            self.edge_ptr = archive["edge_ptr"]
            self.atom_types = archive["atom_type"]
            if "pos" in archive:
                self.cart_coords = archive["pos"]
            elif split == "test":
                self.cart_coords = None
            else:
                raise ValueError("GEOM training and validation require coordinates.")
            self.edge_index = archive["edge_index"]
            self.edge_type = archive["edge_type"]
            if pack_by_molecule:
                if "smiles" not in archive.files:
                    raise ValueError("Packing by molecule requires author SMILES.")
                first_rows, num_references, seen = [], [], {}
                for index, key in enumerate(archive["smiles"].tolist()):
                    if key not in seen:
                        seen[key] = len(first_rows)
                        first_rows.append(index)
                        num_references.append(0)
                    num_references[seen[key]] += 1
                self.first_rows = first_rows
                self.num_references = num_references

    def __len__(self):
        if self.first_rows is not None:
            return len(self.first_rows)
        return len(self.node_ptr) - 1

    def __getitem__(self, index):
        row = self.first_rows[index] if self.first_rows is not None else index
        start, stop = self.node_ptr[row : row + 2]
        edges = slice(*self.edge_ptr[row : row + 2])
        graph = Data(
            num_nodes=int(stop - start),
            atom_types=self.atom_types[start:stop],
            cart_coords=None
            if self.cart_coords is None
            else self.cart_coords[start:stop],
            edge_index=self.edge_index[:, edges],
            edge_type=self.edge_type[edges],
        )
        if self.rdmol_ptr is not None:
            mol_start, mol_stop = self.rdmol_ptr[row : row + 2]
            graph.rdmol_data = self.rdmol_data[mol_start:mol_stop]
            graph.rdmol_size = int(mol_stop - mol_start)
        if self.first_rows is not None:
            graph.num_reference = self.num_references[index]
        return graph

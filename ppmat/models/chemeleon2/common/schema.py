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

import paddle

from ppmat.datasets.build_structure import BuildStructure
from ppmat.datasets.geometric_data_type.batch import Batch
from ppmat.datasets.geometric_data_type.data import Data
from ppmat.utils.crystal import lattice_params_to_matrix_paddle

__all__ = [
    "CrystalBatch",
    "build_structure_array",
    "create_empty_batch",
]


class CrystalBatch(Data):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def num_graphs(self):
        if hasattr(self, "__num_graphs__"):
            return self.__num_graphs__
        if self.batch is not None:
            return int(self.batch.max().item()) + 1
        return 1

    @num_graphs.setter
    def num_graphs(self, v):
        self.__num_graphs__ = v

    def _split_by_batch_index(self):
        if self.batch is None or self.num_atoms is None:
            raise ValueError("batch and num_atoms must be set to split the batch")
        structures = []
        batch_np = self.batch.cpu().numpy()
        num_graphs = int(batch_np.max()) + 1 if len(batch_np) > 0 else 0
        for i in range(num_graphs):
            mask = batch_np == i
            sd = {}
            for k in ("atom_types", "frac_coords", "cart_coords"):
                v = getattr(self, k, None)
                if v is not None:
                    sd[k] = v[mask]
            if self.lattices is not None:
                sd["lattices"] = (
                    self.lattices[i] if self.lattices.ndim == 3 else self.lattices
                )
            structures.append(sd)
        return structures

    def to_atoms(self, frac_coords=True):
        try:
            from ase import Atoms
        except ImportError:
            raise ImportError("ASE is required. pip install ase")
        if self.atom_types is None or self.lattices is None:
            raise ValueError("atom_types and lattices must be set")
        atoms_list = []
        for sd in self._split_by_batch_index():
            types = sd["atom_types"].cpu().numpy()
            lat = sd["lattices"].cpu().numpy().squeeze()
            atoms = Atoms(numbers=types, cell=lat, pbc=True)
            if frac_coords and "frac_coords" in sd:
                atoms.set_scaled_positions(sd["frac_coords"].cpu().numpy())
            elif "cart_coords" in sd:
                atoms.set_positions(sd["cart_coords"].cpu().numpy())
            atoms_list.append(atoms)
        return atoms_list

    def to_structures(self, frac_coords=True):
        if self.atom_types is None or self.lattices is None:
            raise ValueError("atom_types and lattices must be set")
        structure_list = []
        for sd in self._split_by_batch_index():
            if frac_coords and "frac_coords" in sd:
                coords = sd["frac_coords"]
            elif "cart_coords" in sd and not frac_coords:
                coords = sd["cart_coords"]
            else:
                raise ValueError("frac_coords required")
            # Unified parsing interface via BuildStructure.build_one
            crystal_data = {
                "atom_types": sd["atom_types"].cpu().numpy().tolist(),
                "frac_coords": coords.cpu().numpy().tolist(),
                "lattice": sd["lattices"].cpu().numpy().squeeze().tolist(),
            }
            s = BuildStructure.build_one(
                crystal_data, "array", niggli=False, canocial=False
            )
            structure_list.append(s)
        return structure_list


def build_structure_array(batch, structure_array):
    num_atoms = structure_array["num_atoms"]
    batch_size = num_atoms.shape[0]
    total_atoms = num_atoms.sum().item()

    if "atom_types" in structure_array:
        batch.atom_types = structure_array["atom_types"]
    else:
        batch.atom_types = paddle.zeros([total_atoms], dtype="int64")
    batch.num_atoms = num_atoms
    batch.batch = paddle.repeat_interleave(paddle.arange(batch_size), repeats=num_atoms)

    if "frac_coords" in structure_array:
        batch.frac_coords = structure_array["frac_coords"]
    else:
        batch.frac_coords = paddle.rand([total_atoms, 3])

    if "lattice" in structure_array:
        batch.lattices = structure_array["lattice"]
    elif "lengths" in structure_array and "angles" in structure_array:
        batch.lattices = lattice_params_to_matrix_paddle(
            structure_array["lengths"], structure_array["angles"]
        )

    batch.num_nodes = total_atoms
    batch.num_graphs = batch_size
    starts = paddle.cumsum(num_atoms) - num_atoms
    local_offsets = paddle.repeat_interleave(starts, num_atoms)
    batch.token_idx = (
        paddle.arange(total_atoms, dtype=num_atoms.dtype) - local_offsets
    ).astype("int64")

    return batch


def create_empty_batch(num_atoms, atom_types=None):
    data_list = []
    for i, n in enumerate(num_atoms):
        d = CrystalBatch(
            atom_types=paddle.empty([n], dtype="int64")
            if atom_types is None
            else paddle.to_tensor(atom_types[i], dtype="int64"),
            frac_coords=paddle.empty([n, 3]),
            cart_coords=paddle.empty([n, 3]),
            lattices=paddle.empty([1, 3, 3]),
            num_atoms=paddle.to_tensor(n, dtype="int64"),
            lengths=paddle.empty([1, 3]),
            lengths_scaled=paddle.empty([1, 3]),
            angles=paddle.empty([1, 3]),
            angles_radians=paddle.empty([1, 3]),
            token_idx=paddle.arange(n, dtype="int64"),
            num_nodes=n,
        )
        data_list.append(d)
    batch = Batch.from_data_list(data_list)
    return batch

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


import numpy as np
import paddle
from cvve import Structure as CVVEStructure
from pymatgen.core import Structure

from ppmat.models.common.graph_converter import RadiusGraphConverter
from ppmat.utils.crystal import atomic_number_from_symbol


class GPWNOGraphConverter(RadiusGraphConverter):
    """Build GPWNO atom graphs, completing closed-cell boundary images for MP."""

    def __init__(self, pbc=False, **kwargs):
        kwargs.setdefault("cutoff", 3.0)
        kwargs.setdefault("max_num_neighbors", 32)
        kwargs.setdefault("inclusive_cutoff", True)
        kwargs.setdefault("include_distance", False)
        super().__init__(**kwargs)
        self.pbc = pbc

    def __call__(self, structures):
        if isinstance(structures, (Structure, CVVEStructure)):
            return self.from_structure(structures)
        if isinstance(structures, (list, tuple)):
            return [self(item) for item in structures]
        return super().__call__(structures)

    def from_structure(self, structure, node_features=None):
        if isinstance(structure, Structure):
            numbers = np.asarray(structure.atomic_numbers, dtype="int64")
            positions = structure.cart_coords
            cell = structure.lattice.matrix
        elif isinstance(structure, CVVEStructure):
            numbers = np.asarray(
                [atomic_number_from_symbol(symbol) for symbol in structure.symbols],
                dtype="int64",
            )
            positions = structure.cartesian_positions()
            cell = structure.lattice
        else:
            raise TypeError("Expected a pymatgen or cvve structure.")
        return self.from_arrays(numbers, positions, node_features, cell=cell)

    def from_arrays(self, atomic_numbers, positions, node_features=None, *, cell=None):
        if self.pbc:
            if cell is None:
                raise ValueError("Periodic GPWNO graphs require a cell.")
            atomic_numbers, positions, _, sources = self._expand_boundary_atoms(
                paddle.to_tensor(
                    np.asarray(atomic_numbers), dtype="int64", place="cpu"
                ),
                paddle.to_tensor(np.asarray(positions), dtype="float32", place="cpu"),
                paddle.to_tensor(np.zeros(len(positions)), dtype="int64", place="cpu"),
                paddle.to_tensor(
                    np.asarray(cell), dtype="float32", place="cpu"
                ).reshape([1, 3, 3]),
                return_sources=True,
            )
            atomic_numbers, positions = atomic_numbers.numpy(), positions.numpy()
            if node_features is not None:
                node_features = {
                    key: np.asarray(value)[sources.numpy()]
                    for key, value in node_features.items()
                }
        return super().from_arrays(atomic_numbers, positions, node_features)

    def get_radius_edges(self, positions):
        edge_index, distances, directions = super().get_radius_edges(positions)
        # Preserve the released GPWNO source/destination aggregation order.
        order = np.lexsort((edge_index[1], edge_index[0]))
        return edge_index[:, order], distances[order], directions[order]

    @staticmethod
    def _expand_boundary_atoms(
        atom_types, atom_coord, batch_idx, cell, return_sources=False
    ):
        """Restore the closed-cell boundary images used to train MP weights.

        The legacy reader copied atoms on lower cell faces to the corresponding
        upper faces, not to a full 27-cell supercell. Shared field readers now
        return only the original atoms. Keep this compatibility step local to
        this converter and skip images already present in the input. NumPy selects the
        discrete image topology; tensor gathers retain coordinate/cell gradients.
        """
        fractional = paddle.bmm(
            atom_coord.unsqueeze(1), paddle.linalg.inv(cell)[batch_idx]
        ).squeeze(1)
        fractional_np = fractional.detach().numpy()
        types_np = atom_types.numpy()
        batches_np = batch_idx.numpy()
        directions = np.asarray(
            [
                [0, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
                [0, 1, 1],
                [1, 0, 1],
                [1, 1, 0],
                [1, 1, 1],
            ],
            dtype=np.int64,
        )
        tolerance = 1e-6

        existing = {}
        for index, position in enumerate(fractional_np):
            key = (int(batches_np[index]), int(types_np[index]))
            existing.setdefault(key, []).append(position)
        sources, shifts = [], []
        for index, position in enumerate(fractional_np):
            key = (int(batches_np[index]), int(types_np[index]))
            for direction in directions:
                shifted = position + direction
                if direction.any():
                    if np.any(shifted > 1.0 + tolerance):
                        continue
                    if np.any(
                        np.all(
                            np.isclose(existing[key], shifted, atol=tolerance, rtol=0),
                            axis=1,
                        )
                    ):
                        continue
                    existing[key].append(shifted)
                sources.append(index)
                shifts.append(direction)
        if len(sources) == atom_coord.shape[0]:
            result = (atom_types, atom_coord, batch_idx)
            if return_sources:
                return (
                    *result,
                    paddle.to_tensor(
                        np.arange(atom_coord.shape[0]),
                        dtype="int64",
                        place=atom_coord.place,
                    ),
                )
            return result

        sources = paddle.to_tensor(sources, dtype="int64", place=atom_coord.place)
        shifts = paddle.to_tensor(
            np.asarray(shifts), dtype=atom_coord.dtype, place=atom_coord.place
        )
        expanded_batch = batch_idx[sources]
        offset = paddle.bmm(shifts.unsqueeze(1), cell[expanded_batch]).squeeze(1)
        result = (atom_types[sources], atom_coord[sources] + offset, expanded_batch)
        return (*result, sources) if return_sources else result

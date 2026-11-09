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

"""Paddle-safe MatterGen backbone for MatInvent.

``MatterGen`` ports the OCP graph builder, whose PBC check reduces
``pbc[:, i]`` on non-contiguous column views; on GPU ``paddle.all`` mishandles
those views and aborts the graph build. This adapter feeds the same all-True
PBC with contiguous columns, keeping the graph and the state dict unchanged.
"""

import paddle

from ppmat.models.mattergen.mattergen import MatterGen
from ppmat.models.mattergen.mattergen import radius_graph_pbc_ocp
from ppmat.utils.crystal import frac_to_cart_coords_with_lattice


def _radius_graph_pbc(
    cart_coords, lattice, num_atoms, radius, max_neighbors, max_cell_images
):
    """Build the periodic graph with a contiguous-column PBC tensor."""
    # Transposing a row-major [3, B] tensor yields an [B, 3] PBC whose column
    # slices are contiguous, where the Paddle boolean reduction is correct.
    pbc = paddle.ones([3, int(num_atoms.shape[0])], dtype="bool")
    pbc = pbc.to(cart_coords.place).t()
    edge_index, cell_offsets, num_neighbors_image, _, _ = radius_graph_pbc_ocp(
        pos=cart_coords,
        cell=lattice,
        natoms=num_atoms,
        pbc=pbc,
        radius=radius,
        max_num_neighbors_threshold=max_neighbors,
        max_cell_images_per_dim=max_cell_images,
    )
    return edge_index, cell_offsets, num_neighbors_image


class MatinventMatterGen(MatterGen):
    """MatterGen whose interaction graph uses the Paddle-safe PBC build."""

    def _denoise(self, structure_array, times):
        interaction_graph = self._build_interaction_graph(structure_array)
        return self._runtime_denoise(structure_array, times, interaction_graph)

    def _build_interaction_graph(self, structure_array):
        gemnet = self.model.gemnet
        lattice = structure_array["lattice"]
        num_atoms = structure_array["num_atoms"]
        cart_coords = frac_to_cart_coords_with_lattice(
            structure_array["frac_coords"], num_atoms, lattice=lattice
        )
        edge_index, to_jimages, num_bonds = _radius_graph_pbc(
            cart_coords,
            lattice,
            num_atoms,
            gemnet.cutoff,
            gemnet.max_neighbors,
            gemnet.max_cell_images_per_dim,
        )
        # The graph is precomputed here, so the decoder must consume it as is.
        otf_graph = gemnet.otf_graph
        gemnet.otf_graph = False
        try:
            return gemnet.generate_interaction_graph(
                cart_coords, lattice, num_atoms, edge_index, to_jimages, num_bonds
            )
        finally:
            gemnet.otf_graph = otf_graph


__all__ = ["MatinventMatterGen"]

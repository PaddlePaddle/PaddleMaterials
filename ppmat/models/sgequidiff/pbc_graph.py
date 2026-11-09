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
"""Periodic-boundary graph construction for the SGEQuiDiff drift module."""

from typing import Tuple

import paddle

from ppmat.utils.crystal import OFFSET_LIST
from ppmat.utils.scatter import scatter
from ppmat.utils.scatter import scatter_min


def build_pbc_graph(
    cart_coords: paddle.Tensor,
    lattice: paddle.Tensor,
    num_atoms: paddle.Tensor,
) -> Tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor]:
    """Build a periodic-boundary graph over all atom pairs and supercell images.

    For every ordered atom pair inside each crystal, edges to all 26
    supercell image translations are materialized; per (dst, src) node pair
    only the minimum-distance image is kept (periodic self-images of an atom
    with itself are kept when their distance exceeds a numerical tolerance).
    Symmetry-equivalent images whose distance ties within a tolerance are
    all kept, which preserves the Wyckoff symmetry required by the GNN
    drift module.

    Args:
        cart_coords: Packed Cartesian coordinates, shape ``[n_atoms, 3]``.
        lattice: Lattice matrices, shape ``[n_crystals, 3, 3]``.
        num_atoms: Atoms per crystal, shape ``[n_crystals]``.

    Returns:
        Tuple of ``(source_index, destination_index, source_node_image_offsets,
        num_edges_per_crystal)`` where the first two are packed edge endpoint
        indices (both length ``n_edges``) and
        ``source_node_image_offsets`` holds the integer supercell translation
        applied to the source node of each edge, shape ``[n_edges, 3]``.
    """
    batch_size = num_atoms.shape[0]
    atom_pos = cart_coords

    num_atoms_per_crystal_sqr = (num_atoms**2).cast(paddle.int64)

    first_node_index_per_crystal = paddle.cumsum(num_atoms, axis=0) - num_atoms
    first_node_index_per_crystal_expand = paddle.repeat_interleave(
        first_node_index_per_crystal, num_atoms_per_crystal_sqr
    )
    num_atoms_per_crystal_expand = paddle.repeat_interleave(
        num_atoms, num_atoms_per_crystal_sqr
    )

    num_atom_pairs = paddle.sum(num_atoms_per_crystal_sqr)
    index_sqr_offset = (
        paddle.cumsum(num_atoms_per_crystal_sqr, axis=0) - num_atoms_per_crystal_sqr
    )
    index_sqr_offset = paddle.repeat_interleave(
        index_sqr_offset, num_atoms_per_crystal_sqr
    )
    atom_count_sqr = paddle.arange(end=num_atom_pairs, dtype=paddle.int64) - (
        index_sqr_offset
    )

    destination_index = (
        paddle.floor_divide(atom_count_sqr, num_atoms_per_crystal_expand)
        + first_node_index_per_crystal_expand
    )
    source_index = (
        atom_count_sqr % num_atoms_per_crystal_expand
        + first_node_index_per_crystal_expand
    )
    map_edge_to_crystal = paddle.arange(batch_size).repeat_interleave(
        num_atoms_per_crystal_sqr, axis=0
    )

    n_edges_per_crystal_before_masking = num_atoms_per_crystal_sqr

    source_position = atom_pos[source_index]
    destination_position = atom_pos[destination_index]

    num_supercell_images = len(OFFSET_LIST)
    supercell_frac_offsets = paddle.to_tensor(OFFSET_LIST, dtype=paddle.float32)
    batch_supercell_frac_offsets = supercell_frac_offsets.unsqueeze(0).expand(
        [batch_size, num_supercell_images, 3]
    )
    pbc_frac_offsets_per_source_atom = paddle.repeat_interleave(
        batch_supercell_frac_offsets, n_edges_per_crystal_before_masking, axis=0
    )

    pbc_cart_offsets_per_source_atom = paddle.bmm(
        pbc_frac_offsets_per_source_atom,
        paddle.repeat_interleave(lattice, n_edges_per_crystal_before_masking, axis=0),
    )

    destination_position = destination_position.unsqueeze(1).expand(
        [-1, num_supercell_images, -1]
    )
    source_position = (
        source_position.unsqueeze(1).expand([-1, num_supercell_images, -1])
        + pbc_cart_offsets_per_source_atom
    )

    source_index = source_index.unsqueeze(1).expand([-1, num_supercell_images])
    destination_index = destination_index.unsqueeze(1).expand(
        [-1, num_supercell_images]
    )
    map_edge_to_crystal = map_edge_to_crystal.unsqueeze(1).expand(
        [-1, num_supercell_images]
    )

    inter_atom_distances = (source_position - destination_position).norm(axis=-1)

    mask = paddle.logical_or(
        source_index != destination_index,
        inter_atom_distances > 1e-5,
    )

    flat_mask = mask.reshape([-1])
    destination_index = destination_index[mask]
    source_index = source_index[mask]
    map_edge_to_crystal = map_edge_to_crystal[mask]
    pbc_frac_offsets = pbc_frac_offsets_per_source_atom.reshape([-1, 3])[flat_mask]
    pbc_cart_offsets = pbc_cart_offsets_per_source_atom.reshape([-1, 3])[flat_mask]

    source_position_flat = atom_pos[source_index]
    destination_position_flat = atom_pos[destination_index]
    inter_atom_distances = (
        destination_position_flat - source_position_flat - pbc_cart_offsets
    ).norm(axis=-1)

    (
        source_index,
        destination_index,
        source_node_image_offsets,
        num_edges_per_crystal,
    ) = _get_smallest_edge_per_node_pair(
        dst_idx=destination_index,
        src_idx=source_index,
        map_edge_to_crystal=map_edge_to_crystal,
        inter_atom_distances=inter_atom_distances,
        pbc_frac_offsets_per_source_atom=pbc_frac_offsets,
        num_atoms=num_atoms,
    )
    return (
        source_index,
        destination_index,
        source_node_image_offsets,
        num_edges_per_crystal,
    )


def _get_smallest_edge_per_node_pair(
    dst_idx: paddle.Tensor,
    src_idx: paddle.Tensor,
    map_edge_to_crystal: paddle.Tensor,
    inter_atom_distances: paddle.Tensor,
    pbc_frac_offsets_per_source_atom: paddle.Tensor,
    num_atoms: paddle.Tensor,
) -> Tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor]:
    """Keep only the minimum-distance image edge per (dst, src) node pair.

    Edges whose distance is within a ``1e-4`` tolerance of the pair minimum
    are all kept (they correspond to symmetry-equivalent images).
    """
    edges = paddle.stack([dst_idx, src_idx], axis=-1)
    unique_edges, map_edge_to_unique = paddle.unique(edges, axis=0, return_inverse=True)

    min_inter_atom_distances = scatter_min(
        inter_atom_distances,
        map_edge_to_unique,
        dim=0,
        dim_size=unique_edges.shape[0],
    )

    batch_size = num_atoms.shape[0]

    cutoff_distances = 1e-4 + min_inter_atom_distances[map_edge_to_unique]
    keep_edge_mask = inter_atom_distances < cutoff_distances
    indices_to_keep = paddle.nonzero(keep_edge_mask).reshape([-1])

    edges = edges[indices_to_keep]
    pbc_frac_offsets_per_source_atom = pbc_frac_offsets_per_source_atom[indices_to_keep]

    num_edges_per_crystal = scatter(
        src=paddle.ones([edges.shape[0]], dtype=paddle.float32),
        index=map_edge_to_crystal[indices_to_keep],
        dim_size=batch_size,
        reduce="sum",
    ).cast(paddle.int64)

    dst_idx = edges[:, 0]
    src_idx = edges[:, 1]
    return src_idx, dst_idx, pbc_frac_offsets_per_source_atom, num_edges_per_crystal

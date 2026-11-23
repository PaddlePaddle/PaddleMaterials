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
"""SphereNet-specific graph utilities."""

import paddle


def radius_graph(pos, batch, cutoff, loop=False):
    """Build edge indices for a batch of molecules within a cutoff radius.

    Processes each molecule independently to avoid O(N²) memory on the full
    concatenated batch. For each molecule, builds a local N_mol × N_mol
    distance matrix, then remaps edge indices to global positions.

    Args:
        pos: Tensor of shape ``(num_nodes, 3)`` with atomic coordinates.
        batch: Tensor of shape ``(num_nodes,)`` with batch indices.
        cutoff: Neighbor cutoff distance in Ångström.
        loop: Whether to include self-loops (default False).

    Returns:
        edge_index: Tensor of shape ``(2, num_edges)`` with global edge indices.
    """
    num_nodes = pos.shape[0]
    if num_nodes == 0:
        return paddle.empty([2, 0], dtype="int64")

    if batch is None:
        batch = paddle.zeros([num_nodes], dtype="int64")

    _, counts = paddle.unique(batch, return_counts=True)
    edge_list = []
    start = 0
    for i in range(counts.shape[0]):
        n = int(counts[i])
        if n == 0:
            continue
        local_pos = pos[start : start + n]
        diff = local_pos.unsqueeze(1) - local_pos.unsqueeze(0)
        dist_sq = paddle.sum(diff * diff, axis=-1)
        mask = dist_sq < cutoff * cutoff
        if not loop:
            atom_ids = paddle.arange(n, dtype="int64")
            mask = mask & (atom_ids.unsqueeze(0) != atom_ids.unsqueeze(1))
        src, dst = paddle.where(mask)
        if src.shape[0] > 0:
            edge_list.append(paddle.stack([src + start, dst + start], axis=0))
        start += n

    if not edge_list:
        return paddle.empty([2, 0], dtype="int64")
    return paddle.concat(edge_list, axis=1)

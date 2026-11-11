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

"""FlowMol-specific batched complete-graph construction."""

from __future__ import annotations

import numpy as np


def _build_complete_graph_edge_idxs(n_atoms: int) -> np.ndarray:
    """Build the edge index array of a complete graph with ``n_atoms`` nodes.

    The upper-triangle edges are traversed before the lower-triangle edges, so
    the first ``n_atoms * (n_atoms - 1) // 2`` edges are the upper triangle.
    """
    upper_edge_idxs = np.triu_indices(n_atoms, k=1)
    lower_edge_idxs = np.stack((upper_edge_idxs[1], upper_edge_idxs[0]))
    return np.concatenate((upper_edge_idxs, lower_edge_idxs), axis=1)


def build_batched_complete_graph(num_atoms: np.ndarray) -> dict:
    """Build the bookkeeping arrays of a batched complete graph.

    Returns the edge indices with batch offsets, the upper-triangle edge mask
    and the node/edge batch indices of every molecule in the batch.
    """
    num_atoms = np.asarray(num_atoms, dtype="int64")
    edge_idxs_list = [_build_complete_graph_edge_idxs(int(n)) for n in num_atoms]
    num_edges = np.array([e.shape[1] for e in edge_idxs_list], dtype="int64")

    node_offsets = np.concatenate([[0], np.cumsum(num_atoms)[:-1]])
    src = np.concatenate([e[0] for e in edge_idxs_list]) + np.repeat(
        node_offsets, num_edges
    )
    dst = np.concatenate([e[1] for e in edge_idxs_list]) + np.repeat(
        node_offsets, num_edges
    )

    upper_edge_mask = np.zeros(int(num_edges.sum()), dtype=bool)
    offset = 0
    for n_edges in num_edges:
        upper_edge_mask[offset : offset + n_edges // 2] = True
        offset += n_edges

    return {
        "src": src,
        "dst": dst,
        "upper_edge_mask": upper_edge_mask,
        "node_batch_idx": np.repeat(np.arange(len(num_atoms)), num_atoms),
        "edge_batch_idx": np.repeat(np.arange(len(num_atoms)), num_edges),
    }

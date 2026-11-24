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

import numpy as np
from scipy.optimize import linear_sum_assignment


def build_batched_complete_graph(num_atoms: np.ndarray) -> dict:
    """Build the bookkeeping arrays of a batched complete graph.

    Returns the edge indices with batch offsets, the upper-triangle edge mask
    and the node/edge batch indices of every molecule in the batch.
    """
    edge_idxs_list = []
    for n in num_atoms:
        # Upper-triangle edges come first, so the mask is the first half.
        upper_edge_idxs = np.triu_indices(int(n), k=1)
        lower_edge_idxs = np.stack((upper_edge_idxs[1], upper_edge_idxs[0]))
        edge_idxs_list.append(
            np.concatenate((upper_edge_idxs, lower_edge_idxs), axis=1)
        )
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


def align_position_prior(prior: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Align a position prior to the endpoint (Kabsch, after permutation)."""
    cost = np.linalg.norm(dst[:, None, :] - prior[None, :, :], axis=-1)
    _, prior_idx = linear_sum_assignment(cost)
    prior = prior[prior_idx]

    prior_mean = prior.mean(axis=0, keepdims=True)
    dst_mean = dst.mean(axis=0, keepdims=True)
    prior_c = prior - prior_mean
    dst_c = dst - dst_mean

    h = prior_c.T @ dst_c
    u, _, vh = np.linalg.svd(h)
    r = vh.T @ u.T
    aligned = prior_c @ r.T
    return (aligned + dst_mean).astype("float32")

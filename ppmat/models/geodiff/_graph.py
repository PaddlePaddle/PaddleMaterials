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

# Adapted from GeoDiff (https://github.com/MinkaiXu/GeoDiff). See LICENSE.
# Graph operations adapted from PyTorch Geometric 1.7.2, torch-scatter 2.1.2,
# torch-sparse 0.6.18 and torch-cluster 1.6.3. See LICENSE.graph.

import paddle
from rdkit import Chem

from ppmat.utils.scatter import scatter_sum

# Higher-order edges start immediately after the original RDKit bond-type range.
_NUM_BOND_TYPES = len(Chem.rdchem.BondType.names)


def coalesce(index, value, n):
    """Sort the COO coordinates of a square matrix and sum duplicates."""
    if index.shape[1] == 0:
        return index, value
    keys = index[0] * n + index[1]
    keys, groups = paddle.unique(keys, return_inverse=True, sorted=True)
    index = paddle.stack((keys // n, keys % n), axis=0)
    if value is not None:
        value = scatter_sum(value, groups, dim=0, dim_size=keys.shape[0])
    return index, value


def _dense_adjacency(edge_index, num_nodes, edge_attr=None):
    row, col = edge_index
    if edge_attr is None:
        edge_attr = paddle.ones([row.shape[0]], device=edge_index.place)
    adj = scatter_sum(
        edge_attr, row * num_nodes + col, dim=0, dim_size=num_nodes * num_nodes
    )
    return adj.reshape([num_nodes, num_nodes])


@paddle.no_grad()
def _radius(x, y, r, batch_x, batch_y, max_num_neighbors, ignore_same_index):
    if x.numel() == 0 or y.numel() == 0 or max_num_neighbors == 0:
        return paddle.empty([2, 0], dtype="int64", device=x.place)
    if x.ndim == 1:
        x = x.unsqueeze(-1)
    if y.ndim == 1:
        y = y.unsqueeze(-1)

    if batch_x is None and batch_y is None:
        sources = paddle.arange(x.shape[0], dtype="int64", device=x.place)
        points = x.unsqueeze(0)
    else:
        # Each query compares only against the sorted source interval for its batch.
        start = paddle.searchsorted(batch_x, batch_y).unsqueeze(1)
        stop = paddle.searchsorted(batch_x, batch_y, right=True).unsqueeze(1)
        width = int((stop - start).max())
        if width == 0:
            return paddle.empty([2, 0], dtype="int64", device=x.place)
        sources = start + paddle.arange(width, dtype="int64", device=x.place)
        points = paddle.gather(x, sources.clip(max=x.shape[0] - 1).flatten()).reshape(
            [y.shape[0], width, x.shape[1]]
        )

    valid = (points - y.unsqueeze(1)).square().sum(axis=-1) < r * r
    if batch_x is not None:
        valid = valid & (sources < stop)
    if ignore_same_index:
        queries = paddle.arange(y.shape[0], dtype="int64", device=y.place).unsqueeze(1)
        valid = valid & (sources != queries)
    rank = valid.astype(paddle.int32).cumsum(axis=1, dtype="int32")
    pairs = paddle.nonzero(valid & (rank <= max_num_neighbors))
    query, source = pairs[:, 0], pairs[:, 1]
    if batch_x is not None:
        source = source + paddle.gather(start.squeeze(1), query)
    return paddle.stack((query, source), axis=0)


def radius(x, y, r, batch_x=None, batch_y=None, max_num_neighbors=32):
    """Return (query, source) pairs in first-matching source order."""
    return _radius(x, y, r, batch_x, batch_y, max_num_neighbors, False)


def radius_graph(
    x, r, batch=None, loop=False, max_num_neighbors=32, flow="source_to_target"
):
    """Construct capped radius edges, optionally excluding self pairs."""
    assert flow in ("source_to_target", "target_to_source")
    edges = _radius(x, x, r, batch, batch, max_num_neighbors, not loop)
    if flow == "source_to_target":
        return paddle.stack((edges[1], edges[0]), axis=0)
    return edges


def higher_order_edges(num_nodes, edge_index, edge_type, order=3):
    """Return row-major edge indices, types and shortest bond hops as int64."""
    edge_index = edge_index.astype(paddle.int64)
    edge_type = edge_type.astype(paddle.int64)
    if order <= 1 or edge_index.shape[1] == 0:
        index, types = coalesce(edge_index, edge_type, num_nodes)
        return index, types, paddle.ones_like(types)

    bonds = _dense_adjacency(edge_index, num_nodes) > 0
    diagonal = paddle.eye(num_nodes, dtype=paddle.float32, device=edge_index.place) != 0
    adjacency = (bonds | diagonal).astype(paddle.float32)
    hop = bonds.astype(paddle.int64)
    reached = bonds | diagonal
    for distance in range(2, order + 1):
        next_reached = (reached.astype(paddle.float32) @ adjacency) > 0
        hop = paddle.where(next_reached & ~reached, distance, hop)
        reached = next_reached

    coords = paddle.nonzero(hop)
    edge_hop = paddle.gather_nd(hop, coords)
    type_mat = _dense_adjacency(edge_index, num_nodes, edge_type)
    direct_type = paddle.gather_nd(type_mat, coords)
    new_type = paddle.where(edge_hop == 1, direct_type, _NUM_BOND_TYPES + edge_hop - 1)
    return coords.transpose([1, 0]), new_type, edge_hop


def _extend_to_radius_graph(
    pos,
    edge_index,
    edge_type,
    cutoff,
    batch,
    unspecified_type_number=0,
    is_sidechain=None,
):
    assert edge_type.ndim == 1
    N = pos.shape[0]
    if is_sidechain is None:
        rgraph_edge_index = radius_graph(pos, r=cutoff, batch=batch)
    else:
        is_sidechain = is_sidechain.astype(paddle.bool)
        dummy_index = paddle.arange(N, dtype=paddle.int64, device=pos.place)
        sidechain_pos = pos[is_sidechain]
        sidechain_index = dummy_index[is_sidechain]
        sidechain_batch = batch[is_sidechain]
        assign_index = radius(
            x=pos, y=sidechain_pos, r=cutoff, batch_x=batch, batch_y=sidechain_batch
        )
        r_edge_index_x = assign_index[1]
        r_edge_index_y = assign_index[0]
        r_edge_index_y = sidechain_index[r_edge_index_y]
        rgraph_edge_index1 = paddle.stack((r_edge_index_x, r_edge_index_y))
        rgraph_edge_index2 = paddle.stack((r_edge_index_y, r_edge_index_x))
        rgraph_edge_index = paddle.concat(
            (rgraph_edge_index1, rgraph_edge_index2), axis=-1
        )
        rgraph_edge_index = rgraph_edge_index[
            :, rgraph_edge_index[0] != rgraph_edge_index[1]
        ]
    radius_type = paddle.full(
        [rgraph_edge_index.shape[1]],
        unspecified_type_number,
        dtype=paddle.int64,
        device=edge_index.place,
    )
    return coalesce(
        paddle.concat((edge_index, rgraph_edge_index), axis=1),
        paddle.concat((edge_type.astype(paddle.int64), radius_type)),
        N,
    )


def extend_graph_order_radius(
    num_nodes,
    pos,
    edge_index,
    edge_type,
    batch,
    order=3,
    cutoff=10.0,
    extend_order=True,
    extend_radius=True,
    is_sidechain=None,
):
    if extend_order:
        edge_index, edge_type, _ = higher_order_edges(
            num_nodes, edge_index, edge_type, order
        )
    if extend_radius:
        edge_index, edge_type = _extend_to_radius_graph(
            pos=pos,
            edge_index=edge_index,
            edge_type=edge_type,
            cutoff=cutoff,
            batch=batch,
            is_sidechain=is_sidechain,
        )
    return edge_index, edge_type


def get_distance(pos, edge_index):
    return paddle.linalg.norm(pos[edge_index[0]] - pos[edge_index[1]], axis=-1)


def eq_transform(score_d, pos, edge_index, edge_length):
    N = pos.shape[0]
    dd_dr = 1.0 / edge_length * (pos[edge_index[0]] - pos[edge_index[1]])
    return scatter_sum(dd_dr * score_d, edge_index[0], dim=0, dim_size=N) + scatter_sum(
        -dd_dr * score_d, edge_index[1], dim=0, dim_size=N
    )

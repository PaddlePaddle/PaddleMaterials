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

"""DimeNetPlusPlus.triplets against the edge-matrix version it replaced."""

import numpy as np
import paddle
import pytest

from ppmat.models.dimenetpp.dimenetpp import DimeNetPlusPlus


def _edge_matrix_triplets(edge_index, cell_offsets, num_nodes):
    """DimeNetPlusPlus.triplets before the rewrite.

    It lists the edges into every node in a [num_nodes, max in-degree] matrix and
    compares all pairs of edges, so it takes O(E^2) memory.
    """
    row, col = edge_index
    value = paddle.arange(1, row.shape[0] + 1, dtype="int64")
    n = col.shape[0]
    rows = paddle.arange(n).unsqueeze(1)
    cols = paddle.arange(n).unsqueeze(0)
    mask = (col.unsqueeze(1) == col.unsqueeze(0)) & (cols <= rows)
    col_ = mask.astype("int64").sum(axis=1) - 1
    mat = paddle.scatter_nd(
        paddle.stack([col, col_], axis=1),
        value,
        shape=[num_nodes, col_.max().item() + 1],
    )
    idx_kj = mat[row][mat[row] > 0] - 1
    idx_ji = paddle.nonzero(mat[row], as_tuple=False)[:, 0]
    idx_k = paddle.index_select(row, idx_kj, axis=0)
    idx_j = paddle.index_select(row, idx_ji, axis=0)
    idx_i = paddle.index_select(col, idx_ji, axis=0)
    if cell_offsets is None:
        mask2 = idx_i != idx_k
    else:
        cell_offset_kji = cell_offsets[idx_kj] + cell_offsets[idx_ji]
        mask2 = (idx_i != idx_k) | paddle.any(cell_offset_kji != 0, axis=-1)
    return (
        col,
        row,
        idx_i[mask2],
        idx_j[mask2],
        idx_k[mask2],
        idx_kj[mask2],
        idx_ji[mask2],
    )


def _triplets(edge_index, cell_offsets, num_nodes):
    # triplets() does not use the model's state.
    return DimeNetPlusPlus.triplets(None, edge_index, cell_offsets, num_nodes)


def _random_graph(rng, num_nodes, num_edges, self_loops):
    src = rng.integers(0, num_nodes, num_edges)
    dst = rng.integers(0, num_nodes, num_edges)
    if not self_loops:
        same = src == dst
        dst[same] = (dst[same] + 1) % num_nodes
    return num_nodes, src, dst, rng.integers(-1, 2, size=(num_edges, 3))


def _graphs():
    rng = np.random.default_rng(11)
    graphs = {}
    for i in range(12):
        num_nodes = int(rng.integers(3, 12))
        num_edges = int(rng.integers(5, 80))
        graphs[f"random_{i}"] = _random_graph(rng, num_nodes, num_edges, i % 4 == 0)
    graphs["random_large"] = _random_graph(rng, 40, 600, self_loops=False)
    # Three periodic images of every bond in a cell of three atoms.
    pairs = [(i, j) for i in range(3) for j in range(3) if i != j]
    graphs["bond_images"] = (
        3,
        np.repeat([i for i, _ in pairs], 3),
        np.repeat([j for _, j in pairs], 3),
        np.array([[k - 1, 0, 0] for _ in pairs for k in range(3)]),
    )
    # Edges between images of the same atom, where i == k differs only by the
    # cell offset.
    graphs["self_images"] = (
        2,
        np.array([0, 0, 1, 1, 0, 1]),
        np.array([0, 0, 1, 1, 1, 0]),
        np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 0], [0, 0, 0]]),
    )
    # No edge ends where another edge starts.
    graphs["no_triplets"] = (
        5,
        np.array([0, 1, 2, 0, 1]),
        np.array([3, 4, 3, 4, 4]),
        np.zeros((5, 3), dtype="int64"),
    )
    graphs["single_edge"] = (2, np.array([0]), np.array([1]), np.zeros((1, 3)))
    return graphs


GRAPHS = _graphs()


@pytest.mark.parametrize("name", sorted(GRAPHS))
def test_triplets_match_the_edge_matrix_version(name):
    num_nodes, src, dst, offsets = GRAPHS[name]
    edge_index = paddle.to_tensor(np.stack([src, dst]).astype("int64"))
    offsets = offsets.astype("int64")
    for cell_offsets in (offsets, np.zeros_like(offsets), None):
        if cell_offsets is not None:
            cell_offsets = paddle.to_tensor(cell_offsets)
        expected = _edge_matrix_triplets(edge_index, cell_offsets, num_nodes)
        actual = _triplets(edge_index, cell_offsets, num_nodes)
        assert len(actual) == len(expected)
        for a, e in zip(actual, expected):
            assert a.dtype == e.dtype
            # Same triplets in the same order.
            np.testing.assert_array_equal(a.numpy(), e.numpy())


def test_triplets_of_a_graph_without_edges_are_empty():
    # The edge-matrix version fails here: col_.max() of an empty tensor.
    edge_index = paddle.zeros([2, 0], dtype="int64")
    cell_offsets = paddle.zeros([0, 3], dtype="int64")
    for out in _triplets(edge_index, cell_offsets, num_nodes=2):
        assert out.shape == [0]

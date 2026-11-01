# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.

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
import paddle


def _to_np(array):
    """Return a numpy view of a paddle tensor or pass through an ndarray."""
    if hasattr(array, "numpy"):
        return array.numpy()
    return np.asarray(array)


def _concat_edges(src_list, dst_list, dist_list):
    """Flatten per-graph edge lists into global edge arrays."""
    if not src_list:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty.copy(), np.empty(0, dtype=np.float64)
    return (
        np.concatenate(src_list).astype(np.int64),
        np.concatenate(dst_list).astype(np.int64),
        np.concatenate(dist_list).astype(np.float64),
    )


def _cap_edges_per_destination(src, dst, distances, max_num_neighbors):
    """Deterministically retain the nearest ``max_num_neighbors`` edges per destination.

    Edges are grouped by ``dst``; within each group the shortest edge wins, ties are
    broken by ``src`` index. The result is therefore reproducible and independent of
    input ordering or chunk size. ``distances`` may be squared Euclidean lengths;
    only their relative order matters.

    Args:
        src, dst, distances: 1D arrays of equal length describing one edge each.
        max_num_neighbors: per-destination cap; ``None`` or a value >= 1_000_000
            disables truncation (matches the historical sentinel).

    Returns:
        np.ndarray of bool, the keep mask aligned with the inputs.
    """
    src = np.asarray(src, dtype=np.int64)
    dst = np.asarray(dst, dtype=np.int64)
    distances = np.asarray(distances, dtype=np.float64)
    n = src.shape[0]
    if n == 0 or max_num_neighbors is None or max_num_neighbors >= 1000000:
        return np.ones(n, dtype=bool)

    # lexsort keys are reversed: the last key is primary, so sort by
    # (dst, distance, src) -> deterministic nearest-K per destination.
    order = np.lexsort((src, distances, dst))
    dst_sorted = dst[order]
    new_group = np.empty(n, dtype=bool)
    new_group[0] = True
    np.not_equal(dst_sorted[1:], dst_sorted[:-1], out=new_group[1:])
    group_start = np.where(new_group)[0]
    group_id = np.searchsorted(group_start, np.arange(n), side="right") - 1
    rank = np.arange(n) - group_start[group_id]
    keep_sorted = rank < int(max_num_neighbors)
    keep = np.zeros(n, dtype=bool)
    keep[order] = keep_sorted
    return keep


def radius_graph(
    x: paddle.Tensor,
    r: float,
    batch: paddle.Tensor | None = None,
    loop: bool = False,
    max_num_neighbors: int = 32,
) -> paddle.Tensor:
    """Build a radius graph, capping incoming edges per node deterministically.

    Returns ``[src, dst]`` (``[2, num_edges]``), where every node keeps at most
    ``max_num_neighbors`` nearest incoming neighbours.
    """
    x_np = _to_np(x)
    if batch is None:
        batch = paddle.zeros((x_np.shape[0],), dtype="int64")
    b_np = _to_np(batch)

    if x_np.shape[0] <= 1000:
        src, dst, dist = radius_graph_simple(x_np, r, b_np, loop)
    else:
        src, dst, dist = radius_graph_grid(x_np, r, b_np, loop)

    keep = _cap_edges_per_destination(src, dst, dist, max_num_neighbors)
    src, dst = src[keep], dst[keep]
    if src.shape[0] == 0:
        return paddle.zeros([2, 0], dtype="int64")
    return paddle.stack(
        [paddle.to_tensor(src, dtype="int64"), paddle.to_tensor(dst, dtype="int64")],
        axis=0,
    )


@paddle.no_grad()
def radius(
    x: paddle.Tensor,
    y: paddle.Tensor,
    r: float,
    batch_x: paddle.Tensor | None = None,
    batch_y: paddle.Tensor | None = None,
    max_num_neighbors: int = 32,
) -> tuple[paddle.Tensor, paddle.Tensor]:
    """Build a bipartite radius graph from ``x`` (sources) onto ``y`` (destinations).

    Returns ``(dst, src)`` aligned with the historical convention: the first tensor
    indexes ``y`` (e.g. grid points) and the second indexes ``x`` (e.g. atoms).
    Each destination keeps at most ``max_num_neighbors`` nearest sources.
    """
    if x.dtype != y.dtype:
        x, y = x.astype("float64"), y.astype("float64")
    num_x, num_y = x.shape[0], y.shape[0]
    if num_x == 0 or num_y == 0:
        empty = paddle.zeros_like(x[:0, 0], dtype="int64")
        return empty, empty.clone()

    cap = None
    if max_num_neighbors is not None and max_num_neighbors < 1000000:
        cap = int(max_num_neighbors)
    chunk = max(1, (1 << 22) // num_x)
    r2 = paddle.to_tensor(r * r, dtype=x.dtype, place=x.place)
    if batch_x is None:
        batch_x = paddle.zeros_like(x[:, 0], dtype="int64")
    if batch_y is None:
        batch_y = paddle.zeros_like(y[:, 0], dtype="int64")
    batch_x = batch_x.astype("int64").unsqueeze(0)
    batch_y = batch_y.astype("int64").unsqueeze(1)
    x_columns = [x[:, k].unsqueeze(0) for k in range(x.shape[1])]
    dst_list, src_list = [], []
    for start in range(0, num_y, chunk):
        y_chunk = y[start : start + chunk]
        # Sum left to right as NumPy does, so that pairs at the cutoff are the same.
        dist = None
        for k, x_column in enumerate(x_columns):
            diff = y_chunk[:, k : k + 1] - x_column
            dist = diff * diff if dist is None else dist + diff * diff
        mask = (dist <= r2) & (batch_y[start : start + chunk] == batch_x)
        if cap is not None and cap < num_x and int(mask.sum(axis=1).max()) > cap:
            # Stable sort: ties go to the smaller source index.
            key = paddle.where(mask, dist, paddle.full_like(dist, float("inf")))
            rank = paddle.argsort(paddle.argsort(key, axis=1, stable=True), axis=1)
            mask = mask & (rank < cap)
        edges = paddle.nonzero(mask)
        dst_list.append(edges[:, 0] + start)
        src_list.append(edges[:, 1])
    return paddle.concat(dst_list), paddle.concat(src_list)


def radius_graph_simple(
    x_np: np.ndarray,
    r: float,
    batch_np: np.ndarray,
    loop: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Dense radius graph for small node counts.

    Returns ``(src, dst, dist)`` global edge arrays for every within-cutoff pair.
    """
    batch_size = int(batch_np.max()) + 1
    r2 = r * r
    src_list, dst_list, dist_list = [], [], []
    for b in range(batch_size):
        mask = batch_np == b
        idx = np.where(mask)[0]
        if idx.size == 0:
            continue
        sx = x_np[mask]
        d2 = ((sx[:, None, :] - sx[None, :, :]) ** 2).sum(-1)
        adj = d2 <= r2
        if not loop:
            np.fill_diagonal(adj, False)
        row, col = np.where(adj)
        if row.size:
            src_list.append(idx[row])
            dst_list.append(idx[col])
            dist_list.append(d2[row, col])
    return _concat_edges(src_list, dst_list, dist_list)


def radius_graph_grid(
    x_np: np.ndarray,
    r: float,
    batch_np: np.ndarray,
    loop: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Spatially-hashed radius graph for many nodes.

    Returns ``(src, dst, dist)`` global edge arrays for every within-cutoff pair.
    """
    batch_size = int(batch_np.max()) + 1
    r2 = r * r
    src_list, dst_list, dist_list = [], [], []
    for b in range(batch_size):
        mask = batch_np == b
        if not mask.any():
            continue
        sx = x_np[mask]
        idx = np.where(mask)[0]
        n = sx.shape[0]

        min_coords = sx.min(axis=0) - r
        cell_size = r
        cell_of: dict[tuple[int, int, int], list[int]] = {}
        for j in range(n):
            cell_idx = tuple(np.floor((sx[j] - min_coords) / cell_size).astype(int))
            cell_of.setdefault(cell_idx, []).append(j)

        for i in range(n):
            point = sx[i]
            cell_idx = tuple(np.floor((point - min_coords) / cell_size).astype(int))
            nearby: list[int] = []
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        nearby.extend(
                            cell_of.get(
                                (cell_idx[0] + dx, cell_idx[1] + dy, cell_idx[2] + dz),
                                [],
                            )
                        )
            if not nearby:
                continue
            nearby_arr = np.asarray(nearby)
            dists = ((sx[nearby_arr] - point) ** 2).sum(axis=1)
            valid = dists <= r2
            if not loop:
                valid = valid & (nearby_arr != i)
            if valid.any():
                valid_j = nearby_arr[valid]
                src_list.append(np.full(valid_j.shape, idx[i], dtype=np.int64))
                dst_list.append(idx[valid_j])
                dist_list.append(dists[valid])
    return _concat_edges(src_list, dst_list, dist_list)

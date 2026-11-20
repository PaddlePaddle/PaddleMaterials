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

import paddle


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

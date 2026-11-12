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

# This code is adapted from https://github.com/rusty1s/pytorch_scatter/blob/master/torch_scatter/scatter.py

from typing import Literal
from typing import Optional

import paddle

__all__ = [
    "scatter",
    "scatter_argmin",
    "scatter_mean",
    "scatter_min",
    "scatter_sum",
    "scatter_sum_first_order",
]

ReduceType = Literal["sum", "add", "mean", "min"]

# Keep the one-hot fallback for dtypes that index_add does not support on every
# backend (notably complex and CPU bfloat16). Model aggregation uses the common
# floating-point dtypes below and takes the memory-linear index_add path.
_INDEX_ADD_DTYPES = (
    paddle.float16,
    paddle.float32,
    paddle.float64,
    paddle.int32,
    paddle.int64,
)


def _normalize_dim(src: paddle.Tensor, dim: int) -> int:
    if src.ndim == 0:
        raise ValueError("src must have at least one dimension")
    if not -src.ndim <= dim < src.ndim:
        raise ValueError(f"Invalid dim {dim} for a {src.ndim}-dimensional tensor")
    return dim % src.ndim


def _resolve_dim_size(index: paddle.Tensor, dim_size: Optional[int]) -> int:
    if dim_size is None:
        return 0 if index.numel() == 0 else int(index.max()) + 1
    # No non-negative check here on purpose. Under AST conversion dim_size
    # arrives as a traced value that converts to -1, so the check fires during
    # conversion rather than on a bad call: M3GNet's registered checkpoints fail
    # with "dim_size must be non-negative" inside MainBlock.three_body. A
    # negative value still fails immediately in paddle.zeros below.
    return int(dim_size)


def _zeros(src: paddle.Tensor, shape) -> paddle.Tensor:
    # paddle.zeros allocates on the current device, which is where src lives in
    # every supported workflow. An explicit .to(src.place) would add a
    # device-transfer op that static-graph capture cannot trace.
    return paddle.zeros(shape, dtype=src.dtype)


def _broadcast(
    index: paddle.Tensor,
    src: paddle.Tensor,
    dim: int,
) -> paddle.Tensor:
    dim = _normalize_dim(src, dim)
    if index.ndim == 1:
        for _ in range(dim):
            index = index.unsqueeze(0)
    for _ in range(index.ndim, src.ndim):
        index = index.unsqueeze(-1)
    return index.expand(src.shape)


def scatter_argmin(
    src: paddle.Tensor,
    index: paddle.Tensor,
    dim_size: Optional[int] = None,
) -> paddle.Tensor:
    """Return the source index of the minimum value in each group.

    src and index must be one-dimensional. Empty groups are assigned -1.
    Ties are resolved by selecting the first occurrence in src.
    """
    if src.ndim != 1 or index.ndim != 1 or src.shape[0] != index.shape[0]:
        raise ValueError("src and index must be one-dimensional with equal length")

    dim_size = _resolve_dim_size(index, dim_size)
    out = paddle.full([dim_size], -1, dtype="int64").to(src.place)
    if index.numel() == 0:
        return out

    order = paddle.argsort(src, stable=True)
    groups, first = paddle.unique(index[order], return_index=True)
    return paddle.scatter(out, groups, order[first], overwrite=True)


def scatter_sum_first_order(
    src: paddle.Tensor,
    index: paddle.Tensor,
    dim_size: Optional[int] = None,
) -> paddle.Tensor:
    """Sum source rows with memory linear in the input size.

    This first-axis implementation uses paddle.scatter_nd_add and is intended
    for inference and first-order training. Use scatter_sum for force models
    that require second-order gradients.
    """
    # Only the rank is checked: src.shape[0] is -1 under static-graph capture,
    # so a length comparison would reject valid compiled calls.
    if index.ndim != 1:
        raise ValueError("index must be one-dimensional")

    dim_size = _resolve_dim_size(index, dim_size)
    out = _zeros(src, [dim_size, *src.shape[1:]])
    num_rows = src.shape[0]
    # Only a known size can take the shortcut. A dynamic row count (compiled
    # inference with dynamic axes) is a tensor while tracing, and branching on
    # it would trace a cond; scatter_nd_add returns `out` for empty rows.
    if isinstance(num_rows, int) and num_rows == 0:
        return out + src.sum() * 0
    return paddle.scatter_nd_add(out, index.reshape([-1, 1]), src)


def _first_axis_index(
    row_index: paddle.Tensor,
    index: paddle.Tensor,
) -> paddle.Tensor:
    """Destination row of every source row for a first-axis scatter.

    ``index`` is ``row_index`` broadcast to the source shape. With a dynamic row
    count a one-dimensional ``row_index`` already holds the rows, which keeps a
    reshape and a slice out of the traced graph.
    """
    if isinstance(index.shape[0], int):
        return index.reshape([index.shape[0], -1])[:, 0]
    if row_index.ndim == 1:
        return row_index
    return index.flatten(1)[:, 0]


def _scatter_sum(
    src: paddle.Tensor,
    index: paddle.Tensor,
    dim: int = -1,
    dim_size: Optional[int] = None,
) -> paddle.Tensor:
    dim = _normalize_dim(src, dim)
    dim_size = _resolve_dim_size(index, dim_size)
    row_index = index
    index = _broadcast(index, src, dim)
    size = list(src.shape)
    size[dim] = dim_size
    out = _zeros(src, size)

    # Use index_add on the first axis to keep memory linear in the number of
    # source rows while preserving the second-order gradients used by force
    # training. Other axes retain put_along_axis's broadcast semantics.
    if dim == 0:
        num_rows = src.shape[0]
        # Only a known size can take the shortcut. A dynamic row count (compiled
        # inference with dynamic axes) is a tensor while tracing, and branching
        # on it would trace a cond; index_add returns `out` for empty rows.
        if isinstance(num_rows, int) and num_rows == 0:
            return out + src.sum() * 0
        idx_1d = _first_axis_index(row_index, index)
        if src.dtype in _INDEX_ADD_DTYPES:
            return paddle.index_add(
                x=out,
                index=idx_1d,
                axis=0,
                value=src,
            )

        # Preserve the previous behavior for dtypes without a portable
        # index_add kernel, such as complex and CPU bfloat16.
        one_hot = paddle.nn.functional.one_hot(idx_1d, dim_size).cast(src.dtype)
        flat_out = paddle.matmul(
            one_hot,
            src.reshape([src.shape[0], -1]),
            transpose_x=True,
        )
        return flat_out.reshape(out.shape)

    return paddle.put_along_axis(
        arr=out,
        indices=index,
        values=src,
        axis=dim,
        reduce="add",
    )


def _scatter_mean(
    src: paddle.Tensor,
    index: paddle.Tensor,
    dim: int = -1,
    dim_size: Optional[int] = None,
) -> paddle.Tensor:
    dim = _normalize_dim(src, dim)
    result = _scatter_sum(src, index, dim, dim_size)
    index_dim = min(dim, index.ndim - 1)

    # ones_like keeps the counts on the device of index. Reading .place of a
    # tensor inside a captured function would break the SOT graph.
    ones = paddle.ones_like(index, dtype=src.dtype)
    count = _scatter_sum(ones, index, index_dim, result.shape[dim])
    count = paddle.clip(count, min=1)
    count = _broadcast(count, result, dim)
    if result.is_floating_point():
        return paddle.divide(result, count)
    return paddle.floor_divide(result, count)


def _scatter_min(
    src: paddle.Tensor,
    index: paddle.Tensor,
    dim: int = -1,
    dim_size: Optional[int] = None,
) -> paddle.Tensor:
    dim = _normalize_dim(src, dim)
    dim_size = _resolve_dim_size(index, dim_size)
    index = _broadcast(index, src, dim)
    size = list(src.shape)
    size[dim] = dim_size
    out = paddle.full(size, float("inf"), dtype=src.dtype).to(src.place)
    return paddle.put_along_axis(
        arr=out,
        indices=index,
        values=src,
        axis=dim,
        reduce="amin",
    )


def scatter(
    src: paddle.Tensor,
    index: paddle.Tensor,
    dim: int = -1,
    dim_size: Optional[int] = None,
    reduce: ReduceType = "sum",
) -> paddle.Tensor:
    """Aggregate values by index using the requested reduction."""
    if reduce in {"sum", "add"}:
        return _scatter_sum(src, index, dim, dim_size)
    if reduce == "mean":
        return _scatter_mean(src, index, dim, dim_size)
    if reduce == "min":
        return _scatter_min(src, index, dim, dim_size)
    raise ValueError("reduce must be one of: sum, add, mean, min")


def scatter_sum(
    src: paddle.Tensor,
    index: paddle.Tensor,
    dim: int = -1,
    dim_size: Optional[int] = None,
) -> paddle.Tensor:
    return _scatter_sum(src, index, dim, dim_size)


def scatter_mean(
    src: paddle.Tensor,
    index: paddle.Tensor,
    dim: int = -1,
    dim_size: Optional[int] = None,
) -> paddle.Tensor:
    return _scatter_mean(src, index, dim, dim_size)


def scatter_min(
    src: paddle.Tensor,
    index: paddle.Tensor,
    dim: int = -1,
    dim_size: Optional[int] = None,
) -> paddle.Tensor:
    return _scatter_min(src, index, dim, dim_size)

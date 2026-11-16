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

from functools import wraps
from typing import Optional

import paddle
import pgl

from ppmat.utils.execution import in_compiled_execution


def graph_count(graph: pgl.Graph) -> int:
    """Return the number of graphs in a (batched) PGL graph as a host integer."""
    return graph._graph_node_index.shape[0] - 1


def segment_sum(
    data: paddle.Tensor,
    segment_ids: paddle.Tensor,
    num_segments: Optional[int] = None,
) -> paddle.Tensor:
    """pgl.math.segment_sum with an optional number of output rows.

    Eager execution calls pgl.math.segment_sum and appends zero rows up to
    ``num_segments``. Compiled execution needs ``num_segments`` and sums with
    index_add, because CINN cannot infer the output shape of segment_pool's SUM
    mode. Rows of empty segments are zero in both cases.
    """
    if num_segments is None or not in_compiled_execution():
        return _pad_rows(pgl.math.segment_sum(data, segment_ids), num_segments)
    return _index_add_rows(data, segment_ids, num_segments)


def segment_softmax(
    data: paddle.Tensor,
    segment_ids: paddle.Tensor,
    num_segments: Optional[int] = None,
) -> paddle.Tensor:
    """pgl.math.segment_softmax that compiled execution can capture.

    Eager execution calls pgl.math.segment_softmax. Compiled execution with
    ``num_segments`` gives the same values and gradients: the segment maximum is
    detached instead of computed under paddle.no_grad(), which SOT cannot trace,
    and it comes from put_along_axis instead of segment_pool, whose MAX mode has no
    CINN shape inference. Memory stays linear in the size of ``data``.
    """
    if num_segments is None or not in_compiled_execution():
        return pgl.math.segment_softmax(data, segment_ids)
    index = segment_ids
    for _ in range(data.ndim - 1):
        index = index.unsqueeze(-1)
    data_max = paddle.put_along_axis(
        paddle.full([num_segments, *data.shape[1:]], float("-inf"), dtype=data.dtype),
        index.expand(data.shape),
        data,
        axis=0,
        reduce="amax",
    ).detach()
    data = paddle.exp(data - paddle.gather(data_max, segment_ids, axis=0))
    data_sum = _index_add_rows(data, segment_ids, num_segments)
    return data / paddle.gather(data_sum, segment_ids, axis=0)


def _index_add_rows(data, index, num_rows):
    out = paddle.zeros([num_rows, *data.shape[1:]], dtype=data.dtype)
    return paddle.index_add(out, index, 0, data)


def _pad_rows(out, num_rows):
    if num_rows is None or out.shape[0] == num_rows:
        return out
    if out.shape[0] > num_rows:
        raise ValueError(
            f"segment_ids select {out.shape[0]} segments, more than "
            f"num_segments={num_rows}."
        )
    pad = paddle.zeros([num_rows - out.shape[0], *out.shape[1:]], dtype=out.dtype)
    return paddle.concat([out, pad], axis=0)


def patch_pgl_empty_edge_batch() -> None:
    """Allow PGL 2.2.6 to batch graphs when every graph has no edges."""

    if pgl.__version__ != "2.2.6":
        return

    join_edges = pgl.Graph._join_edges
    if getattr(join_edges, "_ppmat_empty_edge_patch", False):
        return

    @wraps(join_edges)
    def join_edges_with_empty_graphs(graph_list):
        if graph_list and all(len(graph.edges) == 0 for graph in graph_list):
            return graph_list[0].edges.reshape([0, 2])
        return join_edges(graph_list)

    join_edges_with_empty_graphs._ppmat_empty_edge_patch = True
    pgl.Graph._join_edges = staticmethod(join_edges_with_empty_graphs)

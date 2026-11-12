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

import pgl


def graph_count(graph: pgl.Graph) -> int:
    """Return the number of graphs in a (batched) PGL graph as a host integer.

    ``graph.num_graph`` is a device tensor once ``Graph.tensor()`` has run, and
    reading it back with ``int()`` or ``.item()`` breaks the SOT graph under
    ``paddle.jit.to_static``. PGL keeps per-graph node offsets with
    ``num_graph + 1`` entries for single and batched graphs alike, and their
    length is a static shape.
    """
    return graph._graph_node_index.shape[0] - 1


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

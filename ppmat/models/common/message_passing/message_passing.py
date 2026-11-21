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

# This code is adapted from the following repository:
# https://github.com/rusty1s/pytorch_geometric

from collections import OrderedDict
from inspect import Parameter
from typing import List
from typing import Optional
from typing import Set
from typing import Tuple

import paddle

from ppmat.utils import paddle_aux  # noqa
from ppmat.utils.scatter import scatter

from .inspector import Inspector

Adj = paddle.Tensor
Size = Optional[Tuple[int, int]]


def expand_left(src: paddle.Tensor, dim: int, dims: int) -> paddle.Tensor:
    for _ in range(dims + dim if dim < 0 else dim):
        src = src.unsqueeze(axis=0)
    return src


class MessagePassing(paddle.nn.Layer):
    """Base class for creating message passing layers of the form

    .. math::
        \\mathbf{x}_i^{\\prime} = \\gamma_{\\mathbf{\\Theta}} \\left( \\mathbf{x}_i,
        \\square_{j \\in \\mathcal{N}(i)} \\, \\phi_{\\mathbf{\\Theta}}
        \\left(\\mathbf{x}_i, \\mathbf{x}_j,\\mathbf{e}_{j,i}\\right) \\right),

    where :math:`\\square` denotes a differentiable, permutation invariant
    function, *e.g.*, sum, mean, min, max or mul, and
    :math:`\\gamma_{\\mathbf{\\Theta}}` and :math:`\\phi_{\\mathbf{\\Theta}}` denote
    differentiable functions such as MLPs.
    See `here <https://pytorch-geometric.readthedocs.io/en/latest/notes/
    create_gnn.html>`__ for the accompanying tutorial.

    Args:
        aggr (string, optional): The aggregation scheme to use
            (:obj:`"add"`, :obj:`"mean"`, :obj:`"min"`, :obj:`"max"`,
            :obj:`"mul"` or :obj:`None`). (default: :obj:`"add"`)
        flow (string, optional): The flow direction of message passing
            (:obj:`"source_to_target"` or :obj:`"target_to_source"`).
            (default: :obj:`"source_to_target"`)
        node_dim (int, optional): The axis along which to propagate.
            (default: :obj:`-2`)
        decomposed_layers (int, optional): The number of feature decomposition
            layers, as introduced in the `"Optimizing Memory Efficiency of
            Graph Neural Networks on Edge Computing Platforms"
            <https://arxiv.org/abs/2104.03058>`_ paper.
            Feature decomposition reduces the peak memory usage by slicing
            the feature dimensions into separated feature decomposition layers
            during GNN aggregation.
            This method can accelerate GNN execution on CPU-based platforms
            (*e.g.*, 2-3x speedup on the
            :class:`~torch_geometric.datasets.Reddit` dataset) for common GNN
            models such as :class:`~torch_geometric.nn.models.GCN`,
            :class:`~torch_geometric.nn.models.GraphSAGE`,
            :class:`~torch_geometric.nn.models.GIN`, etc.
            However, this method is not applicable to all GNN operators
            available, in particular for operators in which message computation
            can not easily be decomposed, *e.g.* in attention-based GNNs.
            The selection of the optimal value of :obj:`decomposed_layers`
            depends both on the specific graph dataset and available hardware
            resources.
            A value of :obj:`2` is suitable in most cases.
            Although the peak memory usage is directly associated with the
            granularity of feature decomposition, the same is not necessarily
            true for execution speedups. (default: :obj:`1`)
    """

    special_args: Set[str] = {
        "edge_index",
        "adj_t",
        "edge_index_i",
        "edge_index_j",
        "size",
        "size_i",
        "size_j",
        "ptr",
        "index",
        "dim_size",
    }

    def __init__(
        self,
        aggr: Optional[str] = "add",
        flow: str = "source_to_target",
        node_dim: int = -2,
        decomposed_layers: int = 1,
    ):
        super().__init__()
        self.aggr = aggr
        assert self.aggr in ["add", "sum", "mean", "min", "max", "mul", None]
        self.flow = flow
        assert self.flow in ["source_to_target", "target_to_source"]
        self.node_dim = node_dim
        self.decomposed_layers = decomposed_layers
        self.inspector = Inspector(self)
        self.inspector.inspect(self.message)
        self.inspector.inspect(self.aggregate, pop_first=True)
        self.inspector.inspect(self.message_and_aggregate, pop_first=True)
        self.inspector.inspect(self.update, pop_first=True)
        self.inspector.inspect(self.edge_update)
        # __collect__ iterates these names on every call. Iterating a set breaks
        # the SOT graph, so keep them in a tuple.
        self.__user_args__ = tuple(
            k
            for k in self.inspector.ordered_keys(["message", "aggregate", "update"])
            if k not in self.special_args
        )
        self.__fused_user_args__ = tuple(
            k
            for k in self.inspector.ordered_keys(["message_and_aggregate", "update"])
            if k not in self.special_args
        )
        self.__edge_user_args__ = tuple(
            k
            for k in self.inspector.ordered_keys(["edge_update"])
            if k not in self.special_args
        )
        self.fuse = self.inspector.implements("message_and_aggregate")
        self._explain = False
        self._edge_mask = None
        self._loop_mask = None
        self._apply_sigmoid = True
        self._propagate_forward_pre_hooks = OrderedDict()
        self._propagate_forward_hooks = OrderedDict()
        self._message_forward_pre_hooks = OrderedDict()
        self._message_forward_hooks = OrderedDict()
        self._aggregate_forward_pre_hooks = OrderedDict()
        self._aggregate_forward_hooks = OrderedDict()
        self._message_and_aggregate_forward_pre_hooks = OrderedDict()
        self._message_and_aggregate_forward_hooks = OrderedDict()
        self._edge_update_forward_pre_hooks = OrderedDict()
        self._edge_update_forward_hooks = OrderedDict()

    def __check_input__(self, edge_index, size):
        the_size: List[Optional[int]] = [None, None]
        if isinstance(edge_index, paddle.Tensor):
            assert edge_index.dtype == paddle.int64
            assert edge_index.dim() == 2
            assert edge_index.shape[0] == 2
            if size is not None:
                the_size[0] = size[0]
                the_size[1] = size[1]
            return the_size
        raise ValueError("`MessagePassing.propagate` only supports `paddle.Tensor`.")

    def __set_size__(self, size: List[Optional[int]], dim: int, src: paddle.Tensor):
        the_size = size[dim]
        if the_size is None:
            size[dim] = src.shape[self.node_dim]
        elif the_size != src.shape[self.node_dim]:
            raise ValueError(
                f"Encountered tensor with size {src.shape[self.node_dim]} in "
                f"dimension {self.node_dim}, but expected size {the_size}."
            )

    def __lift__(self, src, edge_index, dim):
        if isinstance(edge_index, paddle.Tensor):
            index = edge_index[dim]
            if index.is_contiguous():
                index = index.contiguous()

            return src.index_select(axis=self.node_dim, index=index)
        raise ValueError

    def __collect__(self, args, edge_index, size, kwargs):
        i, j = (1, 0) if self.flow == "source_to_target" else (0, 1)
        out = {}
        for arg in args:
            if arg[-2:] not in ["_i", "_j"]:
                out[arg] = kwargs.get(arg, Parameter.empty)
            else:
                dim = j if arg[-2:] == "_j" else i
                data = kwargs.get(arg[:-2], Parameter.empty)
                if isinstance(data, (tuple, list)):
                    assert len(data) == 2
                    if isinstance(data[1 - dim], paddle.Tensor):
                        self.__set_size__(size, 1 - dim, data[1 - dim])
                    data = data[dim]
                if isinstance(data, paddle.Tensor):
                    self.__set_size__(size, dim, data)
                    data = self.__lift__(data, edge_index, dim)
                out[arg] = data
        if isinstance(edge_index, paddle.Tensor):
            out["adj_t"] = None
            out["edge_index"] = edge_index
            out["edge_index_i"] = edge_index[i]
            out["edge_index_j"] = edge_index[j]
            out["ptr"] = None
        out["index"] = out["edge_index_i"]
        out["size"] = size
        out["size_i"] = size[1] if size[1] is not None else size[0]
        out["size_j"] = size[0] if size[0] is not None else size[1]
        out["dim_size"] = out["size_i"]
        return out

    def propagate(self, edge_index: Adj, size: Size = None, **kwargs):
        """The initial call to start propagating messages.

        Args:
            edge_index (Tensor or SparseTensor): A :obj:`torch.LongTensor` or a
                :obj:`torch_sparse.SparseTensor` that defines the underlying
                graph connectivity/message passing flow.
                :obj:`edge_index` holds the indices of a general (sparse)
                assignment matrix of shape :obj:`[N, M]`.
                If :obj:`edge_index` is of type :obj:`torch.LongTensor`, its
                shape must be defined as :obj:`[2, num_messages]`, where
                messages from nodes in :obj:`edge_index[0]` are sent to
                nodes in :obj:`edge_index[1]`
                (in case :obj:`flow="source_to_target"`).
                If :obj:`edge_index` is of type
                :obj:`torch_sparse.SparseTensor`, its sparse indices
                :obj:`(row, col)` should relate to :obj:`row = edge_index[1]`
                and :obj:`col = edge_index[0]`.
                The major difference between both formats is that we need to
                input the *transposed* sparse adjacency matrix into
                :func:`propagate`.
            size (tuple, optional): The size :obj:`(N, M)` of the assignment
                matrix in case :obj:`edge_index` is a :obj:`LongTensor`.
                If set to :obj:`None`, the size will be automatically inferred
                and assumed to be quadratic.
                This argument is ignored in case :obj:`edge_index` is a
                :obj:`torch_sparse.SparseTensor`. (default: :obj:`None`)
            **kwargs: Any additional data which is needed to construct and
                aggregate messages, and to update node embeddings.
        """
        decomposed_layers = 1 if self._explain else self.decomposed_layers
        for hook in self._propagate_forward_pre_hooks.values():
            res = hook(self, (edge_index, size, kwargs))
            if res is not None:
                edge_index, size, kwargs = res
        size = self.__check_input__(edge_index, size)
        if isinstance(edge_index, paddle.Tensor) or not self.fuse:
            if decomposed_layers > 1:
                user_args = self.__user_args__
                decomp_args = {a[:-2] for a in user_args if a[-2:] == "_j"}
                decomp_kwargs = {
                    a: kwargs[a].chunk(chunks=decomposed_layers, axis=-1)
                    for a in decomp_args
                }
                decomp_out = []
            for i in range(decomposed_layers):
                if decomposed_layers > 1:
                    for arg in decomp_args:
                        kwargs[arg] = decomp_kwargs[arg][i]
                coll_dict = self.__collect__(
                    self.__user_args__, edge_index, size, kwargs
                )
                msg_kwargs = self.inspector.distribute("message", coll_dict)
                for hook in self._message_forward_pre_hooks.values():
                    res = hook(self, (msg_kwargs,))
                    if res is not None:
                        msg_kwargs = res[0] if isinstance(res, tuple) else res
                out = self.message(**msg_kwargs)
                for hook in self._message_forward_hooks.values():
                    res = hook(self, (msg_kwargs,), out)
                    if res is not None:
                        out = res
                if self._explain:
                    edge_mask = self._edge_mask
                    if self._apply_sigmoid:
                        edge_mask = edge_mask.sigmoid()
                    if out.shape[self.node_dim] != edge_mask.shape[0]:
                        edge_mask = edge_mask[self._loop_mask]
                        loop = paddle.ones(shape=size[0], dtype=edge_mask.dtype)
                        edge_mask = paddle.concat(x=[edge_mask, loop], axis=0)
                    assert out.shape[self.node_dim] == edge_mask.shape[0]
                    out = out * edge_mask.view([-1] + [1] * (out.dim() - 1))
                aggr_kwargs = self.inspector.distribute("aggregate", coll_dict)
                for hook in self._aggregate_forward_pre_hooks.values():
                    res = hook(self, (aggr_kwargs,))
                    if res is not None:
                        aggr_kwargs = res[0] if isinstance(res, tuple) else res
                out = self.aggregate(out, **aggr_kwargs)
                for hook in self._aggregate_forward_hooks.values():
                    res = hook(self, (aggr_kwargs,), out)
                    if res is not None:
                        out = res
                update_kwargs = self.inspector.distribute("update", coll_dict)
                out = self.update(out, **update_kwargs)
                if decomposed_layers > 1:
                    decomp_out.append(out)
            if decomposed_layers > 1:
                out = paddle.concat(x=decomp_out, axis=-1)
        for hook in self._propagate_forward_hooks.values():
            res = hook(self, (edge_index, size, kwargs), out)
            if res is not None:
                out = res
        return out

    def edge_updater(self, edge_index: Adj, **kwargs):
        """The initial call to compute or update features for each edge in the
        graph.

        Args:
            edge_index (Tensor or SparseTensor): A :obj:`torch.LongTensor` or a
                :obj:`torch_sparse.SparseTensor` that defines the underlying
                graph connectivity/message passing flow.
                See :meth:`propagate` for more information.
            **kwargs: Any additional data which is needed to compute or update
                features for each edge in the graph.
        """
        for hook in self._edge_update_forward_pre_hooks.values():
            res = hook(self, (edge_index, kwargs))
            if res is not None:
                edge_index, kwargs = res
        size = self.__check_input__(edge_index, size=None)
        coll_dict = self.__collect__(self.__edge_user_args__, edge_index, size, kwargs)
        edge_kwargs = self.inspector.distribute("edge_update", coll_dict)
        out = self.edge_update(**edge_kwargs)
        for hook in self._edge_update_forward_hooks.values():
            res = hook(self, (edge_index, kwargs), out)
            if res is not None:
                out = res
        return out

    def message(self, x_j: paddle.Tensor) -> paddle.Tensor:
        """Constructs messages from node :math:`j` to node :math:`i`
        in analogy to :math:`\\phi_{\\mathbf{\\Theta}}` for each edge in
        :obj:`edge_index`.
        This function can take any argument as input which was initially
        passed to :meth:`propagate`.
        Furthermore, tensors passed to :meth:`propagate` can be mapped to the
        respective nodes :math:`i` and :math:`j` by appending :obj:`_i` or
        :obj:`_j` to the variable name, *.e.g.* :obj:`x_i` and :obj:`x_j`.
        """
        return x_j

    def aggregate(
        self,
        inputs: paddle.Tensor,
        index: paddle.Tensor,
        ptr: Optional[paddle.Tensor] = None,
        dim_size: Optional[int] = None,
    ) -> paddle.Tensor:
        """Aggregates messages from neighbors as
        :math:`\\square_{j \\in \\mathcal{N}(i)}`.

        Takes in the output of message computation as first argument and any
        argument which was initially passed to :meth:`propagate`.

        By default, this function will delegate its call to scatter functions
        that support "add", "mean", "min", "max" and "mul" operations as
        specified in :meth:`__init__` by the :obj:`aggr` argument.
        """
        if ptr is not None:
            # ptr = expand_left(ptr, dim=self.node_dim, dims=inputs.dim())
            # return segment_csr(inputs, ptr, reduce=self.aggr)
            raise NotImplementedError()
        else:
            return scatter(
                inputs, index, dim=self.node_dim, dim_size=dim_size, reduce=self.aggr
            )

    def message_and_aggregate(self, adj_t) -> paddle.Tensor:
        """Fuses computations of :func:`message` and :func:`aggregate` into a
        single function.
        If applicable, this saves both time and memory since messages do not
        explicitly need to be materialized.
        This function will only gets called in case it is implemented and
        propagation takes place based on a :obj:`torch_sparse.SparseTensor`.
        """
        raise NotImplementedError

    def update(self, inputs: paddle.Tensor) -> paddle.Tensor:
        """Updates node embeddings in analogy to
        :math:`\\gamma_{\\mathbf{\\Theta}}` for each node
        :math:`i \\in \\mathcal{V}`.
        Takes in the output of aggregation as first argument and any argument
        which was initially passed to :meth:`propagate`.
        """
        return inputs

    def edge_update(self) -> paddle.Tensor:
        """Computes or updates features for each edge in the graph.
        This function can take any argument as input which was initially passed
        to :meth:`edge_updater`.
        Furthermore, tensors passed to :meth:`edge_updater` can be mapped to
        the respective nodes :math:`i` and :math:`j` by appending :obj:`_i` or
        :obj:`_j` to the variable name, *.e.g.* :obj:`x_i` and :obj:`x_j`.
        """
        raise NotImplementedError

    def __repr__(self) -> str:
        if hasattr(self, "in_channels") and hasattr(self, "out_channels"):
            return f"{self.__class__.__name__}({self.in_channels}, {self.out_channels})"
        return f"{self.__class__.__name__}()"

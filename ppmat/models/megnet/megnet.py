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

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Callable
from typing import Literal
from typing import Optional

import paddle
import paddle.nn as nn
import pgl
from paddle.nn import Linear
from pgl.math import segment_sum

from ppmat.models.common import initializer
from ppmat.models.common.runtime import RuntimeMixin
from ppmat.models.common.runtime import runtime_boundary
from ppmat.utils import logger
from ppmat.utils.pgl_compat import graph_count


def segment_softmax(data, segment_ids, num_segments=None):
    """Segment softmax with the same values and gradients as pgl.math.segment_softmax.

    PGL computes the segment max under ``paddle.no_grad()``, and SOT cannot trace
    a ``with`` block, so the max is detached instead. When ``num_segments`` is
    given, segment_pool is not used either: CINN's symbolic shape inference for
    segment_pool only handles MEAN and fails for MAX and SUM. The per-segment max
    then comes from a one-hot membership mask, which needs
    ``len(data) * num_segments`` elements of memory.
    """
    if num_segments is None:
        data_max = paddle.geometric.segment_max(data, segment_ids).detach()
    else:
        # [N, S, D] -> [S, D]
        member = paddle.nn.functional.one_hot(segment_ids, num_segments)
        masked = paddle.where(
            member.astype("bool").unsqueeze(-1),
            data.unsqueeze(1),
            paddle.full([1], float("-inf"), dtype=data.dtype),
        )
        data_max = paddle.max(masked, axis=0).detach()
    data_max = paddle.gather(data_max, segment_ids, axis=0)
    data = paddle.exp(data - data_max)
    sum_data = _segment_sum(data, segment_ids, num_segments)
    sum_data = paddle.gather(sum_data, segment_ids, axis=0)
    return data / sum_data


def _segment_sum(data, segment_ids, num_segments=None):
    """Segment sum; uses index_add instead of segment_pool when num_segments is given.

    See segment_softmax for why segment_pool is avoided. The sum can differ from
    segment_pool in float32 rounding because the accumulation order differs.
    """
    if num_segments is None:
        return segment_sum(data, segment_ids)
    out = paddle.zeros([num_segments, *data.shape[1:]], dtype=data.dtype)
    return paddle.index_add(out, segment_ids, 0, data)


class GaussianExpansion(paddle.nn.Layer):
    """Gaussian Radial Expansion.

    The bond distance is expanded to a vector of shape [m], where m is the number of
    Gaussian basis centers.
    """

    def __init__(
        self,
        initial: float = 0.0,
        final: float = 4.0,
        num_centers: int = 20,
        width: (None | float) = 0.5,
    ):
        """
        Args:
            initial: Location of initial Gaussian basis center.
            final: Location of final Gaussian basis center
            num_centers: Number of Gaussian Basis functions
            width: Width of Gaussian Basis functions.
        """
        super().__init__()
        out_0 = paddle.create_parameter(
            shape=paddle.linspace(start=initial, stop=final, num=num_centers).shape,
            dtype=paddle.linspace(start=initial, stop=final, num=num_centers)
            .numpy()
            .dtype,
            default_initializer=paddle.nn.initializer.Assign(
                paddle.linspace(start=initial, stop=final, num=num_centers)
            ),
        )
        out_0.stop_gradient = not False
        self.centers = out_0
        if width is None:
            self.width = 1.0 / paddle.diff(x=self.centers).mean()
        else:
            self.width = width

    def reset_parameters(self):
        """Reinitialize model parameters."""
        out_1 = paddle.create_parameter(
            shape=self.centers.shape,
            dtype=self.centers.numpy().dtype,
            default_initializer=paddle.nn.initializer.Assign(self.centers),
        )
        out_1.stop_gradient = not False
        self.centers = out_1

    def forward(self, bond_dists):
        """Expand distances.

        Args:
            bond_dists :
                Bond (edge) distances between two atoms (nodes)

        Returns:
            A vector of expanded distance with shape [num_centers]
        """
        diff = bond_dists[:, None] - self.centers[None, :]
        return paddle.exp(x=-self.width * diff**2)


class BondExpansion(paddle.nn.Layer):
    """Expand pair distances into a set of spherical bessel or gaussian functions."""

    def __init__(
        self,
        max_l: int = 3,
        max_n: int = 3,
        cutoff: float = 5.0,
        rbf_type: Literal["SphericalBessel", "Gaussian"] = "SphericalBessel",
        smooth: bool = False,
        initial: float = 0.0,
        final: float = 5.0,
        num_centers: int = 100,
        width: float = 0.5,
    ) -> None:
        """
        Args:
            max_l (int): order of angular part
            max_n (int): order of radial part
            cutoff (float): cutoff radius
            rbf_type (str): type of radial basis function .i.e.
                either "SphericalBessel" or 'Gaussian'
            smooth (bool): whether apply the smooth version of spherical bessel
                functions or not
            initial (float): initial point for gaussian expansion
            final (float): final point for gaussian expansion
            num_centers (int): Number of centers for gaussian expansion.
            width (float): width of gaussian function.
        """
        super().__init__()
        self.max_n = max_n
        self.cutoff = cutoff
        self.max_l = max_l
        self.smooth = smooth
        self.num_centers = num_centers
        self.width = width
        self.initial = initial
        self.final = final
        self.rbf_type = rbf_type
        if rbf_type.lower() == "sphericalbessel":
            raise NotImplementedError("Not implemented yet")
        elif rbf_type.lower() == "gaussian":
            self.rbf = GaussianExpansion(initial, final, num_centers, width)
        else:
            raise ValueError(
                "Undefined rbf_type, please use SphericalBessel or Gaussian instead."
            )

    def forward(self, bond_dist: paddle.Tensor):
        """Forward.

        Args:
        bond_dist: Bond distance

        Return:
        bond_basis: Radial basis functions
        """
        bond_basis = self.rbf(bond_dist)
        return bond_basis


class MLP(paddle.nn.Layer):
    """An implementation of a multi-layer perceptron."""

    def __init__(
        self,
        dims: Sequence[int],
        activation: (Callable[[paddle.Tensor], paddle.Tensor] | None) = None,
        activate_last: bool = False,
        bias_last: bool = True,
    ) -> None:
        super().__init__()
        self._depth = len(dims) - 1
        self.layers = paddle.nn.LayerList()
        bias_attr = paddle.ParamAttr(initializer=paddle.nn.initializer.XavierNormal())
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            if i < self._depth - 1:
                self.layers.append(
                    paddle.nn.Linear(
                        in_features=in_dim, out_features=out_dim, bias_attr=bias_attr
                    )
                )
                if activation is not None:
                    self.layers.append(activation)
            else:
                if bias_last:
                    bias_last = bias_attr
                self.layers.append(
                    paddle.nn.Linear(
                        in_features=in_dim, out_features=out_dim, bias_attr=bias_last
                    )
                )
                if activation is not None and activate_last:
                    self.layers.append(activation)

    @property
    def last_linear(self) -> (Linear | None):
        """:return: The last linear layer."""
        for layer in reversed(self.layers):
            if isinstance(layer, paddle.nn.Linear):
                return layer
        raise RuntimeError

    @property
    def depth(self) -> int:
        """Returns depth of MLP."""
        return self._depth

    @property
    def in_features(self) -> int:
        """Return input features of MLP."""
        return self.layers[0].in_features

    @property
    def out_features(self) -> int:
        """Returns output features of MLP."""
        for layer in reversed(self.layers):
            if isinstance(layer, paddle.nn.Linear):
                return layer.out_features
        raise RuntimeError

    def forward(self, inputs):
        """Applies all layers in turn."""
        x = inputs
        for layer in self.layers:
            x = layer(x)
        return x


class SoftPlus2(paddle.nn.Layer):
    """SoftPlus2 activation function:
    out = log(exp(x)+1) - log(2)
    softplus function that is 0 at x=0, the implementation aims at avoiding overflow.
    """

    def __init__(self) -> None:
        """Initializes the SoftPlus2 class."""
        super().__init__()
        self.ssp = paddle.nn.Softplus()

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        """Evaluate activation function given the input tensor x.

        Args:
            x (paddle.tensor): Input tensor

        Returns:
            out (paddle.tensor): Output tensor
        """
        return self.ssp(x) - math.log(2.0)


class EmbeddingBlock(paddle.nn.Layer):
    """Embedding block for generating node, bond and state features."""

    def __init__(
        self,
        degree_rbf: int,
        activation: paddle.nn.Layer,
        dim_node_embedding: int,
        dim_edge_embedding: (int | None) = None,
        dim_state_feats: (int | None) = None,
        ntypes_node: (int | None) = None,
        include_state: bool = False,
        dim_state_embedding: (int | None) = None,
    ):
        """
        Args:
            degree_rbf (int): number of rbf
            activation (nn.Module): activation type
            dim_node_embedding (int): dimensionality of node features
            dim_edge_embedding (int): dimensionality of edge features
            dim_state_feats: dimensionality of state features
            ntypes_node: number of node labels
            include_state: Whether to include state embedding
            dim_state_embedding: dimensionality of state embedding.
        """
        super().__init__()
        self.include_state = include_state
        self.dim_node_embedding = dim_node_embedding
        self.dim_edge_embedding = dim_edge_embedding
        self.dim_state_feats = dim_state_feats
        self.ntypes_node = ntypes_node
        self.dim_state_embedding = dim_state_embedding
        self.activation = activation

        self.layer_node_embedding = paddle.nn.Embedding(
            num_embeddings=ntypes_node, embedding_dim=dim_node_embedding
        )

        if dim_edge_embedding is not None:
            dim_edges = [degree_rbf, dim_edge_embedding]
            self.layer_edge_embedding = MLP(
                dim_edges, activation=activation, activate_last=True
            )

    def forward(self, node_attr, edge_attr, state_attr):
        """Output embedded features.

        Args:
            node_attr: node attribute
            edge_attr: edge attribute
            state_attr: state attribute

        Returns:
            node_feat: embedded node features
            edge_feat: embedded edge features
            state_feat: embedded state features
        """
        if self.ntypes_node is not None:
            node_feat = self.layer_node_embedding(node_attr)
        else:
            node_feat = self.layer_node_embedding(node_attr.to("float32"))
        if self.dim_edge_embedding is not None:
            edge_feat = self.layer_edge_embedding(edge_attr.to("float32"))
        else:
            edge_feat = edge_attr
        if self.include_state is True:
            state_feat = state_attr
        else:
            state_feat = None
        return node_feat, edge_feat, state_feat


class MEGNetGraphConv(paddle.nn.Layer):
    """A MEGNet graph convolution layer in DGL."""

    def __init__(
        self,
        edge_func: paddle.nn.Layer,
        node_func: paddle.nn.Layer,
        state_func: paddle.nn.Layer,
    ) -> None:
        """
        Args:
            edge_func: Edge update function.
            node_func: Node update function.
            state_func: Global state update function.
        """
        super().__init__()
        self.edge_func = edge_func
        self.node_func = node_func
        self.state_func = state_func

    @staticmethod
    def from_dims(
        edge_dims: list[int],
        node_dims: list[int],
        state_dims: list[int],
        activation: paddle.nn.Layer,
    ) -> MEGNetGraphConv:
        """Create a MEGNet graph convolution layer from dimensions.

        Args:
            edge_dims (list[int]): Edge dimensions.
            node_dims (list[int]): Node dimensions.
            state_dims (list[int]): State dimensions.
            activation (Module): Activation function.

        Returns:
            MEGNetGraphConv: MEGNet graph convolution layer.
        """
        edge_update = MLP(edge_dims, activation, activate_last=True)
        node_update = MLP(node_dims, activation, activate_last=True)
        attr_update = MLP(state_dims, activation, activate_last=True)
        return MEGNetGraphConv(edge_update, node_update, attr_update)

    def edge_update(self, graph, node_feat, edge_feat, u):
        vi = node_feat[graph.edges[:, 0]]  # shape: [num_edges, 32]
        vj = node_feat[graph.edges[:, 1]]  # shape: [num_edges, 32]
        u = u[graph.edges[:, 0]]  # shape: [num_edges, 32]
        edge_feat = paddle.concat(
            [vi, vj, edge_feat, u], axis=1
        )  # shape: [num_edges, 32+32+32+32] = [num_edges, 128]
        edge_feat = self.edge_func(
            edge_feat
        )  # input shape: [num_edges, 128], out shape: [num_edges, 32]
        return edge_feat

    def node_update(self, graph, node_feat, edge_feat, u):
        src, dst, eid = graph.sorted_edges(sort_by="dst")
        # node_feat_e = paddle.geometric.segment_mean(edge_feat[eid], dst)
        # graph.num_nodes is a device tensor after Graph.tensor(); reading it with
        # .item() breaks the SOT graph, so use the row count of node_feat.
        node_feat_e = self.sorted_segment_mean(edge_feat[eid], dst, node_feat.shape[0])
        node_feat = paddle.concat([node_feat, node_feat_e, u], axis=1)
        node_feat = self.node_func(node_feat)
        return node_feat

    def state_update(self, graph, node_feat, edge_feat, state_feat):
        u_edge_feat = paddle.geometric.segment_mean(edge_feat, graph.graph_edge_id)
        u_node_feat = paddle.geometric.segment_mean(node_feat, graph.graph_node_id)
        state = paddle.concat([state_feat, u_edge_feat, u_node_feat], axis=1)
        state_feat = self.state_func(state)
        return state_feat

    @staticmethod
    def sorted_segment_mean(data, segment_ids, num_segments):
        """
        Custom Paddle op to replicate TensorFlow's `unsorted_segment_sum`

        Args:
            data (Tensor [N, D]):      N is the number of edges, and D is the feature
                dimension.
            segment_ids (Tensor [N]):  Each value is the segment ID (int).
            num_segments (int):        Total number of segments (e.g., number of nodes).

        Returns:
            Tensor [num_segments, D]
        """

        feat_dim = data.shape[1]

        # Initialize output and count tensors
        result = paddle.zeros(shape=[num_segments, feat_dim])
        count = paddle.zeros(shape=[num_segments, feat_dim])

        # Expand segment_ids to match data's shape [N, D]
        segment_ids_exp = paddle.unsqueeze(segment_ids, axis=1)
        segment_ids_exp = paddle.expand(segment_ids_exp, shape=[-1, feat_dim])

        # Accumulate values into result tensor (Note: must assign the returned tensor)
        result = paddle.put_along_axis(
            arr=result, indices=segment_ids_exp, values=data, axis=0, reduce="add"
        )

        # Accumulate counts
        ones = paddle.ones_like(data)
        count = paddle.put_along_axis(
            arr=count, indices=segment_ids_exp, values=ones, axis=0, reduce="add"
        )
        count = paddle.clip(count, min=1.0)

        return result / count

    def forward(
        self,
        graph: pgl.Graph,
        edge_feat: paddle.Tensor,
        node_feat: paddle.Tensor,
        state_feat: paddle.Tensor,
    ) -> tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]:
        """Perform sequence of edge->node->attribute updates.

        Args:
            graph: Input g
            edge_feat: Edge features, shape: [num_edges, 32]
            node_feat: Node features, shape: [num_nodes, 32]
            state_feat: Graph attributes (global state)

        Returns:
            (edge features, node features, graph attributes)
        """
        batch_num_nodes = graph._graph_node_index
        batch_num_nodes = batch_num_nodes[1:] - batch_num_nodes[:-1]
        u = paddle.repeat_interleave(
            state_feat, batch_num_nodes, axis=0
        )  # shape: [num_nodes, 32]

        edge_feat = self.edge_update(graph, node_feat, edge_feat, u)
        node_feat = self.node_update(graph, node_feat, edge_feat, u)
        state_feat = self.state_update(graph, node_feat, edge_feat, state_feat)
        return edge_feat, node_feat, state_feat


class MEGNetBlock(paddle.nn.Layer):
    """A MEGNet block comprising a sequence of update operations."""

    def __init__(
        self,
        dims: list[int],
        conv_hiddens: list[int],
        act: paddle.nn.Layer,
        dropout: (float | None) = None,
        skip: bool = True,
    ) -> None:
        """
        Init the MEGNet block with key parameters.

        Args:
            dims: Dimension of dense layers before graph convolution.
            conv_hiddens: Architecture of hidden layers of graph convolution.
            act: Activation type.
            dropout: Randomly zeroes some elements in the input tensor with given
                probability (0 < x < 1) according to a Bernoulli distribution.
            skip: Residual block.
        """
        super().__init__()
        self.has_dense = len(dims) > 1
        self.activation = act
        conv_dim = dims[-1]
        out_dim = conv_hiddens[-1]
        mlp_kwargs = {
            "dims": dims,
            "activation": self.activation,
            "activate_last": True,
            "bias_last": True,
        }
        self.edge_func = MLP(**mlp_kwargs) if self.has_dense else paddle.nn.Identity()
        self.node_func = MLP(**mlp_kwargs) if self.has_dense else paddle.nn.Identity()
        self.state_func = MLP(**mlp_kwargs) if self.has_dense else paddle.nn.Identity()
        edge_in = 2 * conv_dim + conv_dim + conv_dim
        node_in = out_dim + conv_dim + conv_dim
        attr_in = out_dim + out_dim + conv_dim
        self.conv = MEGNetGraphConv.from_dims(
            edge_dims=[edge_in, *conv_hiddens],
            node_dims=[node_in, *conv_hiddens],
            state_dims=[attr_in, *conv_hiddens],
            activation=self.activation,
        )
        self.dropout = paddle.nn.Dropout(p=dropout) if dropout else None
        self.skip = skip

    def forward(
        self,
        graph: pgl.Graph,
        edge_feat: paddle.Tensor,
        node_feat: paddle.Tensor,
        state_feat: paddle.Tensor,
    ) -> tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]:
        """MEGNetBlock forward pass.

        Args:
            graph (pgl.Graph): A Graph.
            edge_feat (Tensor): Edge features, shape [num_edges, 32]
            node_feat (Tensor): Node features, shape [num_nodes, 32]
            state_feat (Tensor): Graph attributes (global state), shape [batch_size, 32]

        Returns:
            tuple[Tensor, Tensor, Tensor]: Updated (edge features,
                node features, graph attributes)
        """
        inputs = edge_feat, node_feat, state_feat
        edge_feat = self.edge_func(edge_feat)
        node_feat = self.node_func(node_feat)
        state_feat = self.state_func(state_feat)
        edge_feat, node_feat, state_feat = self.conv(
            graph, edge_feat, node_feat, state_feat
        )
        if self.dropout:
            edge_feat = self.dropout(edge_feat)
            node_feat = self.dropout(node_feat)
            state_feat = self.dropout(state_feat)
        if self.skip:
            edge_feat = edge_feat + inputs[0]
            node_feat = node_feat + inputs[1]
            state_feat = state_feat + inputs[2]
        return edge_feat, node_feat, state_feat


class Set2Set(nn.Layer):
    """Implementation of Graph Global Pooling "Set2Set".

    Reference Paper: ORDER MATTERS: SEQUENCE TO SEQUENCE

    Args:
        input_dim (int): dimentional size of input
        n_iters: number of iteration
        n_layers: number of LSTM layers
    Return:
        output_feat: output feature of set2set pooling with shape [batch, 2*dim].
    """

    def __init__(self, input_dim, n_iters, n_layers=1):
        super(Set2Set, self).__init__()
        self.input_dim = input_dim
        self.output_dim = 2 * input_dim
        self.n_iters = n_iters
        self.n_layers = n_layers
        self.lstm = paddle.nn.LSTM(
            input_size=self.output_dim,
            hidden_size=self.input_dim,
            num_layers=n_layers,
            time_major=True,
        )

    def forward(self, graph, x, num_graphs=None):
        """Forward function of Graph Global Pooling "Set2Set".

        Args:
            graph: the graph object from (:code:`Graph`)
            x: A tensor with shape (num_nodes, feature_size).
            num_graphs: Number of graphs as a host integer. When it is None, it is
                read from graph_id on the device.
        Return:
            output_feat: A tensor with shape (num_nodes, output_size).
        """
        graph_id = graph.graph_node_id
        batch_size = graph_id.max() + 1 if num_graphs is None else num_graphs
        h = (
            paddle.zeros((self.n_layers, batch_size, self.input_dim)),
            paddle.zeros((self.n_layers, batch_size, self.input_dim)),
        )
        q_star = paddle.zeros((batch_size, self.output_dim))
        for _ in range(self.n_iters):
            q, h = self.lstm(q_star.unsqueeze(0), h)
            q = q.reshape((batch_size, self.input_dim))
            e = (x * q.index_select(graph_id, axis=0)).sum(axis=-1, keepdim=True)
            a = segment_softmax(e, graph_id, num_graphs)
            r = _segment_sum(a * x, graph_id, num_graphs)
            q_star = paddle.concat([q, r], axis=-1)

        return q_star


class EdgeSet2Set(paddle.nn.Layer):
    """Implementation of Set2Set."""

    def __init__(self, input_dim: int, n_iters: int, n_layers: int) -> None:
        """:param input_dim: The size of each input sample.
        :param n_iters: The number of iterations.
        :param n_layers: The number of recurrent layers.
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = 2 * input_dim
        self.n_iters = n_iters
        self.n_layers = n_layers
        self.lstm = paddle.nn.LSTM(
            input_size=self.output_dim,
            hidden_size=self.input_dim,
            num_layers=n_layers,
            time_major=True,
            direction="forward",
        )

    def forward(self, graph, x, num_graphs=None):
        """Forward function of Graph Global Pooling "Set2Set".

        Args:
            graph: the graph object from (:code:`Graph`)
            x: A tensor with shape (num_nodes, feature_size).
            num_graphs: Number of graphs as a host integer. When it is None, it is
                read from graph_id on the device.
        Return:
            output_feat: A tensor with shape (num_nodes, output_size).
        """
        graph_id = graph.graph_edge_id
        batch_size = graph_id.max() + 1 if num_graphs is None else num_graphs
        h = (
            paddle.zeros((self.n_layers, batch_size, self.input_dim)),
            paddle.zeros((self.n_layers, batch_size, self.input_dim)),
        )
        q_star = paddle.zeros((batch_size, self.output_dim))
        for _ in range(self.n_iters):
            q, h = self.lstm(q_star.unsqueeze(0), h)
            q = q.reshape((batch_size, self.input_dim))
            e = (x * q.index_select(graph_id, axis=0)).sum(axis=-1, keepdim=True)
            a = segment_softmax(e, graph_id, num_graphs)
            r = _segment_sum(a * x, graph_id, num_graphs)
            q_star = paddle.concat([q, r], axis=-1)

        return q_star


class MEGNetPlus(RuntimeMixin, paddle.nn.Layer):
    """MegNet: Graph Networks as a Universal Machine Learning Framework for Molecules
    and Crystals

    https://arxiv.org/abs/1812.05055

    Args:
        dim_node_embedding (int, optional): Dimensionality of node (atom) feature
            embeddings. Defaults to 16.
        dim_edge_embedding (int, optional): Dimensionality of edge (bond) feature
            embeddings. Defaults to 100.
        dim_state_embedding (int, optional): Dimensionality of state (graph-level)
            features. Defaults to 2.
        nblocks (int, optional): Number of graph convolution blocks. Defaults to 3.
        hidden_layer_sizes_input (tuple[int, ...], optional): MLP sizes for input
            feature encoding. Defaults to (64, 32).
        hidden_layer_sizes_conv (tuple[int, ...], optional): MLP sizes for convolution
            layers. Defaults to (64, 64, 32).
        hidden_layer_sizes_output (tuple[int, ...], optional): MLP sizes for output
            head. Defaults to (32, 16).
        nlayers_set2set (int, optional): Number of LSTM layers in Set2Set pooling.
            Defaults to 1.
        niters_set2set (int, optional): Number of Set2Set iterations. Defaults to 2.
        include_state (bool, optional): Whether to include state features in processing.
            Defaults to True.
        dropout (float, optional): Dropout rate for regularization. Defaults to 0.0.
        max_element_types (int, optional): Maximum number of atomic species supported.
            Defaults to 119.
        bond_expansion_cfg (_type_, optional): Radial basis function configuration for
            bond distance encoding. Defaults to None.
        property_name (Optional[str], optional): Target property name for prediction.
            Defaults to "formation_energy_per_atom".
        data_mean (float, optional): Mean of the training data. Defaults to 0.0.
        data_std (float, optional): Standard deviation of the training data. Defaults
            to 1.0.
        execution_backend (str, optional): Numerical execution backend. Use
            ``"eager"`` or ``"cinn"``. Defaults to ``"eager"``.
        runtime_options (dict, optional): Per-backend runtime options.
    """

    def __init__(
        self,
        dim_node_embedding: int = 16,
        dim_edge_embedding: int = 100,
        dim_state_embedding: int = 2,
        nblocks: int = 3,
        hidden_layer_sizes_input: tuple[int, ...] = (64, 32),
        hidden_layer_sizes_conv: tuple[int, ...] = (64, 64, 32),
        hidden_layer_sizes_output: tuple[int, ...] = (32, 16),
        nlayers_set2set: int = 1,
        niters_set2set: int = 2,
        include_state: bool = True,
        dropout: float = 0.0,
        max_element_types: int = 119,
        bond_expansion_cfg=None,
        property_name: Optional[str] = "formation_energy_per_atom",
        data_mean: float = 0.0,
        data_std: float = 1.0,
        execution_backend: str = "eager",
        runtime_options: dict | None = None,
        # loss_cfg:dict = {},
    ):
        # loss = build_loss(loss_cfg)
        super().__init__()
        self._init_runtime(execution_backend, runtime_options)
        self.max_element_types = max_element_types
        if bond_expansion_cfg is None:
            bond_expansion_cfg = {
                "rbf_type": "Gaussian",
                "initial": 0.0,
                "final": 5.0,
                "num_centers": 100,
                "width": 0.5,
            }
            logger.info(f"Using bond expansion configuration: {bond_expansion_cfg}")

        self.bond_expansion = BondExpansion(**bond_expansion_cfg)

        if isinstance(property_name, list):
            self.property_name = property_name[0]
        else:
            assert isinstance(property_name, str)
            self.property_name = property_name
        self.register_buffer(tensor=paddle.to_tensor(data_mean), name="data_mean")
        self.register_buffer(tensor=paddle.to_tensor(data_std), name="data_std")

        node_dims = [dim_node_embedding, *hidden_layer_sizes_input]
        edge_dims = [dim_edge_embedding, *hidden_layer_sizes_input]
        state_dims = [dim_state_embedding, *hidden_layer_sizes_input]

        activation = SoftPlus2()
        self.embedding = EmbeddingBlock(
            degree_rbf=dim_edge_embedding,
            dim_node_embedding=dim_node_embedding,
            ntypes_node=max_element_types,
            include_state=include_state,
            dim_state_embedding=dim_state_embedding,
            activation=activation,
        )
        self.edge_encoder = MLP(edge_dims, activation, activate_last=True)
        self.node_encoder = MLP(node_dims, activation, activate_last=True)
        self.state_encoder = MLP(state_dims, activation, activate_last=True)
        dim_blocks_in = hidden_layer_sizes_input[-1]
        dim_blocks_out = hidden_layer_sizes_conv[-1]
        block_args = {
            "conv_hiddens": hidden_layer_sizes_conv,
            "dropout": dropout,
            "act": activation,
            "skip": True,
        }
        blocks = [MEGNetBlock(dims=[dim_blocks_in], **block_args)] + [
            MEGNetBlock(dims=[dim_blocks_out, *hidden_layer_sizes_input], **block_args)
            for _ in range(nblocks - 1)
        ]
        self.blocks = paddle.nn.LayerList(sublayers=blocks)
        s2s_kwargs = {"n_iters": niters_set2set, "n_layers": nlayers_set2set}
        self.edge_s2s = EdgeSet2Set(dim_blocks_out, **s2s_kwargs)
        self.node_s2s = Set2Set(dim_blocks_out, **s2s_kwargs)

        self.fc_out = MLP(
            dims=[
                2 * 2 * dim_blocks_out + dim_blocks_out,
                *hidden_layer_sizes_output,
                1,
            ],
            activation=activation,
            activate_last=False,
        )

        self.dropout = paddle.nn.Dropout(p=dropout) if dropout else None
        self.include_state_embedding = include_state
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            initializer.linear_init_(m)
        elif isinstance(m, nn.Embedding):
            initializer.normal_(m.weight)
        elif isinstance(m, nn.LSTM):
            initializer.lstm_init_(m)

    @runtime_boundary("forward")
    def _forward(self, data):
        #  The data in data['graph'] is numpy.ndarray, convert it to paddle.Tensor
        g = data["graph"].tensor()
        # print(data["id"])
        batch_size = graph_count(g)
        state_attr = paddle.zeros(
            [batch_size, self.embedding.dim_state_embedding], dtype="float32"
        )
        node_attr = g.node_feat["atom_types"]
        edge_attr = self.bond_expansion(g.edge_feat["bond_dist"])
        node_feat, edge_feat, state_feat = self.embedding(
            node_attr, edge_attr, state_attr
        )
        edge_feat = self.edge_encoder(edge_feat)
        node_feat = self.node_encoder(node_feat)
        state_feat = self.state_encoder(state_feat)
        for block in self.blocks:
            output = block(g, edge_feat, node_feat, state_feat)
            edge_feat, node_feat, state_feat = output
        node_vec = self.node_s2s(g, node_feat, batch_size)
        edge_vec = self.edge_s2s(g, edge_feat, batch_size)

        vec = paddle.concat([node_vec, edge_vec, state_feat], axis=1)
        if self.dropout:
            vec = self.dropout(vec)

        result = self.fc_out(vec)

        return result

    def normalize(self, tensor):
        return (tensor - self.data_mean) / self.data_std

    def unnormalize(self, tensor):
        return tensor * self.data_std + self.data_mean

    def forward(self, data, return_loss=True, return_prediction=True):
        assert (
            return_loss or return_prediction
        ), "At least one of return_loss or return_prediction must be True."
        pred = self._forward(data)

        loss_dict = {}
        if return_loss:
            label = data[self.property_name]
            label = self.normalize(label)
            loss = paddle.nn.functional.mse_loss(
                input=pred,
                label=label,
            )
            loss_dict["loss"] = loss

        prediction = {}
        if return_prediction:
            pred = self.unnormalize(pred)
            prediction[self.property_name] = pred
        return {"loss_dict": loss_dict, "pred_dict": prediction}

    @paddle.no_grad()
    def predict(self, samples):
        is_list = isinstance(samples, list)
        samples = samples if is_list else [samples]

        results = []
        for sample in samples:
            result = self._forward({"graph": sample})
            result = self.unnormalize(result).numpy()[0, 0]
            results.append({self.property_name: result})

        return results if is_list else results[0]

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

from math import log
from math import pi

import paddle
from paddle import nn
from paddle.nn import functional as F

from ppmat.models.common.message_passing.message_passing import MessagePassing
from ppmat.utils.scatter import scatter_sum


class MultiLayerPerceptron(nn.Layer):
    """
    Multi-layer Perceptron.
    Note there is no activation or dropout in the last layer.
    Parameters:
        input_dim (int): input dimension
        hidden_dim (list of int): hidden dimensions
        activation (str or function, optional): activation function
        dropout (float, optional): dropout rate
    """

    def __init__(self, input_dim, hidden_dims, activation="relu", dropout=0):
        super(MultiLayerPerceptron, self).__init__()
        self.dims = [input_dim] + hidden_dims
        if isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            self.activation = None
        if dropout:
            self.dropout = paddle.nn.Dropout(dropout)
        else:
            self.dropout = None
        self.layers = nn.LayerList(
            nn.Linear(a, b) for a, b in zip(self.dims, self.dims[1:])
        )

    def forward(self, input):
        """"""
        x = input
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                if self.activation:
                    x = self.activation(x)
                if self.dropout:
                    x = self.dropout(x)
        return x


class MLPEdgeEncoder(paddle.nn.Layer):
    def __init__(self, hidden_dim=100, activation="relu"):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.bond_emb = paddle.nn.Embedding(100, embedding_dim=self.hidden_dim)
        self.mlp = MultiLayerPerceptron(
            1, [self.hidden_dim, self.hidden_dim], activation=activation
        )

    @property
    def out_channels(self):
        return self.hidden_dim

    def forward(self, edge_length, edge_type):
        """
        Input:
            edge_length: The length of edges, shape=(E, 1).
            edge_type: The type pf edges, shape=(E,)
        Returns:
            edge_attr:  The representation of edges. (E, 2 * num_gaussians)
        """
        d_emb = self.mlp(edge_length)
        edge_attr = self.bond_emb(edge_type)
        return d_emb * edge_attr


class ShiftedSoftplus(nn.Layer):
    def __init__(self):
        super(ShiftedSoftplus, self).__init__()
        self.shift = log(2.0)

    def forward(self, x):
        return paddle.nn.functional.softplus(x) - self.shift


class CFConv(nn.Layer):
    def __init__(self, in_channels, out_channels, num_filters, nn, cutoff, smooth):
        super().__init__()
        self.lin1 = paddle.nn.Linear(in_channels, num_filters, bias_attr=False)
        self.lin2 = paddle.nn.Linear(num_filters, out_channels)
        self.nn = nn
        self.cutoff = cutoff
        self.smooth = smooth
        self.reset_parameters()

    def reset_parameters(self):
        xavier = nn.initializer.XavierUniform()
        xavier(self.lin1.weight)
        xavier(self.lin2.weight)
        nn.initializer.Constant(0)(self.lin2.bias)

    def forward(self, x, edge_index, edge_length, edge_attr):
        if self.smooth:
            C = 0.5 * (paddle.cos(edge_length * pi / self.cutoff) + 1.0)
            within_cutoff = (edge_length <= self.cutoff) & (edge_length >= 0.0)
            C = C * within_cutoff.astype(C.dtype)
        else:
            C = (edge_length <= self.cutoff).astype(edge_length.dtype)
        W = self.nn(edge_attr) * C.reshape([-1, 1])
        x = self.lin1(x)
        src, dst = edge_index
        n_dst = x.shape[0]
        messages = self.message(paddle.gather(x, src, axis=-2), W)
        x = self.lin2(scatter_sum(messages, dst, dim=-2, dim_size=n_dst))
        return x

    def message(self, x_j, W):
        return x_j * W


class InteractionBlock(nn.Layer):
    def __init__(self, hidden_channels, num_gaussians, num_filters, cutoff, smooth):
        super(InteractionBlock, self).__init__()
        mlp = nn.Sequential(
            nn.Linear(num_gaussians, num_filters),
            ShiftedSoftplus(),
            nn.Linear(num_filters, num_filters),
        )
        self.conv = CFConv(
            hidden_channels, hidden_channels, num_filters, mlp, cutoff, smooth
        )
        self.act = ShiftedSoftplus()
        self.lin = nn.Linear(hidden_channels, hidden_channels)

    def forward(self, x, edge_index, edge_length, edge_attr):
        x = self.conv(x, edge_index, edge_length, edge_attr)
        x = self.act(x)
        x = self.lin(x)
        return x


class SchNetEncoder(nn.Layer):
    def __init__(
        self,
        hidden_channels=128,
        num_filters=128,
        num_interactions=6,
        edge_channels=100,
        cutoff=10.0,
        smooth=False,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.num_filters = num_filters
        self.num_interactions = num_interactions
        self.cutoff = cutoff
        self.embedding = paddle.nn.Embedding(100, hidden_channels, max_norm=10.0)
        self.interactions = nn.LayerList()
        for _ in range(num_interactions):
            block = InteractionBlock(
                hidden_channels, edge_channels, num_filters, cutoff, smooth
            )
            self.interactions.append(block)

    def forward(self, z, edge_index, edge_length, edge_attr, embed_node=True):
        if embed_node:
            assert z.ndim == 1 and z.dtype == paddle.int64
            h = self.embedding(z)
        else:
            h = z
        for interaction in self.interactions:
            h = h + interaction(h, edge_index, edge_length, edge_attr)
        return h


class GINEConv(MessagePassing):
    def __init__(self, nn, activation="relu"):
        super().__init__(aggr="add")
        self.nn = nn
        self.activation = getattr(F, activation)
        self.register_buffer("eps", paddle.full([1], 0.0))

    def forward(self, x, edge_index, edge_attr):
        assert edge_attr is not None and x.shape[-1] == edge_attr.shape[-1]
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        return self.nn(out + (1 + self.eps) * x)

    def message(self, x_j: paddle.Tensor, edge_attr: paddle.Tensor) -> paddle.Tensor:
        if self.activation:
            return self.activation(x_j + edge_attr)
        else:
            return x_j + edge_attr

    def __repr__(self):
        return "{}(nn={})".format(self.__class__.__name__, self.nn)


class GINEncoder(paddle.nn.Layer):
    def __init__(
        self,
        hidden_dim,
        num_convs=3,
        activation="relu",
        short_cut=True,
        concat_hidden=False,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_convs = num_convs
        self.short_cut = short_cut
        self.concat_hidden = concat_hidden
        self.node_emb = paddle.nn.Embedding(100, hidden_dim)
        if isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            self.activation = None
        self.convs = paddle.nn.LayerList()
        for i in range(self.num_convs):
            self.convs.append(
                GINEConv(
                    MultiLayerPerceptron(
                        hidden_dim, [hidden_dim, hidden_dim], activation=activation
                    ),
                    activation=activation,
                )
            )

    def forward(self, z, edge_index, edge_attr):
        """
        Input:
            z: atom indices for the batched graph
            node_attr: node feature tensor with shape (num_node, hidden)
            edge_attr: edge feature tensor with shape (num_edge, hidden)
        Output:
            node_attr
            graph feature
        """
        node_attr = self.node_emb(z)
        hiddens = []
        conv_input = node_attr
        for conv_idx, conv in enumerate(self.convs):
            hidden = conv(conv_input, edge_index, edge_attr)
            if conv_idx < len(self.convs) - 1 and self.activation is not None:
                hidden = self.activation(hidden)
            assert hidden.shape == conv_input.shape
            if self.short_cut and hidden.shape == conv_input.shape:
                hidden = hidden + conv_input
            hiddens.append(hidden)
            conv_input = hidden
        if self.concat_hidden:
            node_feature = paddle.concat(hiddens, axis=-1)
        else:
            node_feature = hiddens[-1]
        return node_feature

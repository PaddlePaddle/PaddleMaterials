# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Geometric vector perceptron layers for FlowMol.

GVP splits features into scalar and vector channels and updates the vector
channels with rotation-equivariant linear maps. The graph convolution replaces
DGL message passing with direct edge indexing and ppmat scatter reduction.
The implementation follows https://github.com/lucidrains/geometric-vector-perceptron.
"""

from __future__ import annotations

import math
from typing import Union

import paddle
import paddle.nn as nn

from ppmat.utils.scatter import scatter_mean
from ppmat.utils.scatter import scatter_sum


def norm_no_nan(x, axis=-1, keepdims=False, eps=1e-8, sqrt=True):
    """L2 norm of a tensor clamped above a minimum value ``eps``."""
    out = paddle.clip(
        paddle.sum(paddle.square(x), axis=axis, keepdim=keepdims), min=eps
    )
    return paddle.sqrt(out) if sqrt else out


class GVP(nn.Layer):
    """Geometric vector perceptron."""

    def __init__(
        self,
        dim_vectors_in,
        dim_vectors_out,
        dim_feats_in,
        dim_feats_out,
        n_cp_feats,
        feats_activation=nn.SiLU(),
        vectors_activation=nn.Sigmoid(),
    ):
        super().__init__()
        self.dim_vectors_in = dim_vectors_in
        self.dim_feats_in = dim_feats_in
        self.n_cp_feats = n_cp_feats

        dim_h = max(dim_vectors_in, dim_vectors_out)

        wh_k = 1 / math.sqrt(dim_vectors_in)
        self.Wh = self.create_parameter(
            shape=[dim_vectors_in, dim_h],
            default_initializer=nn.initializer.Uniform(low=-wh_k, high=wh_k),
        )

        if n_cp_feats > 0:
            wcp_k = 1 / math.sqrt(dim_vectors_in)
            self.Wcp = self.create_parameter(
                shape=[dim_vectors_in, n_cp_feats * 2],
                default_initializer=nn.initializer.Uniform(low=-wcp_k, high=wcp_k),
            )

        wu_in_dim = dim_h + n_cp_feats
        wu_k = 1 / math.sqrt(wu_in_dim)
        self.Wu = self.create_parameter(
            shape=[wu_in_dim, dim_vectors_out],
            default_initializer=nn.initializer.Uniform(low=-wu_k, high=wu_k),
        )

        self.vectors_activation = vectors_activation

        self.to_feats_out = nn.Sequential(
            nn.Linear(dim_h + n_cp_feats + dim_feats_in, dim_feats_out),
            feats_activation,
        )
        self.scalar_to_vector_gates = nn.Linear(dim_feats_out, dim_vectors_out)

    def forward(self, feats: paddle.Tensor, vectors: paddle.Tensor):
        n = feats.shape[1]
        v, c = vectors.shape[1], vectors.shape[2]

        if c != 3 or v != self.dim_vectors_in:
            raise ValueError("vectors have wrong dimensions")
        if n != self.dim_feats_in:
            raise ValueError("scalar features have wrong dimensions")

        Vh = paddle.einsum("b v c, v h -> b h c", vectors, self.Wh)

        if self.n_cp_feats > 0:
            Vcp = paddle.einsum("b v c, v p -> b p c", vectors, self.Wcp)
            cp_src, cp_dst = paddle.split(
                Vcp, [self.n_cp_feats, self.n_cp_feats], axis=1
            )
            cp = paddle.cross(cp_src, cp_dst, axis=-1)
            Vh = paddle.concat((Vh, cp), axis=1)

        Vu = paddle.einsum("b h c, h u -> b u c", Vh, self.Wu)

        sh = norm_no_nan(Vh)
        s = paddle.concat((feats, sh), axis=1)

        feats_out = self.to_feats_out(s)
        gating = self.scalar_to_vector_gates(feats_out).unsqueeze(-1)
        vectors_out = self.vectors_activation(gating) * Vu
        return feats_out, vectors_out


class _VDropout(nn.Layer):
    """Vector channel dropout which drops whole vector channels together."""

    def __init__(self, drop_rate):
        super().__init__()
        self.drop_rate = drop_rate

    def forward(self, x):
        if not self.training:
            return x
        prob = paddle.full(x.shape[:-1], 1 - self.drop_rate)
        mask = paddle.bernoulli(prob).unsqueeze(-1)
        return mask * x / (1 - self.drop_rate)


class GVPDropout(nn.Layer):
    """Separate dropout for scalars and vectors."""

    def __init__(self, rate):
        super().__init__()
        self.vector_dropout = _VDropout(rate)
        self.feat_dropout = nn.Dropout(rate)

    def forward(self, feats, vectors):
        return self.feat_dropout(feats), self.vector_dropout(vectors)


class GVPLayerNorm(nn.Layer):
    """Layer norm for scalars and RMS norm for vectors."""

    def __init__(self, feats_h_size, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.feat_norm = nn.LayerNorm(feats_h_size)

    def forward(self, feats, vectors):
        normed_feats = self.feat_norm(feats)
        vn = norm_no_nan(vectors, axis=-1, keepdims=True, sqrt=False)
        vn = paddle.sqrt(paddle.mean(vn, axis=-2, keepdim=True) + self.eps) + self.eps
        normed_vectors = vectors / vn
        return normed_feats, normed_vectors


class GVPConv(nn.Layer):
    """GVP graph convolution on a homogeneous graph."""

    def __init__(
        self,
        scalar_size: int,
        vector_size: int,
        n_cp_feats: int,
        n_message_gvps: int,
        n_update_gvps: int,
        rbf_dmax: float,
        rbf_dim: int,
        edge_feat_size: int,
        message_norm: Union[float, str],
        scalar_activation=nn.SiLU,
        vector_activation=nn.Sigmoid,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.scalar_size = scalar_size
        self.edge_feat_size = edge_feat_size
        self.rbf_dmax = rbf_dmax
        self.rbf_dim = rbf_dim
        self.dropout_rate = dropout
        self.message_norm = message_norm

        if isinstance(message_norm, str) and message_norm not in ["mean", "sum"]:
            raise ValueError(
                "message_norm must be either 'mean', 'sum', or a number, "
                f"got {message_norm}"
            )
        if not isinstance(message_norm, (str, float, int)):
            raise TypeError(
                "message_norm must be either 'mean', 'sum', or a number, "
                f"got {message_norm}"
            )

        message_gvps = []
        for i in range(n_message_gvps):
            dim_vectors_in = vector_size
            dim_feats_in = scalar_size

            if i == 0:
                dim_vectors_in += 1
                dim_feats_in += rbf_dim + edge_feat_size

            message_gvps.append(
                GVP(
                    dim_vectors_in=dim_vectors_in,
                    dim_vectors_out=vector_size,
                    n_cp_feats=n_cp_feats,
                    dim_feats_in=dim_feats_in,
                    dim_feats_out=scalar_size,
                    feats_activation=scalar_activation(),
                    vectors_activation=vector_activation(),
                )
            )
        self.edge_message = nn.LayerList(message_gvps)

        update_gvps = []
        for _ in range(n_update_gvps):
            update_gvps.append(
                GVP(
                    dim_vectors_in=vector_size,
                    dim_vectors_out=vector_size,
                    n_cp_feats=n_cp_feats,
                    dim_feats_in=scalar_size,
                    dim_feats_out=scalar_size,
                    feats_activation=scalar_activation(),
                    vectors_activation=vector_activation(),
                )
            )
        self.node_update = nn.LayerList(update_gvps)

        self.dropout = GVPDropout(self.dropout_rate)
        self.message_layer_norm = GVPLayerNorm(self.scalar_size)
        self.update_layer_norm = GVPLayerNorm(self.scalar_size)

        if self.message_norm == "mean":
            self.agg_func = scatter_mean
        else:
            self.agg_func = scatter_sum

    def forward(
        self,
        scalar_feats: paddle.Tensor,
        vec_feats: paddle.Tensor,
        src: paddle.Tensor,
        dst: paddle.Tensor,
        num_nodes: int,
        edge_feats: paddle.Tensor,
        x_diff: paddle.Tensor,
        d: paddle.Tensor,
    ):
        scalar_message, vector_message = self.message(
            scalar_feats, vec_feats, edge_feats, x_diff, d, src, dst
        )

        if isinstance(self.message_norm, str):
            z = 1
        else:
            z = self.message_norm

        scalar_msg = self.agg_func(scalar_message, dst, dim=0, dim_size=num_nodes) / z
        vec_msg = self.agg_func(vector_message, dst, dim=0, dim_size=num_nodes) / z

        scalar_msg, vec_msg = self.dropout(scalar_msg, vec_msg)

        scalar_feat = scalar_feats + scalar_msg
        vec_feat = vec_feats + vec_msg
        scalar_feat, vec_feat = self.message_layer_norm(scalar_feat, vec_feat)

        scalar_residual, vec_residual = scalar_feat, vec_feat
        for gvp in self.node_update:
            scalar_residual, vec_residual = gvp(scalar_residual, vec_residual)
        scalar_residual, vec_residual = self.dropout(scalar_residual, vec_residual)
        scalar_feat = scalar_feat + scalar_residual
        vec_feat = vec_feat + vec_residual
        scalar_feat, vec_feat = self.update_layer_norm(scalar_feat, vec_feat)

        return scalar_feat, vec_feat

    def message(self, scalar_feats, vec_feats, edge_feats, x_diff, d, src, dst):
        vec_inputs = paddle.concat([x_diff.unsqueeze(1), vec_feats[src]], axis=1)

        scalar_inputs = [scalar_feats[src], d]
        if self.edge_feat_size > 0:
            scalar_inputs.append(edge_feats)
        scalar_inputs = paddle.concat(scalar_inputs, axis=1)

        scalar_message, vector_message = scalar_inputs, vec_inputs
        for gvp in self.edge_message:
            scalar_message, vector_message = gvp(scalar_message, vector_message)
        return scalar_message, vector_message

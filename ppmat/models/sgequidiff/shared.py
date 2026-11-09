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

"""Shared submodules reused across the SGEquiDiff model family: space-group
encoder, Fourier feature encoding, and graph aggregation / normalization."""

import math

import paddle
import paddle.nn as nn

from ppmat.models.common.activation import ScaledSiLU as Swish
from ppmat.models.sgequidiff.vocabs import EmbeddingTools
from ppmat.utils.scatter import scatter as paddle_scatter


class SpaceGroupEncoder(nn.Layer):
    """Feature encoder for space group indices."""

    def __init__(
        self,
        embedding_tools: "EmbeddingTools",
        hidden_channels: int = 256,
        space_group_embedding_dim: int = 128,
    ):
        super().__init__()
        self.embedding_tools = embedding_tools
        self.net = nn.Sequential(
            nn.Linear(embedding_tools.space_group_embedding_length, hidden_channels),
            Swish(),
            nn.Linear(hidden_channels, space_group_embedding_dim),
            Swish(),
        )

    def forward(self, space_group_indices):
        return self.net(
            self.embedding_tools.get_space_group_embedding(space_group_indices)
        )


class FourierLinear(nn.Layer):
    """Fourier feature encoding for 3D points."""

    def __init__(
        self,
        input_dim: int,
        num_fourier_frequencies: int,
        scale: float,
        output_dim: int,
        num_layers: int = 1,
        use_bias: bool = True,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}.")
        self.num_fourier_frequencies = num_fourier_frequencies
        self.scale = scale
        self.output_dim = output_dim
        self.num_layers = num_layers

        if self.scale > 0:
            self.fourier_freqs = paddle.create_parameter(
                shape=[input_dim, num_fourier_frequencies],
                dtype="float32",
                default_initializer=nn.initializer.Normal(std=scale),
            )
            self.fourier_freqs.stop_gradient = True
            in_dim = input_dim + 2 * num_fourier_frequencies
            self.layer = nn.Linear(in_dim, output_dim, bias_attr=use_bias)
        else:
            in_dim = input_dim
            self.layer = nn.Linear(in_dim, output_dim, bias_attr=use_bias)
        if num_layers > 1:
            self.layers = nn.LayerList(
                [nn.Linear(output_dim, output_dim) for _ in range(num_layers - 1)]
            )

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        if self.scale > 0:
            with paddle.no_grad():
                v = 2 * math.pi * x @ self.fourier_freqs
                v = paddle.concat([x, v.sin(), v.cos()], axis=-1)
        else:
            v = x
        v = self.layer(v)
        if self.num_layers > 1:
            for layer in self.layers:
                v = nn.functional.silu(layer(v)) + v
        return v


class VariancePreservingAggregation(nn.Layer):
    """Variance preserving aggregation: vpa(X) = sum(X) / sqrt(|X|)."""

    def forward(
        self,
        src: paddle.Tensor,
        index: paddle.Tensor,
        dim_size: int,
    ) -> paddle.Tensor:
        sum_agg = paddle_scatter(src, index, dim=0, dim_size=dim_size, reduce="sum")
        counts = paddle_scatter(
            paddle.ones([src.shape[0]], dtype=src.dtype),
            index,
            dim=0,
            dim_size=dim_size,
            reduce="sum",
        )
        return paddle.nan_to_num(sum_agg / paddle.sqrt(counts).unsqueeze(-1))


class GraphNorm(nn.Layer):
    """Graph normalization layer."""

    def __init__(self, in_channels: int, eps: float = 1e-5):
        super().__init__()
        self.in_channels = in_channels
        self.eps = eps
        self.weight = self.create_parameter(
            [in_channels],
            default_initializer=nn.initializer.Constant(1.0),
        )
        self.bias = self.create_parameter(
            [in_channels],
            default_initializer=nn.initializer.Constant(0.0),
        )
        self.mean_scale = self.create_parameter(
            [in_channels],
            default_initializer=nn.initializer.Constant(1.0),
        )

    def forward(
        self,
        x: paddle.Tensor,
        map_node_to_graph: paddle.Tensor,
        num_graphs: int,
    ) -> paddle.Tensor:
        mean = paddle_scatter(
            x, map_node_to_graph, dim=0, dim_size=num_graphs, reduce="mean"
        )

        out = x - mean[map_node_to_graph] * self.mean_scale

        var = paddle_scatter(
            out**2, map_node_to_graph, dim=0, dim_size=num_graphs, reduce="mean"
        )

        std = (var + self.eps).sqrt()[map_node_to_graph].clip(min=1.0)
        return self.weight * out / std + self.bias

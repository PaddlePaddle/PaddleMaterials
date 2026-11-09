# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import annotations

import paddle

from ppmat.models.common.activation import GateActivation

from .nn.layer_norm import EquivariantRMSNorm
from .nn.so2_layers import SO2_Convolution
from .nn.so3_layers import SO3_Linear


class Edgewise(paddle.nn.Layer):
    """Rotate node features to edges, apply SO(2) convolution, and aggregate."""

    def __init__(
        self,
        sphere_channels: int,
        hidden_channels: int,
        lmax: int,
        mmax: int,
        edge_channels_list: list[int],
        mapping,
    ) -> None:
        super().__init__()
        self.activation = GateActivation(
            lmax=lmax,
            mmax=mmax,
            num_channels=hidden_channels,
            m_prime=True,
        )
        extra_m0_channels = lmax * hidden_channels

        self.so2_conv_1 = SO2_Convolution(
            2 * sphere_channels,
            hidden_channels,
            lmax,
            mmax,
            mapping,
            internal_weights=False,
            edge_channels_list=edge_channels_list,
            extra_m0_output_channels=extra_m0_channels,
        )
        self.so2_conv_2 = SO2_Convolution(
            hidden_channels,
            sphere_channels,
            lmax,
            mmax,
            mapping,
            internal_weights=True,
        )

    def forward(
        self,
        x: paddle.Tensor,
        edge_features: paddle.Tensor,
        edge_index: paddle.Tensor,
        wigner: paddle.Tensor,
        wigner_inv: paddle.Tensor,
    ) -> paddle.Tensor:
        source = x[edge_index[0]]
        target = x[edge_index[1]]
        message = paddle.bmm(wigner, paddle.concat([source, target], axis=2))
        message, gates = self.so2_conv_1(message, edge_features)
        message = self.activation(gates, message)
        message = self.so2_conv_2(message)
        message = paddle.bmm(wigner_inv, message)

        output = paddle.zeros(
            [x.shape[0], message.shape[1], message.shape[2]],
            dtype=message.dtype,
        )
        return output.index_add(
            axis=0,
            index=edge_index[1],
            value=message,
        )


class SpectralAtomwise(paddle.nn.Layer):
    def __init__(
        self,
        sphere_channels: int,
        hidden_channels: int,
        lmax: int,
    ) -> None:
        super().__init__()
        self.scalar_mlp = paddle.nn.Sequential(
            paddle.nn.Linear(
                sphere_channels,
                lmax * hidden_channels,
                bias_attr=True,
            ),
            paddle.nn.SiLU(),
        )
        self.linear_1 = SO3_Linear(sphere_channels, hidden_channels, lmax=lmax)
        self.activation = GateActivation(
            lmax=lmax,
            mmax=lmax,
            num_channels=hidden_channels,
        )
        self.linear_2 = SO3_Linear(hidden_channels, sphere_channels, lmax=lmax)

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        gates = self.scalar_mlp(x[:, :1, :])
        return self.linear_2(self.activation(gates, self.linear_1(x)))


class ESCNMDInteractionBlock(paddle.nn.Layer):
    def __init__(
        self,
        sphere_channels: int,
        hidden_channels: int,
        lmax: int,
        mmax: int,
        mapping,
        edge_channels_list: list[int],
    ) -> None:
        super().__init__()
        self.norm_1 = EquivariantRMSNorm(lmax, sphere_channels)
        self.edgewise = Edgewise(
            sphere_channels,
            hidden_channels,
            lmax,
            mmax,
            edge_channels_list,
            mapping,
        )
        self.norm_2 = EquivariantRMSNorm(lmax, sphere_channels)
        self.atomwise = SpectralAtomwise(sphere_channels, hidden_channels, lmax)

    def forward(
        self,
        x: paddle.Tensor,
        edge_features: paddle.Tensor,
        edge_index: paddle.Tensor,
        wigner: paddle.Tensor,
        wigner_inv: paddle.Tensor,
    ) -> paddle.Tensor:
        residual = x
        x = self.norm_1(x)
        x = residual + self.edgewise(x, edge_features, edge_index, wigner, wigner_inv)
        return x + self.atomwise(self.norm_2(x))

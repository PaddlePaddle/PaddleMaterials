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


class EquivariantRMSNorm(paddle.nn.Layer):
    """RMS normalization balanced across spherical-harmonic degrees."""

    def __init__(
        self,
        lmax: int,
        num_channels: int,
        eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.num_channels = num_channels
        self.eps = eps
        self.weight = self.create_parameter(
            [lmax + 1, num_channels],
            default_initializer=paddle.nn.initializer.Constant(1.0),
        )
        self.bias = self.create_parameter(
            [num_channels],
            default_initializer=paddle.nn.initializer.Constant(0.0),
        )

        expand_index = paddle.zeros([(lmax + 1) ** 2], dtype="int64")
        degree_weight = paddle.zeros([(lmax + 1) ** 2, 1], dtype="float32")
        for degree in range(lmax + 1):
            start = degree**2
            length = 2 * degree + 1
            expand_index[start : start + length] = degree
            degree_weight[start : start + length] = 1.0 / length / (lmax + 1)
        self.register_buffer("expand_index", expand_index, persistable=False)
        self.register_buffer("degree_weight", degree_weight, persistable=False)

    def forward(self, features: paddle.Tensor) -> paddle.Tensor:
        scalar = features[:, :1, :]
        scalar = scalar - scalar.mean(axis=2, keepdim=True)
        features = paddle.concat([scalar, features[:, 1:, :]], axis=1)

        squared_norm = paddle.einsum(
            "nic,ia->nac",
            features.square(),
            self.degree_weight.astype(features.dtype),
        )
        inverse_rms = (squared_norm.mean(axis=2, keepdim=True) + self.eps).rsqrt()
        weight = paddle.index_select(
            self.weight.unsqueeze(0),
            self.expand_index,
            axis=1,
        )
        output = features * inverse_rms * weight
        output_scalar = output[:, :1, :] + self.bias.reshape([1, 1, self.num_channels])
        return paddle.concat([output_scalar, output[:, 1:, :]], axis=1)

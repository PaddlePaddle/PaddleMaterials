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

import math

import paddle

from .radial import RadialMLP


class SO2MConvolution(paddle.nn.Layer):
    """SO(2) convolution for the real and imaginary components of one order."""

    def __init__(
        self,
        order: int,
        sphere_channels: int,
        output_channels: int,
        lmax: int,
    ) -> None:
        super().__init__()
        num_coefficients = lmax - order + 1
        input_channels = num_coefficients * sphere_channels
        self.output_channels = output_channels
        self.output_half = output_channels * num_coefficients
        self.linear = paddle.nn.Linear(
            input_channels,
            2 * self.output_half,
            bias_attr=False,
        )
        with paddle.no_grad():
            self.linear.weight.set_value(self.linear.weight / math.sqrt(2))

    def forward(self, features: paddle.Tensor) -> tuple[paddle.Tensor, paddle.Tensor]:
        features = self.linear(features).reshape(
            [features.shape[0], 4, self.output_half]
        )
        real_0, imag_0, real_1, imag_1 = paddle.unbind(features, axis=1)
        real = real_0 - imag_1
        imag = real_1 + imag_0
        shape = [features.shape[0], -1, self.output_channels]
        return real.reshape(shape), imag.reshape(shape)


class SO2_Convolution(paddle.nn.Layer):
    """Apply SO(2) convolutions to all spherical-harmonic orders."""

    def __init__(
        self,
        sphere_channels: int,
        m_output_channels: int,
        lmax: int,
        mmax: int,
        mappingReduced,
        internal_weights: bool = True,
        edge_channels_list: list[int] | None = None,
        extra_m0_output_channels: int | None = None,
    ) -> None:
        super().__init__()
        self.output_channels = m_output_channels
        self.mapping = mappingReduced
        self.extra_m0_channels = extra_m0_output_channels

        m0_input = (lmax + 1) * sphere_channels
        m0_output = (lmax + 1) * m_output_channels
        if extra_m0_output_channels is not None:
            m0_output += extra_m0_output_channels
        self.m0_linear = paddle.nn.Linear(m0_input, m0_output, bias_attr=True)

        self.m_convolutions = paddle.nn.LayerList(
            [
                SO2MConvolution(
                    order,
                    sphere_channels,
                    m_output_channels,
                    lmax,
                )
                for order in range(1, mmax + 1)
            ]
        )
        self.feature_splits = [self.mapping.m_size[0]] + [
            2 * size for size in self.mapping.m_size[1:]
        ]
        self.radial_splits = [self.m0_linear.weight.shape[0]] + [
            layer.linear.weight.shape[0] for layer in self.m_convolutions
        ]

        self.radial = None
        if not internal_weights:
            if edge_channels_list is None:
                raise ValueError("edge_channels_list is required for external weights.")
            radial_channels = [*edge_channels_list, sum(self.radial_splits)]
            self.radial = RadialMLP(radial_channels)

    def forward(
        self,
        features: paddle.Tensor,
        edge_features: paddle.Tensor | None = None,
    ):
        if self.radial is not None:
            edge_features = self.radial(edge_features)

        features_by_order = paddle.split(features, self.feature_splits, axis=1)
        radial_by_order = (
            paddle.split(edge_features, self.radial_splits, axis=1)
            if edge_features is not None
            else None
        )

        num_edges = features.shape[0]
        scalar = features_by_order[0].reshape([num_edges, -1])
        if radial_by_order is not None:
            scalar = scalar * radial_by_order[0]
        scalar = self.m0_linear(scalar)

        extra = None
        if self.extra_m0_channels is not None:
            extra, scalar = paddle.split(
                scalar,
                [
                    self.extra_m0_channels,
                    scalar.shape[-1] - self.extra_m0_channels,
                ],
                axis=-1,
            )
        output = [scalar.reshape([num_edges, -1, self.output_channels])]

        for order, convolution in enumerate(self.m_convolutions, start=1):
            order_features = features_by_order[order].reshape([num_edges, 2, -1])
            if radial_by_order is not None:
                order_features = order_features * radial_by_order[order].unsqueeze(1)
            output.extend(convolution(order_features))

        output = paddle.concat(output, axis=1)
        return (output, extra) if extra is not None else output

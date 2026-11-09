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

"""Coefficient ordering used by the eSCN SO(2) convolutions."""

from __future__ import annotations

import numpy as np
import paddle


class CoefficientMapping(paddle.nn.Layer):
    """Map spherical coefficients from degree-major to order-major layout."""

    def __init__(self, lmax: int, mmax: int) -> None:
        super().__init__()
        self.lmax = lmax
        self.mmax = mmax

        degrees = []
        orders = []
        for degree in range(lmax + 1):
            degree_orders = np.arange(
                -min(mmax, degree),
                min(mmax, degree) + 1,
                dtype=np.int64,
            )
            degrees.extend([degree] * len(degree_orders))
            orders.extend(degree_orders.tolist())
        degrees = np.asarray(degrees, dtype=np.int64)
        orders = np.asarray(orders, dtype=np.int64)

        to_m = np.zeros((len(degrees), len(degrees)), dtype=np.float32)
        self.m_size = []
        output_offset = 0
        for order in range(mmax + 1):
            real_indices = np.flatnonzero(orders == order)
            imaginary_indices = (
                np.flatnonzero(orders == -order)
                if order > 0
                else np.empty(0, dtype=np.int64)
            )
            self.m_size.append(len(real_indices))
            for index in real_indices:
                to_m[output_offset, index] = 1.0
                output_offset += 1
            for index in imaginary_indices:
                to_m[output_offset, index] = 1.0
                output_offset += 1

        self.register_buffer(
            "to_m",
            paddle.to_tensor(to_m),
            persistable=False,
        )

    def coefficient_idx(self, lmax: int, mmax: int) -> paddle.Tensor:
        indices = []
        full_index = 0
        for degree in range(lmax + 1):
            for order in range(-degree, degree + 1):
                if abs(order) <= mmax:
                    indices.append(full_index)
                full_index += 1
        return paddle.to_tensor(indices, dtype="int64")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}" f"(lmax={self.lmax}, mmax={self.mmax})"

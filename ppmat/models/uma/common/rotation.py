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

"""Wigner-D rotation utilities used by the eSCN interaction blocks."""

from __future__ import annotations

import paddle


def init_edge_rot_euler_angles(
    edge_vectors: paddle.Tensor,
) -> tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]:
    normalized = paddle.nn.functional.normalize(edge_vectors, axis=-1).clip(-1.0, 1.0)
    x, y, z = paddle.unbind(normalized, axis=1)
    beta = paddle.acos(y.clip(-1 + 1e-7, 1 - 1e-7))
    alpha = paddle.atan2(x, z)
    gamma = paddle.rand_like(alpha) * 2 * paddle.pi
    return -gamma, -beta, -alpha


def _z_rotation(angle: paddle.Tensor, degree: int) -> paddle.Tensor:
    size = 2 * degree + 1
    matrix = paddle.zeros([*angle.shape, size, size], dtype=angle.dtype)
    frequencies = range(degree, -degree - 1, -1)
    for index, frequency in enumerate(frequencies):
        matrix[..., index, size - index - 1] = paddle.sin(frequency * angle)
        matrix[..., index, index] = paddle.cos(frequency * angle)
    return matrix


def wigner_d(
    degree: int,
    alpha: paddle.Tensor,
    beta: paddle.Tensor,
    gamma: paddle.Tensor,
    coefficients: paddle.Tensor,
) -> paddle.Tensor:
    alpha, beta, gamma = paddle.broadcast_tensors([alpha, beta, gamma])
    return (
        _z_rotation(alpha, degree)
        @ coefficients
        @ _z_rotation(beta, degree)
        @ coefficients
        @ _z_rotation(gamma, degree)
    )


def eulers_to_wigner(
    eulers: tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor],
    start_degree: int,
    end_degree: int,
    coefficients: list[paddle.Tensor],
) -> paddle.Tensor:
    alpha, beta, gamma = eulers
    size = (end_degree + 1) ** 2 - start_degree**2
    wigner = paddle.zeros([alpha.shape[0], size, size], dtype=alpha.dtype)
    start = 0
    for degree in range(start_degree, end_degree + 1):
        block = wigner_d(
            degree,
            alpha,
            beta,
            gamma,
            coefficients[degree],
        )
        end = start + block.shape[1]
        wigner[:, start:end, start:end] = block
        start = end
    return wigner

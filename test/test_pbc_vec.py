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

import numpy as np
import paddle

from ppmat.utils.crystal import pbc_vec


def test_pbc_vec_wraps_batched_vectors():
    cell = paddle.to_tensor(
        [
            [[2.0, 0, 0], [0, 3.0, 0], [0, 0, 4.0]],
            [[3.0, 0, 0], [0.5, 2.0, 0], [0, 0.2, 4.0]],
        ],
        dtype="float64",
    )
    fractional = paddle.to_tensor(
        [[[0.7, -1.2, 0.1], [-0.4, 0.3, 1.1]], [[1.1, 0.8, -0.9], [0.2, -0.1, 0.3]]],
        dtype="float64",
    )
    wrapped, coords = pbc_vec(fractional @ cell, cell)
    expected = fractional.numpy() - np.round(fractional.numpy())
    np.testing.assert_allclose(coords.numpy(), expected, atol=1e-12)
    np.testing.assert_allclose(wrapped.numpy(), expected @ cell.numpy(), atol=1e-12)


def test_pbc_vec_preserves_coordinate_and_cell_gradients():
    cell = paddle.to_tensor(np.eye(3) * 2, dtype="float64", stop_gradient=False)
    vectors = paddle.to_tensor([[2.4, -1.6, 0.2]], dtype="float64", stop_gradient=False)
    wrapped, fractional = pbc_vec(vectors, cell)
    wrapped.sum().backward()
    np.testing.assert_allclose(vectors.grad.numpy(), np.ones((1, 3)), atol=1e-12)
    np.testing.assert_allclose(
        cell.grad.numpy(), [[-1, -1, -1], [1, 1, 1], [0, 0, 0]], atol=1e-12
    )
    assert not fractional.stop_gradient

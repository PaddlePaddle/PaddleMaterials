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

import numpy as np
import paddle
import pytest

from ppmat.utils.scatter import scatter
from ppmat.utils.scatter import scatter_argmin
from ppmat.utils.scatter import scatter_mean
from ppmat.utils.scatter import scatter_min
from ppmat.utils.scatter import scatter_sum
from ppmat.utils.scatter import scatter_sum_first_order


def test_scatter_sum_first_order_supports_first_order_gradients():
    values = paddle.to_tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], stop_gradient=False)
    groups = paddle.to_tensor([0, 1, 0], dtype="int64")

    result = scatter_sum_first_order(values, groups, dim_size=3)
    result.sum().backward()

    np.testing.assert_array_equal(result.numpy(), [[6, 8], [3, 4], [0, 0]])
    np.testing.assert_array_equal(values.grad.numpy(), np.ones([3, 2]))


def test_scatter_argmin_handles_unsorted_and_empty_groups():
    values = paddle.to_tensor([3.0, -2.0, 4.0, -5.0, 1.0])
    groups = paddle.to_tensor([2, 0, 2, 0, 2], dtype="int64")

    result = scatter_argmin(values, groups, dim_size=4)

    np.testing.assert_array_equal(result.numpy(), [3, -1, 4, -1])


def test_scatter_argmin_selects_first_value_on_ties():
    values = paddle.to_tensor([2.0, 1.0, 1.0, 3.0])
    groups = paddle.to_tensor([0, 0, 0, 1], dtype="int64")

    result = scatter_argmin(values, groups)

    np.testing.assert_array_equal(result.numpy(), [1, 3])


def test_scatter_argmin_handles_empty_input():
    values = paddle.empty([0], dtype="float32")
    groups = paddle.empty([0], dtype="int64")

    result = scatter_argmin(values, groups, dim_size=3)

    np.testing.assert_array_equal(result.numpy(), [-1, -1, -1])


def test_scatter_sum_supports_second_order_gradients():
    values = paddle.to_tensor([[1.0], [2.0], [3.0]], stop_gradient=False)
    groups = paddle.to_tensor([0, 1, 0], dtype="int64")

    result = scatter_sum(values.square(), groups, dim=0)
    first_grad = paddle.grad(result.sum(), values, create_graph=True)[0]
    second_grad = paddle.grad(first_grad.sum(), values)[0]

    np.testing.assert_allclose(first_grad.numpy(), [[2.0], [4.0], [6.0]])
    np.testing.assert_allclose(second_grad.numpy(), np.full([3, 1], 2.0))


def test_scatter_reductions_support_nonzero_and_negative_dim():
    values = paddle.to_tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    groups = paddle.to_tensor([0, 1, 0], dtype="int64")

    summed = scatter_sum(values, groups, dim=-1)
    mean = scatter_mean(values, groups, dim=1)
    minimum = scatter_min(values, groups, dim=1)

    np.testing.assert_array_equal(summed.numpy(), [[4, 2], [10, 5]])
    np.testing.assert_array_equal(mean.numpy(), [[2, 2], [5, 5]])
    np.testing.assert_array_equal(minimum.numpy(), [[1, 2], [4, 5]])


def test_scatter_dispatches_sum_alias():
    values = paddle.to_tensor([1.0, 2.0, 3.0])
    groups = paddle.to_tensor([0, 1, 0], dtype="int64")

    expected = scatter_sum(values, groups)
    actual = scatter(values, groups, reduce="add")

    np.testing.assert_array_equal(actual.numpy(), expected.numpy())


@pytest.mark.parametrize("dtype", ["float32", "float64", "int32", "int64"])
@pytest.mark.parametrize("shape", [(0,), (0, 2), (0, 2, 3)])
@pytest.mark.parametrize("dim_size", [None, 0, 3])
def test_scatter_sum_first_order_handles_empty_input(dtype, shape, dim_size):
    values = paddle.empty(shape, dtype=dtype)
    groups = paddle.empty([0], dtype="int64")

    result = scatter_sum_first_order(values, groups, dim_size=dim_size)

    expected_shape = (dim_size or 0, *shape[1:])
    assert tuple(result.shape) == expected_shape
    assert result.dtype == values.dtype
    assert result.place == values.place
    np.testing.assert_array_equal(result.numpy(), np.zeros(expected_shape, dtype=dtype))


def test_scatter_sum_first_order_empty_input_preserves_gradient():
    values = paddle.empty([0, 2], dtype="float32")
    values.stop_gradient = False
    groups = paddle.empty([0], dtype="int64")

    scatter_sum_first_order(values, groups, dim_size=3).sum().backward()

    assert values.grad is not None
    assert list(values.grad.shape) == [0, 2]

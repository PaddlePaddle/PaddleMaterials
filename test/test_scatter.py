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

from ppmat.utils.execution import compiled_execution
from ppmat.utils.scatter import _resolve_dim_size
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


def test_scatter_mean_supports_second_order_gradients():
    values = paddle.to_tensor([[1.0], [2.0], [3.0]], stop_gradient=False)
    groups = paddle.to_tensor([0, 1, 0], dtype="int64")

    result = scatter_mean(values.square(), groups, dim=0, dim_size=3)
    first_grad = paddle.grad(result.sum(), values, create_graph=True)[0]
    second_grad = paddle.grad(first_grad.sum(), values)[0]

    np.testing.assert_allclose(first_grad.numpy(), [[1.0], [4.0], [3.0]])
    np.testing.assert_allclose(second_grad.numpy(), [[1.0], [2.0], [1.0]])


def test_scatter_sum_dim_zero_supports_higher_rank_and_empty_input():
    values = paddle.arange(24, dtype="float32").reshape([4, 2, 3])
    groups = paddle.to_tensor([0, 2, 0, 2], dtype="int64")

    result = scatter_sum(values, groups, dim=0, dim_size=4)
    expected = np.zeros([4, 2, 3], dtype="float32")
    np.add.at(expected, groups.numpy(), values.numpy())
    np.testing.assert_array_equal(result.numpy(), expected)

    empty = scatter_sum(
        paddle.empty([0, 2, 3], dtype="float32"),
        paddle.empty([0], dtype="int64"),
        dim=0,
        dim_size=4,
    )
    np.testing.assert_array_equal(empty.numpy(), np.zeros([4, 2, 3]))


def test_scatter_sum_dim_zero_preserves_complex_fallback():
    values = paddle.to_tensor(
        [[1.0 + 2.0j], [3.0 - 1.0j], [2.0 + 4.0j]], dtype="complex64"
    )
    groups = paddle.to_tensor([0, 1, 0], dtype="int64")

    result = scatter_sum(values, groups, dim=0, dim_size=3)

    np.testing.assert_allclose(result.numpy(), [[3.0 + 6.0j], [3.0 - 1.0j], [0.0]])


def test_scatter_sum_supports_force_loss_parameter_gradients():
    coords = paddle.to_tensor(
        [[0.1, 0.2], [0.4, -0.2], [-0.3, 0.7]], stop_gradient=False
    )
    weights = paddle.to_tensor([[0.2, 0.4], [0.6, 0.8]], stop_gradient=False)
    groups = paddle.to_tensor([0, 1, 0], dtype="int64")

    messages = paddle.sin(coords @ weights)
    pooled = scatter_sum(messages, groups, dim=0, dim_size=3)
    energy = pooled.square().sum()
    force = -paddle.grad(energy, coords, create_graph=True)[0]
    force.square().mean().backward()

    assert weights.grad is not None
    assert paddle.isfinite(weights.grad).all()
    assert float(paddle.abs(weights.grad).sum()) > 0


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


def test_scatter_sum_first_order_handles_empty_input():
    values = paddle.empty([0, 2], dtype="float32")
    groups = paddle.empty([0], dtype="int64")

    result = scatter_sum_first_order(values, groups, dim_size=3)

    np.testing.assert_array_equal(result.numpy(), np.zeros([3, 2]))


def test_resolve_dim_size_keeps_host_ints_and_reads_tensors():
    index = paddle.to_tensor([0, 1], dtype="int64")
    assert _resolve_dim_size(index, 5) == 5
    assert type(_resolve_dim_size(index, np.int64(5))) is int
    assert _resolve_dim_size(index, paddle.to_tensor(5)) == 5
    assert _resolve_dim_size(index, None) == 2


def test_compiled_execution_scatters_empty_rows_without_the_shortcut():
    groups = paddle.empty([0], dtype="int64")
    eager_values = paddle.empty([0, 2, 3], dtype="float32")
    eager_values.stop_gradient = False
    eager = scatter_sum(eager_values, groups, dim=0, dim_size=4)
    (eager_grad,) = paddle.grad(eager.sum(), eager_values)

    values = paddle.empty([0, 2, 3], dtype="float32")
    values.stop_gradient = False
    with compiled_execution():
        result = scatter_sum(values, groups, dim=0, dim_size=4)
    (grad,) = paddle.grad(result.sum(), values)

    np.testing.assert_array_equal(result.numpy(), eager.numpy())
    assert result.shape == [4, 2, 3]
    assert grad.shape == eager_grad.shape == [0, 2, 3]


def test_scatter_sum_dim_zero_one_dimensional_index_matches_broadcast_index():
    rng = np.random.default_rng(3)
    values_np = rng.normal(size=(6, 2, 3)).astype("float32")
    groups_np = np.array([3, 0, 3, 1, 0, 3], dtype="int64")
    results = []
    for groups in (
        paddle.to_tensor(groups_np),
        paddle.to_tensor(np.broadcast_to(groups_np[:, None, None], (6, 2, 3)).copy()),
    ):
        values = paddle.to_tensor(values_np, stop_gradient=False)
        out = scatter_sum(values, groups, dim=0, dim_size=4)
        (grad,) = paddle.grad((out * out).sum(), values, create_graph=True)
        (grad2,) = paddle.grad(grad.sum(), values)
        results.append((out.numpy(), grad.numpy(), grad2.numpy()))
    for a, b in zip(*results):
        np.testing.assert_array_equal(a, b)

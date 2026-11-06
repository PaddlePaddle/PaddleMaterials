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

from ppmat.models.common.activation import GateActivation
from ppmat.models.uma.nn.radial import RadialMLP


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("m_prime", [False, True])
def test_gate_activation_extreme_inputs_match_analytic_derivatives(dtype, m_prime):
    values = np.asarray([-1000, -120, -100, -87, -80, -20, -1, 0, 1, 20, 1000])
    features = np.ones((len(values), 9, 2), dtype=dtype)
    features[:, 0, :] = values[:, None]
    features = paddle.to_tensor(features, stop_gradient=False)
    gates = paddle.to_tensor(
        np.linspace(-1, 1, len(values) * 4).reshape([-1, 4]).astype(dtype),
        stop_gradient=False,
    )
    model = GateActivation(lmax=2, mmax=2, num_channels=2, m_prime=m_prime)
    output = model(gates, features)
    first, gate_grad = paddle.grad(output.sum(), [features, gates], create_graph=True)
    second = paddle.grad(first.sum(), features)[0]
    sigmoid = np.exp(-np.logaddexp(0, -values.astype("float64")))
    gate_sigmoid = np.exp(-np.logaddexp(0, -gates.numpy().reshape([-1, 2, 2])))
    expected = np.empty(features.shape, dtype="float64")
    expected[:, 0, :] = (values * sigmoid)[:, None]
    expected[:, 1:, :] = gate_sigmoid[:, model.expand_index.numpy(), :]
    expected_first = expected.copy()
    expected_first[:, 0, :] = (sigmoid * (1 + values * (1 - sigmoid)))[:, None]
    expected_second = np.zeros_like(expected)
    expected_second[:, 0, :] = (
        sigmoid * (1 - sigmoid) * (2 + values * (1 - 2 * sigmoid))
    )[:, None]
    expected_gate_grad = (
        gate_sigmoid * (1 - gate_sigmoid) * np.asarray([3, 5]).reshape([1, 2, 1])
    ).reshape([-1, 4])
    for actual, reference in (
        (output, expected),
        (first, expected_first),
        (second, expected_second),
        (gate_grad, expected_gate_grad),
    ):
        assert bool(paddle.isfinite(actual).all())
        np.testing.assert_allclose(actual.numpy(), reference, rtol=1e-5, atol=1e-7)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_radial_mlp_nonuniform_scale_matches_analytic_gradient(dtype):
    model = RadialMLP([3, 3, 1]).to(dtype=dtype)
    model.layers[0].weight.set_value(paddle.eye(3, dtype=dtype))
    model.layers[0].bias.set_value(paddle.zeros([3], dtype=dtype))
    scale = np.asarray([2.0, -3.0, 4.0])
    bias = np.asarray([0.3, -0.2, 0.1])
    model.layers[1].weight.set_value(paddle.to_tensor(scale, dtype=dtype))
    model.layers[1].bias.set_value(paddle.to_tensor(bias, dtype=dtype))
    model.layers[3].weight.set_value(paddle.ones([3, 1], dtype=dtype))
    model.layers[3].bias.set_value(paddle.zeros([1], dtype=dtype))
    values = np.asarray([[0.1, 0.5, 0.7], [-2.0, 1.0, 3.0]])
    inputs = paddle.to_tensor(values, dtype=dtype, stop_gradient=False)
    output = model(inputs)
    first = paddle.grad(output.sum(), inputs, create_graph=True)[0]
    centered = values - values.mean(axis=-1, keepdims=True)
    inverse_std = 1 / np.sqrt(
        np.mean(centered**2, axis=-1, keepdims=True) + model.layers[1]._epsilon
    )
    normalized = centered * inverse_std
    affine = normalized * scale + bias
    sigmoid = np.exp(-np.logaddexp(0, -affine))
    weighted = sigmoid * (1 + affine * (1 - sigmoid)) * scale
    expected = inverse_std * (
        weighted
        - weighted.mean(axis=-1, keepdims=True)
        - normalized * (weighted * normalized).mean(axis=-1, keepdims=True)
    )
    np.testing.assert_allclose(
        output.numpy(),
        (affine * sigmoid).sum(axis=-1, keepdims=True),
        rtol=1e-5,
        atol=1e-6,
    )
    np.testing.assert_allclose(first.numpy(), expected, rtol=1e-4, atol=1e-6)
    second = paddle.grad(first.square().sum(), inputs)[0]
    assert bool(paddle.isfinite(second).all())
    step = 1e-3 if dtype == "float32" else 1e-5
    finite_difference = np.empty_like(values)
    with paddle.no_grad():
        for index in np.ndindex(values.shape):
            plus, minus = values.copy(), values.copy()
            plus[index] += step
            minus[index] -= step
            forward_plus = float(model(paddle.to_tensor(plus, dtype=dtype)).sum())
            forward_minus = float(model(paddle.to_tensor(minus, dtype=dtype)).sum())
            finite_difference[index] = (forward_plus - forward_minus) / (2 * step)
    np.testing.assert_allclose(
        first.numpy(),
        finite_difference,
        rtol=2e-3 if dtype == "float32" else 1e-5,
        atol=2e-3 if dtype == "float32" else 1e-6,
    )

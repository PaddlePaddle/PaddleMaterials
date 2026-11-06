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

from ppmat.models.chgnet.chgnet import MLP
from ppmat.models.chgnet.chgnet import GatedMLP
from ppmat.models.chgnet.chgnet import _LayerNorm


@pytest.mark.parametrize("hidden_dim", [1, [1], [1, 1]])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_mlp_silu_extreme_inputs_have_finite_first_and_second_derivatives(
    hidden_dim, dtype
):
    model = MLP(input_dim=1, output_dim=1, hidden_dim=hidden_dim).to(dtype=dtype)
    for layer in model.sublayers():
        if isinstance(layer, paddle.nn.Linear):
            layer.weight.set_value(paddle.ones_like(layer.weight))
            layer.bias.set_value(paddle.zeros_like(layer.bias))
    values = np.asarray([-1000, -600, -100, -87, -80, -20, -1, 0, 1, 20, 100, 600])
    inputs = paddle.to_tensor(values[:, None], dtype=dtype, stop_gradient=False)
    output = model(inputs)
    first = paddle.grad(output.sum(), inputs, create_graph=True)[0]
    second = paddle.grad(first.sum(), inputs)[0]
    for tensor in (output, first, second):
        assert bool(paddle.isfinite(tensor).all())

    reference = values.astype("float64")
    jacobian, hessian = np.ones_like(reference), np.zeros_like(reference)
    for _ in range(1 if isinstance(hidden_dim, int) else len(hidden_dim)):
        sigmoid = np.exp(-np.logaddexp(0, -reference))
        derivative = sigmoid * (1 + reference * (1 - sigmoid))
        curvature = sigmoid * (1 - sigmoid) * (2 + reference * (1 - 2 * sigmoid))
        hessian = curvature * jacobian**2 + derivative * hessian
        jacobian = derivative * jacobian
        reference = reference * sigmoid
    for actual, expected in ((output, reference), (first, jacobian), (second, hessian)):
        np.testing.assert_allclose(
            actual.numpy().reshape(-1), expected, rtol=1e-5, atol=1e-7
        )


def test_gated_mlp_silu_extreme_inputs_preserve_finite_parameter_gradients():
    model = GatedMLP(input_dim=2, output_dim=2, hidden_dim=2)
    for norm in (model.bn1, model.bn2):
        norm.bias.set_value(paddle.full_like(norm.bias, -1000))
    inputs = paddle.to_tensor([[1000.0, -1000.0]], stop_gradient=False)
    output = model(inputs)
    first = paddle.grad(output.sum(), inputs, create_graph=True, retain_graph=True)[0]
    second = paddle.grad(first.square().sum(), inputs, retain_graph=True)[0]
    output.sum().backward()
    for tensor in (output, first, second):
        assert bool(paddle.isfinite(tensor).all())
    for parameter in model.parameters():
        assert parameter.grad is not None
        assert bool(paddle.isfinite(parameter.grad).all())


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_layer_norm_nonuniform_scale_matches_analytic_gradient(dtype):
    model = _LayerNorm(3).to(dtype=dtype)
    scale = np.asarray([2.0, -3.0, 4.0])
    bias = np.asarray([0.3, -0.2, 0.1])
    model.weight.set_value(paddle.to_tensor(scale, dtype=dtype))
    model.bias.set_value(paddle.to_tensor(bias, dtype=dtype))
    values = np.asarray([[0.1, 0.5, 0.7], [-2.0, 1.0, 3.0]])
    upstream = np.asarray([[1.0, 2.0, -1.0], [0.4, -0.2, 0.1]])
    inputs = paddle.to_tensor(values, dtype=dtype, stop_gradient=False)
    output = model(inputs)
    first = paddle.grad(
        (output * paddle.to_tensor(upstream, dtype=dtype)).sum(),
        inputs,
        create_graph=True,
    )[0]
    second = paddle.grad(first.square().sum(), inputs)[0]
    centered = values - values.mean(axis=-1, keepdims=True)
    inverse_std = 1 / np.sqrt(
        np.mean(centered**2, axis=-1, keepdims=True) + model._epsilon
    )
    normalized = centered * inverse_std
    weighted = upstream * scale
    expected = inverse_std * (
        weighted
        - weighted.mean(axis=-1, keepdims=True)
        - normalized * (weighted * normalized).mean(axis=-1, keepdims=True)
    )
    np.testing.assert_allclose(
        output.numpy(), normalized * scale + bias, rtol=1e-5, atol=1e-6
    )
    np.testing.assert_allclose(first.numpy(), expected, rtol=1e-4, atol=1e-6)
    assert bool(paddle.isfinite(second).all())

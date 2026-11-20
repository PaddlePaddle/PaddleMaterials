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

# Adapted from https://github.com/divelab/AIRS (OpenMat/HIENet)

# Manually decomposed layer_norm / dropout: paddle's fused ops have no
# double-grad, this primitive composition keeps create_graph=True paths alive.
import paddle


class ManualLayerNorm(paddle.nn.Layer):
    def __init__(self, normalized_shape, epsilon=1e-5, *, dtype=None):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = [normalized_shape]
        self._normalized_shape = list(normalized_shape)
        self._epsilon = epsilon
        param_shape = 1
        for s in self._normalized_shape:
            param_shape *= s
        self.weight = self.create_parameter(
            shape=[param_shape],
            default_initializer=paddle.nn.initializer.Constant(1.0),
            dtype=dtype,
        )
        self.bias = self.create_parameter(
            shape=[param_shape],
            default_initializer=paddle.nn.initializer.Constant(0.0),
            is_bias=True,
            dtype=dtype,
        )

    def forward(self, x):
        axes = list(range(-len(self._normalized_shape), 0))
        mu = paddle.mean(x, axis=axes, keepdim=True)
        xc = x - mu
        var = paddle.mean(xc * xc, axis=axes, keepdim=True)
        return xc * paddle.rsqrt(var + self._epsilon) * self.weight + self.bias


class ManualDropout(paddle.nn.Layer):
    def __init__(self, p=0.0):
        super().__init__()
        self.p = p

    def forward(self, x):
        if not self.training or self.p == 0.0:
            return x
        # bernoulli output carries no grad, mask stays a constant in the graph
        mask = paddle.bernoulli(paddle.full(x.shape, 1.0 - self.p, dtype=x.dtype))
        return x * mask / (1.0 - self.p)

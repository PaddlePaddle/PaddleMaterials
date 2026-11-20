# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List

import paddle

from ppmat.models.common.e3nn.math import normalize2mom


class _Layer(paddle.nn.Layer):
    h_in: float
    h_out: float
    var_in: float
    var_out: float
    _profiling_str: str

    def __init__(self, h_in, h_out, act, var_in, var_out):
        super().__init__()
        self.weight = paddle.base.framework.EagerParamBase.from_tensor(
            tensor=paddle.randn(shape=[h_in, h_out])
        )
        self.act = act
        self.h_in = h_in
        self.h_out = h_out
        self.var_in = var_in
        self.var_out = var_out
        self._profiling_str = repr(self)

    def __repr__(self):
        act = self.act
        if hasattr(act, "__name__"):
            act = act.__name__
        elif isinstance(act, paddle.nn.Layer):
            act = act.__class__.__name__
        return f"Layer({self.h_in}->{self.h_out}, act={act})"

    def forward(self, x: paddle.Tensor):
        if self.act is not None:
            w = self.weight / (self.h_in * self.var_in) ** 0.5
            x = x @ w
            x = self.act(x)
            x = x * self.var_out**0.5
        else:
            w = self.weight / (self.h_in * self.var_in / self.var_out) ** 0.5
            x = x @ w
        return x


class FullyConnectedNet(paddle.nn.Sequential):
    """Fully-connected Neural Network

    Parameters
    ----------
    hs : list of int
        input, internal and output dimensions

    act : function
        activation function :math:`\\phi`, it will be automatically normalized by a scaling factor such that

        .. math::

            \\int_{-\\infty}^{\\infty} \\phi(z)^2 \\frac{e^{-z^2/2}}{\\sqrt{2\\pi}} dz = 1
    """

    hs: List[int]

    def __init__(
        self,
        hs,
        act=None,
        variance_in: int = 1,
        variance_out: int = 1,
        out_act: bool = False,
        sample_stream: str = "paddle",
    ):
        super().__init__()
        self.hs = list(hs)
        if act is not None:
            act = normalize2mom(act, sample_stream=sample_stream)
        var_in = variance_in
        for i, (h1, h2) in enumerate(zip(self.hs, self.hs[1:])):
            if i == len(self.hs) - 2:
                var_out = variance_out
                a = act if out_act else None
            else:
                var_out = 1
                a = act
            layer = _Layer(h1, h2, a, var_in, var_out)
            setattr(self, f"layer{i}", layer)
            var_in = var_out

    def __repr__(self):
        return f"{self.__class__.__name__}{self.hs}"

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

import paddle

from ppmat.models.common.e3nn.paddle_utils import *
from ppmat.models.common.e3nn.util import explicit_default_types


def moment(f, n, dtype=None, device=None):
    """
    compute n th moment
    <f(z)^n> for z normal
    """
    dtype, device = explicit_default_types(dtype, device)
    gen = paddle.framework.core.default_cpu_generator().manual_seed(0)
    z = paddle.randn(shape=[1000000], dtype="float64").to(dtype=dtype, device=device)
    return f(z).pow(y=n).mean()


# e3nn estimates ``cst`` from a 1e6-sample Monte-Carlo draw of a seeded torch
# generator, so its value is a fixed (but noisy, ~1.5e-3) estimate.  Paddle's
# generator produces different samples, which shifts the constant enough to
# break reproduction of checkpoints trained with e3nn.  An activation that
# needs that exact parity declares the e3nn value on the callable itself
# (``f.e3nn_cst``); everything else keeps the Monte-Carlo estimate, which is
# the behaviour the existing Paddle checkpoints were trained with.
def _pinned_constant(f):
    return getattr(f, "e3nn_cst", None)


class normalize2mom(paddle.nn.Layer):
    _is_id: bool
    cst: float

    def __init__(self, f, dtype=None, device=None):
        super().__init__()
        if device is None and isinstance(f, paddle.nn.Layer):
            from e3nn.util._argtools import _get_device

            device = _get_device(f)
        with paddle.no_grad():
            cst = _pinned_constant(f)
            if cst is None:
                cst = moment(f, 2, dtype="float64", device="cpu").pow(y=-0.5).item()
        if abs(cst - 1) < 0.0001:
            self._is_id = True
        else:
            self._is_id = False
        self.f = f
        self.cst = cst

    def forward(self, x):
        if self._is_id:
            return self.f(x)
        else:
            return self.f(x) * self.cst  

    @staticmethod
    def _make_tracing_inputs(n: int):
        return [{"forward": (paddle.zeros(shape=(1,)),)}]

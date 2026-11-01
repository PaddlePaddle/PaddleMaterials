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
    # Draw the samples on the CPU, where the generator seeded below is used, so
    # the moment is the same in every process whatever the current device is.
    current_device = paddle.get_device()
    paddle.set_device("cpu")
    try:
        paddle.framework.core.default_cpu_generator().manual_seed(0)
        z = paddle.randn(shape=[1000000], dtype="float64")
    finally:
        paddle.set_device(current_device)
    z = z.to(dtype=dtype, device=device)
    return f(z).pow(y=n).mean()


class normalize2mom(paddle.nn.Layer):
    _is_id: bool
    cst: float

    def __init__(self, f, dtype=None, device=None):
        super().__init__()
        if device is None and isinstance(f, paddle.nn.Layer):
            from e3nn.util._argtools import _get_device

            device = _get_device(f)
        with paddle.no_grad():
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
            return self.f(x).multiply(
                paddle.to_tensor(self.cst)
            )  

    @staticmethod
    def _make_tracing_inputs(n: int):
        return [{"forward": (paddle.zeros(shape=(1,)),)}]

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

from typing import Dict
from typing import Tuple

import numpy as np

from ppmat.models.common.e3nn.paddle_utils import *
from ppmat.models.common.e3nn.util import explicit_default_types

_TORCH_RANDN_CACHE: Dict[int, Tuple[object, paddle.Tensor]] = {}


def _torch_randn_seed0_fp64(num: int) -> paddle.Tensor:
    """Bit-exact reproduction of torch.randn(num, generator=seed0, dtype=float64)
    on CPU. Converted torch checkpoints bake normalization constants computed
    from this stream into their weights.
    """
    n_u32 = 2 * num
    n_state, m = 624, 397
    matrix_a, up, lo = 0x9908B0DF, 0x80000000, 0x7FFFFFFF

    def twist(a, b):
        y = (a & up) | (b & lo)
        return (y >> 1) ^ (matrix_a if (y & 1) else 0)

    st = [0] * n_state
    st[0] = 0
    for j in range(1, n_state):
        prev = st[j - 1]
        st[j] = (1812433253 * (prev ^ (prev >> 30)) + j) & 0xFFFFFFFF
    a = st[:]
    for j in range(n_state - m):
        a[j] = st[j + m] ^ twist(st[j], st[j + 1])
    for j in range(n_state - m, n_state - 1):
        a[j] = a[j + m - n_state] ^ twist(a[j], a[j + 1])
    a[n_state - 1] = a[m - 1] ^ twist(a[n_state - 1], a[0])
    for k in range(n_state, n_u32):
        a.append(a[k - n_state + m] ^ twist(a[k - n_state], a[k - n_state + 1]))

    arr = np.array(a[:n_u32], dtype=np.uint64)
    y = arr.copy()
    y ^= y >> np.uint64(11)
    y ^= (y << np.uint64(7)) & np.uint64(0x9D2C5680)
    y ^= (y << np.uint64(15)) & np.uint64(0xEFC60000)
    y ^= y >> np.uint64(18)
    r64 = (y[0::2] << np.uint64(32)) | y[1::2]
    u = (r64 & np.uint64((1 << 53) - 1)) * (1.0 / (1 << 53))

    z = np.empty(num, dtype=np.float64)
    blocks = u[: num // 16 * 16].reshape(-1, 16)
    u1 = 1.0 - blocks[:, :8]
    u2 = blocks[:, 8:]
    radius = np.sqrt(-2.0 * np.log(u1))
    theta = 2.0 * np.pi * u2
    out = np.empty_like(blocks)
    out[:, :8] = radius * np.cos(theta)
    out[:, 8:] = radius * np.sin(theta)
    z[: num // 16 * 16] = out.reshape(-1)
    return paddle.to_tensor(z)


def moment(f, n, dtype=None, device=None, sample_stream="paddle"):
    """
    compute n th moment
    <f(z)^n> for z normal

    sample_stream: "paddle" uses the framework CPU generator seeded with 0;
    "torch_seed0" reproduces torch.randn(num, generator=seed0) on CPU
    bit-exactly, for checkpoints converted from torch-trained models.
    """
    dtype, device = explicit_default_types(dtype, device)
    # Draw the samples on the CPU, where the generator seeded below is used, so
    # the moment is the same in every process whatever the current device is.
    if sample_stream == "torch_seed0":
        cached = _TORCH_RANDN_CACHE.get(id(f))
        if cached is not None and cached[1].shape[0] >= 1_000_000:
            z = cached[1]
        else:
            z = _torch_randn_seed0_fp64(1_000_000)
            _TORCH_RANDN_CACHE[id(f)] = (f, z)
        z = z.to(dtype=dtype, device=device)
        return f(z).pow(y=n).mean()
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

    def __init__(self, f, dtype=None, device=None, sample_stream="paddle"):
        super().__init__()
        with paddle.no_grad():
            cst = moment(
                f, 2, dtype="float64", device="cpu", sample_stream=sample_stream
            ).pow(y=-0.5).item()
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
            return self.f(x).multiply(paddle.to_tensor(self.cst))

    @staticmethod
    def _make_tracing_inputs(n: int):
        return [{"forward": (paddle.zeros(shape=(1,)),)}]

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

from typing import Dict, List, Tuple

import numpy as np
import paddle
from ..util.default_type import explicit_default_types
from ..util.jit import compile_mode

# ---------------------------------------------------------------------------
# The moment estimate samples a fixed-size standard-normal draw. To keep the
# random stream bit-identical to torch's seeded sampling, the draw is
# reimplemented on numpy (MT19937 + 16-block Box-Muller); the sample is
# computed once and cached for reuse.
# ---------------------------------------------------------------------------
_TORCH_RANDN_CACHE: Dict[int, Tuple[object, paddle.Tensor]] = {}


def _torch_randn_seed0_fp64(num: int) -> paddle.Tensor:
    """Bit-exact reproduction of torch.randn(num, generator=seed0, dtype=float64) on CPU."""
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


def moment(f, n, dtype=None, device=None):
    """
    compute n th moment
    <f(z)^n> for z normal
    """
    dtype, device = explicit_default_types(dtype, device)
    cached = _TORCH_RANDN_CACHE.get(id(f))
    if cached is not None and cached[1].shape[0] >= 1_000_000:
        z = cached[1]
    else:
        z = _torch_randn_seed0_fp64(1_000_000)
        _TORCH_RANDN_CACHE[id(f)] = (f, z)  # hold f so its id cannot be reused
    z = z.to(dtype=dtype, device=device) if dtype is not None else z
    return f(z).pow(n).mean()


@compile_mode("trace")
class normalize2mom(paddle.nn.Module):
    _is_id: bool
    cst: float

    def __init__(self, f, dtype=None, device=None) -> None:
        super().__init__()
        if device is None and isinstance(f, paddle.nn.Module):
            from ..util._argtools import _get_device

            device = _get_device(f)
        with paddle.no_grad():
            cst = moment(f, 2, dtype=paddle.float64, device=device).pow(-0.5).item()
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
            # scalar multiply via the operator form
            return self.f(x) * self.cst

    @staticmethod
    def _make_tracing_inputs(n: int) -> List[Dict[str, Tuple[paddle.Tensor]]]:
        return [{"forward": (paddle.zeros(size=(1,)),)}]

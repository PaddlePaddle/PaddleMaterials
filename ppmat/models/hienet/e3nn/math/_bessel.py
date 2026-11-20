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

import numpy as np
import paddle


def bessel(x: paddle.Tensor, n: int, x_max: float = 1.0) -> paddle.Tensor:
    """Bessel basis functions.

    They obey the following normalization:

    .. math::

        \\int_0^c r^2 B_n(r, c) B_m(r, c) dr = \\delta_{nm}

    Args:
        x (torch.Tensor): input of shape ``[...]``
        n (int): number of basis functions
        x_max (float): maximum value of the input

    Returns:
        torch.Tensor: basis functions of shape ``[..., n]``

    Klicpera, J.; Groß, J.; Günnemann, S. Directional Message Passing for Molecular Graphs; ICLR 2020.
    Equation (7)
    """
    assert isinstance(n, int)
    x = x[..., None]
    n = paddle.arange(1, n + 1, dtype=x.dtype, device=x.device)
    x_nonzero = paddle.where(x == 0.0, 1.0, x)
    return np.sqrt(2.0 / x_max) * paddle.where(
        x == 0,
        n * paddle.pi / x_max,
        paddle.sin(n * paddle.pi / x_max * x_nonzero) / x_nonzero,
    )

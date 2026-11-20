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

import paddle


class _SoftUnitStep(paddle.autograd.Function):
    @staticmethod
    def forward(ctx, x) -> paddle.Tensor:
        ctx.save_for_backward(x)
        y = paddle.zeros_like(x)
        mask = x > 0.0
        safe_x = paddle.where(mask, x, paddle.ones_like(x))
        y = paddle.where(mask, paddle.exp(-1.0 / safe_x), paddle.zeros_like(x))
        return y

    @staticmethod
    def backward(ctx, dy) -> paddle.Tensor:
        (x,) = ctx.saved_tensor()
        mask = x > 0.0
        safe_x = paddle.where(mask, x, paddle.ones_like(x))
        dx = paddle.where(
            mask, paddle.exp(-1.0 / safe_x) / (safe_x * safe_x), paddle.zeros_like(x)
        )
        return dx * dy


def soft_unit_step(x):
    """smooth :math:`C^\\infty` version of the unit step function

    .. math::

        x \\mapsto \\theta(x) e^{-1/x}


    Parameters
    ----------
    x : `torch.Tensor`
        tensor of shape :math:`(...)`

    Returns
    -------
    `torch.Tensor`
        tensor of shape :math:`(...)`

    Examples
    --------

    .. jupyter-execute::
        :hide-code:

        import torch
        from ppmat.models.hienet.e3nn.math import soft_unit_step
        import matplotlib.pyplot as plt

    .. jupyter-execute::

        x = torch.linspace(-1.0, 10.0, 1000)
        plt.plot(x, soft_unit_step(x));
    """
    return _SoftUnitStep.apply(x)

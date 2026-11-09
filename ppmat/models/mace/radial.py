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

"""Paddle port of the ``mace.modules.radial`` blocks used by MACE-MP-0.

Only the Bessel basis and the polynomial cutoff are needed by the released
``mace_mp_0`` (medium) checkpoint. Parameter / buffer names match the official
PyTorch modules one to one.
"""

import numpy as np
import paddle
from paddle import nn

__all__ = ["BesselBasis", "PolynomialCutoff", "RadialEmbeddingBlock"]


class BesselBasis(nn.Layer):
    """Spherical Bessel basis, equation (7) of the MACE paper."""

    def __init__(self, r_max: float, num_basis: int = 8, trainable: bool = False):
        super().__init__()
        dtype = paddle.get_default_dtype()
        bessel_weights = np.pi / r_max * np.linspace(
            1.0, float(num_basis), num_basis, dtype=np.float64
        )
        bessel_weights = paddle.to_tensor(bessel_weights, dtype=dtype)
        if trainable:
            self.bessel_weights = self.create_parameter(
                shape=[num_basis],
                dtype=dtype,
                default_initializer=nn.initializer.Assign(bessel_weights),
            )
        else:
            self.register_buffer("bessel_weights", bessel_weights)

        self.register_buffer("r_max", paddle.to_tensor(float(r_max), dtype=dtype))
        self.register_buffer(
            "prefactor", paddle.to_tensor(np.sqrt(2.0 / r_max), dtype=dtype)
        )

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:  # [..., 1]
        numerator = paddle.sin(self.bessel_weights * x)  # [..., num_basis]
        return self.prefactor * (numerator / x)


class PolynomialCutoff(nn.Layer):
    """Polynomial cutoff that goes from 1 to 0 as ``x`` goes from 0 to r_max."""

    def __init__(self, r_max: float, p: int = 6):
        super().__init__()
        self.register_buffer(
            "p", paddle.to_tensor(float(p), dtype=paddle.get_default_dtype())
        )
        self.register_buffer(
            "r_max", paddle.to_tensor(float(r_max), dtype=paddle.get_default_dtype())
        )

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        return self.calculate_envelope(x, self.r_max, self.p)

    @staticmethod
    def calculate_envelope(
        x: paddle.Tensor, r_max: paddle.Tensor, p: paddle.Tensor
    ) -> paddle.Tensor:
        # ``p`` is stored in the default (float) dtype, as in the converted
        # official checkpoint, but the envelope arithmetic has to happen in the
        # feature dtype. The exponent is kept as a Python float: a tensor
        # exponent requires ``elementwise_pow_grad``, which has no registered
        # gradient and therefore breaks the second-order autograd used for the
        # force and stress losses.
        p = float(p)
        r_over_r_max = x / r_max
        envelope = (
            1.0
            - ((p + 1.0) * (p + 2.0) / 2.0) * paddle.pow(r_over_r_max, p)
            + p * (p + 2.0) * paddle.pow(r_over_r_max, p + 1.0)
            - (p * (p + 1.0) / 2.0) * paddle.pow(r_over_r_max, p + 2.0)
        )
        return envelope * paddle.cast(x < r_max, x.dtype)


class RadialEmbeddingBlock(nn.Layer):
    """Radial embedding: Bessel basis multiplied by the polynomial cutoff."""

    def __init__(
        self,
        r_max: float,
        num_bessel: int,
        num_polynomial_cutoff: int,
        radial_type: str = "bessel",
        apply_cutoff: bool = True,
    ):
        super().__init__()
        if radial_type != "bessel":
            raise NotImplementedError(
                f"MACE-MP-0 only uses the Bessel basis, got radial_type={radial_type}."
            )
        self.bessel_fn = BesselBasis(r_max=r_max, num_basis=num_bessel)
        self.cutoff_fn = PolynomialCutoff(r_max=r_max, p=num_polynomial_cutoff)
        self.out_dim = num_bessel
        self.apply_cutoff = apply_cutoff

    def forward(
        self,
        edge_lengths: paddle.Tensor,  # [n_edges, 1]
        node_attrs=None,
        edge_index=None,
        atomic_numbers=None,
    ):
        cutoff = self.cutoff_fn(edge_lengths)  # [n_edges, 1]
        radial = self.bessel_fn(edge_lengths)  # [n_edges, n_basis]
        if not self.apply_cutoff:
            return radial, cutoff
        # The cutoff is folded into the radial features, so the interaction
        # block receives ``cutoff=None`` and must not apply it a second time.
        return radial * cutoff, None
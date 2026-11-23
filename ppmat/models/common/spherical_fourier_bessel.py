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
"""Numerical spherical Fourier-Bessel embeddings for SphereNet."""

import math
from functools import lru_cache

import numpy as np
import paddle
from scipy import optimize
from scipy import special


def _spherical_jn_root(x, order):
    return special.spherical_jn(order, x)


@lru_cache(maxsize=32)
def _build_basis_constants(num_spherical, num_radial):
    """Build deterministic Fourier-Bessel constants for one basis shape."""
    if num_spherical < 1:
        raise ValueError("num_spherical must be positive.")
    if num_radial < 1:
        raise ValueError("num_radial must be positive.")

    zeros = np.zeros((num_spherical, num_radial), dtype=np.float64)
    zeros[0] = np.arange(1, num_radial + 1, dtype=np.float64) * np.pi
    points = np.arange(1, num_radial + num_spherical, dtype=np.float64) * np.pi
    roots = np.zeros(num_radial + num_spherical - 1, dtype=np.float64)
    for order in range(1, num_spherical):
        for i in range(num_radial + num_spherical - 1 - order):
            roots[i] = optimize.brentq(
                _spherical_jn_root,
                points[i],
                points[i + 1],
                args=(order,),
            )
        points = roots.copy()
        zeros[order] = roots[:num_radial]

    normalizers = np.empty_like(zeros)
    for order in range(num_spherical):
        values = special.spherical_jn(order + 1, zeros[order])
        normalizers[order] = 1.0 / np.sqrt(0.5 * values**2)

    harmonic_prefactors = np.zeros((num_spherical, num_spherical), dtype=np.float64)
    for degree in range(num_spherical):
        for order in range(degree + 1):
            harmonic_prefactors[degree, order] = math.sqrt(
                (2 * degree + 1)
                * math.factorial(degree - order)
                / (4 * math.pi * math.factorial(degree + order))
            )

    zeros.setflags(write=False)
    normalizers.setflags(write=False)
    harmonic_prefactors.setflags(write=False)
    return zeros, normalizers, harmonic_prefactors


def _double_factorial(value):
    result = 1
    for factor in range(value, 0, -2):
        result *= factor
    return result


def _spherical_jn_series(order, x, num_terms=20):
    """Stable spherical-Bessel series around zero."""
    x_squared = x * x
    term = paddle.ones_like(x)
    series = term
    for index in range(1, num_terms):
        term = term * -x_squared / (2 * index * (2 * order + 2 * index + 1))
        series = series + term
    return x**order / _double_factorial(2 * order + 1) * series


class Envelope(paddle.nn.Layer):
    """Smooth polynomial envelope function for radial cutoff."""

    def __init__(self, exponent):
        super().__init__()
        self.p = exponent + 1
        self.a = -(self.p + 1) * (self.p + 2) / 2
        self.b = self.p * (self.p + 2)
        self.c = -self.p * (self.p + 1) / 2

    def forward(self, x):
        p, a, b, c = self.p, self.a, self.b, self.c
        x_pow_p0 = x.pow(p - 1)
        x_pow_p1 = x_pow_p0 * x
        x_pow_p2 = x_pow_p1 * x
        return 1.0 / x + a * x_pow_p0 + b * x_pow_p1 + c * x_pow_p2


class DistEmbedding(paddle.nn.Layer):
    """Radial basis with a smooth envelope cutoff."""

    def __init__(self, num_radial, cutoff=5.0, envelope_exponent=5):
        super().__init__()
        self.cutoff = cutoff
        self.envelope = Envelope(envelope_exponent)
        self.freq = paddle.create_parameter(
            shape=[num_radial],
            dtype=paddle.get_default_dtype(),
            default_initializer=paddle.nn.initializer.Assign(
                paddle.arange(
                    1, num_radial + 1, dtype=paddle.get_default_dtype()
                ).multiply(paddle.to_tensor(math.pi))
            ),
        )

    def reset_parameters(self):
        with paddle.no_grad():
            self.freq.set_value(
                paddle.arange(
                    1,
                    self.freq.shape[0] + 1,
                    dtype=paddle.get_default_dtype(),
                ).multiply(paddle.to_tensor(math.pi))
            )

    def forward(self, dist):
        dist = dist.unsqueeze(-1) / self.cutoff
        return self.envelope(dist) * paddle.sin(self.freq * dist)


class SphericalBesselBasis(paddle.nn.Layer):
    """Normalized spherical-Bessel basis evaluated with Paddle operations."""

    def __init__(self, num_spherical, num_radial, cutoff=5.0):
        super().__init__()
        self.num_spherical = num_spherical
        self.num_radial = num_radial
        self.cutoff = cutoff

        zeros, normalizers, _ = _build_basis_constants(num_spherical, num_radial)
        self.register_buffer(
            "zeros",
            paddle.to_tensor(np.array(zeros, copy=True), dtype="float32"),
            persistable=False,
        )
        self.register_buffer(
            "normalizers",
            paddle.to_tensor(np.array(normalizers, copy=True), dtype="float32"),
            persistable=False,
        )
        self.register_buffer(
            "degree_selectors",
            paddle.eye(num_spherical, dtype="float32").reshape(
                [num_spherical, 1, num_spherical, 1]
            ),
            persistable=False,
        )

    def forward(self, dist):
        scaled_dist = dist.reshape([-1, 1, 1]) / self.cutoff
        arguments = scaled_dist * self.zeros.reshape(
            [1, self.num_spherical, self.num_radial]
        )
        safe_arguments = paddle.where(
            paddle.abs(arguments) < 1e-2,
            paddle.full_like(arguments, 1e-2),
            arguments,
        )

        values = [paddle.sin(safe_arguments) / safe_arguments]
        if self.num_spherical > 1:
            values.append(
                paddle.sin(safe_arguments) / safe_arguments**2
                - paddle.cos(safe_arguments) / safe_arguments
            )
            for degree in range(1, self.num_spherical - 1):
                values.append(
                    (2 * degree + 1) / safe_arguments * values[-1] - values[-2]
                )

        basis = paddle.zeros_like(arguments)
        for degree in range(self.num_spherical):
            degree_arguments = arguments[:, degree, :]
            degree_values = values[degree][:, degree, :]
            degree_values = paddle.where(
                paddle.abs(degree_arguments) < degree + 1.0,
                _spherical_jn_series(degree, degree_arguments),
                degree_values,
            )
            basis = (
                basis
                + (degree_values * self.normalizers[degree]).unsqueeze(1)
                * self.degree_selectors[degree]
            )
        return basis


class RealSphericalHarmonics(paddle.nn.Layer):
    """Real spherical harmonics with the SphereNet ordering convention."""

    def __init__(self, num_spherical):
        super().__init__()
        self.num_spherical = num_spherical
        _, _, prefactors = _build_basis_constants(num_spherical, 1)
        self.register_buffer(
            "prefactors",
            paddle.to_tensor(np.array(prefactors, copy=True), dtype="float32"),
            persistable=False,
        )
        self.register_buffer(
            "m0_indices",
            paddle.to_tensor(
                [degree * degree for degree in range(num_spherical)],
                dtype="int64",
            ),
            persistable=False,
        )
        # Paddle stack/concat double gradients cannot handle the constant l=0
        # harmonic, so assemble columns with fixed non-trainable selectors.
        num_harmonics = num_spherical**2
        self.register_buffer(
            "harmonic_selectors",
            paddle.eye(num_harmonics, dtype="float32").reshape(
                [num_harmonics, 1, num_harmonics]
            ),
            persistable=False,
        )

    def forward(self, cos_angle, sin_angle, cos_torsion, sin_torsion):
        one = paddle.ones_like(cos_angle)
        # Looking up a dict by a tuple key breaks the SOT graph, so the table of
        # associated Legendre polynomials is keyed by degree * n + order.
        n = self.num_spherical
        polynomials = {0: one}

        torsion_cosines = [one]
        torsion_sines = [paddle.zeros_like(one)]
        for _ in range(1, self.num_spherical):
            previous_cos = torsion_cosines[-1]
            previous_sin = torsion_sines[-1]
            torsion_cosines.append(
                previous_cos * cos_torsion - previous_sin * sin_torsion
            )
            torsion_sines.append(
                previous_sin * cos_torsion + previous_cos * sin_torsion
            )

        for order in range(1, self.num_spherical):
            polynomials[order * n + order] = (1 - 2 * order) * polynomials[
                (order - 1) * n + (order - 1)
            ]

        for order in range(self.num_spherical - 1):
            polynomials[(order + 1) * n + order] = (
                (2 * order + 1) * cos_angle * polynomials[order * n + order]
            )

        for order in range(self.num_spherical):
            for degree in range(order + 2, self.num_spherical):
                polynomials[degree * n + order] = (
                    (2 * degree - 1) * cos_angle * polynomials[(degree - 1) * n + order]
                    - (order + degree - 1) * polynomials[(degree - 2) * n + order]
                ) / (degree - order)

        harmonics = []
        sqrt_two = math.sqrt(2.0)
        for degree in range(self.num_spherical):
            harmonics.append(self.prefactors[degree, 0] * polynomials[degree * n])
            for order in range(1, degree + 1):
                harmonics.append(
                    sqrt_two
                    * self.prefactors[degree, order]
                    * polynomials[degree * n + order]
                    * sin_angle**order
                    * torsion_cosines[order]
                )
            for order in range(degree, 0, -1):
                harmonics.append(
                    sqrt_two
                    * self.prefactors[degree, order]
                    * polynomials[degree * n + order]
                    * sin_angle**order
                    * torsion_sines[order]
                )
        result = paddle.zeros(
            [cos_angle.shape[0], self.num_spherical**2],
            dtype=cos_angle.dtype,
        )
        for index, value in enumerate(harmonics):
            result = result + value.unsqueeze(1) * self.harmonic_selectors[index]
        return result


class SphericalFourierBesselEmbedding(paddle.nn.Layer):
    """Shared angle and torsion Fourier-Bessel embedding."""

    def __init__(self, num_spherical, num_radial, cutoff=5.0):
        super().__init__()
        self.num_spherical = num_spherical
        self.num_radial = num_radial
        self.radial_basis = SphericalBesselBasis(num_spherical, num_radial, cutoff)
        self.spherical_harmonics = RealSphericalHarmonics(num_spherical)

    def forward(
        self,
        dist,
        angle_cos,
        angle_sin,
        torsion_cos,
        torsion_sin,
        idx_kj,
    ):
        radial_basis = self.radial_basis(dist)[idx_kj]
        harmonics = self.spherical_harmonics(
            angle_cos, angle_sin, torsion_cos, torsion_sin
        )
        angle_harmonics = paddle.index_select(
            harmonics,
            self.spherical_harmonics.m0_indices,
            axis=1,
        )

        num_triplets = angle_cos.shape[0]
        angle_embedding = (
            radial_basis
            * angle_harmonics.reshape([num_triplets, self.num_spherical, 1])
        ).reshape([num_triplets, self.num_spherical * self.num_radial])
        torsion_embedding = (
            radial_basis.reshape(
                [
                    num_triplets,
                    1,
                    self.num_spherical,
                    self.num_radial,
                ]
            )
            * harmonics.reshape(
                [
                    num_triplets,
                    self.num_spherical,
                    self.num_spherical,
                    1,
                ]
            )
        ).reshape(
            [
                num_triplets,
                self.num_spherical * self.num_spherical * self.num_radial,
            ]
        )
        return angle_embedding, torsion_embedding

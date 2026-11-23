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

"""
Adapted from https://github.com/FAIR-Chem/fairchem/blob/main/src/fairchem/core/models/gemnet/layers/radial_basis.py.
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found at https://github.com/FAIR-Chem/fairchem/blob/main/LICENSE.md.

"""

import math

import numpy as np
import paddle
from paddle import Tensor
from scipy.special import binom


class GaussianSmearing(paddle.nn.Layer):
    def __init__(
        self,
        start: float = 0.0,
        stop: float = 5.0,
        num_gaussians: int = 50,
    ):
        super(GaussianSmearing, self).__init__()
        offset = paddle.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item() ** 2
        self.register_buffer("offset", offset)

    def forward(self, dist: Tensor) -> Tensor:
        dist = dist.reshape([-1, 1]) - self.offset.reshape([1, -1])
        return paddle.exp(self.coeff * paddle.pow(dist, 2))


class PolynomialEnvelope(paddle.nn.Layer):
    """
    Polynomial envelope function that ensures a smooth cutoff.

    Parameters
    ----------
        exponent: int
            Exponent of the envelope function.
    """

    def __init__(self, exponent):
        super().__init__()
        assert exponent > 0
        self.p = exponent
        self.a = -(self.p + 1) * (self.p + 2) / 2
        self.b = self.p * (self.p + 2)
        self.c = -self.p * (self.p + 1) / 2

    def forward(self, d_scaled):
        env_val = (
            1
            + self.a * d_scaled**self.p
            + self.b * d_scaled ** (self.p + 1)
            + self.c * d_scaled ** (self.p + 2)
        )
        return paddle.where(
            condition=d_scaled < 1, x=env_val, y=paddle.zeros_like(x=d_scaled)
        )


class ExponentialEnvelope(paddle.nn.Layer):
    """
    Exponential envelope function that ensures a smooth cutoff,
    as proposed in Unke, Chmiela, Gastegger, Schütt, Sauceda, Müller 2021.
    SpookyNet: Learning Force Fields with Electronic Degrees of Freedom
    and Nonlocal Effects
    """

    def __init__(self):
        super().__init__()

    def forward(self, d_scaled):
        env_val = paddle.exp(x=-(d_scaled**2) / ((1 - d_scaled) * (1 + d_scaled)))
        return paddle.where(
            condition=d_scaled < 1, x=env_val, y=paddle.zeros_like(x=d_scaled)
        )


class SphericalBesselBasis(paddle.nn.Layer):
    """
    1D spherical Bessel basis

    Parameters
    ----------
    num_radial: int
        Controls maximum frequency.
    cutoff: float
        Cutoff distance in Angstrom.
    """

    def __init__(self, num_radial: int, cutoff: float):
        super().__init__()
        self.norm_const = math.sqrt(2 / cutoff**3)
        self.frequencies = paddle.base.framework.EagerParamBase.from_tensor(
            tensor=paddle.to_tensor(
                data=np.pi * np.arange(1, num_radial + 1, dtype=np.float32)
            ),
            trainable=True,
        )

    def forward(self, d_scaled):
        return (
            self.norm_const
            / d_scaled[:, None]
            * paddle.sin(x=self.frequencies * d_scaled[:, None])
        )


class BernsteinBasis(paddle.nn.Layer):
    """
    Bernstein polynomial basis,
    as proposed in Unke, Chmiela, Gastegger, Schütt, Sauceda, Müller 2021.
    SpookyNet: Learning Force Fields with Electronic Degrees of Freedom
    and Nonlocal Effects

    Parameters
    ----------
    num_radial: int
        Controls maximum frequency.
    pregamma_initial: float
        Initial value of exponential coefficient gamma.
        Default: gamma = 0.5 * a_0**-1 = 0.94486,
        inverse softplus -> pregamma = log e**gamma - 1 = 0.45264
    """

    def __init__(self, num_radial: int, pregamma_initial: float = 0.45264):
        super().__init__()
        prefactor = binom(num_radial - 1, np.arange(num_radial))
        self.register_buffer(
            name="prefactor",
            tensor=paddle.to_tensor(data=prefactor, dtype="float32"),
            persistable=False,
        )
        self.pregamma = paddle.base.framework.EagerParamBase.from_tensor(
            tensor=paddle.to_tensor(data=pregamma_initial, dtype="float32"),
            trainable=True,
        )
        self.softplus = paddle.nn.Softplus()
        exp1 = paddle.arange(end=num_radial)
        self.register_buffer(name="exp1", tensor=exp1[None, :], persistable=False)
        exp2 = num_radial - 1 - exp1
        self.register_buffer(name="exp2", tensor=exp2[None, :], persistable=False)

    def forward(self, d_scaled):
        gamma = self.softplus(self.pregamma)
        exp_d = paddle.exp(x=-gamma * d_scaled)[:, None]
        return self.prefactor * exp_d**self.exp1 * (1 - exp_d) ** self.exp2


class RadialBasis(paddle.nn.Layer):
    """

    Parameters
    ----------
    num_radial: int
        Controls maximum frequency.
    cutoff: float
        Cutoff distance in Angstrom.
    rbf: dict = {"name": "gaussian"}
        Basis function and its hyperparameters.
    envelope: dict = {"name": "polynomial", "exponent": 5}
        Envelope function and its hyperparameters.
    """

    def __init__(
        self,
        num_radial: int,
        cutoff: float,
        rbf: dict = {"name": "gaussian"},
        envelope: dict = {"name": "polynomial", "exponent": 5},
    ):
        super().__init__()
        self.num_radial = num_radial
        self.inv_cutoff = 1 / cutoff

        env_name = envelope["name"].lower()
        env_hparams = envelope.copy()
        del env_hparams["name"]
        if env_name == "polynomial":
            self.envelope = PolynomialEnvelope(**env_hparams)
        elif env_name == "exponential":
            self.envelope = ExponentialEnvelope()
        else:
            raise ValueError(f"Unknown envelope function '{env_name}'.")
        rbf_name = rbf["name"].lower()
        rbf_hparams = rbf.copy()
        del rbf_hparams["name"]
        if rbf_name == "gaussian":
            self.rbf = GaussianSmearing(
                start=0, stop=1, num_gaussians=num_radial, **rbf_hparams
            )
        elif rbf_name == "spherical_bessel":
            self.rbf = SphericalBesselBasis(num_radial=num_radial, cutoff=cutoff)
        elif rbf_name == "bernstein":
            self.rbf = BernsteinBasis(num_radial=num_radial, **rbf_hparams)
        else:
            raise ValueError(f"Unknown radial basis function '{rbf_name}'.")

    def forward(self, d):
        d_scaled = d * self.inv_cutoff
        env = self.envelope(d_scaled)
        return env[:, None] * self.rbf(d_scaled)

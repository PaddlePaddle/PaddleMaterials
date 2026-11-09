# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Generic VE-SDE scheduler with exponentially spaced sigmas.

Keeps only the standard VE-SDE mathematics (exponentially spaced sigmas,
predictor-corrector sampling). It deliberately does not own a ``sigma_norms``
buffer: the score-normalization table is model-specific (see
``ppmat.models.sgequidiff.diffusion_model``), so this shared-layer module has no
dependency on any model package and its ``__init_params__`` stay pure
YAML-friendly primitives.
"""

import numpy as np
import paddle
import paddle.nn as nn


def build_ve_sigma_grid(
    num_timesteps: int,
    sigma_min: float,
    sigma_max: float,
) -> paddle.Tensor:
    """Exponentially spaced VE-SDE sigma grid of length ``num_timesteps``."""
    return paddle.to_tensor(
        np.exp(np.linspace(np.log(sigma_min), np.log(sigma_max), num_timesteps)),
        dtype=paddle.float32,
    )


class ASUVESDEScheduler(nn.Layer):
    """VE-SDE scheduler for variance-exploding diffusion.

    Args:
        num_timesteps: Number of diffusion timesteps.
        sigma_min / sigma_max: Diffusion sigma range.

    The sigma grid comes from :func:`build_ve_sigma_grid` so that consumers
    (e.g. the SGEquiDiff sigma-norm table) share one construction instead of
    rebuilding it. Deliberately differs from ``ScoreSdeVeScheduler`` in
    ``scheduling_sde_ve.py`` on three points:

    - this class is an ``nn.Layer`` and registers the sigma grid as a buffer,
      so ``sigmas`` is part of the model state_dict; ``ScoreSdeVeScheduler``
      builds its sigmas lazily from ``set_timesteps``/``set_sigmas``;
    - ``step_pred``/``step_correct`` take an integer timestep plus a noise
      tensor supplied by the caller (Wyckoff-projected noise), and
      ``uniform_sample_timestep`` supplies the training timesteps;
      ``ScoreSdeVeScheduler`` takes a normalized continuous timestep in
      ``[sampling_eps, 1]``, draws its own noise via ``randn_tensor`` and
      has no timestep sampler;
    - ``step_correct`` guards the step size with an epsilon, a NaN-to-zero
      mapping and a ``max_step_size`` clamp; ``ScoreSdeVeScheduler`` has none.
    """

    def __init__(
        self,
        num_timesteps: int,
        sigma_min: float = 0.002,
        sigma_max: float = 0.5,
    ):
        super().__init__()
        self.num_timesteps = num_timesteps
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

        sigmas = build_ve_sigma_grid(num_timesteps, sigma_min, sigma_max)

        self.register_buffer(
            "sigmas",
            paddle.concat([paddle.zeros([1]), sigmas], axis=0),
        )

    def uniform_sample_timestep(self, batch_size: int) -> paddle.Tensor:
        """Uniformly sample integer timesteps in [1, num_timesteps]."""
        return paddle.randint(
            low=1,
            high=self.num_timesteps + 1,
            shape=[batch_size],
            dtype=paddle.int64,
        )

    @paddle.no_grad()
    def step_pred(
        self, x: paddle.Tensor, score: paddle.Tensor, t: int, noise: paddle.Tensor
    ) -> paddle.Tensor:
        """Predictor step: x_{t} = x_{t+1} + (sigma_{t+1}^2 - sigma_t^2) * score
        + sqrt(sigma_{t+1}^2 - sigma_t^2) * noise."""
        sigma_sq_diff = self.sigmas[t + 1] ** 2 - self.sigmas[t] ** 2
        return x + sigma_sq_diff * score + paddle.sqrt(sigma_sq_diff) * noise

    @paddle.no_grad()
    def step_correct(
        self,
        x: paddle.Tensor,
        score: paddle.Tensor,
        noise: paddle.Tensor,
        snr: float = 0.4,
        max_step_size: float = 1e6,
    ):
        """Corrector (Langevin) step: x = x + step_size * score
        + sqrt(2 * step_size) * noise.
        """
        noise_norm = ((noise**2).sum(axis=-1)).sqrt().mean()
        grad_norm = ((score**2).sum(axis=-1)).sqrt().mean()
        step_size = 2 * (snr * noise_norm / (grad_norm + 1e-12)) ** 2
        step_size = paddle.nan_to_num(
            step_size, nan=0.0, posinf=max_step_size, neginf=-max_step_size
        )
        step_size = paddle.where(noise == 0.0, paddle.zeros_like(noise), step_size)
        return x + step_size * score + paddle.sqrt(2 * step_size) * noise

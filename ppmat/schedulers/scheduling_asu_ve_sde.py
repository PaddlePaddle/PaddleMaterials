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

"""Discrete VE-SDE sampling with caller-supplied projected noise."""

from typing import Optional

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
    """VE-SDE scheduler with a checkpointed, one-based sigma grid.

    Unlike ``ScoreSdeVeScheduler``, both steps consume caller-projected noise.
    The corrector uses an explicit mask for constrained coordinates.

    Args:
        num_timesteps: Number of diffusion timesteps.
        sigma_min / sigma_max: Diffusion sigma range.
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
        self.timesteps = None

        sigmas = build_ve_sigma_grid(num_timesteps, sigma_min, sigma_max)

        self.register_buffer(
            "sigmas",
            paddle.concat([paddle.zeros([1]), sigmas], axis=0),
        )

    def set_timesteps(self, num_inference_steps: Optional[int] = None):
        """Select descending training-grid indices, including both endpoints.

        The one-based grid has ``num_inference_steps - 1`` transitions.
        """
        if num_inference_steps is None:
            num_inference_steps = self.num_timesteps
        if not 2 <= num_inference_steps <= self.num_timesteps:
            raise ValueError(
                "num_inference_steps must be between 2 and the training timesteps."
            )
        self.timesteps = np.linspace(
            self.num_timesteps, 1, num_inference_steps, dtype=int
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
        self,
        x: paddle.Tensor,
        score: paddle.Tensor,
        t: int,
        noise: paddle.Tensor,
        previous_t: Optional[int] = None,
    ) -> paddle.Tensor:
        """Reverse VE-SDE from the higher ``previous_t`` index to ``t``."""
        previous_t = t + 1 if previous_t is None else previous_t
        sigma_sq_diff = self.sigmas[previous_t] ** 2 - self.sigmas[t] ** 2
        return x + sigma_sq_diff * score + paddle.sqrt(sigma_sq_diff) * noise

    @paddle.no_grad()
    def step_correct(
        self,
        x: paddle.Tensor,
        score: paddle.Tensor,
        noise: paddle.Tensor,
        update_mask: paddle.Tensor,
        snr: float = 0.4,
        max_step_size: float = 1e6,
    ) -> paddle.Tensor:
        """Corrector (Langevin) step: x = x + step_size * score
        + sqrt(2 * step_size) * noise.
        """
        noise_norm = ((noise**2).sum(axis=-1)).sqrt().mean()
        grad_norm = ((score**2).sum(axis=-1)).sqrt().mean()
        step_size = 2 * (snr * noise_norm / (grad_norm + 1e-12)) ** 2
        step_size = paddle.nan_to_num(
            step_size, nan=0.0, posinf=max_step_size, neginf=0.0
        )
        step_size = paddle.clip(step_size, min=0.0, max=max_step_size)
        step_size = paddle.where(update_mask, step_size, paddle.zeros_like(noise))
        return x + step_size * score + paddle.sqrt(2 * step_size) * noise

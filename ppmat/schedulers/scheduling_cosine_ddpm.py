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

import math

import paddle

from ppmat.schedulers.scheduling_ddpm import DDPMScheduler


class CosineDDPMScheduler(DDPMScheduler):
    """DDPM with a configurable clipped cosine noise table."""

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        s: float = 0.008,
        beta_min: float = 1e-4,
        beta_max: float = 0.9999,
        **kwargs,
    ) -> None:
        if kwargs.get("trained_betas") is None:
            discretization = paddle.linspace(
                0, num_train_timesteps, num_train_timesteps + 1, dtype="float64"
            )
            alphas_cumprod = (
                paddle.cos(
                    (discretization / num_train_timesteps + s) / (1 + s) * math.pi * 0.5
                )
                ** 2
            )
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            kwargs["trained_betas"] = paddle.clip(
                1 - alphas_cumprod[1:] / alphas_cumprod[:-1], beta_min, beta_max
            ).cast("float32")
        super().__init__(num_train_timesteps=num_train_timesteps, **kwargs)

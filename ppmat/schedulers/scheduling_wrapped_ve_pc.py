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

import paddle

from ppmat.schedulers.scheduling_sde_ve import SchedulerOutput
from ppmat.schedulers.scheduling_sde_ve import ScoreSdeVeSchedulerWrapped
from ppmat.schedulers.scheduling_sde_ve import d_log_p_wrapped_normal


class WrappedVESDEPCScheduler(ScoreSdeVeSchedulerWrapped):
    """Wrapped VE predictor-corrector with integer per-atom timestep indices."""

    def __init__(self, step_lr: float = 1e-5, **kwargs) -> None:
        super().__init__(**kwargs)
        self.step_lr = step_lr
        self.sigmas_t = self.discrete_sigmas[:, None]
        self.sigmas_norm_t = self.discrete_sigmas_norm[:, None]

    def step_correct_indexed(
        self,
        model_output: paddle.Tensor,
        timestep: paddle.Tensor,
        sample: paddle.Tensor,
    ) -> SchedulerOutput:
        t_idx = timestep.cast("int64")
        st = self.sigmas_t[t_idx]
        snt = self.sigmas_norm_t[t_idx]
        step_size = self.step_lr * (st / self.sigma_min) ** 2
        drift = -step_size * model_output * paddle.sqrt(snt)
        diffusion = paddle.sqrt(2 * step_size) * paddle.randn(sample.shape)
        return SchedulerOutput(prev_sample=sample + drift + diffusion)

    def step_pred_indexed(
        self,
        model_output: paddle.Tensor,
        timestep: paddle.Tensor,
        sample: paddle.Tensor,
    ) -> SchedulerOutput:
        t_idx = timestep.cast("int64")
        st = self.sigmas_t[t_idx]
        # Clamping to sigma_min makes the last step's drift and diffusion zero.
        st_1 = self.sigmas_t[paddle.maximum(t_idx - 1, paddle.to_tensor(0))]
        snt = self.sigmas_norm_t[t_idx]
        step_size = st**2 - st_1**2
        drift = -step_size * model_output * paddle.sqrt(snt)
        diffusion = paddle.sqrt(
            st_1**2 * (st**2 - st_1**2) / (st**2)
        ) * paddle.randn(sample.shape)
        return SchedulerOutput(prev_sample=(sample + drift + diffusion) % 1.0)

    def score_target(
        self, noise: paddle.Tensor, timestep: paddle.Tensor
    ) -> paddle.Tensor:
        t_idx = timestep.cast("int64")
        st = self.sigmas_t[t_idx]
        snt = self.sigmas_norm_t[t_idx]
        return d_log_p_wrapped_normal(st * noise, st) / paddle.sqrt(snt)

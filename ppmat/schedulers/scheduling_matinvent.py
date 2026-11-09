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


from typing import Optional
from typing import Tuple
from typing import Union

import paddle

from ppmat.schedulers.scheduling_ddpm import DDPMScheduler
from ppmat.schedulers.scheduling_ddpm import DDPMSchedulerOutput
from ppmat.schedulers.scheduling_ddpm import betas_for_alpha_bar
from ppmat.schedulers.scheduling_sde_ve import SchedulerOutput
from ppmat.schedulers.scheduling_sde_ve import ScoreSdeVeSchedulerWrapped
from ppmat.schedulers.scheduling_sde_ve import SdeVeOutput
from ppmat.utils.paddle_utils import randn_tensor


class MatinventDDPMScheduler(DDPMScheduler):
    """MatInvent beta bounds and reference reverse-diffusion arithmetic."""

    def __init__(self, beta_clip_min=0.0, beta_clip_max=0.999, **kwargs):
        if not 0.0 <= beta_clip_min <= beta_clip_max < 1.0:
            raise ValueError("Require 0 <= beta_clip_min <= beta_clip_max < 1")
        self.beta_clip_min = beta_clip_min
        self.beta_clip_max = beta_clip_max
        if (
            kwargs.get("trained_betas") is None
            and kwargs.get("beta_schedule", "linear") == "squaredcos_cap_v2"
        ):
            kwargs["trained_betas"] = betas_for_alpha_bar(
                kwargs.get("num_train_timesteps", 1000), max_beta=beta_clip_max
            ).clip(min=beta_clip_min, max=beta_clip_max)
        super().__init__(**kwargs)
        alphas_cumprod_prev = paddle.concat(
            [
                paddle.ones([1], dtype=self.alphas_cumprod.dtype),
                self.alphas_cumprod[:-1],
            ]
        )
        self.sigmas = paddle.sqrt(
            self.betas * (1.0 - alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )

    def _get_variance(self, t, predicted_variance=None, variance_type=None):
        prev_t = self.previous_timestep(t)

        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else self.one
        current_beta_t = 1 - alpha_prod_t / alpha_prod_t_prev

        if prev_t == t - 1:
            variance = self.sigmas[t].square()
        else:
            variance = (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * current_beta_t

        variance = paddle.clip(variance, min=1e-20)

        if variance_type is None:
            variance_type = self.variance_type

        if variance_type == "fixed_small":
            variance = variance
        elif variance_type == "fixed_small_log":
            variance = paddle.log(variance)
            variance = paddle.exp(0.5 * variance)
        elif variance_type == "fixed_large":
            variance = current_beta_t
        elif variance_type == "fixed_large_log":
            variance = paddle.log(current_beta_t)
        elif variance_type == "learned":
            return predicted_variance
        elif variance_type == "learned_range":
            min_log = paddle.log(variance)
            max_log = paddle.log(current_beta_t)
            frac = (predicted_variance + 1) / 2
            variance = frac * max_log + (1 - frac) * min_log

        return variance

    def step(
        self,
        model_output: paddle.Tensor,
        timestep: int,
        sample: paddle.Tensor,
        generator=None,
        return_dict: bool = True,
    ) -> Union[DDPMSchedulerOutput, Tuple]:
        t = timestep

        prev_t = self.previous_timestep(t)

        if model_output.shape[1] == sample.shape[1] * 2 and self.variance_type in [
            "learned",
            "learned_range",
        ]:
            model_output, predicted_variance = paddle.split(
                model_output,
                [sample.shape[1], model_output.shape[1] - sample.shape[1]],
                axis=1,
            )
        else:
            predicted_variance = None

        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else self.one
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        current_alpha_t = alpha_prod_t / alpha_prod_t_prev
        current_beta_t = 1 - current_alpha_t

        if self.prediction_type == "epsilon":
            pred_original_sample = (
                sample - beta_prod_t ** (0.5) * model_output
            ) / alpha_prod_t ** (0.5)
        elif self.prediction_type == "sample":
            pred_original_sample = model_output
        elif self.prediction_type == "v_prediction":
            pred_original_sample = (alpha_prod_t**0.5) * sample - (
                beta_prod_t**0.5
            ) * model_output
        else:
            raise ValueError(
                f"prediction_type given as {self.prediction_type} must be one of "
                "`epsilon`, `sample` or `v_prediction`  for the DDPMScheduler."
            )

        if self.thresholding:
            pred_original_sample = self._threshold_sample(pred_original_sample)
        elif self.clip_sample:
            pred_original_sample = pred_original_sample.clip(
                -self.clip_sample_range, self.clip_sample_range
            )

        if (
            self.prediction_type == "epsilon"
            and not self.thresholding
            and not self.clip_sample
        ):
            alpha = self.alphas[t] if prev_t == t - 1 else current_alpha_t
            pred_prev_sample = (
                sample - (1 - alpha) / beta_prod_t.sqrt() * model_output
            ) / alpha.sqrt()
        else:
            pred_original_sample_coeff = (
                alpha_prod_t_prev ** (0.5) * current_beta_t
            ) / beta_prod_t
            current_sample_coeff = (
                current_alpha_t ** (0.5) * beta_prod_t_prev / beta_prod_t
            )
            pred_prev_sample = (
                pred_original_sample_coeff * pred_original_sample
                + current_sample_coeff * sample
            )

        variance = 0
        if t > 0:
            variance_noise = randn_tensor(
                model_output.shape, generator=generator, dtype=model_output.dtype
            )
            if self.variance_type == "fixed_small_log":
                variance = (
                    self._get_variance(t, predicted_variance=predicted_variance)
                    * variance_noise
                )
            elif self.variance_type == "learned_range":
                variance = self._get_variance(t, predicted_variance=predicted_variance)
                variance = paddle.exp(0.5 * variance) * variance_noise
            else:
                variance = (
                    self._get_variance(t, predicted_variance=predicted_variance) ** 0.5
                ) * variance_noise

        pred_prev_sample = pred_prev_sample + variance

        if not return_dict:
            return (pred_prev_sample,)

        return DDPMSchedulerOutput(
            prev_sample=pred_prev_sample, pred_original_sample=pred_original_sample
        )


class MatinventSdeVeScheduler(ScoreSdeVeSchedulerWrapped):
    """MatInvent corrector noise and lattice-aligned coordinate timesteps."""

    def _get_timestep_index(self, timestep, timestep_index):
        if self.timesteps is None:
            raise ValueError(
                "`self.timesteps` is not set, you need to run 'set_timesteps' after "
                "creating the scheduler"
            )
        if timestep_index is not None:
            return timestep_index
        return (timestep * (len(self.timesteps) - 1)).cast("int64")

    def step_correct(
        self,
        model_output: paddle.Tensor,
        timestep: float,
        sample: paddle.Tensor,
        generator: Optional[paddle.Generator] = None,
        return_dict: bool = True,
        noise: Optional[paddle.Tensor] = None,
        timestep_index: Optional[int] = None,
    ) -> Union[SchedulerOutput, Tuple]:
        timestep_discrete = self._get_timestep_index(timestep, timestep_index)
        sigma_norm = self.discrete_sigmas_norm[timestep_discrete]
        model_output = model_output * sigma_norm**0.5

        if self.sigmas.dtype != model_output.dtype:
            self.sigmas = self.sigmas.cast(model_output.dtype)

        if noise is None:
            noise = randn_tensor(sample.shape, generator=generator)

        step_size = (
            self.snr
            * (self.discrete_sigmas[timestep_discrete] / self.discrete_sigmas[0]) ** 2
        )
        prev_sample_mean = sample - step_size * model_output
        prev_sample = prev_sample_mean + ((step_size * 2) ** 0.5) * noise

        if not return_dict:
            return (prev_sample,)

        return SchedulerOutput(prev_sample=prev_sample)

    def step_pred(
        self,
        model_output: paddle.Tensor,
        timestep: float,
        sample: paddle.Tensor,
        generator: Optional[paddle.Generator] = None,
        return_dict: bool = True,
        timestep_index: Optional[int] = None,
        previous_timestep_index: Optional[int] = None,
    ) -> Union[SchedulerOutput, Tuple]:
        timestep_discrete = self._get_timestep_index(timestep, timestep_index)
        sigma_norm = self.discrete_sigmas_norm[timestep_discrete]
        model_output = model_output * sigma_norm**0.5

        if self.discrete_sigmas.dtype != model_output.dtype:
            self.discrete_sigmas = self.discrete_sigmas.cast(model_output.dtype)

        sigma = self.discrete_sigmas[timestep_discrete]
        if previous_timestep_index is None:
            adjacent_sigma = self.get_adjacent_sigma(timestep_discrete, timestep)
        elif previous_timestep_index >= 0:
            adjacent_sigma = self.discrete_sigmas[previous_timestep_index]
        else:
            adjacent_sigma = paddle.zeros_like(sigma)
        drift = paddle.zeros_like(sample)
        diffusion = (sigma**2 - adjacent_sigma**2) ** 0.5

        drift = drift + diffusion**2 * model_output

        noise = randn_tensor(sample.shape, generator=generator, dtype=sample.dtype)
        prev_sample_mean = (
            sample - drift
        )  # subtract because `dt` is a small negative timestep
        diffusion2 = adjacent_sigma / sigma

        prev_sample = (
            prev_sample_mean + diffusion * diffusion2 * noise
        )  # add impact of diffusion field g

        if not return_dict:
            return (prev_sample, prev_sample_mean)

        return SdeVeOutput(prev_sample=prev_sample, prev_sample_mean=prev_sample_mean)

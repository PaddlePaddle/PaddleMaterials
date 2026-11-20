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
import paddle.nn.functional as F

from ppmat.schedulers.scheduling_ddpm import DDPMScheduler
from ppmat.schedulers.scheduling_sde_ve import SchedulerOutput
from ppmat.schedulers.scheduling_sde_ve import ScoreSdeVeSchedulerWrapped
from ppmat.schedulers.scheduling_sde_ve import d_log_p_wrapped_normal


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


def _sample_categorical(probs: paddle.Tensor) -> paddle.Tensor:
    """Sample an index from probabilities (Paddle Categorical takes logits)."""
    logits = paddle.log(probs.clip(1e-12))
    return paddle.distribution.Categorical(logits=logits).sample().cast("int64")


class D3PMUniformScheduler:
    """D3PM with uniform transitions and a cosine retention schedule.

    Args:
        num_train_timesteps (int): Number of diffusion steps.
        num_types (int): Number of categorical atom types.
        s (float): Offset of the cosine retention schedule.
        loss_scale (float): Multiplier for the per-atom KL loss.
        kl_eps (float): Numerical offset inside KL logarithms.
    """

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        num_types: int = 100,
        s: float = 0.008,
        loss_scale: float = 1000,
        kl_eps: float = 1e-4,
    ) -> None:
        self.num_train_timesteps = num_train_timesteps
        self.num_types = num_types
        self.loss_scale = loss_scale
        self.kl_eps = kl_eps

        discretization = paddle.arange(1, num_train_timesteps + 1, dtype="float64")
        f_t = paddle.cos(
            (discretization / (num_train_timesteps + 1) + s) / (1 + s) * math.pi / 2
        )
        f_0 = paddle.cos(
            (paddle.to_tensor(0.0, dtype="float64") + s) / (1 + s) * math.pi / 2
        )
        cumprod_alphas_t = f_t / f_0
        cumprod_alphas_t_1 = paddle.concat(
            [paddle.to_tensor([1.0], dtype="float64"), cumprod_alphas_t[:-1]]
        )
        betas_t = 1 - cumprod_alphas_t / cumprod_alphas_t_1

        # Q_t = (1 - beta_t) I + (beta_t / K) J, computed in float64 then cast
        # to float32; cumprod_Q_t accumulates per-step float32 matmuls.
        eye64 = paddle.eye(num_types, dtype="float64")
        ones64 = paddle.ones([1, num_types, num_types], dtype="float64")

        Q_t = (
            (1 - betas_t)[:, None, None] * eye64[None, :, :]
            + (betas_t / float(num_types))[:, None, None] * ones64
        ).cast("float32")

        cumprod_Q_t_list = [Q_t[0]]
        for t_idx in range(1, num_train_timesteps):
            cumprod_Q_t_list.append(paddle.matmul(cumprod_Q_t_list[-1], Q_t[t_idx]))
        cumprod_Q_t = paddle.stack(cumprod_Q_t_list, axis=0)

        cumprod_Q_t_1 = paddle.concat(
            [
                paddle.eye(num_types, dtype="float32").unsqueeze(0),
                cumprod_Q_t[:-1],
            ],
            axis=0,
        )

        self.Q_t = Q_t
        self.cumprod_Q_t = cumprod_Q_t
        self.cumprod_Q_t_1 = cumprod_Q_t_1

    def to_domain(self, atom_types: paddle.Tensor) -> paddle.Tensor:
        return F.one_hot(atom_types, num_classes=self.num_types).cast("float32")

    @staticmethod
    def from_domain(onehot: paddle.Tensor) -> paddle.Tensor:
        return onehot.argmax(axis=-1)

    @staticmethod
    def prediction_to_domain(prediction: paddle.Tensor) -> paddle.Tensor:
        return F.softmax(prediction, axis=-1)

    def add_noise(self, x0: paddle.Tensor, t: paddle.Tensor) -> paddle.Tensor:
        onehot_x0 = self.to_domain(x0)
        xt_probs = paddle.matmul(
            onehot_x0[:, None, :], self.cumprod_Q_t[t.cast("int64")]
        )[:, 0, :]
        xt = _sample_categorical(xt_probs)
        return self.to_domain(xt)

    def _reverse_step_distribution(
        self, onehot_x0: paddle.Tensor, onehot_xt: paddle.Tensor, t: paddle.Tensor
    ) -> paddle.Tensor:
        t_idx = t.cast("int64")
        numerator = (
            paddle.matmul(onehot_xt[:, None, :], self.Q_t[t_idx].transpose([0, 2, 1]))[
                :, 0, :
            ]
            * paddle.matmul(onehot_x0[:, None, :], self.cumprod_Q_t_1[t_idx])[:, 0, :]
        )
        denominator = (
            paddle.matmul(onehot_x0[:, None, :], self.cumprod_Q_t[t_idx])[:, 0, :]
            * onehot_xt
        ).sum(axis=-1)[:, None]
        result = numerator / (denominator + 1e-8)
        result = result / result.sum(axis=-1, keepdim=True)
        # Zero-denominator fallback: uniform distribution.
        return paddle.where(
            paddle.isnan(result),
            paddle.full_like(result, 1.0 / self.num_types),
            result,
        )

    def step(
        self, onehot_pred: paddle.Tensor, onehot_xt: paddle.Tensor, t: paddle.Tensor
    ) -> paddle.Tensor:
        onehot_x0 = self.prediction_to_domain(onehot_pred)
        xt_1_probs = self._reverse_step_distribution(onehot_x0, onehot_xt, t)
        if (t.cast("int64") == 0).all():
            return self.to_domain(self.from_domain(xt_1_probs.cast("float32")))
        xt_1 = _sample_categorical(xt_1_probs)
        return self.to_domain(xt_1)

    def prior_sampling(self, shape: tuple) -> paddle.Tensor:
        xT_probs = paddle.ones(shape, dtype="float32") / self.num_types
        xT = _sample_categorical(xT_probs)
        return self.to_domain(xT)

    def compute_loss(
        self,
        onehot_pred: paddle.Tensor,
        onehot_xt: paddle.Tensor,
        onehot_x0: paddle.Tensor,
        t: paddle.Tensor,
    ) -> paddle.Tensor:
        """Per-atom D3PM KL loss scaled by ``loss_scale``."""
        t_idx = t.cast("int64")
        onehot_x0_pred = self.prediction_to_domain(onehot_pred)
        pred_xt_1_probs = self._reverse_step_distribution(
            onehot_x0_pred, onehot_xt, t_idx
        )
        orig_xt_1_probs = self._reverse_step_distribution(onehot_x0, onehot_xt, t_idx)
        kl_loss = (
            (
                orig_xt_1_probs
                * (
                    paddle.log(orig_xt_1_probs + self.kl_eps)
                    - paddle.log(pred_xt_1_probs + self.kl_eps)
                )
            )
            .reshape([onehot_xt.shape[0], -1])
            .sum(axis=-1)
        )
        return self.loss_scale * kl_loss

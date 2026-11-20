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


def _sample_categorical(probs: paddle.Tensor) -> paddle.Tensor:
    """Sample an index from probabilities (Paddle Categorical takes logits)."""
    logits = paddle.log(probs.clip(1e-12))
    return paddle.distribution.Categorical(logits=logits).sample().cast("int64")


class D3PMUniformScheduler:
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        num_types: int = 100,
        s: float = 0.008,
    ):
        self.num_types = num_types

        discretization = paddle.arange(1, num_train_timesteps + 1, dtype="float64")
        f_t = paddle.cos(
            (discretization / (num_train_timesteps + 1) + s) / (1 + s) * math.pi / 2
        )
        f_0 = paddle.cos(
            (paddle.to_tensor(0.0, dtype="float64") + s) / (1 + s) * math.pi / 2
        )
        a_t = f_t / f_0
        cumprod_alphas_t = a_t
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

        self.Q_t = Q_t.reshape([-1, num_types, num_types])
        self.cumprod_Q_t = cumprod_Q_t.reshape([-1, num_types, num_types])
        self.cumprod_Q_t_1 = cumprod_Q_t_1.reshape([-1, num_types, num_types])


class D3PMUniformDiffusion:
    """D3PM diffusion process over discrete categorical variables with a
    uniform transition kernel (forward noising, reverse sampling, prior
    sampling, and per-atom KL loss)."""

    def __init__(
        self, scheduler_cfg: dict, loss_scale: float = 1000, kl_eps: float = 1e-4
    ) -> None:
        # Deferred import: build_scheduler is defined in ppmat.schedulers.__init__,
        # which imports this module.
        from ppmat.schedulers import build_scheduler

        self.scheduler = build_scheduler(scheduler_cfg)
        self.num_types = self.scheduler.num_types
        self.Q_t = self.scheduler.Q_t
        self.cumprod_Q_t = self.scheduler.cumprod_Q_t
        self.cumprod_Q_t_1 = self.scheduler.cumprod_Q_t_1

        self.loss_scale = loss_scale
        self.kl_eps = kl_eps

    def to_domain(self, atom_types: paddle.Tensor) -> paddle.Tensor:
        return F.one_hot(atom_types, num_classes=self.num_types).cast("float32")

    @staticmethod
    def from_domain(onehot: paddle.Tensor) -> paddle.Tensor:
        return onehot.argmax(axis=-1)

    @staticmethod
    def prediction_to_domain(prediction: paddle.Tensor) -> paddle.Tensor:
        return F.softmax(prediction, axis=-1)

    def output_transform(self, x0: paddle.Tensor) -> paddle.Tensor:
        return self.from_domain(x0)

    def forward_step_sample(self, x0, t):
        onehot_x0 = self.to_domain(x0)
        xt_probs = paddle.matmul(
            onehot_x0[:, None, :], self.cumprod_Q_t[t.cast("int64")]
        )[:, 0, :]
        xt = _sample_categorical(xt_probs)
        return self.to_domain(xt)

    def _reverse_step_distribution(self, onehot_x0, onehot_xt, t):
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

    def reverse_step_sample(self, onehot_pred, onehot_xt, t):
        onehot_x0 = self.prediction_to_domain(onehot_pred)
        xt_1_probs = self._reverse_step_distribution(onehot_x0, onehot_xt, t)
        if (t.cast("int64") == 0).all():
            return self.to_domain(self.from_domain(xt_1_probs.cast("float32")))
        xt_1 = _sample_categorical(xt_1_probs)
        return self.to_domain(xt_1)

    def prior_sample(self, batch):
        na = batch["num_atoms"]
        total = int(na.sum())
        shape = [total, self.num_types]
        xT_probs = paddle.ones(shape, dtype="float32") / self.num_types
        xT = _sample_categorical(xT_probs)
        return self.to_domain(xT)

    def loss(self, onehot_pred, onehot_xt, onehot_x0, t):
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

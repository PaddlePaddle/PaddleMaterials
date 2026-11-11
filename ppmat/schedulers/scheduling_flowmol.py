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

"""Interpolant scheduler for FlowMol flow matching."""

from __future__ import annotations

import paddle


class FlowMolInterpolantScheduler:
    """Schedules the interpolation progress alpha(t) of every feature.

    Not reused: the existing noise/transition schedulers act on discrete
    timesteps and expose neither continuous alpha(t) nor its analytic derivative.

    ``schedule_type`` may be a single schedule name applied to all features or
    a per-feature mapping. Cosine schedules additionally require the ``nu``
    exponent of every feature in ``cosine_params``.
    """

    supported_schedule_types = ["cosine", "linear"]

    def __init__(
        self,
        feature_names: list,
        schedule_type,
        cosine_params: dict,
    ):
        self.feature_names = feature_names

        if not isinstance(schedule_type, (str, dict)):
            raise ValueError("schedule_type must be a string or a dictionary")

        if isinstance(schedule_type, str):
            if schedule_type not in self.supported_schedule_types:
                raise ValueError(f"unsupported schedule_type: {schedule_type}")
            self.schedule_dict = {feat: schedule_type for feat in self.feature_names}
        else:
            for feat in self.feature_names:
                if feat not in schedule_type:
                    raise ValueError(f"must specify schedule_type for feature {feat}")
            self.schedule_dict = schedule_type

        self.cosine_params = cosine_params

        for feat, feat_schedule_type in self.schedule_dict.items():
            if feat_schedule_type == "cosine" and feat not in self.cosine_params:
                raise ValueError(f"must specify cosine_params for feature {feat}")

    def interpolant_weights(self, t: paddle.Tensor):
        """Return the weights of the prior and the endpoint at time ``t``."""
        alpha_t = self.alpha_t(t)
        return 1 - alpha_t, alpha_t

    def loss_weights(self, t: paddle.Tensor) -> paddle.Tensor:
        """Time-dependent loss weights; small at the start of the trajectory."""
        alpha_t = self.alpha_t(t)
        weights = alpha_t / (1 - alpha_t)
        return paddle.clip(weights, min=0.05, max=1.5)

    def alpha_t(self, t: paddle.Tensor) -> paddle.Tensor:
        per_feat_alpha = []
        for feat in self.feature_names:
            schedule_type = self.schedule_dict[feat]
            if schedule_type == "cosine":
                alpha_t = self.cosine_alpha_t(t, nu=self.cosine_params[feat])
            elif schedule_type == "linear":
                alpha_t = self.linear_alpha_t(t)
            else:
                raise ValueError(f"unsupported schedule_type: {schedule_type}")
            per_feat_alpha.append(alpha_t)
        return paddle.concat(per_feat_alpha, axis=1)

    def alpha_t_prime(self, t: paddle.Tensor) -> paddle.Tensor:
        per_feat_alpha_prime = []
        for feat in self.feature_names:
            schedule_type = self.schedule_dict[feat]
            if schedule_type == "cosine":
                alpha_t_prime = self.cosine_alpha_t_prime(
                    t, nu=self.cosine_params[feat]
                )
            elif schedule_type == "linear":
                alpha_t_prime = self.linear_alpha_t_prime(t)
            else:
                raise ValueError(f"unsupported schedule_type: {schedule_type}")
            per_feat_alpha_prime.append(alpha_t_prime)
        return paddle.concat(per_feat_alpha_prime, axis=1)

    def cosine_alpha_t(self, t: paddle.Tensor, nu) -> paddle.Tensor:
        t = t.unsqueeze(-1)
        return 1 - paddle.cos(paddle.pi * 0.5 * paddle.pow(t, nu)).square()

    def cosine_alpha_t_prime(self, t: paddle.Tensor, nu) -> paddle.Tensor:
        t = paddle.clip(t, min=1e-9)
        t = t.unsqueeze(-1)
        sin_input = paddle.pi * paddle.pow(t, nu)
        return paddle.pi * 0.5 * paddle.sin(sin_input) * nu * paddle.pow(t, nu - 1)

    def linear_alpha_t(self, t: paddle.Tensor) -> paddle.Tensor:
        return t.unsqueeze(-1)

    def linear_alpha_t_prime(self, t: paddle.Tensor) -> paddle.Tensor:
        return paddle.ones_like(t).unsqueeze(-1)

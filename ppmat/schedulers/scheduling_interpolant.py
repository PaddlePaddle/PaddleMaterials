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


class InterpolantScheduler:
    """Continuous-time interpolation schedules with analytic derivatives.

    For each feature, linear interpolation uses alpha(t) = t and power-cosine
    interpolation uses alpha(t) = sin(pi * t**nu / 2)**2, with t in [0, 1].
    Inputs have shape [batch_size]; outputs have shape [batch_size, num_features]
    in the order given by feature_names.

    Args:
        feature_names (list): Feature names in interpolation order.
        schedule_type (str | dict): A single schedule name applied to every
            feature, or a per-feature mapping of ``"cosine"``/``"linear"``.
        cosine_params (dict): Finite ``nu >= 0.5`` for every cosine-scheduled
            feature, ensuring a finite right derivative at t=0.
    """

    supported_schedule_types = ["cosine", "linear"]

    def __init__(
        self,
        feature_names: list,
        schedule_type,
        cosine_params: dict,
    ):
        self.feature_names = list(feature_names)

        if not isinstance(schedule_type, (str, dict)):
            raise ValueError("schedule_type must be a string or a dictionary")

        if isinstance(schedule_type, str):
            self.schedule_dict = {feat: schedule_type for feat in self.feature_names}
        else:
            for feat in self.feature_names:
                if feat not in schedule_type:
                    raise ValueError(f"must specify schedule_type for feature {feat}")
            self.schedule_dict = {
                feat: schedule_type[feat] for feat in self.feature_names
            }

        self.cosine_params = dict(cosine_params)

        for feat, feat_schedule_type in self.schedule_dict.items():
            if feat_schedule_type not in self.supported_schedule_types:
                raise ValueError(f"unsupported schedule_type: {feat_schedule_type}")
            if feat_schedule_type == "cosine":
                if feat not in self.cosine_params:
                    raise ValueError(f"must specify cosine_params for feature {feat}")
                nu = self.cosine_params[feat]
                if not math.isfinite(nu) or nu < 0.5:
                    raise ValueError(
                        f"cosine_params for feature {feat} must be finite and >= 0.5"
                    )

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
        return paddle.sin(paddle.pi * 0.5 * paddle.pow(t, nu)).square()

    def cosine_alpha_t_prime(self, t: paddle.Tensor, nu) -> paddle.Tensor:
        t = t.unsqueeze(-1)
        # The sinc form evaluates the right derivative at zero without clipping.
        alpha_t_prime = (
            paddle.pi**2
            * 0.5
            * nu
            * paddle.pow(t, 2 * nu - 1)
            * paddle.sinc(paddle.pow(t, nu))
        )
        return paddle.where(t == 1, paddle.zeros_like(t), alpha_t_prime)

    def linear_alpha_t(self, t: paddle.Tensor) -> paddle.Tensor:
        return t.unsqueeze(-1)

    def linear_alpha_t_prime(self, t: paddle.Tensor) -> paddle.Tensor:
        return paddle.ones_like(t).unsqueeze(-1)

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

from __future__ import annotations

import paddle


class InterpolantScheduler:
    """Schedules the linear interpolation weights alpha_t of every feature."""

    def __init__(self, feature_names: list):
        self.feature_names = feature_names

    def alpha_t(self, t: paddle.Tensor) -> paddle.Tensor:
        return paddle.tile(t.unsqueeze(-1), [1, len(self.feature_names)])

    def alpha_t_prime(self, t: paddle.Tensor) -> paddle.Tensor:
        return paddle.tile(
            paddle.ones_like(t).unsqueeze(-1), [1, len(self.feature_names)]
        )

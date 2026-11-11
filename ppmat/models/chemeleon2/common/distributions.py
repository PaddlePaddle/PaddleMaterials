# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle

__all__ = [
    "DiagonalGaussianDistribution",
]


class DiagonalGaussianDistribution:
    def __init__(self, parameters):
        self.parameters = parameters
        self.mean, self.logvar = paddle.chunk(parameters, 2, axis=1)
        self.logvar = paddle.clip(self.logvar, -30.0, 20.0)
        self.std = paddle.exp(0.5 * self.logvar)
        self.var = paddle.exp(self.logvar)

    def sample(self):
        return self.mean + self.std * paddle.randn(self.mean.shape)

    def kl(self, other=None):
        if other is None:
            return 0.5 * paddle.sum(
                paddle.pow(self.mean, 2) + self.var - 1.0 - self.logvar, axis=[1]
            )
        return 0.5 * paddle.sum(
            paddle.pow(self.mean - other.mean, 2) / other.var
            + self.var / other.var
            - 1.0
            - self.logvar
            + other.logvar,
            axis=[1],
        )

    def mode(self):
        return self.mean

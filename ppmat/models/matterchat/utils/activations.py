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

# Shared activation function registry for MatterChat.

from __future__ import annotations

from paddle import nn


class ScaledSiLU(nn.Layer):
    """Scaled Sigmoid Linear Unit."""

    def __init__(self) -> None:
        super().__init__()
        self.scale_factor = 1 / 0.6
        self._activation = nn.Silu()

    def forward(self, x):
        return self._activation(x) * self.scale_factor


ACT2FN = {
    "gelu": nn.GELU,
    "gelu_new": nn.GELU,
    "gelu_python": nn.GELU,
    "relu": nn.ReLU,
    "silu": nn.Silu,
    "scaledsilu": ScaledSiLU,
    "softplus": nn.Softplus,
    "sigmoid": nn.Sigmoid,
    "tanh": nn.Tanh,
}


def get_activation(name: str) -> nn.Layer:
    if isinstance(name, str):
        if name.lower() in ACT2FN:
            return ACT2FN[name.lower()]()
        raise ValueError(f"Unsupported activation: {name}")
    return name

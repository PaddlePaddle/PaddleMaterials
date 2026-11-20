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

# Shared weight initialization and model utilities for MatterChat.

from __future__ import annotations

import paddle
import paddle.nn as nn


def default_init_weights(module, initializer_range: float = 0.02):
    """Default weight initialization for Linear, Embedding, and LayerNorm."""
    if isinstance(module, nn.Linear):
        module.weight.set_value(
            paddle.normal(shape=module.weight.shape, mean=0.0, std=initializer_range)
        )
        if module.bias is not None:
            module.bias.set_value(paddle.zeros(module.bias.shape))
    elif isinstance(module, nn.Embedding):
        module.weight.set_value(
            paddle.normal(shape=module.weight.shape, mean=0.0, std=initializer_range)
        )
        if hasattr(module, "_padding_idx") and module._padding_idx is not None:
            module.weight[module._padding_idx].set_value(
                paddle.zeros([module.weight.shape[1]])
            )
    elif isinstance(module, nn.LayerNorm):
        module.bias.set_value(paddle.zeros(module.bias.shape))
        module.weight.set_value(paddle.ones(module.weight.shape))

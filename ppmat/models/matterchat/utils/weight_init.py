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

"""Weight initialization for MatterChat transformer blocks."""

import paddle.nn as nn

from ppmat.models.common.initializer import normal_
from ppmat.models.common.initializer import ones_
from ppmat.models.common.initializer import zeros_


def default_init_weights(module, initializer_range: float = 0.02):
    """Default weight initialization for Linear, Embedding, and LayerNorm."""
    if isinstance(module, nn.Linear):
        normal_(module.weight, std=initializer_range)
        if module.bias is not None:
            zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        normal_(module.weight, std=initializer_range)
        if module._padding_idx is not None:
            zeros_(module.weight[module._padding_idx])
    elif isinstance(module, nn.LayerNorm):
        zeros_(module.bias)
        ones_(module.weight)

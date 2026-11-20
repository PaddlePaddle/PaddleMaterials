# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.

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
import paddle.nn as nn
import paddle.nn.functional as F
from typing_extensions import Literal


class MSELoss(nn.Layer):
    r"""Class for mean squared error loss."""

    def __init__(
        self,
        reduction: Literal["mean", "sum"] = "mean",
    ):
        super().__init__()
        if reduction not in ["mean", "sum"]:
            raise ValueError(
                f"reduction should be 'mean' or 'sum', but got {reduction}"
            )
        self.reduction = reduction

    def forward(self, pred, label) -> paddle.Tensor:

        loss = F.mse_loss(pred, label, self.reduction)
        return loss


class MaskMSELoss(nn.Layer):
    r"""Mean squared error over valid entries selected by a boolean mask."""

    def forward(self, pred, label, mask) -> paddle.Tensor:
        diff = paddle.square(pred - label)
        mask = paddle.cast(mask, pred.dtype)
        denom = paddle.clip(paddle.sum(mask), min=1.0)
        return paddle.sum(diff * mask) / denom

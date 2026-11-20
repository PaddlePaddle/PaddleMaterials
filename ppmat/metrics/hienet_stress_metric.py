# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

# Adapted from https://github.com/divelab/AIRS (OpenMat/HIENet)

"""Stress metric for HIENet.

The HIENet contract forward returns stress in Voigt layout ``[B, 6]``
(``[xx, yy, zz, xy, yz, xz]``) while the dataset label keeps the raw 3x3
stress tensor. This metric applies the same label transform as the embedded
loss (negate + Voigt reorder), masks NaN labels, and reports the L1 (MAE)
value — the semantics of ``IgnoreNanMetricWrapper`` wrapping
``paddle.nn.L1Loss``. Self-contained on purpose: imports nothing from
``ppmat.models`` to avoid a circular import through the registries.
"""

import paddle

# Raw 3x3 indices in the model's Voigt order [xx, yy, zz, xy, yz, xz].
_STRESS_VOIGT_3X3 = ((0, 0), (1, 1), (2, 2), (0, 1), (1, 2), (0, 2))


class HIENetStressVoigtMetric(paddle.nn.Layer):
    """MAE between Voigt [B, 6] stress predictions and raw 3x3 labels."""

    def forward(self, pred, label):
        """Compute the NaN-masked L1 between ``pred`` and ``label``.

        Args:
            pred (Tensor): stress in Voigt layout, shape [B, 6].
            label (Tensor): raw 3x3 stress tensor, shape [B, 3, 3] (already
                Voigt [B, 6] labels pass through unchanged).

        Returns:
            Tensor: scalar MAE, or NaN when every label entry is NaN.
        """
        if not isinstance(pred, paddle.Tensor):
            pred = paddle.to_tensor(pred)
        if not isinstance(label, paddle.Tensor):
            label = paddle.to_tensor(label)
        label = paddle.cast(label, pred.dtype)
        if label.ndim == 3 and label.shape[-1] == 3:
            # Raw 3x3 label -> model Voigt convention: the same single
            # transform as the embedded loss (negate + [xx,yy,zz,xy,yz,xz]).
            label = -paddle.stack(
                [label[:, i, j] for (i, j) in _STRESS_VOIGT_3X3], axis=1
            )
        valid = ~paddle.isnan(label)
        valid_label = label[valid]
        valid_pred = pred[valid]
        if valid_label.numel() > 0:
            return paddle.nn.functional.l1_loss(valid_pred, valid_label)
        return paddle.nan

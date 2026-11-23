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

from __future__ import annotations

from typing import Dict
from typing import Optional
from typing import Sequence
from typing import Set

import paddle

from ppmat.metrics.streaming_base import StreamingMetricBase
from ppmat.metrics.streaming_base import _all_reduce_sum_


def calc_psnr(
    pred: paddle.Tensor,
    label: paddle.Tensor,
    data_range: float = 255.0,
    eps: float = 1e-12,
) -> paddle.Tensor:
    """Compute batch PSNR for image tensors with shape [N, C, H, W]."""
    pred = pred.astype("float64")
    label = label.astype("float64")
    diff = (pred - label) / data_range
    mse = paddle.mean(diff * diff)
    mse = paddle.maximum(mse, paddle.to_tensor(eps, dtype=mse.dtype))
    return -10.0 * paddle.log10(mse)


def _gaussian_window_2d(
    channels: int,
    win_size: int = 11,
    win_sigma: float = 1.5,
    dtype: str = "float32",
) -> paddle.Tensor:
    coords = paddle.arange(win_size, dtype=dtype) - (win_size // 2)
    gauss = paddle.exp(-(coords**2) / (2.0 * (win_sigma**2)))
    gauss = gauss / paddle.sum(gauss)
    window_2d = gauss.unsqueeze(1) * gauss.unsqueeze(0)
    window = window_2d.reshape([1, 1, win_size, win_size])
    return paddle.tile(window, [channels, 1, 1, 1])


def calc_ssim(
    pred: paddle.Tensor,
    label: paddle.Tensor,
    data_range: float = 255.0,
    win_size: int = 11,
    win_sigma: float = 1.5,
    k1: float = 0.01,
    k2: float = 0.03,
    nonnegative_ssim: bool = False,
) -> paddle.Tensor:
    """Compute batch SSIM for image tensors with shape [N, C, H, W]."""
    if pred.shape != label.shape:
        raise ValueError(
            f"Input images should have the same dimensions, got {pred.shape} and {label.shape}."
        )
    if len(pred.shape) != 4:
        raise ValueError(
            f"Input images should be 4-d tensors [N, C, H, W], got shape {pred.shape}."
        )
    if win_size % 2 != 1:
        raise ValueError("win_size must be odd.")

    pred = pred.astype("float32")
    label = label.astype("float32")

    channels = pred.shape[1]
    window = _gaussian_window_2d(
        channels=channels,
        win_size=win_size,
        win_sigma=win_sigma,
        dtype=pred.dtype,
    )

    mu1 = paddle.nn.functional.conv2d(pred, window, stride=1, padding=0, groups=channels)
    mu2 = paddle.nn.functional.conv2d(label, window, stride=1, padding=0, groups=channels)

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = paddle.nn.functional.conv2d(
        pred * pred, window, stride=1, padding=0, groups=channels
    ) - mu1_sq
    sigma2_sq = paddle.nn.functional.conv2d(
        label * label, window, stride=1, padding=0, groups=channels
    ) - mu2_sq
    sigma12 = paddle.nn.functional.conv2d(
        pred * label, window, stride=1, padding=0, groups=channels
    ) - mu1_mu2

    c1 = (k1 * data_range) ** 2
    c2 = (k2 * data_range) ** 2
    c1 = paddle.to_tensor(c1, dtype=pred.dtype)
    c2 = paddle.to_tensor(c2, dtype=pred.dtype)

    cs_map = (2.0 * sigma12 + c2) / (sigma1_sq + sigma2_sq + c2)
    ssim_map = ((2.0 * mu1_mu2 + c1) / (mu1_sq + mu2_sq + c1)) * cs_map

    if nonnegative_ssim:
        ssim_map = paddle.nn.functional.relu(ssim_map)

    return paddle.mean(ssim_map)


class PSNRMetric:
    def __init__(self, data_range: float = 255.0, eps: float = 1e-12):
        self.data_range = data_range
        self.eps = eps

    def __call__(self, pred: paddle.Tensor, label: paddle.Tensor):
        return calc_psnr(
            pred=pred,
            label=label,
            data_range=self.data_range,
            eps=self.eps,
        )


class SSIMMetric:
    def __init__(
        self,
        data_range: float = 255.0,
        win_size: int = 11,
        win_sigma: float = 1.5,
        k1: float = 0.01,
        k2: float = 0.03,
        nonnegative_ssim: bool = False,
    ):
        self.data_range = data_range
        self.win_size = win_size
        self.win_sigma = win_sigma
        self.k1 = k1
        self.k2 = k2
        self.nonnegative_ssim = nonnegative_ssim

    def __call__(self, pred: paddle.Tensor, label: paddle.Tensor):
        return calc_ssim(
            pred=pred,
            label=label,
            data_range=self.data_range,
            win_size=self.win_size,
            win_sigma=self.win_sigma,
            k1=self.k1,
            k2=self.k2,
            nonnegative_ssim=self.nonnegative_ssim,
        )


class SFINStreamingAdapter(StreamingMetricBase):
    """Streaming PSNR/SSIM adapter for SFIN image restoration."""

    def __init__(
        self,
        target_name: str,
        pred_name: Optional[str] = None,
        psnr_name: str = "psnr",
        ssim_name: str = "ssim",
        data_range: float = 255.0,
        eps: float = 1e-12,
        win_size: int = 11,
        win_sigma: float = 1.5,
        k1: float = 0.01,
        k2: float = 0.03,
        nonnegative_ssim: bool = False,
        stages: Optional[Sequence[str]] = None,
    ):
        self.target_name = target_name
        self.pred_name = pred_name or target_name
        self.psnr_name = psnr_name
        self.ssim_name = ssim_name
        self.data_range = data_range
        self.eps = eps
        self.win_size = win_size
        self.win_sigma = win_sigma
        self.k1 = k1
        self.k2 = k2
        self.nonnegative_ssim = nonnegative_ssim
        self.stages: Set[str] = set(stages or ("train", "eval"))
        self.reset()

    def reset(self):
        self._sse = 0.0
        self._numel = 0.0
        self._ssim_sum = 0.0
        self._ssim_count = 0.0

    def update_step(self, *, result: Dict, batch: Dict, stage: str):
        if stage not in self.stages:
            return
        pred_dict = result.get("pred_dict", {})
        if self.pred_name not in pred_dict or self.target_name not in batch:
            return

        with paddle.no_grad():
            pred = pred_dict[self.pred_name]
            label = batch[self.target_name]

            pred64 = pred.astype("float64")
            label64 = label.astype("float64")
            diff = pred64 - label64
            self._sse += float(paddle.sum(diff * diff).numpy().item())
            self._numel += float(pred.numel())

            ssim = calc_ssim(
                pred=pred,
                label=label,
                data_range=self.data_range,
                win_size=self.win_size,
                win_sigma=self.win_sigma,
                k1=self.k1,
                k2=self.k2,
                nonnegative_ssim=self.nonnegative_ssim,
            )
            batch_size = float(pred.shape[0])
            self._ssim_sum += float(ssim.numpy().item()) * batch_size
            self._ssim_count += batch_size

    def compute_epoch(self, *, stage: str) -> Dict[str, float]:
        if stage not in self.stages or self._numel <= 0:
            return {}

        values = paddle.to_tensor(
            [self._sse, self._numel, self._ssim_sum, self._ssim_count],
            dtype="float64",
        )
        values = _all_reduce_sum_(values)
        sse, numel, ssim_sum, ssim_count = [float(v) for v in values.numpy()]

        mse = max(sse / max(numel, 1.0), self.eps)
        psnr = 10.0 * paddle.log10(
            paddle.to_tensor((self.data_range**2) / mse, dtype="float64")
        )
        ssim = ssim_sum / max(ssim_count, 1.0)
        return {
            self.psnr_name: float(psnr.numpy().item()),
            self.ssim_name: float(ssim),
        }

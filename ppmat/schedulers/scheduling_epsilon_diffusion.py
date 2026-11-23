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

import copy

import numpy as np
import paddle
from tqdm.auto import tqdm

_MASK_NORM_EPS = 1e-6


class EpsilonDiffusionScheduler:
    """Gaussian epsilon diffusion for any dense latent tensor.

    A model-agnostic wrapper around an epsilon-prediction DDPM scheduler. It
    adds timestep respacing (``ddim50`` style), optional learned-sigma epsilon
    splitting and DDPM/DDIM sampling loops, so it serves any latent diffusion
    model that predicts noise with an optional classifier-free-guidance call.
    """

    def __init__(self, scheduler_cfg, timestep_respacing="", learn_sigma=True):
        from ppmat.schedulers import build_scheduler

        scheduler = build_scheduler(scheduler_cfg)
        if scheduler.prediction_type != "epsilon":
            raise ValueError("EpsilonDiffusionScheduler requires epsilon prediction.")
        if scheduler.variance_type not in ("fixed_small", "fixed_large"):
            raise ValueError("EpsilonDiffusionScheduler supports fixed variance only.")
        if scheduler.thresholding:
            raise ValueError("Latent diffusion does not support dynamic thresholding.")
        self.original_num_steps = scheduler.num_train_timesteps
        if timestep_respacing is None or timestep_respacing == "":
            timestep_respacing = [self.original_num_steps]
        use_timesteps = space_timesteps(self.original_num_steps, timestep_respacing)
        last_alpha = 1.0
        new_betas = []
        self.timestep_map = []
        for index in range(self.original_num_steps):
            if index in use_timesteps:
                new_betas.append(
                    float(1 - scheduler.alphas_cumprod[index] / last_alpha)
                )
                last_alpha = float(scheduler.alphas_cumprod[index])
                self.timestep_map.append(index)
        self.num_timesteps = len(new_betas)
        self._map_tensor = paddle.to_tensor(self.timestep_map)
        spaced_cfg = copy.deepcopy(scheduler_cfg)
        spaced_cfg["__init_params__"].update(
            num_train_timesteps=self.num_timesteps,
            trained_betas=new_betas,
        )
        self.scheduler = build_scheduler(spaced_cfg)
        self.learn_sigma = learn_sigma

    def split_epsilon(self, model_output):
        if self.learn_sigma:
            epsilon, _ = paddle.split(model_output, 2, axis=1)
            return epsilon
        return model_output

    @property
    def alphas_cumprod_prev(self):
        # Upstream convention: prev[0] = 1.0 so the DDIM update is the
        # identity at t=0.
        alphas_cumprod = self.scheduler.alphas_cumprod
        return paddle.concat(
            [paddle.ones_like(alphas_cumprod[:1]), alphas_cumprod[:-1]]
        )

    def training_losses(self, model, x_start, t, model_kwargs=None, noise=None):
        if noise is None:
            noise = paddle.randn_like(x_start)
        x_t = self.scheduler.add_noise(x_start, noise, t)
        mask = (model_kwargs or {}).get("mask")
        model_kwargs_no_mask = dict(model_kwargs or {})
        model_kwargs_no_mask["apply_mask"] = False
        model_output = model(x_t, self._map_tensor[t], **model_kwargs_no_mask)
        model_output = self.split_epsilon(model_output)

        if mask is not None:
            model_output = model_output * mask.unsqueeze(-1).astype(model_output.dtype)

        target = noise

        if mask is not None:
            mask_b = mask.astype(target.dtype)
            while mask_b.ndim < target.ndim:
                mask_b = mask_b.unsqueeze(-1)
            mask_b = paddle.broadcast_to(mask_b, list(target.shape))
            squared = (target - model_output) ** 2 * mask_b
            mse = squared.sum(axis=list(range(1, target.ndim))) / (
                mask_b.sum(axis=list(range(1, target.ndim))).clip(min=_MASK_NORM_EPS)
            )
        else:
            mse = paddle.mean(
                (target - model_output) ** 2, axis=list(range(1, target.ndim))
            )

        return {"mse": mse.mean(), "loss": mse.mean()}

    def p_sample(self, model, x, t, clip_denoised=True, model_kwargs=None, eta=0.0):
        model_output = model(x, self._map_tensor[t], **(model_kwargs or {}))
        model_output = self.split_epsilon(model_output)
        self.scheduler.clip_sample = clip_denoised
        out = self.scheduler.step(model_output, int(t[0].item()), x)
        return {"sample": out.prev_sample, "pred_xstart": out.pred_original_sample}

    def _loop(
        self,
        model,
        shape,
        noise,
        model_kwargs,
        sampler_fn,
        progress,
        clip_denoised,
        eta=0.0,
    ):
        img = noise if noise is not None else paddle.randn(shape)
        indices = list(range(self.num_timesteps))[::-1]
        if progress:
            indices = tqdm(indices)
        for i in indices:
            t = paddle.to_tensor([i] * shape[0], dtype="int64")
            with paddle.no_grad():
                img = sampler_fn(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    model_kwargs=model_kwargs,
                    eta=eta,
                )["sample"]
        return img

    def p_sample_loop(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        model_kwargs=None,
        progress=False,
    ):
        return self._loop(
            model,
            shape,
            noise,
            model_kwargs,
            self.p_sample,
            progress,
            clip_denoised,
        )

    def _extract(self, arr, t, shape):
        res = arr[t]
        while res.ndim < len(shape):
            res = res.unsqueeze(-1)
        return paddle.broadcast_to(res, shape)

    def ddim_sample(self, model, x, t, clip_denoised=True, model_kwargs=None, eta=0.0):
        model_output = model(x, self._map_tensor[t], **(model_kwargs or {}))
        model_output = self.split_epsilon(model_output)
        sched = self.scheduler
        alpha_bar = self._extract(sched.alphas_cumprod, t, x.shape)
        alpha_bar_prev = self._extract(self.alphas_cumprod_prev, t, x.shape)

        pred_x0 = (x - (1 - alpha_bar).sqrt() * model_output) / alpha_bar.sqrt()
        if clip_denoised:
            pred_x0 = pred_x0.clip(-sched.clip_sample_range, sched.clip_sample_range)

        eps = (x - alpha_bar.sqrt() * pred_x0) / (1 - alpha_bar).sqrt()

        sigma = (
            eta
            * ((1 - alpha_bar_prev) / (1 - alpha_bar)).sqrt()
            * (1 - alpha_bar / alpha_bar_prev).sqrt()
        )
        noise = paddle.randn_like(x) * (t != 0).astype("float32").reshape(
            [-1] + [1] * (x.ndim - 1)
        )
        mean = (
            pred_x0 * alpha_bar_prev.sqrt()
            + (1 - alpha_bar_prev - sigma**2).clip(min=0).sqrt() * eps
        )
        return {
            "sample": mean + sigma * noise,
            "pred_xstart": pred_x0,
            "mean": mean,
            "std": sigma,
        }

    def ddim_sample_loop(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        model_kwargs=None,
        progress=False,
        eta=0.0,
    ):
        return self._loop(
            model,
            shape,
            noise,
            model_kwargs,
            self.ddim_sample,
            progress,
            clip_denoised,
            eta=eta,
        )


def space_timesteps(num_timesteps, section_counts):
    if isinstance(section_counts, str) and section_counts.startswith("ddim"):
        desired = int(section_counts[len("ddim") :])
        for i in range(1, num_timesteps):
            if len(range(0, num_timesteps, i)) == desired:
                return set(range(0, num_timesteps, i))
        # Fall back to evenly spaced, unique timesteps for any desired count.
        steps = np.unique(
            np.linspace(0, num_timesteps - 1, desired).round().astype("int64")
        )
        return {int(step) for step in steps}
    counts = (
        [int(x) for x in section_counts.split(",")]
        if isinstance(section_counts, str)
        else section_counts
    )
    size_per = num_timesteps // len(counts)
    extra = num_timesteps % len(counts)
    all_steps = []
    start = 0
    for i, cnt in enumerate(counts):
        size = size_per + (1 if i < extra else 0)
        if size < cnt:
            raise ValueError(f"cannot divide {size} steps into {cnt}")
        stride = (size - 1) / (cnt - 1) if cnt > 1 else 1
        all_steps += [start + round(j * stride) for j in range(cnt)]
        start += size
    return set(all_steps)

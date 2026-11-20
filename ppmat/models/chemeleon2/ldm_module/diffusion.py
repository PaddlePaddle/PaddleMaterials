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

from ppmat.schedulers import build_scheduler


class GaussianDiffusion:
    def __init__(self, learn_sigma=False, **kwargs):
        sched_kwargs = {
            k: v
            for k, v in kwargs.items()
            if k
            in (
                "num_train_timesteps",
                "beta_start",
                "beta_end",
                "beta_schedule",
                "variance_type",
                "prediction_type",
                "clip_sample",
            )
        }
        sched_kwargs.setdefault("variance_type", "fixed_small")
        sched_kwargs.setdefault("prediction_type", "epsilon")
        sched_kwargs.setdefault("beta_schedule", "linear")
        self.scheduler = build_scheduler(
            {
                "__class_name__": "DDPMScheduler",
                "__init_params__": sched_kwargs,
            }
        )
        self.num_timesteps = self.scheduler.num_train_timesteps
        # Whether the denoiser emits extra variance channels that must be
        # stripped before the epsilon output reaches the scheduler.
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

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = paddle.randn_like(x_start)
        return self.scheduler.add_noise(x_start, noise, t)

    def training_losses(self, model, x_start, t, model_kwargs=None, noise=None):
        if noise is None:
            noise = paddle.randn_like(x_start)
        x_t = self.q_sample(x_start, t, noise)
        mask = (model_kwargs or {}).get("mask")
        model_kwargs_no_mask = dict(model_kwargs or {})
        model_kwargs_no_mask["apply_mask"] = False
        model_output = model(x_t, t, **model_kwargs_no_mask)
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
                mask_b.sum(axis=list(range(1, target.ndim))).clip(min=1e-6)
            )
        else:
            mse = paddle.mean(
                (target - model_output) ** 2, axis=list(range(1, target.ndim))
            )

        return {"mse": mse.mean(), "loss": mse.mean()}

    def p_sample(self, model, x, t, clip_denoised=True, model_kwargs=None, eta=0.0):
        model_output = model(x, t, **(model_kwargs or {}))
        model_output = self.split_epsilon(model_output)
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
            try:
                from tqdm.auto import tqdm

                indices = tqdm(indices)
            except ImportError:
                pass
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
            lambda m, x, t, **kw: self.p_sample(m, x, t, **kw),
            progress,
            clip_denoised,
        )

    def _extract(self, arr, t, shape):
        res = arr[t]
        while res.ndim < len(shape):
            res = res.unsqueeze(-1)
        return paddle.broadcast_to(res, shape)

    def ddim_sample(self, model, x, t, clip_denoised=True, model_kwargs=None, eta=0.0):
        model_output = model(x, t, **(model_kwargs or {}))
        model_output = self.split_epsilon(model_output)
        sched = self.scheduler
        alpha_bar = self._extract(sched.alphas_cumprod, t, x.shape)
        alpha_bar_prev = self._extract(self.alphas_cumprod_prev, t, x.shape)

        pred_x0 = (x - (1 - alpha_bar).sqrt() * model_output) / alpha_bar.sqrt()
        if clip_denoised:
            pred_x0 = pred_x0.clip(-1, 1)

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
            lambda m, x, t, **kw: self.ddim_sample(m, x, t, **kw),
            progress,
            clip_denoised,
            eta=eta,
        )


class SpacedDiffusion(GaussianDiffusion):
    def __init__(self, use_timesteps, **kwargs):
        super().__init__(**kwargs)
        use_timesteps = sorted(set(use_timesteps))
        self.use_timesteps = use_timesteps
        self.original_num_steps = self.num_timesteps

        sched = self.scheduler
        last_alpha = 1.0
        new_betas = []
        self.timestep_map = []
        for i in range(sched.num_train_timesteps):
            if i in use_timesteps:
                new_betas.append(float(1 - sched.alphas_cumprod[i] / last_alpha))
                last_alpha = float(sched.alphas_cumprod[i])
                self.timestep_map.append(i)
        self.num_timesteps = len(new_betas)
        self._map_tensor = paddle.to_tensor(self.timestep_map)

        # Rebuild the scheduler from the spaced betas so that every index this
        # class hands to the scheduler (t in step/add_noise and the
        # alphas_cumprod lookups in ddim_sample) refers to the spaced chain.
        # Keeping the original 1000-step alphas_cumprod would make sampling
        # read alphas near 1.0 at the end of the loop instead of near 0.
        self.scheduler = build_scheduler(
            {
                "__class_name__": "DDPMScheduler",
                "__init_params__": {
                    "trained_betas": new_betas,
                    "variance_type": sched.variance_type,
                    "prediction_type": sched.prediction_type,
                    "clip_sample": sched.clip_sample,
                },
            }
        )

    def _wrap_model(self, model):
        if getattr(model, "_spaced_wrapped", False):
            return model
        map_t = self._map_tensor

        def wrapped(x, ts, **kw):
            return model(x, map_t[ts], **kw)

        wrapped._spaced_wrapped = True
        return wrapped

    def p_sample(self, model, *args, **kwargs):
        return super().p_sample(self._wrap_model(model), *args, **kwargs)

    def ddim_sample(self, model, *args, **kwargs):
        return super().ddim_sample(self._wrap_model(model), *args, **kwargs)

    def training_losses(self, model, *args, **kwargs):
        return super().training_losses(self._wrap_model(model), *args, **kwargs)

    def p_sample_loop(self, model, *args, **kwargs):
        return super().p_sample_loop(self._wrap_model(model), *args, **kwargs)

    def ddim_sample_loop(self, model, *args, **kwargs):
        return super().ddim_sample_loop(self._wrap_model(model), *args, **kwargs)


def space_timesteps(num_timesteps, section_counts):
    if isinstance(section_counts, str) and section_counts.startswith("ddim"):
        desired = int(section_counts[len("ddim") :])
        for i in range(1, num_timesteps):
            if len(range(0, num_timesteps, i)) == desired:
                return set(range(0, num_timesteps, i))
        raise ValueError(f"cannot create {num_timesteps} steps with integer stride")
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


def create_diffusion(
    timestep_respacing,
    noise_schedule="linear",
    sigma_small=False,
    learn_sigma=True,
    diffusion_steps=1000,
):
    beta_schedule = (
        noise_schedule
        if noise_schedule in ("linear", "squaredcos_cap_v2")
        else "linear"
    )

    var_type = "fixed_small" if (learn_sigma or sigma_small) else "fixed_large"

    if timestep_respacing is None or timestep_respacing == "":
        timestep_respacing = [diffusion_steps]

    # The factory always respaces (matching the upstream create_diffusion),
    # so every returned diffusion reads scheduler values from the spaced chain.
    use_timesteps = space_timesteps(diffusion_steps, timestep_respacing)
    return SpacedDiffusion(
        use_timesteps=use_timesteps,
        num_train_timesteps=diffusion_steps,
        beta_schedule=beta_schedule,
        variance_type=var_type,
        prediction_type="epsilon",
        learn_sigma=learn_sigma,
    )

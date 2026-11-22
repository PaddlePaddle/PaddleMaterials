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

"""Chemeleon2 reinforcement-learning stage (GRPO over the LDM denoiser).

The module owns a frozen VAE and a trainable DiT denoiser, rolls out a full
DDIM sampling trajectory under ``no_grad``, scores the decoded structures with a
composable reward function, then recomputes the per-step Gaussian log-probs with
gradients enabled to form the clipped GRPO surrogate. It exposes the framework
training contract (``forward`` -> ``{"loss_dict": {"loss": ...}}``) and delegates
``sample``/``predict`` to the wrapped LDM, so it is a drop-in model for
``BaseTrainer`` and ``StructureSampler``.
"""

import copy
import math
from collections import defaultdict

import numpy as np
import paddle
import paddle.nn as nn

from ppmat.models.chemeleon2.chemeleon2 import Chemeleon2LDMModule
from ppmat.models.chemeleon2.common import to_dense_batch
from ppmat.models.chemeleon2.common.utils import build_structure_array
from ppmat.models.chemeleon2.rl_module.components import standardize
from ppmat.models.chemeleon2.rl_module.reward import build_reward
from ppmat.schedulers import build_scheduler

DEFAULT_RL_CONFIGS = {
    "clip_ratio": 0.001,
    "kl_weight": 1.0,
    "entropy_weight": 1e-5,
    "num_group_samples": 1,
    "group_reward_norm": False,
}

DEFAULT_SAMPLING_CONFIGS = {
    "sampler": "ddim",
    "sampling_steps": 50,
    "eta": 1.0,
    "cfg_scale": 2.0,
    "progress": False,
}

_LOG_STD_MIN = 1e-8


def _broadcast_mask(mask, x):
    while mask.ndim < x.ndim:
        mask = mask.unsqueeze(-1)
    return paddle.expand_as(mask, x)


def _calculate_log_prob(x, mean, std, mask, reduce="mean"):
    """Per-structure Gaussian log-probability of ``x`` under ``N(mean, std^2)``."""

    std = paddle.clip(std, min=_LOG_STD_MIN)
    log_prob = -0.5 * (paddle.log(2 * math.pi * std**2) + ((x - mean) / std) ** 2)
    reduce_dims = list(range(1, x.ndim))
    if mask is not None:
        mask_f = _broadcast_mask(mask.astype(x.dtype), x)
        log_prob = log_prob * mask_f
    if reduce == "sum":
        return log_prob.sum(axis=reduce_dims)
    if reduce == "none":
        return log_prob
    summed = log_prob.sum(axis=reduce_dims)
    if mask is not None:
        counts = _broadcast_mask(mask.astype(x.dtype), x).sum(axis=reduce_dims)
        return summed / paddle.clip(counts, min=1e-6)
    return summed / math.prod(log_prob.shape[1:])


def _repeat_condition(value, times):
    if isinstance(value, paddle.Tensor):
        return paddle.repeat_interleave(value, times, axis=0)
    if isinstance(value, np.ndarray):
        return np.repeat(value, times, axis=0)
    if isinstance(value, (list, tuple)):
        return [item for item in value for _ in range(times)]
    return [value] * times


def _safe_structure(data):
    """Build a pymatgen structure with a well-conditioned lattice.

    A frozen but imperfect VAE can decode a near-singular lattice, which makes
    pymatgen's Niggli/LLL reduction in ``StructureMatcher`` either crash or run
    for an extremely long time. Degenerate cells therefore fall back to a cubic
    cell so the reward batch keeps a 1:1 alignment with the rollout batch.
    """

    from pymatgen.core import Element
    from pymatgen.core import Lattice
    from pymatgen.core import Structure

    species = [
        Element.from_Z(min(max(int(number), 1), 118)) for number in data["atom_types"]
    ]
    frac_coords = np.clip(np.asarray(data["frac_coords"], dtype="float64"), 0.0, 1.0)
    lattice = np.asarray(data["lattice"], dtype="float64")

    if np.all(np.isfinite(lattice)) and abs(np.linalg.det(lattice)) > 1e-2:
        try:
            return Structure(
                Lattice(lattice), species, frac_coords, coords_are_cartesian=False
            )
        except Exception:
            pass

    edge = max(len(species), 1) ** (1.0 / 3.0)
    return Structure(
        Lattice.cubic(edge), species, frac_coords, coords_are_cartesian=False
    )


class Chemeleon2RLModule(nn.Layer):
    """GRPO reinforcement-learning wrapper around a Chemeleon2 LDM."""

    def __init__(
        self,
        ldm_cfg=None,
        ldm_pretrained_model_name=None,
        ldm_ckpt_path=None,
        rl_configs=None,
        reward_fn_cfg=None,
        sampling_configs=None,
        denoiser_cfg=None,
        noise_scheduler_cfg=None,
        vae_cfg=None,
        property_embeddings_adapt_cfg=None,
        condition_names=None,
        vae_pretrained_model_name=None,
        vae_ckpt_path=None,
        **_unused_compat,
    ):
        super().__init__()

        # Build the underlying LDM: packaged weights, explicit cfg, or flat args.
        if ldm_pretrained_model_name is not None:
            if ldm_cfg is not None or ldm_ckpt_path is not None:
                raise ValueError(
                    "Choose either a registered LDM or a local LDM config/checkpoint."
                )
            from ppmat.models import build_model_from_name

            self.ldm, _ = build_model_from_name(ldm_pretrained_model_name)
        elif ldm_cfg is not None:
            from ppmat.models import build_model

            self.ldm = build_model(ldm_cfg)
        else:
            self.ldm = Chemeleon2LDMModule(
                denoiser_cfg=denoiser_cfg,
                noise_scheduler_cfg=noise_scheduler_cfg,
                vae_cfg=vae_cfg,
                property_embeddings_adapt_cfg=property_embeddings_adapt_cfg,
                condition_names=condition_names,
                vae_pretrained_model_name=vae_pretrained_model_name,
                vae_ckpt_path=vae_ckpt_path,
            )

        if ldm_ckpt_path is not None:
            from ppmat.utils.save_load import load_pretrain

            load_pretrain(self.ldm, ldm_ckpt_path)

        # Only the denoiser is optimized; the VAE and the condition embedding
        # are frozen exactly as in the upstream RL stage.
        for param in self.ldm.vae.parameters():
            param.stop_gradient = True
        self.ldm.vae.eval()
        if self.ldm.use_cfg:
            for param in self.ldm.condition_module.parameters():
                param.stop_gradient = True

        self.rl_configs = {**DEFAULT_RL_CONFIGS, **(rl_configs or {})}
        self.clip_ratio = self.rl_configs["clip_ratio"]
        self.kl_weight = self.rl_configs["kl_weight"]
        self.entropy_weight = self.rl_configs["entropy_weight"]
        self.num_group_samples = int(self.rl_configs["num_group_samples"])
        self.group_reward_norm = self.rl_configs["group_reward_norm"]
        if self.num_group_samples < 1:
            raise ValueError("num_group_samples must be >= 1.")

        self.sampling_configs = {**DEFAULT_SAMPLING_CONFIGS, **(sampling_configs or {})}
        if self.sampling_configs["sampler"] != "ddim":
            raise ValueError("RL fine-tuning supports the ddim sampler only.")
        if float(self.sampling_configs["eta"]) <= 0:
            raise ValueError(
                "RL fine-tuning requires sampling_configs.eta > 0; eta=0 makes "
                "every step deterministic and yields a zero policy gradient."
            )
        self.sampling_diffusion = self._build_sampling_diffusion()

        self.reward_fn = build_reward(reward_fn_cfg)
        if self.reward_fn is None:
            raise ValueError("reward_fn_cfg must be provided for RL training.")

    def _build_sampling_diffusion(self):
        steps = int(self.sampling_configs["sampling_steps"])
        sampling_cfg = copy.deepcopy(self.ldm.noise_scheduler_cfg)
        sampling_cfg["__init_params__"]["timestep_respacing"] = "ddim" + str(steps)
        return build_scheduler(sampling_cfg)

    @property
    def condition_names(self):
        return self.ldm.condition_names

    @property
    def supports_num_atoms_sampling(self):
        return True

    def before_train(self, trainer):
        self.ldm.vae.eval()
        if self.ldm.use_cfg:
            self.ldm.condition_module.eval()

    def forward(self, batch):
        self.ldm.vae.eval()
        crystal_batch = build_structure_array(batch["structure_array"])
        if self.ldm.use_cfg:
            crystal_batch["y"] = {
                name: batch[name] for name in self.condition_names if name in batch
            }
        loss_dict = self.rl_loss(crystal_batch, training=self.training)
        return {"loss_dict": loss_dict, "label_dict": batch["structure_array"]}

    def sample(self, batch, **kwargs):
        return self.ldm.sample(batch, **kwargs)

    def predict(self, data, **kwargs):
        return self.ldm.predict(data, **kwargs)

    def _denoise(self, x, t, mask=None, y=None, cfg_scale=None):
        if cfg_scale is not None:
            return self.ldm.denoiser.forward_with_cfg(x, t, mask, y, cfg_scale)
        return self.ldm.denoiser(x, t, mask=mask, y=y, apply_mask=True)

    def _prepare_rollout_batch(self, crystal_batch):
        group = self.num_group_samples
        num_atoms = paddle.repeat_interleave(crystal_batch["num_atoms"], group)
        batch = build_structure_array({"num_atoms": num_atoms})
        z = paddle.randn(
            [int(batch["num_atoms"].sum().item()), self.ldm.vae.latent_dim]
        )
        z, mask = to_dense_batch(z, batch["batch"])

        y = None
        if self.ldm.use_cfg:
            cond = {
                name: _repeat_condition(crystal_batch["y"][name], group)
                for name in self.condition_names
            }
            z = paddle.concat([z, z], axis=0)
            mask = paddle.concat([mask, mask], axis=0)
            y = self.ldm.condition_module(cond, training=False)
        return batch, z, mask, y

    def rollout(self, batch, z, mask, y):
        diffusion = self.sampling_diffusion
        eta = self.sampling_configs["eta"]
        cfg_scale = self.sampling_configs["cfg_scale"]
        model_kwargs = {"mask": mask, "y": y}
        if self.ldm.use_cfg:
            model_kwargs["cfg_scale"] = cfg_scale

        indices = list(range(diffusion.num_timesteps))[::-1]
        zs, means, stds = [z], [], []
        img = z
        with paddle.no_grad():
            for index in indices:
                t = paddle.to_tensor([index] * img.shape[0], dtype="int64")
                out = diffusion.ddim_sample(
                    self._denoise,
                    img,
                    t,
                    clip_denoised=False,
                    model_kwargs=model_kwargs,
                    eta=eta,
                )
                img = out["sample"]
                zs.append(img)
                means.append(out["mean"])
                stds.append(out["std"])

        if self.ldm.use_cfg:
            zs = [paddle.chunk(value, 2, axis=0)[0] for value in zs]
            means = [paddle.chunk(value, 2, axis=0)[0] for value in means]
            stds = [paddle.chunk(value, 2, axis=0)[0] for value in stds]
            mask = paddle.chunk(mask, 2, axis=0)[0]
            y = None

        log_probs = []
        for step in range(len(means)):
            log_probs.append(
                _calculate_log_prob(zs[step + 1], means[step], stds[step], mask)
            )
        log_probs = paddle.stack(log_probs, axis=0)
        return {
            "zs": zs,
            "means": means,
            "stds": stds,
            "log_probs": log_probs.detach(),
            "mask": mask,
            "final": zs[-1],
            "batch": batch,
            "y": y,
        }

    def _decode(self, final_latents, mask, batch):
        with paddle.no_grad():
            diffusion_out = final_latents * self.ldm.latent_std
            encoded_batch = {
                "x": diffusion_out[mask],
                "num_atoms": batch["num_atoms"],
                "batch": batch["batch"],
                "token_idx": batch["token_idx"],
            }
            decoder_out = self.ldm.vae.decode(encoded_batch)
            batch_rec = self.ldm.vae.reconstruct(decoder_out, batch)

        array_dicts = []
        offset = 0
        for index, num_atoms in enumerate(batch_rec["num_atoms"].tolist()):
            array_dicts.append(
                {
                    "num_atoms": num_atoms,
                    "atom_types": batch_rec["atom_types"][offset : offset + num_atoms]
                    .numpy()
                    .tolist(),
                    "frac_coords": batch_rec["frac_coords"][offset : offset + num_atoms]
                    .numpy()
                    .tolist(),
                    "lattice": batch_rec["lattice"][index].numpy().tolist(),
                }
            )
            offset += num_atoms
        structures = [_safe_structure(data) for data in array_dicts]
        return array_dicts, structures

    def _advantages(self, rewards):
        if self.group_reward_norm:
            group = self.num_group_samples
            grouped = [
                rewards[start : start + group]
                for start in range(0, rewards.shape[0], group)
            ]
            return paddle.concat([standardize(part) for part in grouped], axis=0)
        return self.reward_fn.normalize(rewards)

    def rl_loss(self, crystal_batch, training=True):
        batch, z, mask, y = self._prepare_rollout_batch(crystal_batch)
        trajectory = self.rollout(batch, z, mask, y)
        _, gen_structures = self._decode(trajectory["final"], trajectory["mask"], batch)

        with paddle.no_grad():
            rewards = self.reward_fn(gen_structures, vae=self.ldm.vae)
            advantages = self._advantages(rewards)

        loss_dict = self._policy_loss(trajectory, advantages, y)
        loss_dict["reward"] = rewards.mean().detach()
        loss_dict["advantages"] = advantages.mean().detach()
        return loss_dict

    def _policy_loss(self, trajectory, advantages, y):
        diffusion = self.sampling_diffusion
        eta = self.sampling_configs["eta"]
        cfg_scale = self.sampling_configs["cfg_scale"]
        mask = trajectory["mask"]
        zs = trajectory["zs"]
        old_log_probs = trajectory["log_probs"]
        indices = list(range(diffusion.num_timesteps))[::-1]

        total_loss = paddle.zeros([1], dtype="float32")
        stats = defaultdict(float)
        counts = 0
        for step, index in enumerate(indices):
            z = zs[step]
            step_mask = mask
            if self.ldm.use_cfg:
                z = paddle.concat([z, z], axis=0)
                step_mask = paddle.concat([mask, mask], axis=0)

            t = paddle.to_tensor([index] * z.shape[0], dtype="int64")
            model_kwargs = {"mask": step_mask, "y": y}
            if self.ldm.use_cfg:
                model_kwargs["cfg_scale"] = cfg_scale
            out = diffusion.ddim_sample(
                self._denoise,
                z,
                t,
                clip_denoised=False,
                model_kwargs=model_kwargs,
                eta=eta,
            )
            mean, std = out["mean"], out["std"]
            if self.ldm.use_cfg:
                mean = paddle.chunk(mean, 2, axis=0)[0]
                std = paddle.chunk(std, 2, axis=0)[0]
                step_mask = paddle.chunk(step_mask, 2, axis=0)[0]

            if index == 0 and float(std.abs().max()) == 0.0:
                continue

            current = _calculate_log_prob(zs[step + 1], mean, std, step_mask)
            old = old_log_probs[step]
            ratio = paddle.exp(current - old)
            clipped = paddle.clip(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio)
            surrogate = paddle.minimum(ratio * advantages, clipped * advantages)

            log_ratio = old - current
            kl = paddle.exp(log_ratio) - 1.0 - log_ratio
            entropy = -current
            policy = -surrogate + self.kl_weight * kl - self.entropy_weight * entropy
            total_loss = total_loss + policy.mean()
            stats["kl"] += float(kl.mean())
            stats["entropy"] += float(entropy.mean())
            stats["surrogate"] += float(surrogate.mean())
            stats["ratio"] += float(ratio.mean())
            counts += 1

        divisor = max(counts, 1)
        return {
            "loss": total_loss / divisor,
            "kl": paddle.to_tensor(stats["kl"] / divisor),
            "entropy": paddle.to_tensor(stats["entropy"] / divisor),
            "surrogate": paddle.to_tensor(stats["surrogate"] / divisor),
            "ratio": paddle.to_tensor(stats["ratio"] / divisor),
        }

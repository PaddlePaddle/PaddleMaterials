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

from functools import partial

import numpy as np
import paddle
import paddle.nn as nn

from ppmat.datasets.build_structure import BuildStructure
from ppmat.models.chemeleon2.common import to_dense_batch
from ppmat.models.chemeleon2.common.utils import build_structure_array
from ppmat.models.common.runtime import RuntimeMixin
from ppmat.models.common.runtime import runtime_boundary
from ppmat.schedulers import build_scheduler
from ppmat.utils import logger
from ppmat.utils.save_load import load_pretrain

# MP-20 caps the structures at 20 atoms per unit cell.
DEFAULT_NUM_ATOMS = 20

# Sampling defaults shared by sample() and predict().
DEFAULT_SAMPLER = "ddim"
DEFAULT_INFERENCE_STEPS = 50
DEFAULT_ETA = 1.0
DEFAULT_CFG_SCALE = 2.0


class Chemeleon2LDMModule(RuntimeMixin, nn.Layer):
    def __init__(
        self,
        denoiser_cfg=None,
        noise_scheduler_cfg=None,
        vae_cfg=None,
        property_embeddings_adapt_cfg=None,
        condition_names=None,
        vae_pretrained_model_name=None,
        vae_ckpt_path=None,
        ldm_ckpt_path=None,
        execution_backend="eager",
        runtime_options=None,
    ):
        super().__init__()
        self._init_runtime(execution_backend, runtime_options)
        from ppmat.models import MODEL_REGISTRY
        from ppmat.models import build_model

        self.denoiser = build_model(denoiser_cfg)
        self.vae = build_model(vae_cfg)
        embedding_cfg = property_embeddings_adapt_cfg
        self.use_cfg = embedding_cfg is not None
        if self.use_cfg:
            self.condition_module = build_model(embedding_cfg)
            if self.denoiser.y_embedder is None:
                raise ValueError(
                    "Conditional generation requires denoiser condition_dim."
                )
            if (
                self.denoiser.y_embedder.weight.shape[0]
                != self.condition_module.hidden_dim
            ):
                raise ValueError(
                    "Condition embedding and denoiser dimensions must match."
                )
            self.condition_names = self.condition_module.target_condition
            if condition_names is not None and condition_names != self.condition_names:
                raise ValueError("condition_names must match the embedding fields.")
        else:
            self.condition_names = []
            if condition_names:
                raise ValueError("condition_names requires property embeddings.")
            if self.denoiser.y_embedder is not None:
                raise ValueError("denoiser condition_dim requires property embeddings.")

        if noise_scheduler_cfg is None:
            raise ValueError("noise_scheduler_cfg is required.")
        self.noise_scheduler_cfg = noise_scheduler_cfg
        self.diffusion = build_scheduler(noise_scheduler_cfg)
        if self.diffusion.learn_sigma != self.denoiser.learn_sigma:
            raise ValueError("Scheduler and denoiser learn_sigma must match.")
        if self.vae.latent_dim != self.denoiser.input_dim:
            raise ValueError("VAE latent_dim must match the denoiser input_dim.")
        self.register_buffer("latent_std", paddle.to_tensor(1.0))
        if vae_pretrained_model_name is not None:
            if vae_ckpt_path is not None:
                raise ValueError("Choose a registered VAE or a local VAE checkpoint.")
            load_pretrain(self.vae, MODEL_REGISTRY[vae_pretrained_model_name])
        if vae_ckpt_path is not None:
            vae_checkpoint = nn.Layer()
            vae_checkpoint.add_sublayer("core", self.vae)
            load_pretrain(vae_checkpoint, vae_ckpt_path)
        if ldm_ckpt_path is not None:
            load_pretrain(self, ldm_ckpt_path)
        for param in self.vae.parameters():
            param.stop_gradient = True
        self.vae.eval()

    def forward(self, batch):
        """Compute diffusion loss; generation metrics consume sample() results."""
        crystal_batch = build_structure_array(batch["structure_array"])
        if self.use_cfg:
            # Condition properties arrive as top-level batch keys.
            crystal_batch["y"] = {
                name: batch[name] for name in self.condition_names if name in batch
            }
        loss_dict = self.calculate_loss(crystal_batch, training=self.training)
        return {
            "loss_dict": loss_dict,
            "label_dict": batch["structure_array"],
        }

    def before_train(self, trainer):
        self.vae.eval()

    @runtime_boundary("denoise_step")
    def _runtime_denoise(
        self, x, t, mask=None, y=None, apply_mask=True, cfg_scale=None
    ):
        if cfg_scale is not None:
            return self.denoiser.forward_with_cfg(x, t, mask, y, cfg_scale)
        return self.denoiser(x, t, mask=mask, y=y, apply_mask=apply_mask)

    def calculate_loss(self, batch, training=True):
        self.vae.eval()

        with paddle.no_grad():
            encoded = self.vae.encode(batch)
            x = encoded["posterior"].sample() / self.latent_std
            x, mask = to_dense_batch(x, encoded["batch"])

        t = paddle.randint(
            0, self.diffusion.num_timesteps, shape=[x.shape[0]], dtype="int64"
        )

        y = None
        if self.use_cfg:
            y = batch["y"]
            if not y:
                raise ValueError(
                    "Batch must carry condition fields "
                    f"{self.condition_names} at the top level when use_cfg=True"
                )
            y = self.condition_module(y, training=training)
            if y.shape[0] != x.shape[0]:
                # CFG doubling from the condition module; align the latent
                # stream with the duplicated batch.
                x = paddle.concat([x, x], axis=0)
                mask = paddle.concat([mask, mask], axis=0)
                t = paddle.concat([t, t], axis=0)

        model_kwargs = {"mask": mask, "y": y}
        loss_dict = self.diffusion.training_losses(
            model=self._runtime_denoise,
            x_start=x,
            t=t,
            model_kwargs=model_kwargs,
        )
        loss_dict["total_loss"] = loss_dict["loss"]
        return loss_dict

    def sample(
        self,
        batch,
        sampler=DEFAULT_SAMPLER,
        num_inference_steps=DEFAULT_INFERENCE_STEPS,
        eta=DEFAULT_ETA,
        cfg_scale=DEFAULT_CFG_SCALE,
        return_atoms=False,
        return_structure=False,
        progress=True,
    ):
        structure_array = batch["structure_array"]
        if "atom_types" in structure_array:
            raise ValueError(
                "Chemeleon2 does not support fixed atom types (by_chemical_formula). "
                "Use by_num_atoms or by_condition."
            )
        if sampler not in ("ddim", "ddpm"):
            raise ValueError(f"Unsupported sampler: {sampler}")
        if num_inference_steps <= 0:
            raise ValueError("num_inference_steps must be positive.")
        count = structure_array["num_atoms"].shape[0]
        entries = {}
        for name in self.condition_names:
            value = batch[name]
            if isinstance(value, (str, int, float)):
                value = [value] * count
            entries[name] = value
        if not self.use_cfg:
            passed = set(batch) - {"structure_array", "id"}
            if passed:
                raise ValueError(f"Unconditional model received conditions: {passed}")
        batch = build_structure_array(structure_array)
        self.vae.eval()

        if sampler == "ddim":
            timestep_respacing = "ddim" + str(num_inference_steps)
        else:
            timestep_respacing = str(num_inference_steps)

        sampling_cfg = {
            "__class_name__": self.noise_scheduler_cfg["__class_name__"],
            "__init_params__": {
                **self.noise_scheduler_cfg["__init_params__"],
                "timestep_respacing": timestep_respacing,
            },
        }
        sampling_diffusion = build_scheduler(sampling_cfg)

        sampler_fn = (
            partial(sampling_diffusion.ddim_sample_loop, eta=eta)
            if sampler == "ddim"
            else sampling_diffusion.p_sample_loop
        )

        if progress:
            logger.info(
                f"Using {sampler} sampler with "
                f"{sampling_diffusion.num_timesteps} timesteps."
            )

        num_nodes = batch["num_atoms"].sum().item()
        z = paddle.randn([num_nodes, self.vae.latent_dim])
        z, mask = to_dense_batch(z, batch["batch"])

        y = None
        if self.use_cfg:
            y = entries
            if not y:
                raise ValueError(
                    "Condition fields must be set on the batch for conditional "
                    f"sampling: {self.condition_names}"
                )
            z = paddle.concat([z, z], axis=0)
            mask = paddle.concat([mask, mask], axis=0)
            y = self.condition_module(y, training=False)

        model_kwargs = {
            "mask": mask,
            "y": y,
        }
        if self.use_cfg:
            model_kwargs["cfg_scale"] = cfg_scale

        diffusion_out = sampler_fn(
            model=self._runtime_denoise,
            shape=z.shape,
            noise=z,
            clip_denoised=False,
            model_kwargs=model_kwargs,
            progress=progress,
        )

        if self.use_cfg:
            diffusion_out, _ = paddle.chunk(diffusion_out, 2, axis=0)
            mask, _ = paddle.chunk(mask, 2, axis=0)

        diffusion_out = diffusion_out * self.latent_std

        encoded_batch = {
            "x": diffusion_out[mask],
            "num_atoms": batch["num_atoms"],
            "batch": batch["batch"],
            "token_idx": batch["token_idx"],
        }
        decoder_out = self.vae.decode(encoded_batch)
        batch_rec = self.vae.reconstruct(decoder_out, batch)

        result_list = []
        offset = 0
        for index, num_atoms in enumerate(batch_rec["num_atoms"].tolist()):
            result_list.append(
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
        if return_atoms or return_structure:
            structures = [
                BuildStructure.build_one(data, "array", niggli=False, canocial=False)
                for data in result_list
            ]
            if return_structure:
                return structures
            from pymatgen.io.ase import AseAtomsAdaptor

            return [AseAtomsAdaptor.get_atoms(structure) for structure in structures]
        return {"result": result_list}

    def predict(
        self,
        data,
        num_inference_steps=DEFAULT_INFERENCE_STEPS,
        sampler=DEFAULT_SAMPLER,
        progress=True,
    ):
        payload = data
        num_samples = payload.get("num_samples", 1)
        batch_size = payload.get("batch_size", num_samples)
        cfg_scale = payload.get("cfg_scale", DEFAULT_CFG_SCALE)
        eta = payload.get("eta", DEFAULT_ETA)

        if "num_atoms" in payload:
            num_atoms = payload["num_atoms"]
            if isinstance(num_atoms, (list, tuple)):
                if len(num_atoms) != num_samples:
                    raise ValueError(
                        "num_atoms list length must match num_samples "
                        f"({len(num_atoms)} != {num_samples})."
                    )
                num_atoms_list = [int(n) for n in num_atoms]
            else:
                num_atoms_list = [int(num_atoms)] * num_samples
        else:
            num_atoms_list = [DEFAULT_NUM_ATOMS] * num_samples

        names = self.condition_names or []
        known_keys = {"num_samples", "batch_size", "num_atoms", "cfg_scale", "eta"}
        unknown = [k for k in payload if k not in known_keys and k not in set(names)]
        if unknown:
            raise ValueError(
                f"predict() payload has unsupported keys {unknown}; supported "
                f"keys are {sorted(known_keys | set(names))}."
            )
        raw_conditions = {name: payload[name] for name in names if name in payload}
        missing = [name for name in names if name not in raw_conditions]
        if missing:
            raise ValueError(
                f"predict() payload is missing required condition fields "
                f"{missing}; expected {sorted(names)}."
            )

        all_results = []
        for i in range(0, num_samples, batch_size):
            cb = min(batch_size, num_samples - i)
            cur = num_atoms_list[i : i + cb]
            batch = {
                "structure_array": {"num_atoms": paddle.to_tensor(cur, dtype="int64")}
            }
            if raw_conditions:
                batch.update(
                    {
                        name: _condition_batch(raw_conditions[name], i, cb, num_samples)
                        for name in raw_conditions
                    }
                )
            with paddle.no_grad():
                result = self.sample(
                    batch,
                    sampler=sampler,
                    num_inference_steps=num_inference_steps,
                    eta=eta,
                    cfg_scale=cfg_scale,
                    progress=progress,
                )
            all_results.extend(result["result"])
        # Downstream contract of StructureSampler: {"result": [...]}
        return {"result": all_results}


def _condition_batch(value, offset, count, total):
    """Align one condition value with the current batch of ``count`` samples.

    Accepts a scalar or a sequence of length ``total`` (per-sample) or ``1``
    (shared). Scalars and length-1 payloads are replicated; ``total``-length
    payloads are sliced. This mirrors the top-level-key convention of
    ``StructureSampler.sample_by_condition``.
    """
    if isinstance(value, paddle.Tensor):
        values = value if value.ndim > 0 else [value]
    elif isinstance(value, np.ndarray):
        values = list(value) if value.ndim > 0 else [value.item()]
    elif isinstance(value, (list, tuple)):
        values = list(value)
    else:
        values = [value]
    n = len(values)
    if n == total:
        return values[offset : offset + count]
    if n == 1:
        if isinstance(values, paddle.Tensor):
            return paddle.concat([values] * count, axis=0)
        return values * count
    raise ValueError(f"condition length {n} must match num_samples ({total}) or be 1.")


class Chemeleon2(RuntimeMixin, nn.Layer):
    """Unified Chemeleon2 model: one object for training and sampling.

    Internally this composes the VAE, the DiT denoiser and (for the RL stage) the
    GRPO wrapper, but the framework only sees a single model that provides both a
    training loss and generation:

    - ``forward(batch)`` returns ``{"loss_dict": {"loss": ...}}`` for the
      configured ``stage`` (``vae`` reconstruction, ``diffusion`` epsilon loss or
      ``rl`` GRPO loss);
    - ``sample(data)`` / ``predict(data)`` run the full generation pipeline
      (latent diffusion followed by VAE decoding) and return ``{"result": [...]}``.

    Generation configurations (``diffusion``/``rl``) and the released
    ``chemeleon2_ldm`` package use this unified entry; the VAE stays a standalone
    pretraining component loaded by the diffusion stage.
    """

    def __init__(
        self,
        stage="diffusion",
        vae_cfg=None,
        denoiser_cfg=None,
        noise_scheduler_cfg=None,
        property_embeddings_adapt_cfg=None,
        condition_names=None,
        vae_pretrained_model_name=None,
        vae_ckpt_path=None,
        ldm_pretrained_model_name=None,
        ldm_ckpt_path=None,
        ckpt_path=None,
        rl_configs=None,
        reward_fn_cfg=None,
        sampling_configs=None,
        execution_backend="eager",
        runtime_options=None,
        energy_calculator=None,
        reference_structures=None,
    ):
        super().__init__()
        if stage not in ("vae", "diffusion", "rl"):
            raise ValueError(
                f"stage must be one of 'vae', 'diffusion', 'rl', got {stage!r}."
            )
        self.stage = stage
        if stage != "rl" and (
            energy_calculator is not None or reference_structures is not None
        ):
            raise ValueError("Reward runtime dependencies require stage='rl'.")

        if stage == "vae":
            from ppmat.models import build_model

            self.core = build_model(vae_cfg)
            if self.core is None:
                raise ValueError("stage='vae' requires a VAE configuration.")
            if vae_pretrained_model_name is not None:
                from ppmat.models import MODEL_REGISTRY

                if vae_ckpt_path is not None:
                    raise ValueError(
                        "Choose a registered VAE or a local VAE checkpoint."
                    )
                load_pretrain(self.core, MODEL_REGISTRY[vae_pretrained_model_name])
            if vae_ckpt_path is not None:
                load_pretrain(self, vae_ckpt_path)
        else:
            has_ldm_components = any(
                value is not None
                for value in (denoiser_cfg, noise_scheduler_cfg, vae_cfg)
            )
            if ldm_pretrained_model_name is None and not has_ldm_components:
                raise ValueError(
                    f"stage={stage!r} requires ldm_pretrained_model_name or explicit "
                    "LDM component configurations (ldm_ckpt_path alone cannot define "
                    "the architecture)."
                )
            if stage == "diffusion":
                if ldm_pretrained_model_name is not None:
                    from ppmat.models import build_model_from_name

                    self.core, _ = build_model_from_name(ldm_pretrained_model_name)
                else:
                    self.core = Chemeleon2LDMModule(
                        denoiser_cfg=denoiser_cfg,
                        noise_scheduler_cfg=noise_scheduler_cfg,
                        vae_cfg=vae_cfg,
                        property_embeddings_adapt_cfg=property_embeddings_adapt_cfg,
                        condition_names=condition_names,
                        vae_pretrained_model_name=vae_pretrained_model_name,
                        vae_ckpt_path=vae_ckpt_path,
                        execution_backend=execution_backend,
                        runtime_options=runtime_options,
                    )
                if ldm_ckpt_path is not None:
                    load_pretrain(self.core, ldm_ckpt_path)
            else:
                from ppmat.models.chemeleon2.rl_module import rl_module

                self.core = rl_module.Chemeleon2RLModule(
                    ldm_pretrained_model_name=ldm_pretrained_model_name,
                    ldm_ckpt_path=ldm_ckpt_path,
                    denoiser_cfg=denoiser_cfg,
                    noise_scheduler_cfg=noise_scheduler_cfg,
                    vae_cfg=vae_cfg,
                    property_embeddings_adapt_cfg=property_embeddings_adapt_cfg,
                    condition_names=condition_names,
                    vae_pretrained_model_name=vae_pretrained_model_name,
                    vae_ckpt_path=vae_ckpt_path,
                    rl_configs=rl_configs,
                    reward_fn_cfg=reward_fn_cfg,
                    sampling_configs=sampling_configs,
                    energy_calculator=energy_calculator,
                    reference_structures=reference_structures,
                    execution_backend=execution_backend,
                    runtime_options=runtime_options,
                )

        self.set_execution_backend(execution_backend)
        self.set_runtime_options(runtime_options or {})
        if ckpt_path is not None:
            load_pretrain(self, ckpt_path)

    @property
    def condition_names(self):
        return getattr(self.core, "condition_names", None)

    @property
    def supports_num_atoms_sampling(self):
        return self.stage != "vae" and getattr(
            self.core, "supports_num_atoms_sampling", True
        )

    def before_train(self, trainer):
        before_train = getattr(self.core, "before_train", None)
        if before_train is not None:
            before_train(trainer)

    def forward(self, batch):
        return self.core(batch)

    def sample(self, data, **kwargs):
        sample_fn = getattr(self.core, "sample", None)
        if sample_fn is None:
            raise ValueError(
                f"stage={self.stage!r} has no sample(); generation requires "
                "stage 'diffusion' or 'rl'."
            )
        return sample_fn(data, **kwargs)

    def predict(self, data, **kwargs):
        predict_fn = getattr(self.core, "predict", None)
        if predict_fn is None:
            raise ValueError(
                f"stage={self.stage!r} has no predict(); generation requires "
                "stage 'diffusion' or 'rl'."
            )
        return predict_fn(data, **kwargs)

    @property
    def _ldm(self):
        return self.core.ldm if self.stage == "rl" else self.core

    @property
    def denoiser(self):
        return self._ldm.denoiser

    @property
    def vae(self):
        return self.core if self.stage == "vae" else self._ldm.vae

    @property
    def diffusion(self):
        return self._ldm.diffusion

    @property
    def condition_module(self):
        return self._ldm.condition_module

    @property
    def latent_std(self):
        return self._ldm.latent_std

    def calculate_loss(self, *args, **kwargs):
        return self.core.calculate_loss(*args, **kwargs)

    @property
    def execution_backend(self):
        return "eager" if self.stage == "vae" else self.core.execution_backend

    def set_execution_backend(self, backend):
        if self.stage == "vae":
            if backend != "eager":
                raise ValueError("The VAE stage supports only the eager backend.")
            return
        self.core.set_execution_backend(backend)

    def set_runtime_options(self, runtime_options):
        if self.stage == "vae":
            if runtime_options:
                raise ValueError("The VAE stage does not accept runtime options.")
            return
        self.core.set_runtime_options(runtime_options)

    def get_runtime_options(self, backend):
        if self.stage == "vae":
            if backend != "eager":
                raise ValueError("The VAE stage supports only the eager backend.")
            return {}
        return self.core.get_runtime_options(backend)

    def invalidate_runtime(self):
        if self.stage != "vae":
            self.core.invalidate_runtime()

    def _run_runtime(self, *args, **kwargs):
        return self.core._run_runtime(*args, **kwargs)

    def _runtime_denoise(self, *args, **kwargs):
        return self._ldm._runtime_denoise(*args, **kwargs)

    @property
    def use_cfg(self):
        return getattr(self._ldm, "use_cfg", False)

    @property
    def noise_scheduler_cfg(self):
        return self._ldm.noise_scheduler_cfg

    def validate_execution_backend(self, *, use_amp=False, world_size=1):
        if self.stage == "vae":
            return
        return self.core.validate_execution_backend(
            use_amp=use_amp, world_size=world_size
        )

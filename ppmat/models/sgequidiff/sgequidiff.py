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

"""Wrappers and samplers for SGEquiDiff."""

from __future__ import annotations

from typing import Any
from typing import Dict
from typing import List
from typing import Optional

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.distribution import Categorical

from ppmat.models.common.runtime import RuntimeMixin
from ppmat.models.sgequidiff.asu_math import asu_to_pymatgen_structure
from ppmat.models.sgequidiff.diffusion_model import EquivariantDiffusionModel
from ppmat.models.sgequidiff.lattice_sampler import TelescopingDiscreteLatticeSampler
from ppmat.models.sgequidiff.sgequidiff_meta import ELEMENT_ENCODING_SIZE
from ppmat.models.sgequidiff.sgequidiff_meta import chemical_symbols
from ppmat.models.sgequidiff.sgequidiff_meta import lattice_parameter_ranges
from ppmat.models.sgequidiff.sgequidiff_meta import max_atoms_per_dataset
from ppmat.models.sgequidiff.vocabs import build_embedding_tools
from ppmat.models.sgequidiff.wyckoff_geometry import build_wyckoff_geometry
from ppmat.models.sgequidiff.wyckoff_transformer import WyckoffElementTransformer
from ppmat.utils import logger
from ppmat.utils.crystal import lattice_params_to_matrix_paddle


def _as_tensor(x, dtype=None):
    """Convert numpy inputs to tensors; pass tensors through without copying."""
    if isinstance(x, paddle.Tensor):
        return x.cast(dtype) if dtype is not None else x
    return paddle.to_tensor(x, dtype=dtype)


class SpaceGroupSampler(nn.Layer):
    def __init__(self):
        super().__init__()
        self.marginal_space_group_logits = paddle.create_parameter(
            shape=[230],
            dtype="float32",
            default_initializer=nn.initializer.Constant(1.0),
        )

    def sample_and_log_prob(self, batch_size: int = 1, temperature: float = 1.0):
        log_probs = F.log_softmax(
            self.marginal_space_group_logits / temperature, axis=-1
        )
        log_probs = log_probs.unsqueeze(0).expand([batch_size, -1])

        dist = Categorical(logits=log_probs)
        sample = dist.sample([1]).squeeze(0)

        sample_log_probs = paddle.take_along_axis(
            log_probs, sample.unsqueeze(-1), axis=-1
        ).squeeze(-1)
        return sample, sample_log_probs

    def log_prob(self, space_group_indices):
        normed_logits = F.log_softmax(self.marginal_space_group_logits, axis=-1)
        return normed_logits[space_group_indices]


class SGEQuiDiff(RuntimeMixin, nn.Layer):
    """Full crystal sampler combining all submodules.

    This is the unified model entry: ``forward(batch)`` returns the
    training loss dict (delegating to the internal coordinate diffusion model),
    and ``sample(batch_data)`` returns structures compatible with
    ``structure_generation/sample.py``.

    SGEQuiDiff is an **unconditional** generator:
    ``sample_by_num_atoms`` raises ``NotImplementedError`` (it declares
    ``supports_num_atoms_sampling = False``), and the formula passed to
    ``sample_by_chemical_formula`` is ignored (it declares
    ``supports_chemical_formula_sampling = False``); use
    ``sample_by_dataloader`` instead.

    Args:
        dataset_name: Dataset name, e.g. ``"mp_20"``.
        diffusion_snr: SNR for the diffusion sampling step.
        temperature: Sampling temperature.
        num_timesteps: Number of diffusion timesteps.
        noise_scheduler_num_monte_carlo_samples: MC samples for sigma norms.
        num_lattice_translations: Lattice-translation neighbors for the
            wrapped-normal noise model.
        sigma_min: Lower bound of the VE-SDE noise schedule.
        sigma_max: Upper bound of the VE-SDE noise schedule.
        model_type: Coordinate drift backbone type (``"gnn"``/``"mlp"``/``"cspnet"``).
        time_emb_dim: Time embedding dimension.
        num_plane_wave_freqs: Number of plane wave frequencies.
        gnn_config: GNN backbone overrides (plain dict of constructor kwargs).
        cspnet_config: CSPNet backbone overrides (plain dict of constructor kwargs).
        noise_scheduler_cfg: Noise scheduler config dict.
        vocab: Prebuilt ``sgequidiff`` vocabulary dict. ``None`` (default)
            resolves it internally via ``ppmat.vocab.build_vocab``, with no
            framework-level wiring required.
        lattice_length_noise: Noise added to lattice lengths during training.
        lattice_angle_noise: Noise added to lattice angles during training.
        space_group_grad_weight: Loss weight for the space-group log-prob.
        lattice_grad_weight: Loss weight for the lattice log-prob.
        wyckoff_element_grad_weight: Loss weight for the wyckoff/element log-prob.
        frac_coord_grad_weight: Loss weight for the coordinate score matching.
        lattice_sampler_config: Overrides for the lattice sampler constructor
            fields. Keys must be valid constructor parameter names; unknown
            keys raise a ``TypeError``. ``None`` uses the defaults, which
            match the released checkpoint structure.
        wyckoff_element_transformer_config: Overrides for the
            Wyckoff-element transformer constructor fields, same semantics as
            ``lattice_sampler_config``.
        execution_backend: Numerical execution backend, ``"eager"`` or
            ``"cinn"``. Forwarded to the coordinate diffusion model, which
            owns the compiled runtime. Defaults to ``"eager"``.
        runtime_options: Per-backend runtime options forwarded to the
            coordinate diffusion model.
    """

    supports_num_atoms_sampling = False
    supports_chemical_formula_sampling = False

    def __init__(
        self,
        dataset_name: str = "mp_20",
        diffusion_snr: float = 0.4,
        temperature: float = 1.0,
        num_timesteps: int = 1000,
        noise_scheduler_num_monte_carlo_samples: int = 2500,
        num_lattice_translations: int = 3,
        sigma_min: float = 0.002,
        sigma_max: float = 0.5,
        model_type: str = "gnn",
        time_emb_dim: int = 128,
        num_plane_wave_freqs: int = 96,
        gnn_config: Any = None,
        cspnet_config: Any = None,
        noise_scheduler_cfg: Optional[dict] = None,
        lattice_length_noise: Optional[float] = 0.0,
        lattice_angle_noise: Optional[float] = 0.0,
        space_group_grad_weight: Optional[float] = 1.0,
        lattice_grad_weight: Optional[float] = 1.0,
        wyckoff_element_grad_weight: Optional[float] = 1.0,
        frac_coord_grad_weight: Optional[float] = 1.0,
        lattice_sampler_config: Optional[Dict[str, Any]] = None,
        wyckoff_element_transformer_config: Optional[Dict[str, Any]] = None,
        execution_backend: str = "eager",
        runtime_options: Optional[dict] = None,
        vocab: Optional[dict] = None,
    ):
        super().__init__()
        self.dataset_name = dataset_name
        self.diffusion_snr = diffusion_snr
        self.temperature = temperature

        lr = lattice_parameter_ranges.get(
            dataset_name, lattice_parameter_ranges["mp_20"]
        )

        if gnn_config is None:
            gnn_config = {"dataset_name": dataset_name}

        lattice_kwargs = {
            "min_lattice_length": lr["min_lattice_length"],
            "max_lattice_length": lr["max_lattice_length"],
            "min_lattice_angle": lr["min_lattice_angle"],
            "max_lattice_angle": lr["max_lattice_angle"],
        }
        if lattice_sampler_config is not None:
            lattice_kwargs.update(lattice_sampler_config)
        we_kwargs = {"dataset_name": dataset_name}
        if wyckoff_element_transformer_config is not None:
            we_kwargs.update(wyckoff_element_transformer_config)

        self.lattice_length_noise = lattice_length_noise
        self.lattice_angle_noise = lattice_angle_noise
        self.space_group_grad_weight = space_group_grad_weight
        self.lattice_grad_weight = lattice_grad_weight
        self.wyckoff_element_grad_weight = wyckoff_element_grad_weight
        self.frac_coord_grad_weight = frac_coord_grad_weight

        # Build static resources explicitly (no global singletons).
        self.wyckoff_geometry = build_wyckoff_geometry(vocab)
        self.embedding_tools = build_embedding_tools(vocab)

        self.atom_coord_diffusion_model = EquivariantDiffusionModel(
            model_type=model_type,
            num_timesteps=num_timesteps,
            noise_scheduler_num_monte_carlo_samples=noise_scheduler_num_monte_carlo_samples,
            num_lattice_translations=num_lattice_translations,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            time_emb_dim=time_emb_dim,
            num_plane_wave_freqs=num_plane_wave_freqs,
            gnn_config=gnn_config,
            cspnet_config=cspnet_config,
            noise_scheduler_cfg=noise_scheduler_cfg,
            wyckoff_geometry=self.wyckoff_geometry,
            embedding_tools=self.embedding_tools,
            execution_backend=execution_backend,
            runtime_options=runtime_options,
        )
        self.space_group_sampler = SpaceGroupSampler()
        self.lattice_sampler = TelescopingDiscreteLatticeSampler(
            embedding_tools=self.embedding_tools,
            **lattice_kwargs,
        )
        self.wyckoff_and_element_sampler = WyckoffElementTransformer(
            wyckoff_geometry=self.wyckoff_geometry,
            embedding_tools=self.embedding_tools,
            **we_kwargs,
        )

        logger.info(
            f"[SGEQuiDiff] Initialized: dataset={dataset_name}, "
            f"snr={diffusion_snr}, temperature={temperature}"
        )

    # Framework-facing runtime protocol: configure_execution_backend in
    # ppmat/utils/execution.py resolves these via getattr (trainer, predictor,
    # and samplers are the real callers). No direct callers exist in this
    # file; do not remove. The property is read-only by design: every
    # mutation goes through set_execution_backend, the single entry point.
    @property
    def execution_backend(self) -> str:
        """Active numerical execution backend (owned by the diffusion model)."""
        return self.atom_coord_diffusion_model.execution_backend

    def set_execution_backend(self, backend: str) -> None:
        self.atom_coord_diffusion_model.set_execution_backend(backend)

    def set_runtime_options(self, runtime_options: dict) -> None:
        self.atom_coord_diffusion_model.set_runtime_options(runtime_options)

    def validate_execution_backend(
        self, use_amp: bool = False, world_size: int = 1
    ) -> None:
        self.atom_coord_diffusion_model.validate_execution_backend(
            use_amp=use_amp, world_size=world_size
        )

    def forward(self, batch) -> Dict:
        """Training entry: MLE on discrete variables + score matching on coords.

        Returns:
            loss_dict : total loss (``loss_dict["loss"]`` is backpropagated).
            pred_dict : detached per-component log-probs / losses, tracked in
                training logs only. SGEQuiDiff is an unconditional generative
                model: its training forward has no per-attribute predictions
                aligned with batch labels, so **non-streaming
                ``compute_metric_func_dict`` metrics are not applicable** here.
                Generations are scored with the streaming
                ``SGEQuiDiffMetric`` (``stage == "sample"``).
            label_dict : supervision fields passed through with keys identical
                to the corresponding ``batch`` labels, so per-key
                pred/label lookups stay aligned when a streaming metric needs
                them.
        """
        loss, artifacts = self.compute_loss(batch)
        label_keys = (
            "space_group_indices",
            "lattice_lengths",
            "lattice_angles",
            "n_atoms_per_asu",
            "element_indices",
            "wyckoff_indices",
            "wyckoff_shape_indices",
            "frac_coords",
        )
        label_dict = {k: batch[k] for k in label_keys if k in batch}
        return {
            "loss_dict": {"loss": loss},
            "pred_dict": artifacts,
            "label_dict": label_dict,
        }

    def compute_loss(self, batch):
        """MLE on discrete variables, score matching on atom coordinates.

        Returns:
            loss: scalar tensor.
            artifacts: dict of detached log-probs / losses for logging.
        """
        space_group_indices = _as_tensor(batch["space_group_indices"])
        lattice_lengths = _as_tensor(batch["lattice_lengths"], paddle.float32)
        lattice_angles = _as_tensor(batch["lattice_angles"], paddle.float32)
        if "lattice_matrices" in batch:
            lattice_matrices = batch["lattice_matrices"]
        else:
            lattice_matrices = lattice_params_to_matrix_paddle(
                lattice_lengths, lattice_angles
            )
        lattice_matrices = _as_tensor(lattice_matrices, paddle.float32)
        element_indices = _as_tensor(batch["element_indices"], paddle.int64)
        wyckoff_indices = _as_tensor(batch["wyckoff_indices"], paddle.int64)
        n_asu_atoms_per_xtal = _as_tensor(batch["n_atoms_per_asu"], paddle.int64)
        wyckoff_shape_indices = _as_tensor(batch["wyckoff_shape_indices"], paddle.int64)
        asu_frac_coords = _as_tensor(batch["frac_coords"], paddle.float32)

        if self.training and (
            self.lattice_length_noise > 0 or self.lattice_angle_noise > 0
        ):
            (
                noisy_lattice_lengths,
                noisy_lattice_angles,
            ) = self.get_noisy_lattice_lengths_and_angles(
                lattice_lengths, lattice_angles, space_group_indices
            )
        else:
            noisy_lattice_lengths = lattice_lengths
            noisy_lattice_angles = lattice_angles

        space_group_log_prob = self.space_group_log_probs(space_group_indices)
        lattice_log_prob = self.lattice_param_log_probs(
            lattice_lengths,
            lattice_angles,
            space_group_indices,
            noisy_lattice_lengths,
            noisy_lattice_angles,
        )
        (
            elements_log_prob,
            wyckoffs_log_prob,
            termination_log_prob,
        ) = self.element_and_wyckoff_log_probs(
            element_indices,
            wyckoff_indices,
            n_asu_atoms_per_xtal,
            noisy_lattice_lengths,
            noisy_lattice_angles,
            space_group_indices,
        )
        score_matching_loss = self.atom_coord_diffusion_model.compute_loss(
            asu_frac_coords,
            element_indices,
            wyckoff_indices,
            space_group_indices,
            n_asu_atoms_per_xtal,
            wyckoff_shape_indices,
            lattice_matrices,
            lattice_lengths,
            lattice_angles,
        )
        (
            space_group_log_prob,
            lattice_log_prob,
            termination_log_prob,
            elements_log_prob,
            wyckoffs_log_prob,
            score_matching_loss,
            detached_space_group_log_prob,
            detached_lattice_log_prob,
            detached_termination_log_prob,
            detached_elements_log_prob,
            detached_wyckoffs_log_prob,
            detached_score_matching_loss,
        ) = self._get_rebalanced_grads(
            space_group_log_prob,
            lattice_log_prob,
            termination_log_prob,
            elements_log_prob,
            wyckoffs_log_prob,
            score_matching_loss,
        )
        artifacts = {
            "space_group_log_prob": detached_space_group_log_prob,
            "lattice_log_prob": detached_lattice_log_prob,
            "elements_log_prob": detached_elements_log_prob,
            "wyckoffs_log_prob": detached_wyckoffs_log_prob,
            "termination_log_prob": detached_termination_log_prob,
            "score_matching_loss": detached_score_matching_loss,
        }
        lattice_log_prob_nonzero = lattice_log_prob[lattice_log_prob != 0.0]
        if lattice_log_prob_nonzero.numel() > 0:
            lattice_nll = lattice_log_prob_nonzero.mean()
        else:
            lattice_nll = paddle.zeros([1], dtype=lattice_log_prob.dtype)
        nll = -1.0 * (
            (space_group_log_prob + termination_log_prob).mean()
            + lattice_nll
            + (elements_log_prob + wyckoffs_log_prob).mean()
        )
        return nll + score_matching_loss, artifacts

    def space_group_log_probs(
        self, space_group_indices: paddle.Tensor
    ) -> paddle.Tensor:
        """Log probabilities of the given space group indices."""
        return self.space_group_sampler.log_prob(space_group_indices)

    def lattice_param_log_probs(
        self,
        lattice_lengths: paddle.Tensor,
        lattice_angles: paddle.Tensor,
        space_group_indices: paddle.Tensor,
        noisy_lattice_lengths: paddle.Tensor = None,
        noisy_lattice_angles: paddle.Tensor = None,
    ) -> paddle.Tensor:
        log_probs = self.lattice_sampler.log_prob(
            lattice_lengths,
            lattice_angles,
            space_group_indices,
            noisy_lattice_lengths,
            noisy_lattice_angles,
        )
        return log_probs

    def element_and_wyckoff_log_probs(
        self,
        element_indices: paddle.Tensor,
        wyckoff_indices: paddle.Tensor,
        n_asu_atoms_per_xtal: paddle.Tensor,
        lattice_lengths: paddle.Tensor,
        lattice_angles: paddle.Tensor,
        space_group_indices: paddle.Tensor,
    ):
        return self.wyckoff_and_element_sampler.log_prob(
            element_indices,
            wyckoff_indices,
            n_asu_atoms_per_xtal,
            lattice_lengths,
            lattice_angles,
            space_group_indices,
            # Static pad width: avoids the per-step GPU sync of
            # ``.max()`` and keeps shapes CINN-friendly.
            max_atoms=max_atoms_per_dataset[self.dataset_name],
        )

    def get_noisy_lattice_lengths_and_angles(
        self,
        lattice_lengths: paddle.Tensor,
        lattice_angles: paddle.Tensor,
        space_group_indices: paddle.Tensor,
    ):
        """Rejection-sample noisy lattice parameters under Bravais constraints.

        NOTE: only invoked when ``lattice_length_noise`` or ``lattice_angle_noise``
        is > 0 (see ``compute_loss``); the default config (0.0) skips this path.
        The while-loop checks ``.item()`` every iteration, forcing a GPU sync;
        this matches the upstream algorithm and is kept for parity.
        """
        batch_size = space_group_indices.shape[0]
        lattice_sampler = self.lattice_sampler
        MIN_LATTICE_LENGTH = lattice_sampler.MIN_LATTICE_LENGTH
        MAX_LATTICE_LENGTH = lattice_sampler.MAX_LATTICE_LENGTH
        MIN_LATTICE_ANGLE = lattice_sampler.MIN_LATTICE_ANGLE
        MAX_LATTICE_ANGLE = lattice_sampler.MAX_LATTICE_ANGLE
        length_transforms = lattice_sampler.bravais_length_transforms[
            space_group_indices
        ]
        angle_transforms = lattice_sampler.bravais_angle_transforms[space_group_indices]
        angle_offsets = lattice_sampler.bravais_angle_offsets[space_group_indices]
        lattice_angle_bounds = paddle.to_tensor(
            [MIN_LATTICE_ANGLE, MAX_LATTICE_ANGLE], dtype=paddle.float32
        )
        lattice_noise_magnitude = paddle.to_tensor(
            [
                self.lattice_length_noise,
                self.lattice_length_noise,
                self.lattice_length_noise,
                self.lattice_angle_noise,
                self.lattice_angle_noise,
                self.lattice_angle_noise,
            ],
            dtype=paddle.float32,
        )
        num_samples_per_iter = 2
        max_iters = 100
        iteration = 0
        lattice_param_is_done = paddle.zeros([batch_size, 6], dtype="bool")
        accepted_noisy_lattice_parameters = paddle.full(
            [batch_size, 6], fill_value=-1.0, dtype=paddle.float32
        )
        # NOTE: .item() per iteration forces a GPU->CPU sync; only active when
        # lattice noise > 0, inherited from the upstream rejection-sampling loop.
        while not paddle.all(lattice_param_is_done).item():
            iteration += 1
            if iteration > max_iters:
                raise RuntimeError(
                    "Failed to sample valid noisy lattice parameters after "
                    f"{max_iters} iterations: "
                    f"{int((~lattice_param_is_done).all(axis=-1).sum())} "
                    f"of {batch_size} lattices still incomplete"
                )
            lattice_is_done = paddle.all(lattice_param_is_done, axis=-1)
            unfinished_lengths = lattice_lengths[~lattice_is_done]
            unfinished_angles = lattice_angles[~lattice_is_done]
            num_lattices_left_to_noise = unfinished_angles.shape[0]
            noise = lattice_noise_magnitude[None, None, :] * (
                2.0 * paddle.rand([num_lattices_left_to_noise, num_samples_per_iter, 6])
                - 1.0
            )
            noisy_lengths = noise[:, :, :3] + unfinished_lengths[:, None, :]
            noisy_angles = noise[:, :, 3:] + unfinished_angles[:, None, :]
            noisy_lengths = paddle.bmm(
                noisy_lengths, length_transforms[~lattice_is_done]
            )
            noisy_angles = (
                paddle.bmm(noisy_angles, angle_transforms[~lattice_is_done])
                + angle_offsets[~lattice_is_done][:, None, :]
            )
            lengths_are_valid = (noisy_lengths >= MIN_LATTICE_LENGTH) & (
                noisy_lengths <= MAX_LATTICE_LENGTH
            )
            (
                min_gamma_angle,
                max_gamma_angle,
            ) = lattice_sampler.get_valid_gamma_angle_interval(
                alpha_and_beta_angles=noisy_angles[:, :, :2].reshape([-1, 2])
            )
            min_gamma_angle = min_gamma_angle + 0.01
            max_gamma_angle = max_gamma_angle - 0.01
            min_allowed_angles = paddle.concat(
                [
                    lattice_angle_bounds[0]
                    .reshape([1, 1, 1])
                    .expand([num_lattices_left_to_noise, num_samples_per_iter, 2]),
                    min_gamma_angle.reshape(
                        [num_lattices_left_to_noise, num_samples_per_iter, 1]
                    ),
                ],
                axis=-1,
            )
            max_allowed_angles = paddle.concat(
                [
                    lattice_angle_bounds[1]
                    .reshape([1, 1, 1])
                    .expand([num_lattices_left_to_noise, num_samples_per_iter, 2]),
                    max_gamma_angle.reshape(
                        [num_lattices_left_to_noise, num_samples_per_iter, 1]
                    ),
                ],
                axis=-1,
            )
            angles_are_valid = (noisy_angles >= min_allowed_angles) & (
                noisy_angles <= max_allowed_angles
            )
            for i in range(3):
                sample_is_valid = lengths_are_valid[:, :, i]
                any_sample_is_valid = paddle.any(sample_is_valid, axis=-1)
                if paddle.any(any_sample_is_valid).item():
                    update_lattice_params_mask = (
                        paddle.nn.functional.one_hot(
                            paddle.to_tensor([i], dtype="int64"), num_classes=6
                        )
                        .cast("bool")
                        .tile([batch_size, 1])
                    )
                    update_lattice_params_mask[lattice_is_done] = False
                    accept_sample_mask = (
                        any_sample_is_valid
                        & ~lattice_param_is_done[~lattice_is_done][:, i]
                    )
                    update_lattice_params_mask[
                        ~lattice_is_done[:, None] & update_lattice_params_mask
                    ] = accept_sample_mask
                    first_accepted_length_idx = paddle.argmax(
                        sample_is_valid.cast("float32"), axis=-1
                    )[accept_sample_mask][:, None]
                    accepted_noisy_lattice_parameters[update_lattice_params_mask] = (
                        noisy_lengths[:, :, i][accept_sample_mask]
                        .take_along_axis(first_accepted_length_idx, axis=1)
                        .reshape([-1])
                    )
                    lattice_param_is_done[update_lattice_params_mask] = True
            sample_is_valid = paddle.all(angles_are_valid, axis=-1)
            any_sample_is_valid = paddle.any(sample_is_valid, axis=-1)
            update_lattice_angles_mask = paddle.to_tensor(
                [[False, False, False, True, True, True]], dtype="bool"
            ).tile([batch_size, 1])
            update_lattice_angles_mask[lattice_is_done] = False
            accept_sample_mask = any_sample_is_valid & paddle.all(
                ~lattice_param_is_done[~lattice_is_done][:, 3:], axis=-1
            )
            update_lattice_angles_mask[
                ~lattice_is_done[:, None] & update_lattice_angles_mask
            ] = (accept_sample_mask[:, None].tile([1, 3]).reshape([-1]))
            first_accepted_angles_idx = paddle.argmax(
                sample_is_valid.cast("float32"), axis=-1
            )[accept_sample_mask]
            _lattice_idx = paddle.arange(first_accepted_angles_idx.shape[0])
            accepted_noisy_lattice_parameters[
                update_lattice_angles_mask
            ] = noisy_angles[accept_sample_mask][
                _lattice_idx, first_accepted_angles_idx
            ].reshape(
                [-1]
            )
            lattice_param_is_done[update_lattice_angles_mask] = True
        noisy_lattice_lengths = accepted_noisy_lattice_parameters[:, :3]
        noisy_lattice_angles = accepted_noisy_lattice_parameters[:, 3:]
        return noisy_lattice_lengths, noisy_lattice_angles

    def _get_rebalanced_grads(
        self,
        space_group_log_prob: paddle.Tensor,
        lattice_log_prob: paddle.Tensor,
        termination_log_prob: paddle.Tensor,
        elements_log_prob: paddle.Tensor,
        wyckoffs_log_prob: paddle.Tensor,
        score_matching_loss: paddle.Tensor,
    ):
        """Apply straight-through estimators to rebalance gradients."""
        detached_space_group_log_prob = space_group_log_prob.detach()
        detached_lattice_log_prob = lattice_log_prob.detach()
        detached_termination_log_prob = termination_log_prob.detach()
        detached_elements_log_prob = elements_log_prob.detach()
        detached_wyckoffs_log_prob = wyckoffs_log_prob.detach()
        detached_score_matching_loss = score_matching_loss.detach()
        space_group_log_prob = (
            self.space_group_grad_weight * space_group_log_prob
            - self.space_group_grad_weight * detached_space_group_log_prob
            + detached_space_group_log_prob
        )
        lattice_log_prob = (
            self.lattice_grad_weight * lattice_log_prob
            - self.lattice_grad_weight * detached_lattice_log_prob
            + detached_lattice_log_prob
        )
        termination_log_prob = (
            self.wyckoff_element_grad_weight * termination_log_prob
            - self.wyckoff_element_grad_weight * detached_termination_log_prob
            + detached_termination_log_prob
        )
        elements_log_prob = (
            self.wyckoff_element_grad_weight * elements_log_prob
            - self.wyckoff_element_grad_weight * detached_elements_log_prob
            + detached_elements_log_prob
        )
        wyckoffs_log_prob = (
            self.wyckoff_element_grad_weight * wyckoffs_log_prob
            - self.wyckoff_element_grad_weight * detached_wyckoffs_log_prob
            + detached_wyckoffs_log_prob
        )
        score_matching_loss = (
            self.frac_coord_grad_weight * score_matching_loss
            - self.frac_coord_grad_weight * detached_score_matching_loss
            + detached_score_matching_loss
        )
        return (
            space_group_log_prob,
            lattice_log_prob,
            termination_log_prob,
            elements_log_prob,
            wyckoffs_log_prob,
            score_matching_loss,
            detached_space_group_log_prob,
            detached_lattice_log_prob,
            detached_termination_log_prob,
            detached_elements_log_prob,
            detached_wyckoffs_log_prob,
            detached_score_matching_loss,
        )

    @paddle.no_grad()
    def sample_crystal(
        self,
        batch_size: int,
        diffusion_snr: float = 0.4,
        temperature: float = 1.0,
        space_group_numbers=None,
        lattice_parameters=None,
        wyckoff_element_data=None,
    ) -> List[Dict]:
        """Full crystal sampling pipeline."""

        if space_group_numbers is None:
            space_group_indices, _ = self.sample_and_log_prob_space_group(
                batch_size, temperature
            )
        else:
            if space_group_numbers.shape[0] != batch_size:
                raise ValueError(
                    "space_group_numbers batch size "
                    f"({space_group_numbers.shape[0]}) does not match "
                    f"batch_size ({batch_size})."
                )
            space_group_indices = space_group_numbers - 1

        if lattice_parameters is None:
            (
                lattice_lengths,
                lattice_angles,
                _,
            ) = self.sample_and_log_prob_lattice_parameters(space_group_indices)
        else:
            lattice_lengths = lattice_parameters[:, :3]
            lattice_angles = lattice_parameters[:, 3:]

        lattice_matrices = lattice_params_to_matrix_paddle(
            lattice_lengths, lattice_angles
        )

        if wyckoff_element_data is None:
            (
                element_indices,
                wyckoff_indices,
                n_asu_atoms_per_xtal,
                elements_log_prob,
                wyckoffs_log_prob,
                termination_log_prob,
            ) = self.sample_and_log_prob_elements_and_wyckoffs(
                lattice_lengths, lattice_angles, space_group_indices, temperature
            )
        else:
            element_indices = wyckoff_element_data["element_indices"]
            wyckoff_indices = wyckoff_element_data["wyckoff_indices"]
            n_asu_atoms_per_xtal = wyckoff_element_data["n_asu_atoms_per_xtal"]

        frac_coords = self.sample_frac_coords(
            wyckoff_indices,
            element_indices,
            space_group_indices,
            n_asu_atoms_per_xtal,
            lattice_matrices,
            lattice_lengths,
            lattice_angles,
            snr=diffusion_snr,
            max_step_size=1e6,
        )

        asu_crystals = []
        atom_offsets = paddle.concat(
            [
                paddle.cumsum(n_asu_atoms_per_xtal, axis=0) - n_asu_atoms_per_xtal,
                n_asu_atoms_per_xtal.sum().unsqueeze(0),
            ],
            axis=0,
        )
        for i in range(n_asu_atoms_per_xtal.shape[0]):
            asu_crystals.append(
                {
                    "space_group_number": 1 + space_group_indices[i],
                    "conventional_lattice_lengths": lattice_lengths[i],
                    "conventional_lattice_angles": lattice_angles[i],
                    "element_indices": element_indices[
                        atom_offsets[i] : atom_offsets[i + 1]
                    ],
                    "wyckoff_indices": wyckoff_indices[
                        atom_offsets[i] : atom_offsets[i + 1]
                    ],
                    "conventional_frac_coords": frac_coords[
                        atom_offsets[i] : atom_offsets[i + 1]
                    ],
                }
            )
        return asu_crystals

    def sample_and_log_prob_space_group(self, batch_size, temperature=1.0):
        return self.space_group_sampler.sample_and_log_prob(batch_size, temperature)

    def sample_and_log_prob_lattice_parameters(self, space_group_indices):
        lengths, angles, log_probs = self.lattice_sampler(space_group_indices)
        return lengths, angles, log_probs

    def sample_and_log_prob_elements_and_wyckoffs(
        self, lattice_lengths, lattice_angles, space_group_indices, temperature=1.0
    ):
        return self.wyckoff_and_element_sampler.sample_and_log_prob(
            lattice_lengths, lattice_angles, space_group_indices, temperature
        )

    def sample_frac_coords(
        self,
        wyckoff_indices,
        element_indices,
        space_group_indices,
        n_asu_atoms_per_xtal,
        lattice_matrices,
        lattice_lengths,
        lattice_angles,
        snr=0.4,
        max_step_size=1e6,
    ):
        trajectory = self.atom_coord_diffusion_model.sample(
            wyckoff_indices,
            element_indices,
            space_group_indices,
            n_asu_atoms_per_xtal,
            lattice_matrices,
            lattice_lengths,
            lattice_angles,
            snr,
            max_step_size,
        )
        return trajectory[0]

    @paddle.no_grad()
    def sample(self, batch_data: Dict, **kwargs) -> Dict:
        """Sample crystals compatible with ``structure_generation/sample.py``.

        Args:
            batch_data: Dict with key ``"structure_array"`` containing at
                least ``"num_atoms"`` (a 1-D int tensor of batch sizes).
                SGEQuiDiff is an **unconditional** generator, so the actual
                atom counts in *batch_data* are ignored -- they only
                determine the *batch size*.

        Returns:
            Dict with key ``"result"`` mapping to a list of dicts, each
            containing the **full conventional cell** (ASU expanded via the
            space-group symmetry operations):
            - ``"num_atoms"``: int
            - ``"atom_types"``: list[int]  (1-indexed atomic numbers)
            - ``"frac_coords"``: list[list[float]]
            - ``"lengths"``: list[float]  (a, b, c in Angstrom)
            - ``"angles"``: list[float]   (alpha, beta, gamma in degrees)
        """
        structure_array = batch_data.get("structure_array")
        if structure_array is not None:
            num_atoms_tensor = structure_array["num_atoms"]
        else:
            # sample_by_dataloader passes an ASU dataset batch without
            # structure_array; SGEQuiDiff is unconditional so the number of
            # generated crystals is simply the batch size.
            num_atoms_tensor = paddle.ones(
                [batch_data["n_atoms_per_asu"].shape[0]], dtype=paddle.int64
            )
        batch_size = num_atoms_tensor.shape[0]

        crystals = self.sample_crystal(
            batch_size=batch_size,
            diffusion_snr=self.diffusion_snr,
            temperature=self.temperature,
        )

        # Expand each ASU crystal into a full conventional cell, so the result
        # is a physically complete structure (not just the asymmetric unit).
        result = []
        for crystal in crystals:
            valid_mask = []
            for i, idx in enumerate(crystal["element_indices"].tolist()):
                if 0 <= idx < ELEMENT_ENCODING_SIZE:
                    elem = chemical_symbols[idx + 1]
                    if elem != "X":
                        valid_mask.append(i)

            if len(valid_mask) == 0:
                logger.warning("Generated crystal has no valid elements, skipping.")
                continue

            mask = paddle.to_tensor(valid_mask, dtype=paddle.int64)
            filtered_crystal = {
                "space_group_number": crystal["space_group_number"],
                "conventional_lattice_lengths": crystal["conventional_lattice_lengths"],
                "conventional_lattice_angles": crystal["conventional_lattice_angles"],
                "element_indices": crystal["element_indices"][mask],
                "wyckoff_indices": crystal["wyckoff_indices"][mask],
                "conventional_frac_coords": crystal["conventional_frac_coords"][mask],
            }
            structure = asu_to_pymatgen_structure(
                filtered_crystal, self.wyckoff_geometry
            )

            result.append(
                {
                    "num_atoms": len(structure),
                    "atom_types": list(structure.atomic_numbers),
                    "frac_coords": structure.frac_coords.tolist(),
                    "lengths": list(structure.lattice.parameters[:3]),
                    "angles": list(structure.lattice.parameters[3:]),
                }
            )

        return {"result": result}

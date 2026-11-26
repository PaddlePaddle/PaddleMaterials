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

"""MatInvent diffusion backbones with per-structure denoising losses."""

import pickle

import numpy as np
import paddle
from tqdm import tqdm

from ppmat.models.common.time_embedding import uniform_sample_t
from ppmat.models.diffcsp.diffcsp import DiffCSP
from ppmat.models.diffcsp.diffcsp import d_log_p_wrapped_normal
from ppmat.models.mattergen.mattergen import MatterGen
from ppmat.models.mattergen.mattergen import radius_graph_pbc_ocp
from ppmat.models.mattergen.mattergen import wrapped_normal_loss
from ppmat.schedulers import build_scheduler
from ppmat.utils.crystal import frac_to_cart_coords_with_lattice
from ppmat.utils.crystal import lattice_params_to_matrix_paddle
from ppmat.utils.misc import make_noise_symmetric_preserve_variance
from ppmat.utils.scatter import scatter


class _MatInventTrainingMixin:
    """Bind RL losses and state to the standard Trainer lifecycle."""

    def before_train(self, trainer):
        if self.training_cfg is None:
            return
        from ppmat.models.matinvent.training import MatInventTraining

        if self._training_context is None:
            context = MatInventTraining(trainer, self.training_cfg)
            self._training_context = context
            if self._pending_training_state is not None:
                context.set_state_dict(self._pending_training_state)
                self._pending_training_state = None
        else:
            self._training_context.trainer = trainer
        trainer.metric_modules["matinvent_rl"] = self._training_context

    def forward(self, batch):
        if self.training and self._training_context is not None:
            return self._training_context.forward(batch)
        return self.diffusion_forward(batch)

    def _reduce_denoise_loss(self, loss_dict):
        losses = {name: value.mean() for name, value in loss_dict.items()}
        losses["loss"] = (
            self.coord_loss_weight * losses["loss_coord"]
            + self.lattice_loss_weight * losses["loss_lattice"]
        )
        if "loss_atom_type" in losses:
            losses["loss"] += self.atom_loss_weight * losses["loss_atom_type"]
        return losses

    def state_dict(self, *args, **kwargs):
        state = super().state_dict(*args, **kwargs)
        training_state = (
            self._training_context.state_dict()
            if self._training_context is not None
            else self._pending_training_state
        )
        if training_state is not None:
            state["_matinvent_training"] = paddle.to_tensor(
                np.frombuffer(pickle.dumps(training_state), dtype=np.uint8).copy()
            )
        return state

    def set_state_dict(self, state_dict, use_structured_name=True):
        state = dict(state_dict)
        training_state = state.pop("_matinvent_training", None)
        if self._training_context is not None and training_state is None:
            raise ValueError(
                "RL resume requires a checkpoint with MatInvent training state"
            )
        result = super().set_state_dict(state, use_structured_name)
        if training_state is not None:
            restored = pickle.loads(np.asarray(training_state).tobytes())
            if not self.continuous_timesteps:
                self.coord_scheduler.discrete_sigmas_norm = paddle.to_tensor(
                    restored["coord_sigmas_norm"]
                )
            if self._training_context is not None:
                self._training_context.set_state_dict(restored)
            else:
                self._pending_training_state = restored
        return result


class MatinventDiffCSP(_MatInventTrainingMixin, DiffCSP):
    """Joint MatInvent diffusion over lattices, coordinates and atom types.

    The decoder uses native CSPNet activations and the configured time offset.
    Unused property projections are removed from the unconditional backbone.

    Args:
        decoder_cfg (dict): Decoder layer configuration. See ``CSPNet``.
        lattice_noise_scheduler_cfg (dict): Lattice noise scheduler configuration.
        coord_noise_scheduler_cfg (dict): Coordinate noise scheduler configuration.
        atom_noise_scheduler_cfg (dict, optional): Atom type noise scheduler
            configuration, required exactly when the decoder predicts atom types.
        num_train_timesteps (int): Number of diffusion steps. Defaults to 1000.
        time_dim (int): Time embedding dimension. Defaults to 256.
        lattice_loss_weight (float, optional): Lattice loss weight. Defaults to 1.0.
        coord_loss_weight (float, optional): Coordinate loss weight. Defaults to 1.0.
        atom_loss_weight (float, optional): Atom type loss weight. Defaults to 0.0.
        time_embedding_offset (int, optional): Timestep offset of the reference time
            embedding. Defaults to 0.
        training_cfg (dict, optional): RL recipe initialized by before_train.
    """

    continuous_timesteps = False

    def __init__(
        self,
        decoder_cfg: dict,
        lattice_noise_scheduler_cfg: dict,
        coord_noise_scheduler_cfg: dict,
        atom_noise_scheduler_cfg: dict | None = None,
        num_train_timesteps: int = 1000,
        time_dim: int = 256,
        lattice_loss_weight: float = 1.0,
        coord_loss_weight: float = 1.0,
        atom_loss_weight: float = 0.0,
        time_embedding_offset: int = 0,
        execution_backend: str = "eager",
        runtime_options: dict | None = None,
        training_cfg: dict | None = None,
    ) -> None:
        self.training_cfg = training_cfg
        self._training_context = None
        self._pending_training_state = None
        super().__init__(
            decoder_cfg=decoder_cfg,
            lattice_noise_scheduler_cfg=lattice_noise_scheduler_cfg,
            coord_noise_scheduler_cfg=coord_noise_scheduler_cfg,
            num_train_timesteps=num_train_timesteps,
            time_dim=time_dim,
            lattice_loss_weight=lattice_loss_weight,
            coord_loss_weight=coord_loss_weight,
            execution_backend=execution_backend,
            runtime_options=runtime_options,
        )
        self.atom_scheduler = build_scheduler(atom_noise_scheduler_cfg)
        self.atom_loss_weight = atom_loss_weight
        self.pred_type = bool(self.decoder.pred_type)
        self.num_classes = self.decoder.num_classes
        self.time_embedding_offset = time_embedding_offset
        if self.pred_type != (self.atom_scheduler is not None):
            raise ValueError(
                "atom_noise_scheduler_cfg must be given exactly when the decoder "
                f"predicts atom types, but pred_type={self.pred_type} and "
                f"atom_scheduler={self.atom_scheduler is not None}"
            )

        for index in range(self.decoder.num_layers):
            layer = getattr(self.decoder, f"csp_layer_{index}")
            del layer.prop_mlp

    @property
    def supports_num_atoms_sampling(self):
        return self.pred_type

    def encode_time(self, timesteps):
        return self.time_embedding(timesteps + self.time_embedding_offset)

    def diffusion_forward(self, batch):
        structure_array = batch["structure_array"]
        times = uniform_sample_t(
            structure_array["num_atoms"].shape[0], self.num_train_timesteps
        )
        denoised = self.denoise_with_time(structure_array, times)
        return {"loss_dict": self._reduce_denoise_loss(denoised["loss_dict"])}

    def denoise_with_time(self, structure_array, times, noise_state=None):
        """Corrupt ``structure_array`` at ``times`` and denoise it.

        Return per-structure losses, predictions and the reusable corruption state.
        Passing ``noise_state`` evaluates the same corrupted inputs and targets.
        """
        num_atoms = structure_array["num_atoms"]
        batch_size = num_atoms.shape[0]
        batch_idx = paddle.repeat_interleave(
            paddle.arange(batch_size), repeats=num_atoms
        )
        times_per_atom = times.repeat_interleave(repeats=num_atoms)
        time_emb = self.encode_time(times)

        if noise_state is None:
            lattices = (
                structure_array["lattice"]
                if "lattice" in structure_array
                else lattice_params_to_matrix_paddle(
                    structure_array["lengths"], structure_array["angles"]
                )
            )
            frac_coords = structure_array["frac_coords"]
            rand_l = paddle.randn(shape=lattices.shape, dtype=lattices.dtype)
            rand_x = paddle.randn(shape=frac_coords.shape, dtype=frac_coords.dtype)

            input_lattice = self.lattice_scheduler.add_noise(
                lattices, rand_l, timesteps=times
            )
            input_frac_coords = self.coord_scheduler.add_noise(
                frac_coords, rand_x, timesteps=times_per_atom
            )
            input_frac_coords = input_frac_coords % 1.0

            noise_state = {"lattice_noise": rand_l, "coord_noise": rand_x}
            if self.pred_type:
                gt_onehot = paddle.nn.functional.one_hot(
                    structure_array["atom_types"] - 1, num_classes=self.num_classes
                ).cast("float32")
                rand_t = paddle.randn(shape=gt_onehot.shape, dtype=gt_onehot.dtype)
                input_atom_types = self.atom_scheduler.add_noise(
                    gt_onehot, rand_t, timesteps=times_per_atom
                )
                noise_state["atom_noise"] = rand_t
            else:
                input_atom_types = structure_array["atom_types"] - 1
            noise_state["noisy_inputs"] = {
                "lattice": input_lattice,
                "frac_coords": input_frac_coords,
                "atom_types": input_atom_types,
            }

        noisy_inputs = noise_state["noisy_inputs"]
        decoded = self._decode(
            time_emb,
            noisy_inputs["atom_types"],
            noisy_inputs["frac_coords"],
            noisy_inputs["lattice"],
            num_atoms,
            batch_idx,
        )
        if self.pred_type:
            pred_l, pred_x, pred_t = decoded
        else:
            pred_l, pred_x = decoded

        sigmas = self.coord_scheduler.discrete_sigmas[times_per_atom][:, None]
        sigmas_norm = self.coord_scheduler.discrete_sigmas_norm[times_per_atom][:, None]
        tar_x = d_log_p_wrapped_normal(
            sigmas * noise_state["coord_noise"], sigmas
        ) / paddle.sqrt(x=sigmas_norm)

        loss_lattice = (
            (pred_l - noise_state["lattice_noise"]).square().mean(axis=[1, 2])
        )
        loss_coord = scatter(
            (pred_x - tar_x).square().mean(axis=1), batch_idx, dim=0, reduce="mean"
        )
        pred_dict = {"frac_coords": pred_x, "lattice": pred_l}
        loss_dict = {"loss_coord": loss_coord, "loss_lattice": loss_lattice}
        if self.pred_type:
            loss_type = scatter(
                (pred_t - noise_state["atom_noise"]).square().mean(axis=1),
                batch_idx,
                dim=0,
                reduce="mean",
            )
            loss_dict["loss_atom_type"] = loss_type
            pred_dict["atom_types"] = pred_t
        return {
            "loss_dict": loss_dict,
            "pred_dict": pred_dict,
            "noise_state": noise_state,
        }

    @paddle.no_grad()
    def sample(self, batch_data, num_inference_steps=1000):
        structure_array = batch_data["structure_array"]
        num_atoms = structure_array["num_atoms"]
        batch_size = num_atoms.shape[0]
        batch_idx = paddle.repeat_interleave(
            paddle.arange(batch_size), repeats=num_atoms
        )
        l_t = paddle.randn(shape=[batch_size, 3, 3])
        x_t = paddle.rand(shape=[num_atoms.sum(), 3])
        if self.pred_type:
            t_t = paddle.randn([num_atoms.sum(), self.num_classes])
        else:
            t_t = structure_array["atom_types"] - 1

        self.lattice_scheduler.set_timesteps(num_inference_steps)
        self.coord_scheduler.set_timesteps(num_inference_steps)
        if self.pred_type:
            self.atom_scheduler.set_timesteps(num_inference_steps)

        for lattice_t, coord_t in tqdm(
            zip(self.lattice_scheduler.timesteps, self.coord_scheduler.timesteps),
            total=num_inference_steps,
            desc="Sampling...",
        ):
            time_emb = self.encode_time(
                paddle.ones([batch_size], dtype="int64") * lattice_t
            )
            decoded = self._decode(
                time_emb,
                t_t,
                x_t,
                l_t,
                num_atoms,
                batch_idx,
            )
            if self.pred_type:
                pred_l, pred_x, pred_t = decoded
            else:
                pred_l, pred_x = decoded
            # The reference corrector is deterministic on the last offset timestep.
            corrector_noise = (
                paddle.zeros_like(x_t)
                if self.time_embedding_offset and int(lattice_t) == 0
                else None
            )
            x_t = self.coord_scheduler.step_correct(
                pred_x,
                coord_t,
                x_t,
                noise=corrector_noise,
                timestep_index=lattice_t,
            ).prev_sample

            decoded = self._decode(
                time_emb,
                t_t,
                x_t,
                l_t,
                num_atoms,
                batch_idx,
            )
            if self.pred_type:
                pred_l, pred_x, pred_t = decoded
            else:
                pred_l, pred_x = decoded
            output = self.coord_scheduler.step_pred(
                pred_x,
                coord_t,
                x_t,
                timestep_index=lattice_t,
                previous_timestep_index=self.lattice_scheduler.previous_timestep(
                    lattice_t
                ),
            )
            x_t, x_t_mean = output.prev_sample, output.prev_sample_mean

            l_t = self.lattice_scheduler.step(pred_l, lattice_t, l_t).prev_sample
            if self.pred_type:
                t_t = self.atom_scheduler.step(pred_t, lattice_t, t_t).prev_sample

            x_t = x_t % 1.0

        x_t = x_t_mean % 1.0
        if self.pred_type:
            atom_types = paddle.argmax(t_t, axis=-1) + 1
        else:
            atom_types = structure_array["atom_types"]

        result = []
        start_idx = 0
        for index in range(batch_size):
            end_idx = start_idx + num_atoms[index]
            result.append(
                {
                    "num_atoms": num_atoms[index].tolist(),
                    "atom_types": atom_types[start_idx:end_idx].tolist(),
                    "frac_coords": x_t[start_idx:end_idx].tolist(),
                    "lattice": l_t[index].tolist(),
                }
            )
            start_idx += num_atoms[index]

        return {"result": result}


def _radius_graph_pbc(
    cart_coords, lattice, num_atoms, radius, max_neighbors, max_cell_images
):
    """Build the periodic graph with a contiguous-column PBC tensor."""
    # Transposing a row-major [3, B] tensor yields an [B, 3] PBC whose column
    # slices are contiguous, where the Paddle boolean reduction is correct.
    pbc = paddle.ones([3, int(num_atoms.shape[0])], dtype="bool")
    pbc = pbc.to(cart_coords.place).t()
    edge_index, cell_offsets, num_neighbors_image, _, _ = radius_graph_pbc_ocp(
        pos=cart_coords,
        cell=lattice,
        natoms=num_atoms,
        pbc=pbc,
        radius=radius,
        max_num_neighbors_threshold=max_neighbors,
        max_cell_images_per_dim=max_cell_images,
    )
    return edge_index, cell_offsets, num_neighbors_image


class MatinventMatterGen(_MatInventTrainingMixin, MatterGen):
    """MatterGen whose interaction graph uses the Paddle-safe PBC build."""

    continuous_timesteps = True

    def __init__(
        self,
        decoder_cfg: dict,
        lattice_noise_scheduler_cfg: dict,
        coord_noise_scheduler_cfg: dict,
        atom_noise_scheduler_cfg: dict | None = None,
        num_train_timesteps: int = 1000,
        max_t: float = 1.0,
        time_dim: int = 256,
        lattice_loss_weight: float = 1.0,
        coord_loss_weight: float = 0.1,
        atom_loss_weight: float = 1.0,
        d3pm_hybrid_lambda: float = 0.01,
        execution_backend: str = "eager",
        runtime_options: dict | None = None,
        training_cfg: dict | None = None,
    ):
        self.training_cfg = training_cfg
        self._training_context = None
        self._pending_training_state = None
        super().__init__(
            decoder_cfg=decoder_cfg,
            lattice_noise_scheduler_cfg=lattice_noise_scheduler_cfg,
            coord_noise_scheduler_cfg=coord_noise_scheduler_cfg,
            atom_noise_scheduler_cfg=atom_noise_scheduler_cfg,
            num_train_timesteps=num_train_timesteps,
            max_t=max_t,
            time_dim=time_dim,
            lattice_loss_weight=lattice_loss_weight,
            coord_loss_weight=coord_loss_weight,
            atom_loss_weight=atom_loss_weight,
            d3pm_hybrid_lambda=d3pm_hybrid_lambda,
            execution_backend=execution_backend,
            runtime_options=runtime_options,
        )
        for block in self.model.gemnet.out_blocks:
            for name in (
                "dense_rbf",
                "scale_sum",
                "layers",
                "seq_energy",
                "out_energy",
            ):
                delattr(block, name)

    def diffusion_forward(self, batch):
        structure_array = batch["structure_array"]
        times = self.timestep_sampler(structure_array["num_atoms"].shape[0])
        denoised = self.denoise_with_time(structure_array, times)
        return {"loss_dict": self._reduce_denoise_loss(denoised["loss_dict"])}

    def denoise_with_time(self, structure_array, times, noise_state=None):
        """Return per-structure losses and reuse an explicit corruption state."""
        num_atoms = structure_array["num_atoms"]
        batch_size = num_atoms.shape[0]
        batch_idx = paddle.repeat_interleave(
            paddle.arange(batch_size), repeats=num_atoms
        )
        frac_coords = structure_array["frac_coords"] % 1.0
        if noise_state is None:
            rand_x = paddle.randn(shape=frac_coords.shape, dtype=frac_coords.dtype)
            input_frac_coords = self.coord_scheduler.add_noise(
                frac_coords,
                rand_x,
                timesteps=times,
                batch_idx=batch_idx,
                num_atoms=num_atoms,
            )
            lattices = (
                structure_array["lattice"]
                if "lattice" in structure_array
                else lattice_params_to_matrix_paddle(
                    structure_array["lengths"], structure_array["angles"]
                )
            )
            rand_l = paddle.randn(shape=lattices.shape, dtype=lattices.dtype)
            rand_l = make_noise_symmetric_preserve_variance(rand_l)
            input_lattice = self.lattice_scheduler.add_noise(
                lattices, rand_l, timesteps=times, num_atoms=num_atoms
            )
            input_atom_types = structure_array["atom_types"]
            if self.atom_scheduler is not None:
                noisy_atom_types = self.atom_scheduler.add_noise(
                    input_atom_types - 1, timesteps=times, batch_idx=batch_idx
                )
                input_atom_types = noisy_atom_types + 1
            noise_state = {
                "lattice_noise": rand_l,
                "noisy_inputs": {
                    "frac_coords": input_frac_coords,
                    "lattice": input_lattice,
                    "atom_types": input_atom_types,
                    "num_atoms": num_atoms,
                    "batch": batch_idx,
                },
            }
        atom_types = structure_array["atom_types"]
        noisy_inputs = noise_state["noisy_inputs"]
        predictions = self._denoise(noisy_inputs, times)
        losses = {
            "loss_coord": wrapped_normal_loss(
                corruption=self.coord_scheduler,
                score_model_output=predictions["frac_coords"],
                t=times,
                batch_idx=batch_idx,
                batch_size=batch_size,
                x=frac_coords,
                noisy_x=noisy_inputs["frac_coords"],
                reduce="sum",
                batch=structure_array,
            ),
            "loss_lattice": (predictions["lattice"] + noise_state["lattice_noise"])
            .square()
            .mean(axis=[1, 2]),
        }
        if self.atom_scheduler is not None:
            (
                losses["loss_atom_type"],
                losses["base_loss_atom_type"],
                losses["cross_entropy_atom_type"],
            ) = self.atom_scheduler.compute_loss(
                score_model_output=predictions["atom_types"],
                t=times,
                batch_idx=batch_idx,
                batch_size=batch_size,
                x=atom_types - 1,
                noisy_x=noisy_inputs["atom_types"] - 1,
                reduce="sum",
                d3pm_hybrid_lambda=self.d3pm_hybrid_lambda,
            )
        return {
            "loss_dict": losses,
            "pred_dict": predictions,
            "noise_state": noise_state,
        }

    def _denoise(self, structure_array, times):
        interaction_graph = self._build_interaction_graph(structure_array)
        return self._runtime_denoise(structure_array, times, interaction_graph)

    def _build_interaction_graph(self, structure_array):
        gemnet = self.model.gemnet
        lattice = structure_array["lattice"]
        num_atoms = structure_array["num_atoms"]
        cart_coords = frac_to_cart_coords_with_lattice(
            structure_array["frac_coords"], num_atoms, lattice=lattice
        )
        edge_index, to_jimages, num_bonds = _radius_graph_pbc(
            cart_coords,
            lattice,
            num_atoms,
            gemnet.cutoff,
            gemnet.max_neighbors,
            gemnet.max_cell_images_per_dim,
        )
        if edge_index.shape[1] == 0:
            empty_index = edge_index[0]
            return (
                edge_index,
                num_bonds,
                paddle.empty([0], dtype=lattice.dtype),
                paddle.empty([0, 3], dtype=lattice.dtype),
                empty_index,
                empty_index,
                empty_index,
                empty_index,
                to_jimages,
            )
        # The graph is precomputed here, so the decoder must consume it as is.
        otf_graph = gemnet.otf_graph
        gemnet.otf_graph = False
        try:
            return gemnet.generate_interaction_graph(
                cart_coords, lattice, num_atoms, edge_index, to_jimages, num_bonds
            )
        finally:
            gemnet.otf_graph = otf_graph

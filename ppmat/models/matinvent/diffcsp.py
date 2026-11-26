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

"""MatInvent diffusion semantics on the native DiffCSP backbone."""

import paddle
import paddle.nn as nn
from tqdm import tqdm

from ppmat.models.common.activation import swish
from ppmat.models.common.time_embedding import uniform_sample_t
from ppmat.models.diffcsp.diffcsp import DiffCSP
from ppmat.models.diffcsp.diffcsp import d_log_p_wrapped_normal
from ppmat.schedulers import build_scheduler
from ppmat.utils.crystal import lattice_params_to_matrix_paddle
from ppmat.utils.scatter import scatter


class _SiLU(nn.Layer):
    def forward(self, x):
        return swish(x)


class MatinventDiffCSP(DiffCSP):
    """Joint MatInvent diffusion over lattices, coordinates and atom types.

    The released MatInvent checkpoints were trained with the reference CSPNet
    activation replacement, the time embedding offset and the frozen property
    projections, so those three details are part of the model definition here.

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
    ) -> None:
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

        self.decoder.act_fn = _SiLU()
        for index in range(self.decoder.num_layers):
            layer = getattr(self.decoder, f"csp_layer_{index}")
            for network in (layer.edge_mlp, layer.node_mlp, layer.prop_mlp):
                for position, activation in enumerate(network):
                    if isinstance(activation, nn.Silu):
                        network[position] = self.decoder.act_fn
            # The released conditional checkpoints carry unused property projections.
            for parameter in layer.prop_mlp.parameters():
                parameter.stop_gradient = True

    @property
    def supports_num_atoms_sampling(self):
        return self.pred_type

    def encode_time(self, timesteps):
        return self.time_embedding(timesteps + self.time_embedding_offset)

    def forward(self, batch):
        structure_array = batch["structure_array"]
        times = uniform_sample_t(
            structure_array["num_atoms"].shape[0], self.num_train_timesteps
        )
        denoised = self.denoise_with_time(structure_array, times)
        return {"loss_dict": self._reduce_denoise_loss(denoised["loss_dict"])}

    def denoise_with_time(self, structure_array, times):
        """Corrupt ``structure_array`` at ``times`` and denoise it.

        Per-graph losses and raw predictions are returned so that callers needing
        per-structure weighting share the exact arithmetic of ``forward``.
        """
        num_atoms = structure_array["num_atoms"]
        batch_size = num_atoms.shape[0]
        batch_idx = paddle.repeat_interleave(
            paddle.arange(batch_size), repeats=num_atoms
        )
        times_per_atom = times.repeat_interleave(repeats=num_atoms)
        time_emb = self.encode_time(times)

        if "lattice" in structure_array.keys():
            lattices = structure_array["lattice"]
        else:
            lattices = lattice_params_to_matrix_paddle(
                structure_array["lengths"], structure_array["angles"]
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

        rand_t = None
        if self.pred_type:
            gt_onehot = paddle.nn.functional.one_hot(
                structure_array["atom_types"] - 1, num_classes=self.num_classes
            ).cast("float32")
            rand_t = paddle.randn(shape=gt_onehot.shape, dtype=gt_onehot.dtype)
            input_atom_types = self.atom_scheduler.add_noise(
                gt_onehot, rand_t, timesteps=times_per_atom
            )
        else:
            input_atom_types = structure_array["atom_types"] - 1

        decoded = self._decode(
            time_emb,
            input_atom_types,
            input_frac_coords,
            input_lattice,
            num_atoms,
            batch_idx,
        )
        if self.pred_type:
            pred_l, pred_x, pred_t = decoded
        else:
            pred_l, pred_x = decoded

        sigmas = self.coord_scheduler.discrete_sigmas[times_per_atom][:, None]
        sigmas_norm = self.coord_scheduler.discrete_sigmas_norm[times_per_atom][:, None]
        tar_x = d_log_p_wrapped_normal(sigmas * rand_x, sigmas) / paddle.sqrt(
            x=sigmas_norm
        )

        loss_lattice = (pred_l - rand_l).square().mean(axis=[1, 2])
        loss_coord = scatter(
            (pred_x - tar_x).square().mean(axis=1), batch_idx, dim=0, reduce="mean"
        )
        pred_dict = {"frac_coords": pred_x, "lattice": pred_l}
        loss_dict = {"loss_coord": loss_coord, "loss_lattice": loss_lattice}
        if self.pred_type:
            loss_type = scatter(
                (pred_t - rand_t).square().mean(axis=1), batch_idx, dim=0, reduce="mean"
            )
            loss_dict["loss_atom_type"] = loss_type
            pred_dict["atom_types"] = pred_t
        return {"loss_dict": loss_dict, "pred_dict": pred_dict}

    def _reduce_denoise_loss(self, loss_dict):
        loss_coord = loss_dict["loss_coord"].mean()
        loss_lattice = loss_dict["loss_lattice"].mean()
        loss = (
            self.coord_loss_weight * loss_coord
            + self.lattice_loss_weight * loss_lattice
        )
        if "loss_atom_type" not in loss_dict:
            return {
                "loss": loss,
                "loss_coord": loss_coord,
                "loss_lattice": loss_lattice,
            }

        loss_atom_type = loss_dict["loss_atom_type"].mean()
        return {
            "loss": loss + self.atom_loss_weight * loss_atom_type,
            "loss_coord": loss_coord,
            "loss_lattice": loss_lattice,
            "loss_atom_type": loss_atom_type,
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

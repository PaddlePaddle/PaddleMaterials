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

"""MatInvent diffusion semantics on the unchanged native DiffCSP backbone."""

import copy

import paddle
import paddle.nn as nn
from tqdm import tqdm

from ppmat.models.common.activation import swish
from ppmat.models.common.time_embedding import uniform_sample_t
from ppmat.models.diffcsp.diffcsp import DiffCSP
from ppmat.models.diffcsp.diffcsp import d_log_p_wrapped_normal
from ppmat.utils.crystal import lattice_params_to_matrix_paddle


class _SiLU(nn.Layer):
    def forward(self, x):
        return swish(x)


class MatinventDiffCSP(DiffCSP):
    """Conditional and joint MatInvent diffusion with native runtime ownership."""

    def __init__(
        self,
        decoder_cfg: dict,
        lattice_noise_scheduler_cfg: dict,
        coord_noise_scheduler_cfg: dict,
        num_train_timesteps: int = 1000,
        time_dim: int = 256,
        lattice_loss_weight: float = 1.0,
        coord_loss_weight: float = 1.0,
        atom_loss_weight: float = 0.0,
        time_embedding_offset: int = 0,
        execution_backend: str = "eager",
        runtime_options: dict | None = None,
    ):
        lattice_noise_scheduler_cfg = copy.deepcopy(lattice_noise_scheduler_cfg)
        coord_noise_scheduler_cfg = copy.deepcopy(coord_noise_scheduler_cfg)
        if lattice_noise_scheduler_cfg["__class_name__"] == "DDPMScheduler":
            lattice_noise_scheduler_cfg["__class_name__"] = "MatinventDDPMScheduler"
        if coord_noise_scheduler_cfg["__class_name__"] == "ScoreSdeVeSchedulerWrapped":
            coord_noise_scheduler_cfg["__class_name__"] = "MatinventSdeVeScheduler"
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
        self.atom_loss_weight = atom_loss_weight
        self.pred_type = bool(self.decoder.pred_type)
        self.num_classes = self.decoder.num_classes
        self.time_embedding_offset = time_embedding_offset
        self.decoder.act_fn = _SiLU()
        for index in range(self.decoder.num_layers):
            layer = getattr(self.decoder, f"csp_layer_{index}")
            for network in (layer.edge_mlp, layer.node_mlp, layer.prop_mlp):
                for position, activation in enumerate(network):
                    if isinstance(activation, nn.Silu):
                        network[position] = self.decoder.act_fn
            for parameter in layer.prop_mlp.parameters():
                parameter.stop_gradient = True

    def encode_time(self, timesteps):
        return self.time_embedding(timesteps + self.time_embedding_offset)

    def forward(self, batch, **kwargs):

        structure_array = batch["structure_array"]
        batch_size = structure_array["num_atoms"].shape[0]
        batch_idx = paddle.repeat_interleave(
            paddle.arange(batch_size), repeats=structure_array["num_atoms"]
        )

        times = uniform_sample_t(batch_size, self.num_train_timesteps)
        times_per_atom = times.repeat_interleave(repeats=structure_array["num_atoms"])

        time_emb = self.encode_time(times)

        if "lattice" in structure_array.keys():
            lattices = structure_array["lattice"]
        else:
            lattices = lattice_params_to_matrix_paddle(
                structure_array["lengths"], structure_array["angles"]
            )

        frac_coords = structure_array["frac_coords"]
        rand_l, rand_x = paddle.randn(
            shape=lattices.shape, dtype=lattices.dtype
        ), paddle.randn(shape=frac_coords.shape, dtype=frac_coords.dtype)

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
                structure_array["atom_types"] - 1,
                num_classes=self.num_classes,
            ).cast("float32")
            rand_t = paddle.randn(shape=gt_onehot.shape, dtype=gt_onehot.dtype)
            alpha_cumprod = self.lattice_scheduler.alphas_cumprod[times_per_atom].cast(
                "float32"
            )[:, None]
            input_atom_types = (
                paddle.sqrt(alpha_cumprod) * gt_onehot
                + paddle.sqrt(1.0 - alpha_cumprod) * rand_t
            )
        else:
            input_atom_types = structure_array["atom_types"] - 1

        decoded = self._decode(
            time_emb,
            input_atom_types,
            input_frac_coords,
            input_lattice,
            structure_array["num_atoms"],
            batch_idx,
        )
        if self.pred_type:
            pred_l, pred_x, pred_t = decoded
        else:
            pred_l, pred_x = decoded

        sigmas_per_atom = self.coord_scheduler.discrete_sigmas[times_per_atom][:, None]
        sigmas_norm_per_atom = self.coord_scheduler.discrete_sigmas_norm[
            times_per_atom
        ][:, None]

        tar_x = d_log_p_wrapped_normal(
            sigmas_per_atom * rand_x, sigmas_per_atom
        ) / paddle.sqrt(x=sigmas_norm_per_atom)

        loss_lattice = paddle.nn.functional.mse_loss(input=pred_l, label=rand_l)
        loss_coord = paddle.nn.functional.mse_loss(input=pred_x, label=tar_x)
        loss = (
            self.lattice_loss_weight * loss_lattice
            + self.coord_loss_weight * loss_coord
        )
        loss_dict = {
            "loss": loss,
            "loss_lattice": loss_lattice,
            "loss_coord": loss_coord,
        }
        if self.pred_type:
            loss_atom_type = paddle.nn.functional.mse_loss(pred_t, rand_t)
            loss = loss + self.atom_loss_weight * loss_atom_type
            loss_dict["loss"] = loss
            loss_dict["loss_atom_type"] = loss_atom_type

        return {
            "loss_dict": loss_dict,
        }

    @paddle.no_grad()
    def sample(self, batch_data, num_inference_steps=1000, **kwargs):
        structure_array = batch_data["structure_array"]
        batch_size = structure_array["num_atoms"].shape[0]
        batch_idx = paddle.repeat_interleave(
            paddle.arange(batch_size), repeats=structure_array["num_atoms"]
        )
        l_T, x_T = paddle.randn(shape=[batch_size, 3, 3]), paddle.rand(
            shape=[structure_array["num_atoms"].sum(), 3]
        )
        l_t, x_t = l_T, x_T
        if self.pred_type:
            t_t = paddle.randn([structure_array["num_atoms"].sum(), self.num_classes])
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
                structure_array["num_atoms"],
                batch_idx,
            )
            if self.pred_type:
                pred_l, pred_x, pred_t = decoded
            else:
                pred_l, pred_x = decoded
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
                structure_array["num_atoms"],
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

            l_t = self.lattice_scheduler.step(
                pred_l,
                lattice_t,
                l_t,
            ).prev_sample
            if self.pred_type:
                t_t = self.lattice_scheduler.step(pred_t, lattice_t, t_t).prev_sample

            x_t = x_t % 1.0

        x_t = x_t_mean % 1.0
        if self.pred_type:
            atom_types_out = paddle.argmax(t_t, axis=-1) + 1
        else:
            atom_types_out = structure_array["atom_types"]

        start_idx = 0
        result = []
        for i in range(batch_size):
            end_idx = start_idx + structure_array["num_atoms"][i]
            result.append(
                {
                    "num_atoms": structure_array["num_atoms"][i].tolist(),
                    "atom_types": atom_types_out[start_idx:end_idx].tolist(),
                    "frac_coords": x_t[start_idx:end_idx].tolist(),
                    "lattice": l_t[i].tolist(),
                }
            )
            start_idx += structure_array["num_atoms"][i]

        return {"result": result}

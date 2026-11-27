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

from typing import Optional

import numpy as np
import paddle
import paddle.nn as nn
from tqdm import tqdm

from ppmat.models.common import initializer
from ppmat.models.common.runtime import RuntimeMixin
from ppmat.models.common.runtime import runtime_boundary
from ppmat.models.common.time_embedding import SinusoidalTimeEmbeddings
from ppmat.models.common.time_embedding import UniformTimestepSampler
from ppmat.models.miad.cspnet import MiADCSPNet
from ppmat.schedulers import build_scheduler
from ppmat.utils.crystal import lattices_to_params_shape_numpy


def _extract_x0(batch, mirage_num_atoms=None):
    """Build the MiAD diffusion state from a structure batch."""
    structure_array = batch["structure_array"]
    num_atoms = paddle.to_tensor(structure_array["num_atoms"], dtype="int64").flatten()
    batch_size = num_atoms.shape[0]
    frac_coords = paddle.to_tensor(structure_array["frac_coords"], dtype="float32")
    atom_types = paddle.to_tensor(structure_array["atom_types"], dtype="int64")
    lattices = paddle.to_tensor(structure_array["lattice"], dtype="float32")

    if mirage_num_atoms is not None:
        padded_frac = []
        padded_types = []
        padded_num_atoms = []
        start = 0
        for i in range(batch_size):
            n = int(num_atoms[i])
            n_m = max(int(mirage_num_atoms), n)
            padded_frac.append(frac_coords[start : start + n])
            padded_types.append(atom_types[start : start + n])
            if n_m > n:
                padded_frac.append(paddle.rand([n_m - n, 3], dtype=frac_coords.dtype))
                padded_types.append(paddle.zeros([n_m - n], dtype="int64"))
            padded_num_atoms.append(n_m)
            start += n
        frac_coords = paddle.concat(padded_frac)
        atom_types = paddle.concat(padded_types)
        num_atoms = paddle.to_tensor(padded_num_atoms, dtype="int64")

    return {
        "lattice": lattices,
        "frac_coords": frac_coords,
        "atom_types": atom_types,
        "num_atoms": num_atoms,
        "batch_idx": paddle.repeat_interleave(
            paddle.arange(batch_size), repeats=num_atoms
        ),
        "batch_size": batch_size,
    }


class MiAD(RuntimeMixin, paddle.nn.Layer):
    """Mirage Atom Diffusion model.

    ``sample`` treats the requested atom count as generation capacity and
    filters type-0 mirage atoms from the output.
    """

    supports_num_atoms_sampling = True

    # Class index 0 is the mirage type.
    _MIRAGE_TYPE = 0
    # Hydrogen fallback when every sampled atom stays mirage.
    _FALLBACK_ATOMIC_NUM = 1
    _TIME_EMBEDDING_SCALE = 1000

    def __init__(
        self,
        hidden_dim: int = 128,
        latent_dim: int = 256,
        num_layers: int = 4,
        act_fn: str = "silu",
        dis_emb: str = "sin",
        num_freqs: int = 10,
        edge_style: str = "fc",
        ln: bool = False,
        ip: bool = True,
        smooth: bool = True,
        pred_type: bool = True,
        num_classes: int = 100,
        mirage_num_atoms: int = 25,
        num_train_timesteps: int = 1000,
        cont_time: bool = False,
        eps: float = 1e-3,
        time_dim: int = 256,
        lattice_noise_scheduler_cfg: Optional[dict] = None,
        coord_noise_scheduler_cfg: Optional[dict] = None,
        atom_noise_scheduler_cfg: Optional[dict] = None,
        execution_backend: str = "eager",
        runtime_options: Optional[dict] = None,
    ) -> None:
        super().__init__()
        self._init_runtime(execution_backend, runtime_options)

        self.mirage_num_atoms = mirage_num_atoms
        self.decoder = MiADCSPNet(
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            num_layers=num_layers,
            act_fn=act_fn,
            dis_emb=dis_emb,
            num_freqs=num_freqs,
            edge_style=edge_style,
            ln=ln,
            ip=ip,
            smooth=smooth,
            pred_type=pred_type,
            num_classes=num_classes,
        )

        self.num_train_timesteps = num_train_timesteps
        self.cont_time = cont_time
        self.eps = float(eps)
        self.time_embedding = SinusoidalTimeEmbeddings(time_dim)
        self.timestep_sampler = UniformTimestepSampler(
            min_t=self.eps, max_t=self.num_train_timesteps - 1
        )

        self.lattice_scheduler = build_scheduler(lattice_noise_scheduler_cfg)
        self.coord_scheduler = build_scheduler(coord_noise_scheduler_cfg)
        self.atom_scheduler = build_scheduler(atom_noise_scheduler_cfg)

        if self.atom_scheduler.num_types != self.decoder.num_classes:
            raise ValueError("Type diffusion num_types must match decoder num_classes")
        schedule_lengths = (
            len(self.lattice_scheduler.betas),
            len(self.coord_scheduler.discrete_sigmas),
            len(self.atom_scheduler.cumprod_Q_t),
        )
        for length in schedule_lengths:
            if length != self.num_train_timesteps:
                raise ValueError(
                    "All diffusion schedules must match num_train_timesteps"
                )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            initializer.linear_init_(m)
        elif isinstance(m, nn.Embedding):
            initializer.normal_(m.weight)

    def _decode(
        self,
        time_emb,
        atom_types,
        frac_coords,
        lattices,
        num_atoms,
        node2graph,
    ):
        edges, frac_diff = self.decoder.gen_edges(num_atoms, frac_coords)
        return self._runtime_decode(
            time_emb,
            atom_types,
            frac_coords,
            lattices,
            num_atoms,
            node2graph,
            edges,
            frac_diff,
        )

    @runtime_boundary("denoise_step")
    def _runtime_decode(
        self,
        time_emb,
        atom_types,
        frac_coords,
        lattices,
        num_atoms,
        node2graph,
        edges,
        frac_diff,
    ):
        return self.decoder.forward_with_edges(
            time_emb,
            atom_types,
            frac_coords,
            lattices,
            num_atoms,
            node2graph,
            edges,
            frac_diff,
        )

    def _time_sample(self, state):
        t = self.timestep_sampler(state["batch_size"])
        if not self.cont_time:
            t = t.round().cast("int64")
        t_per_atom = t.repeat_interleave(state["num_atoms"])
        return [t, t_per_atom]

    def _forward_step_sample(self, state, t):
        l0, f0, a0 = state["lattice"], state["frac_coords"], state["atom_types"]
        t0_idx = t[0].cast("int64")
        t1_idx = t[1].cast("int64")

        lat_noise = paddle.randn(l0.shape)
        lt = self.lattice_scheduler.add_noise(l0, lat_noise, t0_idx)

        frac_noise = paddle.randn(f0.shape)
        ft = self.coord_scheduler.add_noise(f0, frac_noise, t1_idx) % 1.0

        at = self.atom_scheduler.add_noise(a0, t[1])

        return [lt, ft, at], lat_noise, frac_noise

    def _reverse_step_sample(self, xt, t, state):
        lt, ft, at = xt
        _, f_pred, _ = self._model_prediction(xt, t, state)
        ft_05 = self.coord_scheduler.step_correct_indexed(f_pred, t[1], ft).prev_sample
        xt_05 = [lt, ft_05, at]
        l_pred, f_pred, a_pred = self._model_prediction(xt_05, t, state)
        lt_1 = self.lattice_scheduler.step(
            l_pred, t[0].cast("int64")[0], lt
        ).prev_sample
        ft_1 = self.coord_scheduler.step_pred_indexed(f_pred, t[1], ft_05).prev_sample

        at_1 = self.atom_scheduler.step(a_pred, at, t[1])
        return [lt_1, ft_1, at_1]

    def _model_prediction(self, xt, t, state):
        lt, ft, at = xt
        time_emb = self.time_embedding(
            self._TIME_EMBEDDING_SCALE * (t[0] / self.num_train_timesteps) + 1
        )
        return self._decode(
            time_emb, at, ft, lt, state["num_atoms"], state["batch_idx"]
        )

    def forward(self, batch: dict) -> dict:
        labels = {
            key: value for key, value in batch.items() if key != "structure_array"
        }
        state = _extract_x0(batch, mirage_num_atoms=self.mirage_num_atoms)
        t = self._time_sample(state)
        xt, lat_noise, frac_noise = self._forward_step_sample(state, t)
        prediction = self._model_prediction(xt, t, state)

        loss_lattice = paddle.nn.functional.mse_loss(
            input=prediction[0], label=lat_noise
        )

        t_idx = t[1].cast("int64")
        normed_score = self.coord_scheduler.score_target(frac_noise, t_idx)
        coord_err = (
            paddle.nn.functional.mse_loss(
                input=prediction[1], label=normed_score, reduction="none"
            )
            .reshape([-1, 3])
            .mean(axis=1)
        )
        # Mask type-0 mirage atoms from the coordinate loss and rescale.
        mask = (state["atom_types"] != self._MIRAGE_TYPE).cast(coord_err.dtype)
        coef = mask.shape[0] / mask.sum().clip(min=1)
        loss_coord = (coord_err * mask * coef).mean()

        loss_type = self.atom_scheduler.compute_loss(
            prediction[2],
            xt[2],
            self.atom_scheduler.to_domain(state["atom_types"]),
            t[1],
        ).mean()

        loss = loss_lattice + loss_coord + loss_type
        return {
            "loss_dict": {
                "loss": loss,
                "loss_lattice": loss_lattice,
                "loss_coord": loss_coord,
                "loss_type": loss_type,
            },
            "label_dict": labels,
        }

    @paddle.no_grad()
    def sample(
        self, batch_data: dict, num_inference_steps: Optional[int] = None
    ) -> dict:
        if num_inference_steps is not None and (
            type(num_inference_steps) is not int
            or num_inference_steps != self.num_train_timesteps
        ):
            raise ValueError(
                f"MiAD requires num_inference_steps={self.num_train_timesteps} to "
                "preserve the trained diffusion schedules; respacing is unsupported"
            )

        structure_array = batch_data["structure_array"]
        num_atoms = paddle.to_tensor(
            structure_array["num_atoms"], dtype="int64"
        ).flatten()
        batch_size = num_atoms.shape[0]
        state = {
            "num_atoms": num_atoms,
            "batch_idx": paddle.repeat_interleave(
                paddle.arange(batch_size), repeats=num_atoms
            ),
            "batch_size": batch_size,
        }
        total = int(num_atoms.sum())

        prior_at = self.atom_scheduler.prior_sampling(
            (total, self.atom_scheduler.num_types)
        )
        xt = [
            paddle.randn([state["batch_size"], 3, 3], dtype="float32"),
            paddle.rand([total, 3], dtype="float32"),
            prior_at,
        ]
        for t_val in tqdm(
            range(self.num_train_timesteps - 1, -1, -1),
            total=self.num_train_timesteps,
            desc="Sampling...",
        ):
            t = paddle.full([state["batch_size"]], t_val, dtype="float32")
            t = [t, t.repeat_interleave(state["num_atoms"])]
            xt = self._reverse_step_sample(xt, t, state)
        lattices, frac_coords, atom_types = xt
        atom_types = self.atom_scheduler.from_domain(atom_types)

        result = []
        start_idx = 0
        for i in range(state["batch_size"]):
            n = int(num_atoms[i])
            lat_np = lattices[i].numpy()
            fc_np = frac_coords[start_idx : start_idx + n].numpy()
            at_np = atom_types[start_idx : start_idx + n].numpy()
            start_idx += n
            valid_mask = at_np != self._MIRAGE_TYPE
            if valid_mask.any():
                at_np = at_np[valid_mask]
                fc_np = fc_np[valid_mask]
                n = int(valid_mask.sum())
            else:
                # All-mirage fallback keeps the structure non-empty.
                at_np = np.full_like(at_np, self._FALLBACK_ATOMIC_NUM)
            lengths, angles = lattices_to_params_shape_numpy(lat_np.reshape(1, 3, 3))
            result.append(
                {
                    "num_atoms": n,
                    "atom_types": at_np.tolist(),
                    "frac_coords": fc_np.tolist(),
                    "lattice": lat_np.tolist(),
                    "lengths": lengths.flatten().tolist(),
                    "angles": angles.flatten().tolist(),
                }
            )

        return {"result": result}

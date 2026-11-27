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

from typing import Optional

import numpy as np
import paddle

from ppmat.models.common.runtime import RuntimeMixin
from ppmat.models.common.runtime import runtime_boundary
from ppmat.models.common.time_embedding import SinusoidalTimeEmbeddings
from ppmat.models.common.time_embedding import UniformTimestepSampler
from ppmat.models.miad.cspnet import MiADCSPNet
from ppmat.schedulers import build_scheduler
from ppmat.utils.crystal import lattices_to_params_shape_numpy


def _extract_x0(batch, mirage_num_atoms=None):
    """Normalize a batch into the MiAD diffusion state without mutating it."""
    if "x0" in batch:
        return {
            "x0": batch["x0"],
            "num_atoms": batch["num_atoms"],
            "batch_idx": batch["batch_idx"],
            "batch_size": batch["batch_size"],
        }
    structure_array = batch.get("structure_array", batch)
    num_atoms_np = np.asarray(structure_array["num_atoms"], dtype="int64").flatten()
    batch_size = len(num_atoms_np)
    total_atoms = int(num_atoms_np.sum())
    frac_coords = paddle.to_tensor(
        structure_array["frac_coords"], dtype="float32"
    ).reshape(total_atoms, 3)
    atom_types = paddle.to_tensor(structure_array["atom_types"], dtype="int64").reshape(
        total_atoms
    )
    lattice = paddle.to_tensor(structure_array["lattice"], dtype="float32").reshape(
        batch_size, 3, 3
    )
    if mirage_num_atoms is not None:
        # Pad every crystal to mirage_num_atoms with type-0 mirage atoms.
        padded_frac = []
        padded_types = []
        padded_num_atoms = []
        offset = 0
        for n in num_atoms_np:
            n = int(n)
            n_m = max(int(mirage_num_atoms), n)
            padded_frac.append(frac_coords[offset : offset + n])
            padded_types.append(atom_types[offset : offset + n])
            if n_m > n:
                padded_frac.append(
                    paddle.to_tensor(
                        np.random.rand(n_m - n, 3).astype("float32"),
                        place=frac_coords.place,
                    )
                )
                padded_types.append(paddle.zeros([n_m - n], dtype="int64"))
            padded_num_atoms.append(n_m)
            offset += n
        frac_coords = paddle.concat(padded_frac)
        atom_types = paddle.concat(padded_types)
        num_atoms_np = np.array(padded_num_atoms, dtype="int64")
    num_atoms = paddle.to_tensor(num_atoms_np)
    return {
        "x0": [lattice, frac_coords, atom_types],
        "num_atoms": num_atoms,
        "batch_idx": paddle.repeat_interleave(
            paddle.arange(batch_size), repeats=num_atoms
        ),
        "batch_size": batch_size,
    }


class MiAD(RuntimeMixin, paddle.nn.Layer):
    """Mirage Atom Diffusion model.

    ``sample`` uses the requested atom count as generation capacity and
    filters type-0 mirage atoms from the output.
    """

    supports_num_atoms_sampling = True

    # Official MAX_ATOMIC_NUM; class index 0 is the mirage type.
    _MAX_ATOMIC_NUM = 100
    _MIRAGE_TYPE = 0
    # Hydrogen fallback when every sampled atom is mirage.
    _FALLBACK_ATOMIC_NUM = 1
    # Official timestep scale used by the sinusoidal embedding.
    _TIME_EMBEDDING_SCALE = 1000

    def __init__(
        self,
        model_cfg: dict,
        diffusion_cfg: dict,
        execution_backend: str = "eager",
        runtime_options: Optional[dict] = None,
    ) -> None:
        super().__init__()
        self._init_runtime(execution_backend, runtime_options)

        self.mirage_num_atoms = model_cfg.get("mirage_num_atoms")
        model_cfg = dict(model_cfg)
        # max_atoms (MiAD naming) maps to num_classes (CSPNet naming)
        model_cfg.setdefault(
            "num_classes", model_cfg.pop("max_atoms", self._MAX_ATOMIC_NUM)
        )
        # prop_dim/pred_scalar excluded: MiAD has no property-guided branches
        cspnet_keys = {
            "hidden_dim",
            "latent_dim",
            "num_layers",
            "act_fn",
            "dis_emb",
            "num_freqs",
            "edge_style",
            "ln",
            "ip",
            "smooth",
            "pred_type",
            "num_classes",
        }
        unknown_keys = model_cfg.keys() - cspnet_keys - {"mirage_num_atoms"}
        if unknown_keys:
            raise TypeError(f"Unknown MiAD model parameters: {sorted(unknown_keys)}")
        cspnet_kwargs = {k: v for k, v in model_cfg.items() if k in cspnet_keys}
        self.decoder = MiADCSPNet(**cspnet_kwargs)

        self.config = dict(diffusion_cfg)
        diffusion_keys = {
            "cont_time",
            "num_steps",
            "time_embed_dim",
            "eps",
            "lattice_noise_scheduler_cfg",
            "coord_noise_scheduler_cfg",
            "atom_noise_scheduler_cfg",
        }
        unknown_keys = self.config.keys() - diffusion_keys
        if unknown_keys:
            raise TypeError(
                f"Unknown MiAD diffusion parameters: {sorted(unknown_keys)}"
            )
        self.cont_time = self.config["cont_time"]
        self.num_steps = self.config["num_steps"]
        if type(self.num_steps) is not int or self.num_steps < 1:
            raise ValueError("num_steps must be a positive integer")
        # Official default eps=1e-3.
        self.eps = float(self.config.get("eps", 1e-3))
        time_embed_dim = self.config.get("time_embed_dim", 256)
        latent_dim = (
            self.decoder.atom_latent_emb.weight.shape[0]
            - self.decoder.node_embedding.weight.shape[1]
        )
        if (
            type(time_embed_dim) is not int
            or time_embed_dim < 4
            or time_embed_dim % 2
            or time_embed_dim != latent_dim
        ):
            raise ValueError(
                "time_embed_dim must be even, at least 4, and match latent_dim"
            )
        self.time_embedding = SinusoidalTimeEmbeddings(time_embed_dim)
        self.timestep_sampler = UniformTimestepSampler(
            min_t=self.eps, max_t=self.num_steps - 1
        )

        self.lat_scheduler = build_scheduler(self.config["lattice_noise_scheduler_cfg"])
        self.frac_scheduler = build_scheduler(self.config["coord_noise_scheduler_cfg"])
        self.type_diffusion = build_scheduler(self.config["atom_noise_scheduler_cfg"])
        if self.type_diffusion.num_types != self.decoder.num_classes:
            raise ValueError("Type diffusion num_types must match decoder num_classes")
        schedule_lengths = (
            len(self.lat_scheduler.betas),
            len(self.frac_scheduler.discrete_sigmas),
            len(self.type_diffusion.cumprod_Q_t),
        )
        if any(length != self.num_steps for length in schedule_lengths):
            raise ValueError("All diffusion schedules must match num_steps")

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

    def _forward_step_sample(self, x0, t):
        l0, f0, a0 = x0
        t0_idx = t[0].cast("int64")
        t1_idx = t[1].cast("int64")

        lat_noise = paddle.randn(l0.shape)
        lt = self.lat_scheduler.add_noise(l0, lat_noise, t0_idx)

        frac_noise = paddle.randn(f0.shape)
        ft = self.frac_scheduler.add_noise(f0, frac_noise, t1_idx) % 1.0

        at = self.type_diffusion.add_noise(a0, t[1])

        return [lt, ft, at], lat_noise, frac_noise

    def _reverse_step_sample(self, xt, t, state):
        lt, ft, at = xt
        _, f_pred, _ = self._model_prediction(xt, t, state)
        ft_05 = self.frac_scheduler.step_correct_indexed(f_pred, t[1], ft).prev_sample
        xt_05 = [lt, ft_05, at]
        l_pred, f_pred, a_pred = self._model_prediction(xt_05, t, state)
        lt_1 = self.lat_scheduler.step(l_pred, t[0].cast("int64")[0], lt).prev_sample
        ft_1 = self.frac_scheduler.step_pred_indexed(f_pred, t[1], ft_05).prev_sample

        at_1 = self.type_diffusion.step(a_pred, at, t[1])
        return [lt_1, ft_1, at_1]

    def _model_prediction(self, xt, t, state):
        lt, ft, at = xt
        time_emb = self.time_embedding(
            self._TIME_EMBEDDING_SCALE * (t[0] / self.num_steps) + 1
        )
        return self._decode(
            time_emb, at, ft, lt, state["num_atoms"], state["batch_idx"]
        )

    def forward(self, batch: dict) -> dict:
        """Return diffusion loss and labels; use streaming metrics for generation.

        Per-property compute_metric_func_dict is not applicable.
        """
        labels = {
            key: value for key, value in batch.items() if key != "structure_array"
        }
        state = _extract_x0(batch, mirage_num_atoms=self.mirage_num_atoms)
        t = self._time_sample(state)
        xt, lat_noise, frac_noise = self._forward_step_sample(state["x0"], t)
        prediction = self._model_prediction(xt, t, state)

        loss_lat = (
            ((prediction[0] - lat_noise) ** 2).reshape([-1, 9]).mean(axis=1).mean()
        )

        t_idx = t[1].cast("int64")
        normed_score = self.frac_scheduler.score_target(frac_noise, t_idx)
        loss_frac = ((prediction[1] - normed_score) ** 2).reshape([-1, 3]).mean(axis=1)
        # Mask type-0 mirage atoms from the coordinate loss and rescale.
        mask = (state["x0"][2] != 0).cast(loss_frac.dtype)
        coef = mask.shape[0] / mask.sum().clip(min=1)
        loss_frac = (loss_frac * mask * coef).mean()

        loss_type = self.type_diffusion.compute_loss(
            prediction[2],
            xt[2],
            self.type_diffusion.to_domain(state["x0"][2]),
            t[1],
        ).mean()
        loss = loss_lat + loss_frac + loss_type

        return {
            "loss_dict": {"loss": loss},
            "label_dict": labels,
        }

    @paddle.no_grad()
    def sample(
        self, batch_data: dict, num_inference_steps: Optional[int] = None
    ) -> dict:
        if num_inference_steps is not None and (
            type(num_inference_steps) is not int
            or num_inference_steps != self.num_steps
        ):
            raise ValueError(
                f"MiAD requires num_inference_steps={self.num_steps} to preserve "
                "the trained diffusion schedules; timestep respacing is unsupported"
            )
        if "batch_idx" in batch_data:
            state = {
                "num_atoms": batch_data["num_atoms"],
                "batch_idx": batch_data["batch_idx"],
                "batch_size": int(batch_data["batch_size"]),
            }
        else:
            structure_array = batch_data.get("structure_array", batch_data)
            num_atoms = structure_array["num_atoms"].flatten().cast("int64")
            batch_size = num_atoms.shape[0]
            state = {
                "num_atoms": num_atoms,
                "batch_idx": paddle.repeat_interleave(
                    paddle.arange(batch_size), repeats=num_atoms
                ),
                "batch_size": int(batch_size),
            }

        total = int(state["num_atoms"].sum())
        prior_at = self.type_diffusion.prior_sampling(
            (total, self.type_diffusion.num_types)
        )
        xt = [
            paddle.randn([state["batch_size"], 3, 3], dtype="float32"),
            paddle.rand([total, 3], dtype="float32"),
            prior_at,
        ]
        for t_val in range(self.num_steps - 1, -1, -1):
            t = paddle.full([state["batch_size"]], t_val, dtype="float32")
            t = [t, t.repeat_interleave(state["num_atoms"])]
            xt = self._reverse_step_sample(xt, t, state)
        lattices, frac_coords, atom_types = xt
        atom_types = self.type_diffusion.from_domain(atom_types)

        result = []
        num_atoms = state["num_atoms"]
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

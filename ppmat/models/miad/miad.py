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

import numpy as np
import paddle
import paddle.nn as nn

from ppmat.models.common.runtime import RuntimeMixin
from ppmat.models.common.runtime import runtime_boundary
from ppmat.models.common.time_embedding import SinusoidalTimeEmbeddings
from ppmat.models.common.time_embedding import UniformTimestepSampler
from ppmat.models.diffcsp.diffcsp import CSPNet
from ppmat.schedulers import build_scheduler
from ppmat.schedulers.scheduling_sde_ve import d_log_p_wrapped_normal
from ppmat.utils import logger
from ppmat.utils.crystal import lattices_to_params_shape_numpy


class MiADCSPNet(CSPNet):
    """CSPNet with block-diagonal edge generation (avoids GPU crashes) and
    without the ``prop_mlp`` sub-layers absent from the MiAD checkpoint.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for i in range(self.num_layers):
            delattr(getattr(self, "csp_layer_%d" % i), "prop_mlp")

    def gen_edges(self, num_atoms, frac_coords):
        lis = [paddle.ones([int(n), int(n)], dtype="int64") for n in num_atoms]
        fc_graph = paddle.block_diag(lis)
        fc_edges = paddle.nonzero(fc_graph).t()
        return fc_edges, frac_coords[fc_edges[1]] - frac_coords[fc_edges[0]]


def _to_numpy(x):
    if hasattr(x, "numpy"):
        return x.numpy()
    return np.asarray(x)


def _build_batch_idx(num_atoms_np):
    """Per-atom batch indices from per-crystal atom counts."""
    num_atoms_np = np.asarray(num_atoms_np).flatten().astype("int64")
    batch_idx_np = np.concatenate(
        [np.full(int(n), i) for i, n in enumerate(num_atoms_np)]
    ).astype("int64")
    return batch_idx_np, len(num_atoms_np)


def _parse_num_atoms_to_per_crystal(num_atoms_data):
    """Convert per-crystal atom counts to an int64 tensor and numpy array."""
    num_atoms_np = num_atoms_data.numpy().flatten().astype("int64")
    return paddle.to_tensor(num_atoms_np), num_atoms_np


def _extract_x0(batch, mirage_num_atoms=None):
    if "x0" in batch:
        return batch
    sa = batch.get("structure_array", batch)
    num_atoms_np = _to_numpy(sa["num_atoms"]).flatten().astype("int64")
    frac_coords_np = _to_numpy(sa["frac_coords"]).astype("float32")
    atom_types_np = _to_numpy(sa["atom_types"]).flatten().astype("int64")
    lattice_np = _to_numpy(sa["lattice"]).astype("float32")
    if lattice_np.ndim == 3:
        if lattice_np.shape[0] == 1:
            lattice_np = lattice_np.reshape(3, 3)
        else:
            lattice_np = lattice_np.reshape(-1, 3)
    batch_size = len(num_atoms_np)
    total_atoms = int(num_atoms_np.sum())
    if frac_coords_np.shape[0] != total_atoms or atom_types_np.shape[0] != total_atoms:
        raise ValueError(
            f"frac_coords/atom_types length ({frac_coords_np.shape[0]}/"
            f"{atom_types_np.shape[0]}) does not match total atoms ({total_atoms})"
        )
    if lattice_np.ndim != 2 or lattice_np.shape != (batch_size * 3, 3):
        raise ValueError(
            f"lattice must be ({batch_size * 3}, 3) or ({batch_size}, 3, 3), "
            f"got {lattice_np.shape}"
        )
    if mirage_num_atoms is not None:
        # Pad every crystal to mirage_num_atoms with type-0 mirage atoms.
        padded_frac = []
        padded_types = []
        padded_num_atoms = []
        offset = 0
        for i, n in enumerate(num_atoms_np):
            n = int(n)
            n_m = max(int(mirage_num_atoms), n)
            padded_frac.append(frac_coords_np[offset : offset + n])
            padded_types.append(atom_types_np[offset : offset + n])
            if n_m > n:
                padded_frac.append(np.random.rand(n_m - n, 3).astype("float32"))
                padded_types.append(np.zeros(n_m - n, dtype="int64"))
            padded_num_atoms.append(n_m)
            offset += n
        frac_coords_np = np.concatenate(padded_frac)
        atom_types_np = np.concatenate(padded_types)
        num_atoms_np = np.array(padded_num_atoms, dtype="int64")
        total_atoms = int(num_atoms_np.sum())
        batch_size = len(num_atoms_np)
    batch_idx_np, _ = _build_batch_idx(num_atoms_np)
    batch["x0"] = [
        paddle.to_tensor(lattice_np.reshape(batch_size, 3, 3)),
        paddle.to_tensor(frac_coords_np.reshape(total_atoms, 3)),
        paddle.to_tensor(atom_types_np.reshape(total_atoms)),
    ]
    batch["num_atoms"] = paddle.to_tensor(num_atoms_np)
    batch["batch_idx"] = paddle.to_tensor(batch_idx_np)
    batch["atom_types"] = paddle.to_tensor(atom_types_np)
    batch["batch_size"] = batch_size
    return batch


class MiAD(RuntimeMixin, paddle.nn.Layer):
    """Mirage Atom Diffusion model.

    ``sample`` supports num-atoms-based sampling: all atoms start as mirage
    (type 0) and mirage atoms are filtered from the output.
    """

    supports_num_atoms_sampling = True

    # Official MAX_ATOMIC_NUM; class index 0 is the mirage type.
    _MAX_ATOMIC_NUM = 100
    _MIRAGE_TYPE = 0
    # Hydrogen fallback when every sampled atom is mirage.
    _FALLBACK_ATOMIC_NUM = 1

    def __init__(
        self,
        model_cfg=None,
        diffusion_cfg=None,
        execution_backend="eager",
        runtime_options=None,
    ):
        super().__init__()
        self._init_runtime(execution_backend, runtime_options)

        model_cfg = model_cfg or {}
        diffusion_cfg = diffusion_cfg or {}

        self.mirage_num_atoms = model_cfg.get("mirage_num_atoms", None)
        model_cfg = dict(model_cfg)
        # max_atoms (MiAD naming) maps to num_classes (CSPNet naming)
        model_cfg.setdefault(
            "num_classes", model_cfg.pop("max_atoms", self._MAX_ATOMIC_NUM)
        )
        # prop_dim/pred_scalar excluded: MiAD has no property-guided branches
        _cspnet_keys = {
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
        cspnet_kwargs = {k: v for k, v in model_cfg.items() if k in _cspnet_keys}
        self.decoder = MiADCSPNet(**cspnet_kwargs)

        self.config = diffusion_cfg
        self.cont_time = self.config["cont_time"]
        self.num_steps = self.config["num_steps"]
        # Official default eps=1e-3.
        self.eps = float(self.config.get("eps", 1e-3))
        self.time_embedding = SinusoidalTimeEmbeddings(
            self.config.get("time_embed_dim", 256)
        )
        self.timestep_sampler = UniformTimestepSampler(
            min_t=self.eps, max_t=self.num_steps - 1
        )

        lat_cfg = self.config["lat_diffusion"]
        self.lat_scheduler = build_scheduler(lat_cfg["scheduler_cfg"])

        frac_cfg = self.config["frac_diffusion"]
        self.frac_scheduler = build_scheduler(frac_cfg["scheduler_cfg"])
        self.step_lr = frac_cfg["step_lr"]
        self.sigmas_t = self.frac_scheduler.discrete_sigmas[:, None]
        self.sigmas_norm_t = self.frac_scheduler.discrete_sigmas_norm[:, None]
        self.sb = self.frac_scheduler.sigma_min

        self.type_diffusion = build_scheduler(self.config["type_diffusion"])

    def set_state_dict(self, state_dict, use_structured_name=True):
        # Official checkpoints: no "decoder." prefix and PyTorch (out, in)
        # Linear weights. Adapt per key.
        linear_param_names = set()
        for module_name, module in self.decoder.named_modules():
            if isinstance(module, nn.Linear):
                name = (
                    f"decoder.{module_name}.weight" if module_name else "decoder.weight"
                )
                linear_param_names.add(name)
        adapted = {}
        for k, v in state_dict.items():
            if k.startswith("decoder."):
                new_key = k
            else:
                new_key = f"decoder.{k}"
                if (
                    new_key in linear_param_names
                    and hasattr(v, "shape")
                    and len(v.shape) == 2
                ):
                    v = v.T
            adapted[new_key] = v
        state_dict = adapted
        missing_keys = []
        shape_mismatch_keys = []
        param_state = {}
        for name, param in self.named_parameters():
            if name not in state_dict:
                missing_keys.append(name)
                continue
            v = state_dict[name]
            if hasattr(v, "numpy"):
                v = v.numpy()
            elif not isinstance(v, np.ndarray):
                v = np.asarray(v)
            if v.shape == param.shape:
                param_state[name] = v.astype(param.numpy().dtype)
            else:
                shape_mismatch_keys.append(name)
        loaded = set(param_state.keys())
        unexpected_keys = [k for k in state_dict.keys() if k not in loaded]
        if shape_mismatch_keys:
            logger.warning(
                "Shape mismatch, skipped: %s", ", ".join(shape_mismatch_keys)
            )
        for name in param_state:
            self.get_parameter(name).set_value(param_state[name])
        return missing_keys, unexpected_keys

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

    def _time_sample(self, batch):
        t = self.timestep_sampler(batch["batch_size"])
        if not self.cont_time:
            t = t.round().cast("int64")
        t_per_atom = t.repeat_interleave(batch["num_atoms"])
        return [t, t_per_atom]

    def _forward_step_sample(self, x0, t, batch):
        l0, f0, a0 = x0
        t0_idx = t[0].cast("int64")
        t1_idx = t[1].cast("int64")

        lat_noise = paddle.randn(l0.shape)
        batch["lat_noise"] = lat_noise
        lt = self.lat_scheduler.add_noise(l0, lat_noise, t0_idx)

        frac_noise = paddle.randn(f0.shape)
        batch["frac_noise"] = frac_noise
        ft = (f0 + self.sigmas_t[t1_idx] * frac_noise) % 1.0

        at = self.type_diffusion.forward_step_sample(a0, t[1], batch)

        return [lt, ft, at]

    def _reverse_step_sample(self, xt, t, batch):
        lt, ft, at = xt
        _, f_pred, _ = self._model_prediction(xt, t, batch)
        ft_05 = self._frac_reverse_part1(f_pred, ft, t[1])
        xt_05 = [lt, ft_05, at]
        l_pred, f_pred, a_pred = self._model_prediction(xt_05, t, batch)
        lt_1 = self._lat_reverse(l_pred, lt, t[0])
        ft_1 = self._frac_reverse_part2(f_pred, ft_05, t[1])

        at_1 = self.type_diffusion.reverse_step_sample(a_pred, at, t[1], batch)
        return [lt_1, ft_1, at_1]

    def _lat_reverse(self, pred, xt, t):
        t_idx = t.cast("int64")
        return self.lat_scheduler.step(pred, t_idx[0], xt).prev_sample

    def _frac_reverse_part1(self, pred, xt, t):
        t_idx = t.cast("int64")
        st = self.sigmas_t[t_idx]
        snt = self.sigmas_norm_t[t_idx]
        step_size = self.step_lr * (st / self.sb) ** 2
        drift = -step_size * pred * paddle.sqrt(snt)
        diffusion = paddle.sqrt(2 * step_size) * paddle.randn(xt.shape)
        return xt + drift + diffusion

    def _frac_reverse_part2(self, pred, xt, t):
        t_idx = t.cast("int64")
        st = self.sigmas_t[t_idx]
        st_1 = self.sigmas_t[paddle.maximum(t_idx - 1, paddle.to_tensor(0))]
        snt = self.sigmas_norm_t[t_idx]
        step_size = st**2 - st_1**2
        drift = -step_size * pred * paddle.sqrt(snt)
        diffusion = paddle.sqrt(
            st_1**2 * (st**2 - st_1**2) / (st**2)
        ) * paddle.randn(xt.shape)
        return (xt + drift + diffusion) % 1.0

    def _model_prediction(self, xt, t, batch):
        lt, ft, at = xt
        time_emb = self.time_embedding(1000 * (t[0] / self.num_steps) + 1)
        return self._decode(
            time_emb, at, ft, lt, batch["num_atoms"], batch["batch_idx"]
        )

    def _output_transform(self, x0, batch):
        return [x0[0], x0[1], self.type_diffusion.output_transform(x0[2], batch)]

    def forward(self, batch, **kwargs):
        batch = _extract_x0(batch, mirage_num_atoms=self.mirage_num_atoms)
        batch["t"] = self._time_sample(batch)
        batch["xt"] = self._forward_step_sample(batch["x0"], batch["t"], batch)
        batch["prediction"] = self._model_prediction(batch["xt"], batch["t"], batch)

        lat_noise = batch["lat_noise"]
        loss_lat = (
            ((batch["prediction"][0] - lat_noise) ** 2)
            .reshape([-1, 9])
            .mean(axis=1)
            .mean()
        )

        t_idx = batch["t"][1].cast("int64")
        st = self.sigmas_t[t_idx]
        snt = self.sigmas_norm_t[t_idx]
        frac_noise = batch["frac_noise"]
        normed_score = d_log_p_wrapped_normal(st * frac_noise, st) / paddle.sqrt(snt)
        loss_frac = (
            ((batch["prediction"][1] - normed_score) ** 2).reshape([-1, 3]).mean(axis=1)
        )
        # Mask type-0 mirage atoms from the coordinate loss and rescale.
        mask = (batch["x0"][2] != 0).cast(loss_frac.dtype)
        coef = mask.shape[0] / mask.sum().clip(min=1)
        loss_frac = loss_frac * mask * coef
        loss_frac = loss_frac.mean()

        loss = loss_lat + loss_frac

        loss_type = self.type_diffusion.loss(
            batch["prediction"][2],
            batch["xt"][2],
            self.type_diffusion.to_domain(batch["x0"][2]),
            batch["t"][1],
        ).mean()
        loss = loss + loss_type

        return {"loss_dict": {"loss": loss}}

    @paddle.no_grad()
    def sample(self, batch_data, num_inference_steps=None):
        if "structure_array" in batch_data:
            num_atoms_data = batch_data["structure_array"].get("num_atoms", None)
        else:
            num_atoms_data = batch_data.get("num_atoms", None)

        if "batch_idx" in batch_data:
            for key in ("num_atoms", "batch_size"):
                if key not in batch_data:
                    raise ValueError(
                        f"batch_idx provided but missing required key '{key}'"
                    )
        elif num_atoms_data is not None:
            num_atoms, num_atoms_np = _parse_num_atoms_to_per_crystal(num_atoms_data)
            batch_idx_np, batch_size = _build_batch_idx(num_atoms_np)
            batch_data = {
                **batch_data,
                "num_atoms": num_atoms,
                "batch_idx": paddle.to_tensor(batch_idx_np),
                "batch_size": batch_size,
            }

        original_steps = self.num_steps
        if num_inference_steps is not None:
            self.num_steps = num_inference_steps

        try:
            batch = batch_data
            prior_at = self.type_diffusion.prior_sample(batch)
            na = batch["num_atoms"]
            total = int(na.sum()) if na.ndim > 0 else int(na)
            batch["xt"] = [
                paddle.randn([batch["batch_size"], 3, 3], dtype="float32"),
                paddle.rand([total, 3], dtype="float32"),
                prior_at,
            ]
            for t_val in range(self.num_steps - 1, -1, -1):
                t = paddle.full([batch["batch_size"]], t_val, dtype="float32")
                batch["t"] = [t, t.repeat_interleave(batch["num_atoms"])]
                batch["xt"] = self._reverse_step_sample(batch["xt"], batch["t"], batch)
            batch["xt"] = self._output_transform(batch["xt"], batch)
            batch["x0_prediction"] = batch["xt"]
        finally:
            self.num_steps = original_steps

        lattices, frac_coords, atom_types = batch["x0_prediction"]

        result = []
        num_atoms = batch_data.get("num_atoms", None)
        batch_size = batch_data.get("batch_size", lattices.shape[0])

        start_idx = 0
        for i in range(batch_size):
            if num_atoms is not None:
                n = int(num_atoms[i])
            else:
                n = frac_coords.shape[0]
                if i > 0:
                    break
            lat_np = _to_numpy(lattices[i])
            fc_np = _to_numpy(frac_coords[start_idx : start_idx + n])
            at_np = _to_numpy(atom_types[start_idx : start_idx + n])
            start_idx += n
            valid_mask = at_np != self._MIRAGE_TYPE
            if valid_mask.any():
                at_np = at_np[valid_mask]
                fc_np = fc_np[valid_mask]
                n = int(valid_mask.sum())
            else:
                # All-mirage fallback keeps the structure non-empty.
                at_np = np.where(
                    at_np == self._MIRAGE_TYPE, self._FALLBACK_ATOMIC_NUM, at_np
                )
            lat_for_params = lat_np.reshape(1, 3, 3) if lat_np.ndim == 2 else lat_np
            lengths, angles = lattices_to_params_shape_numpy(lat_for_params)
            result.append(
                {
                    "num_atoms": n,
                    "atom_types": at_np,
                    "frac_coords": fc_np,
                    "lattice": lat_np,
                    "lengths": lengths.flatten(),
                    "angles": angles.flatten(),
                }
            )

        return {"result": result}

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

# Adapted from GeoDiff (https://github.com/MinkaiXu/GeoDiff). See LICENSE.

import numpy as np
import paddle
from paddle import nn

from ppmat.datasets.geometric_data_type.batch import Batch
from ppmat.datasets.geometric_data_type.data import Data
from ppmat.schedulers import build_scheduler
from ppmat.utils.scatter import scatter_mean

from ._encoders import GINEncoder
from ._encoders import MLPEdgeEncoder
from ._encoders import MultiLayerPerceptron
from ._encoders import SchNetEncoder
from ._graph import eq_transform
from ._graph import extend_graph_order_radius
from ._graph import get_distance
from ._graph import higher_order_edges


def _get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    if beta_schedule != "sigmoid":
        raise ValueError("GeoDiff uses the sigmoid diffusion schedule.")
    points = np.linspace(-6, 6, num_diffusion_timesteps)
    sigmoid = 1 / (np.exp(-points) + 1)
    return sigmoid * (beta_end - beta_start) + beta_start


def _pair_features(nodes, edge_index, edge_features):
    row = paddle.gather(nodes, edge_index[0])
    col = paddle.gather(nodes, edge_index[1])
    return paddle.concat([row * col, edge_features], axis=-1)


def _clip_norm(vec, limit):
    norm = paddle.linalg.norm(vec, axis=-1, p=2, keepdim=True)
    scale = paddle.where(norm > limit, limit / norm, paddle.ones_like(norm))
    return vec * scale


class GeoDiff(nn.Layer):
    """Dual-encoder coordinate diffusion for GEOM molecular conformers."""

    def __init__(
        self,
        type="diffusion",
        network="dualenc",
        hidden_dim=128,
        num_convs=6,
        num_convs_local=4,
        cutoff=10.0,
        mlp_act="relu",
        beta_schedule="sigmoid",
        beta_start=1.0e-7,
        beta_end=2.0e-3,
        num_diffusion_timesteps=5000,
        edge_order=3,
        edge_encoder="mlp",
        smooth_conv=False,
    ):
        super().__init__()
        if (type, network, edge_encoder) != ("diffusion", "dualenc", "mlp"):
            raise ValueError(
                "GeoDiff supports the dual-encoder diffusion model with MLP edges"
            )
        self.cutoff = cutoff
        self.edge_order = edge_order
        self.edge_encoder_global = MLPEdgeEncoder(hidden_dim, mlp_act)
        self.edge_encoder_local = MLPEdgeEncoder(hidden_dim, mlp_act)
        self.encoder_global = SchNetEncoder(
            hidden_channels=hidden_dim,
            num_filters=hidden_dim,
            num_interactions=num_convs,
            edge_channels=self.edge_encoder_global.out_channels,
            cutoff=cutoff,
            smooth=smooth_conv,
        )
        self.encoder_local = GINEncoder(
            hidden_dim=hidden_dim, num_convs=num_convs_local
        )
        self.grad_global_dist_mlp = MultiLayerPerceptron(
            2 * hidden_dim, [hidden_dim, hidden_dim // 2, 1], activation=mlp_act
        )
        self.grad_local_dist_mlp = MultiLayerPerceptron(
            2 * hidden_dim, [hidden_dim, hidden_dim // 2, 1], activation=mlp_act
        )
        # Keep the aliases used by the released checkpoint and optimizer groups.
        self.model_global = nn.LayerList(
            [self.edge_encoder_global, self.encoder_global, self.grad_global_dist_mlp]
        )
        self.model_local = nn.LayerList(
            [self.edge_encoder_local, self.encoder_local, self.grad_local_dist_mlp]
        )
        # The released schedule evaluates its sigmoid in NumPy before the
        # float32 cast. Supply those exact betas to the shared table builder.
        trained_betas = _get_beta_schedule(
            beta_schedule,
            beta_start=beta_start,
            beta_end=beta_end,
            num_diffusion_timesteps=num_diffusion_timesteps,
        ).astype(np.float32)
        scheduler = build_scheduler(
            {
                "__class_name__": "DDPMScheduler",
                "__init_params__": {
                    "num_train_timesteps": num_diffusion_timesteps,
                    "trained_betas": trained_betas,
                },
            }
        )
        self.register_buffer("betas", scheduler.betas)
        self.register_buffer("alphas", scheduler.alphas_cumprod)
        self.num_timesteps = self.betas.shape[0]

    def denoise_step(
        self,
        atom_type,
        pos,
        bond_index,
        bond_type,
        batch,
        time_step,
        edge_index=None,
        edge_type=None,
        edge_length=None,
        return_edges=False,
        extend_order=True,
        extend_radius=True,
        is_sidechain=None,
    ):
        """Predict global and local edge scores, optionally with their graph.

        The QM9 network does not condition its features on the timestep.
        """
        if edge_index is None or edge_type is None or edge_length is None:
            edge_index, edge_type = extend_graph_order_radius(
                num_nodes=atom_type.shape[0],
                pos=pos,
                edge_index=bond_index,
                edge_type=bond_type,
                batch=batch,
                order=self.edge_order,
                cutoff=self.cutoff,
                extend_order=extend_order,
                extend_radius=extend_radius,
                is_sidechain=is_sidechain,
            )
            edge_length = get_distance(pos, edge_index).unsqueeze(-1)
        local_mask = edge_type > 0
        edge_global = self.edge_encoder_global(
            edge_length=edge_length, edge_type=edge_type
        )
        nodes_global = self.encoder_global(
            atom_type, edge_index, edge_length, edge_global
        )
        score_global = self.grad_global_dist_mlp(
            _pair_features(nodes_global, edge_index, edge_global)
        )
        # Both branches use the global edge encoder; the unused local encoder is
        # still part of the released model state.
        local_index = edge_index[:, local_mask]
        edge_local = edge_global[local_mask]
        nodes_local = self.encoder_local(atom_type, local_index, edge_local)
        score_local = self.grad_local_dist_mlp(
            _pair_features(nodes_local, local_index, edge_local)
        )
        if return_edges:
            return (
                score_global,
                score_local,
                edge_index,
                edge_type,
                edge_length,
                local_mask,
            )
        return score_global, score_local

    def forward(self, batch):
        """Compute the mean per-atom training losses for a collated graph batch."""
        loss, loss_global, loss_local = self._diffusion_loss(
            batch["atom_types"],
            batch["cart_coords"],
            batch["edge_index"],
            batch["edge_type"],
            batch["batch"],
            int(batch["num_graphs"]),
        )
        result = {
            "loss_dict": {
                "loss": loss.mean(),
                "loss_global": loss_global.mean(),
                "loss_local": loss_local.mean(),
            }
        }
        if not self.training:
            result["loss_stats"] = {
                "loss": loss.sum(),
                "loss_global": loss_global.sum(),
                "loss_local": loss_local.sum(),
            }
        return result

    def _diffusion_loss(
        self,
        atom_type,
        pos,
        bond_index,
        bond_type,
        batch,
        num_graphs,
        *,
        replay=None,
        extend_order=True,
        extend_radius=True,
    ):
        """Compute per-atom losses with optional explicit timestep and noise."""
        if replay is None:
            time_step = paddle.randint(
                0,
                self.num_timesteps,
                [num_graphs // 2 + 1],
                dtype=paddle.int64,
                device=pos.place,
            )
            time_step = paddle.concat([time_step, self.num_timesteps - time_step - 1])[
                :num_graphs
            ]
            pos_noise = paddle.randn_like(pos, device=pos.place)
        else:
            time_step, pos_noise = replay
        a = paddle.gather(self.alphas, time_step)
        a_pos = paddle.gather(a, batch).unsqueeze(-1)
        pos_perturbed = pos + pos_noise * (1.0 - a_pos).sqrt() / a_pos.sqrt()
        (
            score_global,
            score_local,
            edge_index,
            _,
            edge_length,
            local_mask,
        ) = self.denoise_step(
            atom_type,
            pos_perturbed,
            bond_index,
            bond_type,
            batch,
            time_step,
            return_edges=True,
            extend_order=extend_order,
            extend_radius=extend_radius,
        )
        edge2graph = paddle.gather(batch, edge_index[0])
        a_edge = paddle.gather(a, edge2graph).unsqueeze(-1)
        d_gt = get_distance(pos, edge_index).unsqueeze(-1)
        d_target = (d_gt - edge_length) / (1.0 - a_edge).sqrt() * a_edge.sqrt()
        global_mask = (edge_length <= self.cutoff) & ~local_mask.unsqueeze(-1)
        target_global = paddle.where(global_mask, d_target, paddle.zeros_like(d_target))
        score_global = paddle.where(
            global_mask, score_global, paddle.zeros_like(score_global)
        )
        target_pos_global = eq_transform(
            target_global, pos_perturbed, edge_index, edge_length
        )
        node_eq_global = eq_transform(
            score_global, pos_perturbed, edge_index, edge_length
        )
        loss_global = 2 * paddle.sum(
            (node_eq_global - target_pos_global) ** 2, axis=-1, keepdim=True
        )

        local_index = edge_index[:, local_mask]
        local_length = edge_length[local_mask]
        target_pos_local = eq_transform(
            d_target[local_mask], pos_perturbed, local_index, local_length
        )
        node_eq_local = eq_transform(
            score_local, pos_perturbed, local_index, local_length
        )
        loss_local = 5 * paddle.sum(
            (node_eq_local - target_pos_local) ** 2, axis=-1, keepdim=True
        )
        return loss_global + loss_local, loss_global, loss_local

    @staticmethod
    def _conformer_count(num_confs, data, position):
        """Return the number of conformers sampled for one molecule.

        ``"Nx"`` gives ``N`` conformers per reference conformer of a molecule
        packed by ``GeomConformerDataset(pack_by_molecule=True)``.
        """
        if isinstance(num_confs, str) and num_confs.endswith("x"):
            if "num_reference" not in data:
                raise ValueError(
                    f"num_confs {num_confs!r} needs molecules packed by "
                    "GeomConformerDataset(pack_by_molecule=True)."
                )
            return int(data["num_reference"][position]) * int(num_confs[:-1])
        num_confs = int(num_confs)
        if num_confs < 1:
            raise ValueError("GeoDiff num_confs must be positive.")
        return num_confs

    @paddle.no_grad()
    def sample(self, data, *, num_confs=1, n_steps=5000):
        """Return one ``pred`` record per input molecule in batch order.

        Each record contains atom numbers, molecule-local covalent edges, and
        ``cart_coords`` of shape ``(num_confs, num_atoms, 3)`` in input atom order.
        ``rdmol`` contains the optional binary RDKit molecule from the input.
        ``num_confs`` is a number of conformers per molecule, or ``"Nx"`` for
        ``N`` conformers per reference conformer of molecules packed by
        ``GeomConformerDataset(pack_by_molecule=True)``.
        """
        if not 1 <= n_steps <= self.num_timesteps:
            raise ValueError(
                f"GeoDiff n_steps must be between 1 and {self.num_timesteps}."
            )

        node_ptr = data["ptr"].tolist()
        edge_index = data["edge_index"]
        edge_type = data["edge_type"]
        if "rdmol_size" in data:
            mol_sizes = data["rdmol_size"].tolist()
            mol_data = data["rdmol_data"].numpy().tobytes()
        else:
            mol_sizes = [0] * (len(node_ptr) - 1)
            mol_data = b""
        mol_start = 0
        pred = []
        for position, (start, stop, mol_size) in enumerate(
            zip(node_ptr[:-1], node_ptr[1:], mol_sizes)
        ):
            count = self._conformer_count(num_confs, data, position)
            atoms = data["atom_types"][start:stop]
            edge_mask = (edge_index[0] >= start) & (edge_index[0] < stop)
            bonds = edge_index[:, edge_mask] - start
            bond_types = edge_type[edge_mask]
            num_atoms = stop - start
            high_index, high_type, _ = higher_order_edges(
                num_atoms, bonds, bond_types, order=self.edge_order
            )
            graph = Data(
                num_nodes=num_atoms,
                atom_types=atoms,
                edge_index=high_index,
                edge_type=high_type,
            )
            repeated = Batch.from_data_list([graph] * count)
            pos_init = paddle.randn([num_atoms * count, 3], device=atoms.place)
            pos, _ = self.langevin_dynamics_sample_diffusion(
                atom_type=repeated.atom_types,
                pos_init=pos_init,
                bond_index=repeated.edge_index,
                bond_type=repeated.edge_type,
                batch=repeated.batch.to(atoms.place),
                num_graphs=count,
                extend_order=False,
                n_steps=n_steps,
                step_lr=1.0e-6,
                global_start_sigma=0.5,
                w_global=1.0,
                clip=1000,
                sampling_type="ld",
                eta=1.0,
            )
            mol_stop = mol_start + mol_size
            pred.append(
                {
                    "atom_types": atoms.numpy(),
                    "edge_index": bonds.numpy(),
                    "edge_type": bond_types.numpy(),
                    "cart_coords": pos.reshape([count, num_atoms, 3]).numpy(),
                    "rdmol": mol_data[mol_start:mol_stop] or None,
                }
            )
            mol_start = mol_stop
        return {"pred": pred}

    @paddle.no_grad()
    def langevin_dynamics_sample_diffusion(
        self,
        atom_type,
        pos_init,
        bond_index,
        bond_type,
        batch,
        num_graphs,
        extend_order,
        extend_radius=True,
        n_steps=100,
        step_lr=1.0e-6,
        clip=1000,
        clip_local=None,
        clip_pos=None,
        global_start_sigma=float("inf"),
        w_global=0.2,
        sampling_type="ddpm_noisy",
        eta=1.0,
        return_trajectory=False,
    ):
        """Return the denoised coordinates and optional CPU trajectory."""
        if sampling_type not in ("generalized", "ddpm_noisy", "ld"):
            raise ValueError(f"Unknown GeoDiff sampling type: {sampling_type}")
        sigmas = (1.0 - self.alphas).sqrt() / self.alphas.sqrt()
        if sampling_type != "ld":
            # Include alpha at t=-1, retaining the original reverse table formula.
            beta = paddle.concat([paddle.zeros_like(self.betas[:1]), self.betas])
            alphas = paddle.cumprod(1 - beta, dim=0)
        pos = pos_init * sigmas[-1]
        pos_traj = []
        start = self.num_timesteps - n_steps
        for i in range(self.num_timesteps - 1, start - 1, -1):
            j = i - 1 if i > start else -1
            t = None
            if sampling_type == "ddpm_noisy":
                t = paddle.full([num_graphs], i, dtype=paddle.int64, device=pos.place)
            (
                score_global,
                score_local,
                edge_index,
                _,
                edge_length,
                local_mask,
            ) = self.denoise_step(
                atom_type,
                pos,
                bond_index,
                bond_type,
                batch,
                t,
                return_edges=True,
                extend_order=extend_order,
                extend_radius=extend_radius,
            )
            node_eq_local = eq_transform(
                score_local, pos, edge_index[:, local_mask], edge_length[local_mask]
            )
            if clip_local is not None:
                node_eq_local = _clip_norm(node_eq_local, clip_local)
            if sigmas[i] < global_start_sigma:
                score_global = score_global * (
                    1 - local_mask.unsqueeze(-1).astype(score_global.dtype)
                )
                node_eq_global = _clip_norm(
                    eq_transform(score_global, pos, edge_index, edge_length), clip
                )
            else:
                node_eq_global = 0
            eps_pos = node_eq_local + node_eq_global * w_global
            noise = paddle.randn_like(pos, device=pos.place)
            if sampling_type == "ld":
                step_size = step_lr * (sigmas[i] / 0.01) ** 2
                pos = (
                    pos
                    + step_size * eps_pos / sigmas[i]
                    + noise * paddle.sqrt(step_size * 2)
                )
            else:
                at = alphas[i + 1 : i + 2]
                at_next = alphas[j + 1 : j + 2]
                if sampling_type == "generalized":
                    c1 = eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
                    c2 = (1 - at_next - c1**2).sqrt()
                    step_size_pos_ld = step_lr * (sigmas[i] / 0.01) ** 2 / sigmas[i]
                    step_size_pos_generalized = 5 * (
                        (1 - at).sqrt() / at.sqrt() - c2 / at_next.sqrt()
                    )
                    step_size_pos = (
                        step_size_pos_ld
                        if step_size_pos_ld < step_size_pos_generalized
                        else step_size_pos_generalized
                    )
                    step_size_noise_ld = paddle.sqrt(
                        step_lr * (sigmas[i] / 0.01) ** 2 * 2
                    )
                    step_size_noise_generalized = 3 * (c1 / at_next.sqrt())
                    step_size_noise = (
                        step_size_noise_ld
                        if step_size_noise_ld < step_size_noise_generalized
                        else step_size_noise_generalized
                    )
                    pos = pos - (-eps_pos) * step_size_pos + noise * step_size_noise
                else:
                    beta_t = 1 - at / at_next
                    pos0_from_e = (1.0 / at).sqrt() * pos - (1.0 / at - 1).sqrt() * (
                        -eps_pos
                    )
                    mean = (
                        at_next.sqrt() * beta_t * pos0_from_e
                        + (1 - beta_t).sqrt() * (1 - at_next) * pos
                    ) / (1.0 - at)
                    mask = 1 - (t[:1] == 0).astype(pos.dtype)
                    pos = mean + mask * paddle.exp(0.5 * beta_t.log()) * noise
            if paddle.isnan(pos).any():
                raise FloatingPointError("NaN detected in GeoDiff reverse diffusion")
            pos = pos - scatter_mean(pos, batch, dim=0)[batch]
            if clip_pos is not None:
                pos = paddle.clamp(pos, min=-clip_pos, max=clip_pos)
            if return_trajectory:
                pos_traj.append(pos.clone().cpu())
        return pos, pos_traj

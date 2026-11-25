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
import math

import paddle
import paddle.nn as nn
from tqdm import tqdm

from ppmat.models.common import initializer
from ppmat.models.common.runtime import RuntimeMixin
from ppmat.models.common.runtime import runtime_boundary
from ppmat.models.common.time_embedding import SinusoidalTimeEmbeddings
from ppmat.models.common.time_embedding import uniform_sample_t
from ppmat.schedulers import build_scheduler
from ppmat.schedulers.scheduling_sde_ve import d_log_p_wrapped_normal
from ppmat.utils.scatter import scatter_mean

_NUM_SPATIAL_DIMS = 3
_NUM_LATTICE_MATRIX_ELEMENTS = _NUM_SPATIAL_DIMS * _NUM_SPATIAL_DIMS
_CG_SITE_SPECIES = ("Xmetal", "Xlinker")


class SinusoidsEmbedding(nn.Layer):
    """Encode fractional displacements with integer-frequency sinusoidal pairs."""

    def __init__(self, n_frequencies):
        super().__init__()
        self.n_frequencies = n_frequencies
        self.register_buffer(
            "frequencies",
            2 * math.pi * paddle.arange(end=self.n_frequencies),
            persistable=False,
        )
        self.dim = self.n_frequencies * 2 * _NUM_SPATIAL_DIMS

    def forward(self, x):
        emb = x.unsqueeze(axis=-1) * self.frequencies[None, None, :]
        emb = emb.reshape([-1, self.n_frequencies * _NUM_SPATIAL_DIMS])
        return paddle.concat(x=(emb.sin(), emb.cos()), axis=-1)


class CSPLayer(nn.Layer):
    """Message passing layer for the coarse-grained MOF CSPNet."""

    def __init__(self, hidden_dim, distance_dim):
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(
                in_features=hidden_dim * 2
                + _NUM_LATTICE_MATRIX_ELEMENTS
                + distance_dim,
                out_features=hidden_dim,
            ),
            nn.Silu(),
            nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
            nn.Silu(),
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(in_features=hidden_dim * 2, out_features=hidden_dim),
            nn.Silu(),
            nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
            nn.Silu(),
        )
        self.layer_norm = nn.LayerNorm(normalized_shape=hidden_dim)

    def forward(self, node_features, edge_index, edge_attributes):
        normed = self.layer_norm(node_features)
        edge_features = self.edge_mlp(
            paddle.concat(
                [normed[edge_index[0]], normed[edge_index[1]], edge_attributes],
                axis=1,
            )
        )
        agg = scatter_mean(
            edge_features,
            edge_index[0],
            dim=0,
            dim_size=node_features.shape[0],
        )
        return node_features + self.node_mlp(paddle.concat([normed, agg], axis=1))


class Xrd2MofCSPNet(nn.Layer):
    """CSPNet decoder conditioned on the XRD feature embedding.

    The node embedding fuses atom type, per-atom XRD feature and time embedding
    before message passing.

    Args:
        hidden_dim (int, optional): Hidden dimension. Defaults to 512.
        feature_dim (int, optional): XRD feature embedding dimension. Defaults to 256.
        latent_dim (int, optional): Time embedding dimension. Defaults to 256.
        num_layers (int, optional): Number of CSPLayer. Defaults to 3.
        num_atom_types (int, optional): Number of coarse-grained atom types.
            Defaults to 2.
        num_freqs (int, optional): Number of frequencies of the periodic distance
            embedding. Defaults to 128.
    """

    def __init__(
        self,
        hidden_dim: int = 512,
        feature_dim: int = 256,
        latent_dim: int = 256,
        num_layers: int = 3,
        num_atom_types: int = 2,
        num_freqs: int = 128,
    ):
        super().__init__()
        self.node_embedding = nn.Embedding(
            num_embeddings=num_atom_types, embedding_dim=hidden_dim
        )
        self.atom_latent_emb = nn.Linear(
            in_features=hidden_dim + latent_dim + feature_dim, out_features=hidden_dim
        )
        self.distance_embedding = SinusoidsEmbedding(n_frequencies=num_freqs)
        self.layers = nn.LayerList(
            [
                CSPLayer(hidden_dim, self.distance_embedding.dim)
                for _ in range(num_layers)
            ]
        )
        self.coord_out = nn.Linear(
            in_features=hidden_dim, out_features=_NUM_SPATIAL_DIMS, bias_attr=False
        )
        self.lattice_out = nn.Linear(
            in_features=hidden_dim,
            out_features=_NUM_LATTICE_MATRIX_ELEMENTS,
            bias_attr=False,
        )
        self.final_layer_norm = nn.LayerNorm(normalized_shape=hidden_dim)

    def gen_edges(self, num_atoms):
        counts = num_atoms.tolist()
        indices_pp = []
        rows = paddle.arange(max(counts))
        ind1, ind2 = paddle.meshgrid(rows, rows)
        index = paddle.stack(x=[ind1, ind2], axis=0)
        offset = 0
        for n in counts:
            indices_pp.append(index[:, :n, :n].reshape((2, -1)) + offset)
            offset += n
        return paddle.concat(x=indices_pp, axis=1)

    def forward_with_edges(
        self,
        feature_embedding,
        time_emb,
        atom_types,
        lattices,
        node2graph,
        edges,
        frac_diff,
    ):
        edge2graph = node2graph[edges[0]]
        node_features = self.node_embedding(atom_types)
        node_features = paddle.concat(
            x=[node_features, feature_embedding[node2graph], time_emb[node2graph]],
            axis=1,
        )
        node_features = self.atom_latent_emb(node_features)

        lattice_ips = paddle.matmul(lattices, lattices, transpose_y=True)
        lattice_ips = lattice_ips.reshape([-1, _NUM_LATTICE_MATRIX_ELEMENTS])
        edge_attributes = paddle.concat(
            [lattice_ips[edge2graph], self.distance_embedding(frac_diff)], axis=1
        )
        for layer in self.layers:
            node_features = layer(node_features, edges, edge_attributes)
        node_features = self.final_layer_norm(node_features)
        coord_out = self.coord_out(node_features)
        graph_features = scatter_mean(
            node_features,
            node2graph,
            dim=0,
            dim_size=lattices.shape[0],
        )
        lattice_out = self.lattice_out(graph_features)
        lattice_out = lattice_out.reshape([-1, _NUM_SPATIAL_DIMS, _NUM_SPATIAL_DIMS])
        return paddle.bmm(lattice_out, lattices), coord_out


class Xrd2Mof(RuntimeMixin, nn.Layer):
    """Coarse-grained MOF crystal structure generation from XRD embeddings.

    The lattice is diffused with a DDPM Gaussian process and the fractional
    coordinates are diffused on a torus with wrapped Gaussian noise; the decoder
    is additionally conditioned on a per-sample XRD feature embedding.

    Args:
        decoder_cfg (dict): Decoder configuration. See `Xrd2MofCSPNet` for details.
        lattice_noise_scheduler_cfg (dict): Noise scheduler configuration for lattice.
        coord_noise_scheduler_cfg (dict): Noise scheduler configuration for coordinates.
        num_train_timesteps (int): Number of diffusion steps. Defaults to 1000.
        lattice_loss_weight (float, optional): Lattice loss weight. Defaults to 0.1.
        coord_loss_weight (float, optional): Coordinate loss weight. Defaults to 1.0.
        p_unconditional (float, optional): Probability of replacing the XRD
            embedding with zeros during training. Defaults to 0.0.
    """

    supports_num_atoms_sampling = False

    def __init__(
        self,
        decoder_cfg: dict,
        lattice_noise_scheduler_cfg: dict,
        coord_noise_scheduler_cfg: dict,
        num_train_timesteps: int = 1000,
        lattice_loss_weight: float = 0.1,
        coord_loss_weight: float = 1.0,
        execution_backend: str = "eager",
        runtime_options: dict | None = None,
        p_unconditional: float = 0.0,
    ) -> None:

        super().__init__()
        self._init_runtime(execution_backend, runtime_options)
        if not 0.0 <= p_unconditional <= 1.0:
            raise ValueError("p_unconditional must be between 0 and 1.")
        self.p_unconditional = p_unconditional

        self.decoder = Xrd2MofCSPNet(**decoder_cfg)

        self.lattice_scheduler = build_scheduler(lattice_noise_scheduler_cfg)
        self.coord_scheduler = build_scheduler(coord_noise_scheduler_cfg)

        self.num_train_timesteps = num_train_timesteps
        self.lattice_loss_weight = lattice_loss_weight
        self.coord_loss_weight = coord_loss_weight

        self.time_embedding = SinusoidalTimeEmbeddings(decoder_cfg["latent_dim"])
        self.apply(self._init_weights)

    def _decode(
        self,
        time_emb,
        feature_embedding,
        atom_types,
        frac_coords,
        lattices,
        num_atoms,
        node2graph,
        edges=None,
        guidance_scale=1.0,
    ):
        if edges is None:
            edges = self.decoder.gen_edges(num_atoms)
        frac_diff = (frac_coords[edges[1]] - frac_coords[edges[0]]) % 1.0
        if guidance_scale == 0.0:
            feature_embedding = paddle.zeros_like(feature_embedding)
        pred_l, pred_x = self._runtime_decode(
            time_emb,
            feature_embedding,
            atom_types,
            lattices,
            node2graph,
            edges,
            frac_diff,
        )
        if guidance_scale in (0.0, 1.0):
            return pred_l, pred_x
        unconditional_l, unconditional_x = self._runtime_decode(
            time_emb,
            paddle.zeros_like(feature_embedding),
            atom_types,
            lattices,
            node2graph,
            edges,
            frac_diff,
        )
        return (
            unconditional_l + guidance_scale * (pred_l - unconditional_l),
            unconditional_x + guidance_scale * (pred_x - unconditional_x),
        )

    @runtime_boundary("denoise_step")
    def _runtime_decode(
        self,
        time_emb,
        feature_embedding,
        atom_types,
        lattices,
        node2graph,
        edges,
        frac_diff,
    ):
        return self.decoder.forward_with_edges(
            feature_embedding,
            time_emb,
            atom_types,
            lattices,
            node2graph,
            edges,
            frac_diff,
        )

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            initializer.linear_init_(m)
        elif isinstance(m, nn.Embedding):
            initializer.normal_(m.weight)

    def forward(self, batch):

        structure_array = batch["structure_array"]
        batch_size = structure_array["num_atoms"].shape[0]
        batch_idx = paddle.repeat_interleave(
            paddle.arange(batch_size), repeats=structure_array["num_atoms"]
        )

        # The time embedding follows the original 1-based diffusion step, while
        # the noise schedulers are indexed from 0 (diffusion step - 1).
        times = uniform_sample_t(batch_size, self.num_train_timesteps)
        times_per_atom = times[batch_idx]
        time_emb = self.time_embedding(times + 1)

        lattices = structure_array["lattice"]

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

        feature_embedding = batch["feature_embedding"]
        if self.training and self.p_unconditional > 0.0:
            unconditional_mask = paddle.rand([batch_size, 1]) < self.p_unconditional
            feature_embedding = paddle.where(
                unconditional_mask,
                paddle.zeros_like(feature_embedding),
                feature_embedding,
            )
        pred_l, pred_x = self._decode(
            time_emb,
            feature_embedding,
            structure_array["atom_types"],
            input_frac_coords,
            input_lattice,
            structure_array["num_atoms"],
            batch_idx,
        )

        sigmas_per_atom = self.coord_scheduler.discrete_sigmas[times_per_atom][:, None]
        sigmas_norm_per_atom = self.coord_scheduler.discrete_sigmas_norm[
            times_per_atom
        ][:, None]

        tar_x = d_log_p_wrapped_normal(
            sigmas_per_atom * rand_x, sigmas_per_atom
        ) / paddle.sqrt(x=sigmas_norm_per_atom)

        loss_lattice = nn.functional.mse_loss(input=pred_l, label=rand_l)
        loss_coord = nn.functional.mse_loss(input=pred_x, label=tar_x)
        loss = (
            self.lattice_loss_weight * loss_lattice
            + self.coord_loss_weight * loss_coord
        )
        loss_dict = {
            "loss": loss,
            "loss_lattice": loss_lattice,
            "loss_coord": loss_coord,
        }

        return {
            "loss_dict": loss_dict,
        }

    @paddle.no_grad()
    def sample(self, batch_data, num_inference_steps=1000, guidance_scale=1.0):
        """Sample CG structures with optional classifier-free guidance.

        A scale of 1 preserves conditional sampling. Other scales require
        weights trained with a positive p_unconditional; 0 uses the zero-XRD
        branch while values above 1 strengthen the XRD condition.
        """
        if "feature_embedding" not in batch_data:
            raise ValueError(
                "Xrd2Mof requires an XRD feature_embedding. "
                "Use sample_by_dataloader with XRD-conditioned data."
            )
        if not math.isfinite(guidance_scale) or guidance_scale < 0.0:
            raise ValueError("guidance_scale must be finite and nonnegative.")
        if guidance_scale != 1.0 and self.p_unconditional == 0.0:
            raise ValueError(
                "Classifier-free guidance requires weights trained with "
                "p_unconditional > 0."
            )
        structure_array = batch_data["structure_array"]
        batch_size = structure_array["num_atoms"].shape[0]
        batch_idx = paddle.repeat_interleave(
            paddle.arange(batch_size), repeats=structure_array["num_atoms"]
        )
        l_t = paddle.randn(shape=[batch_size, _NUM_SPATIAL_DIMS, _NUM_SPATIAL_DIMS])
        x_t = paddle.rand(
            shape=[structure_array["atom_types"].shape[0], _NUM_SPATIAL_DIMS]
        )
        edges = self.decoder.gen_edges(structure_array["num_atoms"])

        self.lattice_scheduler.set_timesteps(num_inference_steps)
        self.coord_scheduler.set_timesteps(num_inference_steps)

        for lattice_t, coord_t in tqdm(
            zip(self.lattice_scheduler.timesteps, self.coord_scheduler.timesteps),
            total=num_inference_steps,
            desc="Sampling...",
        ):
            time_emb = self.time_embedding(
                paddle.ones([batch_size], dtype="int64") * (lattice_t + 1)
            )
            _, pred_x = self._decode(
                time_emb,
                batch_data["feature_embedding"],
                structure_array["atom_types"],
                x_t,
                l_t,
                structure_array["num_atoms"],
                batch_idx,
                edges=edges,
                guidance_scale=guidance_scale,
            )
            x_t = self.coord_scheduler.step_correct(pred_x, coord_t, x_t).prev_sample

            pred_l, pred_x = self._decode(
                time_emb,
                batch_data["feature_embedding"],
                structure_array["atom_types"],
                x_t,
                l_t,
                structure_array["num_atoms"],
                batch_idx,
                edges=edges,
                guidance_scale=guidance_scale,
            )
            output = self.coord_scheduler.step_pred(
                pred_x,
                coord_t,
                x_t,
            )
            x_t, x_t_mean = output.prev_sample, output.prev_sample_mean

            l_t = self.lattice_scheduler.step(
                pred_l,
                lattice_t,
                l_t,
            ).prev_sample

            x_t = x_t % 1.0

        x_t = x_t_mean % 1.0

        num_atoms = structure_array["num_atoms"].tolist()
        atom_types = [
            _CG_SITE_SPECIES[node_type]
            for node_type in structure_array["atom_types"].tolist()
        ]
        frac_coords = x_t.tolist()
        lattices = l_t.tolist()
        mof_ids = batch_data.get("mof_id")
        start_idx = 0
        result = []
        for i, count in enumerate(num_atoms):
            end_idx = start_idx + count
            item = {
                "num_atoms": count,
                "atom_types": atom_types[start_idx:end_idx],
                "frac_coords": frac_coords[start_idx:end_idx],
                "lattice": lattices[i],
            }
            if mof_ids is not None:
                item["mof_id"] = mof_ids[i]
            result.append(item)
            start_idx = end_idx

        return {"result": result}

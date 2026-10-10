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
from ppmat.utils.crystal import lattice_params_to_matrix_paddle
from ppmat.utils.scatter import scatter_mean


def p_wrapped_normal(x, sigma, N=10, T=1.0):
    p_ = 0
    for i in range(-N, N + 1):
        p_ += paddle.exp(x=-((x + T * i) ** 2) / 2 / sigma**2)
    return p_


def d_log_p_wrapped_normal(x, sigma, N=10, T=1.0):
    p_ = 0
    for i in range(-N, N + 1):
        exp1 = paddle.exp(x=-((x + T * i) ** 2) / 2 / sigma**2)
        p_ += (x + T * i) / sigma**2 * exp1
    return p_ / p_wrapped_normal(x, sigma, N, T)


class SinusoidsEmbedding(paddle.nn.Layer):
    def __init__(self, n_frequencies=10, n_space=3):
        super().__init__()
        self.n_frequencies = n_frequencies
        self.n_space = n_space
        self.frequencies = 2 * math.pi * paddle.arange(end=self.n_frequencies)
        self.dim = self.n_frequencies * 2 * self.n_space

    def forward(self, x):
        emb = x.unsqueeze(axis=-1) * self.frequencies[None, None, :]
        emb = emb.reshape([-1, self.n_frequencies * self.n_space])
        emb = paddle.concat(x=(emb.sin(), emb.cos()), axis=-1)
        return emb


class CSPLayer(paddle.nn.Layer):
    """Message passing layer for the coarse-grained MOF CSPNet."""

    def __init__(
        self,
        hidden_dim=512,
        act_fn=None,
        dis_emb=None,
        ln=False,
        ip=True,
    ):
        super(CSPLayer, self).__init__()
        if act_fn is None:
            act_fn = paddle.nn.Silu()
        self.dis_dim = 3
        self.dis_emb = dis_emb
        self.ip = ip
        if dis_emb is not None:
            self.dis_dim = dis_emb.dim
        self.edge_mlp = paddle.nn.Sequential(
            paddle.nn.Linear(
                in_features=hidden_dim * 2 + 9 + self.dis_dim, out_features=hidden_dim
            ),
            act_fn,
            paddle.nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
            act_fn,
        )
        self.node_mlp = paddle.nn.Sequential(
            paddle.nn.Linear(in_features=hidden_dim * 2, out_features=hidden_dim),
            act_fn,
            paddle.nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
            act_fn,
        )
        self.ln = ln
        if self.ln:
            self.layer_norm = paddle.nn.LayerNorm(normalized_shape=hidden_dim)

    def edge_model(
        self,
        node_features,
        frac_coords,
        lattices,
        edge_index,
        edge2graph,
        frac_diff=None,
    ):
        hi, hj = node_features[edge_index[0]], node_features[edge_index[1]]
        if frac_diff is None:
            xi, xj = frac_coords[edge_index[0]], frac_coords[edge_index[1]]
            frac_diff = (xj - xi) % 1.0
        if self.dis_emb is not None:
            frac_diff = self.dis_emb(frac_diff)
        if self.ip:
            lattice_ips = paddle.matmul(lattices, lattices, transpose_y=True)
        else:
            lattice_ips = lattices

        lattice_ips_flatten = lattice_ips.reshape([-1, 9])
        lattice_ips_flatten_edges = lattice_ips_flatten[edge2graph]
        edges_input = paddle.concat(
            x=[hi, hj, lattice_ips_flatten_edges, frac_diff], axis=1
        )
        edge_features = self.edge_mlp(edges_input)
        return edge_features

    def node_model(self, node_features, edge_features, edge_index):
        agg = scatter_mean(
            edge_features,
            edge_index[0],
            dim=0,
            dim_size=node_features.shape[0],
        )
        agg = paddle.concat(x=[node_features, agg], axis=1)
        out = self.node_mlp(agg)
        return out

    def forward(
        self,
        node_features,
        frac_coords,
        lattices,
        edge_index,
        edge2graph,
        frac_diff=None,
    ):
        node_input = node_features
        if self.ln:
            node_features = self.layer_norm(node_input)
        edge_features = self.edge_model(
            node_features, frac_coords, lattices, edge_index, edge2graph, frac_diff
        )
        node_output = self.node_model(node_features, edge_features, edge_index)
        return node_input + node_output


class Xrd2MofCSPNet(paddle.nn.Layer):
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
        act_fn (str, optional): Activation function type. Defaults to "silu".
        dis_emb (str, optional): Distance embedding method, can be 'sin' or 'none'.
            Defaults to "sin".
        num_freqs (int, optional): Number of frequency components (effective only when
            dis_emb='sin'). Defaults to 128.
        edge_style (str, optional): Edge feature encoding method. Only 'fc' (fully
            connected atomic interactions) is supported. Default: "fc".
        ln (bool, optional): Enable LayerNorm after CSPLayer. Defaults to True.
        ip (bool, optional): Apply lattice inner product for O(3)-invariance. Defaults
            to True.
    """

    def __init__(
        self,
        hidden_dim: int = 512,
        feature_dim: int = 256,
        latent_dim: int = 256,
        num_layers: int = 3,
        num_atom_types: int = 2,
        act_fn: str = "silu",
        dis_emb: str = "sin",
        num_freqs: int = 128,
        edge_style: str = "fc",
        ln: bool = True,
        ip: bool = True,
    ):
        super(Xrd2MofCSPNet, self).__init__()
        self.ip = ip
        self.node_embedding = paddle.nn.Embedding(
            num_embeddings=num_atom_types, embedding_dim=hidden_dim
        )
        self.atom_latent_emb = paddle.nn.Linear(
            in_features=hidden_dim + latent_dim + feature_dim, out_features=hidden_dim
        )
        if act_fn == "silu":
            self.act_fn = paddle.nn.Silu()
        else:
            raise ValueError("act_fn only supports 'silu', got %s" % act_fn)
        if dis_emb == "sin":
            self.dis_emb = SinusoidsEmbedding(n_frequencies=num_freqs)
        elif dis_emb == "none":
            self.dis_emb = None
        else:
            raise ValueError("dis_emb supports 'sin' or 'none', got %s" % dis_emb)
        for i in range(0, num_layers):
            self.add_sublayer(
                name="csp_layer_%d" % i,
                sublayer=CSPLayer(
                    hidden_dim,
                    act_fn=self.act_fn,
                    dis_emb=self.dis_emb,
                    ln=ln,
                    ip=ip,
                ),
            )
        self.num_layers = num_layers
        self.coord_out = paddle.nn.Linear(
            in_features=hidden_dim, out_features=3, bias_attr=False
        )
        self.lattice_out = paddle.nn.Linear(
            in_features=hidden_dim, out_features=9, bias_attr=False
        )
        self.ln = ln
        self.edge_style = edge_style
        if self.ln:
            self.final_layer_norm = paddle.nn.LayerNorm(normalized_shape=hidden_dim)

    def gen_edges(self, num_atoms, frac_coords):
        if self.edge_style == "fc":
            cum_num_atoms = paddle.cumsum(x=num_atoms)
            indices_pp = []
            rows = paddle.arange(num_atoms.max())
            ind1, ind2 = paddle.meshgrid(rows, rows)
            index = paddle.stack(x=[ind1, ind2], axis=0)
            for n, cum_n in zip(num_atoms, cum_num_atoms):
                offset = cum_n - n
                indices_pp.append(index[:, :n, :n].reshape((2, -1)) + offset)
            indices_pp = paddle.concat(x=indices_pp, axis=1)
            fc_edges = indices_pp
            return fc_edges, (frac_coords[fc_edges[1]] - frac_coords[fc_edges[0]]) % 1.0
        else:
            raise NotImplementedError("Edge style '%s'" % self.edge_style)

    def forward(
        self,
        feature_embedding,
        time_emb,
        atom_types,
        frac_coords,
        lattices,
        num_atoms,
        node2graph,
    ):
        edges, frac_diff = self.gen_edges(num_atoms, frac_coords)
        return self.forward_with_edges(
            feature_embedding,
            time_emb,
            atom_types,
            frac_coords,
            lattices,
            num_atoms,
            node2graph,
            edges,
            frac_diff,
        )

    def forward_with_edges(
        self,
        feature_embedding,
        time_emb,
        atom_types,
        frac_coords,
        lattices,
        num_atoms,
        node2graph,
        edges,
        frac_diff,
    ):
        edge2graph = node2graph[edges[0]]
        node_features = self.node_embedding(atom_types)
        feature_per_atom = feature_embedding.repeat_interleave(
            repeats=num_atoms, axis=0
        )
        time_per_atom = time_emb.repeat_interleave(repeats=num_atoms, axis=0)
        node_features = paddle.concat(
            x=[node_features, feature_per_atom, time_per_atom], axis=1
        )
        node_features = self.atom_latent_emb(node_features)

        for i in range(0, self.num_layers):
            layer = getattr(self, "csp_layer_%d" % i)
            node_features = layer(
                node_features,
                frac_coords,
                lattices,
                edges,
                edge2graph,
                frac_diff=frac_diff,
            )
        if self.ln:
            node_features = self.final_layer_norm(node_features)
        coord_out = self.coord_out(node_features)
        graph_features = scatter_mean(
            node_features,
            node2graph,
            dim=0,
            dim_size=lattices.shape[0],
        )
        lattice_out = self.lattice_out(graph_features)
        lattice_out = lattice_out.reshape([-1, 3, 3])
        if self.ip:
            lattice_out = paddle.bmm(lattice_out, lattices)
        return lattice_out, coord_out


class Xrd2Mof(RuntimeMixin, paddle.nn.Layer):
    """Coarse-grained MOF crystal structure generation from XRD embeddings.

    The lattice is diffused with a DDPM Gaussian process and the fractional
    coordinates are diffused on a torus with wrapped Gaussian noise; the decoder
    is additionally conditioned on a per-sample XRD feature embedding.

    Args:
        decoder_cfg (dict): Decoder configuration. See `Xrd2MofCSPNet` for details.
        lattice_noise_scheduler_cfg (dict): Noise scheduler configuration for lattice.
        coord_noise_scheduler_cfg (dict): Noise scheduler configuration for coordinates.
        num_train_timesteps (int): Number of diffusion steps. Defaults to 1000.
        time_dim (int): Time embedding dimension. Defaults to 256.
        lattice_loss_weight (float, optional): Lattice loss weight. Defaults to 0.1.
        coord_loss_weight (float, optional): Coordinate loss weight. Defaults to 1.0.
    """

    supports_num_atoms_sampling = False

    def __init__(
        self,
        decoder_cfg: dict,
        lattice_noise_scheduler_cfg: dict,
        coord_noise_scheduler_cfg: dict,
        num_train_timesteps: int = 1000,
        time_dim: int = 256,
        lattice_loss_weight: float = 0.1,
        coord_loss_weight: float = 1.0,
        execution_backend: str = "eager",
        runtime_options: dict | None = None,
    ) -> None:

        super().__init__()
        self._init_runtime(execution_backend, runtime_options)

        self.decoder = Xrd2MofCSPNet(**decoder_cfg)

        self.lattice_scheduler = build_scheduler(lattice_noise_scheduler_cfg)
        self.coord_scheduler = build_scheduler(coord_noise_scheduler_cfg)

        self.num_train_timesteps = num_train_timesteps
        self.time_dim = time_dim
        self.lattice_loss_weight = lattice_loss_weight
        self.coord_loss_weight = coord_loss_weight

        self.time_embedding = SinusoidalTimeEmbeddings(time_dim)
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
    ):
        edges, frac_diff = self.decoder.gen_edges(num_atoms, frac_coords)
        return self._runtime_decode(
            time_emb,
            feature_embedding,
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
        feature_embedding,
        atom_types,
        frac_coords,
        lattices,
        num_atoms,
        node2graph,
        edges,
        frac_diff,
    ):
        return self.decoder.forward_with_edges(
            feature_embedding,
            time_emb,
            atom_types,
            frac_coords,
            lattices,
            num_atoms,
            node2graph,
            edges,
            frac_diff,
        )

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            initializer.linear_init_(m)
        elif isinstance(m, nn.Embedding):
            initializer.normal_(m.weight)

    def forward(self, batch, **kwargs):

        structure_array = batch["structure_array"]
        batch_size = structure_array["num_atoms"].shape[0]
        batch_idx = paddle.repeat_interleave(
            paddle.arange(batch_size), repeats=structure_array["num_atoms"]
        )

        # The time embedding follows the original 1-based diffusion step, while
        # the noise schedulers are indexed from 0 (diffusion step - 1).
        steps = uniform_sample_t(batch_size, self.num_train_timesteps) + 1
        steps_per_atom = steps.repeat_interleave(repeats=structure_array["num_atoms"])
        times = steps - 1
        times_per_atom = steps_per_atom - 1

        time_emb = self.time_embedding(steps)

        # forward is only driven by train/val batches, which always carry
        # lattice parameters; there is no input path carrying a prebuilt matrix.
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

        pred_l, pred_x = self._decode(
            time_emb,
            structure_array["feature_embedding"],
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
            pred_l, pred_x = self._decode(
                time_emb,
                structure_array["feature_embedding"],
                structure_array["atom_types"],
                x_t,
                l_t,
                structure_array["num_atoms"],
                batch_idx,
            )
            x_t = self.coord_scheduler.step_correct(pred_x, coord_t, x_t).prev_sample

            pred_l, pred_x = self._decode(
                time_emb,
                structure_array["feature_embedding"],
                structure_array["atom_types"],
                x_t,
                l_t,
                structure_array["num_atoms"],
                batch_idx,
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

        mof_ids = structure_array.get("mof_id")
        start_idx = 0
        result = []
        for i in range(batch_size):
            end_idx = start_idx + structure_array["num_atoms"][i]
            item = {
                "num_atoms": structure_array["num_atoms"][i].tolist(),
                "atom_types": structure_array["atom_types"][start_idx:end_idx].tolist(),
                "frac_coords": x_t[start_idx:end_idx].tolist(),
                "lattice": l_t[i].tolist(),
            }
            if mof_ids is not None:
                item["mof_id"] = mof_ids[i]
            result.append(item)
            start_idx += structure_array["num_atoms"][i]

        return {"result": result}

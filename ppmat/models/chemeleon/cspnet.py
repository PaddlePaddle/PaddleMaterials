# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

# Adapted from https://github.com/hspark1212/chemeleon

"""Decoder backbone for Chemeleon.

FiLM-conditioned message passing over fully-connected per-graph edges.
Only the fully-connected edge style is implemented; the released
configuration uses it exclusively.
"""

import math
from collections import namedtuple

import paddle

from ppmat.models.common.activation import swish
from ppmat.utils.scatter import scatter_mean

DECODER_OUTPUTS = namedtuple(
    "DECODER_OUTPUTS", ["atom_types_out", "lattice_out", "coords_out", "node_features"]
)


class Swish(paddle.nn.Layer):
    """SiLU as the composite x * sigmoid(x) expression.

    Forward-equal to the fused SiLU kernel; the composite form keeps the
    CPU fp32 backward finite for large-magnitude negative inputs.
    """

    def forward(self, x):
        return swish(x)


class SinusoidalTimeEmbeddings(paddle.nn.Layer):
    """Attention is all you need."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = paddle.exp(paddle.arange(half_dim) * -embeddings)
        embeddings = time.astype(embeddings.dtype)[:, None] * embeddings[None, :]
        embeddings = paddle.concat((embeddings.sin(), embeddings.cos()), axis=-1)
        return embeddings


class SinusoidsEmbedding(paddle.nn.Layer):
    """Embedding for periodic distance features."""

    def __init__(self, n_frequencies=10, n_space=3):
        super().__init__()
        self.n_frequencies = n_frequencies
        self.n_space = n_space
        # plain attribute (not a buffer): stays out of the state_dict
        self.frequencies = 2 * math.pi * paddle.arange(self.n_frequencies)
        self.dim = self.n_frequencies * 2 * self.n_space

    def forward(self, x):
        frequencies = self.frequencies.astype(x.dtype)
        if frequencies.place != x.place:
            frequencies = frequencies.to(x.place)
        emb = x.unsqueeze(-1) * frequencies[None, None, :]
        emb = emb.reshape([-1, self.n_frequencies * self.n_space])
        emb = paddle.concat((emb.sin(), emb.cos()), axis=-1)
        return emb.detach()


class FilmLayer(paddle.nn.Layer):
    """FiLM layer for efficient incorporation of time or conditional embeddings."""

    def __init__(
        self,
        hidden_dim=128,
        time_dim=256,
        text_dim=128,
        act_fn=Swish(),
    ):
        super(FilmLayer, self).__init__()
        self.hidden_dim = hidden_dim
        self.time_dim = time_dim
        self.text_dim = text_dim
        self.act_fn = act_fn
        self.mlp_cond = paddle.nn.Sequential(
            paddle.nn.Linear(time_dim + text_dim, hidden_dim * 2),
            act_fn,
        )
        # block
        self.proj = paddle.nn.Linear(hidden_dim, hidden_dim)
        self.norm = paddle.nn.LayerNorm(hidden_dim)

    def forward(self, x, time_embeds=None, text_embeds=None):
        # Feature-wise Linear Modulation
        if time_embeds is not None and text_embeds is not None:
            cond_emb = self.mlp_cond(paddle.concat([time_embeds, text_embeds], axis=1))
        elif time_embeds is not None:
            cond_emb = self.mlp_cond(time_embeds)
        elif text_embeds is not None:
            cond_emb = self.mlp_cond(text_embeds)
        else:
            raise ValueError("Either time or text embeddings must be provided.")
        scale, shift = cond_emb.chunk(2, dim=1)

        # residual block
        x_init = x.clone()
        x = self.proj(x)
        x = self.norm(x)
        x = x * scale + shift
        x = self.act_fn(x)
        x = x + x_init
        return x


class CSPLayer(paddle.nn.Layer):
    """Message passing layer for cspnet."""

    def __init__(self, hidden_dim=128, act_fn=Swish(), dis_emb=None, ln=False, ip=True):
        super(CSPLayer, self).__init__()

        self.dis_dim = 3
        self.dis_emb = dis_emb
        self.ip = ip
        if dis_emb is not None:
            self.dis_dim = dis_emb.dim
        self.edge_mlp = paddle.nn.Sequential(
            paddle.nn.Linear(hidden_dim * 2 + 9 + self.dis_dim, hidden_dim),
            act_fn,
            paddle.nn.Linear(hidden_dim, hidden_dim),
            act_fn,
        )
        self.node_mlp = paddle.nn.Sequential(
            paddle.nn.Linear(hidden_dim * 2, hidden_dim),
            act_fn,
            paddle.nn.Linear(hidden_dim, hidden_dim),
            act_fn,
        )
        self.ln = ln
        if self.ln:
            self.layer_norm = paddle.nn.LayerNorm(hidden_dim)

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
            frac_diff = self.dis_emb(frac_diff)  # fourier transform
        if self.ip:
            lattice_ips = lattices @ lattices.transpose(-1, -2)
        else:
            lattice_ips = lattices
        lattice_ips_flatten = lattice_ips.reshape([-1, 9])
        lattice_ips_flatten_edges = lattice_ips_flatten[edge2graph]
        edges_input = paddle.concat(
            [hi, hj, lattice_ips_flatten_edges, frac_diff], axis=1
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
        agg = paddle.concat([node_features, agg], axis=1)
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


class CSPNet(paddle.nn.Layer):
    def __init__(
        self,
        hidden_dim=128,
        time_dim=256,
        text_dim=128,
        num_layers=4,
        max_atoms=103,
        act_fn="silu",
        dis_emb="sin",
        num_freqs=10,
        edge_style="fc",
        cutoff=6.0,
        max_neighbors=20,
        ln=False,
        ip=True,
        smooth=True,
        pred_atom_types=True,
    ):
        super(CSPNet, self).__init__()

        self.ip = ip
        self.smooth = smooth
        if self.smooth:
            self.node_embedding = paddle.nn.Linear(max_atoms, hidden_dim)
        else:
            self.node_embedding = paddle.nn.Embedding(max_atoms, hidden_dim)
        if time_dim > 0 or text_dim > 0:
            self.film_layer = FilmLayer(hidden_dim, time_dim, text_dim)
        if act_fn == "silu":
            self.act_fn = Swish()
        if dis_emb == "sin":
            self.dis_emb = SinusoidsEmbedding(n_frequencies=num_freqs)
        elif dis_emb == "none":
            self.dis_emb = None
        for i in range(0, num_layers):
            setattr(
                self,
                f"csp_layer_{i}",
                CSPLayer(hidden_dim, self.act_fn, self.dis_emb, ln=ln, ip=ip),
            )
        self.num_layers = num_layers
        self.coord_out = paddle.nn.Linear(hidden_dim, 3, bias_attr=False)
        self.lattice_out = paddle.nn.Linear(hidden_dim, 9, bias_attr=False)
        self.type_out = paddle.nn.Linear(hidden_dim, max_atoms)
        self.cutoff = cutoff
        self.max_neighbors = max_neighbors
        self.ln = ln
        self.edge_style = edge_style
        if self.ln:
            self.final_layer_norm = paddle.nn.LayerNorm(hidden_dim)
        self.pred_atom_types = pred_atom_types

    def gen_edges(self, num_atoms, frac_coords):
        # fully connected edges per graph, self-loops included; edge_index,
        # edge2graph and frac_diff all derive from this one ordering
        if self.edge_style != "fc":
            raise NotImplementedError(
                f"Edge style '{self.edge_style}' is not supported; use 'fc'."
            )
        cum_num_atoms = paddle.cumsum(x=num_atoms)
        rows = paddle.arange(int(num_atoms.max()))
        ind1, ind2 = paddle.meshgrid(rows, rows)
        index = paddle.stack(x=[ind1, ind2], axis=0)
        indices_pp = []
        for n, cum_n in zip(num_atoms, cum_num_atoms):
            offset = cum_n - n
            indices_pp.append(index[:, :n, :n].reshape((2, -1)) + offset)
        indices_pp = paddle.concat(x=indices_pp, axis=1)
        fc_edges = indices_pp
        return fc_edges, (frac_coords[fc_edges[1]] - frac_coords[fc_edges[0]]) % 1.0

    def forward(
        self,
        atom_types,
        frac_coords,
        lattices,
        num_atoms,
        node2graph,
        t=None,  # time embeddings can be omitted when training contrastive learning
        text_embeds=None,
    ):
        edges, frac_diff = self.gen_edges(num_atoms, frac_coords)
        return DECODER_OUTPUTS(
            *self.forward_with_edges(
                atom_types,
                frac_coords,
                lattices,
                num_atoms,
                node2graph,
                edges,
                frac_diff,
                t=t,
                text_embeds=text_embeds,
            )
        )

    def forward_with_edges(
        self,
        atom_types,
        frac_coords,
        lattices,
        num_atoms,
        node2graph,
        edges,
        frac_diff,
        t=None,  # time embeddings can be omitted when training contrastive learning
        text_embeds=None,
    ):
        # Numeric body on precomputed edges (static shapes; the runtime
        # boundary compiles exactly this). Returns a plain tuple in
        # DECODER_OUTPUTS field order; eager callers re-wrap.
        edge2graph = node2graph[edges[0]]
        node_features = self.node_embedding(atom_types)

        if t is not None:
            t_per_atom = t.repeat_interleave(num_atoms, dim=0)  # [B_n, time_dim]
        else:
            t_per_atom = None

        if text_embeds is not None:
            text_per_atom = text_embeds.repeat_interleave(
                num_atoms, dim=0
            )  # [B_n, text_dim]
        else:
            text_per_atom = None

        for i in range(0, self.num_layers):
            if t_per_atom is not None or text_per_atom is not None:
                node_features = self.film_layer(
                    node_features, time_embeds=t_per_atom, text_embeds=text_per_atom
                )
            node_features = getattr(self, f"csp_layer_{i}")(
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

        graph_features = scatter_mean(node_features, node2graph, dim=0)
        lattice_out = self.lattice_out(graph_features)
        lattice_out = lattice_out.reshape([-1, 3, 3])
        if self.ip:
            lattice_out = paddle.einsum("bij,bjk->bik", lattice_out, lattices)
        if self.pred_atom_types:
            type_out = self.type_out(node_features)
        else:
            type_out = None

        return (
            type_out,  # atom_types_out [B_n, max_atoms]
            lattice_out,  # [B, 3, 3]
            coord_out,  # [B_n, 3]
            node_features,  # [B_n, hidden_dim]
        )

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

from ppmat.models.common.activation import swish
from ppmat.utils.scatter import scatter_mean

__all__ = ["MiADCSPNet"]


class _MiADSiLU(nn.Silu):
    def forward(self, x):
        # Paddle 3.3 CPU SiLU backward overflows for large negative inputs.
        if x.place.is_cpu_place() and paddle.is_grad_enabled():
            return swish(x)
        return super().forward(x)


class _SinusoidsEmbedding(nn.Layer):
    def __init__(self, n_frequencies: int = 10, n_space: int = 3) -> None:
        super().__init__()
        self.n_frequencies = n_frequencies
        self.n_space = n_space
        self.frequencies = 2 * math.pi * paddle.arange(end=self.n_frequencies)
        self.dim = self.n_frequencies * 2 * self.n_space

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        emb = x.unsqueeze(axis=-1) * self.frequencies[None, None, :]
        emb = emb.reshape([-1, self.n_frequencies * self.n_space])
        return paddle.concat(x=(emb.sin(), emb.cos()), axis=-1)


class MiADCSPLayer(nn.Layer):
    """MiAD message-passing layer without property conditioning."""

    def __init__(
        self,
        hidden_dim: int,
        act_fn: nn.Layer,
        dis_emb: nn.Layer | None = None,
        ln: bool = False,
        ip: bool = True,
    ) -> None:
        super().__init__()
        self.dis_dim = 3
        self.dis_emb = dis_emb
        self.ip = ip
        if dis_emb is not None:
            self.dis_dim = dis_emb.dim
        self.edge_mlp = nn.Sequential(
            nn.Linear(
                in_features=hidden_dim * 2 + 9 + self.dis_dim,
                out_features=hidden_dim,
            ),
            act_fn,
            nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
            act_fn,
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(in_features=hidden_dim * 2, out_features=hidden_dim),
            act_fn,
            nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
            act_fn,
        )
        self.ln = ln
        if self.ln:
            self.layer_norm = nn.LayerNorm(normalized_shape=hidden_dim)

    def forward(
        self,
        node_features: paddle.Tensor,
        frac_coords: paddle.Tensor,
        lattices: paddle.Tensor,
        edge_index: paddle.Tensor,
        edge2graph: paddle.Tensor,
        frac_diff: paddle.Tensor | None = None,
    ) -> paddle.Tensor:
        if frac_diff is None:
            xi, xj = frac_coords[edge_index[0]], frac_coords[edge_index[1]]
            frac_diff = (xj - xi) % 1.0
        node_input = node_features
        if self.ln:
            node_features = self.layer_norm(node_input)

        hi, hj = node_features[edge_index[0]], node_features[edge_index[1]]
        if self.dis_emb is not None:
            frac_diff = self.dis_emb(frac_diff)
        if self.ip:
            lattice_ips = paddle.matmul(lattices, lattices, transpose_y=True)
        else:
            lattice_ips = lattices
        lattice_ips_flatten = lattice_ips.reshape([-1, 9])[edge2graph]
        edge_features = self.edge_mlp(
            paddle.concat(
                x=[hi, hj, lattice_ips_flatten, frac_diff],
                axis=1,
            )
        )
        agg = scatter_mean(
            edge_features,
            edge_index[0],
            dim=0,
            dim_size=node_features.shape[0],
        )
        node_output = self.node_mlp(paddle.concat(x=[node_features, agg], axis=1))
        return node_input + node_output


class MiADCSPNet(nn.Layer):
    """CSPNet denoiser specialized for MiAD.

    Uses a fully connected atom graph built with block-diagonal adjacency and
    the atom-type prediction path required by MiAD. Property conditioning is not
    part of the MiAD architecture.
    """

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
        smooth: bool = False,
        pred_type: bool = False,
        num_classes: int = 100,
    ) -> None:
        super().__init__()
        if act_fn != "silu":
            raise ValueError("MiAD only supports act_fn='silu'")
        if not smooth or not pred_type:
            raise ValueError("MiAD requires smooth=True and pred_type=True")
        if edge_style != "fc":
            raise ValueError("MiAD only supports edge_style='fc'")
        if num_layers < 1:
            raise ValueError("MiAD requires at least one message-passing layer")

        self.ip = ip
        self.smooth = smooth
        self.num_classes = num_classes
        self.node_embedding = nn.Linear(
            in_features=num_classes, out_features=hidden_dim
        )
        self.atom_latent_emb = nn.Linear(
            in_features=hidden_dim + latent_dim, out_features=hidden_dim
        )
        self.act_fn = _MiADSiLU()
        if dis_emb == "sin":
            self.dis_emb = _SinusoidsEmbedding(n_frequencies=num_freqs)
        elif dis_emb == "none":
            self.dis_emb = None
        else:
            raise ValueError(f"Unsupported MiAD distance embedding: {dis_emb}")
        for i in range(num_layers):
            self.add_sublayer(
                name=f"csp_layer_{i}",
                sublayer=MiADCSPLayer(
                    hidden_dim=hidden_dim,
                    act_fn=self.act_fn,
                    dis_emb=self.dis_emb,
                    ln=ln,
                    ip=ip,
                ),
            )
        self.num_layers = num_layers
        self.coord_out = nn.Linear(
            in_features=hidden_dim, out_features=3, bias_attr=False
        )
        self.lattice_out = nn.Linear(
            in_features=hidden_dim, out_features=9, bias_attr=False
        )
        self.pred_type = pred_type
        self.ln = ln
        self.edge_style = edge_style
        if self.ln:
            self.final_layer_norm = nn.LayerNorm(normalized_shape=hidden_dim)
        self.type_out = nn.Linear(in_features=hidden_dim, out_features=num_classes)

    def gen_edges(
        self, num_atoms: paddle.Tensor, frac_coords: paddle.Tensor
    ) -> tuple[paddle.Tensor, paddle.Tensor]:
        adjacency_blocks = [
            paddle.ones([int(n), int(n)], dtype="int64") for n in num_atoms
        ]
        fc_graph = paddle.block_diag(adjacency_blocks)
        fc_edges = paddle.nonzero(fc_graph).t()
        return fc_edges, frac_coords[fc_edges[1]] - frac_coords[fc_edges[0]]

    def forward_with_edges(
        self,
        time_emb: paddle.Tensor,
        atom_types: paddle.Tensor,
        frac_coords: paddle.Tensor,
        lattices: paddle.Tensor,
        num_atoms: paddle.Tensor,
        node2graph: paddle.Tensor,
        edges: paddle.Tensor,
        frac_diff: paddle.Tensor,
    ):
        edge2graph = node2graph[edges[0]]
        node_features = self.node_embedding(atom_types)
        t_per_atom = time_emb.repeat_interleave(repeats=num_atoms, axis=0)
        node_features = self.atom_latent_emb(
            paddle.concat(x=[node_features, t_per_atom], axis=1)
        )
        for i in range(self.num_layers):
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
        graph_features = scatter_mean(
            node_features,
            node2graph,
            dim=0,
            dim_size=lattices.shape[0],
        )
        lattice_out = self.lattice_out(graph_features).reshape([-1, 3, 3])
        if self.ip:
            lattice_out = paddle.bmm(lattice_out, lattices)
        type_out = self.type_out(node_features)
        return lattice_out, coord_out, type_out

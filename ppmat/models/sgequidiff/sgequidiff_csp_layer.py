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

"""CSPLayer for SGEquiDiff's CSPNet drift module.

Distinct from DiffCSP's own implementation (``ppmat.models.diffcsp.diffcsp``);
the two models use different lattice representations and do not share code.
"""
import math

import paddle
import paddle.nn as nn


class SinusoidsEmbedding(nn.Layer):
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


class CSPLayer(nn.Layer):
    """Message passing layer for CSPNet.

    Args:
        hidden_dim: Hidden dimension.
        prop_dim: Property embedding dimension.
        act_fn: Activation function.
        dis_emb: Distance embedding module (optional).
        ln: Whether to use LayerNorm.
        use_lattice_ip: If True, use 9-dim lattice inner products (DiffCSP).
                        If False, use 6-dim lattice representation (SGEquiDiff).
        lattice_dim: Dimension of lattice representation
                     (9 for inner products, 6 for lengths+angles).
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        prop_dim: int = 512,
        act_fn: nn.Layer = None,
        dis_emb: nn.Layer = None,
        ln: bool = False,
        use_lattice_ip: bool = True,
        lattice_dim: int = 9,
    ):
        super().__init__()
        self.dis_dim = 3
        self.dis_emb = dis_emb
        self.use_lattice_ip = use_lattice_ip
        self.lattice_dim = lattice_dim
        if dis_emb is not None:
            self.dis_dim = dis_emb.dim

        self.edge_mlp = nn.Sequential(
            nn.Linear(
                in_features=hidden_dim * 2 + self.lattice_dim + self.dis_dim,
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

        self.prop_mlp = nn.Sequential(
            nn.Linear(in_features=prop_dim, out_features=hidden_dim),
            act_fn,
            nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
            act_fn,
        )

        self.ln = ln
        if self.ln:
            self.layer_norm = nn.LayerNorm(normalized_shape=hidden_dim)

    def edge_model(
        self,
        node_features: paddle.Tensor,
        frac_coords: paddle.Tensor,
        lattices: paddle.Tensor,
        edge_index: paddle.Tensor,
        edge2graph: paddle.Tensor,
        frac_diff: paddle.Tensor = None,
    ) -> paddle.Tensor:
        hi, hj = node_features[edge_index[0]], node_features[edge_index[1]]
        if frac_diff is None:
            xi, xj = frac_coords[edge_index[0]], frac_coords[edge_index[1]]
            frac_diff = (xj - xi) % 1.0
        if self.dis_emb is not None:
            frac_diff = self.dis_emb(frac_diff)

        if self.use_lattice_ip:
            # DiffCSP: 9-dim lattice inner products
            x = lattices
            perm_0 = list(range(x.ndim))
            perm_0[-1] = -2
            perm_0[-2] = -1
            lattice_ips = x @ x.transpose(perm=perm_0)
        else:
            # SGEquiDiff: 6-dim lattice representation (lengths + angles)
            lattice_ips = lattices

        lattice_ips_flatten = lattice_ips.reshape([-1, self.lattice_dim])
        lattice_ips_flatten_edges = lattice_ips_flatten[edge2graph]
        edges_input = paddle.concat(
            x=[hi, hj, lattice_ips_flatten_edges, frac_diff], axis=1
        )
        edge_features = self.edge_mlp(edges_input)
        return edge_features

    def node_model(
        self,
        node_features: paddle.Tensor,
        edge_features: paddle.Tensor,
        edge_index: paddle.Tensor,
    ) -> paddle.Tensor:
        agg = paddle.geometric.segment_mean(edge_features, edge_index[0])
        agg = paddle.concat(x=[node_features, agg], axis=1)
        out = self.node_mlp(agg)
        return out

    def forward(
        self,
        node_features: paddle.Tensor,
        frac_coords: paddle.Tensor,
        lattices: paddle.Tensor,
        edge_index: paddle.Tensor,
        edge2graph: paddle.Tensor,
        frac_diff: paddle.Tensor = None,
        num_atoms: paddle.Tensor = None,
        property_emb: paddle.Tensor = None,
        property_mask: paddle.Tensor = None,
    ) -> paddle.Tensor:
        if property_emb is not None:
            property_features = self.prop_mlp(property_emb)
            if property_mask is not None:
                property_features = property_features * property_mask
            property_features = paddle.repeat_interleave(
                property_features, num_atoms, axis=0
            )
            node_features = node_features + property_features

        node_input = node_features
        if self.ln:
            node_features = self.layer_norm(node_input)
        edge_features = self.edge_model(
            node_features, frac_coords, lattices, edge_index, edge2graph, frac_diff
        )
        node_output = self.node_model(node_features, edge_features, edge_index)
        return node_input + node_output

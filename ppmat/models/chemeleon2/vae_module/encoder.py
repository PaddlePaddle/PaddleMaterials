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

import paddle.nn as nn

from ..common import get_index_embedding
from ..common import make_attn_mask
from ..common import set_gelu_approx
from ..common import to_dense_batch


class Chemeleon2TransformerEncoder(nn.Layer):
    def __init__(
        self,
        max_num_elements=100,
        d_model=512,
        nhead=8,
        dim_feedforward=2048,
        activation="gelu",
        dropout=0.0,
        norm_first=True,
        num_layers=8,
    ):
        super().__init__()

        self.max_num_elements = max_num_elements
        self.d_model = d_model
        self.num_layers = num_layers
        self.atom_type_embedder = nn.Embedding(max_num_elements, d_model)
        self.lattices_embedder = nn.Sequential(
            nn.Linear(9, d_model, bias_attr=False),
            nn.Silu(),
            nn.Linear(d_model, d_model),
        )
        self.frac_coords_embedder = nn.Sequential(
            nn.Linear(3, d_model, bias_attr=False),
            nn.Silu(),
            nn.Linear(d_model, d_model),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            normalize_before=norm_first,
        )

        layer_norm = nn.LayerNorm(d_model)
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=layer_norm,
        )

        if activation == "gelu":
            set_gelu_approx(self.transformer)

    @property
    def hidden_dim(self):
        return self.d_model

    def forward(self, batch):
        atom_types = batch.atom_types
        lattices = batch.lattices
        frac_coords = batch.frac_coords
        token_idx = batch.token_idx
        batch_idx = batch.batch
        num_atoms = batch.num_atoms

        x = self.atom_type_embedder(atom_types)
        x += self.lattices_embedder(lattices.reshape([-1, 9]))[batch_idx]
        x += self.frac_coords_embedder(frac_coords)

        x += get_index_embedding(token_idx, self.d_model)

        x_dense, token_mask = to_dense_batch(x, batch_idx)

        attn_mask = make_attn_mask(token_mask)
        x_out = self.transformer(x_dense, src_mask=attn_mask)

        x = x_out[token_mask]

        return {
            "x": x,
            "num_atoms": num_atoms,
            "batch": batch_idx,
            "token_idx": token_idx,
        }

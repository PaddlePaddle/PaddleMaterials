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

from ppmat.utils.scatter import scatter_mean

from ..common import get_index_embedding
from ..common import make_attn_mask
from ..common import set_gelu_approx
from ..common import to_dense_batch


class Chemeleon2TransformerDecoder(nn.Layer):
    def __init__(
        self,
        atom_type_predict=True,
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
        self.atom_type_predict = atom_type_predict

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

        if atom_type_predict:
            self.atom_types_head = nn.Linear(d_model, max_num_elements, bias_attr=True)
        self.frac_coords_head = nn.Linear(d_model, 3, bias_attr=False)
        self.lattice_head = nn.Linear(d_model, 6, bias_attr=False)

    @property
    def hidden_dim(self):
        return self.d_model

    def forward(self, encoded_batch):
        # Out-of-place add: the caller may reuse the passed dict, so the
        # positional embedding must not be baked into encoded_batch["x"].
        x = encoded_batch["x"] + get_index_embedding(
            encoded_batch["token_idx"], self.d_model
        )

        x_dense, token_mask = to_dense_batch(x, encoded_batch["batch"])

        attn_mask = make_attn_mask(token_mask)
        x_out = self.transformer(x_dense, src_mask=attn_mask)

        x = x_out[token_mask]

        x_global = scatter_mean(x, encoded_batch["batch"], dim=0)

        if self.atom_type_predict:
            atom_types_out = self.atom_types_head(x)
        else:
            atom_types_out = None

        lattices_out = self.lattice_head(x_global)

        frac_coords_out = self.frac_coords_head(x)

        result = {
            "atom_types": atom_types_out,
            "lattices": lattices_out,
            "lengths": lattices_out[:, :3],
            "angles": lattices_out[:, 3:],
            "frac_coords": frac_coords_out,
        }
        return result

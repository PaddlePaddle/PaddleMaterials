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

from __future__ import annotations

import paddle
import paddle.nn as nn

from ppmat.models.liflow.geometry import get_unit_vectors_and_lengths
from ppmat.models.liflow.layers import BesselBasis
from ppmat.models.liflow.layers import CosineCutoff
from ppmat.models.liflow.layers import DualMessageBlock
from ppmat.models.liflow.layers import GatedEquivariantBlock
from ppmat.models.liflow.layers import GaussianFourierBasis
from ppmat.models.liflow.layers import UpdateBlock

__all__ = ["DualPaiNN"]


class DualPaiNN(nn.Layer):
    def __init__(
        self,
        num_features: int,
        num_radial_basis: int,
        num_layers: int,
        num_elements: int,
        r_max: float,
        r_offset: float = 0.0,
        ref_temp: float = 1000.0,
    ):
        super().__init__()
        self.num_features = num_features
        self.num_radial_basis = num_radial_basis
        self.num_layers = num_layers
        self.r_max = r_max
        self.r_offset = r_offset
        self.ref_temp = ref_temp

        assert num_features % 2 == 0, "Number of features must be even"
        self.atom_embedding = nn.Embedding(num_elements, num_features)
        self.time_embedding = GaussianFourierBasis(num_basis=num_features // 2)
        self.temp_embedding = GaussianFourierBasis(num_basis=num_features // 2)
        self.radial_embedding = BesselBasis(num_basis=num_radial_basis, r_max=r_max)
        self.cutoff_fn = CosineCutoff(r_max=r_max)
        self.linear_v = nn.Linear(1, num_features, bias_attr=False)

        messages, updates = [], []
        for _ in range(num_layers):
            messages.append(DualMessageBlock(num_features, num_radial_basis))
            updates.append(UpdateBlock(num_features))
        self.messages = nn.LayerList(messages)
        self.updates = nn.LayerList(updates)
        self.output_block = GatedEquivariantBlock(
            num_scalar_inputs=num_features,
            num_vector_inputs=num_features,
        )

    def forward(
        self,
        condition_positions,
        flow_positions,
        edge_index,
        shifts,
        elements,
        node_time,
        node_temperature,
    ):
        unit_vectors_1, lengths_1 = get_unit_vectors_and_lengths(
            condition_positions, edge_index, shifts
        )
        unit_vectors_2, lengths_2 = get_unit_vectors_and_lengths(
            flow_positions, edge_index, shifts
        )

        lengths_1 = (lengths_1 + self.r_offset).clip(max=self.r_max)
        lengths_2 = (lengths_2 + self.r_offset).clip(max=self.r_max)
        radial_embeddings_1 = self.radial_embedding(lengths_1)
        radial_embeddings_2 = self.radial_embedding(lengths_2)
        f_cut_1 = self.cutoff_fn(lengths_1)
        f_cut_2 = self.cutoff_fn(lengths_2)

        s_atom = self.atom_embedding(elements)
        s_time = self.time_embedding(node_time)
        s_temp = self.temp_embedding(node_temperature / self.ref_temp)
        s = s_atom + paddle.concat([s_time, s_temp], axis=-1)
        s = s[:, None, :]
        positions_diff = flow_positions - condition_positions
        v = self.linear_v(positions_diff[..., None])

        for message, update in zip(self.messages, self.updates):
            s, v = message(
                s,
                v,
                radial_embeddings_1,
                radial_embeddings_2,
                f_cut_1,
                f_cut_2,
                unit_vectors_1,
                unit_vectors_2,
                edge_index,
            )
            s, v = update(s, v)
        v_out = self.output_block(s, v).squeeze(-1)
        return v_out

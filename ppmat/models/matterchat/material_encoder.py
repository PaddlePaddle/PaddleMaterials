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

"""Checkpoint-compatible MatterChat features from the framework CHGNet."""

from dataclasses import asdict
from dataclasses import dataclass

import paddle
from paddle import nn

from ppmat.models.chgnet.chgnet import CHGNet
from ppmat.models.chgnet.chgnet import GatedMLP
from ppmat.models.chgnet.chgnet_graph_converter import CHGNetGraphConverter


@dataclass
class MaterialEncoderConfig:
    """CHGNet feature settings matching the released MatterChat checkpoint."""

    atom_fea_dim: int = 64
    bond_fea_dim: int = 64
    angle_fea_dim: int = 64
    composition_model: str = "MPtrj"
    num_radial: int = 9
    num_angular: int = 9
    n_conv: int = 4
    atom_conv_hidden_dim: int = 64
    update_bond: bool = True
    bond_conv_hidden_dim: int = 64
    update_angle: bool = True
    angle_layer_hidden_dim: int = 0
    conv_dropout: float = 0.0
    read_out: str = "ave"
    mlp_hidden_dims: tuple = (64, 64)
    mlp_dropout: float = 0.0
    mlp_first: bool = True
    is_intensive: bool = True
    atom_graph_cutoff: float = 5.0
    bond_graph_cutoff: float = 3.0
    cutoff_coeff: float = 5.0
    learnable_rbf: bool = True
    mlp_out_bias: bool = True


class MatterChatEncoder(CHGNet):
    """Reuse CHGNet graph construction and message passing with dialogue weights."""

    def __init__(self, config=None):
        config = MaterialEncoderConfig(**({} if config is None else config))
        if config.n_conv < 1:
            raise ValueError("n_conv must be positive.")
        super().__init__(**asdict(config), property_names=["energy_per_atom"])
        # Released MatterChat weights contain neither gated nor readout norms.
        for layer in self.sublayers():
            if isinstance(layer, GatedMLP):
                layer.bn1 = nn.Identity()
                layer.bn2 = nn.Identity()
        self.readout_norm = None
        self.graph_converter = CHGNetGraphConverter(
            atom_graph_cutoff=config.atom_graph_cutoff,
            bond_graph_cutoff=config.bond_graph_cutoff,
        )

    @paddle.no_grad()
    def predict_structure_embedding(self, structure):
        """Return final per-atom features, preserving the dialogue encoder API."""
        features = []

        def capture_features(layer, inputs, output):
            features.append(output)

        hook = self.atom_conv_layers[-1].register_forward_post_hook(capture_features)
        try:
            self._forward({"graph": self.graph_converter(structure)})
        finally:
            hook.remove()
        return features[0]

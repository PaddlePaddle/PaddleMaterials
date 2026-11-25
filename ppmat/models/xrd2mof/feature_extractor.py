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
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import pgl
from pgl.nn import GCNConv
from pgl.nn.functional import degree_norm

from ppmat.utils.scatter import scatter_mean

# Input contract of the upstream feature extractor; the encoder widths that
# consume these dimensions are locked by the released checkpoint.
_XRD_POOLED_LENGTH = 82
_METAL_FEATURE_DIM = 65
_LINKER_NODE_FEATURES = 37


class LinkerEncoder(nn.Layer):
    """Encode a batch of linker graphs with normalized graph convolutions."""

    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(input_size=_LINKER_NODE_FEATURES, output_size=64)
        self.conv2 = GCNConv(input_size=64, output_size=128)
        self.conv3 = GCNConv(input_size=128, output_size=64)
        self.fc1 = nn.Linear(in_features=64, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=64)
        self.norm = nn.LayerNorm(normalized_shape=64)

    def forward(self, x, edge_index, batch):
        edge_index = edge_index[:, edge_index[0] != edge_index[1]]
        nodes = paddle.arange(x.shape[0])
        edges = paddle.concat([edge_index, paddle.stack([nodes, nodes])], axis=1)
        graph = pgl.Graph(num_nodes=x.shape[0], edges=edges.t())
        norm = degree_norm(graph)
        x = F.relu(x=self.conv1(graph, x, norm))
        x = F.relu(x=self.conv2(graph, x, norm))
        x = F.relu(x=self.conv3(graph, x, norm))
        graph_features = scatter_mean(x, batch, dim=0)
        x = F.relu(x=self.fc1(graph_features))
        x = F.relu(x=self.fc2(x))
        x = self.norm(x)
        return x


class MetalEncoder(nn.Layer):
    """Encode the fixed metal-node descriptor with one-dimensional convolutions."""

    def __init__(self):
        super().__init__()
        self.ln_in = nn.LayerNorm(normalized_shape=_METAL_FEATURE_DIM)
        self.conv1 = nn.Conv1D(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1D(num_features=16)
        self.conv2 = nn.Conv1D(
            in_channels=16, out_channels=32, kernel_size=3, padding=1
        )
        self.bn2 = nn.BatchNorm1D(num_features=32)
        self.conv3 = nn.Conv1D(
            in_channels=32, out_channels=32, kernel_size=3, padding=1
        )
        self.bn3 = nn.BatchNorm1D(num_features=32)
        self.fc1 = nn.Linear(in_features=32 * _METAL_FEATURE_DIM, out_features=1024)
        self.fc2 = nn.Linear(in_features=1024, out_features=512)
        self.fc3 = nn.Linear(in_features=512, out_features=256)
        self.fc4 = nn.Linear(in_features=256, out_features=128)
        self.fc5 = nn.Linear(in_features=128, out_features=64)
        self.layer_norm = nn.LayerNorm(normalized_shape=64)

    def forward(self, x):
        x = self.ln_in(x)
        x = x.unsqueeze(1)
        x = F.relu(x=self.bn1(self.conv1(x)))
        x = F.relu(x=self.bn2(self.conv2(x)))
        x = F.relu(x=self.bn3(self.conv3(x)))
        x = x.reshape([x.shape[0], -1])
        x = F.relu(x=self.fc1(x))
        x = F.relu(x=self.fc2(x))
        x = F.relu(x=self.fc3(x))
        x = F.relu(x=self.fc4(x))
        x = F.relu(x=self.fc5(x))
        x = self.layer_norm(x)
        return x


class XrdEncoder(nn.Layer):
    """Encode the PXRD intensity sequence with pooled convolutions."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1D(in_channels=1, out_channels=4, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool1D(kernel_size=4, stride=4)
        self.bn1 = nn.BatchNorm1D(num_features=4)
        self.conv2 = nn.Conv1D(in_channels=4, out_channels=8, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool1D(kernel_size=4, stride=4)
        self.bn2 = nn.BatchNorm1D(num_features=8)
        self.conv3 = nn.Conv1D(in_channels=8, out_channels=16, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool1D(kernel_size=2, stride=2)
        self.bn3 = nn.BatchNorm1D(num_features=16)
        self.conv4 = nn.Conv1D(
            in_channels=16, out_channels=32, kernel_size=3, padding=1
        )
        self.pool4 = nn.MaxPool1D(kernel_size=2, stride=2)
        self.bn4 = nn.BatchNorm1D(num_features=32)
        self.fc = nn.Linear(in_features=32 * _XRD_POOLED_LENGTH, out_features=1024)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = F.relu(x=self.conv1(x))
        x = self.pool1(x)
        x = F.relu(x=self.bn1(x))
        x = F.relu(x=self.conv2(x))
        x = self.pool2(x)
        x = F.relu(x=self.bn2(x))
        x = F.relu(x=self.conv3(x))
        x = self.pool3(x)
        x = F.relu(x=self.bn3(x))
        x = F.relu(x=self.conv4(x))
        x = self.pool4(x)
        x = F.relu(x=self.bn4(x))
        x = x.reshape([x.shape[0], -1])
        x = self.fc(x)
        return x


class ProjectionHead(nn.Layer):
    """Project fused descriptors through a residual normalized multilayer head."""

    def __init__(self, embedding_dim, projection_dim):
        super().__init__()
        self.projection = nn.Linear(
            in_features=embedding_dim, out_features=projection_dim
        )
        self.gelu = nn.GELU()
        self.fc1 = nn.Linear(in_features=projection_dim, out_features=projection_dim)
        self.fc2 = nn.Linear(in_features=projection_dim, out_features=projection_dim)
        self.batch_norm = nn.BatchNorm1D(num_features=projection_dim)
        self.layer_norm = nn.LayerNorm(normalized_shape=projection_dim)

    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.fc2(x)
        x = self.batch_norm(x)
        x = x + projected
        x = self.layer_norm(x)
        return x


class Xrd2MofFeatureExtractor(nn.Layer):
    """XRD-conditioned feature extractor for coarse-grained structure generation.

    Fuses the PXRD pattern with the metal-node and organic-linker priors into the
    256-dim conditioning embedding consumed by `Xrd2Mof`. The released weights
    are loaded for inference; this repository does not train the extractor.

    Args:
        projection_dim (int): Output embedding dimension. Defaults to 256.
        fusion_hidden_dim (int): Hidden dimension of the fusion network.
            Defaults to 512.
    """

    def __init__(
        self,
        projection_dim: int = 256,
        fusion_hidden_dim: int = 512,
    ):
        super().__init__()

        self.xrd_encoder = XrdEncoder()
        self.metal_encoder = MetalEncoder()
        self.linker_encoder = LinkerEncoder()
        combined_features = (
            self.xrd_encoder.fc.weight.shape[1]
            + self.metal_encoder.fc5.weight.shape[1]
            + self.linker_encoder.fc2.weight.shape[1]
        )

        self.feature_fusion = nn.Sequential(
            nn.Linear(in_features=combined_features, out_features=fusion_hidden_dim),
            nn.ReLU(),
            nn.Linear(in_features=fusion_hidden_dim, out_features=combined_features),
        )

        self.feature_projection = ProjectionHead(
            embedding_dim=combined_features,
            projection_dim=projection_dim,
        )

    def forward(self, batch):
        xrd_features = self.xrd_encoder(batch["xrd"])
        metal_features = self.metal_encoder(batch["metal"])
        linker = batch["linker"]
        linker_features = self.linker_encoder(
            linker["x"], linker["edge_index"], linker["batch"]
        )

        combined_features = paddle.concat(
            x=[xrd_features, metal_features, linker_features], axis=1
        )
        combined_features = self.feature_fusion(combined_features)
        return {"feature_embedding": self.feature_projection(combined_features)}

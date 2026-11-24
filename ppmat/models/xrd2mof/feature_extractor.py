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
import paddle.nn.functional as F

from ppmat.utils.scatter import scatter_mean
from ppmat.utils.scatter import scatter_sum

# Input contract of the upstream feature extractor; the encoder widths that
# consume these dimensions are locked by the released checkpoint.
_XRD_POOLED_LENGTH = 82
_METAL_FEATURE_DIM = 65
_LINKER_NODE_FEATURES = 37


class GCNConv(paddle.nn.Layer):
    """Graph convolution with symmetric normalization.

    Equivalent to `torch_geometric.nn.GCNConv` with `add_self_loops=True`,
    `normalize=True` and no edge weights.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.lin = paddle.nn.Linear(
            in_features=in_channels, out_features=out_channels, bias_attr=False
        )
        self.bias = self.create_parameter(shape=[out_channels])

    def forward(self, x, edge_index):
        num_nodes = x.shape[0]
        h = self.lin(x)
        row, col = edge_index[0], edge_index[1]
        mask = row != col
        row, col = row[mask], col[mask]
        loop = paddle.arange(end=num_nodes)
        row = paddle.concat(x=[row, loop], axis=0)
        col = paddle.concat(x=[col, loop], axis=0)

        ones = paddle.ones_like(x=col, dtype=h.dtype)
        deg = scatter_sum(ones, col, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = paddle.pow(x=deg, y=-0.5)
        deg_inv_sqrt = paddle.where(
            paddle.isinf(x=deg_inv_sqrt),
            paddle.zeros_like(x=deg_inv_sqrt),
            deg_inv_sqrt,
        )
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        out = scatter_sum(norm.unsqueeze(-1) * h[row], col, dim=0, dim_size=num_nodes)
        return out + self.bias


class LinkerEncoder(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(in_channels=_LINKER_NODE_FEATURES, out_channels=64)
        self.conv2 = GCNConv(in_channels=64, out_channels=128)
        self.conv3 = GCNConv(in_channels=128, out_channels=64)
        self.fc1 = paddle.nn.Linear(in_features=64, out_features=128)
        self.fc2 = paddle.nn.Linear(in_features=128, out_features=64)
        self.norm = paddle.nn.LayerNorm(normalized_shape=64)

    def forward(self, x, edge_index, batch):
        x = F.relu(x=self.conv1(x, edge_index))
        x = F.relu(x=self.conv2(x, edge_index))
        x = F.relu(x=self.conv3(x, edge_index))
        graph_features = scatter_mean(x, batch, dim=0)
        x = F.relu(x=self.fc1(graph_features))
        x = F.relu(x=self.fc2(x))
        x = self.norm(x)
        return x


class MetalEncoder(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.ln_in = paddle.nn.LayerNorm(normalized_shape=_METAL_FEATURE_DIM)
        self.conv1 = paddle.nn.Conv1D(
            in_channels=1, out_channels=16, kernel_size=3, padding=1
        )
        self.bn1 = paddle.nn.BatchNorm1D(num_features=16)
        self.conv2 = paddle.nn.Conv1D(
            in_channels=16, out_channels=32, kernel_size=3, padding=1
        )
        self.bn2 = paddle.nn.BatchNorm1D(num_features=32)
        self.conv3 = paddle.nn.Conv1D(
            in_channels=32, out_channels=32, kernel_size=3, padding=1
        )
        self.bn3 = paddle.nn.BatchNorm1D(num_features=32)
        self.fc1 = paddle.nn.Linear(
            in_features=32 * _METAL_FEATURE_DIM, out_features=1024
        )
        self.fc2 = paddle.nn.Linear(in_features=1024, out_features=512)
        self.fc3 = paddle.nn.Linear(in_features=512, out_features=256)
        self.fc4 = paddle.nn.Linear(in_features=256, out_features=128)
        self.fc5 = paddle.nn.Linear(in_features=128, out_features=64)
        self.layer_norm = paddle.nn.LayerNorm(normalized_shape=64)

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


class XrdEncoder(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.conv1d = paddle.nn.Conv1D(
            in_channels=1, out_channels=4, kernel_size=3, padding=1
        )
        self.pool1 = paddle.nn.MaxPool1D(kernel_size=4, stride=4)
        self.bn1 = paddle.nn.BatchNorm1D(num_features=4)
        self.conv2d = paddle.nn.Conv1D(
            in_channels=4, out_channels=8, kernel_size=3, padding=1
        )
        self.pool2 = paddle.nn.MaxPool1D(kernel_size=4, stride=4)
        self.bn2 = paddle.nn.BatchNorm1D(num_features=8)
        self.conv3d = paddle.nn.Conv1D(
            in_channels=8, out_channels=16, kernel_size=3, padding=1
        )
        self.pool3 = paddle.nn.MaxPool1D(kernel_size=2, stride=2)
        self.bn3 = paddle.nn.BatchNorm1D(num_features=16)
        self.conv4d = paddle.nn.Conv1D(
            in_channels=16, out_channels=32, kernel_size=3, padding=1
        )
        self.pool4 = paddle.nn.MaxPool1D(kernel_size=2, stride=2)
        self.bn4 = paddle.nn.BatchNorm1D(num_features=32)
        self.fc = paddle.nn.Linear(
            in_features=32 * _XRD_POOLED_LENGTH, out_features=1024
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = F.relu(x=self.conv1d(x))
        x = self.pool1(x)
        x = F.relu(x=self.bn1(x))
        x = F.relu(x=self.conv2d(x))
        x = self.pool2(x)
        x = F.relu(x=self.bn2(x))
        x = F.relu(x=self.conv3d(x))
        x = self.pool3(x)
        x = F.relu(x=self.bn3(x))
        x = F.relu(x=self.conv4d(x))
        x = self.pool4(x)
        x = F.relu(x=self.bn4(x))
        x = x.reshape([x.shape[0], -1])
        x = self.fc(x)
        return x


class ProjectionHead(paddle.nn.Layer):
    def __init__(self, embedding_dim, projection_dim):
        super().__init__()
        self.projection = paddle.nn.Linear(
            in_features=embedding_dim, out_features=projection_dim
        )
        self.gelu = paddle.nn.GELU()
        self.fc1 = paddle.nn.Linear(
            in_features=projection_dim, out_features=projection_dim
        )
        self.fc2 = paddle.nn.Linear(
            in_features=projection_dim, out_features=projection_dim
        )
        self.batch_norm = paddle.nn.BatchNorm1D(num_features=projection_dim)
        self.layer_norm = paddle.nn.LayerNorm(normalized_shape=projection_dim)

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


class Xrd2MofFeatureExtractor(paddle.nn.Layer):
    """XRD-conditioned feature extractor for coarse-grained structure generation.

    Fuses the PXRD pattern with the metal-node and organic-linker priors into the
    256-dim conditioning embedding consumed by `Xrd2Mof`. The released weights
    are loaded for inference; this repository does not train the extractor.

    Args:
        combined_features (int): Concatenated XRD, metal and linker feature
            dimension. Defaults to 1152.
        projection_dim (int): Output embedding dimension. Defaults to 256.
    """

    def __init__(
        self,
        combined_features: int = 1152,
        projection_dim: int = 256,
    ):
        super().__init__()

        self.xrd_encoder = XrdEncoder()
        self.metal_encoder = MetalEncoder()
        self.linker_encoder = LinkerEncoder()

        self.feature_fusion = paddle.nn.Sequential(
            paddle.nn.Linear(in_features=combined_features, out_features=512),
            paddle.nn.ReLU(),
            paddle.nn.Linear(in_features=512, out_features=combined_features),
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

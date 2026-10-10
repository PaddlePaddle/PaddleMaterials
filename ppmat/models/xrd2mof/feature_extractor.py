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


class LatticeEncoder(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.conv1 = paddle.nn.Conv1D(
            in_channels=3, out_channels=8, kernel_size=3, padding=1
        )
        self.bn1 = paddle.nn.BatchNorm1D(num_features=8)
        self.conv2 = paddle.nn.Conv1D(
            in_channels=8, out_channels=16, kernel_size=3, padding=1
        )
        self.bn2 = paddle.nn.BatchNorm1D(num_features=16)
        self.conv3 = paddle.nn.Conv1D(
            in_channels=16, out_channels=32, kernel_size=3, padding=1
        )
        self.bn3 = paddle.nn.BatchNorm1D(num_features=32)
        self.conv4 = paddle.nn.Conv1D(
            in_channels=32, out_channels=32, kernel_size=3, padding=1
        )
        self.bn4 = paddle.nn.BatchNorm1D(num_features=32)
        self.conv5 = paddle.nn.Conv1D(
            in_channels=32, out_channels=16, kernel_size=3, padding=1
        )
        self.bn5 = paddle.nn.BatchNorm1D(num_features=16)
        self.fc = paddle.nn.Linear(in_features=16 * 3, out_features=32)

    def forward(self, x):
        x = x.transpose([0, 2, 1])
        x = F.relu(x=self.bn1(self.conv1(x)))
        x = F.relu(x=self.bn2(self.conv2(x)))
        x = F.relu(x=self.bn3(self.conv3(x)))
        x = F.relu(x=self.bn4(self.conv4(x)))
        x = F.relu(x=self.bn5(self.conv5(x)))
        x = F.adaptive_avg_pool1d(x, 3)
        x = x.reshape([x.shape[0], -1])
        x = self.fc(x)
        return x


class AtomEncoder(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.conv1 = paddle.nn.Conv1D(
            in_channels=3, out_channels=16, kernel_size=3, padding=1
        )
        self.bn1 = paddle.nn.BatchNorm1D(num_features=16)
        self.conv2 = paddle.nn.Conv1D(
            in_channels=16, out_channels=32, kernel_size=3, padding=1
        )
        self.bn2 = paddle.nn.BatchNorm1D(num_features=32)
        self.pool = paddle.nn.MaxPool1D(kernel_size=2, stride=2)
        self.conv3 = paddle.nn.Conv1D(
            in_channels=32, out_channels=64, kernel_size=3, padding=1
        )
        self.bn3 = paddle.nn.BatchNorm1D(num_features=64)
        self.conv4 = paddle.nn.Conv1D(
            in_channels=64, out_channels=64, kernel_size=3, padding=1
        )
        self.bn4 = paddle.nn.BatchNorm1D(num_features=64)
        self.conv5 = paddle.nn.Conv1D(
            in_channels=64, out_channels=128, kernel_size=3, padding=1
        )
        self.bn5 = paddle.nn.BatchNorm1D(num_features=128)
        self.conv6 = paddle.nn.Conv1D(
            in_channels=128, out_channels=128, kernel_size=3, padding=1
        )
        self.bn6 = paddle.nn.BatchNorm1D(num_features=128)
        self.global_pool = paddle.nn.AdaptiveAvgPool1D(output_size=1)
        self.fc = paddle.nn.Linear(in_features=128, out_features=256)

    def forward(self, x):
        x = x.transpose([0, 2, 1])
        x = F.relu(x=self.bn1(self.conv1(x)))
        x = F.relu(x=self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = F.relu(x=self.bn3(self.conv3(x)))
        x = F.relu(x=self.bn4(self.conv4(x)))
        x = F.relu(x=self.bn5(self.conv5(x)))
        x = F.relu(x=self.bn6(self.conv6(x)))
        x = self.global_pool(x)
        x = x.reshape([x.shape[0], -1])
        x = self.fc(x)
        return x


class ProjectionHead(paddle.nn.Layer):
    def __init__(self, embedding_dim, projection_dim, dropout):
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
        self.dropout = paddle.nn.Dropout(p=dropout)
        self.layer_norm = paddle.nn.LayerNorm(normalized_shape=projection_dim)

    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.fc2(x)
        x = self.batch_norm(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x


class Xrd2MofFeatureExtractor(paddle.nn.Layer):
    """XRD feature extraction model.

    Aligns structural features (lattice + atom positions) with chemical
    features (XRD pattern + metal node + organic linker) in a shared embedding
    space, so the XRD pattern alone can drive downstream structure generation.

    Args:
        cif_features (int): Concatenated lattice and atom feature dimension.
            Defaults to 288.
        combined_features (int): Concatenated XRD, metal and linker feature
            dimension. Defaults to 1152.
        projection_dim (int): Shared embedding dimension. Defaults to 256.
        dropout (float): Dropout rate of the projection head. Defaults to 0.0.
        temperature (float): Contrastive temperature. Defaults to 1.0.
    """

    def __init__(
        self,
        cif_features: int = 288,
        combined_features: int = 1152,
        projection_dim: int = 256,
        dropout: float = 0.0,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.temperature = temperature

        self.lattice_encoder = LatticeEncoder()
        self.atom_encoder = AtomEncoder()
        self.xrd_encoder = XrdEncoder()
        self.metal_encoder = MetalEncoder()
        self.linker_encoder = LinkerEncoder()

        self.cif_fusion = paddle.nn.Sequential(
            paddle.nn.Linear(in_features=cif_features, out_features=128),
            paddle.nn.ReLU(),
            paddle.nn.Linear(in_features=128, out_features=cif_features),
        )
        self.feature_fusion = paddle.nn.Sequential(
            paddle.nn.Linear(in_features=combined_features, out_features=512),
            paddle.nn.ReLU(),
            paddle.nn.Linear(in_features=512, out_features=combined_features),
        )

        self.cif_projection = ProjectionHead(
            embedding_dim=cif_features,
            projection_dim=projection_dim,
            dropout=dropout,
        )
        self.feature_projection = ProjectionHead(
            embedding_dim=combined_features,
            projection_dim=projection_dim,
            dropout=dropout,
        )

    def _encode(self, batch):
        cif_lattice = batch["cif_lattice"]
        cif_atom_positions = batch["cif_atom_positions"]
        xrd = batch["xrd"]
        metal = batch["metal"]
        linker = batch["linker"]

        lattice_features = self.lattice_encoder(cif_lattice)
        atom_features = self.atom_encoder(cif_atom_positions)
        xrd_features = self.xrd_encoder(xrd)
        metal_features = self.metal_encoder(metal)
        linker_features = self.linker_encoder(
            linker["x"], linker["edge_index"], linker["batch"]
        )

        cif_features = paddle.concat(x=[lattice_features, atom_features], axis=1)
        cif_features = self.cif_fusion(cif_features)
        combined_features = paddle.concat(
            x=[xrd_features, metal_features, linker_features], axis=1
        )
        combined_features = self.feature_fusion(combined_features)

        cif_embeddings = self.cif_projection(cif_features)
        feature_embeddings = self.feature_projection(combined_features)
        return cif_embeddings, feature_embeddings

    def forward(self, batch):
        cif_embeddings, feature_embeddings = self._encode(batch)

        logits = (
            paddle.matmul(feature_embeddings, cif_embeddings, transpose_y=True)
            / self.temperature
        )
        cifs_similarity = paddle.matmul(
            cif_embeddings, cif_embeddings, transpose_y=True
        )
        features_similarity = paddle.matmul(
            feature_embeddings, feature_embeddings, transpose_y=True
        )
        targets = F.softmax(
            (cifs_similarity + features_similarity) / 2 * self.temperature, axis=-1
        )
        features_loss = F.cross_entropy(
            logits, targets, soft_label=True, reduction="none"
        )
        cifs_loss = F.cross_entropy(
            logits.T, targets, soft_label=True, reduction="none"
        )
        loss = (cifs_loss + features_loss) / 2.0
        loss_dict = {
            "loss": loss.mean(),
            "loss_cif": cifs_loss.mean(),
            "loss_feature": features_loss.mean(),
        }
        pred_dict = {
            "cif_embedding": cif_embeddings,
            "feature_embedding": feature_embeddings,
        }
        return {"loss_dict": loss_dict, "pred_dict": pred_dict}

    @paddle.no_grad()
    def extract(self, batch):
        cif_embeddings, feature_embeddings = self._encode(batch)
        return {
            "cif_embedding": cif_embeddings,
            "feature_embedding": feature_embeddings,
        }

# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This code is adapted from https://github.com/divelab/AIRS/tree/main/OpenMat/ComFormer

import math
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
import paddle
import paddle.nn as nn

from ppmat.models.common.message_passing.message_passing import MessagePassing
from ppmat.models.common.runtime import RuntimeMixin
from ppmat.models.common.runtime import runtime_boundary
from ppmat.utils.pgl_compat import graph_count
from ppmat.utils.scatter import scatter


class RBFExpansion(paddle.nn.Layer):
    """Expand interatomic distances with radial basis functions."""

    def __init__(
        self,
        vmin: float = 0,
        vmax: float = 8,
        bins: int = 40,
        lengthscale: Optional[float] = None,
    ):
        super().__init__()
        self.vmin = vmin
        self.vmax = vmax
        self.bins = bins
        self.register_buffer(
            name="centers",
            tensor=paddle.linspace(start=self.vmin, stop=self.vmax, num=self.bins),
        )
        if lengthscale is None:
            self.lengthscale = np.diff(self.centers).mean()
            self.gamma = 1 / self.lengthscale
        else:
            self.lengthscale = lengthscale
            self.gamma = 1 / lengthscale**2

    def forward(self, distance: paddle.Tensor) -> paddle.Tensor:
        """Apply RBF expansion to interatomic distance tensor."""
        return paddle.exp(
            x=-self.gamma * (distance.unsqueeze(axis=1) - self.centers) ** 2
        )


class ComformerConv(MessagePassing):
    _alpha: paddle.Tensor

    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        heads: int = 1,
        concat: bool = True,
        beta: bool = False,
        dropout: float = 0.0,
        edge_dim: Optional[int] = None,
        root_weight: bool = True,
        **kwargs,
    ):
        kwargs.setdefault("aggr", "add")
        super(ComformerConv, self).__init__(node_dim=0, **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.beta = beta and root_weight
        self.root_weight = root_weight
        self.concat = concat
        self.dropout = dropout
        self.edge_dim = edge_dim
        self._alpha = None
        if isinstance(in_channels, int):
            in_channels = in_channels, in_channels
        self.lin_key = paddle.nn.Linear(
            in_features=in_channels[0], out_features=heads * out_channels
        )
        self.lin_query = paddle.nn.Linear(
            in_features=in_channels[1], out_features=heads * out_channels
        )
        self.lin_value = paddle.nn.Linear(
            in_features=in_channels[0], out_features=heads * out_channels
        )
        self.lin_edge = paddle.nn.Linear(
            in_features=edge_dim, out_features=heads * out_channels
        )
        self.lin_concate = paddle.nn.Linear(
            in_features=heads * out_channels, out_features=out_channels
        )
        self.lin_msg_update = paddle.nn.Sequential(
            paddle.nn.Linear(in_features=out_channels * 3, out_features=out_channels),
            paddle.nn.Silu(),
            paddle.nn.Linear(in_features=out_channels, out_features=out_channels),
        )
        self.softplus = paddle.nn.Softplus()
        self.silu = paddle.nn.Silu()
        self.key_update = paddle.nn.Sequential(
            paddle.nn.Linear(in_features=out_channels * 3, out_features=out_channels),
            paddle.nn.Silu(),
            paddle.nn.Linear(in_features=out_channels, out_features=out_channels),
        )
        self.bn = paddle.nn.BatchNorm1D(num_features=out_channels)
        self.bn_att = paddle.nn.BatchNorm1D(num_features=out_channels)
        self.sigmoid = paddle.nn.Sigmoid()

    def forward(
        self,
        x: paddle.Tensor,
        edge_index,
        edge_attr=None,
    ):
        H, C = self.heads, self.out_channels
        if isinstance(x, paddle.Tensor):
            x = (x, x)
        query = self.lin_query(x[1]).reshape([-1, H, C])
        key = self.lin_key(x[0]).reshape([-1, H, C])
        value = self.lin_value(x[0]).reshape([-1, H, C])
        out = self.propagate(
            edge_index,
            query=query,
            key=key,
            value=value,
            edge_attr=edge_attr,
            size=None,
        )
        out = out.reshape([-1, self.heads * self.out_channels])
        out = self.lin_concate(out)
        return self.softplus(x[1] + self.bn(out))

    def message(
        self,
        query_i: paddle.Tensor,
        key_i: paddle.Tensor,
        key_j: paddle.Tensor,
        value_j: paddle.Tensor,
        value_i: paddle.Tensor,
        edge_attr: paddle.Tensor,
    ) -> paddle.Tensor:
        edge_attr = self.lin_edge(edge_attr).reshape(
            [-1, self.heads, self.out_channels]
        )
        key_j = self.key_update(paddle.concat(x=(key_i, key_j, edge_attr), axis=-1))
        alpha = query_i * key_j / math.sqrt(self.out_channels)
        out = self.lin_msg_update(
            paddle.concat(x=(value_i, value_j, edge_attr), axis=-1)
        )
        out = out * self.sigmoid(
            self.bn_att(alpha.reshape([-1, self.out_channels])).reshape(
                [-1, self.heads, self.out_channels]
            )
        )
        return out


class ComformerConv_edge(paddle.nn.Layer):
    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        heads: int = 1,
        concat: bool = True,
        beta: bool = False,
        dropout: float = 0.0,
        edge_dim: Optional[int] = None,
        root_weight: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.beta = beta and root_weight
        self.root_weight = root_weight
        self.concat = concat
        self.dropout = dropout
        self.edge_dim = edge_dim
        if isinstance(in_channels, int):
            in_channels = in_channels, in_channels
        self.lemb = paddle.nn.Embedding(num_embeddings=3, embedding_dim=32)
        self.embedding_dim = 32
        self.lin_key = paddle.nn.Linear(
            in_features=in_channels[0], out_features=heads * out_channels
        )
        self.lin_query = paddle.nn.Linear(
            in_features=in_channels[1], out_features=heads * out_channels
        )
        self.lin_value = paddle.nn.Linear(
            in_features=in_channels[0], out_features=heads * out_channels
        )
        self.lin_key_e1 = paddle.nn.Linear(
            in_features=in_channels[0], out_features=heads * out_channels
        )
        self.lin_value_e1 = paddle.nn.Linear(
            in_features=in_channels[0], out_features=heads * out_channels
        )
        self.lin_key_e2 = paddle.nn.Linear(
            in_features=in_channels[0], out_features=heads * out_channels
        )
        self.lin_value_e2 = paddle.nn.Linear(
            in_features=in_channels[0], out_features=heads * out_channels
        )
        self.lin_key_e3 = paddle.nn.Linear(
            in_features=in_channels[0], out_features=heads * out_channels
        )
        self.lin_value_e3 = paddle.nn.Linear(
            in_features=in_channels[0], out_features=heads * out_channels
        )
        self.lin_edge = paddle.nn.Linear(
            in_features=edge_dim, out_features=heads * out_channels, bias_attr=False
        )
        self.lin_edge_len = paddle.nn.Linear(
            in_features=in_channels[0] + self.embedding_dim, out_features=in_channels[0]
        )
        self.lin_concate = paddle.nn.Linear(
            in_features=heads * out_channels, out_features=out_channels
        )
        self.lin_msg_update = paddle.nn.Sequential(
            paddle.nn.Linear(in_features=out_channels * 3, out_features=out_channels),
            paddle.nn.Silu(),
            paddle.nn.Linear(in_features=out_channels, out_features=out_channels),
        )
        self.silu = paddle.nn.Silu()
        self.softplus = paddle.nn.Softplus()
        self.key_update = paddle.nn.Sequential(
            paddle.nn.Linear(in_features=out_channels * 3, out_features=out_channels),
            paddle.nn.Silu(),
            paddle.nn.Linear(in_features=out_channels, out_features=out_channels),
        )
        self.bn_att = paddle.nn.BatchNorm1D(num_features=out_channels)
        self.bn = paddle.nn.BatchNorm1D(num_features=out_channels)
        self.sigmoid = paddle.nn.Sigmoid()

    def forward(
        self,
        edge: paddle.Tensor,
        edge_nei_len: paddle.Tensor = None,
        edge_nei_angle: paddle.Tensor = None,
    ):
        H, C = self.heads, self.out_channels
        if isinstance(edge, paddle.Tensor):
            edge = (edge, edge)

        query_x = (
            self.lin_query(edge[1])
            .reshape([-1, H, C])
            .unsqueeze(axis=1)
            .tile(repeat_times=[1, 3, 1, 1])
        )
        key_x = (
            self.lin_key(edge[0])
            .reshape([-1, H, C])
            .unsqueeze(axis=1)
            .tile(repeat_times=[1, 3, 1, 1])
        )
        value_x = (
            self.lin_value(edge[0])
            .reshape([-1, H, C])
            .unsqueeze(axis=1)
            .tile(repeat_times=[1, 3, 1, 1])
        )

        key_y = paddle.concat(
            x=(
                self.lin_key_e1(edge_nei_len[:, 0, :]).reshape([-1, 1, H, C]),
                self.lin_key_e2(edge_nei_len[:, 1, :]).reshape([-1, 1, H, C]),
                self.lin_key_e3(edge_nei_len[:, 2, :]).reshape([-1, 1, H, C]),
            ),
            axis=1,
        )
        value_y = paddle.concat(
            x=(
                self.lin_value_e1(edge_nei_len[:, 0, :]).reshape([-1, 1, H, C]),
                self.lin_value_e2(edge_nei_len[:, 1, :]).reshape([-1, 1, H, C]),
                self.lin_value_e3(edge_nei_len[:, 2, :]).reshape([-1, 1, H, C]),
            ),
            axis=1,
        )
        edge_xy = self.lin_edge(edge_nei_angle).reshape([-1, 3, H, C])
        key = self.key_update(paddle.concat(x=(key_x, key_y, edge_xy), axis=-1))
        alpha = query_x * key / math.sqrt(self.out_channels)
        out = self.lin_msg_update(paddle.concat(x=(value_x, value_y, edge_xy), axis=-1))
        out = out * self.sigmoid(
            self.bn_att(alpha.reshape([-1, self.out_channels])).reshape(
                [-1, 3, self.heads, self.out_channels]
            )
        )
        out = out.reshape([-1, 3, self.heads * self.out_channels])
        out = self.lin_concate(out)
        out = out.sum(axis=1)
        return self.softplus(edge[1] + self.bn(out))


def bond_cosine(r1, r2):
    bond_cosine = paddle.sum(x=r1 * r2, axis=-1) / (
        paddle.linalg.norm(x=r1, axis=-1) * paddle.linalg.norm(x=r2, axis=-1)
    )
    bond_cosine = paddle.clip(x=bond_cosine, min=-1, max=1)
    return bond_cosine


class iComformer(RuntimeMixin, nn.Layer):
    """Complete and Efficient Graph Transformers for Crystal Material Property
    Prediction,  https://arxiv.org/pdf/2403.11857

    Args:
        conv_layers (int, optional): The number of ComformerConv layers.
            Defaults to 4.
        edge_layers (int, optional): The number of ComformerConv_edge layers.
            Defaults to 1.
        atom_input_features (int, optional): The dimension of input features.
            Defaults to 92.
        edge_features (int, optional): The dimension of edge feature. Defaults to 256.
        triplet_input_features (int, optional): The dimension of triplet feature.
            Defaults to 256.
        node_features (int, optional): The dimension of node feature. Defaults to 256.
        fc_features (int, optional): The input dimension of the fully connected layer.
            Defaults to 256.
        output_features (int, optional): The output dimension. Defaults to 1.
        node_layer_head (int, optional): Heads of the node layer. Defaults to 1.
        edge_layer_head (int, optional): Heads of the edge layer. Defaults to 1.
        property_name (Optional[str], optional):  Property name of the
            prediction data. Defaults to "formation_energy_per_atom".
        data_mean (float, optional): Mean of the training data. Defaults to 0.0.
        data_std (float, optional): Standard deviation of the training data.
            Defaults to 1.0.
        loss_type (str, optional): Loss type, can be 'mse_loss' or 'l1_loss'. Defaults
            to "mse_loss".
        execution_backend (str, optional): Numerical execution backend. Use
            ``"eager"`` or ``"cinn"``. Defaults to ``"eager"``.
        runtime_options (dict, optional): Per-backend runtime options.
    """

    def __init__(
        self,
        conv_layers: int = 4,
        edge_layers: int = 1,
        atom_input_features: int = 92,
        edge_features: int = 256,
        triplet_input_features: int = 256,
        node_features: int = 256,
        fc_features: int = 256,
        output_features: int = 1,
        node_layer_head: int = 1,
        edge_layer_head: int = 1,
        property_name: Optional[str] = "formation_energy_per_atom",
        data_mean: float = 0.0,
        data_std: float = 1.0,
        loss_type: str = "mse_loss",
        execution_backend: str = "eager",
        runtime_options: dict | None = None,
    ):
        super().__init__()
        self._init_runtime(execution_backend, runtime_options)
        self.conv_layers = conv_layers
        self.edge_layers = edge_layers
        self.atom_input_features = atom_input_features
        self.edge_features = edge_features
        self.triplet_input_features = triplet_input_features
        self.node_features = node_features
        self.fc_features = fc_features
        self.output_features = output_features
        self.node_layer_head = node_layer_head
        self.edge_layer_head = edge_layer_head
        if isinstance(property_name, list):
            self.property_name = property_name[0]
        else:
            assert isinstance(property_name, str)
            self.property_name = property_name
        self.register_buffer(tensor=paddle.to_tensor(data_mean), name="data_mean")
        self.register_buffer(tensor=paddle.to_tensor(data_std), name="data_std")

        self.atom_embedding = nn.Linear(
            in_features=self.atom_input_features, out_features=self.node_features
        )
        self.rbf = nn.Sequential(
            RBFExpansion(vmin=-4.0, vmax=0.0, bins=self.edge_features),
            nn.Linear(in_features=self.edge_features, out_features=self.node_features),
            nn.Softplus(),
        )
        self.rbf_angle = nn.Sequential(
            RBFExpansion(vmin=-1.0, vmax=1.0, bins=self.triplet_input_features),
            nn.Linear(
                in_features=self.triplet_input_features, out_features=self.node_features
            ),
            nn.Softplus(),
        )
        self.att_layers = nn.LayerList(
            sublayers=[
                ComformerConv(
                    in_channels=self.node_features,
                    out_channels=self.node_features,
                    heads=self.node_layer_head,
                    edge_dim=self.node_features,
                )
                for _ in range(self.conv_layers)
            ]
        )
        self.edge_update_layer = ComformerConv_edge(
            in_channels=self.edge_features,
            out_channels=self.node_features,
            heads=self.edge_layer_head,
            edge_dim=self.node_features,
        )
        self.fc = nn.Sequential(
            nn.Linear(in_features=self.node_features, out_features=self.fc_features),
            nn.Silu(),
        )
        self.fc_out = nn.Linear(
            in_features=self.fc_features, out_features=self.output_features
        )
        if loss_type == "mse_loss":
            self.loss_fn = paddle.nn.functional.mse_loss
        elif loss_type == "l1_loss":
            self.loss_fn = paddle.nn.functional.l1_loss
        else:
            raise ValueError(f"Unknown loss type {loss_type}.")

    def normalize(self, tensor):
        return (tensor - self.data_mean) / self.data_std

    def unnormalize(self, tensor):
        return tensor * self.data_std + self.data_mean

    @runtime_boundary("forward")
    def _forward(self, data) -> paddle.Tensor:
        #  The data in data['graph'] is numpy.ndarray, convert it to paddle.Tensor
        graph = data["graph"].tensor()

        batch_idx = graph.graph_node_id
        edges = graph.edges.T.contiguous()

        node_features = self.atom_embedding(
            graph.node_feat["node_feat"].cast("float32")
        )
        edge_feat = -0.75 / paddle.linalg.norm(x=graph.edge_feat["r"], axis=1)
        edge_nei_len = -0.75 / paddle.linalg.norm(x=graph.edge_feat["nei"], axis=-1)
        edge_nei_angle = bond_cosine(
            graph.edge_feat["nei"],
            graph.edge_feat["r"].unsqueeze(1).tile(repeat_times=[1, 3, 1]),
        )
        num_edge = tuple(edge_feat.shape)[0]
        edge_features = self.rbf(edge_feat)
        edge_nei_len = self.rbf(edge_nei_len.reshape([-1])).reshape([num_edge, 3, -1])
        edge_nei_angle = self.rbf_angle(edge_nei_angle.reshape([-1])).reshape(
            [num_edge, 3, -1]
        )
        node_features = self.att_layers[0](node_features, edges, edge_features)

        edge_features = self.edge_update_layer(
            edge_features, edge_nei_len, edge_nei_angle
        )
        for i in range(1, len(self.att_layers)):
            node_features = self.att_layers[i](node_features, edges, edge_features)

        # Without dim_size, scatter() reads index.max() back from the device,
        # which breaks the SOT graph.
        features = scatter(
            node_features, batch_idx, dim=0, dim_size=graph_count(graph), reduce="mean"
        )
        features = self.fc(features)
        result = self.fc_out(features)
        return result

    def forward(self, data, return_loss=True, return_prediction=True):
        assert (
            return_loss or return_prediction
        ), "At least one of return_loss or return_prediction must be True."
        pred = self._forward(data)

        loss_dict = {}
        if return_loss:
            label = data[self.property_name]
            label = self.normalize(label)
            loss = self.loss_fn(
                input=pred,
                label=label,
            )
            loss_dict["loss"] = loss

        prediction = {}
        if return_prediction:
            pred = self.unnormalize(pred)
            prediction[self.property_name] = pred
        return {"loss_dict": loss_dict, "pred_dict": prediction}

    @paddle.no_grad()
    def predict(self, samples):
        is_list = isinstance(samples, list)
        samples = samples if is_list else [samples]

        results = []
        for sample in samples:
            result = self._forward({"graph": sample})
            result = self.unnormalize(result).numpy()[0, 0]
            results.append({self.property_name: result})
        return results if is_list else results[0]

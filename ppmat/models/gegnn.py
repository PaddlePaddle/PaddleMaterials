# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

from ppmat.utils.scatter import scatter_mean
from ppmat.utils.scatter import scatter_sum


def _segment_sum(values, segment_ids, num_segments):
    return scatter_sum(values, segment_ids, dim=0, dim_size=num_segments)


def _segment_mean(values, segment_ids, num_segments):
    return scatter_mean(values, segment_ids, dim=0, dim_size=num_segments)


_ACTIVATION_MAP = {
    "relu": (paddle.nn.ReLU, paddle.nn.functional.relu),
    "elu": (paddle.nn.ELU, paddle.nn.functional.elu),
    "leakyrelu": (paddle.nn.LeakyReLU, paddle.nn.functional.leaky_relu),
    "softplus": (paddle.nn.Softplus, paddle.nn.functional.softplus),
}


def get_activation(activation, get_nn=False):
    key = "relu" if activation is None else activation.lower()
    if key not in _ACTIVATION_MAP:
        raise ValueError(f"Unsupported activation: {activation}")
    nn_cls, func = _ACTIVATION_MAP[key]
    return nn_cls if get_nn else func


class HigherOrderGraphConv(paddle.nn.Layer):
    """GCN convolution equivalent to PGL ``GCNConv(norm=True)``."""

    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear = paddle.nn.Linear(input_size, output_size, bias_attr=False)
        self.bias = self.create_parameter(
            shape=[output_size],
            is_bias=True,
            default_initializer=paddle.nn.initializer.Constant(0.0),
        )

    def forward(self, graph, feature):
        edges = graph.edges
        if not isinstance(edges, paddle.Tensor):
            edges = paddle.to_tensor(edges, dtype="int64")
        else:
            edges = paddle.cast(edges, "int64")
        src = edges[:, 0]
        dst = edges[:, 1]
        num_nodes = int(graph.num_nodes)

        destination_assignment = paddle.nn.functional.one_hot(
            dst, num_classes=num_nodes
        )
        destination_assignment = paddle.cast(destination_assignment, feature.dtype)
        degree = paddle.sum(destination_assignment, axis=0).reshape([-1, 1])
        norm = paddle.rsqrt(paddle.clip(degree, min=1.0))

        transformed = self.linear(feature) * norm
        source_assignment = paddle.nn.functional.one_hot(src, num_classes=num_nodes)
        source_assignment = paddle.cast(source_assignment, feature.dtype)
        edge_messages = paddle.matmul(source_assignment, transformed)
        aggregated = paddle.matmul(
            destination_assignment, edge_messages, transpose_x=True
        )
        return aggregated * norm + self.bias


class NNConv(paddle.nn.Layer):
    def __init__(self, in_size, out_size, edge_network, aggregator="sum"):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.edge_network = edge_network
        self.aggregator = aggregator
        # DGL ``NNConv`` enables a learnable, zero-initialized output bias by
        # default. Keep it explicit so parameters and numerics match upstream.
        self.bias = self.create_parameter(
            shape=[out_size],
            is_bias=True,
            default_initializer=paddle.nn.initializer.Constant(0.0),
        )

    def forward(self, graph, node_feats, edge_feats):
        edges = graph.edges
        src = edges[:, 0]
        dst = edges[:, 1]

        edge_weight = self.edge_network(edge_feats)
        edge_weight = paddle.reshape(edge_weight, [-1, self.in_size, self.out_size])

        source_assignment = paddle.nn.functional.one_hot(
            src, num_classes=int(graph.num_nodes)
        )
        source_assignment = paddle.cast(source_assignment, node_feats.dtype)
        h_src = paddle.matmul(source_assignment, node_feats).unsqueeze(1)
        msg = paddle.matmul(h_src, edge_weight).squeeze(1)

        if self.aggregator == "sum":
            aggregated = _segment_sum(msg, dst, graph.num_nodes)
        elif self.aggregator == "mean":
            aggregated = _segment_mean(msg, dst, graph.num_nodes)
        else:
            raise ValueError(f"Unsupported aggregator: {self.aggregator}")
        return aggregated + self.bias


class HigherOrderGRUCell(paddle.nn.Layer):
    """One GRU step expressed with primitive ops for higher-order autograd."""

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.input_linear = paddle.nn.Linear(input_size, 3 * hidden_size)
        self.hidden_linear = paddle.nn.Linear(hidden_size, 3 * hidden_size)

    def forward(self, inputs, hidden):
        input_gates = self.input_linear(inputs)
        hidden_gates = self.hidden_linear(hidden)
        input_reset, input_update, input_new = paddle.split(
            input_gates, num_or_sections=3, axis=-1
        )
        hidden_reset, hidden_update, hidden_new = paddle.split(
            hidden_gates, num_or_sections=3, axis=-1
        )
        reset_gate = paddle.nn.functional.sigmoid(input_reset + hidden_reset)
        update_gate = paddle.nn.functional.sigmoid(input_update + hidden_update)
        new_gate = paddle.tanh(input_new + reset_gate * hidden_new)
        return new_gate + update_gate * (hidden - new_gate)


class MPNNconv(paddle.nn.Layer):
    def __init__(
        self,
        node_in_feats,
        edge_in_feats,
        node_out_feats=128,
        edge_hidden_feats=32,
        num_step_message_passing=6,
        activation="relu",
    ):
        super().__init__()
        self.mpnn_activation = get_activation(activation)
        self.project_node_feats = paddle.nn.Sequential(
            paddle.nn.Linear(node_in_feats, node_out_feats),
            get_activation(activation, get_nn=True)(),
        )
        self.num_step_message_passing = num_step_message_passing
        edge_network = paddle.nn.Sequential(
            paddle.nn.Linear(edge_in_feats, edge_hidden_feats),
            get_activation(activation, get_nn=True)(),
            paddle.nn.Linear(edge_hidden_feats, node_out_feats * node_out_feats),
        )
        self.gnn_layer = NNConv(
            in_size=node_out_feats,
            out_size=node_out_feats,
            edge_network=edge_network,
            aggregator="sum",
        )
        self.gru = HigherOrderGRUCell(node_out_feats, node_out_feats)

    def forward(self, g, node_feats, edge_feats):
        node_feats = self.project_node_feats(node_feats)
        hidden_feats = node_feats
        for _ in range(self.num_step_message_passing):
            messages = self.mpnn_activation(self.gnn_layer(g, node_feats, edge_feats))
            hidden_feats = self.gru(messages, hidden_feats)
            node_feats = hidden_feats
        return node_feats


class GEGNNBinary(paddle.nn.Layer):
    """GE-GNN for binary activity coefficients."""

    @staticmethod
    def _as_column(value):
        if not isinstance(value, paddle.Tensor):
            value = paddle.to_tensor(value, dtype="float32")
        return value.cast("float32").reshape([-1, 1])

    @staticmethod
    def activity_coefficients(excess_gibbs_energy, x1, derivative):
        """Convert ``G^E`` and its composition derivative to ``ln(gamma)``."""
        gamma1 = excess_gibbs_energy + (1.0 - x1) * derivative
        gamma2 = excess_gibbs_energy - x1 * derivative
        return paddle.concat([gamma1, gamma2], axis=1)

    def __init__(
        self,
        in_dim=74,
        hidden_dim=256,
        n_classes=1,
        mlp_dropout_rate=0.0,
        mlp_activation="relu",
        mpnn_activation="relu",
        mlp_num_hid_layers=2,
        property_name="gamma",
        edge_in_feats=1,
    ):
        super().__init__()
        if n_classes != 1:
            raise ValueError("GEGNNBinary predicts one scalar excess Gibbs energy.")

        if mlp_num_hid_layers != 2:
            raise ValueError(
                "The reference GE-GNN architecture uses exactly two hidden MLP layers."
            )

        if edge_in_feats not in (1, 3):
            raise ValueError("edge_in_feats must be 1 (reference) or 3 (legacy).")
        if mlp_dropout_rate != 0.0:
            raise ValueError("The reference GE-GNN does not apply MLP dropout.")
        self.edge_in_feats = edge_in_feats
        self.conv1 = HigherOrderGraphConv(in_dim, hidden_dim)
        self.conv2 = HigherOrderGraphConv(hidden_dim, hidden_dim)
        self.global_conv1 = MPNNconv(
            node_in_feats=hidden_dim,
            edge_in_feats=edge_in_feats,
            node_out_feats=hidden_dim,
            edge_hidden_feats=32,
            num_step_message_passing=1,
            activation=mpnn_activation,
        )
        self.mlp_activation = get_activation(mlp_activation)
        self.mfp_trans = paddle.nn.Linear(hidden_dim + 1, hidden_dim + 1)
        self.classify1 = paddle.nn.Linear(hidden_dim + 1, hidden_dim)
        self.classify2 = paddle.nn.Linear(hidden_dim, hidden_dim)
        self.classify3 = paddle.nn.Linear(hidden_dim, 1)
        self.mlp_dropout_rate = mlp_dropout_rate
        if isinstance(property_name, (list, tuple)):
            property_name = property_name[0]
        self.property_name = property_name

    def set_state_dict(self, state_dict, *args, **kwargs):
        key = "global_conv1.gnn_layer.edge_network.0.weight"
        if key in state_dict and state_dict[key].shape[0] != self.edge_in_feats:
            raise ValueError(
                f"Checkpoint edge feature dimension is {state_dict[key].shape[0]}, "
                f"but the model expects {self.edge_in_feats}. Set edge_in_feats="
                f"{state_dict[key].shape[0]} for these weights. Legacy checkpoints "
                "require three features; reference-model weights require one."
            )
        return super().set_state_dict(state_dict, *args, **kwargs)

    def _component_features(self, data, batch_size):
        g1 = data["g1"].tensor()
        g2 = data["g2"].tensor()
        h1 = g1.node_feat["h"]
        h2 = g2.node_feat["h"]
        h1 = paddle.nn.functional.relu(self.conv1(g1, h1))
        h1 = paddle.nn.functional.relu(self.conv2(g1, h1))
        h2 = paddle.nn.functional.relu(self.conv1(g2, h2))
        h2 = paddle.nn.functional.relu(self.conv2(g2, h2))
        hg1 = _segment_mean(h1, g1.graph_node_id, batch_size)
        hg2 = _segment_mean(h2, g2.graph_node_id, batch_size)

        inter_hb = self._as_column(data["inter_hb"])
        intra_hb1 = self._as_column(data["intra_hb1"])
        intra_hb2 = self._as_column(data["intra_hb2"])
        if self.edge_in_feats == 1:
            edge_features = paddle.stack(
                [inter_hb, inter_hb, intra_hb1, intra_hb2], axis=1
            ).reshape([-1, 1])
        else:
            edge_features = paddle.stack(
                [
                    paddle.concat([inter_hb, intra_hb1, intra_hb2], axis=1),
                    paddle.concat([inter_hb, intra_hb2, intra_hb1], axis=1),
                    paddle.concat([inter_hb, intra_hb1, intra_hb1], axis=1),
                    paddle.concat([inter_hb, intra_hb2, intra_hb2], axis=1),
                ],
                axis=1,
            ).reshape([-1, 3])
        interaction_graph = data["empty_solvsys"].tensor()
        node_features = paddle.stack([hg1, hg2], axis=1).reshape([2 * batch_size, -1])
        output = self.global_conv1(interaction_graph, node_features, edge_features)
        output = output.reshape([batch_size, 2, -1])
        return paddle.concat([output[:, 0], output[:, 1]], axis=0)

    def _excess_gibbs_energy(self, data, x1, component_features=None):
        batch_size = x1.shape[0]
        if component_features is None:
            component_features = self._component_features(data, batch_size)
        component1 = paddle.concat([component_features[:batch_size], x1], axis=1)
        component2 = paddle.concat([component_features[batch_size:], 1.0 - x1], axis=1)
        component1 = self.mlp_activation(self.mfp_trans(component1))
        component2 = self.mlp_activation(self.mfp_trans(component2))
        pooled_features = 0.5 * (component1 + component2)
        output = self.mlp_activation(self.classify1(pooled_features))
        output = self.mlp_activation(self.classify2(output))
        return self.classify3(output)

    def _predict_head_gradient(self, data):
        return self._predict_with_derivative(data, create_graph=False)

    def _predict_with_derivative(self, data, create_graph):
        x1 = self._as_column(data["x1"])
        # The collator tensors normally stop gradients. This input must remain
        # differentiable because ``ln(gamma)`` is defined by dG^E/dx1.
        x1.stop_gradient = False
        component_features = None
        if not create_graph:
            with paddle.no_grad():
                component_features = self._component_features(data, x1.shape[0])
        excess_gibbs_energy = self._excess_gibbs_energy(data, x1, component_features)
        derivative = paddle.grad(
            outputs=excess_gibbs_energy,
            inputs=x1,
            create_graph=create_graph,
            retain_graph=create_graph,
        )[0]
        prediction = self.activity_coefficients(excess_gibbs_energy, x1, derivative)
        return excess_gibbs_energy, derivative, prediction

    def _forward(self, data, create_graph=False):
        if create_graph:
            return self._predict_with_derivative(data, create_graph=True)
        return self._predict_head_gradient(data)

    def forward(self, data, return_loss=True, return_prediction=True):
        if self.training and return_loss:
            excess_gibbs_energy, derivative, output = self._forward(
                data, create_graph=True
            )
        else:
            excess_gibbs_energy, derivative, output = self._forward(data)
        loss_dict = {}
        if return_loss:
            labels = paddle.concat(
                [self._as_column(data["gamma1"]), self._as_column(data["gamma2"])],
                axis=1,
            )
            supervised_loss = paddle.nn.functional.mse_loss(output, labels)
            loss_dict = {"supervised_loss": supervised_loss, "loss": supervised_loss}

        prediction = {}
        if return_prediction:
            prediction[self.property_name] = output
            prediction["excess_gibbs_energy"] = excess_gibbs_energy
            prediction["d_excess_gibbs_energy_dx1"] = derivative
        return {"loss_dict": loss_dict, "pred_dict": prediction}

    def predict(self, data):
        _, _, output = self._forward(data)
        return {self.property_name: output}


class _TensorGraphView:
    """Graph attributes constructed from traced tensors inside the adapter."""

    def __init__(self, edges, node_features, graph_node_id):
        self.edges = edges
        self.num_nodes = int(node_features.shape[0])
        self.node_feat = {"h": node_features}
        self.graph_node_id = graph_node_id

    def tensor(self):
        return self


class GEGNNTensorPredictor(paddle.nn.Layer):
    """Expose GE-GNN prediction to ``to_static`` through tensor-only inputs.

    Call :meth:`pack` outside the compiled function. PGL graph objects must not
    cross the static input boundary. Node/edge counts may specialize the graph
    and trigger compilation again. This adapter preserves the composition
    derivative; do not wrap prediction in ``paddle.no_grad()``.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    @staticmethod
    def pack(data):
        """Pack an existing collated binary-mixture batch without rebuilding it."""
        flat = {
            key: GEGNNBinary._as_column(data[key])
            for key in ("x1", "inter_hb", "intra_hb1", "intra_hb2")
        }
        for key in ("g1", "g2", "empty_solvsys"):
            graph = data[key].tensor()
            flat[key + "_edges"] = graph.edges
            flat[key + "_graph_node_id"] = graph.graph_node_id
            if key != "empty_solvsys":
                flat[key + "_node_features"] = graph.node_feat["h"]
            else:
                flat[key + "_node_features"] = paddle.zeros(
                    [int(graph.num_nodes), 1], dtype="float32"
                )
        return flat

    def forward(self, flat):
        data = {key: flat[key] for key in ("x1", "inter_hb", "intra_hb1", "intra_hb2")}
        for key in ("g1", "g2", "empty_solvsys"):
            data[key] = _TensorGraphView(
                flat[key + "_edges"],
                flat[key + "_node_features"],
                flat[key + "_graph_node_id"],
            )
        return self.model.predict(data)[self.model.property_name]

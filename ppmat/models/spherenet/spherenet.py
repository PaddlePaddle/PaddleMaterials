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

from collections.abc import Callable
from collections.abc import Mapping
from collections.abc import Sequence
from typing import Any

import paddle
from paddle import nn
from paddle.nn import Embedding
from paddle.nn import Linear

from ppmat.models.common import initializer
from ppmat.models.common.activation import swish
from ppmat.models.common.runtime import RuntimeMixin
from ppmat.models.common.runtime import runtime_boundary
from ppmat.models.common.spherical_fourier_bessel import DistEmbedding
from ppmat.models.common.spherical_fourier_bessel import SphericalFourierBesselEmbedding
from ppmat.models.spherenet.geometry import compute_geometry
from ppmat.models.spherenet.geometry import select_torsion_edges
from ppmat.utils.pgl_compat import graph_count
from ppmat.utils.scatter import scatter_sum
from ppmat.utils.scatter import scatter_sum_first_order


def _aggregate(src, index, dim_size, require_second_order):
    if require_second_order:
        # scatter_nd_add lacks the second derivative required by force loss.
        return scatter_sum(src, index, dim=0, dim_size=dim_size)

    return scatter_sum_first_order(src, index, dim_size)


class SphereNetEmbedding(paddle.nn.Layer):
    def __init__(self, num_spherical, num_radial, cutoff, envelope_exponent):
        super().__init__()
        self.dist_emb = DistEmbedding(num_radial, cutoff, envelope_exponent)
        self.geometry_emb = SphericalFourierBesselEmbedding(
            num_spherical, num_radial, cutoff
        )

    def reset_parameters(self):
        self.dist_emb.reset_parameters()

    def forward(self, dist, angle, torsion, idx_kj):
        dist_emb = self.dist_emb(dist)
        angle_emb, torsion_emb = self.geometry_emb(dist, *angle, *torsion, idx_kj)
        return dist_emb, angle_emb, torsion_emb


class ResidualLayer(paddle.nn.Layer):
    def __init__(self, hidden_channels, act=swish):
        super().__init__()
        self.act = act
        self.lin1 = Linear(hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, hidden_channels)

    def reset_parameters(self):
        initializer.glorot_orthogonal_(self.lin1.weight, scale=2.0)
        initializer.zeros_(self.lin1.bias)
        initializer.glorot_orthogonal_(self.lin2.weight, scale=2.0)
        initializer.zeros_(self.lin2.bias)

    def forward(self, x):
        return x + self.act(self.lin2(self.act(self.lin1(x))))


class InitialEdgeEmbedding(paddle.nn.Layer):
    def __init__(
        self,
        num_radial,
        hidden_channels,
        act=swish,
        use_atom_types=True,
        use_extra_node_feature=False,
        require_second_order=False,
    ):
        super().__init__()
        self.act = act
        self.use_atom_types = use_atom_types
        self.use_extra_node_feature = use_extra_node_feature
        self.require_second_order = require_second_order
        if self.use_atom_types:
            self.emb = Embedding(95, hidden_channels)
        else:
            self.node_embedding = paddle.create_parameter(
                shape=[hidden_channels],
                dtype=paddle.get_default_dtype(),
                default_initializer=paddle.nn.initializer.Normal(),
            )
        self.lin_rbf_0 = Linear(num_radial, hidden_channels)
        if self.use_extra_node_feature:
            self.lin = Linear(5 * hidden_channels, hidden_channels)
        else:
            self.lin = Linear(3 * hidden_channels, hidden_channels)
        self.lin_rbf_1 = Linear(num_radial, hidden_channels, bias_attr=False)

    def reset_parameters(self):
        if self.use_atom_types:
            initializer.uniform_(
                self.emb.weight,
                -(3.0**0.5),
                3.0**0.5,
            )
        else:
            initializer.normal_(self.node_embedding)

        initializer.linear_init_(self.lin_rbf_0)
        initializer.linear_init_(self.lin)
        initializer.glorot_orthogonal_(self.lin_rbf_1.weight, scale=2.0)

    def forward(self, x, node_feature, emb_in, i, j):
        rbf, _, _ = emb_in
        if self.use_atom_types:
            if self.require_second_order and self.training:
                # EmbeddingGradNode has no higher-order backward in Paddle 3.1.
                # This equivalent lookup keeps force-loss gradients trainable.
                x = paddle.nn.functional.one_hot(x, self.emb.weight.shape[0]).astype(
                    self.emb.weight.dtype
                )
                x = paddle.matmul(x, self.emb.weight)
            else:
                x = self.emb(x)
        else:
            x = self.node_embedding.unsqueeze(0).expand([x.shape[0], -1])
        if node_feature is not None and self.use_extra_node_feature:
            x = paddle.concat([x, node_feature], axis=1)
        rbf0 = self.act(self.lin_rbf_0(rbf))
        e1 = self.act(self.lin(paddle.concat([x[i], x[j], rbf0], axis=-1)))
        e2 = self.lin_rbf_1(rbf) * e1
        return e1, e2


class EdgeUpdate(paddle.nn.Layer):
    def __init__(
        self,
        hidden_channels,
        int_emb_size,
        basis_emb_size_dist,
        basis_emb_size_angle,
        basis_emb_size_torsion,
        num_spherical,
        num_radial,
        num_before_skip,
        num_after_skip,
        act=swish,
        require_second_order=False,
    ):
        super().__init__()
        self.act = act
        self.require_second_order = require_second_order
        self.lin_rbf1 = Linear(num_radial, basis_emb_size_dist, bias_attr=False)
        self.lin_rbf2 = Linear(basis_emb_size_dist, hidden_channels, bias_attr=False)
        self.lin_sbf1 = Linear(
            num_spherical * num_radial, basis_emb_size_angle, bias_attr=False
        )
        self.lin_sbf2 = Linear(basis_emb_size_angle, int_emb_size, bias_attr=False)
        self.lin_t1 = Linear(
            num_spherical * num_spherical * num_radial,
            basis_emb_size_torsion,
            bias_attr=False,
        )
        self.lin_t2 = Linear(basis_emb_size_torsion, int_emb_size, bias_attr=False)
        self.lin_rbf = Linear(num_radial, hidden_channels, bias_attr=False)

        self.lin_kj = Linear(hidden_channels, hidden_channels)
        self.lin_ji = Linear(hidden_channels, hidden_channels)

        self.lin_down = Linear(hidden_channels, int_emb_size, bias_attr=False)
        self.lin_up = Linear(int_emb_size, hidden_channels, bias_attr=False)

        self.layers_before_skip = nn.LayerList(
            [ResidualLayer(hidden_channels, act) for _ in range(num_before_skip)]
        )
        self.lin = Linear(hidden_channels, hidden_channels)
        self.layers_after_skip = nn.LayerList(
            [ResidualLayer(hidden_channels, act) for _ in range(num_after_skip)]
        )

    def reset_parameters(self):
        for layer in self.sublayers():
            if isinstance(layer, Linear):
                initializer.glorot_orthogonal_(layer.weight, scale=2.0)
                if layer.bias is not None:
                    initializer.zeros_(layer.bias)

    def forward(self, x, emb_in, idx_kj, idx_ji):
        rbf0, sbf, t = emb_in
        x1, _ = x

        x_ji = self.act(self.lin_ji(x1))
        x_kj = self.act(self.lin_kj(x1))

        rbf = self.lin_rbf1(rbf0)
        rbf = self.lin_rbf2(rbf)
        x_kj = x_kj * rbf

        x_kj = self.act(self.lin_down(x_kj))
        num_triplets = idx_kj.shape[0]
        # A dynamic triplet axis (compiled inference only) is a tensor here and
        # always takes the general path, which also yields a zero message for
        # zero triplets; branching on it would trace a cond that CINN rejects.
        if isinstance(num_triplets, int) and num_triplets == 0:
            # Empty Linear/Matmul inputs have no second-order gradient in
            # Paddle. With no triplets, the aggregated geometric message is 0.
            x_kj = x_kj * 0.0
        else:
            sbf = self.lin_sbf1(sbf)
            sbf = self.lin_sbf2(sbf)
            x_kj = x_kj[idx_kj] * sbf

            t = self.lin_t1(t)
            t = self.lin_t2(t)
            x_kj = x_kj * t

            x_kj = _aggregate(
                x_kj,
                idx_ji,
                x1.shape[0],
                self.require_second_order and self.training,
            )
        x_kj = self.act(self.lin_up(x_kj))

        e1 = x_ji + x_kj
        for layer in self.layers_before_skip:
            e1 = layer(e1)
        e1 = self.act(self.lin(e1)) + x1
        for layer in self.layers_after_skip:
            e1 = layer(e1)
        e2 = self.lin_rbf(rbf0) * e1
        return e1, e2


class NodeUpdate(paddle.nn.Layer):
    def __init__(
        self,
        hidden_channels,
        out_emb_channels,
        out_channels,
        num_output_layers,
        act,
        output_init,
        require_second_order=False,
    ):
        super().__init__()
        self.act = act
        self.output_init = output_init
        self.require_second_order = require_second_order

        self.lin_up = Linear(hidden_channels, out_emb_channels, bias_attr=True)
        self.lins = nn.LayerList()
        for _ in range(num_output_layers):
            self.lins.append(Linear(out_emb_channels, out_emb_channels))
        self.lin = Linear(out_emb_channels, out_channels, bias_attr=False)

    def reset_parameters(self):
        initializer.glorot_orthogonal_(self.lin_up.weight, scale=2.0)
        for lin in self.lins:
            initializer.glorot_orthogonal_(lin.weight, scale=2.0)
            initializer.zeros_(lin.bias)
        if self.output_init == "zeros":
            initializer.zeros_(self.lin.weight)
        elif self.output_init == "glorot_orthogonal":
            initializer.glorot_orthogonal_(self.lin.weight, scale=2.0)
        else:
            raise ValueError(f"Unsupported output initializer: {self.output_init}")

    def forward(self, e, i, dim_size=None):
        _, e2 = e
        v = _aggregate(
            e2,
            i,
            dim_size,
            self.require_second_order and self.training,
        )
        v = self.lin_up(v)
        for lin in self.lins:
            v = self.act(lin(v))
        v = self.lin(v)
        return v


class SphereNet(RuntimeMixin, paddle.nn.Layer):
    """Spherical Message Passing for 3D molecular graph tasks.

    Args:
        cutoff: Neighbor cutoff used by the radial and spherical bases. It must
            match the cutoff used to construct the input graph.
        num_layers: Number of interaction blocks.
        hidden_channels: Hidden feature dimension.
        out_channels: Number of scalar prediction channels.
        int_emb_size: Interaction embedding dimension.
        basis_emb_size_dist: Distance-basis projection dimension.
        basis_emb_size_angle: Angle-basis projection dimension.
        basis_emb_size_torsion: Torsion-basis projection dimension.
        out_emb_channels: Output-block embedding dimension.
        num_spherical: Number of spherical basis functions.
        num_radial: Number of radial basis functions.
        envelope_exponent: Polynomial envelope exponent.
        num_before_skip: Residual layers before each skip connection.
        num_after_skip: Residual layers after each skip connection.
        num_output_layers: Hidden layers in each output block.
        act: Activation name or tensor callable.
        output_init: Output projection initializer. Supported values are
            ``"glorot_orthogonal"`` and ``"zeros"``.
        use_atom_types: Whether to embed atomic numbers. If false, all atoms use
            one shared learned embedding.
        extra_node_feature_dim: Optional extra node-feature width. ``None``
            disables extra node features.
        property_name: Dataset and prediction key for the scalar target.
        derivative_properties: Optional derivatives to predict. Only ``force``
            is currently supported.
        force_key: Dataset and prediction key for forces.
        data_mean: Mean used to normalize the scalar target.
        data_std: Positive standard deviation used to normalize the scalar target.
        loss_type: Scalar and derivative loss, ``"l1_loss"`` or ``"mse_loss"``.
        loss_weights_dict: Weights for every active prediction property.
        execution_backend: Runtime backend name.
        runtime_options: Per-backend runtime options.
    """

    def __init__(
        self,
        cutoff: float = 5.0,
        num_layers: int = 4,
        hidden_channels: int = 128,
        out_channels: int = 1,
        int_emb_size: int = 64,
        basis_emb_size_dist: int = 8,
        basis_emb_size_angle: int = 8,
        basis_emb_size_torsion: int = 8,
        out_emb_channels: int = 256,
        num_spherical: int = 7,
        num_radial: int = 6,
        envelope_exponent: int = 5,
        num_before_skip: int = 1,
        num_after_skip: int = 2,
        num_output_layers: int = 3,
        act: str | Callable[[paddle.Tensor], paddle.Tensor] = "swish",
        output_init: str = "glorot_orthogonal",
        use_atom_types: bool = True,
        extra_node_feature_dim: int | None = None,
        property_name: str = "energy",
        derivative_properties: Sequence[str] | str | None = None,
        force_key: str = "force",
        data_mean: float | Sequence[float] | paddle.Tensor = 0.0,
        data_std: float | Sequence[float] | paddle.Tensor = 1.0,
        loss_type: str = "l1_loss",
        loss_weights_dict: Mapping[str, float] | None = None,
        execution_backend: str = "eager",
        runtime_options: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> None:
        super().__init__()

        act_fn = swish if act in ("swish", "silu") else act
        derivative_properties = (
            (derivative_properties,)
            if isinstance(derivative_properties, str)
            else derivative_properties or ()
        )
        derivative_properties = frozenset(derivative_properties)
        compute_forces = "force" in derivative_properties

        self._init_runtime(execution_backend, runtime_options)

        active_properties = {property_name}
        if compute_forces:
            active_properties.add(force_key)
        self.loss_weights_dict = {
            **dict.fromkeys(active_properties, 1.0),
            **(loss_weights_dict or {}),
        }
        self.loss_fn = {
            "l1_loss": paddle.nn.functional.l1_loss,
            "mse_loss": paddle.nn.functional.mse_loss,
        }[loss_type]

        self.derivative_properties = derivative_properties
        self.compute_forces = compute_forces
        self.use_extra_node_feature = extra_node_feature_dim is not None
        self.property_name = property_name
        self.force_key = force_key
        self.loss_type = loss_type
        self.register_buffer(
            name="data_mean",
            tensor=paddle.to_tensor(data_mean, dtype=paddle.get_default_dtype()),
        )
        self.register_buffer(
            name="data_std",
            tensor=paddle.to_tensor(data_std, dtype=paddle.get_default_dtype()),
        )

        if self.use_extra_node_feature:
            self.extra_emb = Linear(extra_node_feature_dim, hidden_channels)

        self.init_e = InitialEdgeEmbedding(
            num_radial,
            hidden_channels,
            act_fn,
            use_atom_types=use_atom_types,
            use_extra_node_feature=self.use_extra_node_feature,
            require_second_order=compute_forces,
        )
        node_update_cfg = {
            "hidden_channels": hidden_channels,
            "out_emb_channels": out_emb_channels,
            "out_channels": out_channels,
            "num_output_layers": num_output_layers,
            "act": act_fn,
            "output_init": output_init,
            "require_second_order": compute_forces,
        }
        self.init_v = NodeUpdate(**node_update_cfg)
        self.emb_layer = SphereNetEmbedding(
            num_spherical, num_radial, cutoff, envelope_exponent
        )

        self.update_vs = nn.LayerList(
            [NodeUpdate(**node_update_cfg) for _ in range(num_layers)]
        )

        self.update_es = nn.LayerList(
            [
                EdgeUpdate(
                    hidden_channels,
                    int_emb_size,
                    basis_emb_size_dist,
                    basis_emb_size_angle,
                    basis_emb_size_torsion,
                    num_spherical,
                    num_radial,
                    num_before_skip,
                    num_after_skip,
                    act_fn,
                    compute_forces,
                )
                for _ in range(num_layers)
            ]
        )

        self.reset_parameters()

    def reset_parameters(self):
        if self.use_extra_node_feature:
            initializer.linear_init_(self.extra_emb)
        layers = [
            self.init_e,
            self.init_v,
            self.emb_layer,
            *self.update_es,
            *self.update_vs,
        ]
        for layer in layers:
            layer.reset_parameters()

    def _forward(self, data):
        graph = data["graph"].tensor()
        z = graph.node_feat["atom_types"].astype("int64").reshape([-1])
        pos = graph.node_feat["cart_coords"].astype(paddle.get_default_dtype())
        if self.compute_forces:
            # Make coordinates a differentiable leaf before entering PIR. Creating
            # the leaf inside the static graph emits share_data_, whose Paddle 3.3
            # backward graph can miss repeated-input gradient accumulation.
            pos = pos.detach()
            pos.stop_gradient = False
        node_batch = graph.graph_node_id.astype("int64")
        edge_index = paddle.transpose(graph.edges.astype("int64"), [1, 0])
        node_feature = graph.node_feat.get("node_feature")
        if node_feature is not None:
            node_feature = node_feature.astype(paddle.get_default_dtype())
        idx_kj = graph.edge_feat["ti_idx_kj"].astype("int64")
        idx_ji = graph.edge_feat["ti_idx_ji"].astype("int64")
        idx_qj = select_torsion_edges(
            pos,
            edge_index,
            {"idx_kj": idx_kj, "idx_ji": idx_ji},
        )

        return self._runtime_forward(
            z,
            pos,
            node_batch,
            edge_index,
            node_feature,
            idx_kj,
            idx_ji,
            idx_qj,
            graph_count(graph),
        )

    @runtime_boundary(
        "forward",
        # Edge and triplet counts change at every MD step; the atom count stays
        # static. Applies to the AST capture of the potentials only.
        dynamic_dims={"edge_index": 1, "idx_kj": 0, "idx_ji": 0, "idx_qj": 0},
    )
    def _runtime_forward(
        self,
        z,
        pos,
        node_batch,
        edge_index,
        node_feature,
        idx_kj,
        idx_ji,
        idx_qj,
        num_graphs=None,
    ):
        if self.use_extra_node_feature and node_feature is not None:
            extra_node_feature = self.extra_emb(node_feature)
        else:
            extra_node_feature = None

        num_nodes = z.shape[0]
        dist, angle, torsion, i, j, idx_kj, idx_ji = compute_geometry(
            pos,
            edge_index,
            {"idx_kj": idx_kj, "idx_ji": idx_ji},
            idx_qj,
        )

        emb_out = self.emb_layer(dist, angle, torsion, idx_kj)

        e = self.init_e(z, extra_node_feature, emb_out, i, j)
        v = self.init_v(e, i, dim_size=num_nodes)
        require_second_order = self.compute_forces and self.training
        # Without num_graphs, the graph-level sums read index.max() back from the
        # device, which breaks the SOT graph.
        u = _aggregate(v, node_batch, num_graphs, require_second_order)

        for update_e, update_v in zip(self.update_es, self.update_vs):
            e = update_e(e, emb_out, idx_kj, idx_ji)
            v = update_v(e, i, dim_size=num_nodes)
            u = u + _aggregate(v, node_batch, num_graphs, require_second_order)

        forces = None
        if self.compute_forces:
            forces = -paddle.grad(
                self.unnormalize(u).sum(),
                pos,
                create_graph=self.training,
            )[0]

        return u, pos, forces

    def normalize(self, tensor):
        return (tensor - self.data_mean) / self.data_std

    def unnormalize(self, tensor):
        return tensor * self.data_std + self.data_mean

    def forward(self, data, return_loss=True, return_prediction=True):
        """Forward with the PaddleMaterials dict interface."""
        assert (
            return_loss or return_prediction
        ), "At least one of return_loss or return_prediction must be True."

        normalized_pred, _, forces_pred = self._forward(data)
        pred = self.unnormalize(normalized_pred)

        loss_dict = {}
        if return_loss:
            label = data[self.property_name]
            label_tensor = (
                label.astype(paddle.get_default_dtype())
                if isinstance(label, paddle.Tensor)
                else paddle.to_tensor(label, dtype=paddle.get_default_dtype())
            )
            normalized_label = self.normalize(label_tensor)
            property_loss = self.loss_fn(normalized_pred, normalized_label)
            loss_dict[self.property_name] = property_loss
            loss = self.loss_weights_dict[self.property_name] * property_loss

            if self.compute_forces and forces_pred is not None:
                force = data[self.force_key]
                force_tensor = (
                    force.astype(paddle.get_default_dtype())
                    if isinstance(force, paddle.Tensor)
                    else paddle.to_tensor(force, dtype=paddle.get_default_dtype())
                )
                force_loss = self.loss_fn(forces_pred, force_tensor)
                loss_dict[self.force_key] = force_loss
                loss = loss + self.loss_weights_dict[self.force_key] * force_loss
            loss_dict["loss"] = loss

        prediction = {}
        if return_prediction:
            prediction[self.property_name] = pred
            if self.compute_forces:
                prediction[self.force_key] = forces_pred.detach()

        return {"loss_dict": loss_dict, "pred_dict": prediction}

    def predict(self, samples):
        """Inference interface for batch dicts or PGL graphs."""
        is_list = isinstance(samples, list)
        samples = samples if is_list else [samples]

        results = []
        for sample in samples:
            data = sample if isinstance(sample, dict) else {"graph": sample}
            result = self.forward(data, return_loss=False, return_prediction=True)
            results.append(
                {
                    key: value.numpy() if isinstance(value, paddle.Tensor) else value
                    for key, value in result["pred_dict"].items()
                }
            )

        return results if is_list else results[0]

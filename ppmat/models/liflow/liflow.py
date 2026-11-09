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

from collections.abc import Mapping
from typing import Any

import numpy as np
import paddle

from ppmat.models.common.runtime import RuntimeMixin
from ppmat.models.common.runtime import runtime_boundary
from ppmat.models.liflow.dual_painn import DualPaiNN


def _as_tensor(value: Any) -> paddle.Tensor:
    if isinstance(value, paddle.Tensor):
        return value
    if isinstance(value, np.ndarray):
        return paddle.to_tensor(value)
    return value


def _get_field(batch_data: Any, key: str, default: Any = None) -> Any:
    """Read ``key`` from a ``Data``/``Batch`` object or a plain mapping."""
    if isinstance(batch_data, Mapping):
        return batch_data.get(key, default)
    value = batch_data[key]
    return default if value is None else value


def _expand_to_nodes(value: Any, batch_index: paddle.Tensor, num_nodes: int):
    """Broadcast a per-graph (or already per-node) field to ``num_nodes``."""
    value = _as_tensor(value).reshape([-1])
    if value.numel() == num_nodes:
        return value
    if value.numel() == 1:
        return value.expand([num_nodes])
    return value[batch_index]


def _velocity_loss(prediction: paddle.Tensor, label: paddle.Tensor) -> paddle.Tensor:
    # Reference ``FlowModule.compute_loss``: sum over xyz, mean over nodes.
    return paddle.mean(paddle.sum((prediction - label) ** 2, axis=-1))


class LiFlow(RuntimeMixin, DualPaiNN):
    """DualPaiNN flow-matching model.

    The network parameters live directly on this layer (no ``network``
    sub-module) so that the state dict matches the released checkpoints and the
    output of ``molecular_dynamics_integrator/liflow/convert_weights.py``.
    """

    def __init__(
        self,
        num_features: int = 64,
        num_radial_basis: int = 20,
        num_layers: int = 3,
        num_elements: int = 77,
        r_max: float = 5.0,
        r_offset: float = 0.5,
        ref_temp: float = 1000.0,
        prediction_mode: str = "velocity",
        execution_backend: str = "eager",
        runtime_options: dict | None = None,
    ):
        super().__init__(
            num_features=num_features,
            num_radial_basis=num_radial_basis,
            num_layers=num_layers,
            num_elements=num_elements,
            r_max=r_max,
            r_offset=r_offset,
            ref_temp=ref_temp,
        )
        self._init_runtime(execution_backend, runtime_options)
        if prediction_mode not in {"velocity", "data"}:
            raise ValueError("prediction_mode must be 'velocity' or 'data'")
        self.prediction_mode = prediction_mode

    @runtime_boundary("forward")
    def _runtime_forward(
        self,
        condition_positions: paddle.Tensor,
        flow_positions: paddle.Tensor,
        edge_index: paddle.Tensor,
        shifts: paddle.Tensor,
        elements: paddle.Tensor,
        node_time: paddle.Tensor,
        node_temperature: paddle.Tensor,
    ) -> paddle.Tensor:
        return DualPaiNN.forward(
            self,
            condition_positions,
            flow_positions,
            edge_index,
            shifts,
            elements,
            node_time,
            node_temperature,
        )

    def forward(
        self,
        batch_data: Any,
        return_loss: bool = True,
        return_prediction: bool = True,
    ) -> dict[str, dict[str, Any]]:
        assert (
            return_loss or return_prediction
        ), "At least one of return_loss or return_prediction must be True."

        start_positions = _as_tensor(_get_field(batch_data, "start_positions"))
        end_positions = _as_tensor(_get_field(batch_data, "end_positions"))
        prior = _as_tensor(_get_field(batch_data, "prior"))
        num_nodes = int(start_positions.shape[0])

        # ``Batch.from_data_list`` exposes the node -> graph mapping as
        # ``batch``; a plain ``Data`` (or dict without it) is a single graph.
        batch_index = _get_field(batch_data, "batch")
        if batch_index is None:
            batch_index = paddle.zeros([num_nodes], dtype="int64")
        else:
            batch_index = _as_tensor(batch_index).astype("int64")

        flow_time = _get_field(batch_data, "flow_time")
        temperature = _get_field(batch_data, "temperature")
        node_time = _expand_to_nodes(flow_time, batch_index, num_nodes)
        node_temperature = _expand_to_nodes(temperature, batch_index, num_nodes)

        # ``x_0`` uses the shifted start frame; the network conditions on the
        # unshifted frame, exactly as in the reference ``training_step``.
        source = start_positions + prior
        t_node = node_time[:, None]
        flow_positions = (1.0 - t_node) * source + t_node * end_positions

        edge_index = _as_tensor(_get_field(batch_data, "edge_index")).astype("int64")
        shifts = _as_tensor(_get_field(batch_data, "shifts"))
        elements = _as_tensor(_get_field(batch_data, "elements")).astype("int64")

        prediction = self._runtime_forward(
            start_positions,
            flow_positions,
            edge_index,
            shifts,
            elements,
            node_time,
            node_temperature,
        )

        result = {"loss_dict": {}, "pred_dict": {}}
        if self.prediction_mode == "velocity":
            pred_key = "velocity"
            label = end_positions - source
        else:
            pred_key = "displacement"
            label = end_positions - start_positions

        if return_prediction:
            result["pred_dict"][pred_key] = prediction
        if return_loss:
            loss = _velocity_loss(prediction, label)
            result["loss_dict"][pred_key] = loss
            result["loss_dict"]["loss"] = loss
        return result

    @paddle.no_grad()
    def predict(self, batch_data: Any) -> dict[str, paddle.Tensor]:
        condition_positions = _as_tensor(_get_field(batch_data, "condition_positions"))
        flow_positions = _as_tensor(_get_field(batch_data, "flow_positions"))
        edge_index = _as_tensor(_get_field(batch_data, "edge_index")).astype("int64")
        shifts = _as_tensor(_get_field(batch_data, "shifts"))
        elements = _as_tensor(_get_field(batch_data, "elements")).astype("int64")
        num_nodes = int(condition_positions.shape[0])
        node_time = _as_tensor(_get_field(batch_data, "node_time")).reshape([-1])
        node_temperature = _as_tensor(
            _get_field(batch_data, "node_temperature")
        ).reshape([-1])
        if node_time.numel() == 1:
            node_time = node_time.expand([num_nodes])
        if node_temperature.numel() == 1:
            node_temperature = node_temperature.expand([num_nodes])
        prediction = self._runtime_forward(
            condition_positions,
            flow_positions,
            edge_index,
            shifts,
            elements,
            node_time,
            node_temperature,
        )
        pred_key = "velocity" if self.prediction_mode == "velocity" else "displacement"
        return {pred_key: prediction}

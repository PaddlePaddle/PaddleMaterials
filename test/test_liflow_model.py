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

"""Tests for the standard LiFlow model wrapper (:class:`ppmat.models.LiFlow`).

They pin the shared PaddleMaterials ``forward(batch_data, return_loss,
return_prediction)`` / ``predict(batch_data)`` contract and the reference
flow-matching loss formula.
"""

from __future__ import annotations

import numpy as np
import paddle
import pytest

from ppmat.models.liflow.liflow import LiFlow

N_ATOMS = 4


def _make_training_batch():
    start = paddle.to_tensor(
        np.random.RandomState(0).uniform(2.0, 5.0, size=(N_ATOMS, 3)),
        dtype="float32",
    )
    end = start + paddle.to_tensor(
        np.random.RandomState(1).uniform(-0.5, 0.5, size=(N_ATOMS, 3)),
        dtype="float32",
    )
    prior = paddle.zeros([N_ATOMS, 3], dtype="float32")
    velocity = end - (start + prior)
    return {
        "start_positions": start,
        "end_positions": end,
        "prior": prior,
        "velocity": velocity,
        "flow_time": paddle.to_tensor([0.5], dtype="float32"),
        "temperature": paddle.to_tensor([800.0], dtype="float32"),
        "elements": paddle.to_tensor([3, 8, 1, 6], dtype="int64"),
        "edge_index": paddle.to_tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype="int64"),
        "shifts": paddle.zeros([4, 3], dtype="float32"),
        "batch": paddle.zeros([N_ATOMS], dtype="int64"),
    }


def _make_inference_batch():
    return {
        "condition_positions": paddle.to_tensor(
            np.random.RandomState(2).uniform(2.0, 5.0, size=(N_ATOMS, 3)),
            dtype="float32",
        ),
        "flow_positions": paddle.to_tensor(
            np.random.RandomState(3).uniform(2.0, 5.0, size=(N_ATOMS, 3)),
            dtype="float32",
        ),
        "edge_index": paddle.to_tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype="int64"),
        "shifts": paddle.zeros([4, 3], dtype="float32"),
        "elements": paddle.to_tensor([3, 8, 1, 6], dtype="int64"),
        "node_time": paddle.to_tensor([0.2, 0.2, 0.2, 0.2], dtype="float32"),
        "node_temperature": paddle.to_tensor(
            [800.0, 800.0, 800.0, 800.0], dtype="float32"
        ),
    }


@pytest.fixture(scope="module")
def model():
    return LiFlow(
        num_features=16,
        num_radial_basis=8,
        num_layers=2,
        num_elements=77,
        r_max=5.0,
        r_offset=0.5,
        ref_temp=1000.0,
        prediction_mode="velocity",
        execution_backend="eager",
    )


@pytest.fixture
def training_batch():
    return _make_training_batch()


@pytest.fixture
def inference_batch():
    return _make_inference_batch()


def test_liflow_forward_contract(model, training_batch):
    output = model(training_batch)
    assert set(output) == {"loss_dict", "pred_dict"}
    assert set(output["loss_dict"]) == {"velocity", "loss"}
    assert set(output["pred_dict"]) == {"velocity"}
    assert output["loss_dict"]["loss"].ndim == 0
    output["loss_dict"]["loss"].backward()


def test_liflow_loss_matches_reference_formula(model, training_batch):
    output = model(training_batch)
    pred = output["pred_dict"]["velocity"]
    label = training_batch["velocity"]
    expected = paddle.mean(paddle.sum((pred - label) ** 2, axis=-1))
    np.testing.assert_allclose(
        output["loss_dict"]["loss"].numpy(),
        expected.numpy(),
        rtol=1e-6,
        atol=1e-7,
    )


def test_predict_does_not_require_future_endpoint(model, inference_batch):
    assert "end_positions" not in inference_batch
    prediction = model.predict(inference_batch)
    assert set(prediction) == {"velocity"}
    assert prediction["velocity"].shape == [N_ATOMS, 3]

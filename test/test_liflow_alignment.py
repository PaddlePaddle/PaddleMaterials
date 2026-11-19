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

"""End-to-end alignment tests for the pure-tensor ``DualPaiNN`` port.

The reference fixture ``reference_model.npz`` is produced by
``scripts/generate_liflow_reference_model.py`` running the original PyTorch model.
These tests load the exact parameter/buffer arrays (strictly), exercise the ported
forward on the same fixed inputs, and compare output shape, gradients, rotation
equivariance, and the periodic neighbor list against the recorded reference.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import paddle
import pytest

from ppmat.models.liflow.dual_painn import DualPaiNN
from ppmat.models.liflow.geometry import get_neighbor_list_batch

REFERENCE_NPZ = (
    Path(__file__).resolve().parent / "fixtures" / "liflow" / "reference_model.npz"
)

INPUT_KEYS = (
    "condition_positions",
    "flow_positions",
    "edge_index",
    "shifts",
    "elements",
    "node_time",
    "node_temperature",
)


def _load_state(model, ref_state: dict[str, np.ndarray]):
    """Strict-load a torch reference state, transposing Linear weights.

    torch stores ``nn.Linear.weight`` as ``[out, in]`` while Paddle stores it as
    ``[in, out]``; biases and buffers transfer as-is.
    """
    state = {}
    for name, arr in ref_state.items():
        # torch stores Linear weights as [out, in] while Paddle stores them as
        # [in, out]; biases and buffers transfer as-is.  Embedding tables share
        # the [num_embeddings, emb_dim] layout in both frameworks, so they must
        # NOT be transposed.
        is_embedding = name.startswith("atom_embedding")
        if name.endswith(".weight") and arr.ndim == 2 and not is_embedding:
            arr = arr.T
        state[name] = paddle.to_tensor(arr.copy())
    missing, unexpected = model.set_state_dict(state)
    assert not missing, f"missing keys: {missing}"
    assert not unexpected, f"unexpected keys: {unexpected}"
    return model


@pytest.fixture(scope="module")
def reference():
    return np.load(REFERENCE_NPZ, allow_pickle=False)


@pytest.fixture(scope="module")
def ref_state(reference):
    states = {}
    for key in reference.files:
        if "/state/" in key:
            name = key.partition("model/state/")[2]
            states[name] = reference[key]
    return states


def _model_config(reference) -> dict:
    def g(k):
        v = reference[f"model/config/{k}"]
        return float(v) if v.dtype == np.float32 else int(v)

    return {
        k: g(k)
        for k in (
            "num_features",
            "num_radial_basis",
            "num_layers",
            "num_elements",
            "r_max",
            "r_offset",
            "ref_temp",
        )
    }


@pytest.fixture(scope="module")
def model(reference, ref_state):
    m = DualPaiNN(**_model_config(reference))
    _load_state(m, ref_state)
    return m


@pytest.fixture(scope="module")
def reference_batch(reference):
    return {
        key: paddle.to_tensor(reference[f"model/{key}"]).clone() for key in INPUT_KEYS
    }


# ---------------------------------------------------------------------------
# Step 1 / Step 4: shape, parameter and input gradients, state contract
# ---------------------------------------------------------------------------


def test_dual_painn_output_shape_and_gradients(model, reference_batch):
    output = model(**reference_batch)
    loss = paddle.sum(output**2)
    loss.backward()
    assert output.shape == [reference_batch["condition_positions"].shape[0], 3]
    assert all(p.grad is not None for p in model.parameters())


def test_dual_painn_state_dict_matches_reference(model, reference):
    assert set(model.state_dict()) == set(
        k.partition("model/state/")[2] for k in reference.files if "/state/" in k
    )


def test_dual_painn_output_matches_reference(model, reference_batch, reference):
    output = model(**reference_batch)
    np.testing.assert_allclose(
        output.numpy(), reference["model/output"], rtol=1e-5, atol=1e-6
    )


def test_dual_painn_input_gradients_match_reference(model, reference):
    # Leaf endpoint tensors need grad to back-propagate input gradients.
    condition = paddle.to_tensor(
        reference["model/condition_positions"], stop_gradient=False
    )
    flow = paddle.to_tensor(reference["model/flow_positions"], stop_gradient=False)
    batch = {
        "condition_positions": condition,
        "flow_positions": flow,
        "edge_index": paddle.to_tensor(reference["model/edge_index"]),
        "shifts": paddle.to_tensor(reference["model/shifts"]),
        "elements": paddle.to_tensor(reference["model/elements"]),
        "node_time": paddle.to_tensor(reference["model/node_time"]),
        "node_temperature": paddle.to_tensor(reference["model/node_temperature"]),
    }
    output = model(**batch)
    paddle.sum(output**2).backward()
    np.testing.assert_allclose(
        condition.grad.numpy(),
        reference["model/grad/condition_positions"],
        rtol=2e-4,
        atol=2e-5,
    )
    np.testing.assert_allclose(
        flow.grad.numpy(),
        reference["model/grad/flow_positions"],
        rtol=2e-4,
        atol=2e-5,
    )


# ---------------------------------------------------------------------------
# Step 1: rotation equivariance
# ---------------------------------------------------------------------------


@pytest.fixture
def rotation():
    rng = np.random.default_rng(20260910)
    q = rng.normal(size=3)
    q = q / np.linalg.norm(q)
    theta = 0.6
    K = np.array([[0.0, -q[2], q[1]], [q[2], 0.0, -q[0]], [-q[1], q[0], 0.0]])
    return np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)


def test_dual_painn_rotation_equivariance(model, reference_batch, rotation):
    expected = model(**reference_batch).numpy() @ rotation.T
    rotated = dict(reference_batch)
    for key in ("condition_positions", "flow_positions", "shifts"):
        rotated[key] = paddle.to_tensor(
            (reference_batch[key].numpy() @ rotation.T).astype("float32")
        )
    rotated["edge_index"] = reference_batch["edge_index"]
    rotated["elements"] = reference_batch["elements"]
    rotated["node_time"] = reference_batch["node_time"]
    rotated["node_temperature"] = reference_batch["node_temperature"]
    np.testing.assert_allclose(model(**rotated).numpy(), expected, rtol=2e-5, atol=2e-6)


# ---------------------------------------------------------------------------
# Step 1: neighbor list matches the reference fixture
# ---------------------------------------------------------------------------


def test_neighbor_list_matches_reference_fixture(reference):
    edge_index, shifts = get_neighbor_list_batch(
        reference["nb/positions_batch"],
        reference["nb/lattice"],
        cutoff=float(reference["nb/cutoff"]),
        periodic=True,
    )
    np.testing.assert_array_equal(edge_index, reference["nb/edge_index"])
    np.testing.assert_allclose(shifts, reference["nb/shifts"], rtol=0.0, atol=1e-7)

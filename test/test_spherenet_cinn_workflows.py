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

import os
from pathlib import Path

import numpy as np
import paddle
import pytest
from cinn_workflow_harness import WorkflowCase
from cinn_workflow_harness import assert_gpu_cinn_matches_eager
from cinn_workflow_harness import assert_predictor_matches_checkpoint
from cinn_workflow_harness import assert_resume_parity
from cinn_workflow_harness import requires_gpu_cinn

from ppmat.datasets.collate_fn import RadiusGraphCollator
from ppmat.models.common.graph_converter import RadiusGraphConverter
from ppmat.models.spherenet.spherenet import SphereNet
from ppmat.predictor import PropertyPredictor


@pytest.fixture
def tensor_runtime_proxy(monkeypatch):
    """Exercise public workflow hooks without requiring a GPU compiler."""

    monkeypatch.setattr(
        SphereNet,
        "validate_execution_backend",
        lambda self, **kwargs: None,
    )
    monkeypatch.setattr(
        SphereNet,
        "_run_runtime",
        lambda self, name, layer, *args, **kwargs: layer(*args),
    )


def _make_graphs():
    converter = RadiusGraphConverter(
        cutoff=5.0,
        return_triplet_indices=True,
        num_cpus=1,
    )
    graph_a = converter.from_arrays(
        np.asarray([6, 1, 7, 8], dtype=np.int64),
        np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.2, 1.1, 0.1],
                [0.1, 0.3, 1.2],
            ],
            dtype=np.float32,
        ),
    )
    graph_b = converter.from_arrays(
        np.asarray([6, 1, 7, 8, 9], dtype=np.int64),
        np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.2, 0.0, 0.0],
                [0.0, 1.3, 0.0],
                [0.0, 0.0, 1.4],
                [1.0, 1.0, 0.2],
            ],
            dtype=np.float32,
        ),
    )
    return graph_a, graph_b


def _make_samples():
    graph_a, graph_b = _make_graphs()
    return [
        {"graph": graph_a, "mu": np.asarray([0.25], dtype=np.float32)},
        {"graph": graph_b, "mu": np.asarray([-0.75], dtype=np.float32)},
    ]


def _make_loader():
    return paddle.io.DataLoader(
        _make_samples(),
        batch_size=2,
        shuffle=False,
        collate_fn=RadiusGraphCollator(),
        return_list=True,
    )


def _make_model(execution_backend="cinn", compute_forces=False):
    with paddle.utils.unique_name.guard():
        paddle.seed(2026)
        return SphereNet(
            derivative_properties=("force",) if compute_forces else (),
            num_layers=1,
            hidden_channels=8,
            out_channels=1,
            int_emb_size=4,
            basis_emb_size_dist=4,
            basis_emb_size_angle=4,
            basis_emb_size_torsion=4,
            out_emb_channels=8,
            num_spherical=2,
            num_radial=3,
            num_before_skip=1,
            num_after_skip=1,
            num_output_layers=1,
            property_name="mu",
            execution_backend=execution_backend,
            runtime_options={"cinn": {"full_graph": compute_forces}},
        )


def test_force_coordinates_are_leaf_before_runtime_boundary(monkeypatch):
    model = _make_model(execution_backend="eager", compute_forces=True)
    observed = {}

    def capture_coordinates(
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
        observed["stop_gradient"] = pos.stop_gradient
        return (
            paddle.sum(pos).reshape([1, 1]),
            pos,
            -paddle.ones_like(pos),
        )

    monkeypatch.setattr(SphereNet, "_runtime_forward", capture_coordinates)
    model._forward({"graph": _make_graphs()[0]})

    assert observed["stop_gradient"] is False


def _predictor_config(checkpoint_path, execution_backend="cinn"):
    return {
        "Model": {
            "__class_name__": "SphereNet",
            "__init_params__": {
                "num_layers": 1,
                "hidden_channels": 8,
                "out_channels": 1,
                "int_emb_size": 4,
                "basis_emb_size_dist": 4,
                "basis_emb_size_angle": 4,
                "basis_emb_size_torsion": 4,
                "out_emb_channels": 8,
                "num_spherical": 2,
                "num_radial": 3,
                "num_before_skip": 1,
                "num_after_skip": 1,
                "num_output_layers": 1,
                "property_name": "mu",
            },
        },
        "Execution": {
            "backend": execution_backend,
            "__init_params__": {"full_graph": False},
        },
        "Predict": {
            "checkpoint_path": str(checkpoint_path),
            "eval_with_no_grad": True,
            "graph_converter": {
                "__class_name__": "RadiusGraphConverter",
                "__init_params__": {
                    "cutoff": 5.0,
                    "return_triplet_indices": True,
                    "num_cpus": 1,
                },
            },
        },
    }


def _example_xyz(predictor):
    path = (
        Path(__file__).resolve().parents[1]
        / "property_prediction"
        / "example_data"
        / "molecules"
        / "isoguvacine.xyz"
    )
    return predictor.from_xyz_file(str(path))


SPHERENET_CASE = WorkflowCase(
    name="spherenet",
    model_cls=SphereNet,
    make_model=_make_model,
    make_loader=_make_loader,
    predictor_config=_predictor_config,
    predictor_cls=PropertyPredictor,
    property_names={"mu"},
    atol=3e-6,
    gpu_atol=3e-5,
    # SphereNet reads molecules, so the shared CIF/structure batch does not apply.
    predict_one=_example_xyz,
    predict_batch=None,
)


def test_trainer_checkpoint_resume_matches_uninterrupted_training(
    tmp_path, tensor_runtime_proxy
):
    assert_resume_parity(SPHERENET_CASE, tmp_path)


def test_property_predictor_loads_checkpoint_and_predicts_xyz(
    tmp_path, tensor_runtime_proxy
):
    assert_predictor_matches_checkpoint(SPHERENET_CASE, tmp_path)


@pytest.mark.skipif(
    os.environ.get("PPMAT_RUN_CINN_WORKFLOW_TESTS") != "1",
    reason="Set PPMAT_RUN_CINN_WORKFLOW_TESTS=1 for the GPU workflow.",
)
def test_gpu_cinn_force_matches_eager():
    if not paddle.is_compiled_with_cuda():
        pytest.skip("Paddle was not compiled with CUDA.")
    if not paddle.base.is_compiled_with_cinn():
        pytest.skip("Paddle was not compiled with CINN.")

    paddle.set_device("gpu:0")
    eager_model = _make_model(execution_backend="eager", compute_forces=True)
    cinn_model = _make_model(execution_backend="cinn", compute_forces=True)
    cinn_model.set_state_dict(eager_model.state_dict())
    eager_model.eval()
    cinn_model.eval()

    eager_result = eager_model.predict(_make_graphs()[0])
    cinn_result = cinn_model.predict(_make_graphs()[0])
    eager_force = np.asarray(eager_result["force"])
    cinn_force = np.asarray(cinn_result["force"])

    assert cinn_model.get_runtime_options("cinn")["full_graph"] is True
    assert set(cinn_model._runtime_cache) == {("cinn", "eval", "forward")}
    assert np.linalg.norm(eager_force) > 0
    assert np.linalg.norm(cinn_force) > 0
    np.testing.assert_allclose(
        cinn_result["mu"], eager_result["mu"], atol=5e-4, rtol=1e-4
    )
    np.testing.assert_allclose(cinn_force, eager_force, atol=5e-4, rtol=1e-4)


@requires_gpu_cinn
def test_gpu_cinn_trainer_and_predictor_match_eager(tmp_path):
    assert_gpu_cinn_matches_eager(SPHERENET_CASE, tmp_path)

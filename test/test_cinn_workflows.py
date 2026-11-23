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

"""Trainer/Predictor workflow contract for property models with a CINN boundary.

Each model contributes a WorkflowCase; the assertions themselves live in
cinn_workflow_harness so adding a model does not duplicate a test body.
"""

from __future__ import annotations

import numpy as np
import paddle
import pgl
import pytest
from cinn_workflow_harness import WorkflowCase
from cinn_workflow_harness import assert_gpu_cinn_matches_eager
from cinn_workflow_harness import assert_predictor_matches_checkpoint
from cinn_workflow_harness import assert_resume_parity
from cinn_workflow_harness import patch_cinn_runtime
from cinn_workflow_harness import requires_gpu_cinn

from ppmat.datasets.collate_fn import DefaultCollator
from ppmat.models.dimenetpp.dimenetpp import DimeNetPlusPlus
from ppmat.models.megnet.megnet import MEGNetPlus
from ppmat.predictor import PropertyPredictor

PROPERTY = "formation_energy_per_atom"

MEGNET_PARAMS = {
    "dim_node_embedding": 4,
    "dim_edge_embedding": 6,
    "nblocks": 1,
    "hidden_layer_sizes_input": [8, 4],
    "hidden_layer_sizes_conv": [8, 8, 4],
    "hidden_layer_sizes_output": [8, 4],
    "nlayers_set2set": 1,
    "niters_set2set": 2,
    "bond_expansion_cfg": {
        "rbf_type": "Gaussian",
        "initial": 0.0,
        "final": 5.0,
        "num_centers": 6,
        "width": 0.5,
    },
}

DIMENETPP_PARAMS = {
    "out_channels": 1,
    "hidden_channels": 8,
    "num_blocks": 1,
    "int_emb_size": 4,
    "basis_emb_size": 2,
    "out_emb_channels": 8,
    "num_spherical": 2,
    "num_embeddings": 95,
    "num_radial": 2,
    "cutoff": 7.0,
    "num_before_skip": 1,
    "num_after_skip": 1,
    "num_output_layers": 1,
    "readout": "mean",
    "loss_type": "mse_loss",
}


def _loader(samples):
    return paddle.io.DataLoader(
        samples,
        batch_size=2,
        shuffle=False,
        collate_fn=DefaultCollator(),
        return_list=True,
    )


def _predictor_config(class_name, init_params, checkpoint_path, backend):
    return {
        "Model": {
            "__class_name__": class_name,
            "__init_params__": dict(init_params),
        },
        "Execution": {
            "backend": backend,
            "__init_params__": {"full_graph": False},
        },
        "Predict": {
            "checkpoint_path": str(checkpoint_path),
            "eval_with_no_grad": True,
            "graph_converter": {
                "__class_name__": "FindPointsInSpheres",
                "__init_params__": {"cutoff": 4.0, "num_cpus": 1},
            },
        },
    }


# ------------------------------------------------------------------- MEGNet


def _megnet_graph(atom_types):
    return pgl.Graph(
        np.asarray([[0, 1], [1, 0]], dtype=np.int64),
        num_nodes=2,
        node_feat={"atom_types": np.asarray(atom_types, dtype=np.int64)},
        edge_feat={"bond_dist": np.asarray([1.0, 1.0], dtype=np.float32)},
    )


def _megnet_loader():
    return _loader(
        [
            {"graph": _megnet_graph([1, 8]), PROPERTY: np.float32([0.25])},
            {"graph": _megnet_graph([6, 7]), PROPERTY: np.float32([-0.75])},
        ]
    )


def _megnet_model(backend):
    with paddle.utils.unique_name.guard():
        paddle.seed(2026)
        return MEGNetPlus(**MEGNET_PARAMS, execution_backend=backend)


# --------------------------------------------------------------- DimeNet++


def _dimenetpp_graph(atom_types, coordinate_shift=0.0):
    num_nodes = len(atom_types)
    coordinates = (
        np.asarray(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32
        )[:num_nodes]
        + coordinate_shift
    )
    edges = np.asarray(
        [
            [source, destination]
            for source in range(num_nodes)
            for destination in range(num_nodes)
            if source != destination
        ],
        dtype=np.int64,
    )
    return pgl.Graph(
        edges=edges,
        num_nodes=num_nodes,
        node_feat={
            "atom_types": np.asarray(atom_types, dtype=np.int64),
            "frac_coords": coordinates / 10.0,
            "cart_coords": coordinates,
            "lattice": (np.eye(3, dtype=np.float32) * 10.0).reshape([1, 3, 3]),
            "num_atoms": np.asarray([num_nodes], dtype=np.int64),
        },
        edge_feat={
            "pbc_offset": np.zeros([edges.shape[0], 3], dtype=np.int64),
            "num_edges": np.asarray([edges.shape[0]], dtype=np.int64),
        },
    )


def _dimenetpp_loader():
    return _loader(
        [
            {"graph": _dimenetpp_graph([1, 8, 14]), PROPERTY: np.float32([0.25])},
            {
                "graph": _dimenetpp_graph([6, 7, 8], coordinate_shift=0.125),
                PROPERTY: np.float32([-0.75]),
            },
        ]
    )


def _dimenetpp_model(backend):
    with paddle.utils.unique_name.guard():
        paddle.seed(2026)
        model = DimeNetPlusPlus(**DIMENETPP_PARAMS, execution_backend=backend)
    # Deterministic Bessel frequencies keep eager and compiled runs comparable.
    model.rbf.freq.set_value(
        paddle.arange(1, model.rbf.freq.shape[0] + 1, dtype="float32") * np.pi
    )
    return model


CASES = [
    WorkflowCase(
        name="megnet",
        model_cls=MEGNetPlus,
        make_model=_megnet_model,
        make_loader=_megnet_loader,
        predictor_config=lambda checkpoint, backend: _predictor_config(
            "MEGNetPlus",
            {**MEGNET_PARAMS, "property_name": PROPERTY},
            checkpoint,
            backend,
        ),
        predictor_cls=PropertyPredictor,
        property_names={PROPERTY},
        optimizer_kwargs={"beta1": 0.9, "beta2": 0.999},
    ),
    WorkflowCase(
        name="dimenetpp",
        model_cls=DimeNetPlusPlus,
        make_model=_dimenetpp_model,
        make_loader=_dimenetpp_loader,
        predictor_config=lambda checkpoint, backend: _predictor_config(
            "DimeNetPlusPlus",
            {**DIMENETPP_PARAMS, "property_names": PROPERTY},
            checkpoint,
            backend,
        ),
        predictor_cls=PropertyPredictor,
        property_names={PROPERTY},
    ),
]


@pytest.fixture(params=CASES, ids=lambda case: case.name)
def case(request, monkeypatch):
    patch_cinn_runtime(monkeypatch, request.param.model_cls)
    return request.param


def test_trainer_checkpoint_resume_matches_uninterrupted_training(case, tmp_path):
    assert_resume_parity(case, tmp_path)


def test_property_predictor_loads_checkpoint_and_batches_cifs(case, tmp_path):
    assert_predictor_matches_checkpoint(case, tmp_path)


@requires_gpu_cinn
@pytest.mark.parametrize("gpu_case", CASES, ids=lambda case: case.name)
def test_gpu_cinn_matches_eager(gpu_case, tmp_path):
    assert_gpu_cinn_matches_eager(gpu_case, tmp_path)

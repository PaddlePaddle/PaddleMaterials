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

"""LiFlow's forward was restructured into one differentiable tensor graph.

These tests pin the runtime boundary contract the shared
:class:`ppmat.models.common.runtime.RuntimeMixin` exposes: a single
``@runtime_boundary("forward")`` over the whole velocity field, a runtime cache
that is compiled once per (backend, mode) and reused across calls, and config
propagation from the public Trainer / IntegratorPredictor down to both LiFlow
models with ``full_graph=False``.  They drive the public backend hooks directly
(``patch_cinn_compile``) so no GPU CINN compiler is required on CPU/CI.
"""

from __future__ import annotations

import numpy as np
import paddle
import pytest

import ppmat.models.common.cinn as cinn_module
from ppmat.models.liflow.liflow import LiFlow
from ppmat.predictor import IntegratorPredictor

N_ATOMS = 4


def _make_model(execution_backend="eager"):
    with paddle.utils.unique_name.guard():
        paddle.seed(2026)
        return LiFlow(
            num_features=16,
            num_radial_basis=8,
            num_layers=2,
            num_elements=77,
            r_max=5.0,
            r_offset=0.5,
            ref_temp=1000.0,
            prediction_mode="velocity",
            execution_backend=execution_backend,
            runtime_options={"cinn": {"full_graph": False}},
        )


def _make_inference_batch():
    rng_pos = np.random.RandomState(2)
    rng_vel = np.random.RandomState(3)
    return {
        "condition_positions": paddle.to_tensor(
            rng_pos.uniform(2.0, 5.0, size=(N_ATOMS, 3)), dtype="float32"
        ),
        "flow_positions": paddle.to_tensor(
            rng_vel.uniform(2.0, 5.0, size=(N_ATOMS, 3)), dtype="float32"
        ),
        "edge_index": paddle.to_tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype="int64"),
        "shifts": paddle.zeros([4, 3], dtype="float32"),
        "elements": paddle.to_tensor([3, 8, 1, 6], dtype="int64"),
        "node_time": paddle.full([N_ATOMS], 0.2, dtype="float32"),
        "node_temperature": paddle.full([N_ATOMS], 800.0, dtype="float32"),
    }


@pytest.fixture
def inference_batch():
    return _make_inference_batch()


def patch_cinn_compile(monkeypatch):
    """Compile the boundary by running the eager callable it wraps."""

    def fake_compile(function, **kwargs):
        del kwargs
        return function

    monkeypatch.setattr(cinn_module, "compile_cinn", fake_compile)
    monkeypatch.setattr(
        LiFlow, "validate_execution_backend", lambda self, **kwargs: None
    )


def test_liflow_uses_single_forward_boundary(monkeypatch, inference_batch):
    model = _make_model(execution_backend="cinn")
    model.eval()
    boundary_names = []
    monkeypatch.setattr(
        LiFlow,
        "_run_runtime",
        lambda self, name, function, *args, **kwargs: boundary_names.append(name)
        or function(*args, **kwargs),
    )

    prediction = model.predict(inference_batch)

    assert boundary_names == ["forward"]
    assert set(prediction) == {"velocity"}
    assert prediction["velocity"].shape == [N_ATOMS, 3]
    assert model.get_runtime_options("cinn") == {"full_graph": False}


def test_runtime_cache_compiled_once_per_mode(monkeypatch, inference_batch):
    patch_cinn_compile(monkeypatch)
    model = _make_model(execution_backend="cinn")
    model.eval()

    model.predict(inference_batch)
    first = model._runtime_cache[("cinn", "eval", "forward")]
    model.predict(inference_batch)

    assert model._runtime_cache[("cinn", "eval", "forward")] is first


def test_state_dict_keys_unchanged_by_backend():
    assert (
        _make_model(execution_backend="cinn").state_dict().keys()
        == _make_model(execution_backend="eager").state_dict().keys()
    )


def test_config_propagates_to_trainer(monkeypatch, tmp_path):
    patch_cinn_compile(monkeypatch)
    from ppmat.trainer.base_trainer import BaseTrainer

    model = _make_model(execution_backend="eager")
    optimizer = paddle.optimizer.Adam(learning_rate=1e-4, parameters=model.parameters())

    class _EmptyLoader:
        def __iter__(self):
            return iter([])

    trainer = BaseTrainer(
        {
            "max_epochs": 1,
            "output_dir": str(tmp_path),
            "save_freq": 0,
            "log_freq": 100,
            "start_eval_epoch": 1,
            "eval_freq": 1,
            "seed": 2026,
            "compute_metric_during_train": False,
            "use_amp": False,
            "eval_with_no_grad": True,
            "gradient_accumulation_steps": 1,
        },
        model,
        train_dataloader=_EmptyLoader(),
        val_dataloader=_EmptyLoader(),
        optimizer=optimizer,
        execution_config={
            "backend": "cinn",
            "__init_params__": {"full_graph": False},
        },
    )

    assert trainer.execution_backend == "cinn"
    assert trainer.model.get_runtime_options("cinn") == {"full_graph": False}


def test_config_propagates_to_integrator_predictor(monkeypatch):
    patch_cinn_compile(monkeypatch)
    propagator = _make_model(execution_backend="cinn")
    corrector = _make_model(execution_backend="cinn")

    predictor = IntegratorPredictor(
        propagator_model=propagator,
        corrector_model=corrector,
        cutoff=5.0,
        periodic=True,
    )

    assert predictor.propagator.get_runtime_options("cinn") == {"full_graph": False}
    assert predictor.corrector.get_runtime_options("cinn") == {"full_graph": False}

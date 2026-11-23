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

"""Shared scaffolding for the CINN execution-backend workflow tests.

Every model that implements the compiled-runtime protocol is checked the same
way: a Trainer run can be interrupted and resumed with matching weights, a
Predictor restores that checkpoint and batches identically, and selecting the
backend never changes state-dict keys.  Only the model factory and its sample
batch differ per model, so a per-model test file supplies just those.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Callable

import numpy as np
import paddle
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_CIF = (
    REPO_ROOT / "property_prediction" / "example_data" / "cifs" / "mp-18767-LiMnO2.cif"
)


def predict_example_cif(predictor):
    """Default single-input prediction: the shared LiMnO2 example structure."""

    return predictor.from_cif_file(str(EXAMPLE_CIF))


def predict_example_cif_twice(predictor):
    """Default batched prediction: the same structure repeated."""

    from pymatgen.core import Structure

    structure = Structure.from_file(EXAMPLE_CIF)
    return predictor.from_structures([structure, structure])


@dataclass(frozen=True)
class WorkflowCase:
    """Everything the shared workflow assertions need about one model."""

    name: str
    model_cls: type
    make_model: Callable[[str], paddle.nn.Layer]
    make_loader: Callable[[], paddle.io.DataLoader]
    predictor_config: Callable[[Path, str], dict]
    predictor_cls: type
    property_names: set[str]
    atol: float = 2e-6
    gpu_atol: float = 2e-5
    optimizer_kwargs: dict = field(default_factory=dict)
    # Input format is model-specific: crystal models read a CIF, molecular
    # models read an XYZ. Batched prediction is optional.
    predict_one: Callable[[object], list] = predict_example_cif
    predict_batch: Callable[[object], list] | None = predict_example_cif_twice


def patch_cinn_runtime(monkeypatch, model_cls: type) -> None:
    """Drive the public backend hooks without a GPU compiler.

    The workflow contract under test is orchestration: config resolution,
    checkpoint keys, resume arithmetic.  Compilation itself is covered by the
    GPU-gated tests, so here the boundary runs the eager callable it wraps.
    """

    monkeypatch.setattr(
        model_cls,
        "validate_execution_backend",
        lambda self, **kwargs: None,
    )
    monkeypatch.setattr(
        model_cls,
        "_run_runtime",
        lambda self, name, function, *args, **kwargs: function(*args, **kwargs),
    )


def trainer_config(output_dir, max_epochs: int) -> dict:
    return {
        "max_epochs": max_epochs,
        "output_dir": str(output_dir),
        "save_freq": 1,
        "log_freq": 100,
        "start_eval_epoch": 1,
        "eval_freq": 1,
        "seed": 2026,
        "pretrained_model_path": None,
        "resume_from_checkpoint": None,
        "compute_metric_during_train": False,
        "use_amp": False,
        "eval_with_no_grad": True,
        "gradient_accumulation_steps": 1,
        "best_metric_indicator": "eval_loss",
        "name_for_best_metric": "loss",
        "greater_is_better": False,
    }


def build_trainer(case: WorkflowCase, model, output_dir, max_epochs, backend="cinn"):
    """Build a Trainer that selects the backend through the Execution block."""

    from ppmat.trainer.base_trainer import BaseTrainer

    optimizer = paddle.optimizer.Adam(
        learning_rate=1e-3,
        parameters=model.parameters(),
        **case.optimizer_kwargs,
    )
    trainer = BaseTrainer(
        trainer_config(output_dir, max_epochs),
        model,
        train_dataloader=case.make_loader(),
        val_dataloader=case.make_loader(),
        optimizer=optimizer,
        execution_config={"backend": backend},
    )
    return trainer, optimizer


def assert_state_dict_close(actual, expected, atol: float) -> None:
    assert actual.keys() == expected.keys()
    for key in actual:
        np.testing.assert_allclose(
            actual[key].numpy(), expected[key].numpy(), atol=atol, rtol=atol
        )


def assert_checkpoint_keys_unchanged(checkpoint_state, model) -> None:
    """Enabling a compiled backend must not rename or nest any parameter."""

    assert checkpoint_state.keys() == model.state_dict().keys()
    assert all(not key.startswith("model.") for key in checkpoint_state)


def assert_resume_parity(case: WorkflowCase, tmp_path) -> None:
    """One epoch, checkpoint, resume for a second epoch == two epochs straight."""

    initial_state = case.make_model("cinn").state_dict()

    first_model = case.make_model("cinn")
    first_model.set_state_dict(initial_state)
    first_trainer, _ = build_trainer(case, first_model, tmp_path / "first", 1)
    with paddle.utils.unique_name.guard():
        first_trainer.train()

    checkpoint_dir = tmp_path / "first" / "checkpoints"
    for prefix in ("epoch_1", "latest", "best"):
        for suffix in ("pdparams", "pdopt", "pdstates"):
            assert (checkpoint_dir / f"{prefix}.{suffix}").is_file()
    assert_checkpoint_keys_unchanged(
        paddle.load(str(checkpoint_dir / "latest.pdparams")), first_model
    )

    resumed_model = case.make_model("cinn")
    resumed_trainer, resumed_optimizer = build_trainer(
        case, resumed_model, tmp_path / "resumed", 2
    )
    with paddle.utils.unique_name.guard():
        resumed_trainer.train(resume_from_checkpoint=str(checkpoint_dir / "latest"))

    straight_model = case.make_model("cinn")
    straight_model.set_state_dict(initial_state)
    straight_trainer, straight_optimizer = build_trainer(
        case, straight_model, tmp_path / "straight", 2
    )
    with paddle.utils.unique_name.guard():
        straight_trainer.train()

    assert first_trainer.state.global_step == 1
    assert resumed_trainer.state.global_step == 2
    assert resumed_trainer.state.epoch == 2
    assert resumed_optimizer.get_lr() == straight_optimizer.get_lr()
    assert_state_dict_close(
        resumed_model.state_dict(), straight_model.state_dict(), case.atol
    )
    assert_state_dict_close(
        resumed_optimizer.state_dict(), straight_optimizer.state_dict(), case.atol
    )


def _write_predictor_config(case: WorkflowCase, tmp_path, checkpoint_path, backend):
    from omegaconf import OmegaConf

    config_path = tmp_path / f"{case.name}_{backend}.yaml"
    OmegaConf.save(
        OmegaConf.create(case.predictor_config(checkpoint_path, backend)), config_path
    )
    return config_path


def assert_predictor_matches_checkpoint(case: WorkflowCase, tmp_path, device="cpu"):
    """A predictor restores the eager checkpoint and batches consistently."""

    model = case.make_model("eager")
    checkpoint_path = tmp_path / "best.pdparams"
    paddle.save(model.state_dict(), str(checkpoint_path))
    config_path = _write_predictor_config(case, tmp_path, checkpoint_path, "cinn")

    predictor = case.predictor_cls(config_path=config_path, device=device)
    single = case.predict_one(predictor)

    assert predictor.execution_backend == "cinn"
    assert single[0].keys() == case.property_names
    if case.predict_batch is not None:
        batch = case.predict_batch(predictor)
        assert len(batch) == 2
        for key in case.property_names:
            np.testing.assert_allclose(
                [item[key] for item in batch],
                [single[0][key]] * 2,
                atol=case.atol,
                rtol=case.atol,
            )
    assert predictor.model.state_dict().keys() == model.state_dict().keys()
    assert all(not key.startswith("model.") for key in predictor.model.state_dict())


requires_gpu_cinn = pytest.mark.skipif(
    os.environ.get("PPMAT_RUN_CINN_WORKFLOW_TESTS") != "1",
    reason=(
        "Set PPMAT_RUN_CINN_WORKFLOW_TESTS=1 to run the GPU Trainer/Predictor "
        "workflow smoke."
    ),
)


def gpu_cinn_or_skip() -> None:
    if not paddle.is_compiled_with_cuda():
        pytest.skip("Paddle was not compiled with CUDA.")
    if not paddle.base.is_compiled_with_cinn():
        pytest.skip("Paddle was not compiled with CINN.")
    paddle.set_device("gpu:0")


def assert_gpu_cinn_matches_eager(case: WorkflowCase, tmp_path) -> None:
    """On real hardware, one compiled epoch must match one eager epoch."""

    gpu_cinn_or_skip()
    initial_state = {
        key: value.clone()
        for key, value in case.make_model("eager").state_dict().items()
    }

    eager_model = case.make_model("eager")
    eager_model.set_state_dict(initial_state)
    eager_trainer, eager_optimizer = build_trainer(
        case, eager_model, tmp_path / "eager", 1, backend="eager"
    )
    cinn_model = case.make_model("cinn")
    cinn_model.set_state_dict(initial_state)
    cinn_trainer, cinn_optimizer = build_trainer(case, cinn_model, tmp_path / "cinn", 1)
    with paddle.utils.unique_name.guard():
        eager_trainer.train()
    with paddle.utils.unique_name.guard():
        cinn_trainer.train()

    assert eager_trainer.state.global_step == cinn_trainer.state.global_step == 1
    assert_state_dict_close(
        cinn_model.state_dict(), eager_model.state_dict(), case.gpu_atol
    )
    assert_state_dict_close(
        cinn_optimizer.state_dict(), eager_optimizer.state_dict(), case.gpu_atol
    )

    checkpoint = tmp_path / "cinn" / "checkpoints" / "latest.pdparams"
    cinn_result = case.predict_one(
        case.predictor_cls(
            config_path=_write_predictor_config(case, tmp_path, checkpoint, "cinn"),
            device="gpu:0",
        )
    )
    eager_result = case.predict_one(
        case.predictor_cls(
            config_path=_write_predictor_config(case, tmp_path, checkpoint, "eager"),
            device="gpu:0",
        )
    )
    assert cinn_result[0].keys() == case.property_names
    for key in case.property_names:
        np.testing.assert_allclose(
            cinn_result[0][key],
            eager_result[0][key],
            atol=case.gpu_atol,
            rtol=case.gpu_atol,
        )

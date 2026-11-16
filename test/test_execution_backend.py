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

import ast
import inspect
from pathlib import Path

import numpy as np
import paddle
import pytest

import ppmat.models as models
import ppmat.models.common.cinn as cinn_module
import ppmat.models.common.runtime as runtime_module
from ppmat.models.common.runtime import RuntimeMixin
from ppmat.models.common.runtime import register_runtime_backend
from ppmat.models.common.runtime import runtime_boundary
from ppmat.predictor import BasePredictor
from ppmat.trainer.base_trainer import BaseTrainer
from ppmat.utils.execution import configure_execution_backend

RUNTIME_BOUNDARY_NAMES = {
    "forward",
    "denoise_step",
    "graph_encoder",
    "spectrum_encoder",
}


class _HookedModel(paddle.nn.Layer):
    """Small model that implements the shared compiled-runtime protocol."""

    def __init__(self):
        super().__init__()
        self.linear = paddle.nn.Linear(1, 1)
        self.execution_backend = "eager"
        self.forward_calls = 0
        self.predict_calls = []
        self.runtime_options = {}
        self.validation_calls = []

    def set_execution_backend(self, backend):
        if backend not in {"eager", "cinn"}:
            raise ValueError(backend)
        self.execution_backend = backend

    def set_runtime_options(self, runtime_options):
        self.runtime_options = runtime_options

    def validate_execution_backend(self, *, use_amp=False, world_size=1):
        if self.execution_backend != "cinn":
            raise AssertionError("validation should only be used for CINN")
        self.validation_calls.append((use_amp, world_size))

    def forward(self, batch, return_loss=True, return_prediction=True):
        self.forward_calls += 1
        prediction = self.linear(batch["x"])
        output = {"loss_dict": {}, "pred_dict": {"value": prediction}}
        if return_loss:
            output["loss_dict"]["loss"] = paddle.nn.functional.mse_loss(
                prediction, batch["y"]
            )
        if not return_prediction:
            output["pred_dict"] = {}
        return output

    def predict(self, data):
        self.predict_calls.append(data)
        return {"value": self.linear(data["x"]).numpy()[0, 0]}


def test_model_runtime_boundaries_use_stable_names():
    model_root = Path(models.__file__).parent
    runtime_path = Path(runtime_module.__file__).resolve()
    seen_names = set()

    for path in model_root.rglob("*.py"):
        if path.resolve() == runtime_path:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            is_decorator = (
                isinstance(node.func, ast.Name) and node.func.id == "runtime_boundary"
            )
            is_dispatch = (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "_run_runtime"
            )
            if not (is_decorator or is_dispatch):
                continue

            assert node.args, f"Missing runtime boundary name at {path}:{node.lineno}"
            name_node = node.args[0]
            assert isinstance(name_node, ast.Constant) and isinstance(
                name_node.value, str
            ), f"Runtime boundary name must be a string literal at {path}:{node.lineno}"
            name = name_node.value
            assert name in RUNTIME_BOUNDARY_NAMES, (
                f"Unsupported runtime boundary name {name!r} "
                f"at {path}:{node.lineno}"
            )
            seen_names.add(name)

    assert seen_names == RUNTIME_BOUNDARY_NAMES


def _trainer_config(output_dir):
    return {
        "max_epochs": 1,
        "output_dir": str(output_dir),
        "save_freq": 1,
        "log_freq": 100,
        "start_eval_epoch": 1,
        "eval_freq": 1,
        "seed": 2026,
        "compute_metric_during_train": False,
        "use_amp": False,
        "eval_with_no_grad": True,
        "gradient_accumulation_steps": 1,
    }


def _loader():
    samples = [
        {"x": paddle.to_tensor([[1.0]]), "y": paddle.to_tensor([[2.0]])},
        {"x": paddle.to_tensor([[2.0]]), "y": paddle.to_tensor([[4.0]])},
    ]
    return paddle.io.DataLoader(samples, batch_size=2, shuffle=False, return_list=True)


def test_models_accepting_a_backend_implement_the_protocol():
    """A model may stay eager-only; if it advertises the knob it must honor it.

    Requiring the mixin everywhere would push CINN plumbing into models that
    are never compiled (child denoisers, models whose graph does not compile).
    The invariant that matters is the other direction: exposing
    ``execution_backend`` means the workflow hooks are really there.
    """

    model_classes = [
        getattr(models, name)
        for name in models.__all__
        if inspect.isclass(getattr(models, name))
        and issubclass(getattr(models, name), paddle.nn.Layer)
    ]
    assert model_classes

    for cls in model_classes:
        accepts_backend = (
            "execution_backend" in inspect.signature(cls.__init__).parameters
        )
        if not accepts_backend:
            continue
        assert issubclass(cls, RuntimeMixin), cls.__name__
        for hook in ("set_execution_backend", "validate_execution_backend"):
            assert callable(getattr(cls, hook, None)), f"{cls.__name__}.{hook}"


@pytest.mark.parametrize(
    ("model_name", "method_name"),
    [
        ("CHGNet", "_runtime_forward"),
        ("M3GNet", "_runtime_forward"),
        ("SphereNet", "_runtime_forward"),
        ("DiffCSP", "_runtime_decode"),
        ("MatterGen", "_runtime_denoise"),
        ("MatterGenWithCondition", "_runtime_denoise"),
    ],
)
def test_split_runtime_boundaries_use_standard_private_names(model_name, method_name):
    method = getattr(getattr(models, model_name), method_name)

    # functools.wraps on @runtime_boundary exposes the eager implementation.
    assert callable(method)
    assert hasattr(method, "__wrapped__")


def test_execution_backend_protocol_is_generic_for_trainer_and_predictor(tmp_path):
    model = _HookedModel()
    optimizer = paddle.optimizer.Adam(learning_rate=1e-3, parameters=model.parameters())
    trainer = BaseTrainer(
        _trainer_config(tmp_path / "trainer"),
        model,
        train_dataloader=_loader(),
        val_dataloader=_loader(),
        optimizer=optimizer,
        execution_config={
            "backend": "cinn",
            "__init_params__": {"full_graph": False},
        },
    )

    trainer.train()

    assert model.execution_backend == "cinn"
    assert model.runtime_options == {"cinn": {"full_graph": False}}
    assert model.validation_calls == [(False, 1)]
    assert model.forward_calls > 0

    predictor = BasePredictor()
    predictor.model = _HookedModel()
    predictor.model.set_execution_backend("cinn")
    predictor.model.eval()
    predictor.execution_backend = "cinn"
    predictor.eval_with_no_grad = True
    predictor.post_transforms = None
    sample = {"x": paddle.to_tensor([[3.0]])}
    predictor._run_model(sample)
    predictor._run_model(sample)

    assert predictor.model.predict_calls == [sample, sample]


def test_eager_override_is_compatible_with_legacy_models():
    class LegacyModel:
        pass

    model = LegacyModel()
    assert configure_execution_backend(model, "eager", owner="Trainer") == "eager"

    class FixedCompiledModel:
        execution_backend = "cinn"

    with pytest.raises(ValueError, match="does not implement"):
        configure_execution_backend(FixedCompiledModel(), "eager", owner="Trainer")


def test_eager_override_ignores_compiled_runtime_options():
    model = _HookedModel()

    active = configure_execution_backend(
        model,
        "eager",
        init_params={"full_graph": False},
        owner="Predict",
    )

    assert active == "eager"
    assert model.runtime_options == {}
    assert model.validation_calls == []


def test_configure_execution_backend_forwards_workflow_context():
    model = _HookedModel()

    active = configure_execution_backend(
        model,
        "cinn",
        use_amp=True,
        world_size=4,
        owner="Trainer",
    )

    assert active == "cinn"
    assert model.validation_calls == [(True, 4)]


def test_backend_setter_must_honor_requested_backend():
    class MisreportingModel:
        execution_backend = "eager"

        def set_execution_backend(self, backend):
            del backend

    with pytest.raises(ValueError, match="model selected"):
        configure_execution_backend(MisreportingModel(), "cinn", owner="Predict")


def test_compiled_backend_requires_model_hooks():
    with pytest.raises(ValueError, match="does not implement"):
        configure_execution_backend(object(), "cinn", owner="Predictor")

    class UnvalidatedCompiledModel:
        execution_backend = "eager"

        def set_execution_backend(self, backend):
            self.execution_backend = backend

    with pytest.raises(ValueError, match="validated execution runtime"):
        configure_execution_backend(
            UnvalidatedCompiledModel(),
            "cinn",
            owner="Predictor",
        )


def test_runtime_is_validated_and_compiled_once_per_mode(monkeypatch):
    class RuntimeModel(RuntimeMixin, paddle.nn.Layer):
        def __init__(self):
            super().__init__()
            self.linear = paddle.nn.Linear(1, 1)
            self.validation_count = 0
            self._init_runtime("cinn")

        def validate_execution_backend(self, **kwargs):
            del kwargs
            self.validation_count += 1

    compile_modes = []

    def fake_compile(layer, **kwargs):
        del kwargs
        compile_modes.append("train" if model.training else "eval")

        class Runtime:
            training = True

            def __call__(self, *args, **kwargs):
                return layer(*args, **kwargs)

        return Runtime()

    monkeypatch.setattr(cinn_module, "compile_cinn", fake_compile)

    model = RuntimeModel()
    model.train()
    value = paddle.to_tensor([[3.0]], dtype="float32")
    model._run_runtime("forward", model.linear, value)
    train_runtime = model._runtime_cache[("cinn", "train", "forward")]
    model._run_runtime("forward", model.linear, value)
    assert model._runtime_cache[("cinn", "train", "forward")] is train_runtime

    model.eval()
    model._run_runtime("forward", model.linear, value)
    eval_runtime = model._runtime_cache[("cinn", "eval", "forward")]
    assert eval_runtime is not train_runtime

    assert model.validation_count == 2
    assert compile_modes == ["train", "eval"]

    keys = set(model.state_dict())
    state_dict = model.state_dict()
    state_dict["linear.weight"] = paddle.full_like(state_dict["linear.weight"], 2.0)
    state_dict["linear.bias"] = paddle.full_like(state_dict["linear.bias"], 1.0)
    model.set_state_dict(state_dict)

    assert set(model.state_dict()) == keys
    actual = train_runtime(value)
    np.testing.assert_allclose(actual.numpy(), [[7.0]])

    model.set_runtime_options({"cinn": {"full_graph": True}})
    assert model._runtime_cache == {}


def test_runtime_boundary_reuses_the_eager_model_method(monkeypatch):
    class BoundaryModel(RuntimeMixin, paddle.nn.Layer):
        def __init__(self):
            super().__init__()
            self._init_runtime("cinn")

        def validate_execution_backend(self, **kwargs):
            del kwargs

        @runtime_boundary("forward")
        def forward(self, data):
            return data["x"] * 2

    compiled_functions = []

    def fake_compile(function, **kwargs):
        del kwargs
        compiled_functions.append(function)
        return function

    monkeypatch.setattr(cinn_module, "compile_cinn", fake_compile)
    model = BoundaryModel()
    value = paddle.to_tensor([3.0])

    np.testing.assert_allclose(model({"x": value}).numpy(), [6.0])
    np.testing.assert_allclose(model({"x": value}).numpy(), [6.0])
    assert len(compiled_functions) == 1
    assert list(model._runtime_cache) == [("cinn", "train", "forward")]


def test_registered_backend_runs_without_model_changes(monkeypatch):
    class FakeBackend:
        name = "fake"

        def normalize_options(self, options):
            normalized = dict(options or {})
            return {"scale": normalized.get("scale", 1)}

        def validate(self, model, *, use_amp=False, world_size=1):
            del model, use_amp, world_size

        def compile(self, function, *, options):
            return lambda value: function(value) * options["scale"]

    monkeypatch.setattr(
        runtime_module,
        "_RUNTIME_BACKENDS",
        dict(runtime_module._RUNTIME_BACKENDS),
    )
    register_runtime_backend(FakeBackend())

    class RuntimeModel(RuntimeMixin, paddle.nn.Layer):
        def __init__(self):
            super().__init__()
            self._init_runtime(
                "fake",
                {"fake": {"scale": 3}},
            )

    model = RuntimeModel()
    value = paddle.to_tensor([2.0])
    actual = model._run_runtime("forward", lambda data: data + 1, value)

    np.testing.assert_allclose(actual.numpy(), [9.0])
    assert list(model._runtime_cache) == [("fake", "train", "forward")]

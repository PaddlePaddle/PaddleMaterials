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

# Adapted from https://github.com/divelab/AIRS (OpenMat/HIENet)

"""CINN execution-backend workflow tests for HIENet (CPU orchestration).

The real model with the released weights runs in float64: on CPU the fp32
kernel path yields NaN force and stress for this model, while float64 runs
clean, so float64 is the CPU truth form here. GPU compiled-execution parity
is gated separately.

Two workflow-specific patches:
  - the harness Trainer config is patched to eval_with_no_grad=False
    (force/stress are paddle.grad outputs of the energy);
  - PotentialPredictor construction is patched around the registry
    builders: the model gets the float64 one-hot patch applied after the
    registry build, and the graph converter is wrapped to accept pymatgen
    structures (the HIENet converter speaks ase Atoms).
"""

from __future__ import annotations

import copy
import os
import types
from pathlib import Path

import numpy as np
import paddle
import pytest
from ase.io import read
from cinn_workflow_harness import WorkflowCase
from cinn_workflow_harness import assert_gpu_cinn_matches_eager
from cinn_workflow_harness import assert_predictor_matches_checkpoint
from cinn_workflow_harness import assert_resume_parity
from cinn_workflow_harness import requires_gpu_cinn

import ppmat.models.hienet.model_build as _model_build
from ppmat.datasets.collate_fn import DefaultCollator
from ppmat.datasets.custom_data_type import ConcatNumpyWarper
from ppmat.models.hienet.hienet import HIENet
from ppmat.models.hienet.hienet_graph_converter import HIENetGraphConverter
from ppmat.models.hienet.nn.node_embedding import OnehotEmbedding
from ppmat.predictor import PotentialPredictor

ALL_PROPERTIES = ("energy", "force", "stress")

REPO_ROOT = Path(__file__).resolve().parents[1]
YAML = (
    REPO_ROOT / "interatomic_potentials" / "configs" / "hienet" / "hienet_v3.yaml"
)
# The released checkpoint and the pseudo-labeled frames live outside this
# repository (see interatomic_potentials/configs/hienet/README.md);
# HIENET_TEST_WORKSPACE points at the workspace holding them.
_WORKSPACE = Path(os.environ.get("HIENET_TEST_WORKSPACE", "/home/lkyu/baidu/HIENet"))
WEIGHTS = _WORKSPACE / "HIENet_paddle" / "checkpoints" / "HIENet-V3.pdparams"
DATA = _WORKSPACE / "data" / "pseudo_labeled.extxyz"

if not (WEIGHTS.is_file() and DATA.is_file()):
    pytest.skip(
        "Released HIENet weights and pseudo-labeled frames are required for these "
        "workflow tests; set HIENET_TEST_WORKSPACE to the workspace holding "
        "them (see interatomic_potentials/configs/hienet/README.md).",
        allow_module_level=True,
    )

# float64 default dtype, session-scoped so the default is restored for any
# other modules collected in the same run: the fp32 CPU kernel path yields
# NaN force/stress for this model.
@pytest.fixture(autouse=True, scope="session")
def float64_default_dtype():
    original = paddle.get_default_dtype()
    paddle.set_default_dtype("float64")
    yield
    paddle.set_default_dtype(original)


# ---------------------------------------------------------------------------
# float64 build hook (idempotent): the checkpoint stores float32 weights, so
# every backbone parameter/buffer is cast on creation.
# ---------------------------------------------------------------------------
_ORIG_BUILD = _model_build.build_E3_equivariant_model


def _cast_fp64(model):
    seen = set()
    for layer in [model] + list(model.sublayers()):
        if id(layer) in seen:
            continue
        seen.add(id(layer))
        for name, p in list(layer._parameters.items()):
            if p is not None and p.dtype == paddle.float32:
                layer._parameters[name] = paddle.create_parameter(
                    shape=p.shape,
                    dtype=paddle.float64,
                    default_initializer=paddle.nn.initializer.Assign(
                        p.cast(paddle.float64)
                    ),
                )
        for name, b in list(layer._buffers.items()):
            if b is not None and str(b.dtype) == "paddle.float32":
                layer._buffers[name] = b.cast(paddle.float64)
    return model


if not getattr(_ORIG_BUILD, "_hienet_fp64_cast", False):

    def _build_fp64(cfg, parallel=False):
        return _cast_fp64(_ORIG_BUILD(cfg, parallel=parallel))

    _build_fp64._hienet_fp64_cast = True
    _model_build.build_E3_equivariant_model = _build_fp64

_CKPT = paddle.load(str(WEIGHTS))
_CONFIG = dict(_CKPT["config"])
_STATE64 = {k: v.astype("float64") for k, v in _CKPT["model_state_dict"].items()}


def _patch_onehot_fp64(module):
    """One-hot lookup in float64.

    Reads every attribute from ``self`` at call time: default-argument
    closures over ``module`` break dy2static's AST re-exec of the patched
    forward (NameError on the closure cell in the transformed module).
    """

    def forward(self, data):
        inp = data[self.key_x]
        embd = paddle.nn.functional.one_hot(inp, self.num_classes).to(paddle.float64)
        data[self.key_x] = embd
        data[self.key_additional_output] = embd
        data[self.key_save] = inp
        return data

    module.forward = types.MethodType(forward, module)


# ---------------------------------------------------------------------------
# Model / data factories
# ---------------------------------------------------------------------------
def _make_model(execution_backend="cinn"):
    """Real backbone with the released weights, float64, dropout-free
    (deterministic train path)."""

    with paddle.utils.unique_name.guard():
        config = copy.deepcopy(_CONFIG)
        config["dropout"] = 0.0
        config["dropout_attn"] = 0.0
        model = HIENet(
            config,
            execution_backend=execution_backend,
            # SOT (full_graph=False): the AST tier turns every value inside
            # the compiled function into a PIR Value, which MessagePassing's
            # isinstance gate rejects.
            runtime_options={"cinn": {"full_graph": False}},
        )
        missing, unexpected = model.backbone.load_state_dict(_STATE64, strict=False)
        assert not missing and not unexpected, (missing, unexpected)
    for _, child in model.backbone.named_children():
        if isinstance(child, OnehotEmbedding):
            _patch_onehot_fp64(child)
    return model


_FRAMES = read(str(DATA), index=":4")
# Smallest fully-labeled pair (5 and 4 atoms) keeps the CPU runtime bounded.
_LABELED = [_FRAMES[1], _FRAMES[3]]
_CONVERTER = HIENetGraphConverter(cutoff=_CONFIG["cutoff"])
_LABELED_GRAPHS = [_CONVERTER(atoms) for atoms in _LABELED]


def _make_graph():
    atoms = _LABELED[0].copy()
    atoms.calc = None  # prediction inputs are unlabeled
    return _CONVERTER(atoms)


def _make_samples():
    samples = []
    for i, data in enumerate(_LABELED_GRAPHS):
        samples.append(
            {
                "graph": data,
                "id": i,
                "energy": data["energy"][0],
                "force": ConcatNumpyWarper(data["force"]),
                "stress": data["stress"],
            }
        )
    return samples


def _make_loader():
    return paddle.io.DataLoader(
        _make_samples(),
        batch_size=2,
        shuffle=False,
        collate_fn=DefaultCollator(),
        return_list=True,
    )


# ---------------------------------------------------------------------------
# Boundary conservation
# ---------------------------------------------------------------------------
def test_hienet_has_exactly_one_boundary(monkeypatch):
    model = _make_model(execution_backend="cinn")
    model.eval()
    boundary_names = []

    def run_boundary(self, name, function, *args, **kwargs):
        boundary_names.append(name)
        return function(*args, **kwargs)

    monkeypatch.setattr(HIENet, "_run_runtime", run_boundary)
    prediction = model.predict(_make_graph())

    assert boundary_names == ["forward"]
    assert prediction.keys() == set(ALL_PROPERTIES)
    assert np.asarray(prediction["force"]).shape == (5, 3)
    assert np.asarray(prediction["stress"]).shape == (1, 6)
    assert all(np.isfinite(v).all() for v in prediction.values())


def test_dispatch_does_not_move_eager_numbers(monkeypatch):
    """The boundary must be a pass-through when it runs the eager callable.

    Same weights and same graph, once through the plain eager path and once
    through the dispatcher: bit-for-bit agreement.
    """

    eager_model = _make_model(execution_backend="eager")
    state = eager_model.state_dict()
    eager_model.eval()
    eager_prediction = eager_model.predict(_make_graph())

    dispatched_model = _make_model(execution_backend="cinn")
    dispatched_model.set_state_dict(state)
    dispatched_model.eval()
    monkeypatch.setattr(
        HIENet,
        "_run_runtime",
        lambda self, name, function, *args, **kwargs: function(*args, **kwargs),
    )
    dispatched_prediction = dispatched_model.predict(_make_graph())

    assert dispatched_prediction.keys() == eager_prediction.keys()
    for key in eager_prediction:
        np.testing.assert_allclose(
            np.asarray(dispatched_prediction[key]),
            np.asarray(eager_prediction[key]),
            atol=0,
            rtol=0,
        )


def test_state_dict_keys_are_unchanged_by_the_backend():
    assert (
        _make_model(execution_backend="cinn").state_dict().keys()
        == _make_model(execution_backend="eager").state_dict().keys()
    )


# ---------------------------------------------------------------------------
# Workflow orchestration
# ---------------------------------------------------------------------------
@pytest.fixture
def fp64_predictor_builders(monkeypatch):
    """Adapt PotentialPredictor construction to this module's needs.

    The model is built through the ppmat registry reflection, then gets the
    float64 one-hot patch (the session runs in float64); the converter is
    wrapped to accept pymatgen structures, which the ppmat converters get
    for free and the HIENet converter (ase Atoms) does not.
    """

    import ppmat.predictor.base as predictor_base
    from ppmat.models import build_model as registry_build_model

    def fp64_build_model(cfg, vocab=None, **kwargs):
        model = registry_build_model(cfg, vocab=vocab, **kwargs)
        for _, child in model.backbone.named_children():
            if isinstance(child, OnehotEmbedding):
                _patch_onehot_fp64(child)
        return model

    def fp64_build_graph_converter(cfg, vocab=None):
        from pymatgen.io.ase import AseAtomsAdaptor

        sub = dict(cfg)
        cls_name = sub.pop("__class_name__")
        params = sub.pop("__init_params__", {})
        assert cls_name == "HIENetGraphConverter"
        inner = HIENetGraphConverter(**params)

        def convert(structures):
            single = not isinstance(structures, list)
            items = [structures] if single else structures
            graphs = [inner(AseAtomsAdaptor.get_atoms(s)) for s in items]
            return graphs[0] if single else graphs

        return convert

    monkeypatch.setattr(predictor_base, "build_model", fp64_build_model)
    monkeypatch.setattr(
        predictor_base, "build_graph_converter", fp64_build_graph_converter
    )


@pytest.fixture
def tensor_runtime_proxy(monkeypatch, fp64_predictor_builders):
    """Exercise public workflow hooks without a GPU compiler."""

    monkeypatch.setattr(
        HIENet, "validate_execution_backend", lambda self, **kwargs: None
    )
    monkeypatch.setattr(
        HIENet,
        "_run_runtime",
        lambda self, name, layer, *args, **kwargs: layer(*args, **kwargs),
    )


@pytest.fixture(autouse=True)
def potential_eval_with_grad(monkeypatch):
    """Force/stress are paddle.grad outputs, so eval must not use no_grad."""

    import cinn_workflow_harness as harness

    original = harness.trainer_config

    def trainer_config(output_dir, max_epochs):
        config = original(output_dir, max_epochs)
        config["eval_with_no_grad"] = False
        return config

    monkeypatch.setattr(harness, "trainer_config", trainer_config)


def _model_init_params():
    from omegaconf import OmegaConf

    yaml = OmegaConf.to_container(OmegaConf.load(str(YAML)), resolve=True)
    params = dict(yaml["Model"]["__init_params__"])
    params.pop("pretrained_model_path", None)
    params["config"] = dict(params["config"])
    params["config"]["dropout"] = 0.0
    params["config"]["dropout_attn"] = 0.0
    params["runtime_options"] = {"cinn": {"full_graph": False}}
    return params


def _predictor_config(checkpoint_path, execution_backend="cinn"):
    from omegaconf import OmegaConf

    params = _model_init_params()
    params["execution_backend"] = execution_backend
    predict_cfg = OmegaConf.to_container(OmegaConf.load(str(YAML)), resolve=True)[
        "Predict"
    ]
    predict_cfg["checkpoint_path"] = str(checkpoint_path)
    return {
        "Model": {"__class_name__": "HIENet", "__init_params__": params},
        "Predict": predict_cfg,
    }


def _example_structure():
    from pymatgen.io.ase import AseAtomsAdaptor

    atoms = _LABELED[0].copy()
    atoms.calc = None
    return AseAtomsAdaptor.get_structure(atoms)


def _predict_one(predictor):
    return predictor.from_structures([_example_structure()])


def _predict_batch(predictor):
    structure = _example_structure()
    return predictor.from_structures([structure, structure])


HIENET_CASE = WorkflowCase(
    name="hienet",
    model_cls=HIENet,
    make_model=_make_model,
    make_loader=_make_loader,
    predictor_config=_predictor_config,
    predictor_cls=PotentialPredictor,
    property_names=set(ALL_PROPERTIES),
    # Measured on a CUDA+CINN build (float64 SOT): weights 1.8e-11,
    # optimizer 3.9e-15, predictor outputs <= 2.8e-15; one decade of headroom.
    gpu_atol=1e-10,
    predict_one=_predict_one,
    predict_batch=_predict_batch,
)


def test_trainer_checkpoint_resume_matches_uninterrupted_training(
    tmp_path, tensor_runtime_proxy
):
    assert_resume_parity(HIENET_CASE, tmp_path)


def test_potential_predictor_loads_checkpoint_and_predicts(
    tmp_path, tensor_runtime_proxy
):
    assert_predictor_matches_checkpoint(HIENET_CASE, tmp_path)


def test_runtime_cache_key_is_backend_mode_boundary(monkeypatch):
    """With compilation stubbed out, the cache key shape is still exact."""

    import ppmat.models.common.cinn as cinn_module

    monkeypatch.setattr(
        HIENet, "validate_execution_backend", lambda self, **kwargs: None
    )
    monkeypatch.setattr(
        cinn_module, "compile_cinn", lambda function, **kwargs: function
    )
    model = _make_model(execution_backend="cinn")
    model.eval()
    model.predict(_make_graph())

    assert set(model._runtime_cache) == {("cinn", "eval", "forward")}


@requires_gpu_cinn
def test_gpu_cinn_trainer_and_predictor_match_eager(tmp_path, fp64_predictor_builders):
    assert_gpu_cinn_matches_eager(HIENET_CASE, tmp_path)

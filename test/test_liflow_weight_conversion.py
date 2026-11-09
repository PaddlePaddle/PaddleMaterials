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

"""Tests for strict checkpoint validation and torch -> paddle weight conversion."""

from __future__ import annotations

import numpy as np
import paddle
import pytest

from molecular_dynamics_integrator.liflow.convert_weights import convert_state
from ppmat.models.liflow.dual_painn import DualPaiNN
from ppmat.utils import save_load

MODEL_CFG = dict(
    num_features=8,
    num_radial_basis=4,
    num_layers=2,
    num_elements=8,
    r_max=5.0,
    r_offset=0.0,
    ref_temp=1000.0,
)


@pytest.fixture(scope="function")
def model():
    return DualPaiNN(**MODEL_CFG)


@pytest.fixture(scope="function")
def model_state_dict(model):
    return {name: np.asarray(v.numpy()) for name, v in model.state_dict().items()}


@pytest.fixture(scope="function")
def torch_state(model_state_dict):
    """Torch-style state: Linear weights stored ``[out, in]`` and a ``model.``
    prefix.  ``*.weight`` other than ``atom_embedding.weight`` is transposed (as
    torch does); embedding weights and scalar buffers are left as-is.
    """
    state = {}
    for name, arr in model_state_dict.items():
        prefixed = f"model.{name}"
        if (
            arr.ndim == 2
            and name.endswith(".weight")
            and not name.startswith("atom_embedding")
        ):
            state[prefixed] = arr.T.copy()
        else:
            state[prefixed] = arr.copy()
    return state


def test_load_pretrain_strict_rejects_missing_key(model, tmp_path):
    # build a checkpoint missing one parameter
    full = {name: v.numpy() for name, v in model.state_dict().items()}
    del full["updates.1.mlp_a.2.weight"]
    ckpt = tmp_path / "incomplete"
    paddle.save(full, f"{ckpt}.pdparams")
    with pytest.raises(ValueError, match="missing"):
        save_load.load_pretrain(model, str(ckpt), strict=True)


def test_load_pretrain_default_keeps_legacy_warning_behavior(model, tmp_path):
    full = {name: v.numpy() for name, v in model.state_dict().items()}
    del full["updates.1.mlp_a.2.weight"]
    ckpt = tmp_path / "incomplete"
    paddle.save(full, f"{ckpt}.pdparams")
    # default strict=False must not raise
    save_load.load_pretrain(model, str(ckpt))


def test_convert_state_matches_key_set_and_shapes(model, torch_state, model_state_dict):
    converted = convert_state(torch_state, model_state_dict)
    assert set(converted) == set(model_state_dict)
    for name, expected in model_state_dict.items():
        assert converted[name]["array"].shape == tuple(np.asarray(expected).shape)
        assert np.allclose(converted[name]["array"], np.asarray(expected), atol=1e-6)
    # frame linear weights are transposed by shape; embedding weights are not
    assert converted["messages.0.linear_W.weight"]["transposed"] is True
    assert converted["atom_embedding.weight"]["transposed"] is False


def test_convert_state_rejects_missing_key(model, model_state_dict):
    broken = {
        k: v for k, v in model_state_dict.items() if k != "messages.0.linear_W.weight"
    }
    with pytest.raises(ValueError, match="missing"):
        convert_state(broken, model_state_dict)


def test_convert_state_rejects_shape_mismatch(model, model_state_dict):
    broken = dict(model_state_dict)
    key = "messages.0.linear_W.weight"
    broken[key] = np.zeros((1, 1), dtype=np.float32)  # wrong shape
    with pytest.raises(ValueError, match="shape"):
        convert_state(broken, model_state_dict)


def test_converted_state_loads_strictly(model, tmp_path):
    state = {name: v.numpy() for name, v in model.state_dict().items()}
    ckpt = tmp_path / "converted"
    paddle.save(state, f"{ckpt}.pdparams")
    # full key set + matching shapes -> strict load succeeds (no raise)
    save_load.load_pretrain(model, str(ckpt), strict=True)


def test_build_model_from_name_forwards_strict_weights(monkeypatch, tmp_path, model):
    import ppmat.models as models_mod
    from ppmat.models import build_model_from_name

    # Fake package dir with a harmless YAML and a weights placeholder.
    pkg_dir = tmp_path / "liflow_universal_propagator"
    ckpt_dir = pkg_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True)
    yaml_path = pkg_dir / "liflow_universal_propagator.yaml"
    yaml_path.write_text("Model:\n  __class_name__: x\n", encoding="utf-8")
    (ckpt_dir / "best.pdparams").write_bytes(b"placeholder")

    calls = []

    def fake_load_pretrain(model, path, weights_name=None, strict=False):
        calls.append(strict)

    monkeypatch.setattr(save_load, "load_pretrain", fake_load_pretrain)
    registry = dict(models_mod.MODEL_REGISTRY)
    registry["liflow_universal_propagator"] = "local://placeholder"
    monkeypatch.setattr(models_mod, "MODEL_REGISTRY", registry)
    monkeypatch.setattr(
        models_mod.download, "get_weights_path_from_url", lambda n: str(pkg_dir)
    )
    monkeypatch.setattr(
        models_mod, "resolve_model_package_dir", lambda n, p: str(pkg_dir)
    )
    monkeypatch.setattr(
        models_mod, "get_model_config_path", lambda n, p: str(yaml_path)
    )
    # build_model returns the DualPaiNN fixture so the load step sees a real,
    # non-None layer; build_vocab(None) already returns None.
    monkeypatch.setattr(
        models_mod,
        "build_model",
        lambda cfg, vocab=None: model,
    )

    build_model_from_name("liflow_universal_propagator", strict_weights=True)
    assert calls == [True]

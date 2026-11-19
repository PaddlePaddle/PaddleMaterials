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

"""Tests for the ``strict`` mode of the public pretrained-weight loader.

``save_load.load_pretrain(..., strict=True)`` must reject missing/unexpected
keys and shape mismatches, while ``strict=False`` (the default) preserves the
legacy warn-and-load behavior.
"""

from __future__ import annotations

import os

import paddle
import pytest

from ppmat.utils import save_load


def _make_model():
    return paddle.nn.Linear(4, 8)


@pytest.fixture(scope="function")
def model():
    return _make_model()


@pytest.fixture(scope="function")
def full_checkpoint(model, tmp_path):
    state = {name: v.numpy() for name, v in model.state_dict().items()}
    path = str(tmp_path / "model")
    paddle.save(state, f"{path}.pdparams")
    return path


def test_load_pretrain_full_checkpoint_strict(model, full_checkpoint):
    save_load.load_pretrain(model, full_checkpoint, strict=True)


def test_load_pretrain_strict_rejects_missing_key(model, full_checkpoint):
    state = paddle.load(f"{full_checkpoint}.pdparams")
    reduced = {k: v for k, v in state.items() if k != "weight"}
    missing_path = os.path.join(os.path.dirname(full_checkpoint), "missing")
    paddle.save(reduced, f"{missing_path}.pdparams")
    with pytest.raises(ValueError, match="missing"):
        save_load.load_pretrain(model, missing_path, strict=True)


def test_load_pretrain_strict_rejects_unexpected_key(model, full_checkpoint):
    state = paddle.load(f"{full_checkpoint}.pdparams")
    state["extra.key"] = state["weight"]
    extra_path = os.path.join(os.path.dirname(full_checkpoint), "extra")
    paddle.save(state, f"{extra_path}.pdparams")
    with pytest.raises(ValueError, match="unexpected"):
        save_load.load_pretrain(model, extra_path, strict=True)


def test_load_pretrain_strict_rejects_shape_mismatch(model, full_checkpoint):
    state = paddle.load(f"{full_checkpoint}.pdparams")
    state["weight"] = paddle.zeros([2, 3], dtype=paddle.float32)
    wrong_path = os.path.join(os.path.dirname(full_checkpoint), "wrong_shape")
    paddle.save(state, f"{wrong_path}.pdparams")
    with pytest.raises(ValueError, match="shape"):
        save_load.load_pretrain(model, wrong_path, strict=True)


def test_load_pretrain_default_keeps_legacy_warning_behavior(model, full_checkpoint):
    state = paddle.load(f"{full_checkpoint}.pdparams")
    reduced = {k: v for k, v in state.items() if k != "weight"}
    legacy_path = os.path.join(os.path.dirname(full_checkpoint), "legacy")
    paddle.save(reduced, f"{legacy_path}.pdparams")
    # default strict=False must not raise (legacy warn-only path)
    save_load.load_pretrain(model, legacy_path)

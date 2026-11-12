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

"""Dynamic-shape compilation of the potential boundaries.

Every MD step moves atoms, so the edge and triplet counts that enter a force
model's boundary change while the atom count does not. Without dynamic axes the
AST capture traces, and CINN compiles, a new program for every new count. These
tests pin the contract of ``runtime_boundary(..., dynamic_dims=...)``: changing
a declared axis reuses one program and keeps the eager numbers, changing any
other axis or a non-tensor argument builds a new program, empty triplet sets
match eager, and ``dynamic_shape=False`` restores one program per shape.

The capture runs on CPU (``to_static`` with ``backend=None``), which exercises
the same InputSpec and traced graph that CINN compiles; the last test repeats
the core check with CINN on a GPU.
"""

from __future__ import annotations

import numpy as np
import paddle
import pytest

import ppmat.models.common.cinn as cinn_module
from ppmat.models.common.runtime import DynamicShapeRuntime
from ppmat.models.common.runtime import RuntimeMixin
from ppmat.models.common.runtime import boundary_dynamic_dims
from ppmat.models.common.runtime import runtime_boundary


@pytest.fixture
def programs(monkeypatch):
    """Capture boundaries with AST on CPU and record every static program built."""

    from paddle.jit.dy2static import pir_partial_program

    built = []
    init = pir_partial_program.PartialProgramLayer.__init__

    def counting_init(self, *args, **kwargs):
        built.append(self)
        init(self, *args, **kwargs)

    def compile_on_cpu(function, *, full_graph=False, input_spec=None):
        return paddle.jit.to_static(
            function, input_spec=input_spec, backend=None, full_graph=full_graph
        )

    monkeypatch.setattr(
        pir_partial_program.PartialProgramLayer, "__init__", counting_init
    )
    monkeypatch.setattr(cinn_module, "compile_cinn", compile_on_cpu)
    monkeypatch.setattr(cinn_module.CINN_BACKEND, "validate", lambda *a, **k: None)
    return built


# ---------------------------------------------------------------- runtime contract
class _EdgeModel(RuntimeMixin, paddle.nn.Layer):
    """Minimal boundary with an edge axis, a static atom axis and a constant."""

    def __init__(self, **options):
        super().__init__()
        self.linear = paddle.nn.Linear(3, 2)
        self._init_runtime("cinn", {"cinn": {"full_graph": True, **options}})

    @runtime_boundary("forward", dynamic_dims={"edge_vec": 0})
    def _runtime_forward(self, pos, edge_vec, scale):
        return self.linear(edge_vec).sum(axis=0) * scale + pos.sum()


def _expected(model, pos, edge_vec, scale):
    return (model.linear(edge_vec).sum(axis=0) * scale + pos.sum()).numpy()


def _tensor(shape, seed):
    return paddle.to_tensor(np.random.RandomState(seed).rand(*shape).astype("float32"))


def test_dynamic_axis_reuses_one_program(programs):
    model = _EdgeModel()
    model.eval()
    pos = _tensor([4, 3], 0)
    for seed, num_edges in enumerate((3, 7, 5)):
        edge_vec = _tensor([num_edges, 3], seed + 1)
        actual = model._runtime_forward(pos, edge_vec, 2.0)
        np.testing.assert_allclose(
            actual.numpy(), _expected(model, pos, edge_vec, 2.0), rtol=1e-6
        )
    assert len(programs) == 1
    runtime = model._runtime_cache[("cinn", "eval", "forward")]
    assert isinstance(runtime, DynamicShapeRuntime)
    assert list(model._runtime_cache) == [("cinn", "eval", "forward")]


def test_static_axes_and_constants_stay_in_the_cache_key(programs):
    model = _EdgeModel()
    model.eval()
    edge_vec = _tensor([3, 3], 1)
    cases = [
        (_tensor([4, 3], 0), 2.0),
        (_tensor([5, 3], 2), 2.0),
        (_tensor([5, 3], 2), 3.0),
    ]
    for pos, scale in cases:
        actual = model._runtime_forward(pos, edge_vec, scale)
        np.testing.assert_allclose(
            actual.numpy(), _expected(model, pos, edge_vec, scale), rtol=1e-6
        )
    # A new atom count and a new constant each need their own program.
    assert len(programs) == 3


def test_dynamic_shape_false_restores_one_program_per_shape(programs):
    model = _EdgeModel(dynamic_shape=False)
    model.eval()
    pos = _tensor([4, 3], 0)
    for seed, num_edges in enumerate((3, 7)):
        model._runtime_forward(pos, _tensor([num_edges, 3], seed + 1), 2.0)
    assert len(programs) == 2
    runtime = model._runtime_cache[("cinn", "eval", "forward")]
    assert not isinstance(runtime, DynamicShapeRuntime)


@pytest.mark.parametrize("mode", ["train", "sot"])
def test_training_and_sot_boundaries_get_no_input_spec(monkeypatch, mode):
    calls = []

    def fake_compile(function, **kwargs):
        calls.append(kwargs)
        return function

    monkeypatch.setattr(cinn_module, "compile_cinn", fake_compile)
    monkeypatch.setattr(cinn_module.CINN_BACKEND, "validate", lambda *a, **k: None)
    model = _EdgeModel(full_graph=mode != "sot")
    if mode == "train":
        model.train()
    else:
        model.eval()
    model._runtime_forward(_tensor([4, 3], 0), _tensor([3, 3], 1), 1.0)
    assert calls == [{"full_graph": mode != "sot"}]


def test_dynamic_dims_are_validated():
    with pytest.raises(ValueError, match="unknown arguments"):

        @runtime_boundary("forward", dynamic_dims={"missing": 0})
        def _missing(self, x):
            return x

    with pytest.raises(ValueError, match="non-negative axes"):
        runtime_boundary("forward", dynamic_dims={"x": -1})

    assert boundary_dynamic_dims(_EdgeModel._runtime_forward) == {"edge_vec": (0,)}


def test_cinn_dynamic_shape_option_is_normalized():
    backend = cinn_module.CINN_BACKEND
    assert backend.normalize_options(None) == {
        "full_graph": False,
        "dynamic_shape": True,
    }
    assert backend.normalize_options({"full_graph": True, "dynamic_shape": False}) == {
        "full_graph": True,
        "dynamic_shape": False,
    }
    with pytest.raises(TypeError, match="dynamic_shape"):
        backend.normalize_options({"dynamic_shape": "false"})
    assert backend.accepts_input_spec({"full_graph": True, "dynamic_shape": True})
    assert not backend.accepts_input_spec({"full_graph": False, "dynamic_shape": True})
    assert not backend.accepts_input_spec({"full_graph": True, "dynamic_shape": False})

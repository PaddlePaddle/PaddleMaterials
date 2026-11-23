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

"""Boundary contract for the two models whose callable changed on 2026-08-24.

SFIN dropped a shadow `nn.Layer` in favour of decorating `_forward`, and
M3GNet's boundary went from one packed dict to named tensor parameters. Both are
compiled with AST, where the boundary is the whole program, so these
tests pin the two things a rewrite there can break without failing loudly:
there is still exactly one boundary, and dispatching through it does not move
the eager numbers.
"""

from __future__ import annotations

import numpy as np
import paddle
from pymatgen.core import Lattice
from pymatgen.core import Structure

from ppmat.models.mattersim.m3gnet import M3GNet
from ppmat.models.mattersim.m3gnet_graph_converter import M3GNetGraphConvertor
from ppmat.models.sfin.sfin import SFIN


def _record_boundaries(monkeypatch, model_cls, names):
    def run_boundary(self, name, function, *args, **kwargs):
        names.append(name)
        return function(*args, **kwargs)

    monkeypatch.setattr(model_cls, "_run_runtime", run_boundary)


def _make_sfin(execution_backend="eager"):
    with paddle.utils.unique_name.guard():
        paddle.seed(2026)
        return SFIN(
            in_channels=1,
            base_channels=8,
            num_blocks=1,
            input_name="noisy",
            target_name="gt_enhance",
            execution_backend=execution_backend,
            runtime_options={"cinn": {"full_graph": True}},
        )


def _sfin_image():
    paddle.seed(7)
    return paddle.rand([1, 1, 16, 16], dtype="float32")


def test_sfin_has_exactly_one_boundary(monkeypatch):
    model = _make_sfin("cinn")
    model.eval()
    names = []
    _record_boundaries(monkeypatch, SFIN, names)

    model._forward(_sfin_image())

    assert names == ["forward"]
    assert model.get_runtime_options("cinn")["full_graph"] is True


def test_sfin_dispatch_does_not_move_eager_numbers(monkeypatch):
    image = _sfin_image()
    eager_model = _make_sfin("eager")
    eager_model.eval()
    expected = eager_model._forward(image)

    model = _make_sfin("cinn")
    model.set_state_dict(eager_model.state_dict())
    model.eval()
    monkeypatch.setattr(
        SFIN,
        "_run_runtime",
        lambda self, name, function, *args, **kwargs: function(*args, **kwargs),
    )

    np.testing.assert_allclose(
        model._forward(image).numpy(), expected.numpy(), atol=0, rtol=0
    )
    assert model.state_dict().keys() == eager_model.state_dict().keys()


def _make_m3gnet(execution_backend="eager"):
    with paddle.utils.unique_name.guard():
        paddle.seed(2026)
        return M3GNet(
            num_blocks=1,
            units=8,
            max_l=2,
            max_n=2,
            cutoff=5.0,
            max_z=94,
            threebody_cutoff=4.0,
            energy_key="energy",
            force_key="force",
            stress_key=None,
            loss_type="smooth_l1_loss",
            execution_backend=execution_backend,
            runtime_options={"cinn": {"full_graph": True}},
        )


def _m3gnet_batch():
    structure = Structure(
        Lattice.cubic(4.0), ["Li", "O"], [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]
    )
    return {"graph": M3GNetGraphConvertor()(structure)}


def test_m3gnet_has_exactly_one_boundary(monkeypatch):
    model = _make_m3gnet("cinn")
    model.eval()
    names = []
    _record_boundaries(monkeypatch, M3GNet, names)

    model._forward(_m3gnet_batch())

    # One boundary, not one per sublayer: the passthrough that wrapped every
    # embedding, basis, and conv call was removed on 2026-08-24.
    assert names == ["forward"]
    assert model.get_runtime_options("cinn")["full_graph"] is True


def test_m3gnet_dispatch_does_not_move_eager_numbers(monkeypatch):
    eager_model = _make_m3gnet("eager")
    eager_model.eval()
    expected = eager_model._forward(_m3gnet_batch())

    model = _make_m3gnet("cinn")
    model.set_state_dict(eager_model.state_dict())
    model.eval()
    monkeypatch.setattr(
        M3GNet,
        "_run_runtime",
        lambda self, name, function, *args, **kwargs: function(*args, **kwargs),
    )
    actual = model._forward(_m3gnet_batch())

    assert type(actual) is type(expected)
    for left, right in zip(_as_arrays(actual), _as_arrays(expected)):
        np.testing.assert_allclose(left, right, atol=0, rtol=0)
    assert model.state_dict().keys() == eager_model.state_dict().keys()


def _as_arrays(value):
    if isinstance(value, paddle.Tensor):
        return [value.numpy()]
    if isinstance(value, dict):
        return [array for item in value.values() for array in _as_arrays(item)]
    if isinstance(value, (list, tuple)):
        return [array for item in value for array in _as_arrays(item)]
    if value is None:
        return []
    return [np.asarray(value)]

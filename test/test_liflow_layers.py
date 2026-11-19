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

"""Layer-level contract tests for the LiFlow Paddle port."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import paddle
import pytest

from ppmat.models.liflow.layers import BesselBasis
from ppmat.models.liflow.layers import DualMessageBlock

REFERENCE_NPZ = (
    Path(__file__).resolve().parent / "fixtures" / "liflow" / "reference_layers.npz"
)


def _load_reference_state(layer, ref_state: dict[str, np.ndarray]):
    """Load a torch reference state dict, transposing Linear weights.

    torch stores ``nn.Linear.weight`` as ``[out, in]`` while Paddle stores it as
    ``[in, out]``; biases and buffers transfer as-is.
    """
    state = {}
    for name, arr in ref_state.items():
        if name.endswith(".weight") and arr.ndim == 2:
            arr = arr.T
        state[name] = paddle.to_tensor(arr.copy())
    missing, unexpected = layer.set_state_dict(state)
    assert not missing, f"missing keys: {missing}"
    assert not unexpected, f"unexpected keys: {unexpected}"
    return layer


def test_liflow_layer_state_dict_contract():
    layer = DualMessageBlock(num_features=8, num_radial_basis=4)
    assert set(layer.state_dict()) == {
        "mlp_phi.0.weight",
        "mlp_phi.0.bias",
        "mlp_phi.2.weight",
        "mlp_phi.2.bias",
        "linear_W.weight",
        "linear_W.bias",
    }


def test_bessel_basis_is_finite_away_from_zero():
    layer = BesselBasis(num_basis=4, r_max=5.0)
    output = layer(paddle.to_tensor([[0.5], [2.0]], dtype="float32"))
    assert output.shape == [2, 1, 4]
    assert bool(paddle.isfinite(output).all())


def test_update_block_state_dict_contract():
    from ppmat.models.liflow.layers import UpdateBlock

    layer = UpdateBlock(num_features=8)
    assert set(layer.state_dict()) == {
        "mlp_a.0.weight",
        "mlp_a.0.bias",
        "mlp_a.2.weight",
        "mlp_a.2.bias",
        "linear_UV.weight",
    }


def test_cosine_cutoff_respects_cutoff():
    from ppmat.models.liflow.layers import CosineCutoff

    layer = CosineCutoff(r_max=5.0)
    output = layer(paddle.to_tensor([[1.0], [6.0]], dtype="float32"))
    # Inside the cutoff the envelope is in (0, 1]; beyond it the value is 0.
    assert bool((output[0, 0] > 0.0).all())
    assert float(output[1, 0]) == pytest.approx(0.0)


def test_dual_message_block_shape_and_roundtrip():
    n_nodes, n_edges, f, r = 3, 4, 8, 4
    layer = DualMessageBlock(num_features=f, num_radial_basis=r)
    s = paddle.randn([n_nodes, 1, f])
    v = paddle.randn([n_nodes, 3, f])
    edge_index = paddle.to_tensor([[0, 1, 2, 0], [1, 2, 0, 2]], dtype="int64")
    radial = paddle.randn([n_edges, 1, r])
    f_cut = paddle.ones([n_edges, 1])
    units = paddle.nn.functional.normalize(paddle.randn([n_edges, 3]), axis=-1)
    s_out, v_out = layer(
        s,
        v,
        radial,
        radial,
        f_cut,
        f_cut,
        units,
        units,
        edge_index,
    )
    assert s_out.shape == [n_nodes, 1, f]
    assert v_out.shape == [n_nodes, 3, f]
    assert np.isfinite(s_out.numpy()).all()
    assert np.isfinite(v_out.numpy()).all()


@pytest.fixture(scope="module")
def reference_layers():
    data = np.load(REFERENCE_NPZ, allow_pickle=False)
    states = {}
    for key in data.files:
        if "/state/" in key:
            group, _, name = key.partition("/state/")
            states.setdefault(group, {})[name] = data[key]
    return data, states


def _assert_close(actual, expected, rtol=1e-5, atol=1e-6):
    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol)


def test_gaussian_fourier_basis_matches_reference(reference_layers):
    data, states = reference_layers
    from ppmat.models.liflow.layers import GaussianFourierBasis

    layer = GaussianFourierBasis(num_basis=int(data["gfb/num_basis"]))
    _load_reference_state(layer, states["gfb"])
    out = layer(paddle.to_tensor(data["gfb/x"])).numpy()
    _assert_close(out, data["gfb/output"])


def test_bessel_basis_matches_reference(reference_layers):
    data, states = reference_layers
    layer = BesselBasis(num_basis=4, r_max=5.0)
    _load_reference_state(layer, states["bessel"])
    out = layer(paddle.to_tensor(data["bessel/x"])).numpy()
    _assert_close(out, data["bessel/output"])


def test_cosine_cutoff_matches_reference(reference_layers):
    data, states = reference_layers
    from ppmat.models.liflow.layers import CosineCutoff

    layer = CosineCutoff(r_max=5.0)
    _load_reference_state(layer, states["cutoff"])
    out = layer(paddle.to_tensor(data["cutoff/x"])).numpy()
    _assert_close(out, data["cutoff/output"])


def test_dual_message_block_matches_reference(reference_layers):
    data, states = reference_layers
    layer = DualMessageBlock(
        num_features=int(data["dmb/num_features"]),
        num_radial_basis=int(data["dmb/num_radial_basis"]),
    )
    _load_reference_state(layer, states["dmb"])
    edge_index = paddle.to_tensor(data["dmb/edge_index"], dtype="int64")
    s_out, v_out = layer(
        paddle.to_tensor(data["dmb/s"]),
        paddle.to_tensor(data["dmb/v"]),
        paddle.to_tensor(data["dmb/radial"]),
        paddle.to_tensor(data["dmb/radial"]),
        paddle.to_tensor(data["dmb/f_cut"]),
        paddle.to_tensor(data["dmb/f_cut"]),
        paddle.to_tensor(data["dmb/units"]),
        paddle.to_tensor(data["dmb/units"]),
        edge_index,
    )
    _assert_close(s_out.numpy(), data["dmb/output_s"])
    _assert_close(v_out.numpy(), data["dmb/output_v"])


def test_update_block_matches_reference(reference_layers):
    data, states = reference_layers
    from ppmat.models.liflow.layers import UpdateBlock

    layer = UpdateBlock(num_features=int(data["ub/num_features"]))
    _load_reference_state(layer, states["ub"])
    s_out, v_out = layer(
        paddle.to_tensor(data["ub/s"]),
        paddle.to_tensor(data["ub/v"]),
    )
    _assert_close(s_out.numpy(), data["ub/output_s"])
    _assert_close(v_out.numpy(), data["ub/output_v"])


def test_gated_equivariant_block_matches_reference(reference_layers):
    data, states = reference_layers
    from ppmat.models.liflow.layers import GatedEquivariantBlock

    layer = GatedEquivariantBlock(
        num_scalar_inputs=int(data["geb/num_scalar_inputs"]),
        num_vector_inputs=int(data["geb/num_vector_inputs"]),
    )
    _load_reference_state(layer, states["geb"])
    out = layer(
        paddle.to_tensor(data["geb/s"]),
        paddle.to_tensor(data["geb/v"]),
    )
    _assert_close(out.numpy(), data["geb/output"])

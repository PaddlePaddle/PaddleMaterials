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

"""CHGNet's forward was restructured into one differentiable tensor graph.

These tests pin the parts that restructuring can silently break: which
derivatives are still reachable, what the boundary is, and that the eager
numbers do not move when the backend is switched on.
"""

from __future__ import annotations

import numpy as np
import paddle
import pytest
from pymatgen.core import Lattice
from pymatgen.core import Structure

from ppmat.models.chgnet.chgnet import CHGNet
from ppmat.models.chgnet.chgnet_graph_converter import CHGNetGraphConverter

ALL_PROPERTIES = ("energy_per_atom", "force", "stress", "magmom")


def _make_model(execution_backend="eager", property_names=ALL_PROPERTIES):
    with paddle.utils.unique_name.guard():
        paddle.seed(2026)
        return CHGNet(
            atom_fea_dim=8,
            bond_fea_dim=8,
            angle_fea_dim=8,
            composition_model=None,
            num_radial=5,
            num_angular=5,
            n_conv=2,
            atom_conv_hidden_dim=8,
            bond_conv_hidden_dim=8,
            angle_layer_hidden_dim=0,
            mlp_hidden_dims=(8,),
            property_names=property_names,
            execution_backend=execution_backend,
            runtime_options={"cinn": {"full_graph": True}},
        )


def _make_graph():
    structure = Structure(
        Lattice.cubic(4.0),
        ["Li", "O"],
        [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
    )
    return CHGNetGraphConverter(
        atom_graph_cutoff=6.0,
        bond_graph_cutoff=3.0,
        num_cpus=1,
    )(structure)


def test_cinn_uses_one_complete_forward_boundary(monkeypatch):
    model = _make_model(execution_backend="cinn")
    model.eval()
    boundary_names = []

    def run_boundary(self, name, function, *args, **kwargs):
        boundary_names.append(name)
        return function(*args, **kwargs)

    monkeypatch.setattr(CHGNet, "_run_runtime", run_boundary)
    prediction = model.predict(_make_graph())

    assert boundary_names == ["forward"]
    assert prediction.keys() == set(ALL_PROPERTIES)
    assert np.asarray(prediction["force"]).shape == (2, 3)
    assert np.asarray(prediction["stress"]).shape == (3, 3)


@pytest.mark.parametrize(
    "property_names",
    [
        ("energy_per_atom",),
        ("energy_per_atom", "force"),
        ("energy_per_atom", "stress"),
        ALL_PROPERTIES,
    ],
    ids=["energy", "energy_force", "energy_stress", "all"],
)
def test_each_requested_derivative_stays_reachable(property_names):
    """Forces need a differentiable frac_coords leaf, stress needs strains.

    A model configured for forces but not stress used to lose its only
    gradient path, so paddle.grad had nothing to differentiate through.
    """

    model = _make_model(property_names=property_names)
    model.eval()
    prediction = model.predict(_make_graph())

    assert prediction.keys() == set(property_names)
    if "force" in property_names:
        force = np.asarray(prediction["force"])
        assert force.shape == (2, 3)
        assert np.isfinite(force).all()
    if "stress" in property_names:
        stress = np.asarray(prediction["stress"])
        assert stress.shape == (3, 3)
        assert np.isfinite(stress).all()


def test_enabling_the_backend_does_not_move_eager_numbers(monkeypatch):
    """The boundary must be a pass-through when it runs the eager callable.

    This is the guard against the restructuring itself changing results: the
    same weights and the same graph, once through the plain eager path and once
    through the dispatcher, have to agree bit-for-bit.
    """

    eager_model = _make_model(execution_backend="eager")
    state = eager_model.state_dict()
    eager_model.eval()
    eager_prediction = eager_model.predict(_make_graph())

    dispatched_model = _make_model(execution_backend="cinn")
    dispatched_model.set_state_dict(state)
    dispatched_model.eval()
    monkeypatch.setattr(
        CHGNet,
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


def test_stress_is_the_same_whether_force_is_also_requested():
    """Requesting stress alone and alongside force must give the same stress.

    Force and stress differentiate one energy, so the force call has to retain
    the graph. Note this assertion does not reproduce the underlying defect on
    eager CPU — reading freed buffers happens to return the same values there.
    It held while compiled execution disagreed, so treat it as a cheap invariant,
    not as coverage for the compiled path; that needs the GPU requalification.
    """

    stress_only = _make_model(property_names=("energy_per_atom", "stress"))
    stress_only.eval()
    reference = np.asarray(stress_only.predict(_make_graph())["stress"])

    both = _make_model(property_names=("energy_per_atom", "force", "stress"))
    both.set_state_dict(stress_only.state_dict())
    both.eval()
    actual = np.asarray(both.predict(_make_graph())["stress"])

    np.testing.assert_allclose(actual, reference, atol=0, rtol=0)


def test_state_dict_keys_are_unchanged_by_the_backend():
    assert (
        _make_model(execution_backend="cinn").state_dict().keys()
        == _make_model(execution_backend="eager").state_dict().keys()
    )

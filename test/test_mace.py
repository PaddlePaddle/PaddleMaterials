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

import numpy as np
import paddle
import pytest
from pymatgen.core import Lattice
from pymatgen.core import Structure

from ppmat.models import build_model
from ppmat.models.common.graph_converter import FindPointsInSpheres


@pytest.fixture(scope="module")
def mace_model():
    paddle.set_device("cpu")
    paddle.seed(42)
    # A tiny MACE (two elements, 3 Bessel functions, hidden cut at l = 1) keeps
    # the test on CPU and independent of the released 89-element checkpoint.
    return build_model(
        {
            "__class_name__": "MACE",
            "__init_params__": {
                "num_elements": 2,
                "atomic_numbers": [1, 8],
                "num_bessel": 3,
                "max_ell": 1,
                "hidden_irreps": "4x0e+4x1o",
                "MLP_irreps": "4x0e",
                "correlation": 2,
                "radial_MLP": [4],
            },
        }
    )


@pytest.fixture(scope="module")
def structure():
    # H2O in a cubic cell: one oxygen and two hydrogens within the 6 Å cut-off.
    return Structure(
        Lattice.cubic(8.0),
        [8, 1, 1],
        [[0.4, 0.4, 0.4], [0.52, 0.4, 0.4], [0.37, 0.51, 0.4]],
    )


def _graph(structure):
    return FindPointsInSpheres(cutoff=6.0)(structure).tensor()


def test_mace_predict_returns_finite_energy_force_stress(mace_model, structure):
    mace_model.eval()
    pred = mace_model.predict(_graph(structure))
    for name, shape in [
        ("energy_per_atom", [1]),
        ("force", [3, 3]),
        ("stress", [1, 3, 3]),
    ]:
        assert list(pred[name].shape) == shape
        assert np.isfinite(pred[name].numpy()).all()


def test_mace_force_matches_energy_gradient(mace_model, structure):
    mace_model.eval()
    delta = 1e-3
    energies = []
    for sign in (-1, 1):
        displaced = structure.copy()
        displaced.translate_sites(1, [sign * delta, 0.0, 0.0], frac_coords=False)
        energy_per_atom = mace_model.predict(_graph(displaced))["energy_per_atom"]
        energies.append(float(energy_per_atom.numpy()[0]) * len(displaced))
    finite_difference_force = -(energies[1] - energies[0]) / (2 * delta)
    analytic_force = float(mace_model.predict(_graph(structure))["force"].numpy()[1, 0])
    np.testing.assert_allclose(
        analytic_force, finite_difference_force, rtol=0.03, atol=2e-3
    )


def test_mace_loss_backward_reaches_all_parameters(mace_model, structure):
    mace_model.train()
    result = mace_model(
        {
            "graph": _graph(structure),
            "energy_per_atom": paddle.zeros([1]),
            "force": paddle.zeros([3, 3]),
            "stress": paddle.zeros([1, 3, 3]),
        }
    )
    loss = result["loss_dict"]["loss"]
    assert np.isfinite(loss.numpy()).all()
    # Forces and stresses enter through a second derivative, which is the part
    # that needs the transpose/reshape/bmm einsum in e3nn.
    loss.backward()
    gradients = [p.grad for p in mace_model.parameters() if p.grad is not None]
    assert gradients
    assert all(np.isfinite(g.numpy()).all() for g in gradients)
    assert any(np.any(g.numpy() != 0) for g in gradients)
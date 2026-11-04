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

"""Future-free trajectory generation tests.

We inject deterministic zero-velocity backends so the tests exercise the
integrator bookkeeping (shape, solver, centroid correction, finite values)
without needing trained checkpoints.  The public ``run`` API must expose no
future endpoint reference.
"""

from __future__ import annotations

import inspect

import numpy as np
import paddle
import pytest

from ppmat.predictor import IntegratorPredictor


class _ZeroVelocityModel(paddle.nn.Layer):
    """A deterministic LiFlow-compatible ``predict`` returning zero velocity."""

    def __init__(self):
        super().__init__()

    @paddle.no_grad()
    def predict(self, batch_data):
        num_atoms = batch_data["flow_positions"].shape[0]
        return {"velocity": paddle.zeros([num_atoms, 3], dtype="float32")}


def make_predictor_with_deterministic_models(cutoff=5.0) -> IntegratorPredictor:
    # ``element_index`` is the training-time ``Z -> embedding`` table.
    element_index = np.zeros(100, dtype=np.int64)
    element_index[[1, 3, 6, 8]] = [0, 1, 2, 3]
    return IntegratorPredictor(
        propagator_model=_ZeroVelocityModel(),
        corrector_model=_ZeroVelocityModel(),
        cutoff=cutoff,
        periodic=True,
        element_index=element_index,
    )


@pytest.fixture()
def initial_sample():
    rng = np.random.default_rng(0)
    num_atoms = 4
    return {
        "positions": rng.normal(size=(num_atoms, 3)).astype(np.float32),
        "atomic_numbers": np.array([3, 8, 3, 8], dtype=np.int64),
        "lattice": np.eye(3, dtype=np.float32) * 10.0,
        "temperature": 800.0,
    }


def test_predictor_generates_trajectory_without_future_positions(initial_sample):
    predictor = make_predictor_with_deterministic_models()
    trajectory = predictor.run(**initial_sample, steps=3, flow_steps=2, seed=42)
    assert trajectory.shape == (4, initial_sample["positions"].shape[0], 3)
    assert trajectory.dtype == np.float32


@pytest.mark.parametrize("solver", ["euler", "heun"])
def test_solver_is_finite(solver, initial_sample):
    trajectory = make_predictor_with_deterministic_models().run(
        **initial_sample, steps=2, flow_steps=3, solver=solver, seed=42
    )
    assert np.isfinite(trajectory).all()


def test_predictor_api_has_no_future_argument():
    assert "end_positions" not in inspect.signature(IntegratorPredictor.run).parameters


def test_predictor_rejects_unknown_solver(initial_sample):
    predictor = make_predictor_with_deterministic_models()
    with pytest.raises(ValueError, match="solver"):
        predictor.run(**initial_sample, steps=1, flow_steps=2, solver="midpoint")


def test_predictor_centroid_is_preserved(initial_sample):
    predictor = make_predictor_with_deterministic_models()
    trajectory = predictor.run(**initial_sample, steps=2, flow_steps=2, seed=42)
    from ase import data as ase_data

    masses = ase_data.atomic_masses[initial_sample["atomic_numbers"]].astype(np.float64)
    # ``fix_com`` removes the COM *displacement* of each step, so the absolute
    # COM of the initial frame is preserved (the reference implementation does
    # not re-centre structures at the origin).
    reference = (initial_sample["positions"].astype(np.float64) * masses[:, None]).sum(
        axis=0
    ) / masses.sum()
    for frame in trajectory:
        centroid = (frame.astype(np.float64) * masses[:, None]).sum(
            axis=0
        ) / masses.sum()
        assert np.allclose(centroid, reference, atol=1e-3)


def test_predictor_requires_both_local_paths():
    with pytest.raises(ValueError, match="propagator"):
        IntegratorPredictor(
            propagator_config="propagator.yaml", propagator_checkpoint=None
        )

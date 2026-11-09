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

"""Unit tests for the LiFlow scientific evaluation functions."""

from __future__ import annotations

import numpy as np
import pytest

from molecular_dynamics_integrator.evaluate import calculate_average_rdf
from molecular_dynamics_integrator.evaluate import calculate_msd
from molecular_dynamics_integrator.evaluate import evaluate_trajectory


def test_msd_known_translation():
    trajectory = np.array([[[0.0, 0.0, 0.0]], [[1.0, 2.0, 2.0]]])
    assert calculate_msd(trajectory) == pytest.approx(9.0)


def test_rdf_output_shape(trajectory_fixture, lattice_fixture):
    distance, rdf = calculate_average_rdf(
        trajectory_fixture, lattice_fixture, rmax=5.0, nbins=50
    )
    assert distance.shape == (50,)
    assert rdf.shape == (50,)
    assert np.isfinite(rdf).all()


def test_msd_respects_atom_mask():
    traj = np.zeros((2, 3, 3))
    traj[1, 0] = [1.0, 0.0, 0.0]  # only atom 0 moves by 1
    assert calculate_msd(traj, np.array([True, False, False])) == pytest.approx(1.0)
    assert calculate_msd(traj, np.array([False, True, False])) == pytest.approx(0.0)


def test_evaluate_trajectory_matches_reference():
    atomic_numbers = np.array([3, 8, 3, 8])
    lattice = np.eye(3) * 10.0
    rng = np.random.default_rng(0)
    # the reference convention uses traj_ref = positions_ref[500::100], so the
    # reference must span more than 500 frames.
    reference = np.stack([rng.normal(scale=0.1, size=(4, 3)) for _ in range(600)])
    prediction = reference[:30] + 1e-4
    n_frames = prediction.shape[0]
    metrics = evaluate_trajectory(prediction, reference, atomic_numbers, lattice)
    assert set(metrics) == {
        "msd_Li",
        "msd_Li_ref",
        "msd_frame",
        "msd_frame_ref",
        "rdf_mae",
        "final_step",
    }
    assert metrics["final_step"] == n_frames - 1
    assert np.isfinite(metrics["rdf_mae"])


@pytest.fixture()
def trajectory_fixture():
    rng = np.random.default_rng(0)
    return rng.normal(size=(8, 4, 3))


@pytest.fixture()
def lattice_fixture():
    return np.eye(3) * 10.0

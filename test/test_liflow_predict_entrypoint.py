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

"""Tests for ``molecular_dynamics_integrator/predict.py``.

The end-to-end test runs the real entrypoint as a subprocess with a tiny local
checkpoint (no network), and verifies the produced trajectory, metadata and XYZ
file.  The unit tests cover the registry/duplicate-source resolution in
``_build_side``.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import paddle
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLE_INPUT = (
    REPO_ROOT
    / "molecular_dynamics_integrator"
    / "liflow"
    / "example_data"
    / "initial_structure.npz"
)

TINY_INIT_PARAMS = {
    "num_features": 8,
    "num_radial_basis": 4,
    "num_layers": 2,
    "num_elements": 18,
    "r_max": 5.0,
    "r_offset": 0.5,
    "ref_temp": 1000.0,
    "prediction_mode": "velocity",
}


def _load_predict_module():
    path = REPO_ROOT / "molecular_dynamics_integrator" / "predict.py"
    spec = importlib.util.spec_from_file_location("liflow_predict_entrypoint", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_tiny_checkpoint(directory: Path) -> Path:
    from omegaconf import OmegaConf

    from ppmat.models.liflow.liflow import LiFlow

    model = LiFlow(**TINY_INIT_PARAMS)
    checkpoint = directory / "best.pdparams"
    paddle.save(model.state_dict(), str(checkpoint))

    config = {
        "Model": {
            "__class_name__": "LiFlow",
            "__init_params__": TINY_INIT_PARAMS,
        },
        "Predict": {
            "propagator_model": None,
            "propagator_config": str(directory / "model.yaml"),
            "propagator_checkpoint": str(checkpoint),
            "corrector_model": None,
            "corrector_config": None,
            "corrector_checkpoint": None,
            "cutoff": 5.0,
            "periodic": True,
            "element_index": None,
            "temperature": 800.0,
            "steps": 2,
            "flow_steps": 2,
            "solver": "euler",
            "seed": 42,
        },
    }
    OmegaConf.save(OmegaConf.create(config), directory / "train.yaml")
    OmegaConf.save(
        OmegaConf.create({"Model": config["Model"]}), directory / "model.yaml"
    )
    return directory / "train.yaml"


def _run_predict(config: Path, output_dir: Path):
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(REPO_ROOT)] + [p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p]
    )
    return subprocess.run(
        [
            sys.executable,
            "molecular_dynamics_integrator/predict.py",
            "-c",
            str(config),
            "-i",
            str(EXAMPLE_INPUT),
            "-o",
            str(output_dir),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        env=env,
    )


def test_predict_entrypoint_writes_trajectory_with_local_checkpoint(tmp_path):
    config = _write_tiny_checkpoint(tmp_path)
    output_dir = tmp_path / "output"
    result = _run_predict(config, output_dir)
    assert result.returncode == 0, result.stdout + result.stderr

    trajectory = np.load(output_dir / "trajectory.npy")
    assert trajectory.shape == (3, 4, 3)
    assert trajectory.dtype == np.float32
    assert np.isfinite(trajectory).all()

    import json

    metadata = json.loads((output_dir / "run_metadata.json").read_text())
    assert metadata["steps"] == 2
    assert metadata["flow_steps"] == 2
    assert metadata["solver"] == "euler"
    assert metadata["trajectory_shape"] == [3, 4, 3]

    with open(output_dir / "trajectory.xyz") as handle:
        frame_counts = [int(line.strip()) for line in handle if line.strip().isdigit()]
    assert frame_counts == [4, 4, 4]


def test_build_side_resolves_registry_name(monkeypatch):
    module = _load_predict_module()
    monkeypatch.setattr(module, "_build_registered", lambda name: ("registered", name))
    assert module._build_side(
        {"propagator_model": "liflow_universal"}, "propagator"
    ) == (
        "registered",
        "liflow_universal",
    )


def test_build_side_returns_none_for_disabled_side():
    module = _load_predict_module()
    assert module._build_side({}, "corrector") is None


def test_build_side_rejects_two_sources():
    module = _load_predict_module()
    with pytest.raises(ValueError, match="mutually exclusive"):
        module._build_side(
            {"propagator_model": "liflow_universal", "propagator_checkpoint": "x"},
            "propagator",
        )

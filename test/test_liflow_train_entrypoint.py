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

"""Smoke tests for the LiFlow training entrypoint.

Each test launches ``molecular_dynamics_integrator/train.py`` as a real
subprocess for 2 optimizer steps against the tiny ``dataset_mini`` fixture and
asserts a clean exit plus a ``run.log`` under the (timestamp-suffixed) output
directory.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

CONFIGS = {
    "propagator": (
        "molecular_dynamics_integrator/liflow/configs/"
        "liflow_universal_propagator.yaml"
    ),
    "corrector": (
        "molecular_dynamics_integrator/liflow/configs/"
        "liflow_universal_corrector.yaml"
    ),
}


def _find_run_log(output_dir: Path) -> Path | None:
    """The entrypoint appends ``_t_<ts>_s_<seed>`` to ``output_dir``."""
    for candidate in output_dir.parent.glob(output_dir.name + "_t_*/run.log"):
        return candidate
    return None


def _run_entrypoint(config_rel: str, tmp_path: Path) -> None:
    result = _run_entrypoint_raw(config_rel, tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    run_log = _find_run_log(tmp_path)
    assert run_log is not None, result.stdout + result.stderr
    assert run_log.exists()


def _run_entrypoint_raw(config_rel: str, tmp_path: Path):
    import os
    import subprocess

    dataset_path = "test/fixtures/liflow/dataset_mini"
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(REPO_ROOT)] + [p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p]
    )
    return subprocess.run(
        [
            sys.executable,
            "molecular_dynamics_integrator/train.py",
            "-c",
            config_rel,
            "Trainer.max_epochs=1",
            "Trainer.max_iter=2",
            f"Trainer.output_dir={tmp_path.as_posix()}",
            f"Dataset.train.dataset.__init_params__.path={dataset_path}",
            f"Dataset.val.dataset.__init_params__.path={dataset_path}",
            "Dataset.train.loader.num_workers=0",
            "Dataset.val.loader.num_workers=0",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        env=env,
    )


def test_liflow_train_entrypoint_two_steps(tmp_path):
    _run_entrypoint(CONFIGS["propagator"], tmp_path)


def test_liflow_corrector_train_entrypoint_two_steps(tmp_path):
    _run_entrypoint(CONFIGS["corrector"], tmp_path)


def test_liflow_train_entrypoint_reports_eval_loss(tmp_path):
    result = _run_entrypoint_raw(CONFIGS["propagator"], tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "velocity" in result.stdout + result.stderr or "loss" in (
        result.stdout + result.stderr
    )

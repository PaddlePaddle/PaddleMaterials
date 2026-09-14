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

import importlib.util
import json
from pathlib import Path

from ppmat.models import MODEL_REGISTRY

ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str):
    path = ROOT / "benchmark" / "cinn" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"cinn_benchmark_{name}", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_supported_models_cover_every_qualified_registered_weight():
    runner = load_script("run_registry_matrix")
    models = runner.supported_models()

    assert len(MODEL_REGISTRY) == 76
    assert len(models) == 67
    assert set(MODEL_REGISTRY) - set(models) == {
        name for name in MODEL_REGISTRY if name.startswith("infgcn_")
    }


def test_valid_result_requires_matching_success_record(tmp_path):
    runner = load_script("run_registry_matrix")
    result_path = tmp_path / "result.json"
    result_path.write_text(
        json.dumps({"model": "chgnet_mptrj", "backend": "cinn", "status": "passed"}),
        encoding="utf-8",
    )

    assert runner.valid_result(result_path, "chgnet_mptrj", "cinn")
    assert not runner.valid_result(result_path, "chgnet_mptrj", "eager")
    assert not runner.valid_result(result_path, "mattersim_1M", "cinn")


def test_compare_selected_models_ignores_stale_results(tmp_path):
    comparison = load_script("compare_backends")
    eager_dir = tmp_path / "eager"
    cinn_dir = tmp_path / "cinn"
    eager_dir.mkdir()
    cinn_dir.mkdir()

    for directory, backend, warm in (
        (eager_dir, "eager", 0.02),
        (cinn_dir, "cinn", 0.01),
    ):
        record = {
            "model": "chgnet_mptrj",
            "backend": backend,
            "status": "passed",
            "first_seconds": 2.0,
            "warm_median_seconds": warm,
        }
        (directory / "chgnet_mptrj.json").write_text(
            json.dumps(record), encoding="utf-8"
        )
        (directory / "stale.json").write_text(json.dumps(record), encoding="utf-8")

    summary = comparison.compare(tmp_path, ["chgnet_mptrj"])
    assert summary["models"] == 1
    assert summary["compared"] == 1
    assert summary["speedup_median"] == 2.0

    output_path = tmp_path / "comparison.csv"
    comparison.write_csv(output_path, summary["results"])
    assert output_path.read_text(encoding="utf-8").startswith("model,family,")

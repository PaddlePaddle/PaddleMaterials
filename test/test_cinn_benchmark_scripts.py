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
    expected_models = {
        name for name in MODEL_REGISTRY if not name.startswith("infgcn_")
    }

    assert len(models) == len(expected_models)
    assert set(models) == expected_models


def test_valid_result_requires_matching_success_record(tmp_path):
    runner = load_script("run_registry_matrix")
    result_path = tmp_path / "result.json"
    result_path.write_text(
        json.dumps(
            {
                "model": "chgnet_mptrj",
                "backend": "cinn",
                "status": "passed",
                "prediction": {"energy": [1.0]},
            }
        ),
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
    assert output_path.read_text(encoding="utf-8").startswith("model,eager_status,")


def test_compare_computes_prediction_errors(tmp_path):
    comparison = load_script("compare_backends")
    eager_dir = tmp_path / "eager"
    cinn_dir = tmp_path / "cinn"
    eager_dir.mkdir()
    cinn_dir.mkdir()

    records = {
        "eager": {
            "model": "chgnet_mptrj",
            "backend": "eager",
            "status": "passed",
            "first_seconds": 1.0,
            "warm_median_seconds": 0.02,
            "prediction": {
                "energy": [1.0, 2.0],
                "forces": [[0.0, 0.1], [0.2, 0.3]],
            },
        },
        "cinn": {
            "model": "chgnet_mptrj",
            "backend": "cinn",
            "status": "passed",
            "first_seconds": 1.0,
            "warm_median_seconds": 0.01,
            "prediction": {
                "energy": [1.0 + 1e-5, 2.0 - 1e-5],
                "forces": [[0.0, 0.1 + 1e-5], [0.2, 0.3 - 1e-5]],
            },
        },
    }
    for directory, record in (
        (eager_dir, records["eager"]),
        (cinn_dir, records["cinn"]),
    ):
        (directory / "chgnet_mptrj.json").write_text(
            json.dumps(record),
            encoding="utf-8",
        )

    summary = comparison.compare(tmp_path, ["chgnet_mptrj"])
    row = summary["results"][0]

    assert row["precision_status"] == "passed"
    assert row["precision_matched_fields"] == 2
    assert abs(row["precision_max_abs_error"] - 1e-5) < 1e-12
    assert abs(row["precision_max_relative_error"] - 1e-4) < 1e-12
    assert summary["precision_compared"] == 1

    output_path = tmp_path / "comparison.csv"
    comparison.write_csv(output_path, summary["results"])
    header = output_path.read_text(encoding="utf-8").splitlines()[0]
    assert "precision_max_abs_error" in header


def test_prediction_serialization_converts_arrays_and_tensors():
    benchmark = load_script("benchmark_registered")
    value = {
        "energy": benchmark.np.array([1.0, 2.0]),
        "tensor": benchmark.paddle.to_tensor([3.0, 4.0]),
    }

    assert benchmark.to_jsonable(value) == {
        "energy": [1.0, 2.0],
        "tensor": [3.0, 4.0],
    }

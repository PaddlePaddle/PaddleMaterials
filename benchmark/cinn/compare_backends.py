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

"""Compare eager and CINN benchmark results for speed and precision."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

import numpy as np

EPSILON = 1e-12


def is_numeric_array(value) -> bool:
    """Return whether *value* can be used as a numeric array."""

    try:
        array = np.asarray(value)
    except (TypeError, ValueError):
        return False
    return array.dtype.kind in "iufb" and array.size > 0


def iter_numeric_arrays(value, path: str = ""):
    """Yield (path, ndarray) pairs for every numeric array in a prediction."""

    if isinstance(value, dict):
        for key in sorted(value):
            name = f"{path}.{key}" if path else str(key)
            yield from iter_numeric_arrays(value[key], name)
    elif is_numeric_array(value):
        yield path, np.asarray(value)
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            name = f"{path}[{index}]"
            yield from iter_numeric_arrays(item, name)


def array_errors(reference: np.ndarray, candidate: np.ndarray) -> dict:
    """Return the aggregate errors for one eager/CINN array pair."""

    reference = reference.astype(np.float64)
    candidate = candidate.astype(np.float64)
    difference = reference - candidate
    abs_difference = np.abs(difference)
    relative = abs_difference / np.maximum(np.abs(reference), EPSILON)
    reference_norm = float(np.linalg.norm(reference))
    difference_norm = float(np.linalg.norm(difference))
    return {
        "exact": bool(np.array_equal(reference, candidate)),
        "max_abs_error": float(np.max(abs_difference)),
        "max_relative_error": float(np.max(relative)),
        "relative_l2_error": difference_norm / max(reference_norm, EPSILON),
    }


def compare_predictions(eager, cinn) -> dict:
    """Compare eager/CINN prediction outputs and return core error metrics."""

    if eager is None or cinn is None:
        return {
            "status": "not_compared",
            "fields": 0,
            "matched_fields": 0,
            "missing_fields": [],
            "mismatched_fields": [],
            "max_abs_error": None,
            "max_relative_error": None,
            "relative_l2_error": None,
        }

    eager_arrays = dict(iter_numeric_arrays(eager))
    cinn_arrays = dict(iter_numeric_arrays(cinn))
    field_errors = {}
    for path in sorted(set(eager_arrays) | set(cinn_arrays)):
        if path not in eager_arrays or path not in cinn_arrays:
            field_errors[path] = {"status": "missing"}
            continue
        reference = eager_arrays[path]
        candidate = cinn_arrays[path]
        if reference.shape != candidate.shape:
            field_errors[path] = {
                "status": "shape_mismatch",
                "eager_shape": list(reference.shape),
                "cinn_shape": list(candidate.shape),
            }
            continue
        field_errors[path] = {
            "status": "compared",
            **array_errors(reference, candidate),
        }

    matched = [
        metrics for metrics in field_errors.values() if metrics["status"] == "compared"
    ]
    missing_fields = [
        path for path, metrics in field_errors.items() if metrics["status"] == "missing"
    ]
    mismatched_fields = [
        path
        for path, metrics in field_errors.items()
        if metrics["status"] == "shape_mismatch"
    ]
    if not field_errors:
        status = "no_arrays"
    elif missing_fields or mismatched_fields:
        status = "failed"
    else:
        status = "passed"

    return {
        "status": status,
        "fields": len(field_errors),
        "matched_fields": len(matched),
        "missing_fields": missing_fields,
        "mismatched_fields": mismatched_fields,
        "max_abs_error": (
            max(item["max_abs_error"] for item in matched) if matched else None
        ),
        "max_relative_error": (
            max(item["max_relative_error"] for item in matched) if matched else None
        ),
        "relative_l2_error": (
            max(item["relative_l2_error"] for item in matched) if matched else None
        ),
    }


def load_results(directory: Path) -> dict[str, dict]:
    return {
        path.stem: json.loads(path.read_text(encoding="utf-8"))
        for path in directory.glob("*.json")
    }


def compare_scalars(eager, cinn) -> dict:
    """Compare eager/CINN scalar values such as loss and metric outputs."""

    if not isinstance(eager, dict) or not isinstance(cinn, dict):
        return {"status": "not_compared", "values": {}}

    values = {}
    for key in sorted(set(eager) | set(cinn)):
        if key not in eager or key not in cinn:
            values[key] = {"status": "missing"}
            continue
        left = eager[key]
        right = cinn[key]
        if left is None or right is None:
            values[key] = {"status": "not_compared"}
            continue
        if not isinstance(left, (int, float)) or not isinstance(right, (int, float)):
            values[key] = {"status": "not_compared"}
            continue
        absolute = abs(left - right)
        relative = absolute / max(abs(left), EPSILON)
        values[key] = {
            "status": "compared",
            "eager": left,
            "cinn": right,
            "abs_error": absolute,
            "relative_error": relative,
        }

    compared = [item for item in values.values() if item["status"] == "compared"]
    if not values:
        status = "not_compared"
    elif len(compared) != len(values):
        status = "failed"
    else:
        status = "passed"
    return {"status": status, "values": values}


def compare(root: Path, models: list[str] | None = None) -> dict:
    eager_results = load_results(root / "eager")
    cinn_results = load_results(root / "cinn")
    models = sorted(models or (set(eager_results) | set(cinn_results)))
    rows = []
    for model in models:
        eager = eager_results.get(model)
        cinn = cinn_results.get(model)
        row = {
            "model": model,
            "eager_status": eager and eager.get("status"),
            "cinn_status": cinn and cinn.get("status"),
        }
        if row["eager_status"] == row["cinn_status"] == "passed":
            eager_warm = eager["warm_median_seconds"]
            cinn_warm = cinn["warm_median_seconds"]
            row.update(
                {
                    "first_cinn_seconds": cinn["first_seconds"],
                    "eager_warm_seconds": eager_warm,
                    "cinn_warm_seconds": cinn_warm,
                    "speedup": eager_warm / cinn_warm,
                }
            )
            precision = compare_predictions(
                eager.get("prediction"),
                cinn.get("prediction"),
            )
        else:
            precision = {
                "status": "not_compared",
                "fields": 0,
                "matched_fields": 0,
                "missing_fields": [],
                "mismatched_fields": [],
                "max_abs_error": None,
                "max_relative_error": None,
                "relative_l2_error": None,
            }
        row.update(
            {
                "precision_status": precision["status"],
                "precision_fields": precision["fields"],
                "precision_matched_fields": precision["matched_fields"],
                "precision_max_abs_error": precision["max_abs_error"],
                "precision_max_relative_error": precision["max_relative_error"],
                "precision_relative_l2_error": precision["relative_l2_error"],
            }
        )
        extra = compare_scalars(
            {
                **(eager and eager.get("loss") or {}),
                **(eager and eager.get("metrics") or {}),
            },
            {
                **(cinn and cinn.get("loss") or {}),
                **(cinn and cinn.get("metrics") or {}),
            },
        )
        row.update(
            {
                "loss_metric_status": extra["status"],
                "loss_metric": extra["values"],
            }
        )
        rows.append(row)

    speedups = [row["speedup"] for row in rows if row.get("speedup") is not None]
    precision_statuses = [row["precision_status"] for row in rows]
    scalar_statuses = [row["loss_metric_status"] for row in rows]
    return {
        "models": len(rows),
        "compared": len(speedups),
        "eager_passed": sum(row["eager_status"] == "passed" for row in rows),
        "cinn_passed": sum(row["cinn_status"] == "passed" for row in rows),
        "faster": sum(speedup > 1 for speedup in speedups),
        "precision_compared": sum(status == "passed" for status in precision_statuses),
        "precision_failed": sum(
            status in {"failed", "no_arrays"} for status in precision_statuses
        ),
        "precision_not_compared": sum(
            status in {"missing", "not_compared"} for status in precision_statuses
        ),
        "loss_metric_compared": sum(status == "passed" for status in scalar_statuses),
        "loss_metric_failed": sum(status == "failed" for status in scalar_statuses),
        "speedup_median": statistics.median(speedups) if speedups else None,
        "speedup_min": min(speedups) if speedups else None,
        "speedup_max": max(speedups) if speedups else None,
        "results": rows,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    """Write per-weight speed and precision metrics as CSV."""

    fields = (
        "model",
        "eager_status",
        "cinn_status",
        "first_cinn_seconds",
        "eager_warm_seconds",
        "cinn_warm_seconds",
        "speedup",
        "precision_status",
        "precision_fields",
        "precision_matched_fields",
        "precision_max_abs_error",
        "precision_max_relative_error",
        "precision_relative_l2_error",
        "loss_metric_status",
    )
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--models", nargs="+")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    summary = compare(root, args.models)
    (root / "comparison.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    write_csv(root / "comparison.csv", summary["results"])
    print(
        json.dumps(
            {key: value for key, value in summary.items() if key != "results"},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

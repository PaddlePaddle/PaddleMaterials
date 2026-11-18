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
import math
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np

EPSILON = 1e-12


def family_name(model_name: str) -> str:
    if model_name.startswith("spherenet_md17_"):
        return "SphereNet MD17"
    if model_name.startswith("spherenet_qm9_"):
        return "SphereNet QM9"
    families = {
        "chgnet_": "CHGNet",
        "comformer_": "ComFormer",
        "diffcsp_": "DiffCSP",
        "diffnmr_": "DiffNMR",
        "dimenetpp_": "DimeNet++",
        "mattergen_": "MatterGen",
        "mattersim_": "MatterSim",
        "megnet_": "MEGNet",
        "sfin_": "SFIN",
    }
    for prefix, family in families.items():
        if model_name.startswith(prefix):
            return family
    raise ValueError(f"Unknown benchmark family for {model_name!r}")


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


def array_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict:
    """Compare one eager/CINN array pair and return error metrics."""

    if reference.shape != candidate.shape:
        return {
            "status": "shape_mismatch",
            "shape_match": False,
            "eager_shape": list(reference.shape),
            "cinn_shape": list(candidate.shape),
        }
    reference = reference.astype(np.float64)
    candidate = candidate.astype(np.float64)
    difference = reference - candidate
    abs_difference = np.abs(difference)
    relative = abs_difference / np.maximum(np.abs(reference), EPSILON)
    reference_norm = float(np.linalg.norm(reference))
    difference_norm = float(np.linalg.norm(difference))
    return {
        "status": "compared",
        "shape_match": True,
        "eager_shape": list(reference.shape),
        "cinn_shape": list(candidate.shape),
        "elements": int(reference.size),
        "exact": bool(np.array_equal(reference, candidate)),
        "max_abs_error": float(np.max(abs_difference)),
        "mean_abs_error": float(np.mean(abs_difference)),
        "max_relative_error": float(np.max(relative)),
        "mean_relative_error": float(np.mean(relative)),
        "relative_l2_error": difference_norm / max(reference_norm, EPSILON),
    }


def compare_predictions(eager, cinn) -> dict:
    """Compare nested eager/CINN prediction outputs field by field."""

    if eager is None or cinn is None:
        missing = []
        if eager is None:
            missing.append("eager")
        if cinn is None:
            missing.append("cinn")
        return {
            "status": "missing",
            "fields": 0,
            "matched_fields": 0,
            "missing_fields": ["prediction"],
            "mismatched_fields": [],
            "max_abs_error": None,
            "mean_abs_error": None,
            "max_relative_error": None,
            "mean_relative_error": None,
            "relative_l2_error": None,
            "missing_backends": missing,
            "field_metrics": {},
        }

    eager_arrays = dict(iter_numeric_arrays(eager))
    cinn_arrays = dict(iter_numeric_arrays(cinn))
    field_metrics = {}
    for path in sorted(set(eager_arrays) | set(cinn_arrays)):
        if path not in eager_arrays or path not in cinn_arrays:
            field_metrics[path] = {
                "status": "missing",
                "eager_shape": (
                    list(np.asarray(eager_arrays[path]).shape)
                    if path in eager_arrays
                    else None
                ),
                "cinn_shape": (
                    list(np.asarray(cinn_arrays[path]).shape)
                    if path in cinn_arrays
                    else None
                ),
            }
            continue
        field_metrics[path] = array_metrics(
            eager_arrays[path],
            cinn_arrays[path],
        )

    matched = [
        metrics for metrics in field_metrics.values() if metrics.get("shape_match")
    ]
    missing_fields = [
        path
        for path, metrics in field_metrics.items()
        if metrics.get("status") == "missing"
    ]
    mismatched_fields = [
        path
        for path, metrics in field_metrics.items()
        if metrics.get("shape_match") is False
    ]
    if not field_metrics:
        status = "no_arrays"
    elif missing_fields or mismatched_fields:
        status = "failed"
    else:
        status = "passed"

    max_abs_error = None
    mean_abs_error = None
    max_relative_error = None
    mean_relative_error = None
    relative_l2_error = None
    if matched:
        max_abs_error = max(item["max_abs_error"] for item in matched)
        mean_abs_error = float(np.mean([item["mean_abs_error"] for item in matched]))
        max_relative_error = max(item["max_relative_error"] for item in matched)
        mean_relative_error = float(
            np.mean([item["mean_relative_error"] for item in matched])
        )
        relative_l2_error = max(item["relative_l2_error"] for item in matched)

    return {
        "status": status,
        "fields": len(field_metrics),
        "matched_fields": len(matched),
        "missing_fields": missing_fields,
        "mismatched_fields": mismatched_fields,
        "max_abs_error": max_abs_error,
        "mean_abs_error": mean_abs_error,
        "max_relative_error": max_relative_error,
        "mean_relative_error": mean_relative_error,
        "relative_l2_error": relative_l2_error,
        "field_metrics": field_metrics,
    }


def load_results(directory: Path) -> dict[str, dict]:
    return {
        path.stem: json.loads(path.read_text(encoding="utf-8"))
        for path in directory.glob("*.json")
    }


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
            "family": family_name(model),
            "eager_status": eager and eager.get("status"),
            "cinn_status": cinn and cinn.get("status"),
        }
        if row["eager_status"] == row["cinn_status"] == "passed":
            first_cinn = cinn["first_seconds"]
            eager_warm = eager["warm_median_seconds"]
            cinn_warm = cinn["warm_median_seconds"]
            speedup = eager_warm / cinn_warm
            compile_estimate = first_cinn - cinn_warm
            break_even = None
            if eager_warm > cinn_warm:
                break_even = math.ceil(compile_estimate / (eager_warm - cinn_warm))
            row.update(
                {
                    "first_cinn_seconds": first_cinn,
                    "compile_estimate_seconds": compile_estimate,
                    "eager_warm_seconds": eager_warm,
                    "cinn_warm_seconds": cinn_warm,
                    "speedup": speedup,
                    "break_even_calls": break_even,
                }
            )
        if row["eager_status"] == row["cinn_status"] == "passed":
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
                "mean_abs_error": None,
                "max_relative_error": None,
                "mean_relative_error": None,
                "relative_l2_error": None,
                "field_metrics": {},
            }
        row.update(
            {
                "precision": precision,
                "precision_status": precision["status"],
                "precision_matched_fields": precision["matched_fields"],
                "precision_max_abs_error": precision.get("max_abs_error"),
                "precision_mean_abs_error": precision.get("mean_abs_error"),
                "precision_max_relative_error": precision.get("max_relative_error"),
                "precision_mean_relative_error": precision.get("mean_relative_error"),
                "precision_relative_l2_error": precision.get("relative_l2_error"),
            }
        )
        rows.append(row)

    valid_rows = [row for row in rows if row.get("speedup") is not None]
    family_rows = defaultdict(list)
    for row in valid_rows:
        family_rows[row["family"]].append(row)
    families = []
    for family, items in sorted(family_rows.items()):
        speedups = [item["speedup"] for item in items]
        precision_abs = [
            item["precision_max_abs_error"]
            for item in items
            if item["precision_max_abs_error"] is not None
        ]
        precision_relative = [
            item["precision_max_relative_error"]
            for item in items
            if item["precision_max_relative_error"] is not None
        ]
        families.append(
            {
                "family": family,
                "weights": len(items),
                "faster_weights": sum(speedup > 1 for speedup in speedups),
                "first_cinn_median_seconds": statistics.median(
                    item["first_cinn_seconds"] for item in items
                ),
                "eager_warm_median_seconds": statistics.median(
                    item["eager_warm_seconds"] for item in items
                ),
                "cinn_warm_median_seconds": statistics.median(
                    item["cinn_warm_seconds"] for item in items
                ),
                "speedup_median": statistics.median(speedups),
                "speedup_min": min(speedups),
                "speedup_max": max(speedups),
                "precision_compared": sum(
                    item["precision_status"] == "passed" for item in items
                ),
                "precision_max_abs_error_median": (
                    statistics.median(precision_abs) if precision_abs else None
                ),
                "precision_max_relative_error_median": (
                    statistics.median(precision_relative)
                    if precision_relative
                    else None
                ),
            }
        )

    speedups = [row["speedup"] for row in valid_rows]
    precision_statuses = [row["precision_status"] for row in rows]
    return {
        "models": len(rows),
        "compared": len(valid_rows),
        "eager_passed": sum(row["eager_status"] == "passed" for row in rows),
        "cinn_passed": sum(row["cinn_status"] == "passed" for row in rows),
        "faster": sum(speedup > 1 for speedup in speedups),
        "slower_or_equal": sum(speedup <= 1 for speedup in speedups),
        "precision_compared": sum(status == "passed" for status in precision_statuses),
        "precision_failed": sum(
            status in {"failed", "no_arrays"} for status in precision_statuses
        ),
        "precision_missing": sum(
            status in {"missing", "not_compared"} for status in precision_statuses
        ),
        "speedup_median": statistics.median(speedups) if speedups else None,
        "speedup_min": min(speedups) if speedups else None,
        "speedup_max": max(speedups) if speedups else None,
        "families": families,
        "results": rows,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    """Write the most useful per-weight metrics as a spreadsheet-friendly CSV."""

    fields = (
        "model",
        "family",
        "eager_status",
        "cinn_status",
        "first_cinn_seconds",
        "eager_warm_seconds",
        "cinn_warm_seconds",
        "speedup",
        "break_even_calls",
        "precision_status",
        "precision_matched_fields",
        "precision_max_abs_error",
        "precision_mean_abs_error",
        "precision_max_relative_error",
        "precision_mean_relative_error",
        "precision_relative_l2_error",
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
            {
                key: value
                for key, value in summary.items()
                if key not in {"families", "results"}
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

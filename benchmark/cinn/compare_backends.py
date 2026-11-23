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

"""Compare eager and CINN registered-weight benchmark results."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


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
        rows.append(row)

    valid_rows = [row for row in rows if row.get("speedup") is not None]
    family_rows = defaultdict(list)
    for row in valid_rows:
        family_rows[row["family"]].append(row)
    families = []
    for family, items in sorted(family_rows.items()):
        speedups = [item["speedup"] for item in items]
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
            }
        )

    speedups = [row["speedup"] for row in valid_rows]
    return {
        "models": len(rows),
        "compared": len(valid_rows),
        "eager_passed": sum(row["eager_status"] == "passed" for row in rows),
        "cinn_passed": sum(row["cinn_status"] == "passed" for row in rows),
        "faster": sum(speedup > 1 for speedup in speedups),
        "slower_or_equal": sum(speedup <= 1 for speedup in speedups),
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

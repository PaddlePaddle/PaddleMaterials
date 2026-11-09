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

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from molecular_dynamics_integrator.evaluate import evaluate_trajectory

TEMPERATURES = (600, 800, 1000, 1200)
SEEDS = (1, 2, 3)
MODES = ("propagator", "propagator_corrector")

METRIC_KEYS = (
    "msd_Li",
    "msd_Li_ref",
    "msd_frame",
    "msd_frame_ref",
    "rdf_mae",
    "final_step",
)


def _load_structure(path: Path):
    data = np.load(path)
    return (
        np.asarray(data["atomic_numbers"], dtype=np.int64),
        np.asarray(data["lattice"], dtype=np.float64),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction_dir", type=str, required=True)
    parser.add_argument("--reference_dir", type=str, required=True)
    parser.add_argument("--structure", type=str, required=True)
    parser.add_argument("--output", type=str, default="output/liflow_universal_eval")
    parser.add_argument(
        "--temperatures",
        type=lambda s: tuple(int(x) for x in s.split(",")),
        default=TEMPERATURES,
        help="Comma-separated temperatures, e.g. '600,800,1000,1200'",
    )
    parser.add_argument(
        "--seeds", type=lambda s: tuple(int(x) for x in s.split(",")), default=SEEDS
    )
    parser.add_argument(
        "--modes",
        type=lambda s: tuple(x for x in s.split(",")),
        default=MODES,
    )
    args = parser.parse_args()

    prediction_dir = Path(args.prediction_dir)
    reference_dir = Path(args.reference_dir)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    atomic_numbers, lattice = _load_structure(args.structure)

    failures = 0
    total = 0
    per_row = []
    for temp in args.temperatures:
        for seed in args.seeds:
            for mode in args.modes:
                total += 1
                pred_path = prediction_dir / f"temp_{temp}_seed_{seed}_{mode}.npy"
                ref_path = reference_dir / f"temp_{temp}_seed_{seed}.npy"
                if not pred_path.exists() or not ref_path.exists():
                    failures += 1
                    continue
                try:
                    prediction = np.load(pred_path)
                    reference = np.load(ref_path)
                    metrics = evaluate_trajectory(
                        prediction, reference, atomic_numbers, lattice
                    )
                except Exception:  # noqa: BLE001
                    failures += 1
                    continue
                row = {
                    "temperature": temp,
                    "seed": seed,
                    "mode": mode,
                    **metrics,
                }
                per_row.append(row)

    import csv
    for temp in args.temperatures:
        for mode in args.modes:
            rows = [
                r for r in per_row if r["temperature"] == temp and r["mode"] == mode
            ]
            csv_path = output_dir / f"eval_{temp}K_{mode}.csv"
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=["temperature", "seed", "mode", *METRIC_KEYS],
                )
                writer.writeheader()
                writer.writerows(rows)

    aggregate = {
        "temperatures": list(args.temperatures),
        "seeds": list(args.seeds),
        "modes": list(args.modes),
        "total_combinations": total,
        "failed_combinations": failures,
        "failure_rate": failures / total if total else 0.0,
        "by_metric": {},
    }
    for key in METRIC_KEYS:
        values = [r[key] for r in per_row]
        if values:
            aggregate["by_metric"][key] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
            }
        else:
            aggregate["by_metric"][key] = {"mean": None, "std": None}

    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(aggregate, f, indent=2)
    print(f"Summary written to {summary_path}")


if __name__ == "__main__":
    main()

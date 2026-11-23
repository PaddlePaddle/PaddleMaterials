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

"""Run eager and CINN benchmarks for every supported registered weight."""

from __future__ import annotations

import argparse
import json
import os
import queue
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
from pathlib import Path

from ppmat.models import MODEL_REGISTRY

ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_ONE = Path(__file__).with_name("benchmark_registered.py")
COMPARE_BACKENDS = Path(__file__).with_name("compare_backends.py")
CINN_SUPPORTED_PREFIXES = (
    "chgnet_",
    "comformer_",
    "diffcsp_",
    "diffnmr_",
    "dimenetpp_",
    "mattergen_",
    "mattersim_",
    "megnet_",
    "sfin_",
    "spherenet_",
)


def model_timeout(model_name: str) -> int:
    """Return the per-process timeout in seconds."""

    if model_name.startswith("mattergen_"):
        return 1800
    return 900


def supported_models() -> list[str]:
    """Return registered weights with a qualified CINN workflow."""

    return sorted(
        model_name
        for model_name in MODEL_REGISTRY
        if model_name.startswith(CINN_SUPPORTED_PREFIXES)
    )


def valid_result(path: Path, model_name: str, backend: str) -> bool:
    """Return whether *path* is a reusable successful result."""

    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return (
        result.get("model") == model_name
        and result.get("backend") == backend
        and result.get("status") == "passed"
    )


def run_one(
    model_name: str,
    backend: str,
    output_dir: Path,
    gpu_queue: queue.Queue,
    repeats: int,
    resume: bool = False,
) -> dict:
    """Run one registered weight and return its launcher result."""

    gpu = gpu_queue.get()
    started = time.perf_counter()
    result_path = output_dir / backend / f"{model_name}.json"
    log_path = output_dir / "logs" / backend / f"{model_name}.log"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if resume and valid_result(result_path, model_name, backend):
        gpu_queue.put(gpu)
        return {
            "model": model_name,
            "backend": backend,
            "status": "skipped",
            "returncode": 0,
            "gpu": gpu,
            "wall_seconds": 0.0,
            "result": str(result_path),
            "log": str(log_path),
        }

    # Never let a result from an earlier invocation mask a crash or timeout in
    # this one. The worker will create a fresh record, including failures it can
    # catch itself.
    result_path.unlink(missing_ok=True)
    command = [
        sys.executable,
        str(BENCHMARK_ONE),
        "--model_name",
        model_name,
        "--backend",
        backend,
        "--repeats",
        str(repeats),
        "--output",
        str(result_path),
    ]
    timeout = model_timeout(model_name)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    try:
        with log_path.open("w", encoding="utf-8") as log_file:
            process = subprocess.run(
                command,
                cwd=ROOT,
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                check=False,
                timeout=timeout + 60,
            )
        status = (
            "passed"
            if process.returncode == 0
            and valid_result(result_path, model_name, backend)
            else "failed"
        )
        returncode = process.returncode
    except subprocess.TimeoutExpired:
        status = "timeout"
        returncode = 124
    finally:
        gpu_queue.put(gpu)

    return {
        "model": model_name,
        "backend": backend,
        "status": status,
        "returncode": returncode,
        "gpu": gpu,
        "wall_seconds": time.perf_counter() - started,
        "result": str(result_path),
        "log": str(log_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark every CINN-qualified registered weight. By default both "
            "eager and CINN are measured and compared."
        )
    )
    parser.add_argument("--backend", choices=("eager", "cinn", "both"), default="both")
    parser.add_argument("--output_dir", default="output/cinn_benchmark")
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--models",
        nargs="+",
        help="Optional registered-weight names; the default is all supported weights.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse matching successful JSON records from an interrupted run.",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="Print the resolved CINN-supported registry and exit.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the selected backends and weights without downloading or running.",
    )
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be at least 1")
    return args


def run_backend(
    backend: str,
    models: list[str],
    output_dir: Path,
    gpus: list[str],
    repeats: int,
    resume: bool,
) -> dict:
    """Run one backend over *models* and write its launcher summary."""

    gpu_queue = queue.Queue()
    for gpu in gpus:
        gpu_queue.put(gpu)

    launchers = []
    with ThreadPoolExecutor(max_workers=len(gpus)) as executor:
        futures = {
            executor.submit(
                run_one,
                model,
                backend,
                output_dir,
                gpu_queue,
                repeats,
                resume,
            ): model
            for model in models
        }
        for future in as_completed(futures):
            launcher = future.result()
            launchers.append(launcher)
            print(
                f"[{backend} {len(launchers)}/{len(models)}] "
                f"{launcher['status']:7s} gpu={launcher['gpu']} "
                f"{launcher['model']} ({launcher['wall_seconds']:.1f}s)",
                flush=True,
            )

    summary = {
        "backend": backend,
        "models": len(models),
        "passed": sum(item["status"] == "passed" for item in launchers),
        "skipped": sum(item["status"] == "skipped" for item in launchers),
        "failed": sum(
            item["status"] not in {"passed", "skipped"} for item in launchers
        ),
        "results": sorted(launchers, key=lambda item: item["model"]),
    }
    summary_path = output_dir / f"{backend}_launcher_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({key: summary[key] for key in summary if key != "results"}))
    return summary


def main() -> None:
    args = parse_args()
    all_supported = supported_models()
    if args.list_models:
        print("\n".join(all_supported))
        print(f"total: {len(all_supported)}")
        return

    available = set(all_supported)
    models = list(dict.fromkeys(args.models or all_supported))
    unknown = sorted(set(models) - available)
    if unknown:
        raise ValueError("Unsupported CINN benchmark models: " + ", ".join(unknown))

    backends = ["eager", "cinn"] if args.backend == "both" else [args.backend]
    if args.dry_run:
        print(f"backends: {','.join(backends)}")
        print(f"models: {len(models)}")
        print("\n".join(models))
        return

    gpus = list(
        dict.fromkeys(gpu.strip() for gpu in args.gpus.split(",") if gpu.strip())
    )
    if not gpus:
        raise ValueError("--gpus must contain at least one GPU index")

    output_dir = Path(args.output_dir).resolve()
    summaries = [
        run_backend(
            backend,
            models,
            output_dir,
            gpus,
            args.repeats,
            args.resume,
        )
        for backend in backends
    ]

    if args.backend == "both":
        compare = subprocess.run(
            [
                sys.executable,
                str(COMPARE_BACKENDS),
                "--root",
                str(output_dir),
                "--models",
                *models,
            ],
            cwd=ROOT,
            check=False,
        )
        compare_failed = compare.returncode != 0
    else:
        compare_failed = False
    failed = sum(summary["failed"] for summary in summaries)
    raise SystemExit(1 if failed or compare_failed else 0)


if __name__ == "__main__":
    main()

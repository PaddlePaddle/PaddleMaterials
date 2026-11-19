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
import subprocess
import sys
import time
from pathlib import Path

from ppmat.models import MODEL_REGISTRY

ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_ONE = Path(__file__).with_name("benchmark_registered.py")
COMPARE_BACKENDS = Path(__file__).with_name("compare_backends.py")
MODEL_TIMEOUT_SECONDS = 1800
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


def supported_models() -> list[str]:
    """Return registered weights with a qualified CINN workflow."""

    return sorted(
        model_name
        for model_name in MODEL_REGISTRY
        if model_name.startswith(CINN_SUPPORTED_PREFIXES)
    )


def valid_result(path: Path, model_name: str, backend: str) -> bool:
    """Return whether *path* contains a successful result for this backend."""

    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return (
        result.get("model") == model_name
        and result.get("backend") == backend
        and result.get("status") == "passed"
        and result.get("prediction") is not None
    )


def run_one(
    model_name: str,
    backend: str,
    output_dir: Path,
    gpu: str,
    repeats: int,
) -> dict:
    """Run one registered weight in a child process."""

    started = time.perf_counter()
    result_path = output_dir / backend / f"{model_name}.json"
    log_path = output_dir / "logs" / backend / f"{model_name}.log"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
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
                timeout=MODEL_TIMEOUT_SECONDS + 60,
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
            "Benchmark registered weights with eager and CINN, then compare "
            "their speed and prediction precision."
        )
    )
    parser.add_argument(
        "--backend", choices=("eager", "cinn", "both"), default="both"
    )
    parser.add_argument("--output_dir", default="output/cinn_benchmark")
    parser.add_argument("--gpu", default="0", help="CUDA device index")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--models",
        nargs="+",
        help="Registered-weight names; the default is all supported weights.",
    )
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be at least 1")
    return args


def run_backend(
    backend: str,
    models: list[str],
    output_dir: Path,
    gpu: str,
    repeats: int,
) -> tuple[int, int]:
    """Run one backend over *models* and report passed/failed counts."""

    launchers = [
        run_one(model, backend, output_dir, gpu, repeats) for model in models
    ]
    for launcher in launchers:
        print(
            f"[{backend} {launcher['status']:7s}] gpu={launcher['gpu']} "
            f"{launcher['model']} ({launcher['wall_seconds']:.1f}s)",
            flush=True,
        )
    passed = sum(item["status"] == "passed" for item in launchers)
    failed = sum(item["status"] != "passed" for item in launchers)
    print(json.dumps({"backend": backend, "passed": passed, "failed": failed}))
    return passed, failed


def main() -> None:
    args = parse_args()
    available = set(supported_models())
    models = list(dict.fromkeys(args.models or sorted(available)))
    unknown = sorted(set(models) - available)
    if unknown:
        raise ValueError(
            "Unsupported CINN benchmark models: " + ", ".join(unknown)
        )

    output_dir = Path(args.output_dir).resolve()
    backends = ["eager", "cinn"] if args.backend == "both" else [args.backend]
    failure_count = 0
    for backend in backends:
        _, failed = run_backend(
            backend,
            models,
            output_dir,
            args.gpu,
            args.repeats,
        )
        failure_count += failed

    compare_failed = False
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

    raise SystemExit(1 if failure_count or compare_failed else 0)


if __name__ == "__main__":
    main()

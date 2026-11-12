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

"""Measure GE-GNN tensor-adapter inference; excludes preprocessing and compilation."""
import argparse
import copy
import hashlib
import json
import statistics
import time
from pathlib import Path

import numpy as np
import paddle
from omegaconf import OmegaConf

from ppmat.datasets import build_dataloader
from ppmat.models import build_model
from ppmat.models.gegnn import GEGNNTensorPredictor


def measure(model, inputs, repeats, groups):
    durations = []
    for _ in range(groups):
        paddle.device.synchronize()
        start = time.perf_counter()
        for _ in range(repeats):
            model(inputs)
        paddle.device.synchronize()
        durations.append((time.perf_counter() - start) / repeats)
    return durations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config_path",
        default="property_prediction/configs/gegnn/gegnn_binary_activity.yaml",
    )
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--output_path", default="gegnn_cinn_benchmark.json")
    parser.add_argument("--batch_sizes", type=int, nargs="+", default=[1, 2, 8, 32])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--groups", type=int, default=5)
    args = parser.parse_args()
    if min(args.batch_sizes + [args.warmup, args.repeats, args.groups]) <= 0:
        parser.error("Batch sizes and timing counts must be positive.")
    paddle.set_device("gpu:0")
    config = OmegaConf.to_container(OmegaConf.load(args.config_path), resolve=True)
    model = build_model(config["Model"])
    model.set_state_dict(paddle.load(args.checkpoint_path))
    model.eval()
    eager = GEGNNTensorPredictor(model)
    eager.eval()
    # to_static mutates the Layer: independent copies are essential for timing.
    compiled = paddle.jit.to_static(
        copy.deepcopy(eager), full_graph=True, backend="CINN"
    )
    report = {
        "scope": (
            "Inference microbenchmark; excludes data loading, packing, transfer "
            "and compilation from steady-state timing"
        ),
        "paddle": paddle.__version__,
        "device": paddle.device.cuda.get_device_name(),
        "checkpoint_sha256": hashlib.sha256(
            Path(args.checkpoint_path).read_bytes()
        ).hexdigest(),
        "warmup": args.warmup,
        "repeats": args.repeats,
        "groups": args.groups,
        "independent_layer_copies": True,
        "cases": [],
    }
    for size in args.batch_sizes:
        dataset = copy.deepcopy(config["Dataset"]["val"])
        dataset["sampler"]["__init_params__"]["batch_size"] = size
        batch = next(iter(build_dataloader(dataset)))
        inputs = eager.pack(batch)
        reference = model.predict(batch)[model.property_name].numpy()
        adapted = eager(inputs).numpy()
        paddle.device.synchronize()
        start = time.perf_counter()
        actual = compiled(inputs).numpy()
        paddle.device.synchronize()
        first_call = time.perf_counter() - start
        adapter_error = float(np.max(np.abs(adapted - reference)))
        error = float(np.max(np.abs(actual - reference)))
        if not np.isfinite(actual).all() or adapter_error > 1e-6 or error > 1e-4:
            raise RuntimeError(
                f"Prediction mismatch: adapter={adapter_error}, CINN={error}"
            )
        for _ in range(args.warmup):
            eager(inputs)
            compiled(inputs)
        eager_times = measure(eager, inputs, args.repeats, args.groups)
        cinn_times = measure(compiled, inputs, args.repeats, args.groups)
        case = {
            "batch_size": size,
            "nodes1": int(inputs["g1_node_features"].shape[0]),
            "nodes2": int(inputs["g2_node_features"].shape[0]),
            "adapter_max_abs_error": adapter_error,
            "cinn_max_abs_error": error,
            "first_call_seconds_including_compilation": first_call,
            "eager_seconds_per_call": eager_times,
            "cinn_seconds_per_call": cinn_times,
            "steady_state_ratio": statistics.median(eager_times)
            / statistics.median(cinn_times),
        }
        report["cases"].append(case)
        print(json.dumps(case), flush=True)
        Path(args.output_path).write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()

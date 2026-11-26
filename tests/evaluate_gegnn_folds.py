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

"""Reevaluate existing GE-GNN fold checkpoints without training."""

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import paddle
import pandas as pd
from omegaconf import OmegaConf

from ppmat.datasets import build_dataloader
from ppmat.models import build_model


def prediction_metrics(predictions, targets):
    predictions = np.asarray(predictions, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    if predictions.shape != targets.shape or targets.ndim != 2 or not len(targets):
        raise ValueError(
            "Predictions and targets must have the same nonempty [N, 2] shape."
        )
    if (
        targets.shape[1] != 2
        or not np.isfinite(predictions).all()
        or not np.isfinite(targets).all()
    ):
        raise ValueError("Expected finite binary-mixture predictions and targets.")
    error = predictions - targets
    sse = np.square(error).sum(axis=0)
    centered = targets - targets.mean(axis=0)
    sst = np.square(centered).sum(axis=0)
    r2 = np.divide(sse, sst, out=np.full_like(sse, np.nan), where=sst > 0)
    component_r2 = [None if not np.isfinite(x) else float(1 - x) for x in r2]
    return {
        "mae": float(np.abs(error).mean()),
        "rmse": float(np.sqrt(np.square(error).mean())),
        "r2_uniform_average": None
        if None in component_r2
        else float(np.mean(component_r2)),
        "mae_per_component": np.abs(error).mean(axis=0).tolist(),
        "rmse_per_component": np.sqrt(np.square(error).mean(axis=0)).tolist(),
        "r2_per_component": component_r2,
    }


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def evaluate_folds(config_path, checkpoint_dir, data_root, output_path, device="gpu:0"):
    paddle.set_device(device)
    config = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    data_root, checkpoint_dir, output_path = map(
        Path, (data_root, checkpoint_dir, output_path)
    )
    source_path = data_root / "output_binary_with_inf_all.csv"
    source_hash = sha256(source_path)
    source = pd.read_csv(source_path, low_memory=False)
    report = {
        "phase": "evaluating",
        "paddle": paddle.__version__,
        "python": platform.python_version(),
        "device": paddle.get_device(),
        "source_sha256": source_hash,
        "config_sha256": sha256(config_path),
        "training_performed": False,
        "provenance_limit": (
            "Historical README claims comp_inter/seed=2021. Original per-fold "
            "training logs and indices have not been recovered."
        ),
        "results": [],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def save():
        temporary = output_path.with_suffix(".tmp.json")
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        temporary.replace(output_path)

    indices = []
    columns = ["solv1", "solv2", "solv1_x", "solv1_gamma", "solv2_gamma"]
    save()
    for fold in range(5):
        report["current_fold"] = fold
        save()
        split_dir = data_root / f"fold_{fold}"
        manifest_path = split_dir / "split.json"
        manifest = json.loads(manifest_path.read_text())
        if (
            manifest["source_sha256"] != source_hash
            or manifest["fold"] != fold
            or manifest["seed"] != 2021
            or manifest["mode"] != "comp_inter"
        ):
            raise ValueError(f"Fold {fold} does not match the documented source/split.")
        indices.extend(manifest["val_indices"])
        val_path = split_dir / "val.csv"
        actual = pd.read_csv(val_path, low_memory=False)[columns]
        expected = source.iloc[manifest["val_indices"]][columns].reset_index(drop=True)
        pd.testing.assert_frame_equal(
            actual, expected, check_exact=False, rtol=1e-12, atol=1e-12
        )
        val_config = config["Dataset"]["val"]
        params = val_config["dataset"]["__init_params__"]
        params["path"] = str(val_path)
        params["solvent_list_path"] = str(data_root / "solvent_list.csv")
        loader = build_dataloader(val_config)
        model = build_model(config["Model"])
        checkpoint = checkpoint_dir / f"gegnn_binary_gamma_fold{fold}_best.pdparams"
        state = paddle.load(str(checkpoint))
        current = model.state_dict()
        if set(state) != set(current) or any(
            list(state[k].shape) != list(current[k].shape) for k in current
        ):
            raise ValueError(
                f"Fold {fold} checkpoint structure differs from the model."
            )
        model.set_state_dict(state)
        model.eval()
        predictions, targets = [], []
        start = time.perf_counter()
        for batch in loader:
            predictions.append(model.predict(batch)[model.property_name].numpy())
            targets.append(np.asarray(batch["gamma"]))
        predictions, targets = np.concatenate(predictions), np.concatenate(targets)
        if len(targets) != manifest["counts"]["val"]:
            raise ValueError("Evaluation dropped validation records.")
        result = {
            "fold": fold,
            "samples": len(targets),
            **prediction_metrics(predictions, targets),
            "seconds": time.perf_counter() - start,
            "checkpoint_sha256": sha256(checkpoint),
            "split_manifest_sha256": sha256(manifest_path),
            "validation_csv_sha256": sha256(val_path),
        }
        report["results"].append(result)
        save()
        print(json.dumps(result), flush=True)
    if sorted(indices) != list(range(len(source))):
        raise ValueError("The five validation folds do not partition the source rows.")
    report["summary"] = {}
    for metric in ["mae", "rmse", "r2_uniform_average"]:
        values = [x[metric] for x in report["results"]]
        report["summary"][metric] = {
            "mean": float(np.mean(values)),
            "std_population": float(np.std(values)),
        }
    report["phase"] = "complete"
    save()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config_path",
        default=str(
            Path(__file__).resolve().parents[1]
            / "property_prediction/configs/gegnn/gegnn_binary_activity.yaml"
        ),
    )
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--data_root", default="data/binary_activity")
    parser.add_argument("--output_path", default="gegnn_fold_evaluation.json")
    parser.add_argument("--device", default="gpu:0")
    args = parser.parse_args()
    evaluate_folds(**vars(args))


if __name__ == "__main__":
    main()

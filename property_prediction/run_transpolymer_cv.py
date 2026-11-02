# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

"""Run TransPolymer's original five-fold protocol using train.py per fold."""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

import numpy as np
import paddle
import pandas as pd
from omegaconf import OmegaConf
from sklearn.model_selection import KFold

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ppmat.datasets.transpolymer_dataset import TransPolymerDataset


def _get_dataset_config(config):
    return config["Dataset"]["train"]["dataset"]["__init_params__"]


def _find_fold_output(output_dir: Path, existing_outputs: set[Path]) -> Path:
    """Find the output directory created by the current train.py invocation."""
    outputs = set(output_dir.parent.glob(f"{output_dir.name}_t_*"))
    created_outputs = sorted(outputs - existing_outputs)
    if len(created_outputs) != 1:
        raise RuntimeError(
            f"Expected exactly one new train.py output for {output_dir}, found "
            f"{created_outputs}."
        )
    return created_outputs[0]


def _write_summary(summary_path: Path, rows: list[dict]) -> None:
    fieldnames = ["fold", "best_metric", "output_dir"]
    with open(summary_path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the original TransPolymer K-fold protocol with train.py."
    )
    parser.add_argument("-c", "--config", required=True, help="CV config path.")
    parser.add_argument(
        "--output-root",
        default=None,
        help="Directory for generated split CSVs and fold outputs.",
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=None,
        help="Run only one one-based fold, useful for a smoke test.",
    )
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    config = OmegaConf.load(config_path)
    cv_cfg = config.get("TransPolymerCV")
    if cv_cfg is None:
        print(
            "[warning] TransPolymerCV is not configured. Using the original "
            "protocol: KFold(n_splits=5, shuffle=True, random_state=1)."
        )
        cv_cfg = {}

    n_splits = int(cv_cfg.get("n_splits", 5))
    shuffle = bool(cv_cfg.get("shuffle", True))
    random_state = int(cv_cfg.get("random_state", 1))
    dataset_cfg = _get_dataset_config(config)
    source_path = Path(TransPolymerDataset.resolve_path(dataset_cfg["path"]))
    dataframe = pd.read_csv(source_path)

    output_root = Path(args.output_root).resolve() if args.output_root else Path(
        config["Trainer"]["output_dir"] + "_cv"
    ).resolve()
    split_root = output_root / "splits"
    config_root = output_root / "configs"
    split_root.mkdir(parents=True, exist_ok=True)
    config_root.mkdir(parents=True, exist_ok=True)

    splitter = KFold(
        n_splits=n_splits,
        shuffle=shuffle,
        random_state=random_state if shuffle else None,
    )
    selected_fold = args.fold
    if selected_fold is not None and not 1 <= selected_fold <= n_splits:
        raise ValueError(f"--fold must be between 1 and {n_splits}.")

    summary_rows = []
    for fold, (train_index, val_index) in enumerate(splitter.split(dataframe), start=1):
        if selected_fold is not None and fold != selected_fold:
            continue

        fold_dir = split_root / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        train_path = fold_dir / "train.csv"
        val_path = fold_dir / "val.csv"
        dataframe.iloc[train_index].to_csv(train_path, index=False)
        dataframe.iloc[val_index].to_csv(val_path, index=False)

        fold_config = OmegaConf.load(config_path)
        for split_name, split_path in (
            ("train", train_path),
            ("val", val_path),
            ("test", val_path),
        ):
            fold_config["Dataset"][split_name]["dataset"]["__init_params__"][
                "path"
            ] = str(split_path)
        output_dir = output_root / f"fold_{fold}"
        fold_config["Trainer"]["output_dir"] = str(output_dir)
        del fold_config["TransPolymerCV"]
        generated_config_path = config_root / f"fold_{fold}.yaml"
        OmegaConf.save(fold_config, generated_config_path)

        existing_outputs = set(output_dir.parent.glob(f"{output_dir.name}_t_*"))
        subprocess.run(
            [
                sys.executable,
                "-m",
                "property_prediction.train",
                "-c",
                str(generated_config_path),
            ],
            cwd=PROJECT_ROOT,
            check=True,
        )

        fold_output = _find_fold_output(output_dir, existing_outputs)
        state = paddle.load(str(fold_output / "checkpoints" / "best.pdstates"))
        summary_rows.append(
            {
                "fold": fold,
                "best_metric": state.get("best_metric"),
                "output_dir": str(fold_output),
            }
        )

    metrics = np.asarray(
        [row["best_metric"] for row in summary_rows], dtype="float64"
    )
    summary_rows.extend(
        [
            {
                "fold": "mean",
                "best_metric": float(np.mean(metrics)),
                "output_dir": "",
            },
            {
                "fold": "std",
                "best_metric": float(np.std(metrics)),
                "output_dir": "",
            },
        ]
    )
    _write_summary(output_root / "cross_validation_summary.csv", summary_rows)
    print(f"Saved cross-validation summary to {output_root / 'cross_validation_summary.csv'}")


if __name__ == "__main__":
    main()

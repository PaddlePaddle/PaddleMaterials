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

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from ppmat.datasets.gegnn_dataset import BinaryActivityDataset
from ppmat.utils import download


def split_binary_activity(data, mode="comp_inter", fold=0, num_splits=5, seed=2021):
    """Return row indices using the reference stratified five-fold protocols.

    ``comp_inter`` stratifies mixture compositions by TPSA class.
    ``system_extra`` holds out complete ordered solvent pairs.
    Source: https://git.rwth-aachen.de/avt-svt/public/GDI-NN
    """
    if not 0 <= fold < num_splits:
        raise ValueError("fold must be in [0, num_splits).")
    splitter = StratifiedKFold(n_splits=num_splits, shuffle=True, random_state=seed)
    if mode == "comp_inter":
        groups = data
    elif mode == "system_extra":
        groups = data.groupby(["solv1", "solv2"], as_index=False)[
            "tpsa_binary_avg"
        ].mean()
    else:
        raise ValueError(f"Unsupported split mode: {mode}")
    train, val = list(splitter.split(np.arange(len(groups)), groups.tpsa_binary_avg))[
        fold
    ]
    if mode == "system_extra":
        pair_index = pd.MultiIndex.from_frame(data[["solv1", "solv2"]])
        train_pairs = pd.MultiIndex.from_frame(groups.iloc[train][["solv1", "solv2"]])
        val_pairs = pd.MultiIndex.from_frame(groups.iloc[val][["solv1", "solv2"]])
        train = np.flatnonzero(pair_index.isin(train_pairs))
        val = np.flatnonzero(pair_index.isin(val_pairs))
    return train, val


def prepare_splits(
    input_path, output_dir, mode="comp_inter", fold=0, num_splits=5, seed=2021
):
    """Write train/validation CSVs and a source/split manifest."""
    input_path, output_dir = Path(input_path), Path(output_dir)
    if not input_path.exists():
        if input_path.name not in BinaryActivityDataset.md5:
            raise FileNotFoundError(input_path)
        download.get_path_from_url(
            BinaryActivityDataset.url + input_path.name,
            str(input_path.parent),
            md5sum=BinaryActivityDataset.md5[input_path.name],
            decompress=False,
        )
    data = pd.read_csv(input_path, low_memory=False)
    train, val = split_binary_activity(data, mode, fold, num_splits, seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    targets = [output_dir / name for name in ("train.csv", "val.csv", "split.json")]
    if any(path.exists() for path in targets):
        raise FileExistsError(
            f"Split files already exist in {output_dir}; use a new output directory."
        )
    data.iloc[train].to_csv(targets[0], index=False)
    data.iloc[val].to_csv(targets[1], index=False)
    manifest = {
        "source_file": input_path.name,
        "source_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        "mode": mode,
        "fold": fold,
        "num_splits": num_splits,
        "seed": seed,
        "counts": {"train": len(train), "val": len(val)},
        "train_indices": train.tolist(),
        "val_indices": val.tolist(),
    }
    targets[2].write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(
        description="Prepare GE-GNN reference dataset splits"
    )
    parser.add_argument(
        "--input_path", default="data/binary_activity/output_binary_with_inf_all.csv"
    )
    parser.add_argument("--output_dir", default="data/binary_activity/fold_0")
    parser.add_argument(
        "--mode", choices=["comp_inter", "system_extra"], default="comp_inter"
    )
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--num_splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2021)
    args = parser.parse_args()
    result = prepare_splits(**vars(args))
    print(
        json.dumps(
            {key: result[key] for key in ("mode", "fold", "seed", "counts")}, indent=2
        )
    )


if __name__ == "__main__":
    main()

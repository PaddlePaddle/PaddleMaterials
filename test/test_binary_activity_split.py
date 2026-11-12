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

import hashlib
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ppmat.datasets.split_binary_activity_data import prepare_splits
from ppmat.datasets.split_binary_activity_data import split_binary_activity


def activity_records():
    rows = []
    for group in range(3):
        for pair in range(5):
            for x in [0.2, 0.5, 0.8]:
                rows.append(
                    {
                        "solv1": f"A{group}_{pair}",
                        "solv2": "B",
                        "solv1_x": x,
                        "tpsa_binary_avg": group,
                        "solv1_gamma": 0.1,
                        "solv2_gamma": -0.2,
                    }
                )
    return pd.DataFrame(rows)


@pytest.mark.parametrize("mode", ["comp_inter", "system_extra"])
def test_reference_folds_partition_data_and_are_reproducible(mode):
    data = activity_records()
    held_out = []
    for fold in range(5):
        train, val = split_binary_activity(data, mode=mode, fold=fold)
        repeated = split_binary_activity(data, mode=mode, fold=fold)
        np.testing.assert_array_equal(train, repeated[0])
        np.testing.assert_array_equal(val, repeated[1])
        assert not set(train).intersection(val)
        assert sorted([*train, *val]) == list(range(len(data)))
        assert len(val) == 9
        assert set(data.iloc[val].tpsa_binary_avg) == {0, 1, 2}
        if mode == "system_extra":
            assert not set(
                zip(data.iloc[train].solv1, data.iloc[train].solv2)
            ).intersection(zip(data.iloc[val].solv1, data.iloc[val].solv2))
        held_out.extend(val)
    assert sorted(held_out) == list(range(len(data)))


def test_manifest_reconstructs_saved_rows_and_protects_existing_files():
    with tempfile.TemporaryDirectory() as root:
        source = Path(root) / "source.csv"
        output = Path(root) / "fold_0"
        data = activity_records()
        data.to_csv(source, index=False)
        result = prepare_splits(source, output)
        manifest = json.loads((output / "split.json").read_text())
        assert (
            manifest["source_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
        )
        assert manifest["counts"] == {"train": 36, "val": 9}
        pd.testing.assert_frame_equal(
            pd.read_csv(output / "train.csv"),
            data.iloc[result["train_indices"]].reset_index(drop=True),
        )
        pd.testing.assert_frame_equal(
            pd.read_csv(output / "val.csv"),
            data.iloc[result["val_indices"]].reset_index(drop=True),
        )
        before = (output / "train.csv").read_bytes()
        with pytest.raises(FileExistsError):
            prepare_splits(source, output, seed=42)
        assert (output / "train.csv").read_bytes() == before


@pytest.mark.parametrize("fold", [-1, 5])
def test_invalid_fold_is_rejected(fold):
    with pytest.raises(ValueError, match="fold"):
        split_binary_activity(activity_records(), fold=fold)

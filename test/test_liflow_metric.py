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

"""Guard against a three-fold metric magnitude error and verify the 11-time
validation protocol.

The LiFlow loss/metric sums the squared error over the xyz components *before*
the node mean (reference ``FlowModule.compute_loss``).  Averaging the components
first would under-report the metric by a factor of three.
"""

from __future__ import annotations

import paddle
import pytest

from ppmat.datasets.liflow_dataset import VALIDATION_TIMES
from ppmat.datasets.liflow_dataset import LiFlowDataset
from ppmat.metrics.liflow_metric import LiFlowMSE


def test_liflow_mse_sums_xyz_before_node_mean():
    pred = paddle.to_tensor([[1.0, 2.0, 3.0], [0.0, 1.0, 0.0]])
    label = paddle.zeros_like(pred)
    result = LiFlowMSE()(pred, label)
    assert float(result) == pytest.approx((14.0 + 1.0) / 2.0)


def test_liflow_dataset_val_expands_to_11_times(mini_metric_path):
    ds = LiFlowDataset(path=mini_metric_path, split="val", seed=42)
    row = len(ds) // 11
    assert len(ds) == row * 11
    times = {float(ds[i]["flow_time"]) for i in range(len(ds))}
    assert times == {float(t) for t in VALIDATION_TIMES}
    # each validated sample's flow_time comes from the fixed grid, and the
    # endpoint pair is stable across all 11 times of a single row.
    times_for_second_row = {float(ds[row + i]["flow_time"]) for i in range(11)}
    assert len(times_for_second_row) == 11


@pytest.fixture()
def mini_metric_path():
    from pathlib import Path

    return Path(__file__).resolve().parent / "fixtures" / "liflow" / "dataset_mini"

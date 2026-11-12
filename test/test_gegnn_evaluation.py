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

import runpy
from pathlib import Path

import numpy as np
import pytest

metrics = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "tests/evaluate_gegnn_folds.py")
)["prediction_metrics"]


def test_binary_metrics_use_uniform_component_r2():
    result = metrics([[1, 1], [1, 5]], [[0, 2], [2, 4]])
    assert result["mae"] == 1
    assert result["rmse"] == 1
    assert result["r2_uniform_average"] == 0
    assert result["r2_per_component"] == [0, 0]


def test_constant_targets_do_not_emit_nan_json_values():
    result = metrics([[1, 2], [1, 2]], [[1, 2], [1, 2]])
    assert result["mae"] == 0
    assert result["r2_uniform_average"] is None
    assert result["r2_per_component"] == [None, None]


def test_invalid_predictions_are_rejected():
    with pytest.raises(ValueError):
        metrics([[np.nan, 1]], [[0, 1]])
    with pytest.raises(ValueError):
        metrics([[1, 2]], [[0, 1], [2, 3]])

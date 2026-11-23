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

import importlib.util
from pathlib import Path

import pytest

PREDICT_PATH = Path(__file__).parents[1] / "electronic_structure" / "predict.py"
SPEC = importlib.util.spec_from_file_location(
    "electronic_structure_predict", PREDICT_PATH
)
predict_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(predict_module)
parse_args = predict_module.parse_args


@pytest.mark.parametrize(
    "input_format",
    ["mol", "xyz", "cif", "cube", "chgcar", "json"],
)
def test_predict_accepts_input_format(input_format):
    args, overrides = parse_args(
        [
            "--model_name",
            "infgcn_qm9",
            "--input_path",
            "sample",
            "--input_format",
            input_format,
        ]
    )

    assert args.input_path == "sample"
    assert args.input_format == input_format
    assert overrides == []


def test_predict_rejects_unsupported_input_format():
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--model_name",
                "infgcn_qm9",
                "--input_path",
                "sample",
                "--input_format",
                "unsupported",
            ]
        )

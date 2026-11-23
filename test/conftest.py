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

"""Test-wide fixtures."""

from __future__ import annotations

import paddle
import pytest


@pytest.fixture(autouse=True)
def cpu_device():
    """Pin every test to CPU and restore the caller's device afterwards."""

    original_device = paddle.get_device()
    paddle.set_device("cpu")
    yield
    paddle.set_device(original_device)

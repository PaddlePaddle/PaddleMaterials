# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from ppmat.datasets.mp20_dataset import MP20Dataset


class MPTS52Dataset(MP20Dataset):
    """MPTS-52 crystals in the MP20 CSV schema."""

    name = "mpts_52"
    url = "https://paddle-org.bj.bcebos.com/paddlematerials/datasets/mpts_52/mpts_52.zip"  # noqa
    md5 = "d9cf6b0fa281c3eb06b1aa2edb53cb76"

    def __init__(self, path: str = "./data/mpts_52/train.csv", **kwargs):
        super().__init__(path=path, **kwargs)

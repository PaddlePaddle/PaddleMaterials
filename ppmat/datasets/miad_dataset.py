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

import numpy as np
from paddle.io import Dataset

from ppmat.datasets.custom_data_type import ConcatData


class MiADSamplingDataset(Dataset):
    """Provide the fixed candidate capacity used by MiAD generation."""

    def __init__(self, total_num: int, num_atoms: int):
        super().__init__()
        self.total_num = total_num
        self.num_atoms = num_atoms

    def __len__(self):
        return self.total_num

    def __getitem__(self, index):
        return {
            "structure_array": {
                "num_atoms": ConcatData(np.array([self.num_atoms], dtype="int64")),
            },
            "id": index,
        }

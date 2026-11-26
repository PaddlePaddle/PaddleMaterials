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


class MatInventDataset(MP20Dataset):
    """MP20 anchors with an ordered diffusion-timestep index for each RL round.

    Args:
        finetune_cfg (dict): Timestep, accumulation and fine-tuning pass counts.
        **kwargs: Arguments forwarded to MP20Dataset.
    """

    def __init__(self, finetune_cfg, **kwargs):
        super().__init__(**kwargs)
        self.finetune_cfg = finetune_cfg
        timesteps = finetune_cfg["timesteps"]
        accumulation = finetune_cfg["accum_steps"]
        self.padded_timesteps = (
            (timesteps + accumulation - 1) // accumulation * accumulation
        )
        self.num_steps = self.padded_timesteps * finetune_cfg["epochs"]

    def __len__(self):
        return self.num_steps

    def __getitem__(self, index):
        data = super().__getitem__(index % self.num_samples)
        data["rl_index"] = index
        return data

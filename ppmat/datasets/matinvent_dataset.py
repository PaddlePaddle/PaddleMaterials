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

from paddle.io import Dataset


class MatInventStepDataset(Dataset):
    """Ordered timestep indices for one RL round, independent of crystal data.

    Args:
        timesteps (int): Diffusion timesteps in each fine-tuning pass.
        epochs (int): Fine-tuning passes in each RL round.
        accum_steps (int): Microbatches in each optimizer update.
    """

    def __init__(self, timesteps: int, epochs: int, accum_steps: int):
        super().__init__()
        if min(timesteps, epochs, accum_steps) <= 0:
            raise ValueError("Timestep, pass and accumulation counts must be positive")
        self.timesteps = timesteps
        self.epochs = epochs
        self.accum_steps = accum_steps
        self.padded_timesteps = (
            (timesteps + accum_steps - 1) // accum_steps * accum_steps
        )
        self.num_steps = self.padded_timesteps * epochs

    def __len__(self):
        return self.num_steps

    def __getitem__(self, index):
        return {"rl_index": index}

# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

# Adapted from https://github.com/divelab/AIRS (OpenMat/HIENet)

import paddle


class DenoisingBlock(paddle.nn.Module):
    """
    Force Block: Output block computing the per atom forces

    Args:
        num_channels (int):         Number of channels
        num_sphere_samples (int):   Number of samples used to approximate the integral on the sphere
        act (function):             Non-linear activation function
    """

    def __init__(self, num_channels, num_sphere_samples, act):
        super(ForceBlock, self).__init__()
        self.num_channels = num_channels
        self.num_sphere_samples = num_sphere_samples
        self.act = act
        self.fc1 = paddle.compat.nn.Linear(self.num_channels, self.num_channels)
        self.fc2 = paddle.compat.nn.Linear(self.num_channels, self.num_channels)
        self.fc3 = paddle.compat.nn.Linear(self.num_channels, 1, bias=False)

    def forward(self, x_pt, sphere_points):
        x_pt = self.act(self.fc1(x_pt))
        x_pt = self.act(self.fc2(x_pt))
        x_pt = self.fc3(x_pt)
        x_pt = x_pt.reshape(-1, self.num_sphere_samples, 1)
        forces = x_pt * sphere_points.reshape(1, self.num_sphere_samples, 3)
        forces = paddle.sum(forces, dim=1) / self.num_sphere_samples
        return forces

# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle

from ppmat.datasets.graph_utils.infgcn_graph_utils import radius


class AtomGridRadiusGraphConverter:
    """Build PGL-style ``[atom, grid]`` edge rows for one grid chunk."""

    def __init__(self, cutoff: float, max_num_neighbors: int = 32) -> None:
        self.cutoff = cutoff
        self.max_num_neighbors = max_num_neighbors

    def __call__(
        self,
        atom_coord: paddle.Tensor,
        grid_coord: paddle.Tensor,
        atom_batch: paddle.Tensor,
        grid_batch: paddle.Tensor,
    ) -> paddle.Tensor:
        grid_dst, atom_src = radius(
            atom_coord,
            grid_coord,
            self.cutoff,
            atom_batch,
            grid_batch,
            self.max_num_neighbors,
        )
        return paddle.stack([atom_src, grid_dst], axis=1)

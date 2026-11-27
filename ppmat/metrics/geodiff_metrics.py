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

import paddle.distributed as dist

from ppmat.metrics.streaming_base import StreamingMetricBase


class GeoDiffValidationLoss(StreamingMetricBase):
    """Aggregate GeoDiff validation losses over atoms, not batches."""

    _LOSS_KEYS = ("loss", "loss_global", "loss_local")

    def __init__(self):
        if dist.get_world_size() != 1:
            raise ValueError("GeoDiff validation requires a single process.")
        self.reset()

    def reset(self):
        self._totals = dict.fromkeys(self._LOSS_KEYS, 0.0)
        self._atom_count = 0

    def update_step(self, *, result, batch, stage):
        if stage != "eval":
            return
        stats = result["loss_stats"]
        for key in self._LOSS_KEYS:
            self._totals[key] += stats[key].item()
        self._atom_count += batch["atom_types"].shape[0]

    def compute_epoch(self, *, stage):
        if stage != "eval":
            return {}

        totals, atom_count = self._totals, self._atom_count
        # Step validation does not reset shared streaming metrics.
        self.reset()
        if not atom_count:
            return {}
        return {f"geodiff_{key}": total / atom_count for key, total in totals.items()}

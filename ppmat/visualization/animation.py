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

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import imageio.v2 as imageio
import numpy as np


class AnimationWriter:
    """Write image sequences as animations."""

    @staticmethod
    def save_gif(
        frames: Sequence,
        path: str | Path,
        *,
        duration: float = 200,  # milliseconds (imageio's unit)
        hold_last: int = 10,
    ) -> Path:
        arrays = [np.asarray(frame) for frame in frames]
        if not arrays:
            raise ValueError("frames must not be empty.")
        if hold_last > 0:
            arrays.extend([arrays[-1]] * int(hold_last))
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(path, arrays, duration=duration)
        return path

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

from __future__ import annotations

import abc

import numpy as np
from ase import units

__all__ = [
    "Prior",
    "NormalPrior",
    "UniformScaleNormalPrior",
    "MaxwellBoltzmannPrior",
    "AdaptiveMaxwellBoltzmannPrior",
]


def _get_rng(default_rng, rng=None):
    """Return the RNG used to draw a sample.

    ``rng`` is optional so that callers (e.g. the LiFlow dataset) can pass a
    per-access generator.  Without it the prior falls back to its own seeded
    generator, which keeps standalone usage reproducible.
    """
    return default_rng if rng is None else rng


class Prior(abc.ABC):
    @abc.abstractmethod
    def sample(self, shape: tuple, rng=None) -> np.ndarray:
        ...


class NormalPrior(Prior):
    def __init__(self, scale: float = 1.0, seed: int = 42):
        self.scale = scale
        self.rng = np.random.default_rng(seed)

    def sample(self, shape: tuple, rng=None) -> np.ndarray:
        rng = _get_rng(self.rng, rng)
        return rng.normal(scale=self.scale, size=shape)


class UniformScaleNormalPrior(Prior):
    def __init__(self, scale: float = 1.0, seed: int = 42):
        self.scale = scale
        self.rng = np.random.default_rng(seed)

    def sample(self, shape: tuple, rng=None) -> np.ndarray:
        rng = _get_rng(self.rng, rng)
        scale = rng.uniform(0, self.scale, size=shape[0])
        return rng.normal(scale=scale[:, None], size=shape)


class MaxwellBoltzmannPrior(Prior):
    def __init__(self, scale: float = 1.0, seed: int = 42):
        self.scale = scale
        self.rng = np.random.default_rng(seed)

    def sample(
        self, temperature: float, masses: np.ndarray, shape: tuple, rng=None
    ) -> np.ndarray:
        rng = _get_rng(self.rng, rng)
        scale = self.scale * np.sqrt(units.kB * float(temperature) / masses)
        return rng.normal(scale=scale[:, None], size=shape)


class AdaptiveMaxwellBoltzmannPrior(Prior):
    def __init__(
        self,
        scale: list[list[float]] | None = None,
        seed: int = 42,
    ):
        # ``[Li_scale, frame_scale]`` per prior class; defaults mirror train.yaml.
        self.scale = scale if scale is not None else [[1.0, 10.0], [0.316, 3.16]]
        self.rng = np.random.default_rng(seed)

    def sample(
        self,
        temperature: float,
        atomic_numbers: np.ndarray,
        masses: np.ndarray,
        scale_Li_index: int,
        scale_frame_index: int,
        shape: tuple,
        rng=None,
    ) -> np.ndarray:
        rng = _get_rng(self.rng, rng)
        li_scale = self.scale[0][scale_Li_index]
        frame_scale = self.scale[1][scale_frame_index]
        prefactor = np.where(atomic_numbers == 3, li_scale, frame_scale)
        scale = prefactor * np.sqrt(units.kB * float(temperature) / masses)
        return rng.normal(scale=scale[:, None], size=shape)

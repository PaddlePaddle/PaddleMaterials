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

import numpy as np

from ppmat.datasets.liflow_dataset import LiFlowTrajectoryDataset
from ppmat.models.liflow.prior import MaxwellBoltzmannPrior
from ppmat.models.liflow.prior import Prior
from ppmat.models.liflow.prior import UniformScaleNormalPrior

__all__ = ["NoisedFrameDataset"]


class NoisedFrameDataset(LiFlowTrajectoryDataset):
    """Corrector dataset (reference ``NoisedFrameDataset``).

    Unlike the propagator dataset, a corrector sample is a *single* frame: the
    clean frame at a randomly drawn time is perturbed with
    :class:`UniformScaleNormalPrior` and the model has to transport the noisy
    frame back to the clean one.  Only the clean frame is real data; ``end`` is
    the clean frame and ``start`` is ``clean + noise``.  Following the reference
    implementation::

        time  ~ randint(t_start + time_delay_steps, t_end)
        noise ~ UniformScaleNormalPrior(scale=0.25)
        start = clean + noise
        end   = clean
        prior = MaxwellBoltzmannPrior(scale=0.1)

    The neighbour list is built from the noisy positions, because that is the
    geometry the model actually sees at inference time.  The shared
    flow-matching target of :class:`LiFlowTrajectoryDataset` then gives::

        x_0 = start + prior
        velocity = end - x_0

    Time, noise and prior are sampled on every access (train split); the
    validation split draws them deterministically per row so that the
    ``num_validation_times`` flow times of one row share a single sample.

    Args:
        path: Dataset directory (see :class:`LiFlowTrajectoryDataset`).
        split: ``"train"``, ``"val"`` or ``"test"``.
        time_delay_steps: ``skip_steps`` of the reference: the frame is drawn
            from ``[t_start + time_delay_steps, t_end)``.
        cutoff: Neighbour-list cutoff in Angstrom.
        periodic: Whether to use periodic boundary conditions.
        seed: Seed for the deterministic validation draw.
        in_memory: Cache the raw trajectory arrays in memory.
        num_validation_times: Flow times per validation sample.
        prior: Optional prior override (default ``MaxwellBoltzmannPrior(0.1)``).
        noise: Optional noise override (default
            ``UniformScaleNormalPrior(0.25)``).
        url: Optional archive URL (``md5`` required).  No default URL.
        md5: Archive checksum.
    """

    def __init__(
        self,
        path: str,
        split: str = "train",
        *,
        time_delay_steps: int = 100,
        cutoff: float = 5.0,
        periodic: bool = True,
        seed: int = 42,
        in_memory: bool = False,
        num_validation_times: int = 11,
        prior: Prior | None = None,
        noise: Prior | None = None,
        url: str | None = None,
        md5: str | None = None,
    ):
        self.noise = noise if noise is not None else UniformScaleNormalPrior(scale=0.25)
        super().__init__(
            path=path,
            split=split,
            time_delay_steps=time_delay_steps,
            cutoff=cutoff,
            periodic=periodic,
            seed=seed,
            in_memory=in_memory,
            num_validation_times=num_validation_times,
            prior=prior,
            url=url,
            md5=md5,
        )

    def _default_prior(self) -> Prior:
        return MaxwellBoltzmannPrior(scale=0.1)

    def _build_pair(
        self,
        *,
        name: str,
        row,
        trajectory,
        atomic_numbers: np.ndarray,
        masses: np.ndarray,
        temperature: float,
        rng: np.random.Generator,
    ) -> dict:
        t_start = int(row["t_start"])
        t_end = int(row["t_end"])
        if t_end - t_start <= self.time_delay_steps:
            raise ValueError(
                f"{name}: trajectory window [{t_start}, {t_end}) is shorter "
                f"than time_delay_steps={self.time_delay_steps}."
            )
        time = int(rng.integers(t_start + self.time_delay_steps, t_end))
        clean_positions = np.asarray(trajectory[time], dtype=np.float32)
        noise = self.noise.sample(shape=clean_positions.shape, rng=rng)
        start_positions = clean_positions + np.asarray(noise, dtype=np.float32)
        end_positions = clean_positions
        prior = self.prior.sample(
            temperature=temperature,
            masses=masses,
            shape=start_positions.shape,
            rng=rng,
        )
        return {
            "start_positions": start_positions,
            "end_positions": end_positions,
            "prior": np.asarray(prior, dtype=np.float32),
            "neighbor_positions": [start_positions],
        }

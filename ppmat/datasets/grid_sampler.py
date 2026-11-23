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


from __future__ import annotations

import hashlib
import numbers

import numpy as np


class DensityGridSampler:
    """Sample a fixed number of points from a density grid."""

    def __init__(
        self,
        *,
        n_samples: int,
        sampling_mode: str = "uniform",
        uniform_random_offset: bool = False,
        sampling_seed: int | None = None,
        resample_each_epoch: bool = True,
        importance_threshold: float = 1e-5,
        importance_ratio: float = 0.8,
        extreme_threshold: float | None = None,
        extreme_ratio: float = 0.05,
    ):
        if isinstance(n_samples, bool) or not isinstance(n_samples, numbers.Integral):
            raise TypeError("n_samples must be a positive integer.")
        if n_samples <= 0:
            raise ValueError("n_samples must be a positive integer.")
        self.n_samples = int(n_samples)

        sampling_mode = sampling_mode.lower()
        if sampling_mode not in {"uniform", "random", "importance"}:
            raise ValueError(
                f"Unsupported sampling_mode '{sampling_mode}'. "
                "Use 'uniform', 'random', or 'importance'."
            )
        self.sampling_mode = sampling_mode
        self.uniform_random_offset = bool(uniform_random_offset)

        self.importance_threshold = importance_threshold
        self.importance_ratio = float(importance_ratio)
        self.extreme_threshold = extreme_threshold
        self.extreme_ratio = float(extreme_ratio)
        if not 0.0 <= self.importance_ratio <= 1.0:
            raise ValueError("importance_ratio must be between 0 and 1.")
        if not 0.0 <= self.extreme_ratio <= 1.0:
            raise ValueError("extreme_ratio must be between 0 and 1.")

        if sampling_seed is not None and (
            isinstance(sampling_seed, bool)
            or not isinstance(sampling_seed, numbers.Integral)
        ):
            raise TypeError("sampling_seed must be an integer or None.")
        if not isinstance(resample_each_epoch, bool):
            raise TypeError("resample_each_epoch must be a boolean.")
        self.sampling_seed = None if sampling_seed is None else int(sampling_seed)
        self.resample_each_epoch = resample_each_epoch
        self.uses_random_sampling = (
            self.sampling_mode == "importance"
            or self.sampling_mode == "random"
            or self.uniform_random_offset
        )
        if (
            self.uses_random_sampling
            and not self.resample_each_epoch
            and self.sampling_seed is None
        ):
            raise ValueError(
                "sampling_seed is required when resample_each_epoch is False."
            )

    def _rng_for_index(self, index):
        """Build a per-sample random generator."""

        if not self.uses_random_sampling:
            return None
        if self.resample_each_epoch:
            epoch_seed = np.random.randint(0, 2**63 - 1)
            seed_material = repr((self.sampling_seed, epoch_seed, index)).encode(
                "utf-8"
            )
            seed = int.from_bytes(
                hashlib.sha256(seed_material).digest()[:8],
                "little",
            )
            return np.random.default_rng(seed)
        seed_material = repr((self.sampling_seed, index)).encode("utf-8")
        seed = int.from_bytes(hashlib.sha256(seed_material).digest()[:8], "little")
        return np.random.default_rng(seed)

    def _importance_indices(self, density, total, rng):
        target_samples = self.n_samples
        # Rank a grid point by its largest absolute channel value so that
        # multi-channel fields share the single-channel path.
        dense_vals = np.abs(density)
        if dense_vals.ndim > 1:
            dense_vals = dense_vals.reshape(total, -1).max(axis=1)
        high_mask = dense_vals >= float(self.importance_threshold)
        high_idx = np.flatnonzero(high_mask)

        if self.extreme_threshold is None:
            extreme_idx = np.array([], dtype=np.intp)
            mid_idx = high_idx
        else:
            # Keep extreme a subset of high so the quotas nest.
            extreme_mask = high_mask & (dense_vals >= self.extreme_threshold)
            extreme_idx = np.flatnonzero(extreme_mask)
            mid_idx = np.flatnonzero(high_mask & ~extreme_mask)

        high_quota = min(
            target_samples,
            max(0, int(target_samples * self.importance_ratio)),
        )
        extreme_quota = min(
            high_quota,
            max(0, int(target_samples * self.extreme_ratio)),
        )

        extreme_take = min(len(extreme_idx), extreme_quota)
        indices_extreme = (
            rng.choice(extreme_idx, extreme_take, replace=False)
            if extreme_take > 0
            else np.array([], dtype=np.intp)
        )

        mid_take = min(
            len(mid_idx),
            max(0, high_quota - len(indices_extreme)),
        )
        indices_mid = (
            rng.choice(mid_idx, mid_take, replace=False)
            if mid_take > 0
            else np.array([], dtype=np.intp)
        )

        selected = np.concatenate([indices_extreme, indices_mid])
        high_needed = min(
            max(0, high_quota - len(selected)),
            max(0, len(high_idx) - len(selected)),
        )
        if high_needed > 0:
            selected_mask = np.zeros(total, dtype=bool)
            selected_mask[selected.astype(np.intp)] = True
            remaining_high = high_idx[~selected_mask[high_idx]]
            indices_high = rng.choice(remaining_high, high_needed, replace=False)
            selected = np.concatenate([selected, indices_high])

        remaining = target_samples - len(selected)
        if remaining <= 0:
            return selected

        taken = np.zeros(total, dtype=bool)
        taken[selected.astype(np.intp)] = True
        low_candidates = np.flatnonzero(~taken)
        if len(low_candidates) == 0:
            low_candidates = np.arange(total)
        indices_low = rng.choice(
            low_candidates, remaining, replace=remaining > len(low_candidates)
        )
        return np.concatenate([selected, indices_low])

    def _uniform_indices(self, total, rng):
        target_samples = self.n_samples
        if not self.uniform_random_offset:
            return np.linspace(0, total - 1, num=target_samples, dtype=int)
        if target_samples <= total:
            # One point per equally sized bin, so points never repeat.
            bin_edges = np.linspace(0, total, num=target_samples + 1, dtype=int)
            return rng.integers(bin_edges[:-1], bin_edges[1:])
        offset = int(rng.integers(0, total))
        strided = np.linspace(0, total, num=target_samples, endpoint=False, dtype=int)
        return (strided + offset) % total

    def indices(self, density, index):
        """Return the sorted grid point indices drawn for one field."""

        total = int(density.shape[0])
        if total == 0:
            raise ValueError("Cannot sample from an empty density field.")
        rng = self._rng_for_index(index)
        if self.sampling_mode == "importance":
            indices = self._importance_indices(density, total, rng)
        elif self.sampling_mode == "uniform":
            indices = self._uniform_indices(total, rng)
        else:
            indices = rng.choice(total, self.n_samples, replace=self.n_samples > total)
        indices.sort()
        return indices

    def __call__(self, data, index):
        """Replace ``density`` and ``grid_coord`` in place with a sampled draw."""

        density = data["density"]
        grid_coord = data["grid_coord"]
        if int(density.shape[0]) != int(grid_coord.shape[0]):
            raise ValueError(
                f"Density length ({int(density.shape[0])}) and grid length "
                f"({int(grid_coord.shape[0])}) must match."
            )
        indices = self.indices(density, index)
        data["density"] = density[indices]
        data["grid_coord"] = grid_coord[indices]
        return data

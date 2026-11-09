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

"""Build ASU crystal fields from packed flat arrays. """

from __future__ import annotations

import numpy as np
from p_tqdm import p_map


class BuildAsuCrystal:
    """Build one ASU crystal field dict from a packed flat array.

    The packed layout (NE = element encoding size, n = num atoms) is defined
    in the ``AsymmetricUnitDataset`` docstring. Archives without the optional
    trailing wyckoff-shape segment are normalized with zero shape indices
    (no shape decomposition). Fields whose order depends on the atom order
    are stably sorted by (wyckoff index, element index), so the built dict is
    the final per-sample representation.

    Args:
        element_encoding_size (int): Size of the element encoding NE.
        num_cpus (int, optional): Number of CPUs used when building a list
            of arrays. Defaults to 1.
    """

    def __init__(self, element_encoding_size: int, num_cpus: int = 1):
        self.element_encoding_size = element_encoding_size
        self.num_cpus = num_cpus

        # Packed-layout offsets that depend on the element encoding size.
        self._idx_lengths = 2 + element_encoding_size
        self._idx_angles = 5 + element_encoding_size
        self._idx_atoms = 8 + element_encoding_size

    @staticmethod
    def build_one(
        flat: np.ndarray,
        idx_lengths: int,
        idx_angles: int,
        idx_atoms: int,
    ) -> dict:
        """Build one ASU crystal field dict from one packed flat array.

        ``np.lexsort`` takes the keys in reverse order, so the last key of
        the tuple is the primary one.
        """
        n = int(flat[0])
        if len(flat) > idx_atoms + 5 * n:
            shape_indices = flat[idx_atoms + 5 * n : idx_atoms + 6 * n].astype(np.int64)
        else:
            shape_indices = np.zeros(n, dtype=np.int64)
        element_indices = flat[idx_atoms : idx_atoms + n].astype(np.int64)
        wyckoff_indices = flat[idx_atoms + n : idx_atoms + 2 * n].astype(np.int64)
        frac_coords = (
            flat[idx_atoms + 2 * n : idx_atoms + 5 * n].reshape(n, 3).astype(np.float32)
        )
        order = np.lexsort((element_indices, wyckoff_indices))
        return {
            "n_atoms_per_asu": n,
            "space_group_indices": int(flat[1]) - 1,
            "batch_chemistries": flat[2:idx_lengths].astype(np.float32),
            "lattice_lengths": flat[idx_lengths:idx_angles].astype(np.float32),
            "lattice_angles": flat[idx_angles:idx_atoms].astype(np.float32),
            "element_indices": element_indices[order],
            "wyckoff_indices": wyckoff_indices[order],
            "wyckoff_shape_indices": shape_indices[order],
            "frac_coords": frac_coords[order],
        }

    def __call__(self, flat_crystals):
        """Build ASU crystal field dicts from packed flat array(s)."""
        if isinstance(flat_crystals, list):
            return p_map(
                BuildAsuCrystal.build_one,
                flat_crystals,
                [self._idx_lengths] * len(flat_crystals),
                [self._idx_angles] * len(flat_crystals),
                [self._idx_atoms] * len(flat_crystals),
                num_cpus=self.num_cpus,
            )
        return BuildAsuCrystal.build_one(
            flat_crystals,
            self._idx_lengths,
            self._idx_angles,
            self._idx_atoms,
        )

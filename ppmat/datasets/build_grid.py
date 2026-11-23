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

from collections.abc import Mapping
from collections.abc import Sequence
from typing import Any
from typing import Literal
from typing import Optional

import numpy as np
from cvve import GridSpec
from p_tqdm import p_map

from ppmat.utils.crystal import normalize_coordinate_unit


class BuildGrid:
    """Build a uniform three-dimensional grid.

    Args:
        format (Literal["array", "bounding_box"]): Input format.
            - ``"array"``: Build from a mapping containing ``shape`` and
              ``voxel_vectors``.
            - ``"bounding_box"``: Build around Cartesian coordinates.
        shape (Optional[Sequence[int]], optional): Grid shape used by
            ``"bounding_box"``. Defaults to None.
        padding (float, optional): Padding added on each side of a bounding box.
            Defaults to 0.0.
        coordinate_unit (str, optional): Unit used by the grid geometry. Defaults
            to ``"angstrom"``.
        num_cpus (Optional[int], optional): Number of CPUs used when building a
            list of grids. Defaults to None, which uses one CPU.
    """

    def __init__(
        self,
        format: Literal["array", "bounding_box"],
        shape: Optional[Sequence[int]] = None,
        padding: float = 0.0,
        coordinate_unit: str = "angstrom",
        num_cpus: Optional[int] = None,
    ):
        self.format = format
        self.shape = shape
        self.padding = padding
        self.coordinate_unit = coordinate_unit
        self.num_cpus = num_cpus if num_cpus is not None else 1

    @staticmethod
    def build_one(
        grid_data: Any,
        format: Literal["array", "bounding_box"],
        shape: Optional[Sequence[int]] = None,
        padding: float = 0.0,
        coordinate_unit: str = "angstrom",
    ) -> GridSpec:
        """Build one grid from array data or Cartesian coordinates."""

        coordinate_unit = normalize_coordinate_unit(coordinate_unit)

        if format == "array":
            if not isinstance(grid_data, Mapping):
                raise TypeError("Array grid data must be a mapping.")
            return GridSpec(
                shape=grid_data["shape"],
                origin=grid_data.get("origin", np.zeros(3)),
                vectors=grid_data["voxel_vectors"],
                length_unit=coordinate_unit,
                value_unit=grid_data.get("value_unit", "unknown"),
                periodic=grid_data.get("periodic", (False, False, False)),
                cell=grid_data.get("cell"),
            )

        if format == "bounding_box":
            if shape is None:
                raise ValueError("shape is required for a bounding-box grid.")
            shape = tuple(int(size) for size in shape)
            if len(shape) != 3 or min(shape) <= 0:
                raise ValueError(f"Invalid grid shape: {shape}")
            if padding < 0:
                raise ValueError("padding must be non-negative.")

            coordinates = np.asarray(grid_data, dtype=np.float32)
            if (
                coordinates.ndim != 2
                or coordinates.shape[0] == 0
                or coordinates.shape[1] != 3
            ):
                raise ValueError("Coordinates must have shape [num_atoms, 3].")
            axis_len = np.maximum(np.ptp(coordinates, axis=0), 1e-3)
            axis_len += 2 * padding
            origin = (coordinates.min(axis=0) + coordinates.max(axis=0) - axis_len) / 2
            return GridSpec(
                shape=shape,
                origin=origin,
                vectors=np.diag(axis_len / np.asarray(shape)),
                length_unit=coordinate_unit,
            )

        raise ValueError(f"Unsupported grid format: {format}")

    def __call__(self, grids_data: Any) -> GridSpec | list[GridSpec]:
        is_batch = isinstance(grids_data, list) and (
            self.format == "array"
            or (len(grids_data) > 0 and np.asarray(grids_data[0]).ndim == 2)
        )
        if is_batch:
            return p_map(
                BuildGrid.build_one,
                grids_data,
                [self.format] * len(grids_data),
                [self.shape] * len(grids_data),
                [self.padding] * len(grids_data),
                [self.coordinate_unit] * len(grids_data),
                num_cpus=self.num_cpus,
            )
        return BuildGrid.build_one(
            grids_data,
            self.format,
            self.shape,
            self.padding,
            self.coordinate_unit,
        )

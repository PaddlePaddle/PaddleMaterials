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

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from dataclasses import replace
from itertools import chain
from itertools import islice
from typing import Any

import cvve
import numpy as np
from cvve import GridField
from cvve import GridSpec
from cvve import Structure
from p_tqdm import p_map

from ppmat.datasets.build_grid import BuildGrid
from ppmat.utils.crystal import normalize_coordinate_unit
from ppmat.utils.io import materialize_text_path
from ppmat.utils.io import open_text


@dataclass(frozen=True)
class BuildField:
    """Build one real scalar field from arrays or a volumetric field file.

    Args:
        name: Semantic field name, such as ``"density"`` or ``"potential"``.
        value_unit: Optional expected unit for field values. File formats keep
            the parsed unit when this is omitted.
        coordinate_unit: Optional expected coordinate unit. File formats use
            the unit parsed from the source. Array grids default to angstrom
            when no unit is supplied by a dataset-specific workflow.
        format: Input format. ``"array"`` accepts real-space values together
            with a keyword-only :class:`cvve.GridSpec`. ``"fft"`` accepts
            half-space packed FFT coefficients for that same grid and inverts
            them. ``"cube"`` and ``"chgcar"`` parse their own grid and optional
            atom metadata with cvve. ``"json"`` accepts a raw mapping or a
            plain/compressed JSON path from the Materials Project density
            release.
        num_cpus: Number of processes used when building a list or tuple.
    """

    name: str
    format: str
    value_unit: str | None = None
    coordinate_unit: str | None = None
    num_cpus: int = 1

    @staticmethod
    def build_grid_one(
        grid_data: Mapping[str, Any],
        coordinate_unit: str,
    ) -> GridSpec:
        """Build one affine grid from an array-style grid mapping."""

        return BuildGrid(
            format="array",
            coordinate_unit=coordinate_unit,
        )(grid_data)

    @staticmethod
    def build_one(
        field_data: Any,
        format: str,
        name: str,
        value_unit: str | None,
        coordinate_unit: str | None,
        grid: GridSpec | None = None,
        validate_coordinate_unit: bool = True,
    ) -> GridField:
        """Build one scalar field from a volumetric file or raw mapping.

        Args:
            field_data: A raw mapping or path for ``"json"``; a file path for
                another supported file format (``"cube"``, ``"chgcar"``).
            format: One of ``"cube"``, ``"chgcar"``, or ``"json"``.
            name: Semantic field name.
            value_unit: Expected field-value unit.
            coordinate_unit: Expected grid coordinate unit.
            grid: Optional independently built grid to validate and reuse for
                file data.
            validate_coordinate_unit: Whether a parsed file grid must already
                use ``coordinate_unit``.

        Returns:
            The normalized scalar field.
        """

        configured_coordinate_unit = (
            normalize_coordinate_unit(coordinate_unit)
            if coordinate_unit is not None
            else None
        )
        if format == "json":
            if not isinstance(field_data, Mapping):
                if not isinstance(field_data, (str, os.PathLike)):
                    raise TypeError("json field data must be a mapping or a file path.")
                with open_text(field_data) as file_obj:
                    field_data = json.load(file_obj)
            scale = float(field_data["vector"][0][0])
            cell = np.asarray(field_data["lattice"][0], dtype=float) * scale
            shape = tuple(int(size) for size in field_data["FFTgrid"][0])
            parsed_grid = GridSpec(
                shape=shape,
                origin=np.zeros(3),
                vectors=cell / np.asarray(shape)[:, None],
                length_unit="angstrom",
                value_unit="electron/angstrom^3",
                periodic=(True, True, True),
                cell=cell,
            )
            if (
                validate_coordinate_unit
                and configured_coordinate_unit is not None
                and parsed_grid.length_unit != configured_coordinate_unit
            ):
                raise ValueError(
                    "json grid uses "
                    f"{parsed_grid.length_unit!r}, but coordinate_unit is "
                    f"{coordinate_unit!r}."
                )
            if grid is not None and not grid.same_geometry(parsed_grid):
                raise ValueError(
                    "Provided grid does not match the grid parsed from the "
                    "field source."
                )
            parsed_value_unit = parsed_grid.value_unit
            if value_unit is not None and parsed_value_unit != value_unit:
                raise ValueError(
                    f"json field uses {parsed_value_unit!r}, but value_unit is "
                    f"{value_unit!r}."
                )

            num_values = int(np.prod(shape))
            raw_values = islice(
                chain.from_iterable(field_data["chargedensity"][0]),
                num_values,
            )
            values = np.fromiter(
                (
                    0.0
                    if isinstance(value, str) and value.startswith("*")
                    else float(value)
                    for value in raw_values
                ),
                dtype=float,
                count=num_values,
            )
            values = values.reshape(shape[::-1]).transpose(2, 1, 0)
            values /= abs(float(np.linalg.det(cell)))

            symbols = [
                str(symbol)
                for symbol, count in zip(
                    field_data["elements"][0],
                    field_data["elements_number"][0],
                )
                for _ in range(int(count))
            ]
            structure = Structure(
                symbols=symbols,
                positions=np.asarray(field_data["coordinates"][0], dtype=float),
                position_unit="angstrom",
                coordinate_mode="fractional",
                lattice=cell,
                periodic=(True, True, True),
            )
            output_grid = parsed_grid if grid is None else grid
            value_unit = value_unit or parsed_value_unit
            return GridField(
                data=values,
                grid=replace(output_grid, value_unit=value_unit),
                structure=structure,
                name=name,
                kind="density" if name == "density" else "unknown",
                source_format="json",
            )
        with materialize_text_path(field_data) as path:
            field = cvve.read_grid_field(
                path,
                format=format,
                name=name,
                kind="density" if name == "density" else "unknown",
            )
        if (
            validate_coordinate_unit
            and configured_coordinate_unit is not None
            and field.grid.length_unit != configured_coordinate_unit
        ):
            raise ValueError(
                f"{format} grid uses {field.grid.length_unit!r}, but "
                f"coordinate_unit is {coordinate_unit!r}."
            )
        if grid is not None and not grid.same_geometry(field.grid):
            raise ValueError(
                "Provided grid does not match the grid parsed from the field source."
            )

        parsed_value_unit = field.grid.value_unit
        if (
            value_unit is not None
            and parsed_value_unit != "unknown"
            and parsed_value_unit != value_unit
        ):
            raise ValueError(
                f"{format} field uses {parsed_value_unit!r}, but value_unit is "
                f"{value_unit!r}."
            )
        output_grid = field.grid if grid is None else grid
        value_unit = value_unit or parsed_value_unit or "unknown"
        return replace(
            field,
            grid=replace(output_grid, value_unit=value_unit),
            name=name,
        )

    def __call__(
        self,
        field_data: Any,
        *,
        grid: GridSpec | list[GridSpec] | tuple[GridSpec, ...] | None = None,
        validate_coordinate_unit: bool = True,
    ) -> GridField | list[GridField]:
        """Build one field or a list of fields."""

        if isinstance(field_data, (list, tuple)):
            if not field_data:
                return []
            if isinstance(grid, (list, tuple)):
                if len(grid) != len(field_data):
                    raise ValueError(
                        "grid and field_data must contain the same number of items."
                    )
                grids = grid
            else:
                grids = [grid] * len(field_data)
            return p_map(
                BuildField.build_one,
                field_data,
                [self.format] * len(field_data),
                [self.name] * len(field_data),
                [self.value_unit] * len(field_data),
                [self.coordinate_unit] * len(field_data),
                grids,
                [validate_coordinate_unit] * len(field_data),
                num_cpus=self.num_cpus,
                desc="Building fields",
                dynamic_ncols=True,
                mininterval=0.2,
            )

        return self.build_one(
            field_data,
            self.format,
            self.name,
            self.value_unit,
            self.coordinate_unit,
            grid=grid,
            validate_coordinate_unit=validate_coordinate_unit,
        )

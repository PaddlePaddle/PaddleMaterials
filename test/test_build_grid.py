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

import numpy as np

from ppmat.datasets.build_field import BuildField
from ppmat.datasets.build_grid import BuildGrid
from ppmat.utils.io import write_cube


def test_build_array_grid():
    grid_data = {
        "shape": (2, 3, 4),
        "voxel_vectors": np.diag([0.5, 1.0, 1.5]),
        "origin": [1.0, 2.0, 3.0],
        "periodic": (True, True, True),
    }
    grid = BuildGrid(format="array", coordinate_unit="bohr")(grid_data)

    assert grid.shape == (2, 3, 4)
    assert grid.length_unit == "bohr"
    assert grid.periodic == (True, True, True)
    np.testing.assert_allclose(grid.origin, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(grid.cell_vectors, np.diag([1.0, 3.0, 6.0]))

    compatibility_grid = BuildField.build_grid_one(grid_data, "bohr")
    assert compatibility_grid.same_geometry(grid)


def test_build_bounding_box_grid():
    coordinates = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [2.0, 4.0, 6.0],
        ],
        dtype=np.float32,
    )
    grid = BuildGrid(
        format="bounding_box",
        shape=(2, 3, 4),
        padding=1.0,
    )(coordinates)

    assert grid.shape == (2, 3, 4)
    assert grid.length_unit == "angstrom"
    np.testing.assert_allclose(grid.origin, [-1.0, -1.0, -1.0])
    np.testing.assert_allclose(grid.cell_vectors, np.diag([4.0, 6.0, 8.0]))


def test_build_grid_batch():
    array_builder = BuildGrid(format="array", num_cpus=1)
    array_grids = array_builder(
        [
            {
                "shape": (2, 2, 2),
                "voxel_vectors": np.eye(3),
            },
            {
                "shape": (3, 3, 3),
                "voxel_vectors": np.eye(3) * 0.5,
            },
        ]
    )
    assert [grid.shape for grid in array_grids] == [(2, 2, 2), (3, 3, 3)]

    bounding_box_builder = BuildGrid(
        format="bounding_box",
        shape=(4, 4, 4),
        num_cpus=1,
    )
    coordinate_grids = bounding_box_builder(
        [
            np.asarray([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]),
            np.asarray([[0.0, 0.0, 0.0], [2.0, 2.0, 2.0]]),
        ]
    )
    assert len(coordinate_grids) == 2
    assert all(grid.shape == (4, 4, 4) for grid in coordinate_grids)


def test_build_bounding_box_grid_from_coordinate_list():
    grid = BuildGrid(format="bounding_box", shape=(2, 2, 2))(
        [[0.0, 0.0, 0.0], [2.0, 2.0, 2.0]]
    )

    assert grid.shape == (2, 2, 2)
    np.testing.assert_allclose(grid.origin, [0.0, 0.0, 0.0])


def test_write_cube_uses_grid_geometry(tmp_path):
    grid = BuildGrid(format="array")(
        {
            "shape": (2, 3, 4),
            "voxel_vectors": np.diag([0.5, 0.5, 0.5]),
            "origin": [-1.0, -2.0, -3.0],
        }
    )
    density = np.arange(grid.npts, dtype=np.float32)
    cube_path = tmp_path / "density.cube"

    write_cube(
        cube_path,
        atom_numbers=[1],
        atom_coord=[[0.0, 0.0, 0.0]],
        density=density,
        grid=grid,
    )
    field = BuildField(format="cube", name="density")(
        cube_path,
        validate_coordinate_unit=False,
    )
    expected_grid = grid.to_length_unit("bohr")

    assert field.grid.shape == expected_grid.shape
    np.testing.assert_allclose(field.grid.origin, expected_grid.origin, atol=1e-5)
    np.testing.assert_allclose(
        field.grid.cell_vectors,
        expected_grid.cell_vectors,
        atol=1e-5,
    )
    np.testing.assert_allclose(field.flat, density, atol=1e-5)

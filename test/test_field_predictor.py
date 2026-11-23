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

from types import SimpleNamespace

import numpy as np
import paddle
from pymatgen.core import Lattice
from pymatgen.core import Structure

from ppmat.datasets.build_grid import BuildGrid
from ppmat.predictor import FieldPredictor
from ppmat.predictor import field_predictor as field_predictor_module


def test_from_structures_uses_model_predict_contract():
    predictor = object.__new__(FieldPredictor)
    predictor.predict_config = {"grid_batch_size": 2}
    predictor.model = SimpleNamespace(target_name="density")
    graph = object()
    predictor.graph_converter_fn = SimpleNamespace(
        from_structure=lambda structure: graph
    )

    batches = []

    def run_model(batch):
        batches.append(batch)
        return {"density": paddle.ones([3])}

    predictor._run_model = run_model
    grid = BuildGrid(format="array")(
        {
            "shape": (3, 1, 1),
            "voxel_vectors": np.eye(3, dtype=np.float32),
        }
    )
    result = predictor.from_structures(
        structures=object(),
        grid=grid,
    )

    assert len(batches) == 1
    assert batches[0]["graph"] is graph
    assert batches[0]["grid"] is grid
    assert batches[0]["grid_batch_size"] == 2
    assert list(result["density"].shape) == [3]


def test_from_molecule_uses_graph_and_grid():
    predictor = object.__new__(FieldPredictor)
    predictor.predict_config = {"grid_batch_size": 2}
    predictor.model = SimpleNamespace(target_name="density")

    molecule = object()
    graph = SimpleNamespace(
        node_feat={
            "cart_coords": np.asarray(
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32
            ),
        }
    )
    predictor.graph_converter_fn = lambda value: graph
    batches = []
    predictor._run_model = lambda data: batches.append(data) or {"density": None}

    grid = BuildGrid(
        format="bounding_box",
        shape=(2, 3, 4),
        padding=1.0,
    )(graph.node_feat["cart_coords"])
    predictor.from_molecule(
        molecule,
        grid,
    )

    assert batches[0]["graph"] is graph
    assert batches[0]["grid"] is grid


def test_from_xyz_file_builds_molecule_grid(tmp_path):
    predictor = object.__new__(FieldPredictor)
    predictor.predict_config = {"grid_batch_size": 2}
    predictor.model = SimpleNamespace(target_name="density")

    graph = object()
    predictor.graph_converter_fn = lambda value: graph
    batches = []
    predictor._run_model = lambda data: batches.append(data) or {
        "density": paddle.ones([8])
    }

    xyz_path = tmp_path / "hydrogen.xyz"
    xyz_path.write_text("2\nhydrogen\nH 0 0 0\nH 0 0 0.74\n")

    results = predictor.from_xyz_file(
        str(xyz_path),
        save_path=str(tmp_path / "output"),
        grid_shape=(2, 2, 2),
        grid_padding=1.0,
    )

    assert len(results) == 1
    assert batches[0]["graph"] is graph
    assert batches[0]["grid"].shape == (2, 2, 2)
    assert (tmp_path / "output" / "hydrogen_pred.cube").is_file()


def test_from_cif_file_builds_periodic_cell_grid(tmp_path):
    predictor = object.__new__(FieldPredictor)
    predictor.predict_config = {"grid_batch_size": 2}
    predictor.model = SimpleNamespace(target_name="density")

    graph = object()
    predictor.graph_converter_fn = SimpleNamespace(from_structure=lambda value: graph)
    batches = []
    predictor._run_model = lambda data: batches.append(data) or {
        "density": paddle.ones([8])
    }

    structure = Structure(
        Lattice.from_parameters(3.0, 4.0, 5.0, 80.0, 90.0, 100.0),
        ["C"],
        [[0.25, 0.25, 0.25]],
    )
    cif_path = tmp_path / "sample.cif"
    structure.to(filename=cif_path)

    results = predictor.from_cif_file(
        str(cif_path),
        save_path=str(tmp_path / "output"),
        grid_shape=(2, 2, 2),
    )

    assert len(results) == 1
    assert batches[0]["graph"] is graph
    assert batches[0]["grid"].shape == (2, 2, 2)
    assert batches[0]["grid"].periodic == (True, True, True)
    np.testing.assert_allclose(
        batches[0]["grid"].cell_vectors,
        structure.lattice.matrix,
    )
    assert (tmp_path / "output" / "sample_pred.cube").is_file()


def test_field_file_formats_use_source_structure_and_grid(tmp_path, monkeypatch):
    predictor = object.__new__(FieldPredictor)
    predictor.device = "cpu"
    predictor.predict_config = {"grid_batch_size": 2}
    predictor.model = SimpleNamespace(target_name="density")

    grid = BuildGrid(format="array")(
        {
            "shape": (2, 1, 1),
            "voxel_vectors": np.eye(3, dtype=np.float32),
        }
    )
    structure = object()
    graph = SimpleNamespace()
    converted_structures = []
    predictor.graph_converter_fn = SimpleNamespace(
        from_structure=lambda value: converted_structures.append(value) or graph
    )
    predictor._run_model = lambda data: {"density": paddle.ones([2])}

    formats = []

    class FakeBuildField:
        def __init__(self, format, name):
            formats.append((format, name))

        def __call__(self, path):
            return SimpleNamespace(structure=structure, grid=grid)

    monkeypatch.setattr(field_predictor_module, "BuildField", FakeBuildField)

    inputs = {
        "cube": tmp_path / "sample.cube.lz4",
        "chgcar": tmp_path / "sample.CHGCAR.lz4",
        "json": tmp_path / "sample.json.xz",
    }
    for field_format, field_path in inputs.items():
        field_path.touch()
        results = getattr(predictor, f"from_{field_format}_file")(str(field_path))
        assert list(results[0]["density"].shape) == [2]

    assert formats == [
        ("cube", "density"),
        ("chgcar", "density"),
        ("json", "density"),
    ]
    assert converted_structures == [structure, structure, structure]

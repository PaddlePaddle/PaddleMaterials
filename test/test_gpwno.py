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

import json
from pathlib import Path

import numpy as np
import paddle
import pgl
import pytest
from omegaconf import OmegaConf

from ppmat.datasets.build_field import BuildField
from ppmat.datasets.build_grid import BuildGrid
from ppmat.datasets.collate_fn import DensityCollator
from ppmat.datasets.density_dataset import DensityDataset
from ppmat.datasets.transform import build_transforms
from ppmat.datasets.transform.preprocess import AddFourierGrid
from ppmat.models import MODEL_REGISTRY
from ppmat.models.gpwno.GPWNO import GPWNO
from ppmat.models.gpwno.graph_converter import GPWNOGraphConverter
from ppmat.predictor import FieldPredictor
from ppmat.predictor import base as predictor_base
from ppmat.utils.io import write_cube


@pytest.mark.parametrize(
    "model_name",
    ["gpwno_md17_benzene", "gpwno_md17_ethane", "gpwno_mp", "gpwno_qm9"],
)
def test_pretrained_model_registry(model_name):
    assert MODEL_REGISTRY[model_name] == (
        "https://paddle-org.bj.bcebos.com/"
        f"paddlematerials/checkpoints/electronic_structure/gpwno/{model_name}.zip"
    )


def small_model_params(**overrides):
    params = {
        "n_atom_type": 3,
        "num_radial": 2,
        "num_spherical": 1,
        "radial_embed_size": 4,
        "radial_hidden_size": 4,
        "num_fourier": 4,
        "num_fourier_time": 2,
        "num_gcn_layer": 1,
        "num_local_gcn_layers": 1,
        "width": 2,
        "padding": 0,
        "max_cell_size": 4,
        "probe_cutoff": 3,
        "grid_cutoff": 3,
        "residual": False,
    }
    params.update(overrides)
    return params


def build_small_model(**overrides):
    return GPWNO(**small_model_params(**overrides))


def test_fourier_coordinates_preserve_query_grid_and_are_reused():
    transform = AddFourierGrid([4, 3, 1])
    query = np.asarray([[2.0, 3.0, 4.0]], dtype="float32")
    sample = {"grid_coord": query}
    result = transform(sample)
    assert "fourier_grid" not in sample
    assert result["grid_coord"] is query
    assert transform(sample)["fourier_grid"] is result["fourier_grid"]
    expected = np.stack(
        np.meshgrid(np.linspace(0, 1, 4), np.linspace(0, 1, 3), [0], indexing="ij"),
        axis=-1,
    )
    np.testing.assert_allclose(result["fourier_grid"], expected, atol=1e-7)


@pytest.mark.parametrize("periodic", [False, True])
def test_dataset_training_checkpoint_and_prediction(periodic, tmp_path, monkeypatch):
    """Exercise cached CVVE graphs and shared preprocessing for train and predict."""
    paddle.seed(42)
    vocab = {"atom": {"num_embeddings": 3, "atomic_number_to_id": {1: 0, 6: 1, 8: 2}}}
    converter_cfg = {
        "__class_name__": "GPWNOGraphConverter",
        "__init_params__": {"pbc": periodic, "cutoff": 3.0},
    }
    transform_cfg = [
        {"__class_name__": "AddFourierGrid", "__init_params__": {"shape": [4] * 3}}
    ]
    grid = BuildGrid(format="array")(
        {
            "shape": [2, 2, 2],
            "origin": np.asarray([0.5, 0.6, 0.7]),
            "voxel_vectors": np.eye(3) * 2,
        }
    )
    cube_path = tmp_path / "sample.cube"
    write_cube(
        cube_path,
        [6, 1, 8],
        [[0, 0.3, 0.4], [1.1, 0.3, 0.4], [0.2, 1.2, 0.4]],
        np.ones(8),
        grid,
    )
    field_path, field_format = cube_path, "cube"
    if periodic:
        # MP stores its lattice with fractional coordinates in JSON, whereas
        # molecular CUBE metadata intentionally has no structure lattice.
        field_path, field_format = tmp_path / "sample.json", "json"
        field_path.write_text(
            json.dumps(
                {
                    "vector": [[1]],
                    "lattice": [(np.eye(3) * 4).tolist()],
                    "FFTgrid": [[2, 2, 2]],
                    "chargedensity": [[[64.0] * 8]],
                    "elements": [["C", "H", "O"]],
                    "elements_number": [[1, 1, 1]],
                    "coordinates": [
                        [[0, 0.075, 0.1], [0.275, 0.075, 0.1], [0.05, 0.3, 0.1]]
                    ],
                }
            )
        )
    index_path = tmp_path / "index.json"
    index_path.write_text(json.dumps({"train": [field_path.name] * 2}))
    dataset = DensityDataset(
        path=str(index_path),
        split="train",
        vocab=vocab,
        build_graph_cfg=converter_cfg,
        build_field_cfg={"format": field_format, "name": "density", "num_cpus": 1},
        transforms=build_transforms(transform_cfg),
    )
    sample = dataset[0]
    assert sample["graph"].num_nodes == (4 if periodic else 3)
    loader = paddle.io.DataLoader(dataset, batch_size=2, collate_fn=DensityCollator())
    batch = next(iter(loader))
    params = small_model_params(pbc=periodic, use_max_cell=not periodic)
    model = GPWNO(**params)
    optimizer = paddle.optimizer.Adam(learning_rate=1e-4, parameters=model.parameters())
    loss = model(batch)["loss_dict"]["loss"]
    assert bool(paddle.isfinite(loss))
    loss.backward()
    assert model.embedding.weight.grad is not None
    assert bool(paddle.isfinite(model.embedding.weight.grad).all())
    optimizer.step()
    optimizer.clear_grad()
    model.eval()
    with paddle.no_grad():
        full = model(batch, return_loss=False)["pred_dict"]["density"]
        chunked = model({**batch, "grid_batch_size": 2}, return_loss=False)[
            "pred_dict"
        ]["density"]
    np.testing.assert_allclose(chunked.numpy(), full.numpy(), rtol=1e-5, atol=1e-5)

    config_path = tmp_path / "config.yaml"
    OmegaConf.save(
        OmegaConf.create(
            {
                "Model": {"__class_name__": "GPWNO", "__init_params__": params},
                "Predict": {
                    "graph_converter": converter_cfg,
                    "transforms": transform_cfg,
                },
            }
        ),
        config_path,
    )
    checkpoint = tmp_path / "best.pdparams"
    paddle.save(model.state_dict(), str(checkpoint))
    monkeypatch.setattr(predictor_base, "build_vocab", lambda name: vocab)
    paddle.seed(42)
    predictor = FieldPredictor(
        config_path=str(config_path), checkpoint_path=str(checkpoint), device="cpu"
    )
    field = BuildField(format=field_format, name="density", num_cpus=1)(str(field_path))
    prediction = predictor.from_structures(
        field.structure, field.grid, grid_batch_size=2
    )["density"]
    np.testing.assert_allclose(
        prediction.numpy(), full[0].numpy(), rtol=1e-5, atol=1e-5
    )
    if periodic:
        cube = BuildField(format="cube", name="density", num_cpus=1)(str(cube_path))
        assert cube.structure.lattice is None
        cube_prediction = predictor.from_structures(cube.structure, cube.grid)[
            "density"
        ]
        assert bool(paddle.isfinite(cube_prediction).all())
        assert cube.structure.lattice is None


@pytest.mark.parametrize(
    "name", ["gpwno_md17_benzene", "gpwno_md17_ethane", "gpwno_qm9", "gpwno_mp"]
)
def test_configs_share_training_and_prediction_preprocessing(name):
    path = (
        Path(__file__).resolve().parents[1]
        / "electronic_structure/configs/gpwno"
        / f"{name}.yaml"
    )
    cfg = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    expected_shape = [cfg["Model"]["__init_params__"]["num_fourier"]] * 3
    assert cfg["Global"]["transforms"][0]["__init_params__"]["shape"] == expected_shape
    assert (
        cfg["Global"]["graph_converter"]["__init_params__"]["pbc"]
        == cfg["Model"]["__init_params__"]["pbc"]
    )
    for split in ("train", "val", "test"):
        params = cfg["Dataset"][split]["dataset"]["__init_params__"]
        assert params["transforms"] == cfg["Predict"]["transforms"]
        assert params["build_graph_cfg"] == cfg["Predict"]["graph_converter"]


def prepare_density_inputs(model, atom_types, atoms, grid, batch, info=None):
    converter = GPWNOGraphConverter(cutoff=model.cutoff, pbc=model.pbc)
    graphs = []
    for index in range(grid.shape[0]):
        selected = batch.numpy() == index
        types = atom_types.numpy()[selected]
        cell = info["cell"][index].numpy() if info is not None else None
        graphs.append(
            converter.from_arrays(
                types, atoms.numpy()[selected], {"x": types}, cell=cell
            )
        )
    coordinates = AddFourierGrid([model.num_fourier] * 3).coordinates
    return {
        "graph": pgl.Graph.batch(graphs),
        "grid_coord": grid,
        "info": info,
        "fourier_grid": np.broadcast_to(
            coordinates, [grid.shape[0], *coordinates.shape]
        ).copy(),
    }


def run_density(model, atom_types, atoms, grid, batch, info=None):
    return model._forward(
        prepare_density_inputs(model, atom_types, atoms, grid, batch, info)
    )


@pytest.mark.parametrize("case", ["crowded", "coincident", "boundary", "empty"])
def test_atom_graph_matches_nearest_neighbors_without_self(case):
    rng = np.random.default_rng(42)
    if case == "crowded":
        positions = rng.uniform(0, 0.25, (74, 3)).astype("float32")
        batches = np.arange(74, dtype="int64") % 2
    elif case == "coincident":
        positions = np.zeros((40, 3), dtype="float32")
        batches = np.zeros(40, dtype="int64")
    elif case == "boundary":
        positions = np.array([[0, 0, 0], [1, 0, 0], [3, 0, 0]], dtype="float32")
        batches = np.zeros(3, dtype="int64")
    else:
        positions = np.empty((0, 3), dtype="float32")
        batches = np.empty(0, dtype="int64")
    converter = GPWNOGraphConverter(cutoff=1.0)
    edges = []
    for graph_id in np.unique(batches):
        indices = np.flatnonzero(batches == graph_id)
        graph = converter.from_arrays(
            np.ones(len(indices), dtype="int64"), positions[indices]
        )
        edges.extend(indices[graph.edges].tolist())
    edges = np.asarray(edges, dtype="int64").reshape(-1, 2).T
    expected = []
    for dst in range(len(positions)):
        distances = ((positions - positions[dst]) ** 2).sum(axis=1)
        candidates = np.flatnonzero(
            (distances <= 1.0)
            & (batches == batches[dst])
            & (np.arange(len(positions)) != dst)
        )
        nearest = candidates[np.lexsort((candidates, distances[candidates]))][:32]
        expected.extend((int(src), dst) for src in nearest)
    expected.sort(key=lambda pair: (batches[pair[0]], *pair))
    expected = np.asarray(expected, dtype="int64").reshape(-1, 2).T
    np.testing.assert_array_equal(edges, expected)


@pytest.mark.parametrize(
    "model_sharing,rno_degree,expected_degree",
    [(True, None, 1), (True, 2, 1), (False, None, 1), (False, 2, 2)],
)
def test_resolved_rno_degree(model_sharing, rno_degree, expected_degree):
    model = build_small_model(model_sharing=model_sharing, num_spherical_RNO=rno_degree)
    assert model.irreps_sh_RNO.lmax == expected_degree
    assert model.irreps_feat_RNO.dim == 2 * (expected_degree + 1) ** 2


@pytest.mark.parametrize("using_ff", [False, True])
@pytest.mark.parametrize("activation", ["norm", "scalar"])
def test_centered_prediction_preserves_inputs_and_translation(using_ff, activation):
    paddle.seed(42)
    model = build_small_model(using_ff=using_ff, activation=activation)
    model.eval()
    atoms = paddle.to_tensor(
        [
            [0.25, 0.125, 0.0625],
            [1.125, 0.25, 0.5],
            [0.375, 1.75, 0.75],
            [0.5, 0.625, 2.25],
        ],
        dtype="float32",
    )
    grid = paddle.to_tensor(
        [[[0.5, 0.5, 0.5], [1.0, 0.75, 1.25], [0.25, 1.0, 1.5]]],
        dtype="float32",
    )
    atom_types = paddle.to_tensor([0, 1, 2, 0], dtype="int64")
    batch = paddle.zeros([4], dtype="int64")
    original_atoms, original_grid = atoms.numpy().copy(), grid.numpy().copy()

    with paddle.no_grad():
        density, _ = run_density(model, atom_types, atoms, grid, batch, None)
        repeated, _ = run_density(model, atom_types, atoms, grid, batch, None)
        translation = paddle.to_tensor([4.0, -2.0, 1.0])
        translated, _ = run_density(
            model, atom_types, atoms + translation, grid + translation, batch, None
        )

    np.testing.assert_array_equal(atoms.numpy(), original_atoms)
    np.testing.assert_array_equal(grid.numpy(), original_grid)
    assert np.isfinite(density.numpy()).all()
    np.testing.assert_allclose(density.numpy(), repeated.numpy(), atol=1e-6)
    np.testing.assert_allclose(density.numpy(), translated.numpy(), atol=1e-5)


def test_legacy_local_branch_configuration_and_weights():
    model = build_small_model(model_sharing=False)
    restored = build_small_model(
        model_sharing=False,
        num_local_gcn_layers=None,
        num_infgcn_layer=1,
        input_infgcn=False,
    )
    assert restored.num_local_gcn_layers == 1
    assert not restored.input_local_density
    legacy_state = {
        name.replace("local_gcns.", "infgcns.", 1): tensor.clone()
        for name, tensor in model.state_dict().items()
    }
    assert any(name.startswith("infgcns.") for name in legacy_state)
    missing, unexpected = restored.set_state_dict(legacy_state)
    assert not missing
    assert not unexpected
    for name, tensor in model.state_dict().items():
        np.testing.assert_array_equal(
            tensor.numpy(), restored.state_dict()[name].numpy()
        )


@pytest.mark.parametrize(
    "options",
    [
        {"num_local_gcn_layers": 1, "num_infgcn_layer": 2},
        {"input_local_density": False, "input_infgcn": True},
    ],
)
def test_conflicting_legacy_local_branch_options(options):
    with pytest.raises(ValueError, match="Conflicting local"):
        build_small_model(**options)


@pytest.mark.parametrize("model_sharing", [False, True])
@pytest.mark.parametrize("periodic", [False, True])
def test_local_density_input_matches_independent_graphs(model_sharing, periodic):
    model = build_small_model(
        input_local_density=True,
        model_sharing=model_sharing,
        pbc=periodic,
        use_max_cell=not periodic,
        equivariant_frame=False,
    )
    model.eval()
    atom_types = paddle.to_tensor([0, 1, 2, 0], dtype="int64")
    atoms = paddle.to_tensor(
        [[0.2, 0.3, 0.4], [1.0, 0.3, 0.4], [0.3, 0.5, 0.4], [0.3, 1.3, 0.4]],
        dtype="float32",
    )
    grid = paddle.to_tensor(
        [[[0.5, 0.5, 0.5], [1.2, 0.8, 0.6]], [[0.6, 0.5, 0.8], [0.8, 1.1, 0.7]]],
        dtype="float32",
    )
    batch = paddle.to_tensor([0, 0, 1, 1], dtype="int64")
    cells = paddle.eye(3).unsqueeze(0).tile([2, 1, 1]) * 3
    info = {"cell": cells} if periodic else None
    with paddle.no_grad():
        prediction, _ = run_density(model, atom_types, atoms, grid, batch, info)
        separate = []
        for index in range(2):
            selected = batch == index
            value, _ = run_density(
                model,
                atom_types[selected],
                atoms[selected],
                grid[index : index + 1],
                paddle.zeros([2], dtype="int64"),
                {"cell": cells[index : index + 1]} if periodic else None,
            )
            separate.append(value)
    assert bool(paddle.isfinite(prediction).all())
    np.testing.assert_allclose(
        prediction.numpy(), paddle.concat(separate).numpy(), rtol=1e-5, atol=1e-5
    )


@pytest.mark.parametrize("skewed", [False, True])
def test_periodic_boundary_images_match_legacy_order_and_are_idempotent(skewed):
    matrix = (
        [[4.0, 0.4, 0.2], [0.0, 3.0, 0.3], [0.1, 0.2, 5.0]] if skewed else np.eye(3) * 4
    )
    cell = paddle.to_tensor([matrix], dtype="float32")
    fractional = paddle.to_tensor(
        [[0.0, 0.25, 0.5], [0.2, 0.4, 0.6], [0.5, 0.0, 0.0]], dtype="float32"
    )
    atoms = fractional @ cell[0]
    atom_types = paddle.to_tensor([0, 1, 2], dtype="int64")
    batch = paddle.zeros([3], dtype="int64")
    original = atoms.numpy().copy()
    (
        expanded_types,
        expanded_atoms,
        expanded_batch,
    ) = GPWNOGraphConverter._expand_boundary_atoms(atom_types, atoms, batch, cell)
    sources = np.array([0, 0, 1, 2, 2, 2, 2])
    shifts = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 0, 0], [0, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 1]]
    )
    np.testing.assert_array_equal(expanded_types.numpy(), atom_types.numpy()[sources])
    np.testing.assert_allclose(
        expanded_atoms.numpy(),
        original[sources] + shifts @ np.asarray(matrix),
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_array_equal(expanded_batch.numpy(), np.zeros(7))
    np.testing.assert_array_equal(atoms.numpy(), original)
    repeated = GPWNOGraphConverter._expand_boundary_atoms(
        expanded_types, expanded_atoms, expanded_batch, cell
    )
    for actual, expected in zip(
        repeated, (expanded_types, expanded_atoms, expanded_batch)
    ):
        assert actual is expected


def test_periodic_boundary_images_use_each_graph_cell_and_species():
    cells = paddle.to_tensor([np.eye(3) * 3, np.eye(3) * 5], dtype="float32")
    atom_types = paddle.to_tensor([0, 1, 0, 1], dtype="int64")
    atoms = paddle.to_tensor(
        [[0, 0.6, 0.9], [0, 0.6, 0.9], [0, 1.0, 1.5], [1, 1, 1]], dtype="float32"
    )
    batch = paddle.to_tensor([0, 0, 1, 1], dtype="int64")
    expanded = GPWNOGraphConverter._expand_boundary_atoms(
        atom_types, atoms, batch, cells
    )
    expected_types, expected_atoms, expected_batch = [], [], []
    for index in range(2):
        selected = batch == index
        types, positions, local_batch = GPWNOGraphConverter._expand_boundary_atoms(
            atom_types[selected],
            atoms[selected],
            paddle.zeros([2], dtype="int64"),
            cells[index : index + 1],
        )
        expected_types.append(types)
        expected_atoms.append(positions)
        expected_batch.append(local_batch + index)
    for actual, expected in zip(
        expanded, (expected_types, expected_atoms, expected_batch)
    ):
        np.testing.assert_allclose(actual.numpy(), paddle.concat(expected).numpy())
    assert expanded[0].shape[0] == 7


def test_periodic_boundary_images_preserve_coordinate_and_cell_gradients():
    atoms = paddle.to_tensor([[0.0, 0.4, 0.5]], stop_gradient=False)
    cell = paddle.to_tensor([np.eye(3) * 2], dtype="float32", stop_gradient=False)
    _, positions, _ = GPWNOGraphConverter._expand_boundary_atoms(
        paddle.zeros([1], dtype="int64"), atoms, paddle.zeros([1], dtype="int64"), cell
    )
    positions.sum().backward()
    np.testing.assert_array_equal(atoms.grad.numpy(), [[2, 2, 2]])
    np.testing.assert_array_equal(
        cell.grad.numpy(), [[[1, 1, 1], [0, 0, 0], [0, 0, 0]]]
    )


@pytest.mark.parametrize("offset", [0.0, 1e-7])
def test_periodic_corner_images_tolerate_coordinate_roundoff(offset):
    cell = paddle.eye(3).unsqueeze(0) * 2
    expanded = GPWNOGraphConverter._expand_boundary_atoms(
        paddle.zeros([1], dtype="int64"),
        paddle.to_tensor([[offset, 0.0, 0.0]]),
        paddle.zeros([1], dtype="int64"),
        cell,
    )
    assert expanded[0].shape[0] == 8
    repeated = GPWNOGraphConverter._expand_boundary_atoms(*expanded, cell)
    for actual, expected in zip(repeated, expanded):
        assert actual is expected


@pytest.mark.parametrize("local_density", [False, True])
def test_periodic_prediction_expands_boundaries_without_mutating_graph(local_density):
    model = build_small_model(
        pbc=True,
        use_max_cell=False,
        equivariant_frame=False,
        input_local_density=local_density,
        model_sharing=False,
    )
    model.eval()
    positions = np.asarray([[0.0, 0.7, 1.0], [1.2, 1.4, 1.6]], dtype=np.float32)
    graph = GPWNOGraphConverter(pbc=True).from_arrays(
        np.asarray([0, 1]), positions, {"x": np.asarray([0, 1])}, cell=np.eye(3) * 4
    )
    prepared_positions = graph.node_feat["cart_coords"].copy()
    transform = AddFourierGrid([model.num_fourier] * 3)
    grid = BuildGrid(format="array")(
        {
            "shape": (2, 2, 2),
            "origin": np.asarray([0.1, 0.2, 0.3]),
            "voxel_vectors": np.eye(3) * 2,
        }
    )
    with paddle.no_grad():
        full = model.predict(
            transform({"graph": graph, "grid": grid, "grid_batch_size": 8})
        )["density"]
        chunked = model.predict(
            transform({"graph": graph, "grid": grid, "grid_batch_size": 2})
        )["density"]
        expected, _ = run_density(
            model,
            paddle.to_tensor([0, 0, 1], dtype="int64"),
            paddle.to_tensor(
                [positions[0], positions[0] + [4, 0, 0], positions[1]], dtype="float32"
            ),
            paddle.to_tensor(grid.cartesian_coordinates(), dtype="float32").reshape(
                [1, -1, 3]
            ),
            paddle.zeros([3], dtype="int64"),
            {"cell": paddle.eye(3).unsqueeze(0) * 4},
        )
    np.testing.assert_allclose(
        full.numpy(), expected.numpy().reshape(-1), rtol=1e-5, atol=1e-5
    )
    np.testing.assert_allclose(full.numpy(), chunked.numpy(), rtol=1e-5, atol=1e-5)
    np.testing.assert_array_equal(
        graph.node_feat["cart_coords"].numpy(), prepared_positions
    )

    model.train()
    output = model(
        {
            "graph": graph,
            "grid_coord": paddle.to_tensor(
                grid.cartesian_coordinates(), dtype="float32"
            ).reshape([1, -1, 3]),
            "fourier_grid": transform.coordinates,
            "density": paddle.ones([1, 8]),
            "info": {"cell": paddle.eye(3).unsqueeze(0) * 4},
        }
    )
    loss = output["loss_dict"]["loss"]
    assert bool(paddle.isfinite(loss))
    loss.backward()
    assert model.embedding.weight.grad is not None
    assert bool(paddle.isfinite(model.embedding.weight.grad).all())


def test_periodic_prediction_requires_cell():
    model = build_small_model(pbc=True)
    with pytest.raises(ValueError, match="require"):
        run_density(
            model,
            paddle.zeros([1], dtype="int64"),
            paddle.zeros([1, 3]),
            paddle.zeros([1, 2, 3]),
            paddle.zeros([1], dtype="int64"),
            None,
        )

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
import paddle
import pgl
import pytest

from ppmat.datasets.build_grid import BuildGrid
from ppmat.models import MODEL_REGISTRY
from ppmat.models.gpwno.GPWNO import GPWNO


@pytest.mark.parametrize(
    "model_name",
    ["gpwno_md17_benzene", "gpwno_md17_ethane", "gpwno_mp", "gpwno_qm9"],
)
def test_pretrained_model_registry(model_name):
    assert MODEL_REGISTRY[model_name] == (
        "https://paddle-org.bj.bcebos.com/"
        f"paddlematerials/checkpoints/electronic_structure/gpwno/{model_name}.zip"
    )


def build_small_model(**overrides):
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
    return GPWNO(**params)


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
        density, _ = model._forward_density(atom_types, atoms, grid, batch, None)
        repeated, _ = model._forward_density(atom_types, atoms, grid, batch, None)
        translation = paddle.to_tensor([4.0, -2.0, 1.0])
        translated, _ = model._forward_density(
            atom_types, atoms + translation, grid + translation, batch, None
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
        prediction, _ = model._forward_density(atom_types, atoms, grid, batch, info)
        separate = []
        for index in range(2):
            selected = batch == index
            value, _ = model._forward_density(
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
    expanded_types, expanded_atoms, expanded_batch = GPWNO._expand_boundary_atoms(
        atom_types, atoms, batch, cell
    )
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
    repeated = GPWNO._expand_boundary_atoms(
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
    expanded = GPWNO._expand_boundary_atoms(atom_types, atoms, batch, cells)
    expected_types, expected_atoms, expected_batch = [], [], []
    for index in range(2):
        selected = batch == index
        types, positions, local_batch = GPWNO._expand_boundary_atoms(
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
    _, positions, _ = GPWNO._expand_boundary_atoms(
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
    expanded = GPWNO._expand_boundary_atoms(
        paddle.zeros([1], dtype="int64"),
        paddle.to_tensor([[offset, 0.0, 0.0]]),
        paddle.zeros([1], dtype="int64"),
        cell,
    )
    assert expanded[0].shape[0] == 8
    repeated = GPWNO._expand_boundary_atoms(*expanded, cell)
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
    graph = pgl.Graph(
        edges=np.empty((0, 2), dtype=np.int64),
        num_nodes=2,
        node_feat={
            "x": np.asarray([0, 1], dtype=np.int64),
            "cart_coords": positions.copy(),
        },
    )
    grid = BuildGrid(format="array")(
        {
            "shape": (2, 2, 2),
            "origin": np.asarray([0.1, 0.2, 0.3]),
            "voxel_vectors": np.eye(3) * 2,
        }
    )
    with paddle.no_grad():
        full = model.predict({"graph": graph, "grid": grid, "grid_batch_size": 8})[
            "density"
        ]
        chunked = model.predict({"graph": graph, "grid": grid, "grid_batch_size": 2})[
            "density"
        ]
        expected, _ = model._forward_density(
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
    np.testing.assert_array_equal(graph.node_feat["cart_coords"].numpy(), positions)

    model.train()
    output = model(
        {
            "graph": graph,
            "grid_coord": paddle.to_tensor(
                grid.cartesian_coordinates(), dtype="float32"
            ).reshape([1, -1, 3]),
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
    with pytest.raises(ValueError, match="requires info"):
        model._forward_density(
            paddle.zeros([1], dtype="int64"),
            paddle.zeros([1, 3]),
            paddle.zeros([1, 2, 3]),
            paddle.zeros([1], dtype="int64"),
            None,
        )

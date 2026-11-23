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

import math

import numpy as np
import paddle
from rdkit import Chem
from scipy import special

from ppmat.datasets.build_molecule import BuildMolecule
from ppmat.datasets.collate_fn import RadiusGraphCollator
from ppmat.models.common import initializer
from ppmat.models.common.graph_converter import RadiusGraphConverter
from ppmat.models.common.spherical_fourier_bessel import RealSphericalHarmonics
from ppmat.models.common.spherical_fourier_bessel import SphericalBesselBasis
from ppmat.models.common.spherical_fourier_bessel import SphericalFourierBesselEmbedding
from ppmat.models.common.spherical_fourier_bessel import _build_basis_constants
from ppmat.models.spherenet.geometry import compute_geometry
from ppmat.models.spherenet.spherenet import ResidualLayer
from ppmat.models.spherenet.spherenet import SphereNet


def setup_module():
    paddle.set_device("cpu")


def test_radius_graph_matches_pyg_edge_and_triplet_order():
    molecule = BuildMolecule(format="dict", sanitize=False)(
        {
            "atomic_numbers": np.array([6, 7, 8]),
            "positions": np.array(
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                dtype=np.float32,
            ),
        }
    )
    graph = RadiusGraphConverter(cutoff=2.0, return_triplet_indices=True)(molecule)

    np.testing.assert_array_equal(
        graph.edges,
        [
            [1, 0],
            [2, 0],
            [0, 1],
            [2, 1],
            [0, 2],
            [1, 2],
        ],
    )
    np.testing.assert_array_equal(graph.edge_feat["ti_idx_kj"], [3, 5, 1, 4, 0, 2])
    np.testing.assert_array_equal(graph.edge_feat["ti_idx_ji"], [0, 1, 2, 3, 4, 5])


def test_build_molecule_from_dict():
    molecule = BuildMolecule(format="dict", sanitize=False)(
        {
            "atomic_numbers": np.array([8, 1, 1]),
            "positions": np.array(
                [[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]]
            ),
        }
    )

    assert [atom.GetAtomicNum() for atom in molecule.GetAtoms()] == [8, 1, 1]
    np.testing.assert_allclose(
        molecule.GetConformer().GetPositions(),
        [[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]],
    )


def test_basis_constants_are_cached_by_shape():
    _build_basis_constants.cache_clear()
    constants_7_6 = _build_basis_constants(7, 6)
    constants_3_4 = _build_basis_constants(3, 4)

    assert constants_7_6 is _build_basis_constants(7, 6)
    assert constants_7_6[0].shape == (7, 6)
    assert constants_3_4[0].shape == (3, 4)
    assert not constants_7_6[0].flags.writeable
    assert _build_basis_constants.cache_info().currsize == 2


def test_glorot_orthogonal_matches_pyg_target_variance():
    paddle.seed(7)
    weight = paddle.empty([16, 32], dtype="float32")
    initializer.glorot_orthogonal_(weight, scale=2.0)

    np.testing.assert_allclose(float(paddle.var(weight)), 2.0 / (16 + 32), rtol=1e-5)


def test_spherenet_swish_has_finite_cpu_gradient_for_extreme_inputs():
    activation = ResidualLayer(1).act

    values = paddle.to_tensor([-100.0, -500.0, 500.0])
    values.stop_gradient = False
    gradient = paddle.grad(activation(values).sum(), values, create_graph=True)[0]
    second_gradient = paddle.grad(gradient.sum(), values)[0]

    assert bool(paddle.isfinite(gradient).all())
    assert bool(paddle.isfinite(second_gradient).all())


def test_spherical_bessel_matches_scipy_value_and_gradient():
    num_spherical = 7
    num_radial = 6
    distances = np.array(
        [1e-4, 0.03, 0.15, 0.7, 1.3, 2.6, 4.9],
        dtype=np.float32,
    )
    dist = paddle.to_tensor(distances, stop_gradient=False)
    actual = SphericalBesselBasis(num_spherical, num_radial)(dist)

    zeros, normalizers, _ = _build_basis_constants(num_spherical, num_radial)
    arguments = distances[:, None, None].astype(np.float64) / 5.0
    arguments = arguments * zeros[None]
    expected = np.empty_like(arguments)
    expected_gradient = np.zeros(len(distances), dtype=np.float64)
    for degree in range(num_spherical):
        expected[:, degree, :] = (
            special.spherical_jn(degree, arguments[:, degree, :]) * normalizers[degree]
        )
        expected_gradient += np.sum(
            special.spherical_jn(
                degree,
                arguments[:, degree, :],
                derivative=True,
            )
            * normalizers[degree]
            * zeros[degree]
            / 5.0,
            axis=1,
        )

    np.testing.assert_allclose(actual.numpy(), expected, rtol=1e-4, atol=1e-5)
    actual_gradient = paddle.grad(paddle.sum(actual), dist)[0]
    np.testing.assert_allclose(
        actual_gradient.numpy(),
        expected_gradient,
        rtol=1e-4,
        atol=3e-5,
    )


def test_real_spherical_harmonics_components_preserve_spherenet_order():
    angle = paddle.to_tensor([math.pi / 2], dtype="float32")
    torsion = paddle.zeros([1], dtype="float32")
    harmonics = RealSphericalHarmonics(2)(
        paddle.cos(angle),
        paddle.sin(angle),
        paddle.cos(torsion),
        paddle.sin(torsion),
    )

    scale = math.sqrt(3.0 / (4.0 * math.pi))
    expected = np.array(
        [[1.0 / math.sqrt(4.0 * math.pi), 0.0, -scale, 0.0]],
        dtype=np.float32,
    )
    np.testing.assert_allclose(harmonics.numpy(), expected, rtol=1e-6, atol=1e-6)


def test_embedding_supports_dynamic_shapes_and_empty_triplets():
    for num_spherical, num_radial in ((1, 1), (3, 4), (7, 6)):
        embedding = SphericalFourierBesselEmbedding(num_spherical, num_radial)
        dist = paddle.to_tensor([0.8, 1.2], dtype="float32")
        angle = paddle.to_tensor([0.5], dtype="float32")
        torsion = paddle.to_tensor([0.2], dtype="float32")
        idx_kj = paddle.to_tensor([1], dtype="int64")
        angle_embedding, torsion_embedding = embedding(
            dist,
            paddle.cos(angle),
            paddle.sin(angle),
            paddle.cos(torsion),
            paddle.sin(torsion),
            idx_kj,
        )
        assert angle_embedding.shape == [1, num_spherical * num_radial]
        assert torsion_embedding.shape == [
            1,
            num_spherical * num_spherical * num_radial,
        ]

    empty = paddle.empty([0], dtype="float32")
    empty_index = paddle.empty([0], dtype="int64")
    angle_embedding, torsion_embedding = embedding(
        dist, empty, empty, empty, empty, empty_index
    )
    assert angle_embedding.shape == [0, 42]
    assert torsion_embedding.shape == [0, 294]


def test_radius_graph_uses_edges_as_endpoint_indices():
    xyz_blocks = [
        "3\nwater\nO 0 0 0\nH 0.96 0 0\nH -0.24 0.93 0\n",
        (
            "5\nmethane\nC 0 0 0\nH 0.63 0.63 0.63\n"
            "H -0.63 -0.63 0.63\nH -0.63 0.63 -0.63\n"
            "H 0.63 -0.63 -0.63\n"
        ),
    ]
    converter = RadiusGraphConverter(cutoff=5.0, return_triplet_indices=True)
    graphs = converter([Chem.MolFromXYZBlock(block) for block in xyz_blocks])
    for graph in graphs:
        assert "ti_i" not in graph.edge_feat
        assert "ti_j" not in graph.edge_feat

    batch = RadiusGraphCollator()(
        [{"graph": graph, "id": index} for index, graph in enumerate(graphs)]
    )
    graph = batch["graph"].tensor()
    edge_index = paddle.transpose(graph.edges.astype("int64"), [1, 0])
    triplet_indices = {
        "idx_kj": graph.edge_feat["ti_idx_kj"].astype("int64"),
        "idx_ji": graph.edge_feat["ti_idx_ji"].astype("int64"),
    }
    result = compute_geometry(
        graph.node_feat["cart_coords"], edge_index, triplet_indices
    )
    np.testing.assert_array_equal(result[3].numpy(), edge_index[1].numpy())
    np.testing.assert_array_equal(result[4].numpy(), edge_index[0].numpy())


def test_radius_graph_batch_matches_individual_energy_and_force():
    paddle.seed(2026)
    xyz_blocks = [
        "3\nwater\nO 0 0 0\nH 0.96 0 0\nH -0.24 0.93 0\n",
        (
            "5\nmethane\nC 0 0 0\nH 0.63 0.63 0.63\n"
            "H -0.63 -0.63 0.63\nH -0.63 0.63 -0.63\n"
            "H 0.63 -0.63 -0.63\n"
        ),
    ]
    converter = RadiusGraphConverter(cutoff=5.0, return_triplet_indices=True)

    def build_graphs():
        return converter([Chem.MolFromXYZBlock(block) for block in xyz_blocks])

    model = SphereNet(
        derivative_properties=("force",),
        property_name="energy",
        num_layers=1,
        hidden_channels=16,
        int_emb_size=8,
        basis_emb_size_dist=4,
        basis_emb_size_angle=4,
        basis_emb_size_torsion=4,
        out_emb_channels=16,
        num_spherical=2,
        num_radial=2,
        num_output_layers=1,
    )
    model.eval()

    individual = [model.predict(graph) for graph in build_graphs()]
    batch = RadiusGraphCollator()(
        [{"graph": graph, "id": index} for index, graph in enumerate(build_graphs())]
    )
    prediction = model(batch, return_loss=False, return_prediction=True)["pred_dict"]

    np.testing.assert_allclose(
        prediction["energy"].numpy(),
        np.concatenate([result["energy"] for result in individual]),
        rtol=2e-5,
        atol=2e-6,
    )
    np.testing.assert_allclose(
        prediction["force"].numpy(),
        np.concatenate([result["force"] for result in individual]),
        rtol=2e-4,
        atol=2e-5,
    )


def test_geometry_components_match_angle_and_torsion_definitions():
    molecule = BuildMolecule(format="dict", sanitize=False)(
        {
            "atomic_numbers": np.array([6, 7, 8, 1]),
            "positions": np.array(
                [
                    [0.1, 0.2, 0.3],
                    [1.2, 0.1, 0.4],
                    [-0.2, 1.1, 0.5],
                    [0.4, -0.3, 1.4],
                ],
                dtype=np.float32,
            ),
        }
    )
    graph = RadiusGraphConverter(cutoff=5.0, return_triplet_indices=True)(
        molecule
    ).tensor()
    pos = graph.node_feat["cart_coords"]
    edge_index = paddle.transpose(graph.edges.astype("int64"), [1, 0])
    triplet_indices = {
        key: graph.edge_feat[f"ti_{key}"].astype("int64")
        for key in ("idx_kj", "idx_ji")
    }
    assert "ti_idx_qj" not in graph.edge_feat

    _, angle, torsion, i, j, idx_kj, idx_ji = compute_geometry(
        pos, edge_index, triplet_indices
    )

    src, dst = edge_index
    edge_vector = pos[dst] - pos[src]
    axis = edge_vector[idx_ji]
    reference = -edge_vector[idx_kj]
    angle_cross = paddle.linalg.cross(axis, reference)
    expected_angle = paddle.atan2(
        paddle.sqrt(paddle.sum(angle_cross * angle_cross, axis=-1)),
        paddle.sum(axis * reference, axis=-1),
    )

    expected_torsion = []
    src_array = src.numpy()
    dst_array = dst.numpy()
    for triplet_id, ji_edge in enumerate(idx_ji.numpy()):
        candidate_edges = np.flatnonzero(
            (dst_array == src_array[ji_edge]) & (src_array != dst_array[ji_edge])
        )
        candidate_index = paddle.to_tensor(candidate_edges, dtype="int64")
        candidate_axis = axis[triplet_id].tile([len(candidate_edges), 1])
        candidate_reference = reference[triplet_id].tile([len(candidate_edges), 1])
        candidate = -edge_vector[candidate_index]
        reference_plane = paddle.linalg.cross(candidate_axis, candidate_reference)
        candidate_plane = paddle.linalg.cross(candidate_axis, candidate)
        torsion_values = paddle.atan2(
            paddle.sum(
                paddle.linalg.cross(reference_plane, candidate_plane) * candidate_axis,
                axis=-1,
            )
            / paddle.sqrt(paddle.sum(candidate_axis * candidate_axis, axis=-1)),
            paddle.sum(reference_plane * candidate_plane, axis=-1),
        )
        torsion_values = paddle.where(
            torsion_values <= 0,
            torsion_values + 2 * np.pi,
            torsion_values,
        )
        expected_torsion.append(torsion_values.min())
    expected_torsion = paddle.stack(expected_torsion)

    np.testing.assert_allclose(
        angle[0].numpy(),
        paddle.cos(expected_angle).numpy(),
        rtol=1e-6,
        atol=1e-7,
    )
    np.testing.assert_allclose(
        angle[1].numpy(),
        paddle.sin(expected_angle).numpy(),
        rtol=1e-6,
        atol=1e-7,
    )
    np.testing.assert_allclose(
        torsion[0].numpy(),
        paddle.cos(expected_torsion).numpy(),
        rtol=1e-6,
        atol=1e-7,
    )
    np.testing.assert_allclose(
        torsion[1].numpy(),
        paddle.sin(expected_torsion).numpy(),
        rtol=1e-6,
        atol=1e-7,
    )


def test_predict_returns_energy_and_force():
    molecule = Chem.MolFromXYZBlock("3\nwater\nO 0 0 0\nH 0.96 0 0\nH -0.24 0.93 0\n")
    graph = RadiusGraphConverter(cutoff=5.0, return_triplet_indices=True)(molecule)
    model = SphereNet(
        derivative_properties=("force",),
        property_name="energy",
        num_layers=0,
        hidden_channels=16,
        int_emb_size=8,
        basis_emb_size_dist=4,
        basis_emb_size_angle=4,
        basis_emb_size_torsion=4,
        out_emb_channels=16,
        num_spherical=2,
        num_radial=2,
        num_output_layers=1,
    )

    prediction = model.predict(graph)

    assert prediction["energy"].shape == (1, 1)
    assert prediction["force"].shape == (3, 3)
    assert np.isfinite(prediction["energy"]).all()
    assert np.isfinite(prediction["force"]).all()


def test_model_scaling_returns_physical_energy_and_force():
    molecule = Chem.MolFromXYZBlock("3\nwater\nO 0 0 0\nH 0.96 0 0\nH -0.24 0.93 0\n")
    graph = RadiusGraphConverter(cutoff=5.0, return_triplet_indices=True)(molecule)
    model = SphereNet(
        derivative_properties=("force",),
        property_name="energy",
        num_layers=0,
        hidden_channels=16,
        int_emb_size=8,
        basis_emb_size_dist=4,
        basis_emb_size_angle=4,
        basis_emb_size_torsion=4,
        out_emb_channels=16,
        num_spherical=2,
        num_radial=2,
        num_output_layers=1,
        data_mean=-10.0,
        data_std=2.5,
    )
    model.eval()

    normalized_energy, pos, _ = model._forward({"graph": graph})
    normalized_gradient = paddle.grad(normalized_energy.sum(), pos)[0]
    result = model({"graph": graph}, return_loss=False, return_prediction=True)[
        "pred_dict"
    ]

    np.testing.assert_allclose(
        result["energy"].numpy(),
        normalized_energy.numpy() * 2.5 - 10.0,
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        result["force"].numpy(),
        -normalized_gradient.numpy() * 2.5,
        rtol=1e-5,
        atol=1e-6,
    )
    assert "data_mean" in model.state_dict()
    assert "data_std" in model.state_dict()


def test_force_loss_produces_parameter_gradients():
    paddle.seed(7)
    molecule = BuildMolecule(format="dict", sanitize=False)(
        {
            "atomic_numbers": np.array([6, 7, 8, 1]),
            "positions": np.array(
                [
                    [0.1, 0.2, 0.3],
                    [1.2, 0.1, 0.4],
                    [-0.2, 1.1, 0.5],
                    [0.4, -0.3, 1.4],
                ],
                dtype=np.float32,
            ),
        }
    )
    graph = RadiusGraphConverter(cutoff=5.0, return_triplet_indices=True)(molecule)
    model = SphereNet(
        derivative_properties=("force",),
        property_name="energy",
        num_layers=1,
        hidden_channels=16,
        int_emb_size=8,
        basis_emb_size_dist=4,
        basis_emb_size_angle=4,
        basis_emb_size_torsion=4,
        out_emb_channels=16,
        num_spherical=2,
        num_radial=2,
        num_output_layers=1,
        output_init="zeros",
        loss_weights_dict={"energy": 1.0, "force": 100.0},
    )
    model.train()
    energy, _, _ = model._forward({"graph": graph})
    force = paddle.to_tensor(
        [
            [1.0, -0.4, 0.5],
            [-0.3, 0.2, 0.7],
            [0.1, 0.4, -0.8],
            [-0.8, -0.2, -0.4],
        ],
        dtype="float32",
    )
    result = model(
        {
            "graph": graph,
            "energy": energy.detach(),
            "force": force,
        }
    )

    expected_force_loss = paddle.nn.functional.l1_loss(
        result["pred_dict"]["force"], force
    )
    assert set(result["loss_dict"]) == {"energy", "force", "loss"}
    np.testing.assert_allclose(
        result["loss_dict"]["force"].item(),
        expected_force_loss.item(),
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        result["loss_dict"]["loss"].item(),
        100.0 * expected_force_loss.item(),
        rtol=1e-6,
    )

    result["loss_dict"]["loss"].backward()
    gradients = [
        parameter.grad for parameter in model.parameters() if parameter.grad is not None
    ]

    assert gradients
    assert all(bool(paddle.isfinite(gradient).all()) for gradient in gradients)
    assert float(sum(gradient.abs().sum() for gradient in gradients)) > 0


def test_training_and_eval_paths_match_energy_and_force():
    paddle.seed(13)
    molecule = BuildMolecule(format="dict", sanitize=False)(
        {
            "atomic_numbers": np.array([6, 7, 8, 1]),
            "positions": np.array(
                [
                    [0.1, 0.2, 0.3],
                    [1.2, 0.1, 0.4],
                    [-0.2, 1.1, 0.5],
                    [0.4, -0.3, 1.4],
                ],
                dtype=np.float32,
            ),
        }
    )
    graph = RadiusGraphConverter(cutoff=5.0, return_triplet_indices=True)(molecule)
    model = SphereNet(
        derivative_properties=("force",),
        property_name="energy",
        num_layers=1,
        hidden_channels=16,
        int_emb_size=8,
        basis_emb_size_dist=4,
        basis_emb_size_angle=4,
        basis_emb_size_torsion=4,
        out_emb_channels=16,
        num_spherical=2,
        num_radial=2,
        num_output_layers=1,
    )

    model.train()
    train_energy, train_pos, _ = model._forward({"graph": graph})
    train_force = -paddle.grad(train_energy.sum(), train_pos)[0]

    model.eval()
    eval_energy, eval_pos, _ = model._forward({"graph": graph})
    eval_force = -paddle.grad(eval_energy.sum(), eval_pos)[0]

    np.testing.assert_allclose(
        train_energy.numpy(), eval_energy.numpy(), rtol=1e-5, atol=2e-5
    )
    np.testing.assert_allclose(
        train_force.numpy(), eval_force.numpy(), rtol=1e-5, atol=2e-5
    )


def test_force_loss_supports_molecules_without_triplets():
    molecule = BuildMolecule(format="dict", sanitize=False)(
        {
            "atomic_numbers": np.array([1, 1]),
            "positions": np.array(
                [[0.0, 0.0, 0.0], [0.74, 0.0, 0.0]], dtype=np.float32
            ),
        }
    )
    graph = RadiusGraphConverter(cutoff=5.0, return_triplet_indices=True)(molecule)
    assert graph.edge_feat["ti_idx_kj"].shape == (0,)

    model = SphereNet(
        derivative_properties=("force",),
        property_name="energy",
        num_layers=1,
        hidden_channels=16,
        int_emb_size=8,
        basis_emb_size_dist=4,
        basis_emb_size_angle=4,
        basis_emb_size_torsion=4,
        out_emb_channels=16,
        num_spherical=2,
        num_radial=2,
        num_output_layers=1,
    )
    result = model(
        {
            "graph": graph,
            "energy": np.zeros([1], dtype=np.float32),
            "force": np.zeros([2, 3], dtype=np.float32),
        }
    )
    loss = result["loss_dict"]["loss"]

    assert bool(paddle.isfinite(loss))
    assert bool(paddle.isfinite(result["pred_dict"]["energy"]).all())
    assert bool(paddle.isfinite(result["pred_dict"]["force"]).all())
    loss.backward()
    gradients = [
        parameter.grad for parameter in model.parameters() if parameter.grad is not None
    ]
    assert gradients
    assert all(bool(paddle.isfinite(gradient).all()) for gradient in gradients)


def test_force_matches_finite_difference_with_fixed_torsion_branch():
    paddle.seed(11)
    atomic_numbers = np.array([6, 7, 8])
    positions = np.array(
        [
            [0.1, 0.2, 0.3],
            [1.2, 0.1, 0.4],
            [-0.2, 1.1, 0.5],
        ],
        dtype=np.float32,
    )
    converter = RadiusGraphConverter(cutoff=5.0, return_triplet_indices=True)

    def build_graph(coordinates):
        molecule = BuildMolecule(format="dict", sanitize=False)(
            {
                "atomic_numbers": atomic_numbers,
                "positions": coordinates,
            }
        )
        return converter(molecule)

    model = SphereNet(
        derivative_properties=("force",),
        property_name="energy",
        num_layers=1,
        hidden_channels=16,
        int_emb_size=8,
        basis_emb_size_dist=4,
        basis_emb_size_angle=4,
        basis_emb_size_torsion=4,
        out_emb_channels=16,
        num_spherical=2,
        num_radial=2,
        num_output_layers=1,
        output_init="glorot_orthogonal",
    )
    model.eval()

    energy, position_tensor, _ = model._forward({"graph": build_graph(positions)})
    actual = paddle.grad(energy.sum(), position_tensor)[0][0, 0].item()

    epsilon = 1e-3
    positions_plus = positions.copy()
    positions_minus = positions.copy()
    positions_plus[0, 0] += epsilon
    positions_minus[0, 0] -= epsilon
    energy_plus = model._forward({"graph": build_graph(positions_plus)})[0].sum()
    energy_minus = model._forward({"graph": build_graph(positions_minus)})[0].sum()
    expected = ((energy_plus - energy_minus) / (2 * epsilon)).item()

    np.testing.assert_allclose(actual, expected, rtol=5e-3, atol=2e-3)

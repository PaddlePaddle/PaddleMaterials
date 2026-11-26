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

"""GeoDiff: model contract, data, sampling, SDF output and conformer evaluation."""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import paddle
import pytest
from omegaconf import OmegaConf
from rdkit import Chem
from rdkit.Chem import rdDepictor

from ppmat.datasets import GeomConformerDataset
from ppmat.datasets import build_dataloader
from ppmat.metrics import build_metric
from ppmat.metrics.geodiff_conformers import read_covmat_inputs
from ppmat.models import GeoDiff
from ppmat.models import build_model
from ppmat.trainer.base_trainer import BaseTrainer
from ppmat.utils.molecular_conformers import write_molecular_conformers


def _model(**params):
    model = build_model(
        {
            "__class_name__": "GeoDiff",
            "__init_params__": {
                "hidden_dim": 8,
                "num_convs": 1,
                "num_convs_local": 1,
                "cutoff": 1.5,
                **params,
            },
        }
    )
    model.eval()
    return model


def test_denoise_preserves_bonds_and_strict_same_graph_radius_edges():
    with paddle.no_grad():
        result = _model().denoise_step(
            atom_type=paddle.to_tensor([6, 1, 8, 1, 6], dtype="int64"),
            pos=paddle.to_tensor(
                [[0.0, 0, 0], [1.0, 0, 0], [2.5, 0, 0], [-0.25, 0, 0], [0.0, 0, 0]],
                dtype="float32",
            ),
            bond_index=paddle.to_tensor([[1, 0], [0, 1]], dtype="int64"),
            bond_type=paddle.to_tensor([1, 1], dtype="int64"),
            batch=paddle.to_tensor([0, 0, 0, 0, 1], dtype="int64"),
            time_step=paddle.to_tensor([0, 0], dtype="int64"),
            return_edges=True,
        )
    global_score, local_score, edges, types, lengths, local = result
    assert edges.tolist() == [[0, 0, 1, 1, 3, 3], [1, 3, 0, 3, 0, 1]]
    assert types.tolist() == [1, 0, 1, 0, 0, 0]
    assert local.tolist() == [True, False, True, False, False, False]
    assert lengths.flatten().tolist() == [1.0, 0.25, 1.0, 1.25, 0.25, 1.25]
    assert list(global_score.shape) == [6, 1]
    assert list(local_score.shape) == [2, 1]


def test_denoise_caps_radius_neighbors_in_source_order_not_distance_order():
    with paddle.no_grad():
        _, _, edges, types, _, local = _model().denoise_step(
            atom_type=paddle.full([35], 6, dtype="int64"),
            pos=paddle.to_tensor(
                [[1.0 - i / 64, 0, 0] for i in range(35)], dtype="float32"
            ),
            bond_index=paddle.empty([2, 0], dtype="int64"),
            bond_type=paddle.empty([0], dtype="int64"),
            batch=paddle.zeros([35], dtype="int64"),
            time_step=paddle.zeros([1], dtype="int64"),
            return_edges=True,
        )
    expected = [
        (source, target)
        for source in range(33)
        for target in range(35)
        if source != target and (source < 32 or target < 32)
    ]
    assert list(zip(*edges.tolist())) == expected
    assert types.tolist() == [0] * (35 * 32)
    assert local.tolist() == [False] * (35 * 32)


def test_forward_exposes_scalar_diffusion_losses_and_gradients():
    model = _model()
    model.train()
    output = model(
        {
            "atom_types": paddle.to_tensor([6, 1, 1], dtype="int64"),
            "cart_coords": paddle.to_tensor(
                [[0.0, 0, 0], [1.0, 0, 0], [0, 1.0, 0]], dtype="float32"
            ),
            "edge_index": paddle.to_tensor([[0, 1], [1, 0]], dtype="int64"),
            "edge_type": paddle.to_tensor([1, 1], dtype="int64"),
            "batch": paddle.zeros([3], dtype="int64"),
            "ptr": paddle.to_tensor([0, 3], dtype="int64"),
            "num_graphs": 1,
        }
    )
    losses = output["loss_dict"]
    assert set(losses) == {"loss", "loss_global", "loss_local"}
    assert all(
        loss.ndim == 0 and bool(paddle.isfinite(loss)) for loss in losses.values()
    )
    assert bool(
        paddle.allclose(losses["loss"], losses["loss_global"] + losses["loss_local"])
    )
    losses["loss"].backward()
    assert any(
        parameter.grad is not None for parameter in model.model_global.parameters()
    )
    assert any(
        parameter.grad is not None for parameter in model.model_local.parameters()
    )


def test_geodiff_rejects_singular_noise_schedule():
    with pytest.raises(ValueError, match="sigmoid"):
        _model(beta_schedule="jsd")


def test_geodiff_chain_assigns_distinct_second_and_third_bond_hops():
    atom_types = paddle.full([4], 6, dtype="int64")
    pos = paddle.to_tensor([[0.0, 0, 0], [1.0, 0, 0], [2.0, 0, 0], [3.0, 0, 0]])
    bonds = paddle.to_tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]])
    with paddle.no_grad():
        _, _, edges, types, _, mask = _model().denoise_step(
            atom_types,
            pos,
            bonds,
            paddle.ones([6], dtype="int64"),
            paddle.zeros([4], dtype="int64"),
            paddle.zeros([1], dtype="int64"),
            return_edges=True,
        )
    edge_types = dict(zip(map(tuple, edges.T.tolist()), types.tolist()))
    count = len(Chem.BondType.names)
    assert edge_types == {
        (0, 1): 1,
        (0, 2): count + 1,
        (0, 3): count + 2,
        (1, 0): 1,
        (1, 2): 1,
        (1, 3): count + 1,
        (2, 0): count + 1,
        (2, 1): 1,
        (2, 3): 1,
        (3, 0): count + 2,
        (3, 1): count + 1,
        (3, 2): 1,
    }
    assert mask.tolist() == [True] * 12


# Dataset and collator


@pytest.mark.parametrize("num_workers", [0, 1])
def test_geom_conformer_loader_preserves_isolated_atoms_and_offsets_bonds(
    tmp_path, num_workers
):
    path = tmp_path / "val_data_5k.npz"
    np.savez_compressed(
        path,
        atom_type=np.array([6, 8, 1, 1], dtype=np.int64),
        pos=np.array([[0, 0, 0], [2, 0, 0], [2, 1, 0], [3, 0, 0]], dtype=np.float32),
        edge_index=np.array([[0, 0, 1, 2], [1, 2, 0, 0]], dtype=np.int64),
        edge_type=np.array([1, 1, 1, 1], dtype=np.int64),
        node_ptr=np.array([0, 1, 4], dtype=np.int64),
        edge_ptr=np.array([0, 0, 4], dtype=np.int64),
    )
    loader = build_dataloader(
        {
            "dataset": {
                "__class_name__": "GeomConformerDataset",
                "__init_params__": {"path": str(tmp_path), "split": "val"},
            },
            "sampler": {
                "__class_name__": "BatchSampler",
                "__init_params__": {"batch_size": 2},
            },
            "loader": {
                "num_workers": num_workers,
                "collate_fn": "GeomConformerCollator",
            },
        }
    )
    (batch,) = list(loader)
    assert batch["atom_types"].tolist() == [6, 8, 1, 1]
    assert batch["ptr"].tolist() == [0, 1, 4]
    assert batch["batch"].tolist() == [0, 1, 1, 1]
    assert batch["edge_index"].tolist() == [[1, 1, 2, 3], [2, 3, 1, 1]]
    assert batch["edge_type"].tolist() == [1, 1, 1, 1]
    assert batch["cart_coords"][:, 0].tolist() == [0, 2, 2, 3]


def _packed_file(tmp_path):
    """Write three reference rows of two molecules, authored as CO, CO, N."""
    path = tmp_path / "test_data_1k.npz"
    np.savez_compressed(
        path,
        atom_type=np.array([6, 8, 6, 8, 7], dtype=np.int64),
        pos=np.zeros((5, 3), dtype=np.float32),
        edge_index=np.array([[0, 1, 2, 3], [1, 0, 3, 2]], dtype=np.int64),
        edge_type=np.array([1, 1, 1, 1], dtype=np.int64),
        node_ptr=np.array([0, 2, 4, 5], dtype=np.int64),
        edge_ptr=np.array([0, 2, 4, 4], dtype=np.int64),
        smiles=np.array(["CO", "CO", "N"]),
    )
    return path


def test_packed_dataset_keeps_first_row_graph_and_reference_count(tmp_path):
    dataset = GeomConformerDataset(
        str(_packed_file(tmp_path)), "test", pack_by_molecule=True
    )
    assert len(dataset) == 2
    assert dataset[0].atom_types.tolist() == [6, 8]
    assert dataset[0].num_reference == 2
    assert dataset[1].atom_types.tolist() == [7]
    assert dataset[1].num_reference == 1


def test_packed_loader_passes_reference_count_to_the_model(tmp_path):
    _packed_file(tmp_path)
    loader = build_dataloader(
        {
            "dataset": {
                "__class_name__": "GeomConformerDataset",
                "__init_params__": {
                    "path": str(tmp_path),
                    "split": "test",
                    "pack_by_molecule": True,
                },
            },
            "sampler": {
                "__class_name__": "BatchSampler",
                "__init_params__": {"batch_size": 2},
            },
            "loader": {"num_workers": 0, "collate_fn": "GeomConformerCollator"},
        }
    )
    (batch,) = list(loader)
    assert batch["atom_types"].tolist() == [6, 8, 7]
    assert batch["num_reference"].tolist() == [2, 1]


# Sampling and the sampling CLI


@pytest.mark.parametrize(
    ("preset", "with_rdkit"),
    [
        pytest.param("qm9", False, id="qm9-numeric"),
        pytest.param("qm9", True, id="qm9-rdkit"),
        pytest.param("drugs", True, id="drugs-rdkit"),
    ],
)
def test_molecular_graph_cli_writes_two_conformers_without_training_data(
    tmp_path, preset, with_rdkit
):
    input_path = tmp_path / "ethylene.npz"
    missing_path = tmp_path / "unavailable.npz"
    metadata = {}
    if with_rdkit:
        params = Chem.SmilesParserParams()
        params.removeHs = False
        mol = Chem.MolFromSmiles("[2H]C([H])=C([H])[H]", params)
        binary = mol.ToBinary()
        metadata = {
            "rdmol_ptr": np.array([0, len(binary)], dtype=np.int64),
            "rdmol_data": np.frombuffer(binary, dtype=np.uint8),
        }
    np.savez_compressed(
        input_path,
        atom_type=np.array([1, 6, 1, 6, 1, 1], dtype=np.int64),
        edge_index=np.array(
            [[0, 1, 1, 1, 2, 3, 3, 3, 4, 5], [1, 0, 2, 3, 1, 1, 4, 5, 3, 3]],
            dtype=np.int64,
        ),
        edge_type=np.array([1, 1, 1, 2, 1, 2, 1, 1, 1, 1], dtype=np.int64),
        node_ptr=np.array([0, 6], dtype=np.int64),
        edge_ptr=np.array([0, 10], dtype=np.int64),
        smiles=np.array(["C=C"]),
        **metadata,
    )
    model_params = {"hidden_dim": 8, "num_convs": 1, "num_convs_local": 1}
    repo = Path(__file__).resolve().parents[1]
    config = OmegaConf.load(
        repo / "structure_generation/configs/geodiff" / f"geodiff_{preset}_default.yaml"
    )
    config.Model.__init_params__.update(model_params)
    checkpoint_path = tmp_path / "geodiff.pdparams"
    kwargs = OmegaConf.to_container(config.Model.__init_params__, resolve=True)
    paddle.save(GeoDiff(**kwargs).state_dict(), str(checkpoint_path))
    config.Global.data_dir = str(missing_path)
    config_path = tmp_path / "geodiff.yaml"
    output_path = tmp_path / "output"
    config.Sampler.output_dir = str(output_path)
    OmegaConf.save(config, config_path)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "structure_generation.sample",
            "--mode=by_molecular_graph",
            f"--config_path={config_path}",
            f"--checkpoint_path={checkpoint_path}",
            f"--input_path={input_path}",
            f"--output_path={output_path}",
            "Sampler.model_sample_params.num_confs=2",
            "Sampler.model_sample_params.n_steps=2",
        ],
        cwd=repo,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    molecules = list(
        Chem.SDMolSupplier(str(output_path / "conformers.sdf"), removeHs=False)
    )
    assert len(molecules) == 2
    assert not list(output_path.rglob("*.cif"))
    coordinates = []
    for mol in molecules:
        assert mol is not None
        assert [atom.GetAtomicNum() for atom in mol.GetAtoms()] == [1, 6, 1, 6, 1, 1]
        assert mol.GetAtomWithIdx(0).GetIsotope() == (2 if with_rdkit else 0)
        assert {
            (
                *sorted((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())),
                str(bond.GetBondType()),
            )
            for bond in mol.GetBonds()
        } == {
            (0, 1, "SINGLE"),
            (1, 2, "SINGLE"),
            (1, 3, "DOUBLE"),
            (3, 4, "SINGLE"),
            (3, 5, "SINGLE"),
        }
        positions = mol.GetConformer().GetPositions()
        assert positions.shape == (6, 3)
        assert np.isfinite(positions).all()
        coordinates.append(positions)
    assert not np.allclose(coordinates[0], coordinates[1], atol=1e-4)


def _reference_file(tmp_path, rows):
    """Write ``rows`` reference conformers of one ethylene molecule."""
    path = tmp_path / "test_data_1k.npz"
    np.savez_compressed(
        path,
        atom_type=np.tile(np.array([1, 6, 1, 6, 1, 1], dtype=np.int64), rows),
        edge_index=np.tile(
            np.array(
                [[0, 1, 1, 1, 2, 3, 3, 3, 4, 5], [1, 0, 2, 3, 1, 2, 1, 1, 3, 3]],
                dtype=np.int64,
            ),
            (1, rows),
        ),
        edge_type=np.tile(
            np.array([1, 1, 1, 2, 1, 2, 1, 1, 1, 1], dtype=np.int64), rows
        ),
        node_ptr=np.arange(rows + 1) * 6,
        edge_ptr=np.arange(rows + 1) * 10,
        smiles=np.array(["C=C"] * rows),
    )
    return path


def _packed_batch(path):
    loader = build_dataloader(
        {
            "dataset": {
                "__class_name__": "GeomConformerDataset",
                "__init_params__": {
                    "path": str(path.parent),
                    "split": "test",
                    "pack_by_molecule": True,
                },
            },
            "sampler": {
                "__class_name__": "BatchSampler",
                "__init_params__": {"batch_size": 1},
            },
            "loader": {"num_workers": 0, "collate_fn": "GeomConformerCollator"},
        }
    )
    (batch,) = list(loader)
    return batch


def _tiny_model():
    repo = Path(__file__).resolve().parents[1]
    config = OmegaConf.load(
        repo / "structure_generation/configs/geodiff/geodiff_qm9_default.yaml"
    )
    config.Model.__init_params__.update(
        {"hidden_dim": 8, "num_convs": 1, "num_convs_local": 1}
    )
    kwargs = OmegaConf.to_container(config.Model.__init_params__, resolve=True)
    return GeoDiff(**kwargs)


def test_packed_molecule_samples_two_conformers_per_reference(tmp_path):
    batch = _packed_batch(_reference_file(tmp_path, 3))
    (record,) = _tiny_model().sample(batch, num_confs="2x", n_steps=2)["pred"]
    assert record["cart_coords"].shape == (6, 6, 3)
    assert record["atom_types"].tolist() == [1, 6, 1, 6, 1, 1]


def test_fixed_num_confs_is_unchanged_for_a_packed_molecule(tmp_path):
    batch = _packed_batch(_reference_file(tmp_path, 3))
    (record,) = _tiny_model().sample(batch, num_confs=3, n_steps=2)["pred"]
    assert record["cart_coords"].shape == (3, 6, 3)


def test_multiple_of_references_requires_packed_molecules(tmp_path):
    batch = _packed_batch(_reference_file(tmp_path, 1))
    del batch["num_reference"]
    with pytest.raises(ValueError, match="pack"):
        _tiny_model().sample(batch, num_confs="2x", n_steps=2)


# SDF output


def _ring():
    mol = Chem.MolFromSmiles("c1ccccc1")
    coords = np.arange(18, dtype=np.float32).reshape((1, 6, 3)) / 7.0
    return {"rdmol": mol.ToBinary(), "cart_coords": coords}


def test_sdf_keeps_aromatic_bond_encoding_and_coordinate_order(tmp_path):
    path = tmp_path / "conformers.sdf"
    record = _ring()
    write_molecular_conformers(path, [record])
    (actual,) = Chem.SDMolSupplier(str(path), sanitize=False, removeHs=False)
    assert {(str(b.GetBondType())) for b in actual.GetBonds()} == {"AROMATIC"}
    np.testing.assert_allclose(
        actual.GetConformer().GetPositions(),
        record["cart_coords"][0],
        atol=1e-4,
        rtol=0,
    )


def test_sdf_source_is_the_conditioning_graph_not_generated_handedness(tmp_path):
    mol = Chem.MolFromSmiles("F[C@](Cl)(Br)I")
    source = Chem.MolToSmiles(mol)
    points = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, 1], [-1, 1, -1], [1, -1, -1]])
    coords = np.stack([points, points * [-1, 1, 1]]).astype(np.float64)
    path = tmp_path / "chiral.sdf"
    write_molecular_conformers(path, [{"rdmol": mol.ToBinary(), "cart_coords": coords}])
    poses = list(Chem.SDMolSupplier(str(path), removeHs=False))
    assert len(poses) == 2
    assert [pose.GetProp("GEODIFF_SOURCE_SMILES") for pose in poses] == [source, source]
    assert len({Chem.MolToSmiles(pose) for pose in poses}) == 2
    for pose, expected in zip(poses, coords):
        np.testing.assert_array_equal(pose.GetConformer().GetPositions(), expected)


def test_invalid_graph_does_not_replace_existing_conformers(tmp_path):
    path = tmp_path / "conformers.sdf"
    write_molecular_conformers(path, [_ring()])
    previous = path.read_bytes()
    invalid = {
        "rdmol": None,
        "atom_types": [7, 1, 1, 1, 1],
        "edge_index": np.array([[0, 0, 0, 0], [1, 2, 3, 4]]),
        "edge_type": np.ones(4, dtype=np.int64),
        "cart_coords": np.zeros([1, 5, 3], dtype=np.float32),
    }
    with pytest.raises(Chem.AtomValenceException):
        write_molecular_conformers(path, [_ring(), invalid])
    assert path.read_bytes() == previous


def test_molecule_rdkit_cannot_sanitise_is_written_with_its_source_tag(tmp_path):
    # RDKit rejects a +3 sulphur with five bonds.
    mol = Chem.MolFromSmiles("[H][S+3]([H])([H])([H])[H]", sanitize=False)
    mol.UpdatePropertyCache(strict=False)
    with pytest.raises(Chem.AtomValenceException):
        Chem.RemoveHs(mol)
    coords = np.arange(18, dtype=np.float32).reshape((1, 6, 3)) / 7.0
    path = tmp_path / "conformers.sdf"
    write_molecular_conformers(path, [{"rdmol": mol.ToBinary(), "cart_coords": coords}])
    (actual,) = Chem.SDMolSupplier(str(path), sanitize=False, removeHs=False)
    tagged = Chem.MolFromSmiles(actual.GetProp("GEODIFF_SOURCE_SMILES"), sanitize=False)
    assert sorted(a.GetAtomicNum() for a in tagged.GetAtoms()) == [1] * 5 + [16]
    assert tagged.GetNumBonds() == 5
    np.testing.assert_allclose(
        actual.GetConformer().GetPositions(), coords[0], atol=1e-4, rtol=0
    )


# COV/MAT metrics


def _group(smiles, ref_lengths, gen_lengths):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    mol.AddConformer(Chem.Conformer(mol.GetNumAtoms()))

    def positions(lengths, hydrogen_y):
        xyz = np.zeros((len(lengths), mol.GetNumAtoms(), 3))
        xyz[:, 1, 0] = lengths
        xyz[:, 2:, 1] = hydrogen_y
        return xyz

    return smiles, mol, positions(ref_lengths, 3), positions(gen_lengths, 5)


def test_bidirectional_heavy_atom_metrics_filter_then_use_first_budget():
    from ppmat.metrics.geodiff_covmat import compute_covmat

    # After alignment of a diatomic, RMSD is half the bond length difference.
    # The fifth conformer (length 3) would perfectly match the second reference.
    groups = [
        _group("CC", [1, 3], [1, 1, 1, 9, 3]),
        _group("CO", [2], [2, 2]),
        _group("CF", [1, 1], [1]),
        _group("CN", [2], []),
        _group("C.C", [2], [2, 2]),
    ]
    result = compute_covmat(iter(groups), thresholds=[0, 1])
    assert result["counts"] == {
        "input_molecules": 5,
        "eligible_molecules": 2,
        "disconnected": 1,
        "missing_generated": 1,
        "insufficient_generated": 1,
    }
    assert result["thresholds"] == [0, 1]
    first, second = result["per_molecule"]
    assert first["smiles"] == "CC" and second["smiles"] == "CO"
    assert (
        first["num_reference"],
        first["num_generated"],
        first["num_generated_used"],
    ) == (2, 5, 4)
    assert first["CoverageR"] == [0.5, 1]
    assert first["CoverageP"] == [0.75, 0.75]
    assert first["MatchingR"] == pytest.approx(0.5, abs=1e-8)
    assert first["MatchingP"] == pytest.approx(0.75, abs=1e-8)
    for metric, center, spread in (
        ("CoverageR", [0.75, 1], [0.25, 0]),
        ("CoverageP", [0.875, 0.875], [0.125, 0.125]),
        ("MatchingR", 0.25, 0.25),
        ("MatchingP", 0.375, 0.375),
    ):
        for stat, expected in (("mean", center), ("median", center), ("std", spread)):
            np.testing.assert_allclose(
                result["summary"][metric][stat], expected, atol=1e-8, rtol=0
            )


def test_all_filtered_is_not_a_perfect_score():
    from ppmat.metrics.geodiff_covmat import compute_covmat

    counts = r"disconnected=1, missing_generated=0, insufficient_generated=1"
    with pytest.raises(ValueError, match=counts):
        compute_covmat([_group("CC", [1], [1]), _group("C.C", [2], [2, 2])])


def test_unmatched_reference_shapes_do_not_block_an_eligible_molecule():
    from ppmat.metrics.geodiff_covmat import compute_covmat

    unmatched = (
        "CN",
        Chem.MolFromSmiles("CN"),
        [np.zeros((2, 3)), np.zeros((7, 3))],
        [],
    )
    result = compute_covmat([unmatched, _group("CC", [1], [1, 1])])
    assert result["counts"]["missing_generated"] == 1
    assert result["counts"]["insufficient_generated"] == 0
    assert [record["smiles"] for record in result["per_molecule"]] == ["CC"]


# Pairing and the evaluation CLI


SOURCE = "F[C@](Cl)(Br)I"
OPPOSITE = "F[C@@](Cl)(Br)I"
# The central atom (index 1) sits at the origin with four labelled neighbours.
TETRAHEDRON = np.array(
    [[1, 1, 1], [0, 0, 0], [-1, -1, 1], [-1, 1, -1], [1, -1, -1]], dtype=float
)


def reference_file(tmp_path, rows, poses=None):
    """Write a GEOM reference NPZ. Each row is (author SMILES, molecule SMILES, x)."""
    atoms, positions, edges, types, binaries = [], [], [], [], []
    nodes, bonds, offsets = [0], [0], [0]
    for index, (_, smiles, x) in enumerate(rows):
        mol = Chem.MolFromSmiles(smiles)
        atoms.extend(a.GetAtomicNum() for a in mol.GetAtoms())
        if poses is None:
            positions.extend([[x, 0, 0]] * mol.GetNumAtoms())
        else:
            positions.extend(poses[index])
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            edges.extend([(i, j), (j, i)])
            types.extend([int(bond.GetBondType())] * 2)
        binary = mol.ToBinary()
        binaries.append(binary)
        offsets.append(offsets[-1] + len(binary))
        nodes.append(len(atoms))
        bonds.append(len(edges))
    path = tmp_path / "reference.npz"
    np.savez(
        path,
        smiles=[key for key, _, _ in rows],
        rdmol_ptr=offsets,
        rdmol_data=np.frombuffer(b"".join(binaries), dtype=np.uint8),
        node_ptr=nodes,
        atom_type=atoms,
        pos=np.asarray(positions, dtype=np.float32),
        edge_ptr=bonds,
        edge_type=np.asarray(types, dtype=np.int64),
        edge_index=np.asarray(edges, dtype=np.int64).reshape(-1, 2).T,
    )
    return path


def generated_file(tmp_path, rows):
    """Write tagged 2D SDF records. Each row is (molecule SMILES, source tag, x)."""
    path = tmp_path / "generated.sdf"
    with Chem.SDWriter(str(path)) as writer:
        for smiles, tag, x in rows:
            mol = Chem.MolFromSmiles(smiles)
            rdDepictor.Compute2DCoords(mol)
            conf = mol.GetConformer()
            origin = conf.GetAtomPosition(0).x
            for i in range(mol.GetNumAtoms()):
                point = conf.GetAtomPosition(i)
                conf.SetAtomPosition(i, (point.x - origin + x, point.y, 0))
            if tag is not None:
                mol.SetProp("GEODIFF_SOURCE_SMILES", tag)
            writer.write(mol)
    return path


def test_source_tag_retains_stereo_changes_and_input_order(tmp_path):
    ref = reference_file(
        tmp_path,
        [(SOURCE, SOURCE, 1), ("OC", "CO", 2), (SOURCE, SOURCE, 3), ("N", "N", 4)],
    )
    sdf = generated_file(
        tmp_path, [("CO", "CO", 8), (OPPOSITE, SOURCE, 9), (SOURCE, SOURCE, 7)]
    )
    groups, stereo = read_covmat_inputs(ref, sdf)
    groups = list(groups)
    assert [key for key, *_ in groups] == [SOURCE, "OC", "N"]
    assert Chem.MolToSmiles(groups[0][1]) == SOURCE
    assert [[p[0, 0] for p in refs] for _, _, refs, _ in groups] == [[1, 3], [2], [4]]
    assert [[p[0, 0] for p in gens] for _, _, _, gens in groups] == [[9, 7], [8], []]
    assert stereo["matched_records"] == 3
    assert stereo["mismatch_count"] == 1
    assert stereo["mismatches_by_source"] == {SOURCE: 1, "CO": 0, "N": 0}


@pytest.mark.parametrize(
    ("smiles", "tag", "error"),
    [
        ("CO", None, "[Rr]e-?sampl"),
        ("CO", "CC", "[Uu]nknown"),
        ("OC", "CO", "topology"),
        ("[13CH3]O", "CO", "topology"),
        ("C[O-]", "CO", "topology"),
        ("C=O", "CO", "topology"),
    ],
)
def test_bad_provenance_or_ordered_chemistry_is_rejected(tmp_path, smiles, tag, error):
    ref = reference_file(tmp_path, [("CO", "CO", 0)])
    sdf = generated_file(tmp_path, [(smiles, tag, 0)])
    with pytest.raises(ValueError, match=error):
        read_covmat_inputs(ref, sdf)


def test_reference_rows_of_one_molecule_are_scored_together(tmp_path):
    # The author's packed test set keeps the graph of a molecule's first row
    # and the coordinates of all its rows, whatever the later rows look like.
    ref = reference_file(tmp_path, [("CO", "CO", 0), ("CO", "OC", 1)])
    sdf = generated_file(
        tmp_path, [("CO", "CO", 5), ("CO", "CO", 6), ("CO", "CO", 7), ("CO", "CO", 8)]
    )
    groups, stereo = read_covmat_inputs(ref, sdf)
    assert [(key, len(refs), len(gens)) for key, _, refs, gens in groups] == [
        ("CO", 2, 4)
    ]
    assert Chem.MolToSmiles(next(iter(groups))[1]) == "CO"
    assert stereo["matched_records"] == 4


def test_source_shared_by_two_author_groups_cannot_be_paired(tmp_path):
    # Both author groups start from the same conditioning graph, so a source
    # tag cannot tell which molecule a generated conformer belongs to.
    ref = reference_file(tmp_path, [("CO", "CO", 0), ("OC", "CO", 1)])
    sdf = generated_file(tmp_path, [("CO", "CO", 5)])
    with pytest.raises(ValueError, match="[Aa]mbiguous"):
        read_covmat_inputs(ref, sdf)


def test_reference_rows_differing_only_in_stereo_share_one_group(tmp_path):
    ref = reference_file(tmp_path, [(SOURCE, SOURCE, 1), (SOURCE, OPPOSITE, 3)])
    sdf = generated_file(tmp_path, [(SOURCE, SOURCE, 5)])
    groups, _ = read_covmat_inputs(ref, sdf)
    assert [(key, len(refs), len(gens)) for key, _, refs, gens in groups] == [
        (SOURCE, 2, 1)
    ]


def test_stereo_mismatch_is_limited_to_centres_the_source_specifies(tmp_path):
    untagged = "CC(O)Cl"
    mirror = TETRAHEDRON * [-1, 1, 1]
    ref = reference_file(
        tmp_path,
        [(untagged, untagged, 0), (SOURCE, SOURCE, 0), (SOURCE, SOURCE, 0)],
        poses=[TETRAHEDRON[:4], TETRAHEDRON, mirror],
    )
    sdf = tmp_path / "generated.sdf"
    write_molecular_conformers(
        sdf,
        [
            {
                "rdmol": Chem.MolFromSmiles(untagged).ToBinary(),
                "cart_coords": [TETRAHEDRON[:4]],
            },
            {
                "rdmol": Chem.MolFromSmiles(SOURCE).ToBinary(),
                "cart_coords": [TETRAHEDRON, mirror],
            },
        ],
    )
    _, stereo = read_covmat_inputs(ref, sdf)
    # One generated pose is the enantiomer of the source; the untagged molecule
    # specifies no stereochemistry, so its perceived centre is not a mismatch.
    assert stereo["mismatch_count"] == 1
    assert stereo["mismatches_by_source"] == {untagged: 0, SOURCE: 1}


def test_stereo_mismatch_reports_double_bonds_the_source_specifies(tmp_path):
    source = "C/C=C/C"
    trans = np.array([[-0.62, 1.08, 0], [0, 0, 0], [1.33, 0, 0], [1.95, -1.08, 0]])
    cis = trans.copy()
    cis[3, 1] = 1.08
    ref = reference_file(
        tmp_path, [(source, source, 0), (source, source, 0)], poses=[trans, cis]
    )
    sdf = tmp_path / "generated.sdf"
    write_molecular_conformers(
        sdf,
        [{"rdmol": Chem.MolFromSmiles(source).ToBinary(), "cart_coords": [trans, cis]}],
    )
    _, stereo = read_covmat_inputs(ref, sdf)
    assert stereo["mismatch_count"] == 1


def run_cli(monkeypatch, *arguments):
    from structure_generation import evaluate_geodiff

    monkeypatch.setattr(sys, "argv", ["evaluate_geodiff", *arguments])
    evaluate_geodiff.main()


def test_cli_rejects_an_existing_output_before_reading_inputs(
    tmp_path, monkeypatch, capsys
):
    output = tmp_path / "result.json"
    output.write_text("keep")
    with pytest.raises(SystemExit) as exc:
        run_cli(
            monkeypatch,
            "covmat",
            "--reference",
            str(tmp_path / "absent.npz"),
            "--generated",
            str(tmp_path / "absent.sdf"),
            "--output",
            str(output),
        )
    assert exc.value.code == 2
    assert "--output" in capsys.readouterr().err
    assert output.read_text() == "keep"


def test_cli_reports_a_truncated_reference_as_a_usage_error(tmp_path, monkeypatch):
    truncated = tmp_path / "truncated.npz"
    truncated.write_bytes(b"PK\x03\x04truncated")
    with pytest.raises(SystemExit) as exc:
        run_cli(
            monkeypatch,
            "covmat",
            "--reference",
            str(truncated),
            "--generated",
            str(tmp_path / "absent.sdf"),
        )
    assert exc.value.code == 2


def test_cli_writes_a_result_for_a_scored_molecule(tmp_path, monkeypatch):
    line = np.array([[0.0, 0, 0], [1.4, 0, 0]])
    ref = reference_file(tmp_path, [("CO", "CO", 0)], poses=[line])
    sdf = tmp_path / "generated.sdf"
    write_molecular_conformers(
        sdf,
        [
            {
                "rdmol": Chem.MolFromSmiles("CO").ToBinary(),
                "cart_coords": [line * 2, line * 3],
            }
        ],
    )
    output = tmp_path / "result.json"
    run_cli(
        monkeypatch,
        "covmat",
        "--reference",
        str(ref),
        "--generated",
        str(sdf),
        "--output",
        str(output),
    )
    result = json.loads(output.read_text())
    assert result["counts"]["eligible_molecules"] == 1
    assert result["per_molecule"][0]["smiles"] == "CO"


# Validation losses


class _ValidationModel(paddle.nn.Layer):
    def forward(self, batch):
        loss = batch["atom_types"].astype("float32")
        per_atom = {
            "loss": loss,
            "loss_global": 0.25 * loss,
            "loss_local": 0.75 * loss,
        }
        result = {"loss_dict": {key: value.mean() for key, value in per_atom.items()}}
        if not self.training:
            result["loss_stats"] = {key: value.sum() for key, value in per_atom.items()}
        return result


class _Molecules(paddle.io.Dataset):
    def __init__(self, atom_types):
        self.atom_types = atom_types

    def __len__(self):
        return len(self.atom_types)

    def __getitem__(self, index):
        return {"atom_types": paddle.to_tensor(self.atom_types[index])}


def _loader(*atom_types):
    return paddle.io.DataLoader(
        _Molecules(atom_types), batch_size=1, collate_fn=lambda samples: samples[0]
    )


def test_geodiff_validation_losses_are_per_atom_and_independent(tmp_path):
    metrics = build_metric(
        {
            "validation_loss": {
                "__class_name__": "GeoDiffValidationLoss",
                "__init_params__": {},
            }
        }
    )
    model = _ValidationModel()
    trainer = BaseTrainer(
        {
            "max_epochs": 1,
            "output_dir": str(tmp_path),
            "save_freq": 1,
            "log_freq": 1,
            "start_eval_epoch": 1,
            "eval_freq": 1,
            "seed": 42,
            "compute_metric_during_train": False,
        },
        model,
    )
    trainer.attach_metrics(metrics)

    # Training results do not carry validation totals.
    train_batch = {"atom_types": paddle.to_tensor([3])}
    metric = metrics["validation_loss"]
    metric.update_step(result=model(train_batch), batch=train_batch, stage="train")
    assert metric.compute_epoch(stage="train") == {}

    _, _, first = trainer.eval(_loader([1]))
    assert {key: meter.avg for key, meter in first.items()} == pytest.approx(
        {"geodiff_loss": 1.0, "geodiff_loss_global": 0.25, "geodiff_loss_local": 0.75}
    )

    # Step validation deliberately skips the trainer's metric reset.
    _, _, weighted = trainer.eval_epoch(
        _loader([1], [2, 2, 2]), reset_streaming_metrics=False
    )
    assert {key: meter.avg for key, meter in weighted.items()} == pytest.approx(
        {
            "geodiff_loss": 1.75,
            "geodiff_loss_global": 0.4375,
            "geodiff_loss_local": 1.3125,
        }
    )

    _, _, next_eval = trainer.eval_epoch(
        _loader([2, 2, 2]), reset_streaming_metrics=False
    )
    assert {key: meter.avg for key, meter in next_eval.items()} == pytest.approx(
        {"geodiff_loss": 2.0, "geodiff_loss_global": 0.5, "geodiff_loss_local": 1.5}
    )

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

"""PropMolFlow integration tests over the training/sampling/evaluation chains."""

from __future__ import annotations

import csv
import os
import os.path as osp
from pathlib import Path

import numpy as np
import paddle
import pandas as pd
import pytest
from omegaconf import OmegaConf

from ppmat.datasets import build_dataloader
from ppmat.datasets.collate_fn import DefaultCollator
from ppmat.datasets.custom_data_type import ConcatData
from ppmat.datasets.propmolflow_dataset import PropMolFlowDataset
from ppmat.datasets.propmolflow_dataset import PropMolFlowSamplingDataset
from ppmat.datasets.qm9_dataset import QM9Dataset
from ppmat.metrics import PropMolFlowMoleculeMetric
from ppmat.models import PropMolFlowVectorField
from ppmat.models import build_model
from ppmat.optimizer import build_optimizer
from ppmat.sampler import MolecularSampler
from ppmat.trainer.base_trainer import BaseTrainer

CONFIG_PATH = osp.join(
    osp.dirname(__file__),
    "..",
    "..",
    "structure_generation",
    "configs",
    "propmolflow",
    "propmolflow_qm9_alpha_sum.yaml",
)

PRIOR_CONFIG = {
    "positions": {"type": "centered-normal", "align": True, "kwargs": {"std": 1.0}},
    "atom_types": {"type": "ctmc"},
    "atom_charges": {"type": "ctmc"},
    "bond_types": {"type": "ctmc"},
}

SMALL_VECTOR_FIELD = {
    "n_charges": 6,
    "n_bond_types": 5,
    "charge_offset": 2,
    "n_hidden_scalars": 32,
    "n_hidden_edge_feats": 16,
    "n_vec_channels": 8,
    "n_cp_feats": 2,
    "n_message_gvps": 2,
    "n_update_gvps": 2,
    "n_position_update_gvps": 3,
    "n_molecule_updates": 2,
    "convs_per_update": 1,
    "n_recycles": 1,
    "rbf_dim": 8,
    "rbf_dmax": 14,
    "message_norm": 100,
    "separate_mol_updaters": True,
    "update_edge_w_distance": True,
    "stochasticity": 10.0,
    "high_confidence_threshold": 0.0,
    "cat_temperature": 0.05,
}


def _build_model(vector_field_config=None, **overrides):
    init_params = {
        "atom_type_map": ["C", "H", "N", "O", "F"],
        "total_loss_weights": {
            "positions": 3.0,
            "atom_types": 0.4,
            "atom_charges": 1.0,
            "bond_types": 2.0,
        },
        "interpolant_scheduler_config": {
            "__class_name__": "InterpolantScheduler",
            "__init_params__": {
                "schedule_type": {
                    "positions": "linear",
                    "atom_types": "linear",
                    "atom_charges": "linear",
                    "bond_types": "linear",
                },
                "cosine_params": {
                    "positions": 1,
                    "atom_types": 2,
                    "atom_charges": 2,
                    "bond_types": 1.5,
                },
            },
        },
        "vector_field_config": (
            SMALL_VECTOR_FIELD if vector_field_config is None else vector_field_config
        ),
        "prior_config": PRIOR_CONFIG,
        "property_embedding_dim": 32,
        "properties_handle_method": "sum",
        "gaussian_expansion": False,
        "n_gaussians": 5,
        "gaussian_start": None,
        "gaussian_stop": None,
    }
    init_params.update(overrides)
    model = build_model(
        {"__class_name__": "PropMolFlow", "__init_params__": init_params}
    )
    model.eval()
    return model


_SYNTHETIC_SMILES = (
    "C",
    "O",
    "N",
    "C#N",
    "CCO",
    "CC(=O)O",
    "c1ccccc1",
    "CCN",
    "CO",
    "C=C",
)
_SYNTHETIC_PROPERTIES = ("alpha", "mu", "Cv", "gap", "homo", "lumo")


def _write_synthetic_qm9(tmp_path, num_molecules=6, seed=0):
    """Write an official-QM9-shaped split CSV with embedded 3D molecules."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    rows = []
    for index, smiles in enumerate(_SYNTHETIC_SMILES[:num_molecules]):
        mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
        AllChem.EmbedMolecule(mol, randomSeed=seed + index)
        conformer = mol.GetConformer()
        lines = [str(mol.GetNumAtoms()), f"synthetic {index}"]
        for atom in mol.GetAtoms():
            position = conformer.GetAtomPosition(atom.GetIdx())
            lines.append(
                f"{atom.GetSymbol()} {position.x:.8f} {position.y:.8f} {position.z:.8f}"
            )
        rows.append((index + 1, "\n".join(lines)))

    rng = np.random.RandomState(seed)
    properties = (
        rng.rand(len(rows), len(_SYNTHETIC_PROPERTIES)).astype("float64") * 10.0
    )
    path = tmp_path / "train.csv"
    with open(path, "w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["standard_xyz", "molecule_id", *_SYNTHETIC_PROPERTIES])
        for row_index, (molecule_id, xyz) in enumerate(rows):
            writer.writerow([xyz, molecule_id, *properties[row_index].tolist()])
    return str(path), properties


def _build_batch(num_atoms_list):
    rng = np.random.RandomState(0)
    samples = []
    for n_atoms in num_atoms_list:
        one_hot_a = np.zeros((n_atoms, 5), dtype="float32")
        one_hot_a[np.arange(n_atoms), rng.randint(0, 5, n_atoms)] = 1.0
        one_hot_c = np.zeros((n_atoms, 6), dtype="float32")
        one_hot_c[np.arange(n_atoms), rng.randint(0, 6, n_atoms)] = 1.0
        upper_i, upper_j = np.triu_indices(n_atoms, k=1)
        bond_types = rng.randint(0, 5, upper_i.shape[0])
        one_hot_e = np.zeros((upper_i.shape[0], 5), dtype="float32")
        one_hot_e[np.arange(upper_i.shape[0]), bond_types] = 1.0
        samples.append(
            {
                "structure_array": {
                    "num_atoms": n_atoms,
                    "positions": rng.standard_normal((n_atoms, 3)).astype("float32"),
                    "atom_types": one_hot_a,
                    "atom_charges": one_hot_c,
                    "bond_types": one_hot_e,
                },
                "prop": float(rng.standard_normal()),
            }
        )
    for sample in samples:
        structure_array = sample["structure_array"]
        structure_array["num_atoms"] = np.array(
            [structure_array["num_atoms"]], dtype="int64"
        )
        sample["structure_array"] = {
            key: ConcatData(value) for key, value in structure_array.items()
        }
    return DefaultCollator()(samples)


def _sampling_dataset(tmp_path, **kwargs):
    path, _ = _write_synthetic_qm9(tmp_path)
    dataset_kwargs = {
        "path": path,
        "normalization_source": path,
        "property_name": "alpha",
        "mode": "in_distribution",
        "total_num": 2,
        "seed": 0,
    }
    dataset_kwargs.update(kwargs)
    return PropMolFlowSamplingDataset(**dataset_kwargs)


def _sampling_inputs(dataset):
    batch = DefaultCollator()([dataset[i] for i in range(len(dataset))])
    return {
        "structure_array": {
            "num_atoms": paddle.to_tensor(batch["structure_array"]["num_atoms"])
        },
        "properties": paddle.to_tensor(batch["properties"]),
    }


def _assert_molecule_dicts(result, num_atoms):
    assert len(result) == num_atoms.shape[0]
    for molecule, expected_atoms in zip(result, num_atoms.numpy().tolist()):
        assert molecule["num_atoms"] == expected_atoms
        assert len(molecule["atom_types"]) == expected_atoms
        assert np.asarray(molecule["positions"]).shape == (expected_atoms, 3)
        assert all(
            symbol in ["C", "H", "N", "O", "F", "Se"]
            for symbol in molecule["atom_types"]
        )
        for src, dst, bond_type in molecule["bonds"]:
            assert 0 <= src < expected_atoms
            assert 0 <= dst < expected_atoms
            assert 1 <= bond_type <= 4


# ----------------------------- training chain -----------------------------


def test_train_forward_from_config():
    config = OmegaConf.load(CONFIG_PATH)
    config = OmegaConf.to_container(config, resolve=True)
    model = build_model(config["Model"])
    model.eval()

    batch = _build_batch([4, 5])
    output = model(batch)
    assert set(output["loss_dict"]) == {
        "loss",
        "loss_positions",
        "loss_atom_types",
        "loss_atom_charges",
        "loss_bond_types",
    }
    for value in output["loss_dict"].values():
        assert np.isfinite(float(value))
    output["loss_dict"]["loss"].backward()
    assert all(
        parameter.grad is not None
        for parameter in model.parameters()
        if not parameter.stop_gradient
    )

    # Training-forward variants over the released configuration space.
    for overrides in (
        {"gaussian_expansion": True, "gaussian_start": 6.31, "gaussian_stop": 196.62},
        {
            "vector_field_config": dict(
                SMALL_VECTOR_FIELD, n_molecule_updates=1, convs_per_update=2
            )
        },
        {"vector_field_config": dict(SMALL_VECTOR_FIELD, separate_mol_updaters=False)},
        *(
            {"properties_handle_method": method}
            for method in [
                "concatenate",
                "sum",
                "multiply",
                "concatenate_sum",
                "concatenate_multiply",
            ]
        ),
    ):
        variant = _build_model(**overrides)
        loss = variant(batch)["loss_dict"]["loss"]
        assert np.isfinite(float(loss))
        loss.backward()
        assert all(
            parameter.grad is not None
            for parameter in variant.parameters()
            if not parameter.stop_gradient
        )

    saturated_batch = _build_batch([4, 5])
    saturated_batch["prop"] = paddle.to_tensor([-1000.0, 1000.0])
    saturated_model = _build_model()
    saturated_loss = saturated_model(saturated_batch)["loss_dict"]["loss"]
    saturated_loss.backward()
    assert np.isfinite(float(saturated_loss))
    assert all(
        np.isfinite(parameter.grad.numpy()).all()
        for parameter in saturated_model.parameters()
        if not parameter.stop_gradient
    )

    with pytest.raises(ValueError):
        _build_model(gaussian_expansion=True, gaussian_start=None, gaussian_stop=None)
    with pytest.raises(ValueError):
        _build_model(properties_handle_method="unsupported")
    with pytest.raises(ValueError, match="at least two convolutions"):
        _build_model(vector_field_config=dict(SMALL_VECTOR_FIELD, n_molecule_updates=1))
    for name in ("n_recycles", "n_molecule_updates", "convs_per_update"):
        with pytest.raises(ValueError, match="must be positive"):
            _build_model(vector_field_config=dict(SMALL_VECTOR_FIELD, **{name: 0}))


def test_train_data_pipeline_feeds_forward(tmp_path, monkeypatch):
    path, properties = _write_synthetic_qm9(tmp_path)

    config = OmegaConf.to_container(OmegaConf.load(CONFIG_PATH), resolve=True)
    dataset_params = config["Dataset"]["train"]["dataset"]["__init_params__"]
    dataset_params.update(
        path=path,
        cache_dir=str(tmp_path / "cache"),
        normalization_source=path,
    )
    config["Dataset"]["train"]["sampler"]["__init_params__"]["shuffle"] = False
    data_loader = build_dataloader(config["Dataset"]["train"])
    batch = next(iter(data_loader))
    assert isinstance(data_loader.dataset, QM9Dataset)
    assert all(osp.isfile(path) for path in data_loader.dataset.molecules)

    atom_counts = batch["structure_array"]["num_atoms"].numpy()
    assert len(atom_counts) == 6
    assert len(set(atom_counts.tolist())) > 1
    assert batch["structure_array"]["positions"].shape[0] == atom_counts.sum()
    assert batch["structure_array"]["bond_types"].shape[0] == sum(
        n * (n - 1) // 2 for n in atom_counts
    )

    property_idx = _SYNTHETIC_PROPERTIES.index("alpha")
    mean = float(properties[:, property_idx].mean())
    std = float(properties[:, property_idx].std())
    restored = float(batch["prop"][0]) * std + mean
    assert np.isclose(restored, float(properties[0, property_idx]), atol=1e-5)

    model = _build_model()
    assert np.isfinite(float(model(batch)["loss_dict"]["loss"]))

    original_cache = data_loader.dataset.cache_path
    frame = pd.read_csv(path)
    frame = frame.iloc[[1, 0, *range(2, len(frame))]]
    frame.to_csv(path, index=False)
    changed_loader = build_dataloader(config["Dataset"]["train"])
    changed_batch = next(iter(changed_loader))
    assert changed_loader.dataset.cache_path != original_cache
    assert osp.isfile(original_cache)
    assert int(batch["structure_array"]["num_atoms"][0]) == 5
    assert int(changed_batch["structure_array"]["num_atoms"][0]) == 3
    restored = float(changed_batch["prop"][0]) * std + mean
    assert np.isclose(restored, float(properties[1, property_idx]), atol=1e-5)
    assert np.isfinite(float(model(changed_batch)["loss_dict"]["loss"]))

    cached_loader = build_dataloader(config["Dataset"]["train"])
    assert cached_loader.dataset.cache_path == changed_loader.dataset.cache_path
    np.testing.assert_array_equal(
        cached_loader.dataset.positions, changed_loader.dataset.positions
    )
    assert not cached_loader.dataset.raw_data
    assert len(list((tmp_path / "cache").glob("*.npz"))) == 2

    side_effects = {
        name: Path(name).read_bytes() if osp.isfile(name) else None
        for name in ("run.out", "nul")
    }
    with monkeypatch.context() as context:

        def require_hueckel(*args, **kwargs):
            raise ValueError("Exercise isolated Hueckel perception")

        context.setattr(
            "ppmat.datasets.propmolflow_dataset.rdDetermineBonds.DetermineBonds",
            require_hueckel,
        )
        fallback = PropMolFlowDataset(**dict(dataset_params, rebuild=True))
        fallback_batch = DefaultCollator()([fallback[i] for i in range(len(fallback))])
        assert np.isfinite(float(model(fallback_batch)["loss_dict"]["loss"]))
        np.testing.assert_array_equal(
            fallback.bond_types, cached_loader.dataset.bond_types
        )
    for name, previous in side_effects.items():
        actual = Path(name).read_bytes() if osp.isfile(name) else None
        assert actual == previous
    assert not list((tmp_path / "cache").glob("perception_*"))

    alpha_values = frame["alpha"].to_numpy()
    ordered = np.sort(alpha_values)
    interval = [float(ordered[1]), float(ordered[4])]
    selected = PropMolFlowDataset(**dict(dataset_params, property_range=interval))
    expected = alpha_values[
        (alpha_values >= interval[0]) & (alpha_values < interval[1])
    ]
    np.testing.assert_array_equal(selected.property_values, expected)
    assert selected.cache_path != cached_loader.dataset.cache_path
    selected_batch = DefaultCollator()([selected[i] for i in range(len(selected))])
    assert np.isfinite(float(model(selected_batch)["loss_dict"]["loss"]))
    with pytest.raises(ValueError, match="No QM9 samples"):
        PropMolFlowDataset(
            **dict(dataset_params, property_range=[float(ordered[-1]) + 1, 100.0])
        )

    with monkeypatch.context() as context:
        context.setattr(
            "ppmat.datasets.propmolflow_dataset.dist.is_initialized", lambda: True
        )
        rank = [0]
        broadcasts = []
        context.setattr(
            "ppmat.datasets.propmolflow_dataset.dist.get_rank", lambda: rank[0]
        )

        def broadcast_status(status, src):
            assert src == 0
            if rank[0] == 0:
                broadcasts.append(status[0])
            else:
                status[0] = broadcasts[-1]

        context.setattr(
            "ppmat.datasets.propmolflow_dataset.dist.broadcast_object_list",
            broadcast_status,
        )
        for rank[0] in (0, 1):
            distributed_loader = build_dataloader(config["Dataset"]["train"])
            distributed_batch = next(iter(distributed_loader))
            assert np.isfinite(float(model(distributed_batch)["loss_dict"]["loss"]))
        assert broadcasts == [None, None]

        broken_path = tmp_path / "broken.csv"
        broken_frame = frame.copy()
        broken_frame["standard_xyz"] = "1\ninvalid\nX 0 0 0"
        broken_frame.to_csv(broken_path, index=False)
        for rank[0] in (0, 1):
            with pytest.raises(RuntimeError, match="Prewarm this dataset"):
                PropMolFlowDataset(**dict(dataset_params, path=str(broken_path)))
        assert "Prewarm this dataset" in broadcasts[-1]

    val_path = tmp_path / "val.csv"
    frame.iloc[:2].to_csv(val_path, index=False)
    val_dataset = PropMolFlowDataset(**dict(dataset_params, path=str(val_path)))
    assert val_dataset.property_mean == changed_loader.dataset.property_mean
    assert val_dataset.property_std == changed_loader.dataset.property_std

    isolated_dir = tmp_path / "isolated"
    isolated_dir.mkdir()
    isolated_path = isolated_dir / "val.csv"
    frame.iloc[:2].to_csv(isolated_path, index=False)
    sampling_params = dict(
        path=str(isolated_path),
        property_name="alpha",
        mode="in_distribution",
        total_num=2,
        seed=42,
    )
    checkpoint_dataset = PropMolFlowSamplingDataset(**sampling_params)
    assert checkpoint_dataset.property_mean == 75.19867706298828
    assert checkpoint_dataset.property_std == 8.181739807128906
    reference_training = PropMolFlowDataset(
        **dict(dataset_params, path=str(isolated_path), normalization_source=None)
    )
    assert reference_training.property_mean == checkpoint_dataset.property_mean
    assert reference_training.property_std == checkpoint_dataset.property_std
    sampling_dataset = PropMolFlowSamplingDataset(
        **sampling_params, normalization_source=path
    )
    assert sampling_dataset.property_mean == val_dataset.property_mean
    assert sampling_dataset.property_std == val_dataset.property_std

    def unexpected_download(*args, **kwargs):
        pytest.fail("An explicit normalization source must not trigger a download")

    with monkeypatch.context() as context:
        context.setattr(
            "ppmat.datasets.propmolflow_dataset.download.get_datasets_path_from_url",
            unexpected_download,
        )
        with pytest.raises(FileNotFoundError, match="normalization_source"):
            PropMolFlowSamplingDataset(
                **sampling_params,
                normalization_source=str(tmp_path / "missing" / "train.csv"),
            )

    frame["alpha"] = 7.0
    frame.to_csv(isolated_dir / "train.csv", index=False)
    with pytest.raises(ValueError, match="positive variance"):
        PropMolFlowSamplingDataset(
            **sampling_params, normalization_source=str(isolated_dir / "train.csv")
        )


# ----------------------------- sampling chain -----------------------------


def test_sample_produces_valid_molecules(tmp_path):
    model = _build_model()
    sampling_datasets = (
        _sampling_dataset(tmp_path),
        _sampling_dataset(
            tmp_path,
            mode="out_of_distribution",
            property_value=3.0,
        ),
        _sampling_dataset(tmp_path, n_atoms=4, property_value=2.0),
    )

    for dataset in sampling_datasets:
        batch_data = _sampling_inputs(dataset)
        for invalid_steps in (0, 1):
            with pytest.raises(ValueError, match="at least 2"):
                model.sample(batch_data, num_inference_steps=invalid_steps)
        for sample_kwargs in (
            {},
            {"stochasticity": 8.0, "high_confidence_threshold": 0.9},
        ):
            result = model.sample(batch_data, num_inference_steps=2, **sample_kwargs)[
                "pred"
            ]
            _assert_molecule_dicts(result, batch_data["structure_array"]["num_atoms"])


def test_sampling_framework_end_to_end(tmp_path):
    path, _ = _write_synthetic_qm9(tmp_path)
    checkpoint_path = str(tmp_path / "best.pdparams")
    paddle.save(_build_model().state_dict(), checkpoint_path)

    config = OmegaConf.load(CONFIG_PATH)
    config.Model.__init_params__.vector_field_config = OmegaConf.create(
        SMALL_VECTOR_FIELD
    )
    config.Model.__init_params__.property_embedding_dim = 32
    sampler_config_path = tmp_path / "sampler_config.yaml"
    OmegaConf.save(config, str(sampler_config_path))

    for condition_overrides in (
        [],
        [
            "Sampler.data.dataset.__init_params__.mode=out_of_distribution",
            "Sampler.data.dataset.__init_params__.property_value=3.0",
        ],
    ):
        sampler = MolecularSampler(
            config_path=str(sampler_config_path),
            checkpoint_path=checkpoint_path,
            config_overrides=[
                "Dataset.train=null",
                f"Sampler.data.dataset.__init_params__.path={path}",
                "Sampler.data.dataset.__init_params__.total_num=4",
                f"Sampler.output_dir={tmp_path / 'out'}",
                "Sampler.model_sample_params.num_inference_steps=2",
            ]
            + condition_overrides,
        )
        results = sampler.sample_by_dataloader(save_path=str(tmp_path / "out"))
        assert results is sampler.last_samples
        assert len(results["pred"]) == 4

        # API-level evaluation and serialization (no top-level Metric section
        # is shipped: the SG task train.py has no generation-model metrics).
        metrics = PropMolFlowMoleculeMetric()(results["pred"])
        assert set(metrics) == {
            "rdkit_valid",
            "valid_and_unique",
            "close_shell_ratio",
            "atom_stable",
            "mol_stable",
        }

    # The one-line model-owned serialization writes the SDF file.
    save_samples = getattr(sampler.model, "save_samples", None)
    assert callable(save_samples)
    save_samples(sampler.last_samples["pred"], sampler.output_dir)
    assert osp.exists(str(tmp_path / "out" / "propmolflow_samples.sdf"))


# ---------------------------- evaluation chain ----------------------------


def test_metric_one_shot_matches_streaming():
    from rdkit import rdBase

    log_status = rdBase.LogStatus()
    hydrogen = {
        "atom_types": ["H", "H"],
        "charges": [0, 0],
        "positions": [[0.0, 0.0, 0.0], [0.7, 0.0, 0.0]],
        "bonds": [[0, 1, 1]],
    }
    invalid_carbon = {
        "atom_types": ["C"] + ["H"] * 6,
        "charges": [0] * 7,
        "positions": [[0.0, 0.0, 0.0]] * 7,
        "bonds": [[0, index, 1] for index in range(1, 7)],
    }
    molecules = [hydrogen, invalid_carbon, hydrogen]
    one_shot = PropMolFlowMoleculeMetric()(molecules)
    assert one_shot == pytest.approx(
        {
            "rdkit_valid": 2 / 3,
            "valid_and_unique": 1 / 3,
            "close_shell_ratio": 1.0,
            "atom_stable": 10 / 11,
            "mol_stable": 2 / 3,
        }
    )

    streaming = PropMolFlowMoleculeMetric()
    for batch in (molecules[:2], molecules[2:]):
        streaming.update_step(
            result={"samples": {"pred": batch}}, batch=None, stage="sample"
        )
    streamed = streaming.compute_epoch(stage="sample")

    assert set(one_shot) == set(streamed)
    for name, value in one_shot.items():
        assert np.isclose(value, streamed[name])

    hydride = {
        "atom_types": ["H"],
        "charges": [-1],
        "positions": [[0.0, 0.0, 0.0]],
        "bonds": [],
    }
    hydrogen_radical = dict(hydride, charges=[0])
    hydrogen_ion = dict(hydrogen, charges=[1, 0])
    charged_molecules = [hydride, hydrogen_radical, hydrogen_ion]
    charged_metric = PropMolFlowMoleculeMetric()
    for molecule in charged_molecules:
        charged_metric.update_step(
            result={"samples": {"pred": [molecule]}}, batch=None, stage="sample"
        )
    assert charged_metric.compute_epoch(stage="sample")["close_shell_ratio"] == (
        pytest.approx(1 / 3)
    )
    assert PropMolFlowMoleculeMetric()(charged_molecules)["close_shell_ratio"] == (
        pytest.approx(1 / 3)
    )

    streaming.reset()
    # Non-sample stages carry no molecules; the metric must stay untouched.
    streaming.update_step(result={"loss_dict": {}}, batch=None, stage="eval")
    with pytest.raises(ValueError):
        streaming.compute_epoch(stage="sample")
    assert rdBase.LogStatus() == log_status
    with pytest.raises(KeyError):
        PropMolFlowMoleculeMetric()([{"atom_types": ["H"]}])
    assert rdBase.LogStatus() == log_status


# ------------------------ runtime orchestration (CINN) ------------------------


def test_trainer_checkpoint_resume_restores_training_state(tmp_path, monkeypatch):
    monkeypatch.setattr(
        PropMolFlowVectorField,
        "validate_execution_backend",
        lambda self, **kwargs: None,
    )
    monkeypatch.setattr(
        PropMolFlowVectorField,
        "_run_runtime",
        lambda self, name, function, *args, **kwargs: function(*args, **kwargs),
    )
    path, _ = _write_synthetic_qm9(tmp_path)
    config = OmegaConf.to_container(OmegaConf.load(CONFIG_PATH), resolve=True)
    data_config = config["Dataset"]["train"]
    data_config["dataset"]["__init_params__"].update(
        path=path, cache_dir=str(tmp_path / "cache")
    )
    data_config["sampler"]["__init_params__"].update(batch_size=3, shuffle=False)
    loader = build_dataloader(data_config)
    optimizer_config = config["Optimizer"]
    optimizer_config["__init_params__"]["lr"] = {
        "__class_name__": "ExponentialDecay",
        "__init_params__": {
            "learning_rate": 0.00025,
            "gamma": 0.9,
            "decay_steps": 1,
        },
    }

    def build_trainer(name, max_epochs, backend):
        with paddle.utils.unique_name.guard():
            model = _build_model()
            optimizer, scheduler = build_optimizer(
                optimizer_config,
                (model,),
                epochs=max_epochs,
                iters_per_epoch=len(loader),
            )
        trainer_config = dict(
            config["Trainer"],
            output_dir=str(tmp_path / name),
            max_epochs=max_epochs,
            save_freq=1,
        )
        return BaseTrainer(
            trainer_config,
            model,
            train_dataloader=loader,
            val_dataloader=loader,
            optimizer=optimizer,
            lr_scheduler=scheduler,
            execution_config={"backend": backend},
        )

    first = build_trainer("first", 1, "cinn")
    with paddle.utils.unique_name.guard():
        first.train()
    checkpoint_dir = tmp_path / "first" / "checkpoints"
    for prefix in ("epoch_1", "latest", "best"):
        for suffix in ("pdparams", "pdopt", "pdstates"):
            assert (checkpoint_dir / f"{prefix}.{suffix}").is_file()
    weights = paddle.load(str(checkpoint_dir / "latest.pdparams"))
    optimizer_state = paddle.load(str(checkpoint_dir / "latest.pdopt"))
    assert weights.keys() == first.model.state_dict().keys()

    resumed = build_trainer("resumed", 2, "cinn")
    restored_steps = []

    def check_restored_state(model, inputs):
        if restored_steps:
            return
        assert resumed.state.global_step == first.state.global_step == len(loader)
        assert resumed.state.epoch == first.state.epoch + 1
        assert resumed.optimizer.get_lr() == first.optimizer.get_lr()
        actual = resumed.optimizer.state_dict()
        assert actual.keys() == optimizer_state.keys()
        for name, value in optimizer_state.items():
            if isinstance(value, paddle.Tensor):
                np.testing.assert_array_equal(actual[name].numpy(), value.numpy())
            else:
                assert actual[name] == value
        for name, value in weights.items():
            np.testing.assert_array_equal(
                model.state_dict()[name].numpy(), value.numpy()
            )
        restored_steps.append(resumed.state.global_step)

    hook = resumed.model.register_forward_pre_hook(check_restored_state)
    with paddle.utils.unique_name.guard():
        resumed.train(resume_from_checkpoint=str(checkpoint_dir / "latest"))
    hook.remove()
    assert restored_steps == [len(loader)]

    straight = build_trainer("straight", 2, "eager")
    with paddle.utils.unique_name.guard():
        straight.train()
    assert resumed.model.execution_backend == "cinn"
    assert resumed.state.global_step == straight.state.global_step == 2 * len(loader)
    assert resumed.state.epoch == straight.state.epoch == 2
    assert resumed.optimizer.get_lr() == straight.optimizer.get_lr()
    assert resumed.lr_scheduler.state_dict() == straight.lr_scheduler.state_dict()
    assert (
        resumed.optimizer.state_dict().keys() == straight.optimizer.state_dict().keys()
    )
    assert resumed.model.state_dict().keys() == straight.model.state_dict().keys()


def test_cinn_dispatch_boundary_and_eager_equivalence(monkeypatch):
    """CINN orchestration: boundary dispatch and eager equivalence of the
    full training forward (fixed seed keeps the stochastic path deterministic)."""
    monkeypatch.setattr(
        PropMolFlowVectorField,
        "validate_execution_backend",
        lambda self, **kwargs: None,
    )
    boundary_names = []

    def spy(self, name, function, *args, **kwargs):
        boundary_names.append(name)
        return function(*args, **kwargs)

    monkeypatch.setattr(PropMolFlowVectorField, "_run_runtime", spy)

    batch = _build_batch([4, 5])

    eager_model = _build_model()
    paddle.seed(42)
    eager_losses = {
        name: float(value) for name, value in eager_model(batch)["loss_dict"].items()
    }
    assert boundary_names == ["denoise_step"]

    cinn_model = _build_model(execution_backend="cinn")
    cinn_model.set_state_dict(eager_model.state_dict())
    assert sorted(eager_model.state_dict()) == sorted(cinn_model.state_dict())

    boundary_names.clear()
    paddle.seed(42)
    cinn_losses = {
        name: float(value) for name, value in cinn_model(batch)["loss_dict"].items()
    }
    assert boundary_names == ["denoise_step"]

    for name, value in eager_losses.items():
        np.testing.assert_allclose(cinn_losses[name], value, rtol=0.0, atol=1e-5)

    # The sampling path dispatches the same boundary once per integration step.
    boundary_names.clear()
    cinn_model.sample(
        {
            "structure_array": {"num_atoms": paddle.to_tensor([3, 4], dtype="int64")},
            "properties": paddle.to_tensor([[0.1], [0.2]], dtype="float32"),
        },
        num_inference_steps=3,
    )
    assert boundary_names == ["denoise_step", "denoise_step"]


def _build_deterministic_state(model, num_atoms_list):
    batch = model._prepare_training_batch(_build_batch(num_atoms_list))
    rng = np.random.RandomState(0)
    num_nodes = batch["node_batch_idx"].shape[0]
    num_edges = batch["src"].shape[0]

    state = {
        key: batch[key]
        for key in (
            "src",
            "dst",
            "upper_edge_mask",
            "node_batch_idx",
            "edge_batch_idx",
        )
    }
    state["batch_size"] = batch["batch_size"]
    state["positions"] = paddle.to_tensor(
        rng.standard_normal((num_nodes, 3)).astype("float32")
    )
    for key, num_classes in (("atom_types", 6), ("atom_charges", 7)):
        one_hot = np.zeros((num_nodes, num_classes), dtype="float32")
        one_hot[np.arange(num_nodes), rng.randint(0, num_classes, num_nodes)] = 1.0
        state[key] = paddle.to_tensor(one_hot)
    edge_one_hot = np.zeros((num_edges, 6), dtype="float32")
    edge_one_hot[np.arange(num_edges), rng.randint(0, 6, num_edges)] = 1.0
    state["bond_types"] = paddle.to_tensor(edge_one_hot)

    properties = paddle.to_tensor([[0.5], [0.5]], dtype="float32")
    t = paddle.to_tensor([0.3, 0.7])
    prop_emb = model.property_embedder(properties)
    state["t_combined"] = paddle.concat([t.unsqueeze(-1), prop_emb], axis=-1)
    return state


def _run_vector_field(model, state):
    t_combined = state["t_combined"]
    with paddle.no_grad():
        return model.vector_field(
            state,
            t_combined,
            node_batch_idx=state["node_batch_idx"],
            upper_edge_mask=state["upper_edge_mask"],
        )


@pytest.mark.skipif(
    os.environ.get("PPMAT_RUN_CINN_WORKFLOW_TESTS") != "1",
    reason="Set PPMAT_RUN_CINN_WORKFLOW_TESTS=1 to run the GPU CINN comparison.",
)
def test_gpu_cinn_forward_matches_eager():
    if not paddle.is_compiled_with_cuda():
        pytest.skip("Paddle was not compiled with CUDA.")
    if not paddle.base.is_compiled_with_cinn():
        pytest.skip("Paddle was not compiled with CINN.")
    paddle.set_device("gpu:0")

    # The vector field is called directly because the GPU comparison needs a
    # fully deterministic input; the top-level forward/sample paths are
    # stochastic (time sampling, CTMC transitions).
    eager_model = _build_model()
    cinn_model = _build_model(execution_backend="cinn")
    cinn_model.set_state_dict(eager_model.state_dict())

    state = _build_deterministic_state(eager_model, [4, 5])
    eager_output = _run_vector_field(eager_model, state)
    cinn_output = _run_vector_field(cinn_model, state)
    # The eight GVP convolutions accumulate a per-layer compilation error of
    # about 1e-6, which the logits heads scale to a few 1e-4.
    for key in ("positions", "atom_types", "atom_charges", "bond_types"):
        np.testing.assert_allclose(
            eager_output[key].numpy(), cinn_output[key].numpy(), atol=5e-4
        )

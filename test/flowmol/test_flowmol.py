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

"""FlowMol integration tests over the training/sampling/evaluation chains."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import paddle
import pandas as pd
import pytest
from cinn_workflow_harness import assert_checkpoint_keys_unchanged
from cinn_workflow_harness import gpu_cinn_or_skip
from cinn_workflow_harness import patch_cinn_runtime
from cinn_workflow_harness import requires_gpu_cinn
from cinn_workflow_harness import trainer_config
from omegaconf import OmegaConf
from rdkit import Chem
from rdkit.Chem import AllChem

from ppmat.datasets import build_dataloader
from ppmat.datasets.collate_fn import DefaultCollator
from ppmat.datasets.custom_data_type import ConcatData
from ppmat.datasets.flowmol_sampling_dataset import FlowMol3GeomSamplingDataset
from ppmat.models import build_model
from ppmat.models.flowmol.vector_field import FlowMolVectorField
from ppmat.sampler import MolecularSampler
from ppmat.sampler import molecular_sampler
from ppmat.trainer.base_trainer import BaseTrainer
from structure_generation import sample as sampling_cli

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = (
    REPO_ROOT / "structure_generation" / "configs" / "flowmol" / "flowmol_qm9.yaml"
)
FLOWMOL3_CONFIG_PATH = (
    REPO_ROOT / "structure_generation" / "configs" / "flowmol" / "flowmol3.yaml"
)

SMALL_VECTOR_FIELD = {
    "n_charges": 6,
    "n_bond_types": 4,
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

SMALL_VECTOR_FIELD_RELEASED = {
    **SMALL_VECTOR_FIELD,
    "n_vec_channels": 8,
    "time_embedding_dim": 16,
    "a_token_dim": 8,
    "c_token_dim": 8,
    "e_token_dim": 8,
    "self_conditioning": True,
    "scprop": 0.5,
    "high_confidence_threshold": 0.9,
}

# the released flowmol3 vector field, used for real CINN compilation coverage
RELEASED_VECTOR_FIELD = OmegaConf.to_container(
    OmegaConf.load(FLOWMOL3_CONFIG_PATH), resolve=True
)["Model"]["__init_params__"]["vector_field_config"]


def _build_model(vector_field_config=None, **overrides):
    config = OmegaConf.to_container(OmegaConf.load(CONFIG_PATH).Model, resolve=True)
    init_params = config["__init_params__"]
    init_params["vector_field_config"] = (
        SMALL_VECTOR_FIELD if vector_field_config is None else vector_field_config
    )
    init_params["default_n_timesteps"] = 4
    init_params.update(overrides)
    model = build_model(config)
    model.eval()
    return model


def _write_synthetic_qm9_csv(tmp_path):
    rows = []
    for idx, smiles in enumerate(
        ("CCO", "c1ccccc1", "C#N", "C[N+](=O)[O-]", "CF", "O")
    ):
        mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
        assert AllChem.EmbedMolecule(mol, randomSeed=idx + 1) == 0
        rows.append(
            {
                "standard_xyz": Chem.MolToXYZBlock(mol),
                "molecule_id": idx + 1,
                "num_atoms": mol.GetNumAtoms(),
            }
        )
    path = tmp_path / "train.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return str(path)


def _training_dataset(tmp_path, fake_atom_p=0.0):
    config = OmegaConf.to_container(OmegaConf.load(CONFIG_PATH), resolve=True)
    data_config = config["Dataset"]["train"]
    params = data_config["dataset"]["__init_params__"]
    params["path"] = _write_synthetic_qm9_csv(tmp_path)
    params["cache_path"] = str(tmp_path / "qm9_cache")
    params["transforms"][0]["__init_params__"]["fake_atom_p"] = fake_atom_p
    params["build_molecule_cfg"]["num_cpus"] = 1
    params["build_graph_cfg"]["__init_params__"]["num_cpus"] = 1
    data_config["sampler"]["__init_params__"]["batch_size"] = 3
    data_config["sampler"]["__init_params__"]["shuffle"] = False
    return build_dataloader(data_config).dataset


def _build_batch(num_atoms_list):
    rng = np.random.RandomState(0)
    samples = []
    for n_atoms in num_atoms_list:
        one_hot_a = np.zeros((n_atoms, 5), dtype="float32")
        one_hot_a[np.arange(n_atoms), rng.randint(0, 5, n_atoms)] = 1.0
        one_hot_c = np.zeros((n_atoms, 6), dtype="float32")
        one_hot_c[np.arange(n_atoms), rng.randint(0, 6, n_atoms)] = 1.0
        upper_i, upper_j = np.triu_indices(n_atoms, k=1)
        bond_types = rng.randint(0, 4, upper_i.shape[0])
        one_hot_e = np.zeros((upper_i.shape[0], 4), dtype="float32")
        one_hot_e[np.arange(upper_i.shape[0]), bond_types] = 1.0
        prior = rng.standard_normal((n_atoms, 3)).astype("float32")
        prior = prior - prior.mean(axis=0, keepdims=True)
        samples.append(
            {
                "structure_array": {
                    "num_atoms": ConcatData(np.array([n_atoms], dtype="int64")),
                    "positions": ConcatData(
                        rng.standard_normal((n_atoms, 3)).astype("float32")
                    ),
                    "atom_types": ConcatData(one_hot_a),
                    "atom_charges": ConcatData(one_hot_c),
                    "bond_types": ConcatData(
                        np.concatenate((one_hot_e, one_hot_e), axis=0)
                    ),
                },
                "prior_positions": ConcatData(prior),
            }
        )
    return DefaultCollator()(samples)


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

    for schedule in (
        {"convs_per_update": 2, "separate_mol_updaters": True},
        {"convs_per_update": 1, "separate_mol_updaters": False},
    ):
        scheduled_model = _build_model(
            vector_field_config={**SMALL_VECTOR_FIELD, **schedule}
        )
        loss = scheduled_model(batch)["loss_dict"]["loss"]
        assert np.isfinite(float(loss))
        loss.backward()
        assert all(
            parameter.grad is not None for parameter in scheduled_model.parameters()
        )

    with pytest.raises(ValueError, match="at least one position update"):
        _build_model(
            vector_field_config={**SMALL_VECTOR_FIELD, "n_molecule_updates": 1}
        )

    # the released flowmol3 structure (token embeddings, time embedding and
    # the self-conditioning residual layer) must also run a training forward;
    # distort_p>0 additionally exercises the upstream position-distortion path
    variant = _build_model(
        vector_field_config=SMALL_VECTOR_FIELD_RELEASED,
        time_scaled_loss=True,
        fake_atom_p=0.3,
        distort_p=0.7,
        distort_t=0.25,
    )
    assert np.isfinite(float(variant(batch)["loss_dict"]["loss"]))

    with pytest.raises(ValueError):
        _build_model(prior_config={"positions": {"type": "unsupported"}})


@pytest.mark.parametrize("fake_atom_p", [0.0, 0.3])
def test_train_data_pipeline_feeds_forward(tmp_path, fake_atom_p):
    dataset = _training_dataset(tmp_path, fake_atom_p)
    assert len(dataset) == 6
    expected_heavy_bonds = ([1, 1], [1, 1, 1, 2, 2, 2], [3], [1, 1, 2], [1], [])
    for idx, expected in enumerate(expected_heavy_bonds):
        graph = dataset.load_from_cache(dataset.graphs[idx])
        bonds = graph["bond_index"]
        heavy = np.all(graph["atomic_numbers"][bonds] != 1, axis=1)
        assert sorted(graph["bond_class"][heavy].tolist()) == expected
    charged = dataset.load_from_cache(dataset.graphs[3])
    assert sorted(charged["formal_charges"].tolist()) == [-1, 0, 0, 0, 0, 0, 1]

    sample = dataset[0]
    assert "graph" not in sample
    assert np.allclose(
        sample["structure_array"]["positions"].data.mean(axis=0), 0.0, atol=1e-6
    )
    second = dataset[0]
    assert not np.array_equal(
        sample["prior_positions"].data, second["prior_positions"].data
    )
    cached = _training_dataset(tmp_path, fake_atom_p)
    np.testing.assert_array_equal(
        cached[0]["structure_array"]["bond_types"].data,
        sample["structure_array"]["bond_types"].data,
    )

    loader = paddle.io.DataLoader(
        dataset, batch_size=3, collate_fn=DefaultCollator(), return_list=True
    )
    batch = next(iter(loader))
    model = _build_model(fake_atom_p=fake_atom_p)
    model.train()
    optimizer = paddle.optimizer.Adam(learning_rate=1e-3, parameters=model.parameters())
    before = {key: value.numpy().copy() for key, value in model.state_dict().items()}
    loss = model(batch)["loss_dict"]["loss"]
    assert np.isfinite(float(loss))
    loss.backward()
    for parameter in model.parameters():
        if not parameter.stop_gradient:
            assert parameter.grad is not None, parameter.name
            assert np.isfinite(parameter.grad.numpy()).all(), parameter.name
    optimizer.step()
    assert all(
        np.isfinite(value.numpy()).all() for value in model.state_dict().values()
    )
    assert any(
        not np.array_equal(before[key], value.numpy())
        for key, value in model.state_dict().items()
    )


def test_mixed_interpolation_training_and_sampling():
    scheduler_config = {
        "__class_name__": "InterpolantScheduler",
        "__init_params__": {
            "schedule_type": {
                "positions": "cosine",
                "atom_types": "linear",
                "atom_charges": "cosine",
                "bond_types": "cosine",
            },
            "cosine_params": {
                "positions": 2.0,
                "atom_charges": 0.5,
                "bond_types": 0.75,
            },
        },
    }
    model = _build_model(interpolant_scheduler_config=scheduler_config)
    scheduler = model.interpolant_scheduler
    for dtype in ("float32", "float64"):
        t = paddle.to_tensor([0.0, 1e-4, 0.01, 0.3, 0.7, 1.0], dtype=dtype)
        times = t.numpy().astype("float64")
        expected = np.stack(
            [
                np.sin(np.pi * 0.5 * times**2) ** 2,
                times,
                np.sin(np.pi * 0.5 * times**0.5) ** 2,
                np.sin(np.pi * 0.5 * times**0.75) ** 2,
            ],
            axis=1,
        )
        np.testing.assert_allclose(
            scheduler.alpha_t(t).numpy(), expected, rtol=1e-5, atol=0
        )
        derivative = scheduler.alpha_t_prime(t).numpy()
        assert np.isfinite(derivative).all()
        np.testing.assert_allclose(derivative[0], [0, 1, np.pi**2 / 4, 0])
        np.testing.assert_array_equal(derivative[-1], [0, 1, 0, 0])

        interior = t[1:-1].detach()
        interior.stop_gradient = False
        gradient = paddle.grad(scheduler.alpha_t(interior).sum(), interior)[0]
        np.testing.assert_allclose(
            scheduler.alpha_t_prime(interior).sum(axis=1).numpy(),
            gradient.numpy(),
            rtol=1e-5,
        )

    batch = _build_batch([4, 5])
    optimizer = paddle.optimizer.Adam(learning_rate=1e-3, parameters=model.parameters())
    loss = model(batch)["loss_dict"]["loss"]
    assert np.isfinite(float(loss))
    loss.backward()
    assert all(
        parameter.grad is not None and np.isfinite(parameter.grad.numpy()).all()
        for parameter in model.parameters()
    )
    optimizer.step()
    assert all(np.isfinite(parameter.numpy()).all() for parameter in model.parameters())
    result = model.sample(batch, num_inference_steps=4)["pred"]
    _assert_molecule_dicts(result, paddle.to_tensor([4, 5]))
    assert all(np.isfinite(molecule["positions"]).all() for molecule in result)

    params = scheduler_config["__init_params__"]
    for schedule_type in (
        "unsupported",
        {**params["schedule_type"], "positions": "unsupported"},
        {"positions": "linear"},
    ):
        with pytest.raises(ValueError, match="schedule_type"):
            _build_model(
                interpolant_scheduler_config={
                    **scheduler_config,
                    "__init_params__": {**params, "schedule_type": schedule_type},
                }
            )
    for nu in (0.0, 0.25, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="finite and >= 0.5"):
            _build_model(
                interpolant_scheduler_config={
                    **scheduler_config,
                    "__init_params__": {
                        **params,
                        "cosine_params": {**params["cosine_params"], "positions": nu},
                    },
                }
            )


def test_ctmc_ignore_semantics():
    """Regression test: already-unmasked nodes must be ignored in the loss."""
    model = _build_model()
    vf = model.vector_field
    batch = model._prepare_batch(_build_batch([4]))
    n_nodes = int(batch["positions"].shape[0])

    t = paddle.full([1], 0.0)
    state = vf.sample_conditional_path(
        batch,
        t,
        batch["prior_positions"],
        batch["node_batch_idx"],
        batch["edge_batch_idx"],
        batch["upper_edge_mask"],
    )
    # at t=0 every categorical feature is the mask token
    assert int((state["atom_types"].argmax(-1) == 5).sum()) == n_nodes

    targets = model._build_targets(batch, state, batch["upper_edge_mask"])
    assert int((targets["atom_types"] != -100).sum()) == n_nodes

    # reveal one node: its target must flip to the ignore index
    revealed = state["atom_types"].clone()
    revealed[0] = 0.0
    revealed[0, 3] = 1.0
    state["atom_types"] = revealed
    targets = model._build_targets(batch, state, batch["upper_edge_mask"])
    assert int(targets["atom_types"][0]) == -100
    assert int((targets["atom_types"][1:] != -100).sum()) == n_nodes - 1


# ----------------------------- sampling chain -----------------------------


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


def test_sample_with_fake_atoms_decodes_real_atoms():
    # fix the RNG so the decoded real-atom count is reproducible
    paddle.seed(7)
    np.random.seed(7)
    model = _build_model(
        vector_field_config=SMALL_VECTOR_FIELD_RELEASED,
        fake_atom_p=0.3,
    )
    dataset = FlowMol3GeomSamplingDataset(total_num=2, seed=7)
    batch = DefaultCollator()([dataset[i] for i in range(len(dataset))])
    num_atoms = paddle.to_tensor(batch["structure_array"]["num_atoms"])

    result = model.sample(batch, num_inference_steps=4)["pred"]
    assert len(result) == num_atoms.shape[0]
    decoded_atoms = paddle.to_tensor([molecule["num_atoms"] for molecule in result])
    assert bool(((decoded_atoms > 0) & (decoded_atoms <= num_atoms)).all())
    _assert_molecule_dicts(result, decoded_atoms)


def test_sampling_framework_end_to_end(tmp_path, monkeypatch):
    csv_path = Path(_write_synthetic_qm9_csv(tmp_path))
    monkeypatch.chdir(tmp_path)
    checkpoint_path = str(tmp_path / "best.pdparams")
    paddle.save(_build_model().state_dict(), checkpoint_path)

    config = OmegaConf.load(CONFIG_PATH)
    config.Model.__init_params__.vector_field_config = OmegaConf.create(
        SMALL_VECTOR_FIELD
    )
    config.Model.__init_params__.default_n_timesteps = 4
    config.Sampler.data.dataset.__init_params__.path = "???"
    config.Sampler.input_reference = "${Sampler.data.dataset.__init_params__.path}"
    sampler_config_path = tmp_path / "sampler_config.yaml"
    OmegaConf.save(config, str(sampler_config_path))

    overrides = [
        "Dataset.train=null",
        "Sampler.data.dataset.__init_params__.path=ignored.csv",
        "Sampler.data.dataset.__init_params__.total_num=4",
        "Sampler.data.sampler.__init_params__.batch_size=2",
        "Sampler.model_sample_params.num_inference_steps=4",
    ]
    overrides.append(f"Sampler.data.dataset.__init_params__.path={csv_path}")
    sampler = MolecularSampler(
        config_path=str(sampler_config_path),
        checkpoint_path=checkpoint_path,
        config_overrides=overrides,
    )
    assert sampler.sample_config["input_reference"] == str(csv_path)
    results = sampler.sample_by_dataloader(save_path=str(tmp_path / "out"))
    assert results is sampler.last_samples
    assert len(results["pred"]) == 4
    sampled_num_atoms = sampler._sample_loader.dataset.sampled_num_atoms
    _assert_molecule_dicts(results["pred"], paddle.to_tensor(sampled_num_atoms))

    saved = Chem.SDMolSupplier(
        str(tmp_path / "out" / "flowmol_samples.sdf"),
        sanitize=False,
        removeHs=False,
    )
    assert len(saved) == 4
    assert all(molecule is not None for molecule in saved)

    sampler.sample_config.pop("save_samples")
    sampler.sample_by_dataloader(save_path=str(tmp_path / "disabled"))
    assert not (tmp_path / "disabled" / "flowmol_samples.sdf").exists()

    cli_args = [
        "sample.py",
        "--sampler=MolecularSampler",
        f"--config_path={sampler_config_path}",
        f"--checkpoint_path={checkpoint_path}",
        f"--input_path={csv_path}",
        f"--output_path={tmp_path / 'cli'}",
        *overrides[:-1],
    ]
    monkeypatch.setattr(sys, "argv", [*cli_args, "--mode=by_dataloader"])
    sampling_cli.main()
    cli_samples = Chem.SDMolSupplier(
        str(tmp_path / "cli" / "flowmol_samples.sdf"), sanitize=False
    )
    assert len(cli_samples) == 4

    monkeypatch.setattr(sys, "argv", cli_args)
    with pytest.raises(SystemExit) as error:
        sampling_cli.main()
    assert error.value.code == 2


def test_registered_sampling_framework_end_to_end(tmp_path, monkeypatch):
    csv_path = _write_synthetic_qm9_csv(tmp_path)
    package_dir = tmp_path / "flowmol_test"
    checkpoint_dir = package_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    paddle.save(_build_model().state_dict(), str(checkpoint_dir / "selected.pdparams"))

    config = OmegaConf.load(CONFIG_PATH)
    config.Model.__init_params__.vector_field_config = OmegaConf.create(
        SMALL_VECTOR_FIELD
    )
    config.Model.__init_params__.default_n_timesteps = 4
    config.Dataset.train = None
    config.Sampler.data.dataset.__init_params__.path = "???"
    config.Sampler.data.dataset.__init_params__.total_num = 2
    config.Sampler.data.sampler.__init_params__.batch_size = 2
    config.Sampler.model_sample_params.num_inference_steps = 4
    config.Sampler.output_dir = str(tmp_path / "registered_output")
    config.Sampler.input_reference = "${Sampler.data.dataset.__init_params__.path}"
    OmegaConf.save(config, str(package_dir / "flowmol_test.yaml"))

    package_url = "https://example.com/flowmol_test.zip"
    monkeypatch.setitem(molecular_sampler.MODEL_REGISTRY, "flowmol_test", package_url)
    downloads = []

    def resolve_package(url):
        downloads.append(url)
        return str(tmp_path)

    monkeypatch.setattr(
        molecular_sampler.download, "get_weights_path_from_url", resolve_package
    )
    sampler = MolecularSampler(
        model_name="flowmol_test",
        weights_name="selected.pdparams",
        config_overrides=[f"Sampler.data.dataset.__init_params__.path={csv_path}"],
    )
    assert downloads == [package_url]
    assert sampler.pretrained_weight_name == "selected.pdparams"
    assert sampler.sample_config["input_reference"] == csv_path

    class SampleCountMetric:
        def update_step(self, result, batch, stage):
            self.count = len(result["samples"]["pred"])

        def compute_epoch(self, stage):
            return {"sample_count": self.count}

    sampler.metric_modules = {"count": SampleCountMetric()}
    metrics = sampler.compute_metric(save_path=str(tmp_path / "metrics"))
    assert metrics == {"sample_count": 2}
    saved = Chem.SDMolSupplier(
        str(tmp_path / "metrics" / "flowmol_samples.sdf"), sanitize=False
    )
    assert len(saved) == 2

    sampler.rank = 1
    sampler.sample_by_dataloader(save_path=str(tmp_path / "other_rank"))
    assert not (tmp_path / "other_rank" / "flowmol_samples.sdf").exists()

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sample.py",
            "--model_name=flowmol_test",
            "--sampler=MolecularSampler",
            "--weights_name=selected.pdparams",
            f"--input_path={csv_path}",
            f"--output_path={tmp_path / 'registered_cli'}",
            "--mode=by_dataloader",
        ],
    )
    sampling_cli.main()
    assert downloads == [package_url, package_url]
    saved = Chem.SDMolSupplier(
        str(tmp_path / "registered_cli" / "flowmol_samples.sdf"), sanitize=False
    )
    assert len(saved) == 2


def test_integrate_feeds_evolving_state_to_network():
    """Regression: the sampled trajectory must feed the network its own state.

    The network reads ``state["atom_types"]``; the CTMC integration must
    update the same keys, otherwise every denoise step would see the initial
    mask prior and the sampled molecules collapse to junk fragments.
    """
    model = _build_model()
    vf = model.vector_field
    num_atoms = paddle.to_tensor([4, 5], dtype="int64")
    state = model._build_sampling_state(num_atoms.numpy(), paddle.get_device())

    seen_masks = []
    orig_step = vf._denoise_step

    def spy(state_in, *args, **kwargs):
        seen_masks.append(int((state_in["atom_types"].argmax(-1) == 5).sum()))
        return orig_step(state_in, *args, **kwargs)

    vf._denoise_step = spy
    with paddle.no_grad():
        vf.integrate(
            state,
            state["node_batch_idx"],
            state["upper_edge_mask"],
            n_timesteps=10,
            stochasticity=0.0,
            high_confidence_threshold=0.0,
        )
    vf._denoise_step = orig_step

    total_masked = int(num_atoms.sum())
    assert seen_masks[0] == total_masked
    assert min(seen_masks[1:]) < total_masked


# ------------------------------- CINN runtime ------------------------------


def _build_unique_model(vector_field_config=None):
    with paddle.utils.unique_name.guard():
        return _build_model(vector_field_config=vector_field_config)


def _training_loader(dataset, batch_size=2):
    samples = [dataset[i] for i in range(2)]
    return paddle.io.DataLoader(
        samples,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=DefaultCollator(),
        return_list=True,
    )


def _build_trainer(
    model, output_dir, max_epochs, backend, train_loader, val_loader, optimizer_cls=None
):
    optimizer_cls = paddle.optimizer.Adam if optimizer_cls is None else optimizer_cls
    optimizer = optimizer_cls(learning_rate=1e-3, parameters=model.parameters())
    trainer = BaseTrainer(
        trainer_config(output_dir, max_epochs),
        model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        optimizer=optimizer,
        execution_config={"backend": backend},
    )
    return trainer, optimizer


@pytest.mark.parametrize(
    "vector_field_config",
    [SMALL_VECTOR_FIELD, SMALL_VECTOR_FIELD_RELEASED],
    ids=["base", "released"],
)
def test_cinn_dispatch_is_numerically_identical(vector_field_config, monkeypatch):
    """Selecting the CINN backend must not change the eager numerics.

    The compiled runtime is mocked out so the test runs on CPU: the worker
    records the boundary name while running the eager callable it wraps.
    """

    paddle.seed(7)
    eager_model = _build_model(vector_field_config=vector_field_config)
    paddle.seed(7)
    cinn_model = _build_model(vector_field_config=vector_field_config)

    batch = _build_batch([4, 5])
    paddle.seed(11)
    eager_loss = eager_model(batch)["loss_dict"]

    recorded = []

    def record_run_runtime(self, name, function, *args, **kwargs):
        recorded.append(name)
        return function(*args, **kwargs)

    monkeypatch.setattr(FlowMolVectorField, "_run_runtime", record_run_runtime)
    cinn_model.set_execution_backend("cinn")
    assert cinn_model.execution_backend == "cinn"
    paddle.seed(11)
    cinn_loss = cinn_model(batch)["loss_dict"]

    assert recorded == ["denoise_step"]
    for key in eager_loss:
        assert float(eager_loss[key]) == float(cinn_loss[key])


def test_cinn_trainer_resume_orchestration(tmp_path, monkeypatch):
    """Checkpoint save/resume stays consistent with CINN dispatch selected."""

    patch_cinn_runtime(monkeypatch, FlowMolVectorField)
    dataset = _training_dataset(tmp_path)
    train_loader = _training_loader(dataset)
    val_loader = _training_loader(dataset)

    initial_state = {
        key: value.clone() for key, value in _build_unique_model().state_dict().items()
    }

    first_model = _build_unique_model()
    first_model.set_state_dict(initial_state)
    first_trainer, _ = _build_trainer(
        first_model, tmp_path / "first", 1, "cinn", train_loader, val_loader
    )
    with paddle.utils.unique_name.guard():
        first_trainer.train()

    checkpoint_dir = tmp_path / "first" / "checkpoints"
    for prefix in ("epoch_1", "latest", "best"):
        for suffix in ("pdparams", "pdopt", "pdstates"):
            assert (checkpoint_dir / f"{prefix}.{suffix}").is_file()
    assert_checkpoint_keys_unchanged(
        paddle.load(str(checkpoint_dir / "latest.pdparams")), first_model
    )

    resumed_model = _build_unique_model()
    resumed_trainer, resumed_optimizer = _build_trainer(
        resumed_model, tmp_path / "resumed", 2, "cinn", train_loader, val_loader
    )
    saved_model = paddle.load(str(checkpoint_dir / "latest.pdparams"))
    saved_optimizer = paddle.load(str(checkpoint_dir / "latest.pdopt"))
    resumed_epoch = resumed_trainer.train_epoch
    restored = []

    def check_restored_state(*args, **kwargs):
        assert resumed_trainer.state.global_step == first_trainer.state.global_step
        assert resumed_optimizer.get_lr() == first_trainer.optimizer.get_lr()
        for key, value in resumed_model.state_dict().items():
            np.testing.assert_array_equal(value.numpy(), saved_model[key].numpy())
        optimizer_state = resumed_optimizer.state_dict()
        assert optimizer_state.keys() == saved_optimizer.keys()
        for key, value in optimizer_state.items():
            np.testing.assert_array_equal(value, saved_optimizer[key])
        restored.append(True)
        return resumed_epoch(*args, **kwargs)

    monkeypatch.setattr(resumed_trainer, "train_epoch", check_restored_state)
    with paddle.utils.unique_name.guard():
        resumed_trainer.train(resume_from_checkpoint=str(checkpoint_dir / "latest"))
    assert restored == [True]

    straight_model = _build_unique_model()
    straight_model.set_state_dict(initial_state)
    straight_trainer, straight_optimizer = _build_trainer(
        straight_model, tmp_path / "straight", 2, "cinn", train_loader, val_loader
    )
    with paddle.utils.unique_name.guard():
        straight_trainer.train()

    assert first_trainer.state.global_step == 1
    assert resumed_trainer.state.global_step == 2
    assert resumed_trainer.state.epoch == 2
    assert resumed_optimizer.get_lr() == straight_optimizer.get_lr()
    assert resumed_model.state_dict().keys() == straight_model.state_dict().keys()
    resumed_state = resumed_optimizer.state_dict()
    straight_state = straight_optimizer.state_dict()
    assert resumed_state
    assert resumed_state.keys() == straight_state.keys()


@pytest.mark.parametrize(
    "vector_field_config",
    [SMALL_VECTOR_FIELD, {**SMALL_VECTOR_FIELD_RELEASED, "scprop": 0.0}],
    ids=["base", "self_conditioning"],
)
@requires_gpu_cinn
def test_gpu_cinn_matches_eager(tmp_path, vector_field_config):
    """Compare compiled training with eager, including forced self-conditioning.

    Warm up compilation before resetting the seeds so both modes sample the
    same interpolation times and masks during the compared training epoch.
    """

    gpu_cinn_or_skip()
    dataset = _training_dataset(tmp_path)
    train_loader = _training_loader(dataset)
    val_loader = _training_loader(dataset)

    eager_model = _build_unique_model(vector_field_config)
    cinn_model = _build_unique_model(vector_field_config)
    cinn_model.set_state_dict(eager_model.state_dict())
    # SGD keeps the weight comparison free of Adam's first-step sign
    # normalization, which would amplify compiled gradient noise.
    eager_trainer, _ = _build_trainer(
        eager_model,
        tmp_path / "eager",
        1,
        "eager",
        train_loader,
        val_loader,
        optimizer_cls=paddle.optimizer.SGD,
    )
    cinn_trainer, _ = _build_trainer(
        cinn_model,
        tmp_path / "cinn",
        1,
        "cinn",
        train_loader,
        val_loader,
        optimizer_cls=paddle.optimizer.SGD,
    )

    cinn_model.train()
    cinn_model(_build_batch([4, 5]))
    assert cinn_model.vector_field._runtime_cache, "CINN compiled no boundary"

    paddle.seed(2026)
    np.random.seed(2026)
    with paddle.utils.unique_name.guard():
        eager_trainer.train()
    paddle.seed(2026)
    np.random.seed(2026)
    with paddle.utils.unique_name.guard():
        cinn_trainer.train()

    assert eager_trainer.state.global_step == cinn_trainer.state.global_step == 1
    eager_state = eager_model.state_dict()
    cinn_state = cinn_model.state_dict()
    assert eager_state.keys() == cinn_state.keys()
    for key in eager_state:
        np.testing.assert_allclose(
            cinn_state[key].numpy(),
            eager_state[key].numpy(),
            atol=2e-5,
            rtol=2e-5,
        )


@pytest.mark.parametrize(
    "vector_field_config",
    [SMALL_VECTOR_FIELD, RELEASED_VECTOR_FIELD],
    ids=["base", "released"],
)
@requires_gpu_cinn
def test_gpu_cinn_sampling_matches_eager(vector_field_config):
    """The compiled sampling workflow must reproduce the eager molecules.

    Sampling is the documented CINN workflow for FlowMol; the released
    self-conditioning vector field is exercised here on real hardware.
    """

    gpu_cinn_or_skip()
    eager_model = _build_unique_model(vector_field_config)
    eager_model.eval()
    cinn_model = _build_unique_model(vector_field_config)
    cinn_model.set_state_dict(eager_model.state_dict())
    cinn_model.eval()
    cinn_model.set_execution_backend("cinn")

    batch = {"structure_array": {"num_atoms": np.array([5, 6], dtype="int64")}}
    cinn_model.sample(batch, num_inference_steps=2)
    assert cinn_model.vector_field._runtime_cache, "CINN compiled no boundary"

    paddle.seed(123)
    eager_mols = eager_model.sample(batch, num_inference_steps=6)["pred"]
    paddle.seed(123)
    cinn_mols = cinn_model.sample(batch, num_inference_steps=6)["pred"]

    for eager_mol, cinn_mol in zip(eager_mols, cinn_mols):
        assert eager_mol["num_atoms"] == cinn_mol["num_atoms"]
        np.testing.assert_allclose(
            np.asarray(eager_mol["positions"]),
            np.asarray(cinn_mol["positions"]),
            atol=1e-5,
        )

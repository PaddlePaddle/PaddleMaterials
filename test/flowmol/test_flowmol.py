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

from pathlib import Path

import numpy as np
import paddle
import pytest
from cinn_workflow_harness import assert_checkpoint_keys_unchanged
from cinn_workflow_harness import gpu_cinn_or_skip
from cinn_workflow_harness import patch_cinn_runtime
from cinn_workflow_harness import requires_gpu_cinn
from cinn_workflow_harness import trainer_config
from omegaconf import OmegaConf

from ppmat.datasets.collate_fn import DefaultCollator
from ppmat.datasets.custom_data_type import ConcatData
from ppmat.datasets.flowmol_dataset import _FLOWMOL3_GEOM_ATOM_COUNTS
from ppmat.datasets.flowmol_dataset import FlowMol3GeomSamplingDataset
from ppmat.datasets.flowmol_dataset import FlowMolDataset
from ppmat.models import build_model
from ppmat.models.flowmol.vector_field import CTMCVectorField
from ppmat.sampler import MolecularSampler
from ppmat.trainer.base_trainer import BaseTrainer

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = (
    REPO_ROOT / "structure_generation" / "configs" / "flowmol" / "flowmol_qm9.yaml"
)
FLOWMOL3_CONFIG_PATH = (
    REPO_ROOT / "structure_generation" / "configs" / "flowmol" / "flowmol3.yaml"
)

PRIOR_CONFIG = {
    "positions": {"type": "centered-normal", "align": True, "kwargs": {"std": 1.0}},
    "atom_types": {"type": "ctmc"},
    "atom_charges": {"type": "ctmc"},
    "bond_types": {"type": "ctmc"},
}

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
    init_params = {
        "atom_type_map": ["C", "H", "N", "O", "F"],
        "total_loss_weights": {
            "positions": 3.0,
            "atom_types": 0.4,
            "atom_charges": 1.0,
            "bond_types": 2.0,
        },
        "interpolant_scheduler_config": {
            "__class_name__": "FlowMolInterpolantScheduler",
            "__init_params__": {
                "schedule_type": {
                    "positions": "linear",
                    "atom_types": "linear",
                    "atom_charges": "linear",
                    "bond_types": "linear",
                },
                "cosine_params": {},
            },
        },
        "vector_field_config": (
            SMALL_VECTOR_FIELD if vector_field_config is None else vector_field_config
        ),
        "prior_config": PRIOR_CONFIG,
        "time_scaled_loss": False,
        "fake_atom_p": 0.0,
        "distort_p": 0.0,
        "default_n_timesteps": 4,
    }
    init_params.update(overrides)
    model = build_model({"__class_name__": "FlowMol", "__init_params__": init_params})
    model.eval()
    return model


def _write_synthetic_split(tmp_path, num_molecules=6, seed=0):
    rng = np.random.RandomState(seed)
    positions, atom_types, atom_charges = [], [], []
    bond_types, bond_idxs = [], []
    node_idx_array = []
    edge_idx_array = []
    node_offset = 0
    edge_offset = 0
    for _ in range(num_molecules):
        n_atoms = int(rng.randint(3, 6))
        positions.append(rng.standard_normal((n_atoms, 3)).astype("float32"))
        one_hot = np.zeros((n_atoms, 5), dtype=bool)
        one_hot[np.arange(n_atoms), rng.randint(0, 5, n_atoms)] = True
        atom_types.append(one_hot)
        atom_charges.append(rng.randint(-2, 4, n_atoms).astype("int32"))

        upper_i, upper_j = np.triu_indices(n_atoms, k=1)
        keep = rng.rand(upper_i.shape[0]) < 0.4
        mol_bond_idxs = np.stack((upper_i[keep], upper_j[keep]), axis=1).astype("int32")
        bond_idxs.append(mol_bond_idxs)
        bond_types.append(rng.randint(0, 4, mol_bond_idxs.shape[0]).astype("int32"))

        node_idx_array.append([node_offset, node_offset + n_atoms])
        edge_idx_array.append([edge_offset, edge_offset + mol_bond_idxs.shape[0]])
        node_offset += n_atoms
        edge_offset += mol_bond_idxs.shape[0]

    path = tmp_path / "train_data_processed.npz"
    np.savez(
        path,
        positions=np.concatenate(positions, axis=0),
        atom_types=np.concatenate(atom_types, axis=0).astype("float32"),
        atom_charges=np.concatenate(atom_charges, axis=0),
        bond_types=np.concatenate(bond_types, axis=0),
        bond_idxs=np.concatenate(bond_idxs, axis=0),
        node_idx_array=np.array(node_idx_array, dtype="int32"),
        edge_idx_array=np.array(edge_idx_array, dtype="int32"),
    )
    np.savez(
        tmp_path / "train_data_n_atoms_histogram.npz",
        n_atoms=np.array([3, 4, 5], dtype="int64"),
        n_atom_counts=np.array([10, 20, 30], dtype="int64"),
    )
    return str(path)


def _training_dataset(tmp_path, **kwargs):
    path = _write_synthetic_split(tmp_path)
    dataset_kwargs = {
        "path": path,
        "atom_map": ["C", "H", "N", "O", "F"],
        "prior_align": True,
        "seed": 0,
    }
    dataset_kwargs.update(kwargs)
    return FlowMolDataset(**dataset_kwargs)


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
        _build_model(
            prior_config={
                **PRIOR_CONFIG,
                "positions": {"type": "unsupported", "align": False, "kwargs": {}},
            }
        )


def test_train_data_pipeline_feeds_forward(tmp_path):
    dataset = _training_dataset(tmp_path)
    assert len(dataset) == 6

    sample = dataset[0]
    n_atoms = int(sample["structure_array"]["num_atoms"].data[0])
    positions = sample["structure_array"]["positions"].data
    assert positions.shape == (n_atoms, 3)
    assert np.allclose(positions.sum(axis=0), 0.0, atol=1e-4)

    batch = DefaultCollator()([dataset[i] for i in range(3)])
    assert batch["structure_array"]["num_atoms"].shape[0] == 3

    model = _build_model()
    prepared = model._prepare_batch(batch)
    mask = prepared["upper_edge_mask"]
    assert prepared["batch_size"] == 3
    assert int(mask.sum()) == mask.shape[0] // 2
    assert np.isfinite(float(model(batch)["loss_dict"]["loss"]))


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


def test_sample_produces_valid_molecules(tmp_path):
    model = _build_model()
    num_atoms = paddle.to_tensor([4, 5, 6], dtype="int64")
    batch = {"structure_array": {"num_atoms": num_atoms}}

    result = model.sample(batch, num_inference_steps=4)["pred"]
    _assert_molecule_dicts(result, num_atoms)

    model.save_samples(result, str(tmp_path / "out"))
    assert (tmp_path / "out" / "flowmol_samples.sdf").exists()


def test_sample_with_fake_atoms_decodes_real_atoms(tmp_path):
    # fix the RNG so the decoded real-atom count is reproducible
    paddle.seed(7)
    np.random.seed(7)
    model = _build_model(
        vector_field_config=SMALL_VECTOR_FIELD_RELEASED,
        fake_atom_p=0.3,
    )
    num_atoms = paddle.to_tensor([4, 5], dtype="int64")
    batch = {"structure_array": {"num_atoms": num_atoms}}

    result = model.sample(batch, num_inference_steps=4)["pred"]
    assert len(result) == num_atoms.shape[0]
    for molecule, expected_atoms in zip(result, num_atoms.numpy().tolist()):
        # decoded fake atoms are dropped, so the real atom count can shrink
        assert 0 < molecule["num_atoms"] <= expected_atoms
        assert len(molecule["atom_types"]) == molecule["num_atoms"]
        assert np.asarray(molecule["positions"]).shape == (molecule["num_atoms"], 3)
        assert all(
            symbol in ["C", "H", "N", "O", "F", "Se"]
            for symbol in molecule["atom_types"]
        )
        for src, dst, bond_type in molecule["bonds"]:
            assert 0 <= src < molecule["num_atoms"]
            assert 0 <= dst < molecule["num_atoms"]
            assert 1 <= bond_type <= 4


def test_sampling_framework_end_to_end(tmp_path):
    path = _write_synthetic_split(tmp_path)
    histogram_path = Path(path).with_name("train_data_n_atoms_histogram.npz")
    checkpoint_path = str(tmp_path / "best.pdparams")
    paddle.save(_build_model().state_dict(), checkpoint_path)

    config = OmegaConf.load(CONFIG_PATH)
    config.Model.__init_params__.vector_field_config = OmegaConf.create(
        SMALL_VECTOR_FIELD
    )
    config.Model.__init_params__.default_n_timesteps = 4
    sampler_config_path = tmp_path / "sampler_config.yaml"
    OmegaConf.save(config, str(sampler_config_path))

    sampler = MolecularSampler(
        config_path=str(sampler_config_path),
        checkpoint_path=checkpoint_path,
        config_overrides=[
            "Dataset.train=null",
            f"Sampler.data.dataset.__init_params__.path={histogram_path}",
            "Sampler.data.dataset.__init_params__.total_num=4",
            f"Sampler.output_dir={tmp_path / 'out'}",
            "Sampler.model_sample_params.num_inference_steps=4",
        ],
    )
    results = sampler.sample_by_dataloader(save_path=str(tmp_path / "out"))
    assert results is sampler.last_samples
    assert len(results["pred"]) == 4
    sampled_num_atoms = sampler._sample_loader.dataset.sampled_num_atoms
    _assert_molecule_dicts(results["pred"], paddle.to_tensor(sampled_num_atoms))

    save_samples = getattr(sampler.model, "save_samples", None)
    assert callable(save_samples)
    save_samples(sampler.last_samples["pred"], sampler.output_dir)
    assert (tmp_path / "out" / "flowmol_samples.sdf").exists()


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
    orig_step = vf._runtime_denoise_step

    def spy(state_in, *args, **kwargs):
        seen_masks.append(int((state_in["atom_types"].argmax(-1) == 5).sum()))
        return orig_step(state_in, *args, **kwargs)

    vf._runtime_denoise_step = spy
    with paddle.no_grad():
        vf.integrate(
            state,
            state["node_batch_idx"],
            state["upper_edge_mask"],
            n_timesteps=10,
            stochasticity=0.0,
            high_confidence_threshold=0.0,
        )
    vf._runtime_denoise_step = orig_step

    total_masked = int(num_atoms.sum())
    assert seen_masks[0] == total_masked
    assert min(seen_masks[1:]) < total_masked


# ----------------------------- evaluation chain ----------------------------


def test_geom_sampling_histogram_is_inlined():
    """Regression: the released-model histogram must be inlined and exact."""
    assert len(_FLOWMOL3_GEOM_ATOM_COUNTS) == 157
    assert sum(_FLOWMOL3_GEOM_ATOM_COUNTS.values()) == 5730535
    assert not hasattr(FlowMol3GeomSamplingDataset, "url")

    dataset = FlowMol3GeomSamplingDataset(total_num=4, seed=0)
    repeated = FlowMol3GeomSamplingDataset(total_num=4, seed=0)
    assert len(dataset) == 4
    assert np.array_equal(dataset.sampled_num_atoms, repeated.sampled_num_atoms)
    assert dataset[0] == {
        "structure_array": {"num_atoms": int(dataset.sampled_num_atoms[0])}
    }


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

    monkeypatch.setattr(CTMCVectorField, "_run_runtime", record_run_runtime)
    cinn_model.set_execution_backend("cinn")
    assert cinn_model.execution_backend == "cinn"
    paddle.seed(11)
    cinn_loss = cinn_model(batch)["loss_dict"]

    assert recorded == ["denoise_step"]
    for key in eager_loss:
        assert float(eager_loss[key]) == float(cinn_loss[key])


def test_cinn_trainer_resume_orchestration(tmp_path, monkeypatch):
    """Checkpoint save/resume stays consistent with CINN dispatch selected."""

    patch_cinn_runtime(monkeypatch, CTMCVectorField)
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
    with paddle.utils.unique_name.guard():
        resumed_trainer.train(resume_from_checkpoint=str(checkpoint_dir / "latest"))

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


@requires_gpu_cinn
def test_gpu_cinn_matches_eager(tmp_path):
    """One real compiled training epoch must match the eager weights.

    Only the base vector field is trained here: the released self-conditioning
    configuration hits a Paddle CINN symbol-shape limitation on the training
    (backward) path, so the documented sampling workflow is covered by
    ``test_gpu_cinn_sampling_matches_eager`` instead. Compilation itself draws
    from the global RNG, so the compiled boundary is warmed up before the
    seeding below; otherwise the eager and CINN runs would sample different
    interpolation times and masks.
    """

    gpu_cinn_or_skip()
    dataset = _training_dataset(tmp_path)
    train_loader = _training_loader(dataset)
    val_loader = _training_loader(dataset)

    eager_model = _build_unique_model()
    cinn_model = _build_unique_model()
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

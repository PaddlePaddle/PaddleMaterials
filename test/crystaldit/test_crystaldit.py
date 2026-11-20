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
"""Integration tests for the CrystalDiT model.

Covers the end-to-end chains required by the project conventions:
config -> model building -> forward loss, dataset -> collate -> forward,
and the DDPM sampling loop with post-processing back to crystal structures.
"""

import os

import numpy as np
import paddle
import paddle.io as io
import pandas as pd
import pytest
from omegaconf import OmegaConf

from ppmat.datasets import build_dataloader
from ppmat.datasets.collate_fn import DefaultCollator
from ppmat.models import build_model

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONFIG_PATH = os.path.join(
    REPO_ROOT,
    "structure_generation",
    "configs",
    "crystaldit",
    "crystaldit_mp20.yaml",
)


def build_small_model():
    config = {
        "__class_name__": "CrystalDiT",
        "__init_params__": {
            "max_atoms": 20,
            "hidden_size": 64,
            "depth": 2,
            "num_heads": 4,
            "mlp_ratio": 4.0,
            "max_length": 46.7425,
            "num_train_timesteps": 50,
        },
    }
    return build_model(config)


def build_batch(batch_size=4):
    paddle.seed(42)
    np.random.seed(42)
    rng = np.random.RandomState(0)
    lattice_vectors = rng.randn(batch_size, 3, 3).astype("float32")
    atom_features = rng.randn(batch_size, 20, 5).astype("float32")
    return {
        "lattice_vectors": paddle.to_tensor(lattice_vectors),
        "atom_features": paddle.to_tensor(atom_features),
    }


def test_config_builds_model_with_expected_structure():
    config = OmegaConf.to_container(OmegaConf.load(CONFIG_PATH), resolve=True)
    model = build_model(config["Model"])
    assert type(model).__name__ == "CrystalDiT"
    assert model.num_train_timesteps == 1000
    named_params = dict(model.named_parameters())
    assert "backbone.crystal_embedder.lattice_embedder.weight" in named_params
    assert "backbone.blocks.0.dit_block.attn.qkv.weight" in named_params


def test_forward_returns_loss_dict_with_expected_keys():
    model = build_small_model()
    model.train()
    batch = build_batch()
    result = model(batch)
    assert "loss_dict" in result
    assert "loss" in result["loss_dict"]
    assert "lattice_loss" in result["loss_dict"]
    assert "atom_loss" in result["loss_dict"]
    assert "label_dict" in result
    loss = result["loss_dict"]["loss"]
    assert loss.ndim == 0
    float(loss)
    loss.backward()


def test_forward_loss_is_deterministic_given_fixed_seed():
    model = build_small_model()
    model.eval()
    losses = []
    for _ in range(2):
        paddle.seed(7)
        np.random.seed(7)
        result = model(build_batch())
        losses.append(float(result["loss_dict"]["loss"]))
    assert abs(losses[0] - losses[1]) < 1e-6


def test_dataset_and_collate_feed_forward():
    from ppmat.datasets import CrystalDiTDataset

    dataset = CrystalDiTDataset(
        path=os.path.join(REPO_ROOT, "data", "mp_20", "val.csv"),
        max_atoms=20,
        max_length=46.7425,
        num_cpus=2,
    )
    loader = build_dataloader(
        {
            "dataset": {
                "__class_name__": "CrystalDiTDataset",
                "__init_params__": {
                    "path": os.path.join(REPO_ROOT, "data", "mp_20", "val.csv"),
                    "max_atoms": 20,
                    "max_length": 46.7425,
                    "num_cpus": 2,
                },
            },
            "sampler": {
                "__class_name__": "BatchSampler",
                "__init_params__": {
                    "shuffle": False,
                    "drop_last": False,
                    "batch_size": 4,
                },
            },
        }
    )
    batch = next(iter(loader))
    assert batch["lattice_vectors"].shape == (4, 3, 3)
    assert batch["atom_features"].shape == (4, 20, 5)

    model = build_small_model()
    model.eval()
    result = model(batch)
    assert float(result["loss_dict"]["loss"]) > 0.0
    val_csv = os.path.join(REPO_ROOT, "data", "mp_20", "val.csv")
    assert len(dataset) == len(pd.read_csv(val_csv))


def test_sample_generates_parsable_structure_results():
    model = build_small_model()
    model.eval()
    batch = build_batch(batch_size=2)
    with paddle.no_grad():
        output = model.sample(batch, num_inference_steps=3)
    results = output["result"]
    assert len(results) == 2
    for item in results:
        assert set(item.keys()) == {
            "num_atoms",
            "atom_types",
            "frac_coords",
            "lattice",
        }
        num_atoms = item["num_atoms"]
        assert 0 <= num_atoms <= 20
        assert len(item["atom_types"]) == num_atoms
        assert len(item["frac_coords"]) == num_atoms
        lattice = np.array(item["lattice"]).reshape(3, 3)
        assert lattice.shape == (3, 3)
        assert np.all(np.abs(lattice) <= 46.7425 + 1e-3)
        for atomic_number in item["atom_types"]:
            assert 0 < atomic_number <= 94


def test_sample_from_registered_weights_runs():
    from ppmat.models import MODEL_REGISTRY

    if "crystaldit_mp20" not in MODEL_REGISTRY:
        pytest.skip("crystaldit_mp20 is not registered")
    from ppmat.models import build_model_from_name

    model, config = build_model_from_name("crystaldit_mp20")
    assert type(model).__name__ == "CrystalDiT"
    paddle.seed(0)
    batch = build_batch(batch_size=1)
    with paddle.no_grad():
        output = model.sample(batch, num_inference_steps=3)
    assert len(output["result"]) == 1
    assert output["result"][0]["num_atoms"] >= 0


def _make_loader(batch_size=4):
    rng = np.random.RandomState(3)
    lattice = rng.randn(8, 3, 3).astype("float32")
    atoms = rng.randn(8, 20, 5).astype("float32")

    class _FixedDataset(io.Dataset):
        def __len__(self):
            return len(lattice)

        def __getitem__(self, index):
            return {
                "lattice_vectors": lattice[index],
                "atom_features": atoms[index],
            }

    return io.DataLoader(
        _FixedDataset(),
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        collate_fn=DefaultCollator(),
        return_list=True,
        num_workers=0,
    )


def _make_trainer(model, output_dir, max_epochs, loader):
    from ppmat.trainer.base_trainer import BaseTrainer

    trainer_config = {
        "max_epochs": max_epochs,
        "output_dir": str(output_dir),
        "save_freq": 1,
        "log_freq": 100,
        "start_eval_epoch": 1,
        "eval_freq": 1,
        "seed": 2026,
        "pretrained_model_path": None,
        "resume_from_checkpoint": None,
        "compute_metric_during_train": False,
        "use_amp": False,
        "eval_with_no_grad": True,
        "gradient_accumulation_steps": 1,
        "best_metric_indicator": "train_loss",
        "name_for_best_metric": "loss",
        "greater_is_better": False,
    }
    optimizer = paddle.optimizer.Adam(learning_rate=1e-3, parameters=model.parameters())
    return (
        BaseTrainer(
            trainer_config,
            model,
            train_dataloader=loader,
            val_dataloader=loader,
            optimizer=optimizer,
            execution_config=None,
        ),
        optimizer,
    )


def test_runtime_dispatch_records_denoise_step_and_matches_eager(monkeypatch):
    """CINN dispatch (mocked compilation) hits only the denoise_step boundary
    and reproduces the eager forward bitwise on CPU."""
    from ppmat.models.crystaldit.crystaldit import CrystalDiT

    records = []

    def recording_run(self, name, function, *args, **kwargs):
        records.append(name)
        return function(*args, **kwargs)

    monkeypatch.setattr(CrystalDiT, "_run_runtime", recording_run)
    monkeypatch.setattr(
        CrystalDiT, "validate_execution_backend", lambda self, **kwargs: None
    )

    def make_model(backend):
        paddle.seed(11)
        np.random.seed(11)
        return CrystalDiT(
            max_atoms=20,
            hidden_size=32,
            depth=1,
            num_heads=4,
            num_train_timesteps=50,
            execution_backend=backend,
            runtime_options={"cinn": {"full_graph": False}},
        )

    batch = {
        "lattice_vectors": paddle.to_tensor(
            np.random.RandomState(0).randn(2, 3, 3).astype("float32")
        ),
        "atom_features": paddle.to_tensor(
            np.random.RandomState(1).randn(2, 20, 5).astype("float32")
        ),
    }

    eager_model = make_model("eager")
    cinn_model = make_model("cinn")
    assert eager_model.state_dict().keys() == cinn_model.state_dict().keys()
    for key in eager_model.state_dict():
        np.testing.assert_array_equal(
            eager_model.state_dict()[key].numpy(),
            cinn_model.state_dict()[key].numpy(),
        )

    records.clear()
    np.random.seed(99)
    paddle.seed(99)
    cinn_model.train()
    cinn_out = cinn_model(batch)
    assert set(records) == {"denoise_step"}

    records.clear()
    np.random.seed(99)
    paddle.seed(99)
    eager_model.train()
    eager_out = eager_model(batch)
    assert set(records) == {"denoise_step"}

    np.testing.assert_array_equal(
        cinn_out["loss_dict"]["loss"].numpy(),
        eager_out["loss_dict"]["loss"].numpy(),
    )


def test_trainer_resume_checkpoint_structure(tmp_path):
    """Checkpoint triple exists and a resume run continues without changing
    the parameter structure (structural-level parity only: diffusion training
    draws fresh noise and the trainer does not restore RNG state)."""
    loader = _make_loader()

    with paddle.utils.unique_name.guard():
        first_model = build_small_model()
        first_trainer, _ = _make_trainer(first_model, tmp_path / "first", 1, loader)
    first_trainer.train()

    checkpoint_dir = tmp_path / "first" / "checkpoints"
    for suffix in ("pdparams", "pdopt", "pdstates"):
        assert (checkpoint_dir / f"latest.{suffix}").is_file()
    first_global_step = first_trainer.state.global_step
    saved = paddle.load(str(checkpoint_dir / "latest.pdparams"))
    assert saved.keys() == first_model.state_dict().keys()

    with paddle.utils.unique_name.guard():
        resumed_model = build_small_model()
        resumed_trainer, resumed_optimizer = _make_trainer(
            resumed_model, tmp_path / "resumed", 2, loader
        )
    with paddle.utils.unique_name.guard():
        resumed_trainer.train(resume_from_checkpoint=str(checkpoint_dir / "latest"))

    assert resumed_trainer.state.global_step > first_global_step
    assert resumed_trainer.state.epoch == 2
    assert resumed_model.state_dict().keys() == first_model.state_dict().keys()
    assert resumed_optimizer.get_lr() > 0.0


@pytest.mark.skipif(
    os.environ.get("PPMAT_RUN_CINN_WORKFLOW_TESTS") != "1",
    reason="real CINN compilation is gated behind PPMAT_RUN_CINN_WORKFLOW_TESTS=1 "
    "and requires a CUDA Paddle build",
)
def test_gpu_cinn_matches_eager(tmp_path):
    """Gated real-compilation check: eager and cinn backends train one epoch
    from identical seeds and end with close weights."""
    if not paddle.is_compiled_with_cuda():
        pytest.skip("Paddle was not compiled with CUDA.")
    if not paddle.base.is_compiled_with_cinn():
        pytest.skip("Paddle was not compiled with CINN.")
    paddle.set_device("gpu:0")

    from ppmat.models.crystaldit.crystaldit import CrystalDiT

    loader = _make_loader()
    final_state_dicts = {}
    for backend in ("eager", "cinn"):
        paddle.seed(5)
        np.random.seed(5)
        model = CrystalDiT(
            max_atoms=20,
            hidden_size=64,
            depth=2,
            num_heads=4,
            num_train_timesteps=50,
            execution_backend=backend,
            runtime_options={"cinn": {"full_graph": False}},
        )
        trainer, _ = _make_trainer(model, tmp_path / backend, 1, loader)
        with paddle.utils.unique_name.guard():
            trainer.train()
        final_state_dicts[backend] = model.state_dict()

    eager_state = final_state_dicts["eager"]
    cinn_state = final_state_dicts["cinn"]
    assert eager_state.keys() == cinn_state.keys()
    for key in eager_state:
        np.testing.assert_allclose(
            eager_state[key].numpy(), cinn_state[key].numpy(), atol=2e-5, rtol=2e-5
        )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))

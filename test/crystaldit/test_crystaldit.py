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
from pathlib import Path

import numpy as np
import paddle
import pandas as pd
import pytest
from omegaconf import OmegaConf
from pymatgen.io.cif import CifWriter

from ppmat.datasets import build_dataloader
from ppmat.datasets.build_structure import BuildStructure
from ppmat.datasets.collate_fn import DefaultCollator
from ppmat.datasets.custom_data_type import ConcatData
from ppmat.metrics import build_metric
from ppmat.models import build_model
from ppmat.models.crystaldit.crystaldit import CrystalDiT
from ppmat.sampler import StructureSampler
from ppmat.trainer.base_trainer import BaseTrainer
from structure_generation.sample import main as sample_main

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "structure_generation/configs/crystaldit/crystaldit_mp20.yaml"


def _make_scheduler_cfg(num_train_timesteps):
    return {
        "__class_name__": "DDPMScheduler",
        "__init_params__": {
            "num_train_timesteps": num_train_timesteps,
            "beta_start": 1e-4,
            "beta_end": 0.02,
            "beta_schedule": "linear",
            "variance_type": "fixed_small",
            "clip_sample": True,
            "clip_sample_range": 1.0,
        },
    }


def _make_small_model():
    config = {
        "__class_name__": "CrystalDiT",
        "__init_params__": {
            "noise_scheduler_cfg": _make_scheduler_cfg(50),
            "max_atoms": 20,
            "hidden_size": 64,
            "depth": 2,
            "num_heads": 4,
            "mlp_ratio": 4.0,
            "max_length": 46.7425,
        },
    }
    return build_model(config)


def _make_batch(batch_size=4):
    rng = np.random.RandomState(0)
    num_atoms = np.arange(2, batch_size + 2, dtype="int64")
    total = int(num_atoms.sum())
    return {
        "structure_array": {
            "num_atoms": paddle.to_tensor(num_atoms),
            "atom_types": paddle.to_tensor(rng.randint(1, 95, size=total)),
            "frac_coords": paddle.to_tensor(rng.rand(total, 3).astype("float32")),
            "lattice": paddle.to_tensor(
                np.tile(np.eye(3, dtype="float32") * 4, (batch_size, 1, 1))
            ),
        }
    }


def test_config_builds_model_with_expected_structure():
    config = OmegaConf.to_container(OmegaConf.load(CONFIG_PATH), resolve=True)
    model = build_model(config["Model"])
    assert type(model).__name__ == "CrystalDiT"
    assert model.num_train_timesteps == 1000
    named_params = dict(model.named_parameters())
    assert "backbone.crystal_embedder.lattice_embedder.weight" in named_params
    assert "backbone.blocks.0.attn.q_proj.weight" in named_params


def test_forward_returns_loss_dict_with_expected_keys():
    model = _make_small_model()
    model.train()
    batch = _make_batch()
    result = model(batch)
    assert "loss_dict" in result
    assert "loss" in result["loss_dict"]
    assert "lattice_loss" in result["loss_dict"]
    assert "atom_loss" in result["loss_dict"]
    loss = result["loss_dict"]["loss"]
    assert loss.ndim == 0
    assert np.isfinite(float(loss))
    assert result["label_dict"]["structure_array"] is batch["structure_array"]
    optimizer = paddle.optimizer.Adam(learning_rate=1e-3, parameters=model.parameters())
    initial = model.backbone.final_layer.atom_linear.weight.numpy().copy()
    loss.backward()
    for parameter in model.parameters():
        if not parameter.stop_gradient:
            assert parameter.grad is not None
            assert np.isfinite(parameter.grad.numpy()).all()
    optimizer.step()
    assert not np.array_equal(
        initial, model.backbone.final_layer.atom_linear.weight.numpy()
    )


def test_forward_loss_is_deterministic_given_fixed_seed():
    model = _make_small_model()
    model.eval()
    losses = []
    for _ in range(2):
        paddle.seed(7)
        np.random.seed(7)
        result = model(_make_batch())
        losses.append(float(result["loss_dict"]["loss"]))
    assert abs(losses[0] - losses[1]) < 1e-6


def _make_real_loader(tmp_path):
    rows = pd.read_csv(REPO_ROOT / "data" / "mp_20" / "val.csv").iloc[:4]
    source = tmp_path / "review.csv"
    rows.to_csv(source, index=False)
    config = OmegaConf.to_container(OmegaConf.load(CONFIG_PATH), resolve=True)
    data_config = config["Dataset"]["val"]
    data_config["dataset"]["__init_params__"].update(
        path=str(source), cache_path=str(tmp_path / "cache")
    )
    data_config["dataset"]["__init_params__"]["build_structure_cfg"]["num_cpus"] = 1
    data_config["sampler"]["__init_params__"]["batch_size"] = 4
    return build_dataloader(data_config)


def test_dataset_and_collate_feed_forward(tmp_path):
    loader = _make_real_loader(tmp_path)
    batch = next(iter(loader))
    structure = batch["structure_array"]
    assert structure["num_atoms"].shape == (4,)
    assert structure["atom_types"].shape[0] == int(structure["num_atoms"].sum())
    assert len(set(structure["num_atoms"].numpy().tolist())) > 1
    result = _make_small_model()(batch)
    assert np.isfinite(float(result["loss_dict"]["loss"]))
    assert len(loader.dataset) == 4


def test_sample_generates_parsable_structure_results(tmp_path):
    model = _make_small_model()
    model.eval()
    batch = {"structure_array": {"num_atoms": paddle.to_tensor([2, 3])}}
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
        assert 0 < num_atoms <= 20
        assert len(item["atom_types"]) == num_atoms
        assert len(item["frac_coords"]) == num_atoms
        lattice = np.array(item["lattice"]).reshape(3, 3)
        assert lattice.shape == (3, 3)
        assert np.all(np.abs(lattice) <= 46.7425 + 1e-3)
        for atomic_number in item["atom_types"]:
            assert 0 < atomic_number <= 94
        structure = BuildStructure.build_one(
            item, "array", niggli=False, canocial=False
        )
        path = tmp_path / f"sample_{len(structure)}.cif"
        CifWriter(structure).write_file(path)
        restored = BuildStructure.build_one(
            str(path), "cif_file", niggli=False, canocial=False
        )
        assert len(restored) == num_atoms


def test_generation_metrics_preserve_global_population(tmp_path, monkeypatch):
    structures = [
        {
            "num_atoms": 1,
            "atom_types": [number],
            "frac_coords": [[0.0, 0.0, 0.0]],
            "lattice": (np.eye(3) * 4).tolist(),
        }
        for number in (14, 13)
    ]
    cifs = [
        str(CifWriter(BuildStructure.build_one(item, "array", niggli=False)))
        for item in structures
    ]
    train = tmp_path / "train.csv"
    test = tmp_path / "test.csv"
    pd.DataFrame({"cif": cifs[:1]}).to_csv(train, index=False)
    pd.DataFrame({"cif": cifs}).to_csv(test, index=False)
    metric = build_metric(
        {
            "__class_name__": "CrystalGenerationMetric",
            "__init_params__": {
                "train_file_path": str(train),
                "test_file_path": str(test),
                "num_cpus": 1,
            },
        }
    )
    invalid = {"atom_types": [], "frac_coords": [], "lattice": np.eye(3).tolist()}
    population = [structures[0], structures[1], structures[1], invalid]
    one_shot = metric(population)
    assert one_shot["structural_validity"] == 0.75
    assert one_shot["chemical_validity"] == 0.75
    assert one_shot["overall_validity"] == 0.75
    assert one_shot["unique_rate"] == 0.5
    assert one_shot["novel_rate"] == 0.5
    assert one_shot["un_rate"] == 0.25
    assert all(np.isfinite(value) for value in one_shot.values())
    reference = metric.reference
    metric.reset()
    metric.update_step(result={"result": population[:2]})
    metric.update_step(result={"result": population[2:]})
    assert metric.compute_epoch() == one_shot
    metric.reset()
    assert metric.reference is reference
    assert metric.total == 0
    invalid_coordinates = dict(structures[0], frac_coords=[[float("nan"), 0.0, 0.0]])
    assert metric([structures[0], invalid_coordinates])["overall_validity"] == 0.5

    downloads = []

    def cached_archive(url, md5):
        downloads.append((url, md5))
        return str(tmp_path.parent)

    monkeypatch.setattr(
        "ppmat.metrics.crystal_generation_metric.download.get_datasets_path_from_url",
        cached_archive,
    )
    downloaded_metric = build_metric(
        {
            "__class_name__": "CrystalGenerationMetric",
            "__init_params__": {
                "train_file_path": str(tmp_path / "missing" / "train.csv"),
                "test_file_path": str(tmp_path / "missing" / "test.csv"),
                "num_cpus": 1,
                "reference_archive_cfg": {
                    "url": "https://example.invalid/mp20.zip",
                    "md5": "reference-digest",
                    "name": tmp_path.name,
                },
            },
        }
    )
    assert downloads == [("https://example.invalid/mp20.zip", "reference-digest")] * 2
    assert downloaded_metric(population) == one_shot


def test_sample_from_registered_weights_runs(tmp_path, monkeypatch):
    original_load = CrystalDiT.set_state_dict
    loaded = []

    def checked_load(model, state, *args, **kwargs):
        result = original_load(model, state, *args, **kwargs)
        assert result == ([], [])
        loaded.append(len(state))
        return result

    monkeypatch.setattr(CrystalDiT, "set_state_dict", checked_load)
    sampler = StructureSampler(
        model_name="crystaldit_mp20",
        config_overrides=[
            "Sample.data.dataset.__init_params__.total_num=2",
            "Sample.data.sampler.__init_params__.batch_size=2",
            "Sample.model_sample_params.num_inference_steps=3",
        ],
    )
    assert type(sampler.model).__name__ == "CrystalDiT"
    assert loaded == [len(sampler.model.state_dict())]
    with pytest.raises(ValueError, match="unconditional"):
        sampler.sample_by_chemical_formula("Li2O")
    paddle.seed(0)
    results = sampler.sample_by_dataloader(save_path=str(tmp_path))
    assert len(results) == 2
    assert len(list(tmp_path.glob("*.cif"))) == 2
    for path in tmp_path.glob("*.cif"):
        structure = BuildStructure.build_one(
            str(path), "cif_file", niggli=False, canocial=False
        )
        assert 0 < len(structure) <= sampler.model.max_atoms

    cli_output = tmp_path / "cli"
    monkeypatch.setattr(
        "sys.argv",
        [
            "sample.py",
            "--model_name=crystaldit_mp20",
            "--mode=by_dataloader",
            f"--output_path={cli_output}",
            "Sample.data.dataset.__init_params__.total_num=2",
            "Sample.model_sample_params.num_inference_steps=3",
        ],
    )
    sample_main()
    assert len(list(cli_output.glob("*.cif"))) == 2

    sampler.model.train()
    batch = next(iter(_make_real_loader(tmp_path)))
    optimizer = paddle.optimizer.SGD(
        learning_rate=1e-4, parameters=sampler.model.parameters()
    )
    loss = sampler.model(batch)["loss_dict"]["loss"]
    assert np.isfinite(float(loss))
    loss.backward()
    for parameter in sampler.model.parameters():
        if not parameter.stop_gradient:
            assert parameter.grad is not None
            assert np.isfinite(parameter.grad.numpy()).all()
    optimizer.step()
    for parameter in sampler.model.parameters():
        assert np.isfinite(parameter.numpy()).all()


def _make_loader(batch_size=4):
    batch = _make_batch(batch_size=8)["structure_array"]
    samples = []
    start = 0
    for i, count in enumerate(batch["num_atoms"].numpy()):
        samples.append(
            {
                "structure_array": {
                    "num_atoms": ConcatData(np.array([count], dtype="int64")),
                    "atom_types": ConcatData(
                        batch["atom_types"][start : start + count].numpy()
                    ),
                    "frac_coords": ConcatData(
                        batch["frac_coords"][start : start + count].numpy()
                    ),
                    "lattice": ConcatData(batch["lattice"][i : i + 1].numpy()),
                }
            }
        )
        start += count

    return paddle.io.DataLoader(
        samples,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        collate_fn=DefaultCollator(),
        return_list=True,
        num_workers=0,
    )


def _make_trainer(model, output_dir, max_epochs, loader):
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
            noise_scheduler_cfg=_make_scheduler_cfg(50),
            max_atoms=20,
            hidden_size=32,
            depth=1,
            num_heads=4,
            execution_backend=backend,
            runtime_options={"cinn": {"full_graph": False}},
        )

    batch = _make_batch(batch_size=2)

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
    """Verify model, optimizer and step restoration before resumed updates."""
    loader = _make_loader()

    with paddle.utils.unique_name.guard():
        first_model = _make_small_model()
        first_trainer, first_optimizer = _make_trainer(
            first_model, tmp_path / "first", 1, loader
        )
    first_trainer.train()

    checkpoint_dir = tmp_path / "first" / "checkpoints"
    for suffix in ("pdparams", "pdopt", "pdstates"):
        assert (checkpoint_dir / f"latest.{suffix}").is_file()
    first_global_step = first_trainer.state.global_step
    saved = paddle.load(str(checkpoint_dir / "latest.pdparams"))
    assert saved.keys() == first_model.state_dict().keys()

    with paddle.utils.unique_name.guard():
        resumed_model = _make_small_model()
        resumed_trainer, resumed_optimizer = _make_trainer(
            resumed_model, tmp_path / "resumed", 2, loader
        )
    saved_optimizer = paddle.load(str(checkpoint_dir / "latest.pdopt"))
    restored_steps = []

    def check_restored_state(layer, inputs):
        if restored_steps:
            return
        for key, value in saved.items():
            np.testing.assert_array_equal(
                layer.state_dict()[key].numpy(), value.numpy()
            )
        optimizer_state = resumed_optimizer.state_dict()
        for key, value in saved_optimizer.items():
            if isinstance(value, paddle.Tensor):
                np.testing.assert_array_equal(
                    optimizer_state[key].numpy(), value.numpy()
                )
        assert resumed_optimizer.get_lr() == first_optimizer.get_lr()
        assert resumed_trainer.state.global_step == first_global_step
        restored_steps.append(first_global_step)

    handle = resumed_model.register_forward_pre_hook(check_restored_state)
    with paddle.utils.unique_name.guard():
        resumed_trainer.train(resume_from_checkpoint=str(checkpoint_dir / "latest"))
    handle.remove()
    assert restored_steps == [first_global_step]

    assert resumed_trainer.state.global_step == first_global_step + len(loader)
    assert resumed_trainer.state.epoch == 2
    assert resumed_model.state_dict().keys() == first_model.state_dict().keys()
    with paddle.utils.unique_name.guard():
        control_model = _make_small_model()
        control_trainer, control_optimizer = _make_trainer(
            control_model, tmp_path / "control", 2, loader
        )
    with paddle.utils.unique_name.guard():
        control_trainer.train()
    assert resumed_trainer.state.global_step == control_trainer.state.global_step
    assert resumed_trainer.state.epoch == control_trainer.state.epoch
    assert resumed_optimizer.get_lr() == control_optimizer.get_lr()
    assert (
        resumed_optimizer.state_dict().keys() == control_optimizer.state_dict().keys()
    )


@pytest.mark.skipif(
    os.environ.get("PPMAT_RUN_CINN_WORKFLOW_TESTS") != "1",
    reason="real CINN compilation is gated behind PPMAT_RUN_CINN_WORKFLOW_TESTS=1 "
    "and requires a CUDA Paddle build",
)
def test_gpu_cinn_matches_eager(tmp_path):
    """Compare compiled training, registered sampling and runtime cache reuse."""
    if not paddle.is_compiled_with_cuda():
        pytest.skip("Paddle was not compiled with CUDA.")
    if not paddle.base.is_compiled_with_cinn():
        pytest.skip("Paddle was not compiled with CINN.")
    paddle.set_device("gpu:0")

    loader = _make_loader()
    final_state_dicts = {}
    for backend in ("eager", "cinn"):
        paddle.seed(5)
        np.random.seed(5)
        model = CrystalDiT(
            noise_scheduler_cfg=_make_scheduler_cfg(50),
            max_atoms=20,
            hidden_size=64,
            depth=2,
            num_heads=4,
            execution_backend=backend,
            runtime_options={"cinn": {"full_graph": False}},
        )
        trainer, _ = _make_trainer(model, tmp_path / backend, 1, loader)
        with paddle.utils.unique_name.guard():
            trainer.train()
        final_state_dicts[backend] = model.state_dict()
        if backend == "cinn":
            assert {key[1] for key in model._runtime_cache} == {"train", "eval"}

    eager_state = final_state_dicts["eager"]
    cinn_state = final_state_dicts["cinn"]
    assert eager_state.keys() == cinn_state.keys()
    for key in eager_state:
        np.testing.assert_allclose(
            eager_state[key].numpy(), cinn_state[key].numpy(), atol=2e-5, rtol=2e-5
        )

    sampled = {}
    for backend in ("eager", "cinn"):
        sampler = StructureSampler(
            model_name="crystaldit_mp20",
            config_overrides=[
                f"Execution.backend={backend}",
                "Execution.__init_params__.full_graph=False",
                "Sample.data.dataset.__init_params__.total_num=2",
                "Sample.model_sample_params.num_inference_steps=3",
            ],
        )
        paddle.seed(37)
        np.random.seed(37)
        destination = tmp_path / f"sample_{backend}"
        sampled[backend] = sampler.sample_by_dataloader(save_path=str(destination))
        assert len(list(destination.glob("*.cif"))) == 2
        if backend == "cinn":
            cached = dict(sampler.model._runtime_cache)
            assert cached
            paddle.seed(37)
            sampler.sample({"structure_array": {"num_atoms": paddle.to_tensor([2, 3])}})
            assert sampler.model._runtime_cache == cached

    for eager, compiled in zip(sampled["eager"], sampled["cinn"]):
        assert eager["atom_types"] == compiled["atom_types"]
        for field in ("lattice", "frac_coords"):
            np.testing.assert_allclose(
                eager[field], compiled[field], atol=1e-4, rtol=1e-4
            )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))

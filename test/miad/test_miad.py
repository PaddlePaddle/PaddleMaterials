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

import copy
import os
import tempfile
from pathlib import Path
from unittest import mock

import numpy as np
import paddle
import pytest
from cinn_workflow_harness import REPO_ROOT
from omegaconf import OmegaConf

from ppmat.datasets import NumAtomsCrystalDataset
from ppmat.datasets import build_dataloader
from ppmat.datasets.collate_fn import DefaultCollator
from ppmat.datasets.custom_data_type import ConcatData
from ppmat.datasets.mp20_dataset import MP20Dataset
from ppmat.models import build_model
from ppmat.models.miad.miad import MiAD
from ppmat.models.miad.miad import _extract_x0
from ppmat.sampler import StructureSampler
from ppmat.schedulers import DDPMScheduler
from ppmat.trainer.base_trainer import BaseTrainer

_MP20_TEST_CSV = REPO_ROOT / "data" / "mp_20" / "test.csv"
_MIAD_YAML = REPO_ROOT / "structure_generation" / "configs" / "miad" / "miad_mp20.yaml"


def setup_module():
    paddle.seed(42)


def _load_miad_yaml(num_steps=None):
    config = OmegaConf.load(_MIAD_YAML)
    if num_steps is not None:
        config.Global.num_train_timesteps = num_steps
    return OmegaConf.to_container(config, resolve=True)


def _make_model_from_yaml(num_steps=None):
    return build_model(_load_miad_yaml(num_steps)["Model"])


TINY_MODEL_CFG = {
    "hidden_dim": 64,
    "latent_dim": 32,
    "num_layers": 2,
    "num_classes": 100,
    "mirage_num_atoms": 8,
    "act_fn": "silu",
    "dis_emb": "sin",
    "num_freqs": 10,
    "edge_style": "fc",
    "ln": False,
    "ip": True,
    "smooth": True,
    "pred_type": True,
}

TINY_DIFFUSION_CFG = {
    "cont_time": False,
    "num_train_timesteps": 10,
    "time_dim": 32,
    "eps": 1e-3,
    "lattice_noise_scheduler_cfg": {
        "__class_name__": "CosineDDPMScheduler",
        "__init_params__": {
            "num_train_timesteps": 10,
            "clip_sample": False,
        },
    },
    "coord_noise_scheduler_cfg": {
        "__class_name__": "WrappedVESDEPCScheduler",
        "__init_params__": {
            "num_train_timesteps": 10,
            "step_lr": 1e-5,
            "sigma_min": 0.005,
            "sigma_max": 0.5,
            "sampling_eps": 0.001,
        },
    },
    "atom_noise_scheduler_cfg": {
        "__class_name__": "D3PMUniformScheduler",
        "__init_params__": {
            "loss_scale": 1000,
            "num_train_timesteps": 10,
            "num_types": 100,
        },
    },
}


def _make_synthetic_batch(batch_size=2, atoms_per_crystal=5):
    num_atoms = paddle.full([batch_size], atoms_per_crystal, dtype="int64")
    total_atoms = batch_size * atoms_per_crystal
    lattices = paddle.randn([batch_size, 3, 3], dtype="float32")
    frac_coords = paddle.rand([total_atoms, 3], dtype="float32")
    atom_types = paddle.randint(1, 10, [total_atoms], dtype="int64")
    return {
        "structure_array": {
            "num_atoms": num_atoms.reshape([batch_size, 1]),
            "frac_coords": frac_coords,
            "atom_types": atom_types,
            "lattice": lattices,
        },
    }


_SAMPLE_NUM_ATOMS = (5, 7)


def _make_tiny_model(execution_backend="eager"):
    # Isolated name scope: identical parameter names across instances.
    with paddle.utils.unique_name.guard():
        return MiAD(
            **TINY_MODEL_CFG,
            **TINY_DIFFUSION_CFG,
            execution_backend=execution_backend,
        )


def _clone_state_dict(model):
    return {key: value.clone() for key, value in model.state_dict().items()}


def _record_cinn_boundaries(names):
    def run_boundary(self, name, function, *args, **kwargs):
        names.append(name)
        return function(*args, **kwargs)

    return run_boundary


def _make_structure_sample(num_atoms, seed):
    rng = np.random.RandomState(seed)
    return {
        "structure_array": {
            "frac_coords": ConcatData(rng.rand(num_atoms, 3).astype("float32")),
            "atom_types": ConcatData(
                rng.randint(1, 10, size=num_atoms).astype("int64")
            ),
            "lattice": ConcatData(rng.rand(1, 3, 3).astype("float32")),
            "num_atoms": ConcatData(np.array([num_atoms], dtype="int64")),
        }
    }


def _make_structure_loader():
    # Mixed atom counts exercise collate and mirage padding.
    samples = [
        _make_structure_sample(4, seed=1),
        _make_structure_sample(3, seed=2),
    ]
    return paddle.io.DataLoader(
        samples,
        batch_size=2,
        shuffle=False,
        collate_fn=DefaultCollator(),
        return_list=True,
    )


def _trainer_config(output_dir, max_epochs):
    return {
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
        "best_metric_indicator": "eval_loss",
        "name_for_best_metric": "loss",
        "greater_is_better": False,
    }


def _build_trainer(model, output_dir, max_epochs, backend="cinn"):
    scheduler = paddle.optimizer.lr.ExponentialDecay(learning_rate=1e-3, gamma=0.9)
    scheduler.by_epoch = False
    with paddle.utils.unique_name.guard():
        optimizer = paddle.optimizer.Adam(
            learning_rate=scheduler, parameters=model.parameters()
        )
    trainer = BaseTrainer(
        _trainer_config(output_dir, max_epochs),
        model,
        train_dataloader=_make_structure_loader(),
        val_dataloader=_make_structure_loader(),
        optimizer=optimizer,
        lr_scheduler=scheduler,
        execution_config={"backend": backend},
    )
    return trainer, optimizer


def _sample_small_batch(model, num_inference_steps=None):
    return model.sample(
        {
            "structure_array": {
                "num_atoms": paddle.to_tensor(_SAMPLE_NUM_ATOMS, dtype="int64").reshape(
                    [-1, 1]
                )
            }
        },
        num_inference_steps=num_inference_steps,
    )["result"]


class TestMiADSmoke:
    """End-to-end model flows: training forward and sampling."""

    def test_forward_smoke(self):
        model = _make_tiny_model()
        model.train()
        bias = model.decoder.csp_layer_0.edge_mlp[0].bias
        bias.set_value(paddle.full_like(bias, -100.0))
        bias = model.decoder.csp_layer_0.node_mlp[2].bias
        bias.set_value(paddle.full_like(bias, -100.0))
        train_num_atoms = paddle.to_tensor([5, 3], dtype="int64")
        total_atoms = int(train_num_atoms.sum())
        batch = {
            "target": paddle.to_tensor([1.0, 2.0]),
            "structure_array": {
                "num_atoms": train_num_atoms,
                "frac_coords": paddle.rand([total_atoms, 3], dtype="float32"),
                "atom_types": paddle.randint(1, 10, [total_atoms], dtype="int64"),
                "lattice": paddle.randn([2, 3, 3], dtype="float32"),
            },
        }
        output = model(batch)
        assert output["label_dict"]["target"] is batch["target"]
        loss = output["loss_dict"]["loss"]
        assert paddle.isfinite(loss)
        loss.backward()
        optimizer = paddle.optimizer.Adam(parameters=model.parameters())
        for name, parameter in model.named_parameters():
            if not parameter.stop_gradient:
                assert parameter.grad is not None, name
                assert paddle.isfinite(parameter.grad).all(), name
        optimizer.step()
        for name, parameter in model.named_parameters():
            assert paddle.isfinite(parameter).all(), name

    def test_sample_output_format(self):
        model = _make_tiny_model()
        model.eval()
        for steps in (0, -1, 5, 11, 10.0, True):
            with pytest.raises(ValueError, match="trained diffusion schedules"):
                _sample_small_batch(model, num_inference_steps=steps)
            assert (
                model.num_train_timesteps == TINY_DIFFUSION_CFG["num_train_timesteps"]
            )
        paddle.seed(123)
        result = _sample_small_batch(model)
        paddle.seed(123)
        explicit = _sample_small_batch(
            model, num_inference_steps=model.num_train_timesteps
        )
        for default_entry, explicit_entry in zip(result, explicit):
            for key in ("atom_types", "frac_coords", "lattice"):
                np.testing.assert_array_equal(default_entry[key], explicit_entry[key])
        assert len(result) == 2
        for entry in result:
            for key in ("num_atoms", "atom_types", "frac_coords", "lattice"):
                assert key in entry
            assert len(entry["frac_coords"][0]) == 3
            assert entry["num_atoms"] == len(entry["atom_types"])
            assert (np.asarray(entry["atom_types"]) != 0).all()


class TestMiADConfig:
    """End-to-end flow driven by the released YAML config."""

    def test_forward_and_sample_from_yaml(self):
        for key, value in (
            ("smooth", False),
            ("pred_type", False),
            ("edge_style", "knn"),
            ("num_layers", 0),
            ("num_classes", 99),
        ):
            config = _load_miad_yaml(
                num_steps=TINY_DIFFUSION_CFG["num_train_timesteps"]
            )
            config["Model"]["__init_params__"][key] = value
            with pytest.raises(ValueError):
                build_model(config["Model"])
        config = _load_miad_yaml(num_steps=TINY_DIFFUSION_CFG["num_train_timesteps"])
        config["Model"]["__init_params__"]["num_train_timesteps"] += 1
        with pytest.raises(ValueError, match="must match num_train_timesteps"):
            build_model(config["Model"])
        paddle.seed(2026)
        model = _make_model_from_yaml(
            num_steps=TINY_DIFFUSION_CFG["num_train_timesteps"]
        )
        assert isinstance(model.lattice_scheduler, DDPMScheduler)

        lattice = paddle.full([1, 3, 3], 4.0)
        recovered = model.lattice_scheduler.step(
            paddle.zeros_like(lattice), 0, lattice
        ).prev_sample
        expected = lattice / paddle.sqrt(model.lattice_scheduler.alphas[0])
        np.testing.assert_allclose(recovered.numpy(), expected.numpy(), atol=1e-5)
        coords = paddle.rand([7, 3])
        recovered_coords = model.coord_scheduler.step_pred_indexed(
            paddle.ones_like(coords), paddle.zeros([7], dtype="int64"), coords
        ).prev_sample
        np.testing.assert_array_equal(recovered_coords.numpy(), coords.numpy())
        model.eval()
        synthetic_batch = _make_synthetic_batch()
        paddle.seed(2026)
        with paddle.no_grad():
            output = model(copy.deepcopy(synthetic_batch))
        assert paddle.isfinite(output["loss_dict"]["loss"])

        source = DefaultCollator()(
            [_make_structure_sample(4, seed=1), _make_structure_sample(3, seed=2)]
        )["structure_array"]
        original_counts = source["num_atoms"].flatten()
        for capacity in (None, model.mirage_num_atoms):
            expected_counts = [
                int(count) if capacity is None else max(int(count), capacity)
                for count in original_counts
            ]
            state = _extract_x0(
                {"structure_array": dict(source)}, mirage_num_atoms=capacity
            )
            with paddle.no_grad():
                output = model({"structure_array": dict(source)})
            assert paddle.isfinite(output["loss_dict"]["loss"])
            np.testing.assert_array_equal(state["num_atoms"].numpy(), expected_counts)
            np.testing.assert_array_equal(
                state["lattice"].numpy(), source["lattice"].reshape(2, 3, 3)
            )
            if capacity is None:
                np.testing.assert_array_equal(
                    state["frac_coords"].numpy(), source["frac_coords"]
                )
                np.testing.assert_array_equal(
                    state["atom_types"].numpy(), source["atom_types"]
                )

        sample_config = _load_miad_yaml(num_steps=model.num_train_timesteps)["Sample"][
            "data"
        ]
        sample_config["dataset"]["__init_params__"]["total_num"] = 2
        sample_loader = build_dataloader(sample_config)
        assert isinstance(sample_loader.dataset, NumAtomsCrystalDataset)
        sample_batch = next(iter(sample_loader))
        paddle.seed(123)
        result = model.sample(
            copy.deepcopy(sample_batch),
            num_inference_steps=model.num_train_timesteps,
        )["result"]
        assert len(result) == 2
        for entry in result:
            assert entry["num_atoms"] == len(entry["atom_types"])
            assert (np.asarray(entry["atom_types"]) != 0).all()
            assert len(entry["frac_coords"][0]) == 3
            assert np.asarray(entry["lattice"]).shape == (3, 3)


@pytest.mark.skipif(
    not _MP20_TEST_CSV.exists(),
    reason=f"MP-20 test data not found at {_MP20_TEST_CSV}; skipping dataset tests",
)
class TestMiADDataset:
    """Real-data pipeline: dataset -> collate -> model forward."""

    def test_collate_to_forward(self):
        dataset = MP20Dataset(
            path=_MP20_TEST_CSV,
            build_structure_cfg={"format": "cif_str", "num_cpus": 1},
        )
        assert len(dataset) > 0
        collator = DefaultCollator()
        samples = [dataset[i] for i in range(min(4, len(dataset)))]
        batch = collator(samples)
        model = _make_tiny_model()
        model.eval()
        with paddle.no_grad():
            output = model(batch)
        assert paddle.isfinite(output["loss_dict"]["loss"])


class TestMiADFrameworkEntry:
    """StructureSampler and metric entry points driven by config files."""

    def test_sampler_and_metric_entry(self, tmp_path):
        import pandas as pd
        from pymatgen.core import Lattice
        from pymatgen.core import Structure

        reference = Structure(Lattice.cubic(4), ["Na", "Cl"], [[0, 0, 0], [0.5] * 3])
        reference_path = tmp_path / "reference.csv"
        pd.DataFrame({"cif": [reference.to(fmt="cif")]}).to_csv(
            reference_path, index=False
        )

        config = _load_miad_yaml(num_steps=TINY_DIFFUSION_CFG["num_train_timesteps"])
        config["Global"]["time_embed_dim"] = TINY_DIFFUSION_CFG["time_dim"]
        config["Model"]["__init_params__"].update(TINY_MODEL_CFG)
        diffusion_params = copy.deepcopy(TINY_DIFFUSION_CFG)
        # Untrained tiny models are numerically unstable at ten steps; clipping
        # keeps this entry-point test focused on the sampler contract.
        diffusion_params["lattice_noise_scheduler_cfg"]["__init_params__"][
            "clip_sample"
        ] = True
        config["Model"]["__init_params__"].update(diffusion_params)
        config["Sample"]["data"]["dataset"]["__init_params__"]["total_num"] = 2
        config["Sample"]["metrics"]["__init_params__"].update(
            {
                "reference_file_path": str(reference_path),
                "cache_dir": str(tmp_path),
            }
        )
        config_path = tmp_path / "miad.yaml"
        OmegaConf.save(OmegaConf.create(config), config_path)
        checkpoint_path = tmp_path / "best.pdparams"
        paddle.seed(2026)
        paddle.save(build_model(config["Model"]).state_dict(), str(checkpoint_path))

        sampler = StructureSampler(
            config_path=str(config_path), checkpoint_path=str(checkpoint_path)
        )
        sample_dir = tmp_path / "num_atoms_samples"
        paddle.seed(2026)
        sampled = sampler.sample_by_num_atoms(num_atoms=5, save_path=str(sample_dir))
        assert len(sampled) == 1
        assert len(list(sample_dir.glob("*.cif"))) == 1

        metric_dir = tmp_path / "metric_samples"
        paddle.seed(2026)
        metrics = sampler.compute_metric(save_path=str(metric_dir))
        assert metrics["total"] == 2
        for key in ("uniqueness_rate", "novelty_rate"):
            assert 0.0 <= metrics[key] <= 100.0
        assert "stability_rate" not in metrics
        assert "sun_rate" not in metrics


class TestMiADStateDict:
    """Native Paddle layout load through the default Layer API."""

    def test_native_layout_weight_load(self):
        model = _make_model_from_yaml()
        state = {key: value.clone() for key, value in model.state_dict().items()}
        assert not any(("prop_mlp" in key for key in state))
        model2 = _make_model_from_yaml()
        missing, unexpected = model2.set_state_dict(state)
        assert len(missing) == 0
        assert len(unexpected) == 0
        params1 = list(model.named_parameters())
        params2 = list(model2.named_parameters())
        assert len(params1) == len(params2)
        for (name, param1), (_, param2) in zip(params1, params2):
            np.testing.assert_array_equal(
                param1.numpy(), param2.numpy(), err_msg=f"parameter {name} differs"
            )
        model2.eval()
        with paddle.no_grad():
            output = model2(_make_synthetic_batch())
        assert paddle.isfinite(output["loss_dict"]["loss"])


class TestMiADSUNMetric:
    """S.U.N. evaluation driven by the YAML metric config."""

    def test_sun_metric_from_yaml(self):
        import pandas as pd
        from pymatgen.core import Lattice
        from pymatgen.core import Structure

        from ppmat.metrics import build_metric

        reference = Structure(Lattice.cubic(4), ["Na", "Cl"], [[0, 0, 0], [0.5] * 3])
        novel = Structure(Lattice.cubic(4), ["Li", "F"], [[0, 0, 0], [0.5] * 3])
        generated = [novel.to(fmt="cif"), reference.to(fmt="cif")]
        generated += [generated[0], "invalid cif"]
        energies = [None, -0.1, -0.1, -0.1]
        with tempfile.TemporaryDirectory() as tmp_dir:
            reference_path = Path(tmp_dir) / "reference.csv"
            pd.DataFrame({"cif": [reference.to(fmt="cif")]}).to_csv(
                reference_path, index=False
            )
            config = _load_miad_yaml()["Sample"]["metrics"]
            config["__init_params__"].update(
                reference_file_path=str(reference_path), cache_dir=tmp_dir
            )
            metrics_fn = build_metric(config)
            results = metrics_fn(generated, energy_above_hull=energies)
            assert results == {
                "total": 4,
                "valid": 3,
                "non_trivial": 3,
                "stability_rate": 75.0,
                "uniqueness_rate": 50.0,
                "novelty_rate": 50.0,
                "sun_rate": 0.0,
                "sun_count": 0,
                "metastable_rate": 75.0,
                "msun_rate": 0.0,
                "msun_count": 0,
            }
            metrics_fn.reset()
            metrics_fn.update_step(
                result={"result": generated[:1]}, batch=None, stage="sample"
            )
            metrics_fn.update_step(
                result={"result": generated[1:], "energy_above_hull": energies[1:]},
                batch=None,
                stage="sample",
            )
            assert metrics_fn.compute_epoch(stage="sample") == results
            with pytest.raises(ValueError, match="one entry per structure"):
                metrics_fn.update_step(
                    result={"result": generated, "energy_above_hull": [-0.1]},
                    batch=None,
                    stage="sample",
                )
            assert metrics_fn.compute_epoch(stage="sample") == results
            model = _make_tiny_model()
            model.eval()
            sampled_results = metrics_fn(_sample_small_batch(model))
            assert sampled_results["total"] == len(_SAMPLE_NUM_ATOMS)
            for key in ("uniqueness_rate", "novelty_rate"):
                assert np.isfinite(sampled_results[key])
            assert "stability_rate" not in sampled_results
            assert "sun_rate" not in sampled_results

    def test_novelty_reference_lifecycle(self):
        import pandas as pd
        from pymatgen.core import Lattice
        from pymatgen.core import Structure

        from ppmat.metrics import build_metric

        structure = Structure(Lattice.cubic(4), ["Na", "Cl"], [[0, 0, 0], [0.5] * 3])
        generated = [structure.to(fmt="cif")]
        with tempfile.TemporaryDirectory() as tmp_dir:
            reference_path = Path(tmp_dir) / "reference.csv"
            cache_dir = Path(tmp_dir) / "cache"
            config = _load_miad_yaml()["Sample"]["metrics"]
            config["__init_params__"].update(
                reference_file_path=str(reference_path), cache_dir=str(cache_dir)
            )
            metric = build_metric(config)
            with pytest.raises(FileNotFoundError):
                metric(generated)
            for content in ("", "formula\nNaCl\n", "cif\ninvalid\n"):
                reference_path.write_text(content)
                with pytest.raises(ValueError):
                    metric(generated)

            pd.DataFrame({"cif": generated}).to_csv(reference_path, index=False)
            assert metric(generated)["novelty_rate"] == 0.0
            with mock.patch(
                "pandas.read_csv", side_effect=AssertionError("cache miss")
            ):
                assert build_metric(config)(generated)["novelty_rate"] == 0.0
            changed = Structure(Lattice.cubic(4), ["Li", "F"], [[0, 0, 0], [0.5] * 3])
            pd.DataFrame({"cif": [changed.to(fmt="cif")]}).to_csv(
                reference_path, index=False
            )
            assert metric(generated)["novelty_rate"] == 100.0
            assert build_metric(config)(generated)["novelty_rate"] == 100.0
            assert len(list(cache_dir.glob("*.pkl"))) == 2
            for cache_path in cache_dir.glob("*.pkl"):
                cache_path.write_bytes(b"incomplete cache")
            assert build_metric(config)(generated)["novelty_rate"] == 100.0

            pd.DataFrame({"cif": generated}).to_csv(reference_path, index=False)
            with mock.patch("ppmat.metrics.sun_metric.os.replace", side_effect=OSError):
                assert build_metric(config)(generated)["novelty_rate"] == 0.0
            assert not list(cache_dir.glob("*.tmp"))
            reference_path.unlink()
            with pytest.raises(FileNotFoundError):
                metric(generated)


def _make_dispatch_pair():
    # Same-seed construction: sigma_norm() is a Monte-Carlo constant outside
    # state_dict, and bitwise dispatch comparison needs identical instances.
    initial_state = _clone_state_dict(_make_tiny_model())
    paddle.seed(2026)
    eager_model = _make_tiny_model()
    paddle.seed(2026)
    cinn_model = _make_tiny_model(execution_backend="cinn")
    eager_model.set_state_dict(initial_state)
    cinn_model.set_state_dict(initial_state)
    eager_model.eval()
    cinn_model.eval()
    return eager_model, cinn_model


class TestMiADCinnDispatch:
    """Boundary dispatch must not move eager numbers or the state layout."""

    def test_forward_dispatch_matches_eager(self):
        eager_model, cinn_model = _make_dispatch_pair()

        paddle.seed(7)
        eager_batch = _make_synthetic_batch()
        paddle.seed(7)
        cinn_batch = _make_synthetic_batch()

        paddle.seed(42)
        with paddle.no_grad():
            eager_loss = eager_model(eager_batch)["loss_dict"]["loss"]

        names = []
        paddle.seed(42)
        with mock.patch.object(MiAD, "_run_runtime", _record_cinn_boundaries(names)):
            with paddle.no_grad():
                cinn_loss = cinn_model(cinn_batch)["loss_dict"]["loss"]

        assert names == ["denoise_step"]
        np.testing.assert_allclose(
            cinn_loss.numpy(), eager_loss.numpy(), atol=0, rtol=0
        )
        assert cinn_model.state_dict().keys() == eager_model.state_dict().keys()

    def test_sampling_dispatch_matches_eager(self):
        eager_model, cinn_model = _make_dispatch_pair()

        paddle.seed(123)
        eager_result = _sample_small_batch(eager_model)

        names = []
        paddle.seed(123)
        with mock.patch.object(MiAD, "_run_runtime", _record_cinn_boundaries(names)):
            cinn_result = _sample_small_batch(cinn_model)

        assert names
        assert set(names) == {"denoise_step"}
        assert len(cinn_result) == len(eager_result)
        for eager_entry, cinn_entry in zip(eager_result, cinn_result):
            assert eager_entry["num_atoms"] == cinn_entry["num_atoms"]
            for key in ("atom_types", "frac_coords", "lattice"):
                np.testing.assert_allclose(
                    cinn_entry[key], eager_entry[key], atol=0, rtol=0
                )


class TestMiADCinnTrainer:
    """Trainer orchestration with the CINN backend selected end to end."""

    @pytest.fixture(autouse=True)
    def runtime_dispatch(self):
        # Orchestration test only: runtime hooks are mocked, no GPU compiler.
        validate_patcher = mock.patch.object(
            MiAD, "validate_execution_backend", lambda self, **kwargs: None
        )
        self.boundaries = []
        runtime_patcher = mock.patch.object(
            MiAD,
            "_run_runtime",
            _record_cinn_boundaries(self.boundaries),
        )
        with validate_patcher, runtime_patcher:
            yield

    def test_checkpoint_resume_workflow(self):
        initial_state = _clone_state_dict(_make_tiny_model())
        paddle.seed(2026)
        first_model = _make_tiny_model(execution_backend="cinn")
        first_model.set_state_dict(initial_state)
        with tempfile.TemporaryDirectory() as tmp_dir:
            first_trainer, _ = _build_trainer(first_model, Path(tmp_dir) / "first", 1)
            with paddle.utils.unique_name.guard():
                first_trainer.train()

            checkpoint_dir = Path(tmp_dir) / "first" / "checkpoints"
            for prefix in ("epoch_1", "latest", "best"):
                for suffix in ("pdparams", "pdopt", "pdstates"):
                    assert (checkpoint_dir / f"{prefix}.{suffix}").is_file()
            saved_state = paddle.load(str(checkpoint_dir / "latest.pdparams"))
            saved_optimizer = paddle.load(str(checkpoint_dir / "latest.pdopt"))
            assert saved_state.keys() == first_model.state_dict().keys()

            paddle.seed(2026)
            resumed_model = _make_tiny_model(execution_backend="cinn")
            resumed_trainer, resumed_optimizer = _build_trainer(
                resumed_model, Path(tmp_dir) / "resumed", 2
            )
            original_train_epoch = resumed_trainer.train_epoch

            def check_restored_state(*args, **kwargs):
                assert resumed_trainer.state.global_step == 1
                for key, value in resumed_model.state_dict().items():
                    np.testing.assert_array_equal(value.numpy(), saved_state[key])
                restored_optimizer = resumed_optimizer.state_dict()
                assert restored_optimizer.keys() == saved_optimizer.keys()
                for key, value in restored_optimizer.items():
                    if isinstance(value, dict):
                        assert value == saved_optimizer[key]
                    else:
                        np.testing.assert_array_equal(
                            value.numpy(), saved_optimizer[key]
                        )
                assert resumed_optimizer.get_lr() == 0.001 * 0.9
                return original_train_epoch(*args, **kwargs)

            with mock.patch.object(
                resumed_trainer, "train_epoch", side_effect=check_restored_state
            ) as resumed_epoch:
                with paddle.utils.unique_name.guard():
                    resumed_trainer.train(
                        resume_from_checkpoint=str(checkpoint_dir / "latest")
                    )
                resumed_epoch.assert_called_once()

            paddle.seed(2026)
            straight_model = _make_tiny_model(execution_backend="cinn")
            straight_model.set_state_dict(initial_state)
            straight_trainer, straight_optimizer = _build_trainer(
                straight_model, Path(tmp_dir) / "straight", 2
            )
            with paddle.utils.unique_name.guard():
                straight_trainer.train()

        assert first_trainer.state.global_step == 1
        assert resumed_trainer.state.global_step == 2
        assert resumed_trainer.state.epoch == 2
        assert resumed_optimizer.get_lr() == straight_optimizer.get_lr()
        assert (
            resumed_trainer.lr_scheduler.state_dict()
            == straight_trainer.lr_scheduler.state_dict()
        )
        assert (
            resumed_optimizer.state_dict().keys()
            == straight_optimizer.state_dict().keys()
        )
        assert self.boundaries
        assert set(self.boundaries) == {"denoise_step"}
        for model in (resumed_model, straight_model):
            assert model.state_dict().keys() == first_model.state_dict().keys()
            for value in model.state_dict().values():
                assert paddle.isfinite(value).all()
        # Diffusion training is stochastic and BaseTrainer does not restore the
        # RNG state on resume, so weight parity is checked structurally above.


@pytest.mark.skipif(
    os.environ.get("PPMAT_RUN_CINN_WORKFLOW_TESTS") != "1",
    reason="Set PPMAT_RUN_CINN_WORKFLOW_TESTS=1 to run the GPU CINN compile test",
)
class TestMiADGpuCinn:
    """Real CINN compilation on GPU: compiled run must match eager."""

    def test_gpu_cinn_matches_eager(self):
        if (
            not paddle.is_compiled_with_cuda()
            or not paddle.base.is_compiled_with_cinn()
        ):
            pytest.skip("Paddle must be compiled with CUDA and CINN")
        paddle.set_device("gpu:0")

        initial_state = _clone_state_dict(_make_tiny_model())
        with tempfile.TemporaryDirectory() as tmp_dir:
            paddle.seed(2026)
            eager_model = _make_tiny_model()
            eager_model.set_state_dict(initial_state)
            paddle.seed(2026)
            cinn_model = _make_tiny_model(execution_backend="cinn")
            cinn_model.set_state_dict(initial_state)

            # Identical seed streams isolate the compiled-numerics delta.
            paddle.seed(2026)
            np.random.seed(2026)
            eager_trainer, _ = _build_trainer(
                eager_model, Path(tmp_dir) / "eager", 1, backend="eager"
            )
            with paddle.utils.unique_name.guard():
                eager_trainer.train()

            paddle.seed(2026)
            np.random.seed(2026)
            cinn_trainer, _ = _build_trainer(cinn_model, Path(tmp_dir) / "cinn", 1)
            with paddle.utils.unique_name.guard():
                cinn_trainer.train()

            assert eager_trainer.state.global_step == cinn_trainer.state.global_step
            for key in initial_state:
                np.testing.assert_allclose(
                    cinn_model.state_dict()[key].numpy(),
                    eager_model.state_dict()[key].numpy(),
                    atol=2e-5,
                    rtol=2e-5,
                )

            # Same weights, same seed: compiled eval forward stays close.
            cinn_model.set_state_dict(eager_model.state_dict())
            cinn_model.eval()
            eager_model.eval()
            paddle.seed(7)
            eager_batch = _make_synthetic_batch()
            paddle.seed(7)
            cinn_batch = _make_synthetic_batch()
            with paddle.no_grad():
                paddle.seed(42)
                eager_loss = eager_model(eager_batch)["loss_dict"]["loss"]
                paddle.seed(42)
                cinn_loss = cinn_model(cinn_batch)["loss_dict"]["loss"]
            np.testing.assert_allclose(
                cinn_loss.numpy(), eager_loss.numpy(), atol=2e-5, rtol=2e-5
            )

            result = _sample_small_batch(cinn_model)
            assert len(result) == 2
            for entry in result:
                assert (np.asarray(entry["atom_types"]) != 0).all()

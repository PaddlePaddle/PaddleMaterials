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

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import paddle
from cinn_workflow_harness import REPO_ROOT
from omegaconf import OmegaConf

from ppmat.datasets.collate_fn import DefaultCollator
from ppmat.datasets.custom_data_type import ConcatData
from ppmat.datasets.mp20_dataset import MP20Dataset
from ppmat.models import build_model
from ppmat.models.miad.miad import MiAD
from ppmat.trainer.base_trainer import BaseTrainer

_MP20_TEST_CSV = REPO_ROOT / "data" / "mp_20" / "test.csv"
_MIAD_YAML = REPO_ROOT / "structure_generation" / "configs" / "miad" / "miad_mp20.yaml"


def setUpModule():
    paddle.seed(42)


def _load_miad_yaml():
    config = OmegaConf.load(_MIAD_YAML)
    return OmegaConf.to_container(config, resolve=True)


def _make_model_from_yaml():
    return build_model(_load_miad_yaml()["Model"])


TINY_MODEL_CFG = {
    "hidden_dim": 64,
    "latent_dim": 32,
    "num_layers": 2,
    "max_atoms": 100,
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
    "num_steps": 10,
    "time_embed_dim": 32,
    "eps": 1e-3,
    "lat_diffusion": {
        "scheduler_cfg": {
            "__class_name__": "DDPMScheduler",
            "__init_params__": {
                "num_train_timesteps": 10,
                "beta_schedule": "clipped_cosine",
            },
        },
    },
    "frac_diffusion": {
        "step_lr": 1e-5,
        "scheduler_cfg": {
            "__class_name__": "ScoreSdeVeSchedulerWrapped",
            "__init_params__": {
                "num_train_timesteps": 10,
                "sigma_min": 0.005,
                "sigma_max": 0.5,
                "sampling_eps": 0.001,
            },
        },
    },
    "type_diffusion": {
        "__class_name__": "D3PMUniformDiffusion",
        "__init_params__": {
            "loss_scale": 1000,
            "scheduler_cfg": {
                "__class_name__": "D3PMUniformScheduler",
                "__init_params__": {
                    "num_train_timesteps": 10,
                    "num_types": 100,
                },
            },
        },
    },
}


def _make_synthetic_batch(batch_size=2, atoms_per_crystal=5):
    num_atoms = paddle.full([batch_size], atoms_per_crystal, dtype="int64")
    total_atoms = batch_size * atoms_per_crystal
    batch_idx = paddle.concat(
        [paddle.full([atoms_per_crystal], i, dtype="int64") for i in range(batch_size)]
    )
    lattices = paddle.randn([batch_size, 3, 3], dtype="float32")
    frac_coords = paddle.rand([total_atoms, 3], dtype="float32")
    atom_types = paddle.randint(1, 10, [total_atoms], dtype="int64")
    return {
        "x0": [lattices, frac_coords, atom_types],
        "batch_size": batch_size,
        "num_atoms": num_atoms,
        "batch_idx": batch_idx,
        "atom_types": atom_types,
    }


_SAMPLE_NUM_ATOMS = (5, 7)
_SAMPLE_INFERENCE_STEPS = 5


def _make_tiny_model(execution_backend="eager"):
    # Isolated name scope: identical parameter names across instances.
    with paddle.utils.unique_name.guard():
        return MiAD(
            model_cfg=TINY_MODEL_CFG,
            diffusion_cfg=TINY_DIFFUSION_CFG,
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
    optimizer = paddle.optimizer.Adam(learning_rate=1e-3, parameters=model.parameters())
    trainer = BaseTrainer(
        _trainer_config(output_dir, max_epochs),
        model,
        train_dataloader=_make_structure_loader(),
        val_dataloader=_make_structure_loader(),
        optimizer=optimizer,
        execution_config={"backend": backend},
    )
    return trainer, optimizer


def _sample_small_batch(model, num_inference_steps=_SAMPLE_INFERENCE_STEPS):
    return model.sample(
        {"num_atoms": paddle.to_tensor(_SAMPLE_NUM_ATOMS, dtype="int64")},
        num_inference_steps=num_inference_steps,
    )["result"]


class MiADSmokeTest(unittest.TestCase):
    """End-to-end model flows: training forward and sampling."""

    def test_forward_smoke(self):
        model = _make_tiny_model()
        model.train()
        train_num_atoms = paddle.to_tensor([5, 3], dtype="int64")
        total_atoms = int(train_num_atoms.sum())
        batch = {
            "structure_array": {
                "num_atoms": train_num_atoms,
                "frac_coords": paddle.rand([total_atoms, 3], dtype="float32"),
                "atom_types": paddle.randint(1, 10, [total_atoms], dtype="int64"),
                "lattice": paddle.randn([2, 3, 3], dtype="float32"),
            }
        }
        loss = model(batch)["loss_dict"]["loss"]
        self.assertTrue(paddle.isfinite(loss))

    def test_sample_output_format(self):
        model = _make_tiny_model()
        model.eval()
        result = _sample_small_batch(model)
        self.assertEqual(len(result), 2)
        for entry in result:
            for key in ("num_atoms", "atom_types", "frac_coords", "lattice"):
                self.assertIn(key, entry)
            self.assertEqual(entry["frac_coords"].shape[-1], 3)
            self.assertEqual(entry["num_atoms"], entry["atom_types"].shape[0])
            self.assertTrue((entry["atom_types"] != 0).all())


class MiADConfigTest(unittest.TestCase):
    """End-to-end flow driven by the released YAML config."""

    def test_forward_and_sample_from_yaml(self):
        model = _make_model_from_yaml()
        model.eval()
        with paddle.no_grad():
            output = model(_make_synthetic_batch())
        self.assertTrue(paddle.isfinite(output["loss_dict"]["loss"]))
        result = _sample_small_batch(model)
        self.assertEqual(len(result), 2)
        for entry in result:
            self.assertIn("num_atoms", entry)
            self.assertIn("lattice", entry)


@unittest.skipUnless(
    os.path.exists(_MP20_TEST_CSV),
    f"MP-20 test data not found at {_MP20_TEST_CSV}; skipping dataset tests",
)
class MiADDatasetTest(unittest.TestCase):
    """Real-data pipeline: dataset -> collate -> model forward."""

    def test_collate_to_forward(self):
        dataset = MP20Dataset(
            path=_MP20_TEST_CSV,
            build_structure_cfg={"format": "cif_str", "num_cpus": 1},
        )
        self.assertGreater(len(dataset), 0)
        collator = DefaultCollator()
        samples = [dataset[i] for i in range(min(4, len(dataset)))]
        batch = collator(samples)
        model = _make_tiny_model()
        model.eval()
        with paddle.no_grad():
            output = model(batch)
        self.assertTrue(paddle.isfinite(output["loss_dict"]["loss"]))


class MiADStateDictTest(unittest.TestCase):
    """Official checkpoint layout compatibility, end to end."""

    def test_official_layout_weight_load(self):
        # Layout facts: no "decoder." prefix, PyTorch (out, in) Linear weights,
        # no prop_mlp; the registered checkpoint download is exercised by
        # weight-registry checks.
        model = _make_model_from_yaml()
        sd = model.state_dict()
        self.assertFalse(any("prop_mlp" in k for k in sd.keys()))
        official = {}
        for k, v in sd.items():
            if k.startswith("decoder."):
                name = k[len("decoder.") :]
                if name.endswith(".weight") and len(v.shape) == 2:
                    v = v.T
                official[name] = v
        model2 = _make_model_from_yaml()
        missing, unexpected = model2.set_state_dict(official)
        self.assertEqual(len(missing), 0)
        self.assertEqual(len(unexpected), 0)
        params1 = list(model.named_parameters())
        params2 = list(model2.named_parameters())
        self.assertEqual(len(params1), len(params2))
        for (n1, p1), (n2, p2) in zip(params1, params2):
            self.assertTrue(
                np.allclose(p1.numpy(), p2.numpy()),
                f"parameter {n1} differs after official-layout load",
            )
        model2.eval()
        with paddle.no_grad():
            output = model2(_make_synthetic_batch())
        self.assertTrue(paddle.isfinite(output["loss_dict"]["loss"]))


class MiADSUNMetricTest(unittest.TestCase):
    """S.U.N. evaluation driven by the YAML metric config."""

    def test_sun_metric_from_yaml(self):
        from ppmat.metrics import build_metric

        metrics_fn = build_metric(_load_miad_yaml()["Sample"]["metrics"])
        model = _make_tiny_model()
        model.eval()
        structures = _sample_small_batch(model)
        results = metrics_fn(structures)
        for key in (
            "total",
            "valid",
            "non_trivial",
            "stability_rate",
            "uniqueness_rate",
            "novelty_rate",
            "sun_rate",
            "sun_count",
        ):
            self.assertIn(key, results)
        for key in (
            "stability_rate",
            "uniqueness_rate",
            "novelty_rate",
            "sun_rate",
        ):
            self.assertTrue(np.isfinite(results[key]), f"{key} must be finite")


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


class MiADCinnDispatchTest(unittest.TestCase):
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

        self.assertEqual(names, ["denoise_step"])
        np.testing.assert_allclose(
            cinn_loss.numpy(), eager_loss.numpy(), atol=0, rtol=0
        )
        self.assertEqual(
            cinn_model.state_dict().keys(), eager_model.state_dict().keys()
        )

    def test_sampling_dispatch_matches_eager(self):
        eager_model, cinn_model = _make_dispatch_pair()

        paddle.seed(123)
        eager_result = _sample_small_batch(eager_model)

        names = []
        paddle.seed(123)
        with mock.patch.object(MiAD, "_run_runtime", _record_cinn_boundaries(names)):
            cinn_result = _sample_small_batch(cinn_model)

        self.assertTrue(names)
        self.assertEqual(set(names), {"denoise_step"})
        self.assertEqual(len(cinn_result), len(eager_result))
        for eager_entry, cinn_entry in zip(eager_result, cinn_result):
            self.assertEqual(eager_entry["num_atoms"], cinn_entry["num_atoms"])
            for key in ("atom_types", "frac_coords", "lattice"):
                np.testing.assert_allclose(
                    cinn_entry[key], eager_entry[key], atol=0, rtol=0
                )


class MiADCinnTrainerTest(unittest.TestCase):
    """Trainer orchestration with the CINN backend selected end to end."""

    def setUp(self):
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
        validate_patcher.start()
        runtime_patcher.start()
        self.addCleanup(validate_patcher.stop)
        self.addCleanup(runtime_patcher.stop)

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
                    self.assertTrue((checkpoint_dir / f"{prefix}.{suffix}").is_file())
            saved_state = paddle.load(str(checkpoint_dir / "latest.pdparams"))
            self.assertEqual(saved_state.keys(), first_model.state_dict().keys())

            paddle.seed(2026)
            resumed_model = _make_tiny_model(execution_backend="cinn")
            resumed_trainer, resumed_optimizer = _build_trainer(
                resumed_model, Path(tmp_dir) / "resumed", 2
            )
            with paddle.utils.unique_name.guard():
                resumed_trainer.train(
                    resume_from_checkpoint=str(checkpoint_dir / "latest")
                )

            paddle.seed(2026)
            straight_model = _make_tiny_model(execution_backend="cinn")
            straight_model.set_state_dict(initial_state)
            straight_trainer, straight_optimizer = _build_trainer(
                straight_model, Path(tmp_dir) / "straight", 2
            )
            with paddle.utils.unique_name.guard():
                straight_trainer.train()

        self.assertEqual(first_trainer.state.global_step, 1)
        self.assertEqual(resumed_trainer.state.global_step, 2)
        self.assertEqual(resumed_trainer.state.epoch, 2)
        self.assertEqual(resumed_optimizer.get_lr(), straight_optimizer.get_lr())
        self.assertEqual(
            resumed_optimizer.state_dict().keys(),
            straight_optimizer.state_dict().keys(),
        )
        self.assertTrue(self.boundaries)
        self.assertEqual(set(self.boundaries), {"denoise_step"})
        for model in (resumed_model, straight_model):
            self.assertEqual(model.state_dict().keys(), first_model.state_dict().keys())
            for value in model.state_dict().values():
                self.assertTrue(paddle.isfinite(value).all())
        # Diffusion training is stochastic and BaseTrainer does not restore the
        # RNG state on resume, so weight parity is checked structurally above.


@unittest.skipUnless(
    os.environ.get("PPMAT_RUN_CINN_WORKFLOW_TESTS") == "1",
    "Set PPMAT_RUN_CINN_WORKFLOW_TESTS=1 to run the GPU CINN compile test",
)
class MiADGpuCinnTest(unittest.TestCase):
    """Real CINN compilation on GPU: compiled run must match eager."""

    def test_gpu_cinn_matches_eager(self):
        if (
            not paddle.is_compiled_with_cuda()
            or not paddle.base.is_compiled_with_cinn()
        ):
            self.skipTest("Paddle must be compiled with CUDA and CINN")
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

            self.assertEqual(
                eager_trainer.state.global_step, cinn_trainer.state.global_step
            )
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
            self.assertEqual(len(result), 2)
            for entry in result:
                self.assertTrue((entry["atom_types"] != 0).all())


if __name__ == "__main__":
    unittest.main()

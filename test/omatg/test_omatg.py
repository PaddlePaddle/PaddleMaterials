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
import unittest
from unittest import mock

import numpy as np
import paddle

from ppmat.models.omatg.omatg import OMatG
from ppmat.models.omatg.omatg import OMATGCSPNetFull
from ppmat.models.omatg.si.core import BIG_TIME
from ppmat.models.omatg.si.core import DEFAULT_MAX_ATOMS
from ppmat.models.omatg.si.core import SMALL_TIME


def _make_batch(batch_size=2, atoms_per_struct=(3, 4), max_z=10):
    """Build a minimal PPMAT-style structure_array batch for forward/loss tests."""
    num_list = list(atoms_per_struct[:batch_size])
    total = sum(num_list)
    return {
        "structure_array": {
            "num_atoms": paddle.to_tensor(num_list, dtype="int64"),
            "atom_types": paddle.randint(1, max_z, [total], dtype="int64"),
            "frac_coords": paddle.rand([total, 3]),
            "lattice": paddle.rand([batch_size, 3, 3]) * 2.0
            + paddle.eye(3).unsqueeze(0),
        }
    }


def _make_concat_sample(n):
    """Build a ConcatData-wrapped structure_array sample (DefaultCollator-friendly)."""
    from ppmat.datasets.custom_data_type import ConcatData

    cell = (paddle.eye(3) * 5.0).numpy()
    nums = np.arange(1, n + 1, dtype="int64")
    pos = paddle.rand([n, 3]).numpy()
    return {
        "structure_array": {
            "num_atoms": ConcatData(np.array([n], dtype="int64")),
            "atom_types": ConcatData(nums),
            "lattice": ConcatData(cell.reshape(1, 3, 3)),
            "frac_coords": ConcatData(pos),
        }
    }


def _train_step(model, batch):
    optimizer = paddle.optimizer.Adam(learning_rate=1e-4, parameters=model.parameters())
    output = model(batch)
    assert bool(paddle.isfinite(output["loss_dict"]["loss"]))
    output["loss_dict"]["loss"].backward()
    frozen = {}
    for name, parameter in model.named_parameters():
        if parameter.stop_gradient:
            frozen[name] = parameter.numpy().copy()
        else:
            assert parameter.grad is not None, name
            assert bool(paddle.isfinite(parameter.grad).all()), name
    optimizer.step()
    optimizer.clear_grad()
    for name, parameter in model.named_parameters():
        assert bool(paddle.isfinite(parameter).all()), name
        if name in frozen:
            np.testing.assert_array_equal(parameter.numpy(), frozen[name])
    return output


def _csp_si_scheduler_cfg(integration_time_steps=210):
    """Build the SI scheduler config for CSP (Linear-ODE-style).

    Every kwarg of ``SingleStochasticInterpolant.__init__`` is listed
    explicitly (gamma/epsilon/integrator_kwargs ``None``,
    ``correct_center_of_mass_motion`` as a bool) so the config doubles as
    a regression check against silent default drift.
    """
    return {
        "species": {
            "__class_name__": "SingleStochasticInterpolantIdentity",
            "__init_params__": {},
        },
        "pos": {
            "__class_name__": "SingleStochasticInterpolant",
            "__init_params__": {
                "interpolant": {
                    "__class_name__": "PeriodicLinearInterpolant",
                    "__init_params__": {},
                },
                "gamma": None,
                "epsilon": None,
                "differential_equation_type": "ODE",
                "integrator_kwargs": None,
                "correct_center_of_mass_motion": True,
                "velocity_annealing_factor": 10.18,
            },
        },
        "cell": {
            "__class_name__": "SingleStochasticInterpolant",
            "__init_params__": {
                "interpolant": {
                    "__class_name__": "LinearInterpolant",
                    "__init_params__": {},
                },
                "gamma": None,
                "epsilon": None,
                "differential_equation_type": "ODE",
                "integrator_kwargs": None,
                "correct_center_of_mass_motion": False,
                "velocity_annealing_factor": 1.82,
            },
        },
        "integration_time_steps": integration_time_steps,
        "relative_si_costs": {
            "species_loss": 0.0,
            "pos_loss_b": 0.9994,
            "cell_loss_b": 0.0006,
        },
    }


def _dng_si_scheduler_cfg(integration_time_steps=710):
    """Build the SI scheduler config for DNG (Linear-SDE + DFM mask)."""
    return {
        "species": {
            "__class_name__": "DiscreteFlowMatchingMask",
            "__init_params__": {"noise": 0.189},
        },
        "pos": {
            "__class_name__": "SingleStochasticInterpolant",
            "__init_params__": {
                "interpolant": {
                    "__class_name__": "PeriodicLinearInterpolant",
                    "__init_params__": {},
                },
                "gamma": {
                    "__class_name__": "LatentGammaSqrt",
                    "__init_params__": {"a": 0.018},
                },
                "epsilon": {
                    "__class_name__": "VanishingEpsilon",
                    "__init_params__": {"c": 9.7, "mu": 0.17, "sigma": 0.029},
                },
                "differential_equation_type": "SDE",
                "integrator_kwargs": None,
                "correct_center_of_mass_motion": True,
                "velocity_annealing_factor": 6.33,
            },
        },
        "cell": {
            "__class_name__": "SingleStochasticInterpolant",
            "__init_params__": {
                "interpolant": {
                    "__class_name__": "LinearInterpolant",
                    "__init_params__": {},
                },
                "gamma": None,
                "epsilon": None,
                "differential_equation_type": "ODE",
                "integrator_kwargs": None,
                "correct_center_of_mass_motion": False,
                "velocity_annealing_factor": 1.07,
            },
        },
        "integration_time_steps": integration_time_steps,
        "relative_si_costs": {
            "species_loss": 0.5918,
            "pos_loss_b": 0.1309,
            "pos_loss_z": 0.2708,
            "cell_loss_b": 0.0065,
        },
    }


def _build_omatg_model(
    pred_type,
    steps=10,
    backend="eager",
    hidden_dim=32,
    num_layers=1,
    time_embed_dim=16,
):
    """Build a small OMatG model from the config-dict style constructor."""
    si_cfg = (
        _dng_si_scheduler_cfg(steps) if pred_type else _csp_si_scheduler_cfg(steps)
    )
    decoder_cfg = {
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "max_atoms": DEFAULT_MAX_ATOMS,
        "time_embed_dim": time_embed_dim,
        "edge_style": "fc",
        "pred_type": pred_type,
        # Additive lattice updates keep the few-step sampling of a randomly
        # initialised test model numerically stable.
        "ip": False,
    }
    sampler_cfg = {
        "dataset_name": "mp_20",
        "mirror_species": not pred_type,
        "mask_species": pred_type,
    }
    with paddle.utils.unique_name.guard():
        paddle.seed(2026)
        model = OMatG(
            decoder_cfg=decoder_cfg,
            si_scheduler_cfg=si_cfg,
            sampler_cfg=sampler_cfg,
            execution_backend=backend,
        )
    return model


class TestOMatG(unittest.TestCase):
    """CSP / DNG training modes and the released-package constructor shim."""

    def test_csp_forward_loss(self):
        """CSP mode forward returns finite loss_dict."""
        model = _build_omatg_model(False)
        self.assertFalse(model.supports_num_atoms_sampling)
        output = model(_make_batch())
        self.assertIn("loss_dict", output)
        loss_val = float(output["loss_dict"]["loss"])
        self.assertFalse(loss_val != loss_val, "loss is NaN")

    def test_dng_forward_loss(self):
        """DNG mode forward returns finite loss_dict including species_loss."""
        model = _build_omatg_model(True)
        self.assertTrue(model.supports_num_atoms_sampling)
        output = model(_make_batch())
        self.assertIn("species_loss", output["loss_dict"])
        loss_val = float(output["loss_dict"]["loss"])
        self.assertFalse(loss_val != loss_val, "DNG loss is NaN")

    def test_released_package_constructor_matches(self):
        """OMATGCSPNetFull keeps the flat constructor used by the released YAMLs."""
        cfg = _csp_si_scheduler_cfg(10)
        sampler_cfg = {"dataset_name": "mp_20", "mirror_species": True}
        legacy = OMATGCSPNetFull(
            hidden_dim=32,
            num_layers=1,
            max_atoms=DEFAULT_MAX_ATOMS,
            time_embed_dim=16,
            si_scheduler_cfg=cfg,
            sampler_cfg=sampler_cfg,
        )
        model = OMatG(
            decoder_cfg={
                "hidden_dim": 32,
                "num_layers": 1,
                "max_atoms": DEFAULT_MAX_ATOMS,
                "time_embed_dim": 16,
                "pred_type": False,
                "ln": True,
                "num_freqs": 128,
            },
            si_scheduler_cfg=cfg,
            sampler_cfg=sampler_cfg,
        )
        legacy_state = legacy.state_dict()
        model_state = model.state_dict()
        self.assertEqual(set(legacy_state), set(model_state))
        for key in legacy_state:
            self.assertEqual(legacy_state[key].shape, model_state[key].shape)


class TestSITrainingPath(unittest.TestCase):
    """SI velocity-matching training path (CSP and DNG), including CSP sampling."""

    def _build_csp_si_model(self):
        """Build a small CSP model with SI (Linear-ODE)."""
        return _build_omatg_model(False, steps=10)

    def test_csp_si_training_and_sampling(self):
        """CSP SI path trains (velocity matching) and samples via si.integrate."""
        model = self._build_csp_si_model()
        bias = model.csp_layer_0.edge_mlp[0].bias
        bias.set_value(paddle.full_like(bias, -100.0))
        batch = _make_batch()
        out = _train_step(model, batch)
        self.assertIn("loss", out["loss_dict"])
        loss_val = float(out["loss_dict"]["loss"])
        self.assertFalse(loss_val != loss_val, "SI loss is NaN")

        result = model.sample(batch)
        self.assertEqual(len(result["result"]), 2)
        self.assertIn("frac_coords", result["result"][0])

    def test_dng_si_training_and_sampling(self):
        """DNG trains every active head and samples synchronously from masks."""
        cfg = _dng_si_scheduler_cfg(integration_time_steps=710)
        model = _build_omatg_model(True, steps=710)
        batch = _make_batch()
        out = _train_step(model, batch)
        self.assertIn("loss", out["loss_dict"])
        loss_val = float(out["loss_dict"]["loss"])
        self.assertFalse(loss_val != loss_val, "DNG SDE loss is NaN")
        states = []
        predictions = []
        noises = []
        original_forward = model.forward_dict
        original_randn = paddle.randn

        def record_state(**kwargs):
            states.append(
                {
                    key: kwargs[key].numpy().copy()
                    for key in ("atom_types", "frac_coords", "lattices")
                }
            )
            output = original_forward(**kwargs)
            predictions.append(
                {key: value.numpy().copy() for key, value in output.items()}
            )
            return output

        def record_noise(*args, **kwargs):
            value = original_randn(*args, **kwargs)
            if tuple(value.shape) == (
                int(batch["structure_array"]["num_atoms"].sum()),
                3,
            ):
                noises.append(value.numpy().copy())
            return value

        paddle.seed(42)
        with mock.patch.object(
            model, "forward_dict", side_effect=record_state
        ), mock.patch.object(paddle, "randn", side_effect=record_noise):
            result = model.sample(batch, num_inference_steps=10)
        for start in range(0, len(states), 3):
            for state in states[start + 1 : start + 3]:
                for key, value in states[start].items():
                    np.testing.assert_allclose(state[key], value, atol=1e-7)
        self.assertTrue(np.any(states[3]["atom_types"] == 0))
        pos_params = cfg["pos"]["__init_params__"]
        epsilon_params = pos_params["epsilon"]["__init_params__"]
        time = float(np.float32(SMALL_TIME))
        dt = (BIG_TIME - SMALL_TIME) / 9
        epsilon = epsilon_params["c"] / (
            (1 + np.exp(-(time - epsilon_params["mu"]) / epsilon_params["sigma"]))
            * (1 + np.exp(-(1 - epsilon_params["mu"] - time) / epsilon_params["sigma"]))
        )
        gamma = np.sqrt(pos_params["gamma"]["__init_params__"]["a"] * time * (1 - time))
        expected_pos = (
            states[0]["frac_coords"]
            + (1 + pos_params["velocity_annealing_factor"] * time)
            * predictions[1]["pos_b"]
            * dt
            - epsilon / gamma * predictions[1]["pos_eta"] * dt
            + np.sqrt(2 * epsilon * dt) * noises[0]
        ) % 1.0
        np.testing.assert_allclose(states[3]["frac_coords"], expected_pos, atol=5e-6)
        for item in result["result"]:
            self.assertTrue(all(number > 0 for number in item["atom_types"]))
            self.assertTrue(np.isfinite(item["frac_coords"]).all())


class TestSIFactory(unittest.TestCase):
    """Config-driven SI end-to-end: YAML -> build_model -> forward -> sample."""

    def test_config_driven_dng_si_from_yaml(self):
        """The DNG yaml drives build -> forward -> sample and rejects bad configs."""
        import copy

        from omegaconf import OmegaConf

        from ppmat.datasets.collate_fn import DefaultCollator
        from ppmat.models import build_model

        cfg = OmegaConf.load("structure_generation/configs/omatg/omatg_mp20_dng.yaml")
        model_cfg = OmegaConf.to_container(cfg["Model"], resolve=True)
        init_params = model_cfg["__init_params__"]
        init_params["decoder_cfg"].update(
            hidden_dim=32, num_layers=1, time_embed_dim=16
        )
        init_params["si_scheduler_cfg"]["integration_time_steps"] = 10
        model = build_model(copy.deepcopy(model_cfg))
        self.assertTrue(model.use_min_perm_dist)
        self.assertTrue(model.supports_num_atoms_sampling)

        batch = DefaultCollator()([_make_concat_sample(3), _make_concat_sample(4)])
        out = model(batch)
        loss_val = float(out["loss_dict"]["loss"])
        self.assertFalse(loss_val != loss_val, "DNG SI loss is NaN")
        result = model.sample(batch, num_inference_steps=10)
        self.assertEqual(len(result["result"]), 2)
        for entry in result["result"]:
            coords = paddle.to_tensor(entry["frac_coords"])
            self.assertFalse(
                bool(paddle.isnan(coords).any()), "sampled frac_coords contain NaN"
            )

        # A typo in a loss weight must fail at build time instead of silently
        # reweighting the training loss.
        costs = model_cfg["__init_params__"]["si_scheduler_cfg"]["relative_si_costs"]
        costs["pos_loss_bb"] = costs.pop("pos_loss_b")
        with self.assertRaises(ValueError):
            build_model(copy.deepcopy(model_cfg))

    def test_config_driven_csp_si_from_yaml(self):
        """The CSP yaml drives build -> forward -> sample with the PPMAT contract."""
        import copy

        from omegaconf import OmegaConf

        from ppmat.datasets.collate_fn import DefaultCollator
        from ppmat.models import build_model

        cfg = OmegaConf.load("structure_generation/configs/omatg/omatg_mp20_csp.yaml")
        model_cfg = OmegaConf.to_container(cfg["Model"], resolve=True)
        init_params = model_cfg["__init_params__"]
        init_params["decoder_cfg"].update(
            hidden_dim=32, num_layers=1, time_embed_dim=16, ip=False
        )
        init_params["si_scheduler_cfg"]["integration_time_steps"] = 10
        model = build_model(copy.deepcopy(model_cfg))
        self.assertFalse(model.supports_num_atoms_sampling)
        self.assertIsNone(model._min_perm_dist_corrector)

        batch = DefaultCollator()([_make_concat_sample(3), _make_concat_sample(4)])
        out = model(batch)
        loss_val = float(out["loss_dict"]["loss"])
        self.assertFalse(loss_val != loss_val, "CSP SI loss is NaN")

        result = model.sample(batch, num_inference_steps=10)
        self.assertEqual(len(result["result"]), 2)
        for entry in result["result"]:
            self.assertEqual(
                set(entry), {"num_atoms", "atom_types", "frac_coords", "lattice"}
            )
            coords = paddle.to_tensor(entry["frac_coords"])
            self.assertFalse(bool(paddle.isnan(coords).any()))

    def test_csv_dataset_to_model_pipeline(self):
        """CSV -> MP20Dataset -> DefaultCollator -> SI forward."""
        import tempfile
        from pathlib import Path

        import pandas as pd
        from pymatgen.core import Lattice
        from pymatgen.core import Structure

        from ppmat.datasets.collate_fn import DefaultCollator
        from ppmat.datasets.mp20_dataset import MP20Dataset

        lengths = [4.0, 4.0, 4.0]
        angles = [90.0, 90.0, 90.0]
        rng = np.random.RandomState(5)
        cifs = []
        for n, species in ((2, [3, 8]), (3, [11, 8, 3]), (4, [8, 8, 3, 11])):
            structure = Structure(
                lattice=Lattice.from_parameters(*(lengths + angles)),
                species=species,
                coords=rng.rand(n, 3),
                coords_are_cartesian=False,
            )
            cifs.append(structure.to(fmt="cif"))

        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "train.csv"
            pd.DataFrame({"cif": cifs}).to_csv(csv_path, index=False)
            dataset = MP20Dataset(
                path=str(csv_path),
                build_structure_cfg={
                    "format": "cif_str",
                    "primitive": False,
                    "niggli": False,
                    "num_cpus": 1,
                },
                cache_path=str(Path(tmp) / "cache"),
            )
            samples = [dataset[i] for i in range(len(dataset))]
            for sample, cif in zip(samples, cifs):
                ref = Structure.from_str(cif, fmt="cif")
                np.testing.assert_allclose(
                    np.asarray(sample["structure_array"]["frac_coords"].data),
                    ref.frac_coords,
                    atol=1e-6,
                )

        self.assertEqual(len(samples), 3)
        batch = DefaultCollator()(samples)
        structure_array = batch["structure_array"]
        self.assertEqual(len(structure_array["num_atoms"]), 3)
        self.assertEqual(len(structure_array["atom_types"]), 9)

        model = _make_cinn_model("eager")
        out = model(batch)
        loss_val = float(out["loss_dict"]["loss"])
        self.assertFalse(loss_val != loss_val, "dataset-pipeline loss is NaN")


def _make_cinn_model(execution_backend="eager"):
    """Build a masked-species DNG model with a fixed seed per backend."""
    cfg = _dng_si_scheduler_cfg(integration_time_steps=10)
    with paddle.utils.unique_name.guard():
        paddle.seed(2026)
        model = OMATGCSPNetFull(
            hidden_dim=32,
            num_layers=1,
            max_atoms=DEFAULT_MAX_ATOMS,
            time_embed_dim=16,
            pred_type=True,
            use_si=True,
            si_scheduler_cfg=cfg,
            sampler_cfg={
                "dataset_name": "mp_20",
                "mask_species": True,
                "mirror_species": False,
            },
            execution_backend=execution_backend,
        )
    return model


def _passthrough_run_runtime(self, name, function, *args, **kwargs):
    return function(*args, **kwargs)


def _recording_run_runtime(names):
    def run_runtime(self, name, function, *args, **kwargs):
        names.append(name)
        return function(*args, **kwargs)

    return run_runtime


def _make_loader():
    from ppmat.datasets.collate_fn import DefaultCollator

    class _Samples(paddle.io.Dataset):
        def __init__(self):
            self.samples = [_make_concat_sample(n) for n in (3, 4)]

        def __getitem__(self, idx):
            return self.samples[idx]

        def __len__(self):
            return len(self.samples)

    return paddle.io.DataLoader(
        _Samples(), batch_size=2, shuffle=False, collate_fn=DefaultCollator()
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


def _build_trainer(model, output_dir, max_epochs, backend):
    from ppmat.trainer.base_trainer import BaseTrainer

    with paddle.utils.unique_name.guard():
        scheduler = paddle.optimizer.lr.StepDecay(
            learning_rate=1e-3, step_size=1, gamma=0.5
        )
        scheduler.by_epoch = False
        optimizer = paddle.optimizer.Adam(
            learning_rate=scheduler, parameters=model.parameters()
        )
    loader = _make_loader()
    trainer = BaseTrainer(
        _trainer_config(output_dir, max_epochs),
        model,
        train_dataloader=loader,
        val_dataloader=loader,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        execution_config={"backend": backend},
    )
    return trainer, optimizer


class TestCINNDispatch(unittest.TestCase):
    """CINN dispatch boundary for the SI training path."""

    def test_si_dispatch_records_only_denoise_step(self):
        """One boundary name is dispatched and eager/cinn numbers stay bitwise equal."""
        batch = _make_batch()
        eager_model = _make_cinn_model("eager")
        paddle.seed(2027)
        expected = eager_model(batch)["loss_dict"]

        names = []
        dispatched = _make_cinn_model("cinn")
        dispatched.set_state_dict(eager_model.state_dict())
        paddle.seed(2027)
        with mock.patch.object(
            OMATGCSPNetFull, "_run_runtime", _recording_run_runtime(names)
        ):
            actual = dispatched(batch)["loss_dict"]
            dispatched.eval()
            dispatched.sample(batch, num_inference_steps=3)

        self.assertTrue(names)
        self.assertEqual(set(names), {"denoise_step"})
        self.assertEqual(
            dispatched.state_dict().keys(), eager_model.state_dict().keys()
        )
        self.assertEqual(actual.keys(), expected.keys())
        for key in expected:
            np.testing.assert_array_equal(actual[key].numpy(), expected[key].numpy())


class TestCINNTrainerOrchestration(unittest.TestCase):
    """Trainer resume orchestration through the cinn backend (CPU)."""

    def setUp(self):
        validate_patch = mock.patch.object(
            OMATGCSPNetFull,
            "validate_execution_backend",
            lambda self, **kwargs: None,
        )
        run_patch = mock.patch.object(
            OMATGCSPNetFull, "_run_runtime", _passthrough_run_runtime
        )
        validate_patch.start()
        run_patch.start()
        self.addCleanup(validate_patch.stop)
        self.addCleanup(run_patch.stop)

    def test_resume_orchestration_with_cinn_backend(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            initial_state = _make_cinn_model("cinn").state_dict()

            first_model = _make_cinn_model("cinn")
            first_model.set_state_dict(initial_state)
            first_trainer, _ = _build_trainer(
                first_model, tmp_path / "first", 1, "cinn"
            )
            with paddle.utils.unique_name.guard():
                first_trainer.train()

            checkpoint_dir = tmp_path / "first" / "checkpoints"
            for prefix in ("epoch_1", "latest", "best"):
                for suffix in ("pdparams", "pdopt", "pdstates"):
                    self.assertTrue((checkpoint_dir / f"{prefix}.{suffix}").is_file())
            checkpoint_state = paddle.load(str(checkpoint_dir / "latest.pdparams"))
            self.assertEqual(checkpoint_state.keys(), first_model.state_dict().keys())
            self.assertTrue(
                all(not key.startswith("model.") for key in checkpoint_state)
            )

            resumed_model = _make_cinn_model("cinn")
            resumed_trainer, resumed_optimizer = _build_trainer(
                resumed_model, tmp_path / "resumed", 2, "cinn"
            )
            optimizer_state = paddle.load(str(checkpoint_dir / "latest.pdopt"))
            restored = []
            original_train_epoch = resumed_trainer.train_epoch

            def verify_restored(*args, **kwargs):
                self.assertEqual(resumed_trainer.state.global_step, 1)
                for key, value in checkpoint_state.items():
                    np.testing.assert_array_equal(
                        resumed_model.state_dict()[key].numpy(), value.numpy()
                    )
                current = resumed_optimizer.state_dict()
                self.assertEqual(current.keys(), optimizer_state.keys())
                for key, value in optimizer_state.items():
                    if isinstance(value, dict):
                        self.assertEqual(current[key], value)
                    else:
                        np.testing.assert_array_equal(
                            current[key].numpy(), value.numpy()
                        )
                self.assertEqual(resumed_optimizer.get_lr(), 5e-4)
                restored.append(True)
                return original_train_epoch(*args, **kwargs)

            with paddle.utils.unique_name.guard(), mock.patch.object(
                resumed_trainer, "train_epoch", side_effect=verify_restored
            ):
                resumed_trainer.train(
                    resume_from_checkpoint=str(checkpoint_dir / "latest")
                )
            self.assertEqual(restored, [True])

            straight_model = _make_cinn_model("cinn")
            straight_model.set_state_dict(initial_state)
            straight_trainer, straight_optimizer = _build_trainer(
                straight_model, tmp_path / "straight", 2, "cinn"
            )
            with paddle.utils.unique_name.guard():
                straight_trainer.train()

            self.assertEqual(first_trainer.state.global_step, 1)
            self.assertEqual(resumed_trainer.state.global_step, 2)
            self.assertEqual(resumed_trainer.state.epoch, 2)
            self.assertEqual(resumed_optimizer.get_lr(), straight_optimizer.get_lr())
            self.assertEqual(
                resumed_trainer.lr_scheduler.state_dict(),
                straight_trainer.lr_scheduler.state_dict(),
            )
            self.assertEqual(
                resumed_model.state_dict().keys(), straight_model.state_dict().keys()
            )
            self.assertEqual(
                resumed_optimizer.state_dict().keys(),
                straight_optimizer.state_dict().keys(),
            )


@unittest.skipUnless(
    os.environ.get("PPMAT_RUN_CINN_WORKFLOW_TESTS") == "1",
    "Set PPMAT_RUN_CINN_WORKFLOW_TESTS=1 to run the GPU CINN compile parity.",
)
class TestCINNGPUCompileParity(unittest.TestCase):
    """Real CINN compilation parity for one training epoch."""

    def _require_gpu_cinn(self):
        if not paddle.is_compiled_with_cuda():
            self.skipTest("Paddle was not compiled with CUDA.")
        if not paddle.base.is_compiled_with_cinn():
            self.skipTest("Paddle was not compiled with CINN.")
        paddle.set_device("gpu:0")

    def test_one_training_epoch_matches_eager(self):
        """One compiled training epoch must track eager within the SI tolerance.

        The full SI sampling loop is not comparable on GPU: ``segment_mean`` uses
        atomic scatter, so even two eager runs diverge inside the chaotic SDE
        integration. The training epoch is the stable end-to-end parity check.
        """
        import tempfile
        from pathlib import Path

        self._require_gpu_cinn()
        initial_state = {
            key: value.clone()
            for key, value in _make_cinn_model("eager").state_dict().items()
        }

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            eager_model = _make_cinn_model("eager")
            eager_model.set_state_dict(initial_state)
            cinn_model = _make_cinn_model("cinn")
            cinn_model.set_state_dict(initial_state)

            paddle.seed(2026)
            np.random.seed(2026)
            eager_trainer, _ = _build_trainer(
                eager_model, tmp_path / "eager", 1, "eager"
            )
            with paddle.utils.unique_name.guard():
                eager_trainer.train()

            paddle.seed(2026)
            np.random.seed(2026)
            cinn_trainer, _ = _build_trainer(cinn_model, tmp_path / "cinn", 1, "cinn")
            with paddle.utils.unique_name.guard():
                cinn_trainer.train()

            self.assertEqual(eager_trainer.state.global_step, 1)
            self.assertEqual(cinn_trainer.state.global_step, 1)
            for key in initial_state:
                np.testing.assert_allclose(
                    cinn_model.state_dict()[key].numpy(),
                    eager_model.state_dict()[key].numpy(),
                    atol=1e-3,
                    rtol=1e-3,
                )


def _metric_sample(num_atoms, seed, species=(3, 8, 11)):
    """Deterministic cubic-crystal sample dict matching model.sample() output."""
    rng = np.random.RandomState(seed)
    return {
        "num_atoms": num_atoms,
        "atom_types": [species[i % len(species)] for i in range(num_atoms)],
        "frac_coords": rng.rand(num_atoms, 3).round(4).tolist(),
        "lengths": [4.0, 4.0, 4.0],
        "angles": [90.0, 90.0, 90.0],
    }


class TestOMatGMetric(unittest.TestCase):
    """CSP matching (shared CSPMetric) and OMatG de novo metrics."""

    @staticmethod
    def _write_gt_csv(csv_path, ref):
        import pandas as pd
        from pymatgen.core import Lattice
        from pymatgen.core import Structure

        structures = [
            Structure(
                lattice=Lattice.from_parameters(*(item["lengths"] + item["angles"])),
                species=item["atom_types"],
                coords=item["frac_coords"],
                coords_are_cartesian=False,
            )
            for item in ref
        ]
        pd.DataFrame({"cif": [s.to(fmt="cif") for s in structures]}).to_csv(
            csv_path, index=False
        )

    def test_csp_metric_reuses_framework_metric(self):
        """CSP evaluation goes through the shared CSPMetric."""
        import tempfile
        from pathlib import Path

        from ppmat.metrics.csp_metric import CSPMetric

        ref = [_metric_sample(3, 0), _metric_sample(4, 1), _metric_sample(5, 2)]
        gen = [dict(ref[0]), dict(ref[1]), _metric_sample(5, seed=3)]

        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "gt.csv"
            self._write_gt_csv(csv_path, ref)
            metrics = CSPMetric(gt_file_path=str(csv_path))(gen)

        self.assertEqual(set(metrics), {"match_rate", "rms_dist"})
        self.assertGreater(metrics["match_rate"], 0.0)
        self.assertTrue(np.isfinite(metrics["rms_dist"]))

    def test_dng_metrics_smoke(self):
        """DNG mode reports validity, distribution, METRe and COV metrics."""
        import tempfile
        from pathlib import Path

        from ppmat.metrics.omatg_metric import OMatGMetric

        ref = [
            {
                "lengths": [5.0, 5.0, 5.0],
                "angles": [90.0, 90.0, 90.0],
                "atom_types": [11, 17],
                "frac_coords": [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
            },
            {
                "lengths": [5.0, 5.0, 5.0],
                "angles": [90.0, 90.0, 90.0],
                "atom_types": [3, 9],
                "frac_coords": [[0.0, 0.0, 0.0], [0.3, 0.0, 0.0]],
            },
        ]
        invalid = dict(ref[0], lengths=[0.0, 0.0, 0.0])
        gen = [ref[0], invalid, ref[0]]
        metrics = OMatGMetric(dataset_name="mp_20")(gen, gt_data=ref)
        expected_keys = {
            "valid_rate",
            "composition_valid_rate",
            "structural_valid_rate",
            "wdist_density",
            "wdist_narity",
            "wdist_coordination_numbers",
            "metre_rate",
            "metre_mean_rmsd",
            "metre_corr_rmsd",
            "metre_valid_rate",
            "metre_valid_mean_rmsd",
            "metre_valid_corr_rmsd",
            "cov_precision",
            "cov_recall",
            "dng_eval",
        }
        self.assertTrue(expected_keys <= set(metrics))
        for value in metrics.values():
            self.assertFalse(value != value, "metric value is NaN")
        self.assertAlmostEqual(metrics["valid_rate"], 2.0 / 3.0)
        # Composition validity is only evaluated for structurally valid entries,
        # so the copy with zero lattice lengths counts as composition-invalid.
        self.assertAlmostEqual(metrics["composition_valid_rate"], 2.0 / 3.0)
        self.assertAlmostEqual(metrics["structural_valid_rate"], 2.0 / 3.0)
        self.assertAlmostEqual(metrics["metre_rate"], 0.5)
        self.assertAlmostEqual(metrics["wdist_coordination_numbers"], 0.5)

        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "gt.csv"
            self._write_gt_csv(csv_path, ref)
            csv_metrics = OMatGMetric(
                dataset_name="mp_20", gt_file_path=str(csv_path)
            )(gen)
        for key, value in csv_metrics.items():
            self.assertAlmostEqual(metrics[key], value)
        self.assertEqual(
            OMatGMetric(dataset_name="mp_20")([invalid], ref),
            {
                "valid_rate": 0.0,
                "composition_valid_rate": 0.0,
                "structural_valid_rate": 0.0,
            },
        )


if __name__ == "__main__":
    unittest.main()

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

from ppmat.models.omatg.model import OMATGCSPNetFull
from ppmat.models.omatg.si.core import DEFAULT_MAX_ATOMS


def _make_batch(batch_size=2, atoms_per_struct=(3, 4), max_z=10):
    """Build a minimal OMatG sample dict batch for forward/loss tests."""
    num_list = list(atoms_per_struct[:batch_size])
    total = sum(num_list)
    atom_types = paddle.randint(1, max_z, [total], dtype="int64")
    frac_coords = paddle.rand([total, 3])
    lattices = paddle.rand([batch_size, 3, 3]) * 2.0 + paddle.eye(3).unsqueeze(0)
    num_atoms = paddle.to_tensor(num_list, dtype="int64")
    node2graph = paddle.repeat_interleave(
        paddle.arange(batch_size, dtype="int64"), num_atoms
    )
    return {
        "n_atoms": num_atoms,
        "species": atom_types,
        "cell": lattices,
        "pos": frac_coords,
        "pos_is_fractional": paddle.ones([batch_size], dtype="bool"),
        "batch": node2graph,
        "ptr": paddle.concat(
            [
                paddle.to_tensor([0], dtype="int64"),
                paddle.cumsum(num_atoms, axis=0).cast("int64"),
            ]
        ),
    }


def _make_concat_sample(n):
    """Build a ConcatData-wrapped sample dict (DefaultCollator-friendly)."""
    from ppmat.datasets.custom_data_type import ConcatData

    cell = (paddle.eye(3) * 5.0).numpy()
    nums = np.arange(1, n + 1, dtype="int64")
    pos = paddle.rand([n, 3]).numpy()
    return {
        "n_atoms": ConcatData(np.array([n], dtype="int64")),
        "species": ConcatData(nums),
        "cell": ConcatData(cell.reshape(1, 3, 3)),
        "pos": ConcatData(pos),
        "pos_is_fractional": ConcatData(np.array([True], dtype="bool")),
    }


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


class TestOMATGCSPNetFull(unittest.TestCase):
    """CSP / DNG training mode minimal forward and loss tests."""

    def test_csp_forward_loss(self):
        """CSP mode forward returns finite loss_dict."""
        model = OMATGCSPNetFull(
            hidden_dim=64,
            num_layers=2,
            max_atoms=DEFAULT_MAX_ATOMS,
            time_embed_dim=32,
            edge_style="fc",
            pred_type=False,
        )
        output = model(_make_batch())
        self.assertIn("loss_dict", output)
        loss_val = float(output["loss_dict"]["loss"])
        self.assertFalse(loss_val != loss_val, "loss is NaN")

    def test_dng_forward_loss(self):
        """DNG mode forward returns finite loss_dict including loss_type."""
        model = OMATGCSPNetFull(
            hidden_dim=64,
            num_layers=2,
            max_atoms=DEFAULT_MAX_ATOMS,
            time_embed_dim=32,
            edge_style="fc",
            pred_type=True,
        )
        output = model(_make_batch())
        self.assertIn("loss_type", output["loss_dict"])
        loss_val = float(output["loss_dict"]["loss"])
        self.assertFalse(loss_val != loss_val, "DNG loss is NaN")


class TestSITrainingPath(unittest.TestCase):
    """SI velocity-matching training path (CSP and DNG), including CSP sampling."""

    def _build_csp_si_model(self):
        """Build a small CSP model with SI (Linear-ODE)."""
        cfg = _csp_si_scheduler_cfg(integration_time_steps=210)
        return OMATGCSPNetFull(
            hidden_dim=32,
            num_layers=1,
            max_atoms=DEFAULT_MAX_ATOMS,
            time_embed_dim=16,
            pred_type=False,
            use_si=True,
            si_scheduler_cfg=cfg,
            sampler_cfg={"dataset_name": "mp_20", "mirror_species": True},
        )

    def test_csp_si_training_and_sampling(self):
        """CSP SI path trains (velocity matching) and samples via si.integrate."""
        model = self._build_csp_si_model()
        batch = _make_batch()
        out = model(batch)
        self.assertIn("loss", out["loss_dict"])
        loss_val = float(out["loss_dict"]["loss"])
        self.assertFalse(loss_val != loss_val, "SI loss is NaN")

        result = model.sample(batch, num_inference_steps=10)
        self.assertEqual(len(result["result"]), 2)
        self.assertIn("frac_coords", result["result"][0])

    def test_dng_si_forward_loss(self):
        """DNG SI training path (SDE + gamma + DFM mask) stays finite."""
        cfg = _dng_si_scheduler_cfg(integration_time_steps=710)
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
        )
        out = model(_make_batch())
        self.assertIn("loss", out["loss_dict"])
        loss_val = float(out["loss_dict"]["loss"])
        self.assertFalse(loss_val != loss_val, "DNG SDE loss is NaN")


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
        init_params.update(hidden_dim=32, num_layers=1, time_embed_dim=16)
        init_params["si_scheduler_cfg"]["integration_time_steps"] = 10
        model = build_model(copy.deepcopy(model_cfg))
        self.assertTrue(model.use_si)
        self.assertTrue(model.use_min_perm_dist)

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

    def test_csv_dataset_to_model_pipeline(self):
        """CSV -> OMATGStructureDataset -> DefaultCollator -> SI forward."""
        import tempfile
        from pathlib import Path

        import pandas as pd
        from pymatgen.core import Lattice
        from pymatgen.core import Structure

        from ppmat.datasets.collate_fn import DefaultCollator
        from ppmat.datasets.omatg_dataset import OMATGStructureDataset

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
            dataset = OMATGStructureDataset(file_path=str(csv_path), lazy_storage=False)
            samples = [dataset[i] for i in range(len(dataset))]

        # CSV samples carry fractional coords; convert_to_fractional must be
        # a no-op on them (regression guard against double conversion).
        for sample, cif in zip(samples, cifs):
            ref = Structure.from_str(cif, fmt="cif")
            np.testing.assert_allclose(
                np.asarray(sample["pos"].data), ref.frac_coords, atol=1e-6
            )

        self.assertEqual(len(samples), 3)
        batch = DefaultCollator()(samples)
        self.assertEqual(len(batch["n_atoms"]), 3)
        self.assertEqual(len(batch["species"]), 9)

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

    optimizer = paddle.optimizer.Adam(learning_rate=1e-3, parameters=model.parameters())
    loader = _make_loader()
    trainer = BaseTrainer(
        _trainer_config(output_dir, max_epochs),
        model,
        train_dataloader=loader,
        val_dataloader=loader,
        optimizer=optimizer,
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
            with paddle.utils.unique_name.guard():
                resumed_trainer.train(
                    resume_from_checkpoint=str(checkpoint_dir / "latest")
                )

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
    """Metric end-to-end: one-shot call matches streaming accumulation."""

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

    def test_match_one_shot_equals_streaming(self):
        import tempfile
        from pathlib import Path

        from ppmat.metrics.omatg_metric import OMatGMetric

        ref = [_metric_sample(3, 0), _metric_sample(4, 1), _metric_sample(5, 2)]
        gen = [dict(ref[0]), dict(ref[1]), _metric_sample(5, seed=3)]

        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "gt.csv"
            self._write_gt_csv(csv_path, ref)

            one_shot = OMatGMetric(metric_type="match", gt_file_path=str(csv_path))
            one_shot_metrics = one_shot(gen)

            streaming = OMatGMetric(metric_type="match", gt_file_path=str(csv_path))
            streaming.update_step(result={"result": gen[:2]}, batch=None, stage="eval")
            streaming.update_step(result={"result": gen[2:]}, batch=None, stage="eval")
            streamed_metrics = streaming.compute_epoch(stage="eval")

        self.assertEqual(set(one_shot_metrics), set(streamed_metrics))
        for key in one_shot_metrics:
            self.assertAlmostEqual(one_shot_metrics[key], streamed_metrics[key])
        self.assertGreater(one_shot_metrics["match_rate"], 0.0)

    def test_dng_metrics_smoke(self):
        """DNG mode reports validity, distribution, METRe and COV metrics."""
        from ppmat.metrics.omatg_metric import OMatGMetric

        ref = [_metric_sample(3, 0), _metric_sample(4, 1), _metric_sample(5, 2)]
        gen = [_metric_sample(3, 0), _metric_sample(4, 1), _metric_sample(5, 3)]
        metrics = OMatGMetric(metric_type="dng", dataset_name="mp_20")(gen, gt_data=ref)
        expected_keys = {
            "valid_rate",
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


if __name__ == "__main__":
    unittest.main()

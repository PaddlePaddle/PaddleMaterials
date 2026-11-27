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

"""End-to-end tests for the OMatG integration.

The tests cover the public flows only: YAML -> build_model -> forward/sample,
CSV -> dataset -> collator -> model, one optimiser step, and the CSP / DNG
metric entry points.
"""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import paddle
import pandas as pd
from omegaconf import OmegaConf
from pymatgen.core import Lattice
from pymatgen.core import Structure

from ppmat.datasets.collate_fn import DefaultCollator
from ppmat.datasets.mp20_dataset import MP20Dataset
from ppmat.metrics.csp_metric import CSPMetric
from ppmat.metrics.dng_metric import DNGMetric
from ppmat.models import build_model
from ppmat.models.omatg.omatg import OMatG

_CONFIGS = {
    "csp": "structure_generation/configs/omatg/omatg_mp20_csp.yaml",
    "dng": "structure_generation/configs/omatg/omatg_mp20_dng.yaml",
}


def _load_small_model_config(kind):
    cfg = OmegaConf.load(_CONFIGS[kind])
    model_cfg = OmegaConf.to_container(cfg["Model"], resolve=True)
    params = model_cfg["__init_params__"]
    params["decoder_cfg"].update(hidden_dim=32, num_layers=1, time_embed_dim=16)
    params["si_scheduler_cfg"]["integration_time_steps"] = 10
    return model_cfg


def _build_small_model(kind):
    with paddle.utils.unique_name.guard():
        paddle.seed(2026)
        return build_model(_load_small_model_config(kind))


def _make_batch(atoms_per_struct=(3, 4), max_z=10):
    batch_size = len(atoms_per_struct)
    total = sum(atoms_per_struct)
    return {
        "structure_array": {
            "num_atoms": paddle.to_tensor(atoms_per_struct, dtype="int64"),
            "atom_types": paddle.randint(1, max_z, [total], dtype="int64"),
            "frac_coords": paddle.rand([total, 3]),
            "lattice": paddle.rand([batch_size, 3, 3]) * 2.0
            + paddle.eye(3).unsqueeze(0),
        }
    }


def _train_one_step(model, batch):
    optimizer = paddle.optimizer.Adam(learning_rate=1e-4, parameters=model.parameters())
    output = model(batch)
    loss = output["loss_dict"]["loss"]
    loss_val = float(loss)
    assert np.isfinite(loss_val), "training loss is not finite"
    loss.backward()
    for parameter in model.parameters():
        if not parameter.stop_gradient:
            assert parameter.grad is not None
            assert np.isfinite(parameter.grad.numpy()).all()
    optimizer.step()
    assert all(np.isfinite(p.numpy()).all() for p in model.parameters())
    optimizer.clear_grad()
    return loss_val


def _crystal_dict(cell, atom_types, frac_coords):
    return {
        "lengths": [cell, cell, cell],
        "angles": [90.0, 90.0, 90.0],
        "atom_types": atom_types,
        "frac_coords": frac_coords,
    }


class TestOMatGFlows(unittest.TestCase):
    """YAML -> model -> forward/sample and one training step."""

    def test_yaml_build_forward_and_sample(self):
        for kind in _CONFIGS:
            with self.subTest(kind=kind):
                model = _build_small_model(kind)
                self.assertIsInstance(model, OMatG)
                self.assertEqual(model.supports_num_atoms_sampling, kind == "dng")

                batch = _make_batch()
                loss_dict = model(batch)["loss_dict"]
                self.assertIn("pos_loss_b", loss_dict)
                self.assertIn("cell_loss_b", loss_dict)
                self.assertFalse(np.isnan(float(loss_dict["loss"])))
                if kind == "dng":
                    self.assertIn("species_loss", loss_dict)

                result = model.sample(batch, num_inference_steps=10)
                self.assertEqual(len(result["result"]), 2)
                for entry in result["result"]:
                    self.assertEqual(
                        set(entry),
                        {"num_atoms", "atom_types", "frac_coords", "lattice"},
                    )
                    coords = np.asarray(entry["frac_coords"])
                    self.assertTrue(np.isfinite(coords).all())
                    self.assertTrue((coords >= 0.0).all() and (coords < 1.0).all())
                    self.assertTrue(min(entry["atom_types"]) >= 1)

    def test_training_step_then_sampling(self):
        for kind in _CONFIGS:
            with self.subTest(kind=kind):
                model = _build_small_model(kind)
                batch = _make_batch()
                loss_val = _train_one_step(model, batch)
                self.assertFalse(np.isnan(loss_val))

                result = model.sample(batch, num_inference_steps=5)
                self.assertEqual(len(result["result"]), 2)
                for entry in result["result"]:
                    self.assertTrue(np.isfinite(entry["frac_coords"]).all())

    def test_runtime_dispatch_preserves_outputs(self):
        calls = set()

        def dispatch(model, boundary, function, *args, **kwargs):
            calls.add((model.execution_backend, boundary))
            return function(*args, **kwargs)

        for kind in _CONFIGS:
            batch = _make_batch()
            outputs = []
            for backend in ("eager", "cinn"):
                cfg = _load_small_model_config(kind)
                cfg["__init_params__"]["execution_backend"] = backend
                paddle.seed(2026)
                model = build_model(cfg)
                with patch.object(OMatG, "_run_runtime", dispatch):
                    paddle.seed(42)
                    loss = float(model(batch)["loss_dict"]["loss"])
                    model.eval()
                    sample = model.sample(batch, num_inference_steps=5)
                outputs.append((loss, sample))
            self.assertEqual(outputs[0], outputs[1])
        self.assertEqual(calls, {("eager", "denoise_step"), ("cinn", "denoise_step")})


class TestDataFlow(unittest.TestCase):
    """CSV -> MP20Dataset -> DefaultCollator -> model."""

    def test_csv_dataset_to_model(self):
        rng = np.random.RandomState(5)
        cifs = []
        for n, species in ((2, [3, 8]), (3, [11, 8, 3]), (4, [8, 8, 3, 11])):
            structure = Structure(
                lattice=Lattice.from_parameters(4.0, 4.0, 4.0, 90.0, 90.0, 90.0),
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
                    "canocial": False,
                    "num_cpus": 1,
                },
                cache_path=str(Path(tmp) / "cache"),
            )
            samples = [dataset[i] for i in range(len(dataset))]

        batch = DefaultCollator()(samples)
        structure_array = batch["structure_array"]
        self.assertEqual(len(structure_array["num_atoms"]), 3)
        self.assertEqual(len(structure_array["atom_types"]), 9)

        model = _build_small_model("csp")
        loss_val = float(model(batch)["loss_dict"]["loss"])
        self.assertFalse(np.isnan(loss_val))


class TestMetrics(unittest.TestCase):
    """CSP matching and de novo generation metric entry points."""

    @staticmethod
    def _write_gt_csv(csv_path, ref):
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

    def test_csp_metric(self):
        ref = [
            _crystal_dict(5.0, [11, 17], [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]),
            _crystal_dict(4.0, [3, 9], [[0.0, 0.0, 0.0], [0.3, 0.2, 0.1]]),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "gt.csv"
            self._write_gt_csv(csv_path, ref)
            metrics = CSPMetric(gt_file_path=str(csv_path))([ref[0], ref[1]])

        self.assertEqual(set(metrics), {"match_rate", "rms_dist"})
        self.assertAlmostEqual(metrics["match_rate"], 1.0)

    def test_dng_metric(self):
        ref = [
            _crystal_dict(5.0, [11, 17], [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]),
            _crystal_dict(5.0, [3, 9], [[0.0, 0.0, 0.0], [0.3, 0.0, 0.0]]),
        ]
        invalid = dict(ref[0], lengths=[0.0, 0.0, 0.0])
        overlapping = dict(ref[1], frac_coords=[[0.0, 0.0, 0.0], [0.01, 0.0, 0.0]])
        gen = [ref[0], invalid, overlapping, ref[0]]

        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "gt.csv"
            self._write_gt_csv(csv_path, ref)
            metric = DNGMetric(gt_file_path=str(csv_path))
            metrics = metric(gen)
            metric.update_step(result={"result": gen[:2]}, batch={}, stage="sample")
            metric.update_step(result={"result": gen[2:]}, batch={}, stage="sample")
            self.assertEqual(metrics, metric.compute_epoch(stage="sample"))
            metric.update_step(result={"loss_dict": {}}, batch={}, stage="eval")
            self.assertEqual(metric.compute_epoch(stage="eval"), {})
            metric.update_step(result={"result": [ref[0]]}, batch={}, stage="eval")
            self.assertAlmostEqual(
                metric.compute_epoch(stage="eval")["valid_rate"], 1.0
            )
            self.assertEqual(metrics, metric.compute_epoch(stage="sample"))
            cached_reference = metric._gt_crystals
            metric.reset()
            self.assertIs(metric._gt_crystals, cached_reference)
            self.assertEqual(metric.compute_epoch(stage="sample"), {})
        expected = {
            "valid_rate",
            "composition_valid_rate",
            "structural_valid_rate",
            "metre_rate",
            "cov_precision",
            "cov_recall",
            "wdist_density",
            "dng_eval",
        }
        self.assertTrue(expected <= set(metrics))
        for value in metrics.values():
            self.assertFalse(np.isnan(value))
        self.assertAlmostEqual(metrics["valid_rate"], 0.5)
        self.assertAlmostEqual(metrics["structural_valid_rate"], 0.5)
        self.assertAlmostEqual(metrics["cov_precision"], 0.5)
        self.assertAlmostEqual(metrics["cov_recall"], 0.5)
        self.assertAlmostEqual(metrics["metre_rate"], 1.0)
        self.assertAlmostEqual(metrics["metre_valid_rate"], 0.5)

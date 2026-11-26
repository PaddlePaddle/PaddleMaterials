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

"""SGEQuiDiff integration tests.
"""

import os
from pathlib import Path

import numpy as np
import paddle
import pytest

from ppmat.metrics.streaming_base import StreamingMetricBase
from ppmat.models.sgequidiff.asu import build_embedding_tools
from ppmat.models.sgequidiff.asu import build_wyckoff_geometry
from ppmat.models.sgequidiff.sgequidiff import EquivariantDiffusionModel
from ppmat.models.sgequidiff.sgequidiff import SGEQuiDiff
from ppmat.utils.execution import configure_execution_backend


def _make_synthetic_batch(batch_size=2, atoms_per_crystal=2):
    """Create a synthetic batch dict with valid P1 (space group 1) data."""
    total = batch_size * atoms_per_crystal
    return {
        "frac_coords": paddle.rand([total, 3]),
        "element_indices": paddle.to_tensor([1, 2] * batch_size, dtype=paddle.int64),
        "wyckoff_indices": paddle.to_tensor([0] * total, dtype=paddle.int64),
        "space_group_indices": paddle.to_tensor([0] * batch_size, dtype=paddle.int64),
        "n_atoms_per_asu": paddle.to_tensor(
            [atoms_per_crystal] * batch_size, dtype=paddle.int64
        ),
        "wyckoff_shape_indices": paddle.to_tensor([0] * total, dtype=paddle.int64),
        "lattice_lengths": paddle.to_tensor(
            [[5.0, 5.0, 5.0]] * batch_size, dtype=paddle.float32
        ),
        "lattice_angles": paddle.to_tensor(
            [[90.0, 90.0, 90.0]] * batch_size, dtype=paddle.float32
        ),
    }


def _make_mixed_space_group_batch(batch_size=3):
    """Synthetic batch with multiple space groups (P1, P2, Pm)."""
    sg_indices = paddle.to_tensor([0, 3, 5], dtype=paddle.int64)[:batch_size]
    n_crystals = sg_indices.shape[0]
    n_atoms = 2
    total = n_crystals * n_atoms
    return {
        "space_group_indices": sg_indices,
        "lattice_lengths": paddle.full([n_crystals, 3], 5.0),
        "lattice_angles": paddle.full([n_crystals, 3], 90.0),
        "n_atoms_per_asu": paddle.full([n_crystals], n_atoms, dtype=paddle.int64),
        "element_indices": paddle.to_tensor([1, 2] * n_crystals, dtype=paddle.int64),
        "wyckoff_indices": paddle.zeros([total], dtype=paddle.int64),
        "wyckoff_shape_indices": paddle.zeros([total], dtype=paddle.int64),
        "frac_coords": paddle.rand([total, 3]),
    }


def _make_sgequidiff(num_timesteps=2):
    """Build a lightweight SGEQuiDiff for fast integration tests."""
    return SGEQuiDiff(
        dataset_name="mp_20",
        num_timesteps=num_timesteps,
        noise_scheduler_num_monte_carlo_samples=2,
        num_lattice_translations=1,
    )


@pytest.fixture(scope="module")
def model():
    """Lightweight MLP EquivariantDiffusionModel, shared across the class.

    Kept small (10 timesteps, tiny embeddings) so the forward/loss checks
    run quickly while still exercising the real score-matching path.
    """
    wyckoff_geometry = build_wyckoff_geometry()
    embedding_tools = build_embedding_tools()
    return EquivariantDiffusionModel(
        model_type="mlp",
        num_timesteps=10,
        num_lattice_translations=1,
        noise_scheduler_num_monte_carlo_samples=10,
        time_emb_dim=32,
        num_plane_wave_freqs=16,
        wyckoff_geometry=wyckoff_geometry,
        embedding_tools=embedding_tools,
    )


class TestModelBuildForward:
    """Build + forward + loss validity of the coordinate diffusion model.

    Overall question: can the model be constructed from a light config and
    complete one training forward pass that yields a valid scalar loss?
    """

    def test_build_from_config_produces_valid_output(self, model):
        """Forward pass returns a scalar, finite, positive loss."""
        batch_data = _make_synthetic_batch()
        output = model(batch_data)

        assert "loss_dict" in output
        loss = output["loss_dict"]["loss"]
        assert loss.ndim == 0
        assert not paddle.isnan(loss).item()
        assert float(loss) > 0.0


class TestSGEQuiDiffEndToEnd:
    """SGEQuiDiff end-to-end: build -> sample / forward.

    Overall questions: does unconditional sampling produce a complete,
    internally-consistent crystal result, and does the full model forward
    produce a valid loss?
    """

    @pytest.fixture(scope="class")
    def sampler(self, tmp_path_factory):
        from ppmat.models.sgequidiff import sgequidiff

        paddle.seed(0)
        cache = tmp_path_factory.mktemp("sgequidiff_sampling")
        with pytest.MonkeyPatch.context() as context:
            context.setattr(sgequidiff, "get_data_directory", lambda: cache)
            return _make_sgequidiff(num_timesteps=4)

    def test_sample_returns_valid_result_format(self, sampler, monkeypatch, tmp_path):
        """Unconditional sampling yields crystals with self-consistent fields.

        Each generated structure must report a positive atom count whose
        atom_types / frac_coords match, plus 3 lattice lengths and angles.
        """
        batch_data = {
            "structure_array": {
                "num_atoms": paddle.to_tensor([2], dtype=paddle.int64),
            }
        }
        time_steps = []
        time_embedder = sampler.atom_coord_diffusion_model.time_embedder
        hook = time_embedder.register_forward_pre_hook(
            lambda layer, args: time_steps.append(float(args[0].item()))
        )
        out = sampler.sample(batch_data, num_inference_steps=2)
        hook.remove()
        assert time_steps == [3.0, 1.0]
        scheduler = sampler.atom_coord_diffusion_model.noise_scheduler
        np.testing.assert_array_equal(scheduler.timesteps, [4, 1])
        with pytest.raises(ValueError, match="num_inference_steps"):
            sampler.sample(batch_data, num_inference_steps=1)
        assert len(out["result"]) == 1
        assert "result" in out
        for crystal in out["result"]:
            from ppmat.datasets.build_structure import BuildStructure

            structure = BuildStructure.build_one(
                crystal, "array", niggli=False, canocial=False
            )
            assert len(structure) == crystal["num_atoms"]
            assert np.asarray(crystal["lattice"]).shape == (3, 3)
            assert crystal["num_atoms"] > 0
            assert len(crystal["atom_types"]) == crystal["num_atoms"]
            assert len(crystal["frac_coords"]) == crystal["num_atoms"]
            assert len(crystal["lengths"]) == 3
            assert len(crystal["angles"]) == 3
            for pt in crystal["frac_coords"]:
                assert len(pt) == 3
        from ppmat.sampler.structure_sampler import StructureSampler

        workflow = StructureSampler.__new__(StructureSampler)
        workflow.model = sampler
        workflow.sample_config = {}
        with pytest.raises(NotImplementedError):
            workflow.sample_by_num_atoms(3)
        with pytest.raises(NotImplementedError, match="chemical formula"):
            workflow.sample_by_chemical_formula("Li2O")
        with pytest.raises(NotImplementedError, match="not a conditional model"):
            workflow.sample_by_condition(3, {"band_gap": 1.0})

        crystal = {
            "space_group_number": paddle.to_tensor(1, dtype=paddle.int64),
            "conventional_lattice_lengths": paddle.to_tensor(
                [[5.0, 5.0, 5.0]], dtype=paddle.float32
            ),
            "conventional_lattice_angles": paddle.to_tensor(
                [[90.0, 90.0, 90.0]], dtype=paddle.float32
            ),
            "element_indices": paddle.to_tensor([0, 1, 2], dtype=paddle.int64),
            "wyckoff_indices": paddle.to_tensor([0, 0, 0], dtype=paddle.int64),
            "conventional_frac_coords": paddle.to_tensor(
                [[0.1, 0.2, 0.3], [0.3, 0.4, 0.5], [0.6, 0.7, 0.8]],
                dtype=paddle.float32,
            ),
        }

        def _fake_sample_crystal(
            batch_size, diffusion_snr=0.4, temperature=1.0, **kwargs
        ):
            return [crystal]

        monkeypatch.setattr(sampler, "sample_crystal", _fake_sample_crystal)
        out = sampler.sample(
            {
                "structure_array": {
                    "num_atoms": paddle.to_tensor([1], dtype=paddle.int64)
                }
            }
        )
        assert len(out["result"]) == 1
        atom_types = out["result"][0]["atom_types"]
        assert 1 in atom_types, f"H (Z=1) was filtered out: {atom_types}"
        assert set(atom_types) == {1, 2, 3}
        from ppmat.models.sgequidiff import asu as build_asu

        monkeypatch.setattr(build_asu.download, "DATASETS_HOME", str(tmp_path))
        cold_shapes = build_asu.load_wyckoff_shapes(sampler.wyckoff_geometry)
        warm_shapes = build_asu.load_wyckoff_shapes(sampler.wyckoff_geometry)
        assert cold_shapes["1"]["a"]["dim"] == 3
        np.testing.assert_array_equal(
            cold_shapes["1"]["a"]["volumes"], warm_shapes["1"]["a"]["volumes"]
        )


class TestFullTrainingObjective:
    """Full training objective: MLE on discrete parts + score matching.

    Overall questions: can the model be trained end-to-end (forward +
    backward with gradients flowing to all four submodules), does the loss
    actually decrease, and do sampled structures stay physically valid?
    """

    @pytest.fixture(scope="class")
    def sampler(self):
        paddle.seed(0)
        return _make_sgequidiff()

    def test_mixed_space_group_batch_trains(self, sampler):
        """Training on mixed space groups: forward + backward propagate
        gradients into every trainable submodule."""
        sampler.train()
        batch = _make_mixed_space_group_batch()
        out = sampler(batch)
        loss = out["loss_dict"]["loss"]
        assert not paddle.isnan(loss).item()
        pred = out["pred_dict"]
        for key in (
            "space_group_log_prob",
            "lattice_log_prob",
            "elements_log_prob",
            "wyckoffs_log_prob",
            "termination_log_prob",
            "score_matching_loss",
        ):
            assert key in pred, f"missing artifact: {key}"
        loss.backward()
        for name, p in sampler.named_parameters():
            if not p.stop_gradient:
                assert p.grad is not None, name
            if p.grad is not None and float(p.grad.abs().sum()) > 0:
                assert name.split(".")[0] in (
                    "atom_coord_diffusion_model",
                    "space_group_sampler",
                    "lattice_sampler",
                    "wyckoff_and_element_sampler",
                )
        from ppmat.optimizer import build_optimizer

        optimizer, _ = build_optimizer(
            {
                "__class_name__": "AdamW",
                "__init_params__": {"lr": 1e-3, "weight_decay": 0.0},
            },
            sampler,
            epochs=1,
            iters_per_epoch=1,
        )
        selected = {}
        for name, parameter in sampler.named_parameters():
            root = name.split(".")[0]
            if parameter.grad is not None:
                assert bool(paddle.isfinite(parameter.grad).all())
                if root not in selected and float(parameter.grad.abs().max()) > 1e-3:
                    selected[root] = (parameter, parameter.numpy().copy())
        assert len(selected) == 4
        optimizer.step()
        for root, (parameter, before) in selected.items():
            expected = (
                1e-4
                if root in ("lattice_sampler", "wyckoff_and_element_sampler")
                else 1e-3
            )
            assert float(np.abs(parameter.numpy() - before).max()) == pytest.approx(
                expected, rel=1e-3, abs=1e-7
            )
        assert all(
            bool(paddle.isfinite(parameter).all()) for parameter in sampler.parameters()
        )
        optimizer.clear_grad()

    def test_training_loss_decreases_over_steps(self):
        """Gradient descent drives the total training loss down."""
        batch = _make_mixed_space_group_batch()
        n_steps = 40
        window = 5
        for seed in (0, 1, 2):
            paddle.seed(seed)
            sampler = _make_sgequidiff()
            sampler.train()
            opt = paddle.optimizer.Adam(
                parameters=sampler.parameters(), learning_rate=1e-3
            )
            losses = []
            for _ in range(n_steps):
                out = sampler(batch)
                loss = out["loss_dict"]["loss"]
                losses.append(float(loss))
                opt.clear_grad()
                loss.backward()
                opt.step()
            initial = sum(losses[:window]) / window
            best = min(
                sum(losses[i : i + window]) / window
                for i in range(n_steps - window + 1)
            )
            assert best < initial, (
                f"seed {seed}: loss did not improve "
                f"(initial={initial:.2f}, best={best:.2f})"
            )


def _write_crystal_csv(root_dir, name):
    """Write CIFs spanning general and special Wyckoff positions."""
    import pandas as pd
    from pymatgen.core import Lattice
    from pymatgen.core import Structure

    structures = [
        Structure(
            Lattice.from_parameters(4.1, 5.2, 6.3, 77, 83, 101),
            ["Na", "Cl", "K"],
            [[0.13, 0.27, 0.38], [0.41, 0.53, 0.62], [0.73, 0.19, 0.84]],
        ),
        Structure.from_spacegroup(
            225, Lattice.cubic(5.64), ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]]
        ),
        Structure.from_spacegroup(227, Lattice.cubic(5.43), ["Si"], [[0, 0, 0]]),
        Structure.from_spacegroup(
            152,
            Lattice.hexagonal(4.91, 5.4),
            ["Si", "O"],
            [[0.4697, 0, 0], [0.4133, 0.2672, 0.1188]],
        ),
        Structure.from_spacegroup(
            6,
            Lattice.monoclinic(4, 5, 6, 101),
            ["Na", "Cl"],
            [[0.1, 0, 0.2], [0.37, 0, 0.63]],
        ),
    ]
    root = Path(root_dir) / name
    root.mkdir(parents=True)
    pd.DataFrame(
        {
            "material_id": [f"crystal-{i}" for i in range(len(structures))],
            "formation_energy_per_atom": np.arange(len(structures), dtype=float),
            "cif": [structure.to(fmt="cif") for structure in structures],
        }
    ).to_csv(root / "train.csv", index=False)
    return root


@pytest.mark.parametrize("dataset_class", ["MP20Dataset", "MPTS52Dataset"])
def test_dataset_collate_to_model_forward(tmp_path, monkeypatch, dataset_class):
    """CSV -> base dataset -> ASU transform -> default collator -> model loss."""
    import copy

    from omegaconf import OmegaConf

    import ppmat.datasets as datasets
    from ppmat.datasets import build_dataloader
    from ppmat.datasets.build_structure import BuildStructure
    from ppmat.models.sgequidiff import asu
    from ppmat.models.sgequidiff.sgequidiff import expand_asu
    from ppmat.utils import download

    cls = getattr(datasets, dataset_class)
    model_name = "mp20" if dataset_class == "MP20Dataset" else "mpts_52"
    extract_root = tmp_path / "download"
    root = _write_crystal_csv(extract_root, cls.name)
    calls = []

    def fake_download(url, md5sum=None):
        calls.append((url, md5sum))
        return str(extract_root)

    monkeypatch.setattr(download, "get_datasets_path_from_url", fake_download)
    config = OmegaConf.to_container(
        OmegaConf.load(
            f"structure_generation/configs/sgequidiff/sgequidiff_{model_name}.yaml"
        ),
        resolve=True,
    )["Dataset"]["train"]
    params = config["dataset"]["__init_params__"]
    params.update(
        path=str(tmp_path / "missing" / "train.csv"),
        cache_path=str(tmp_path / "structures"),
        property_names=["cif", "formation_energy_per_atom"],
    )
    params["transforms"][0]["__init_params__"]["cache_dir"] = str(tmp_path / "asu")
    config["sampler"]["__init_params__"].update(
        batch_size=5, shuffle=False, drop_last=False
    )
    loader = build_dataloader(copy.deepcopy(config))
    assert isinstance(loader.dataset, datasets.MP20Dataset)
    assert calls == [(cls.url, cls.md5)]
    assert loader.dataset.path == str(root / "train.csv")
    batch = next(iter(loader))
    assert "structure_array" in batch
    np.testing.assert_array_equal(batch["space_group_indices"], [0, 224, 226, 151, 5])
    np.testing.assert_array_equal(
        batch["formation_energy_per_atom"].flatten(), range(5)
    )
    assert len(set(batch["n_atoms_per_asu"].numpy().tolist())) > 1
    transform = loader.dataset.transforms.transforms[0]
    cache_files = list(transform.cache_dir.glob("*.pkl"))
    assert len(cache_files) == 5
    with monkeypatch.context() as context:

        def fail_convert(*args, **kwargs):
            raise AssertionError("A warm cache must not recompute symmetry")

        context.setattr(asu.AsymmetricUnitConverter, "__call__", fail_convert)
        warm = next(iter(loader))
    np.testing.assert_array_equal(warm["frac_coords"], batch["frac_coords"])
    cache_files[0].write_bytes(b"invalid cache")
    rebuilt = next(iter(loader))
    np.testing.assert_array_equal(rebuilt["frac_coords"], batch["frac_coords"])
    from pymatgen.analysis.structure_matcher import StructureMatcher

    geometry = build_wyckoff_geometry()
    for index in range(len(loader.dataset)):
        sample = loader.dataset[index]
        expanded = expand_asu(
            {
                "space_group_number": sample["space_group_indices"] + 1,
                "element_indices": paddle.to_tensor(sample["element_indices"].data),
                "conventional_frac_coords": paddle.to_tensor(
                    sample["frac_coords"].data
                ),
                "conventional_lattice_lengths": paddle.to_tensor(
                    sample["lattice_lengths"]
                ),
                "conventional_lattice_angles": paddle.to_tensor(
                    sample["lattice_angles"]
                ),
            },
            geometry,
        )
        parsed = BuildStructure.build_one(
            expanded, "array", niggli=False, canocial=False
        )
        array = sample["structure_array"]
        original = BuildStructure.build_one(
            {
                "atom_types": array["atom_types"].data,
                "frac_coords": array["frac_coords"].data,
                "lengths": array["lengths"].data.reshape(3),
                "angles": array["angles"].data.reshape(3),
            },
            "array",
            niggli=False,
            canocial=False,
        )
        assert StructureMatcher().fit(original, parsed)
    sampler = _make_sgequidiff()
    sampler.train()
    out = sampler(batch)
    loss = out["loss_dict"]["loss"]
    assert loss.ndim == 0 and bool(paddle.isfinite(loss))
    loss.backward()
    assert any(parameter.grad is not None for parameter in sampler.parameters())
    for key, value in out["label_dict"].items():
        np.testing.assert_array_equal(value, batch[key])


def test_sgequidiff_metric(tmp_path, monkeypatch):
    """Evaluation pipeline: generation-quality metric over synthetic structures.

    Overall question: does CrystalGenerationMetric return the complete expected
    metric set (validity / uniqueness / novelty / coverage / distances)
    with values in valid ranges? Uses self-contained synthetic CIFs written
    to ``tmp_path``, so the test runs without external data files.
    """
    import pandas as pd
    from omegaconf import OmegaConf
    from pymatgen.core import Lattice
    from pymatgen.core import Structure

    from ppmat.metrics import build_metric

    structures = [
        Structure(
            Lattice.from_parameters(5.0, 5.0, 5.0, 90.0, 90.0, 90.0),
            ["Si", "Si"],
            [[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]],
        ),
        Structure(
            Lattice.from_parameters(4.0, 4.0, 4.0, 90.0, 90.0, 90.0),
            ["C", "C"],
            [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
        ),
        Structure(
            Lattice.from_parameters(3.0, 3.0, 3.0, 90.0, 90.0, 90.0),
            ["Na", "Cl"],
            [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
        ),
        Structure(
            Lattice.from_parameters(6.0, 6.0, 6.0, 90.0, 90.0, 90.0),
            ["Fe", "Fe"],
            [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
        ),
    ]
    cifs = [structure.to(fmt="cif") for structure in structures]
    gt_csv = tmp_path / "gt.csv"
    pd.DataFrame({"cif": cifs}).to_csv(gt_csv, index=False)

    pred_dicts = [
        {
            "num_atoms": len(structure),
            "atom_types": list(structure.atomic_numbers),
            "frac_coords": structure.frac_coords.tolist(),
            "lattice": structure.lattice.matrix,
        }
        for structure in structures
    ]
    # The first structure is repeated across "batches": it must be counted
    # once for uniqueness, so the streaming accumulation (which spreads the
    # duplicate over two steps) matches the batch interface.
    invalid = {**pred_dicts[0], "frac_coords": [[0, 0, 0], [0, 0, 0]]}
    duplicated = pred_dicts + [pred_dicts[0], invalid]
    cfg = OmegaConf.to_container(
        OmegaConf.load("structure_generation/configs/sgequidiff/sgequidiff_mp20.yaml"),
        resolve=True,
    )["Sample"]["metrics"]
    cfg["__init_params__"]["gt_file_path"] = str(gt_csv)
    cfg["__init_params__"]["coverage_cfg"]["distance_batch_size"] = 2
    metric = build_metric(cfg)
    result = metric(duplicated)

    for key in (
        "validity",
        "uniqueness",
        "novelty",
        "cov_recall",
        "cov_precision",
        "amsd_recall",
        "amsd_precision",
        "amcd_recall",
        "amcd_precision",
    ):
        assert key in result, f"missing metric: {key}"
    assert result["validity"] == pytest.approx(5 / 6)
    assert result["uniqueness"] == pytest.approx(4 / 5)
    assert result["novelty"] == 0.0
    assert result["cov_recall"] == 1.0
    assert result["cov_precision"] == 1.0

    basic = build_metric(
        {
            "__class_name__": "CrystalGenerationMetric",
            "__init_params__": {},
        }
    )
    assert basic(duplicated) == {
        "validity": pytest.approx(5 / 6),
        "uniqueness": pytest.approx(4 / 5),
    }
    offline = build_metric(
        {
            "__class_name__": "CrystalGenerationMetric",
            "__init_params__": {"input_format": "cif_str"},
        }
    )
    assert offline(cifs) == {"validity": 1.0, "uniqueness": 1.0}
    invalid_result = metric([invalid])
    assert invalid_result["validity"] == 0.0
    assert invalid_result["cov_recall"] == 0.0
    assert np.isfinite(invalid_result["amsd_recall"])

    novelty_csv = tmp_path / "novelty.csv"
    pd.DataFrame({"cif": cifs[:2]}).to_csv(novelty_csv, index=False)
    cfg["__init_params__"]["novelty_gt_file_path"] = str(novelty_csv)
    separate = build_metric(cfg)(duplicated)
    assert separate["novelty"] == pytest.approx(2 / 5)
    assert separate["cov_recall"] == 1.0
    del cfg["__init_params__"]["novelty_gt_file_path"]

    from ppmat.metrics.utils import CrystalNNFP

    def fail_fingerprint(*args, **kwargs):
        raise ValueError("Unavailable structural fingerprint")

    with monkeypatch.context() as context:
        context.setattr(CrystalNNFP, "featurize", fail_fingerprint)
        unavailable = metric(pred_dicts)
    assert unavailable["validity"] == 1.0
    assert unavailable["cov_recall"] == 0.0
    assert np.isnan(unavailable["amsd_recall"])

    # Streaming contract: per-step accumulation must match the batch interface.
    assert isinstance(metric, StreamingMetricBase)

    stream_metric = build_metric(cfg)
    assert stream_metric.compute_epoch(stage="sample") == {}
    assert stream_metric.compute_epoch(stage="eval") == {}
    stream_metric.update_step(
        result={"samples": {"result": pred_dicts[:4]}}, batch=None, stage="sample"
    )
    stream_metric.update_step(
        result={"result": duplicated[4:]},
        batch=None,
        stage="sample",
    )
    stream_metric.update_step(result={"result": duplicated}, batch=None, stage="eval")
    streamed = stream_metric.compute_epoch(stage="sample")
    for key, value in result.items():
        assert streamed[key] == pytest.approx(value)
    stream_metric.reset()
    assert stream_metric.compute_epoch(stage="sample") == {}


def test_trainer_checkpoint_resume(tmp_path):
    """Resume model, optimizer moments, scheduler and Trainer step counters."""
    from omegaconf import OmegaConf

    from ppmat.optimizer.optimizer import AdamW
    from ppmat.trainer.base_trainer import BaseTrainer

    def make_trainer(name, epochs):
        with paddle.utils.unique_name.guard():
            paddle.seed(42)
            model = _make_small_gnn_sgequidiff()
            scheduler = paddle.optimizer.lr.StepDecay(1e-3, step_size=1, gamma=0.5)
            scheduler.by_epoch = False
            optimizer = AdamW(learning_rate=scheduler, weight_decay=0.0)(model)
        config = OmegaConf.to_container(
            OmegaConf.load(
                "structure_generation/configs/sgequidiff/sgequidiff_mp20.yaml"
            )
        )["Trainer"]
        config.update(
            max_epochs=epochs,
            output_dir=str(tmp_path / name),
            save_freq=1,
            pretrained_model_path=None,
            use_tensorboard=False,
            zero_nan_grad=False,
        )
        return BaseTrainer(
            config,
            model,
            train_dataloader=[_make_mixed_space_group_batch()],
            optimizer=optimizer,
            lr_scheduler=scheduler,
        )

    def assert_equal(current, expected):
        if isinstance(expected, dict):
            assert set(current) == set(expected)
            for key in expected:
                assert_equal(current[key], expected[key])
        elif isinstance(expected, (paddle.Tensor, np.ndarray)):
            if isinstance(current, paddle.Tensor):
                current = current.numpy()
            if isinstance(expected, paddle.Tensor):
                expected = expected.numpy()
            np.testing.assert_array_equal(current, expected)
        else:
            assert current == expected

    uninterrupted = make_trainer("uninterrupted", 2)
    with paddle.utils.unique_name.guard():
        uninterrupted.train()
    interrupted = make_trainer("interrupted", 1)
    with paddle.utils.unique_name.guard():
        interrupted.train()
    checkpoint = tmp_path / "interrupted" / "checkpoints" / "latest"
    saved_model = paddle.load(str(checkpoint) + ".pdparams")
    saved_optimizer = paddle.load(str(checkpoint) + ".pdopt")
    assert Path(str(checkpoint) + ".pdstates").is_file()
    saved_lr = interrupted.optimizer.get_lr()
    saved_scheduler = interrupted.lr_scheduler.state_dict()
    resumed = make_trainer("resumed", 2)
    step = resumed.optimizer.step
    observed = []

    def observe_first_step():
        if not observed:
            assert_equal(resumed.model.state_dict(), saved_model)
            assert_equal(resumed.optimizer.state_dict(), saved_optimizer)
            assert resumed.optimizer.get_lr() == saved_lr
            assert_equal(resumed.lr_scheduler.state_dict(), saved_scheduler)
            assert resumed.state.global_step == interrupted.state.global_step + 1
            observed.append(True)
        step()

    resumed.optimizer.step = observe_first_step
    with paddle.utils.unique_name.guard():
        resumed.train(resume_from_checkpoint=str(checkpoint))
    assert observed
    assert resumed.state.epoch == uninterrupted.state.epoch == 2
    assert resumed.state.global_step == uninterrupted.state.global_step == 2
    assert_equal(
        resumed.lr_scheduler.state_dict(), uninterrupted.lr_scheduler.state_dict()
    )
    assert set(resumed.model.state_dict()) == set(uninterrupted.model.state_dict())
    assert set(resumed.optimizer.state_dict()) == set(
        uninterrupted.optimizer.state_dict()
    )


def test_execution_backend_protocol():
    """Runtime protocol: backend selection is owned by the diffusion model.

    Overall question: does the SGEQuiDiff <-> EquivariantDiffusionModel
    delegation honor the framework execution-backend contract (backend
    propagation, eager fallback, invalid input rejection, AMP/world-size
    constraints) without changing any state_dict key?
    """
    model = _make_sgequidiff()
    boundary = type(model.atom_coord_diffusion_model)._runtime_denoise_step
    assert callable(boundary) and hasattr(boundary, "__wrapped__")
    state_dict_before = {k: tuple(v.shape) for k, v in model.state_dict().items()}

    assert configure_execution_backend(model, None, owner="Trainer") == "eager"
    model.set_execution_backend("cinn")
    assert model.execution_backend == "cinn"
    assert model.atom_coord_diffusion_model.execution_backend == "cinn"
    model.set_runtime_options({"cinn": {"full_graph": False}})
    state_dict_after = {k: tuple(v.shape) for k, v in model.state_dict().items()}
    assert state_dict_after == state_dict_before, "state_dict changed on backend switch"

    model.set_execution_backend("eager")
    assert model.atom_coord_diffusion_model.execution_backend == "eager"

    with pytest.raises(ValueError):
        model.set_execution_backend("not-a-backend")
    model.set_execution_backend("cinn")
    with pytest.raises(ValueError):
        model.validate_execution_backend(use_amp=True)
    with pytest.raises(ValueError):
        model.validate_execution_backend(world_size=2)


def _make_small_gnn_sgequidiff(num_timesteps=2):
    """Tiny GNN SGEQuiDiff so CINN compilation stays fast in tests."""
    return SGEQuiDiff(
        dataset_name="mp_20",
        num_timesteps=num_timesteps,
        noise_scheduler_num_monte_carlo_samples=2,
        num_lattice_translations=1,
        gnn_config={
            "num_cartesian_distance_gaussians": 8,
            "edge_hidden_dim": 16,
            "atom_hidden_dim": 16,
            "num_plane_wave_freqs": 8,
            "num_msg_pass_steps": 2,
            "use_graph_norm": False,
        },
    )


@pytest.mark.skipif(
    not (
        os.environ.get("PPMAT_RUN_CINN_WORKFLOW_TESTS") == "1"
        and paddle.is_compiled_with_cuda()
        and paddle.device.cuda.device_count() > 0
        and paddle.is_compiled_with_cinn()
    ),
    reason="requires PPMAT_RUN_CINN_WORKFLOW_TESTS=1, CUDA and CINN",
)
def test_cinn_sampling_parity(tmp_path, monkeypatch):
    """CINN workflow: compiled denoise_step matches eager sampling.

    Overall question: does sampling with ``execution_backend="cinn"`` reuse a
    single cached compiled runtime and match fixed-input eager denoising and
    the sampled coordinates within GPU float32 tolerances?
    """
    from ppmat.models.sgequidiff import sgequidiff

    monkeypatch.setattr(sgequidiff, "get_data_directory", lambda: tmp_path)
    paddle.set_device("gpu")
    try:
        paddle.seed(42)
        np.random.seed(42)
        model = _make_small_gnn_sgequidiff(num_timesteps=3)
        model.eval()
        batch = {"structure_array": {"num_atoms": paddle.to_tensor([1])}}
        diffusion = model.atom_coord_diffusion_model
        predict = diffusion.predict_equivariant_vectors
        eager_steps = []

        def record_eager_step(*args, **kwargs):
            result = predict(*args, **kwargs)
            eager_steps.append(
                (
                    tuple(value.clone() for value in args),
                    {key: value.clone() for key, value in kwargs.items()},
                    result.clone(),
                )
            )
            return result

        def sample_frac_coords():
            paddle.seed(42)
            np.random.seed(42)
            result = model.sample(dict(batch))["result"]
            return [crystal["frac_coords"] for crystal in result]

        model.set_execution_backend("eager")
        monkeypatch.setattr(diffusion, "predict_equivariant_vectors", record_eager_step)
        coords_eager = sample_frac_coords()
        monkeypatch.setattr(diffusion, "predict_equivariant_vectors", predict)

        model.set_execution_backend("cinn")
        model.set_runtime_options({"cinn": {"full_graph": False}})
        assert len(eager_steps) == 4
        for args, kwargs, expected in eager_steps:
            actual = predict(*args, **kwargs)
            np.testing.assert_allclose(
                actual.numpy(), expected.numpy(), rtol=2e-5, atol=2e-6
            )
        coords_cinn = sample_frac_coords()

        cache = model.atom_coord_diffusion_model._runtime_cache
        assert list(cache.keys()) == [("cinn", "eval", "denoise_step")]

        assert len(coords_cinn) == len(coords_eager)
        # The adaptive corrector amplifies rounding along a sampled trajectory.
        for actual, expected in zip(coords_cinn, coords_eager):
            np.testing.assert_allclose(actual, expected, rtol=0, atol=3e-5)
    finally:
        paddle.set_device("cpu")

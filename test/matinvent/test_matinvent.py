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

"""Integration tests for the MatInvent RL workflow.

Covers the workflow end to end: config to backbone, registered package
build/load/sample, dataset to diffusion loss, reward scoring, the RL loop, and
the CINN execution backend (nested runtime ownership plus boundary dispatch).
"""

import copy
import gzip
import os
import pathlib
import pickle
import tempfile
import unittest.mock as mock

import lmdb
import numpy as np
import paddle
import pandas as pd
import pytest
from ase import units
from omegaconf import OmegaConf
from paddle.framework import core
from pymatgen.core.structure import Lattice
from pymatgen.core.structure import Structure
from pymatgen.entries.computed_entries import ComputedStructureEntry
from pymatgen.io.ase import AseAtomsAdaptor

from ppmat.calculator.ase import ASECalculator
from ppmat.datasets import build_dataloader
from ppmat.datasets.build_structure import BuildStructure
from ppmat.models import MODEL_REGISTRY
from ppmat.models import build_model
from ppmat.models.diffcsp.diffcsp import DiffCSP
from ppmat.models.matinvent.dataset import RLDataset
from ppmat.models.matinvent.dataset import create_rl_dataloader
from ppmat.models.matinvent.diffcsp import MatinventDiffCSP
from ppmat.models.matinvent.mattergen import MatinventMatterGen
from ppmat.models.matinvent.qualification import _MatterSimCalculator
from ppmat.models.matinvent.rewards.reward import CALCULATOR_REGISTRY
from ppmat.models.matinvent.rewards.reward import Reward
from ppmat.models.matinvent.rl_finetune import MatinventRL
from ppmat.models.matinvent.rl_loop import MatInvent
from ppmat.models.matinvent.sampler import MatInventSampler
from ppmat.predictor.potential_predictor import PotentialPredictor
from ppmat.schedulers import scheduling_matinvent
from ppmat.schedulers.scheduling_sde_ve import d_log_p_wrapped_normal
from ppmat.trainer.base_trainer import BaseTrainer
from ppmat.utils import download
from ppmat.utils import save_load
from ppmat.utils.execution import configure_execution_backend
from ppmat.utils.model_package import resolve_model_package_dir

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_CONFIG_ROOT = _REPO_ROOT / "structure_generation" / "configs"
_TASK_CONFIG = {
    "mattergen": _CONFIG_ROOT / "matinvent" / "matinvent_mattergen.yaml",
    "diffcsp": _CONFIG_ROOT / "matinvent" / "matinvent_diffcsp.yaml",
}
# Canonical backbone architecture lives in the reference task configs.
_REFERENCE_CONFIG = {
    "mattergen": _CONFIG_ROOT / "mattergen" / "mattergen_mp20.yaml",
    "diffcsp": _CONFIG_ROOT / "diffcsp" / "diffcsp_mp20.yaml",
}
_BACKBONE_NAME = {"mattergen": "MatinventMatterGen", "diffcsp": "MatinventDiffCSP"}
_BACKBONE_CLASS = {"mattergen": MatinventMatterGen, "diffcsp": MatinventDiffCSP}
_BACKBONE_METHOD = {"mattergen": "_denoise", "diffcsp": "_decode"}
_WEIGHTS_NAME = "best.pdparams"
_DISPATCH_SEED = 2026


def _read_config(path):
    return OmegaConf.to_container(OmegaConf.load(path), resolve=True)


def _load_config(model_type):
    return _read_config(_TASK_CONFIG[model_type])


def _make_structure(species, coords, a=5.43):
    return Structure(lattice=Lattice.cubic(a), species=species, coords=coords)


def _reference_params(model_type):
    config = _read_config(_REFERENCE_CONFIG[model_type])
    return copy.deepcopy(config["Model"]["__init_params__"])


def _tiny_backbone_cfg(model_type):
    """Shrunk copy of the released backbone: same code path, fast numerics."""
    settings = _reference_params(model_type)
    if model_type == "mattergen":
        # Hidden/latent widths must stay coupled for the denoiser concat.
        settings["decoder_cfg"]["hidden_dim"] = 32
        gemnet = settings["decoder_cfg"]["gemnet_cfg"]
        gemnet["latent_dim"] = 32
        gemnet["atom_embedding_cfg"]["emb_size"] = 32
        gemnet["emb_size_atom"] = 32
        gemnet["emb_size_edge"] = 32
        gemnet["num_blocks"] = 1
        gemnet["max_neighbors"] = 12
        gemnet["cutoff"] = 5.0
    else:
        settings["time_dim"] = 32
        decoder = settings["decoder_cfg"]
        decoder["hidden_dim"] = 32
        decoder["latent_dim"] = 32
        decoder["num_layers"] = 1
        decoder["num_freqs"] = 6
    return {
        "__class_name__": _BACKBONE_NAME[model_type],
        "__init_params__": settings,
    }


def _write_model_package(directory, model_type, backbone_cfg=None, package_name=None):
    """Write a MatInvent package laid out like the MODEL_REGISTRY archives."""
    config = _load_config(model_type)
    if backbone_cfg is not None:
        config["Model"]["__init_params__"]["backbone_cfg"] = backbone_cfg
    package_name = package_name or f"test_matinvent_{model_type}"
    package_dir = pathlib.Path(directory) / package_name
    (package_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    with paddle.utils.unique_name.guard():
        paddle.seed(_DISPATCH_SEED)
        model = build_model(config["Model"])
    paddle.save(
        model.backbone().state_dict(),
        str(package_dir / "checkpoints" / _WEIGHTS_NAME),
    )
    OmegaConf.save(OmegaConf.create(config), package_dir / f"{package_name}.yaml")
    return package_name, package_dir


def _make_backbone(model_type, execution_backend="eager"):
    with paddle.utils.unique_name.guard():
        paddle.seed(_DISPATCH_SEED)
        backbone = build_model(_tiny_backbone_cfg(model_type))
    backbone.set_execution_backend(execution_backend)
    return backbone


def _clone_state_dict(model):
    return {key: value.clone() for key, value in model.state_dict().items()}


def _make_dispatch_pair(model_type):
    initial_state = _clone_state_dict(_make_backbone(model_type))
    paddle.seed(_DISPATCH_SEED)
    eager_backbone = _make_backbone(model_type)
    paddle.seed(_DISPATCH_SEED)
    cinn_backbone = _make_backbone(model_type, execution_backend="cinn")
    eager_backbone.set_state_dict(initial_state)
    cinn_backbone.set_state_dict(initial_state)
    eager_backbone.eval()
    cinn_backbone.eval()
    return eager_backbone, cinn_backbone


def _record_boundaries(names):
    def run_boundary(self, name, function, *args, **kwargs):
        names.append(name)
        return function(*args, **kwargs)

    return run_boundary


def _make_denoise_inputs(model_type, backbone, batch_size=1):
    """Deterministic boundary inputs, no dependence on sampler randomness."""
    num_atoms = paddle.to_tensor([2] * batch_size, dtype="int64")
    total_atoms = int(num_atoms.sum())
    batch_idx = paddle.repeat_interleave(paddle.arange(batch_size), repeats=num_atoms)
    lattice = paddle.eye(3).unsqueeze(0).tile([batch_size, 1, 1]) * 5.43
    frac_coords = paddle.to_tensor(
        [0.0, 0.0, 0.0, 0.25, 0.25, 0.25] * batch_size,
        dtype="float32",
    ).reshape([total_atoms, 3])
    atom_types = paddle.to_tensor([14] * total_atoms, dtype="int64")
    structure_array = {
        "frac_coords": frac_coords,
        "lattice": lattice,
        "atom_types": atom_types,
        "num_atoms": num_atoms,
        "batch": batch_idx,
    }
    if model_type == "mattergen":
        # Fractional diffusion time in (0, 1).
        return (structure_array, paddle.full([batch_size], 0.5, dtype="float32"))
    # DiffCSP._decode consumes the embedded time, matching DiffCSP.forward.
    time_emb = backbone.time_embedding(paddle.full([batch_size], 500, dtype="int64"))
    return (time_emb, atom_types - 1, frac_coords, lattice, num_atoms, batch_idx)


def _flatten_result(output):
    if isinstance(output, dict):
        for value in output.values():
            yield from _flatten_result(value)
    elif isinstance(output, (list, tuple)):
        for value in output:
            yield from _flatten_result(value)
    else:
        yield output


def _make_adapter(model_type, execution_backend="eager"):
    params = _load_config(model_type)["Model"]["__init_params__"]
    return MatinventRL(
        model_type=model_type,
        num_inference_steps=3,
        backbone_cfg=_tiny_backbone_cfg(model_type),
        rl_cfg=params["rl_cfg"],
        build_structure_cfg=params["build_structure_cfg"],
        execution_backend=execution_backend,
    )


class _FakeTrainer:
    """Minimal stand-in for BaseTrainer: ``before_train`` reads output_dir."""

    def __init__(self, output_dir):
        self.output_dir = str(output_dir)
        self.resume_from_checkpoint = None


def _patched_rl_environment(monkeypatch, tmp_path, model_type, execution):
    """Registered package, no download, isolated output, runtime hooks."""
    tmp_path = pathlib.Path(tmp_path)
    backbone_cfg = _tiny_backbone_cfg(model_type)
    package_name, _ = _write_model_package(tmp_path, model_type, backbone_cfg)
    config = _load_config(model_type)
    rl_cfg = config["Model"]["__init_params__"]["rl_cfg"]
    rl_cfg["pretrained_model_name"] = package_name
    rl_cfg["replay_cfg"] = None
    rl_cfg["rl_epoch"] = 1
    rl_cfg["save_freq"] = 1
    rl_cfg["sample_cfg"].update(
        {"batch_size": 1, "num_batches": 1, "num_inference_steps": 2}
    )
    rl_cfg["finetune_cfg"].update(
        {"batch_size": 2, "epochs": 1, "timesteps": 4, "accum_steps": 2}
    )
    config["Model"]["__init_params__"]["backbone_cfg"] = backbone_cfg
    config["Execution"] = {"backend": execution, "__init_params__": {}}
    config["Trainer"]["output_dir"] = str(tmp_path / "output")

    monkeypatch.setitem(
        MODEL_REGISTRY, package_name, f"http://localhost/{package_name}.zip"
    )
    monkeypatch.setattr(
        download, "get_weights_path_from_url", lambda url: str(tmp_path)
    )
    return config


def test_matinvent_joint_type_diffusion():
    """Joint atom-type diffusion returns a finite type loss and samples types."""
    config = _tiny_backbone_cfg("diffcsp")
    params = config["__init_params__"]
    params["decoder_cfg"]["pred_type"] = True
    params["decoder_cfg"]["smooth"] = True
    params["atom_loss_weight"] = 20.0
    params["time_embedding_offset"] = 1
    params["lattice_noise_scheduler_cfg"]["__init_params__"].update(
        {"beta_clip_min": 1e-4, "beta_clip_max": 0.9999}
    )
    with paddle.utils.unique_name.guard():
        paddle.seed(_DISPATCH_SEED)
        model = build_model(config)
    model.eval()

    num_atoms = paddle.to_tensor([1, 3], dtype="int64")
    batch = {
        "structure_array": {
            "frac_coords": paddle.to_tensor(
                [
                    [0.0, 0.0, 0.0],
                    [0.25, 0.25, 0.25],
                    [0.1, 0.1, 0.1],
                    [0.3, 0.3, 0.3],
                ],
                dtype="float32",
            ),
            "lattice": paddle.eye(3).unsqueeze(0).tile([2, 1, 1]) * 5.43,
            "atom_types": paddle.to_tensor([14, 14, 8, 8], dtype="int64"),
            "num_atoms": num_atoms,
        }
    }
    noises = []
    predictions = []
    draw_noise = paddle.randn
    decode = model._decode

    def record_noise(*args, **kwargs):
        value = draw_noise(*args, **kwargs)
        noises.append(value)
        return value

    def record_predictions(*args):
        result = decode(*args)
        predictions.extend(result)
        return result

    with mock.patch.object(
        paddle, "randn", side_effect=record_noise
    ), mock.patch.object(model, "_decode", side_effect=record_predictions):
        loss_dict = model(batch)["loss_dict"]
    expected_type_loss = np.square(predictions[2].numpy() - noises[2].numpy()).mean()
    np.testing.assert_allclose(
        loss_dict["loss_atom_type"].numpy(), expected_type_loss, rtol=1e-6
    )
    assert "loss_atom_type" in loss_dict
    assert np.isfinite(float(loss_dict["loss"]))
    assert np.isfinite(float(loss_dict["loss_atom_type"]))
    loss_dict["loss"].backward()
    for name, parameter in model.named_parameters():
        if ".prop_mlp." in name:
            assert parameter.stop_gradient
            assert parameter.grad is None
        else:
            assert not parameter.stop_gradient
            assert parameter.grad is not None, name
            assert bool(paddle.isfinite(parameter.grad).all()), name

    assert type(model.lattice_scheduler).__name__ == "MatinventDDPMScheduler"
    assert type(model.coord_scheduler).__name__ == "MatinventSdeVeScheduler"
    assert params["lattice_noise_scheduler_cfg"]["__class_name__"] == "DDPMScheduler"
    assert (
        params["coord_noise_scheduler_cfg"]["__class_name__"]
        == "ScoreSdeVeSchedulerWrapped"
    )
    scheduler = model.lattice_scheduler
    scheduler.set_timesteps(model.num_train_timesteps)
    times = np.linspace(0, 1, model.num_train_timesteps + 1)
    cosine = np.cos((times + 0.008) / 1.008 * np.pi / 2) ** 2
    betas = np.clip(1 - cosine[1:] / cosine[:-1], 1e-4, 0.9999).astype("float32")
    alphas = 1 - betas
    cumulative_alphas = np.cumprod(alphas)
    posterior_sigmas = np.sqrt(
        betas * (1 - np.r_[1.0, cumulative_alphas[:-1]]) / (1 - cumulative_alphas)
    )
    np.testing.assert_allclose(scheduler.betas.numpy(), betas, atol=1e-7)
    sigma_values = np.r_[0.0, model.coord_scheduler.discrete_sigmas.numpy()]
    norm_values = np.r_[0.0, model.coord_scheduler.discrete_sigmas_norm.numpy()]
    rng = np.random.default_rng(_DISPATCH_SEED)
    for label in (1000, 501, 2, 1):
        for shape in ((2, 3, 3), (4, model.num_classes)):
            sample, prediction, noise = [
                rng.standard_normal(shape).astype("float32") for _ in range(3)
            ]
            alpha = alphas[label - 1]
            cumulative = cumulative_alphas[label - 1]
            sigma = posterior_sigmas[label - 1]
            expected = (
                sample - (1 - alpha) / np.sqrt(1 - cumulative) * prediction
            ) / np.sqrt(alpha)
            if label > 1:
                expected += sigma * noise
            with mock.patch.object(
                scheduling_matinvent,
                "randn_tensor",
                return_value=paddle.to_tensor(noise),
            ):
                actual = scheduler.step(
                    paddle.to_tensor(prediction),
                    label - 1,
                    paddle.to_tensor(sample),
                ).prev_sample
            np.testing.assert_allclose(actual.numpy(), expected, atol=2e-6, rtol=2e-6)

    correct = model.coord_scheduler.step_correct
    predict = model.coord_scheduler.step_pred
    correct_steps, predict_steps = [], []

    def check_correct(prediction, time, sample, **kwargs):
        index = int(kwargs["timestep_index"])
        assert index == indices[len(correct_steps)]
        correct_steps.append(index)
        sigma = sigma_values[index + 1]
        step_size = model.coord_scheduler.snr * (sigma / sigma_values[1]) ** 2
        expected = sample.numpy() - step_size * prediction.numpy() * np.sqrt(
            norm_values[index + 1]
        )
        noise = kwargs["noise"]
        expected += np.sqrt(2 * step_size) * (1 if noise is None else noise.numpy())
        result = correct(prediction, time, sample, **kwargs)
        np.testing.assert_allclose(
            result.prev_sample.numpy(), expected, atol=2e-5, rtol=2e-5
        )
        return result

    def check_predict(prediction, time, sample, **kwargs):
        position = len(predict_steps)
        index = indices[position]
        previous = indices[position + 1] if position + 1 < len(indices) else -1
        assert int(kwargs["timestep_index"]) == index
        actual_previous = int(kwargs["previous_timestep_index"])
        if previous >= 0:
            assert actual_previous == previous
        else:
            assert actual_previous < 0
        predict_steps.append(index)
        sigma = sigma_values[index + 1]
        adjacent = sigma_values[previous + 1] if previous >= 0 else 0.0
        variance = sigma**2 - adjacent**2
        expected_mean = sample.numpy() - variance * prediction.numpy() * np.sqrt(
            norm_values[index + 1]
        )
        expected = expected_mean + np.sqrt(adjacent**2 * variance / sigma**2)
        result = predict(prediction, time, sample, **kwargs)
        np.testing.assert_allclose(
            result.prev_sample_mean.numpy(), expected_mean, atol=2e-5, rtol=2e-5
        )
        np.testing.assert_allclose(
            result.prev_sample.numpy(), expected, atol=2e-5, rtol=2e-5
        )
        return result

    for steps in (2, 10):
        scheduler.set_timesteps(steps)
        indices = scheduler.timesteps.numpy().tolist()
        correct_steps.clear()
        predict_steps.clear()
        with mock.patch.object(
            model.coord_scheduler, "step_correct", side_effect=check_correct
        ), mock.patch.object(
            model.coord_scheduler, "step_pred", side_effect=check_predict
        ), mock.patch.object(
            scheduling_matinvent,
            "randn_tensor",
            side_effect=lambda shape, **kwargs: paddle.ones(
                shape, dtype=kwargs.get("dtype", "float32")
            ),
        ):
            sampled = model.sample(batch, num_inference_steps=steps)["result"]
        assert correct_steps == predict_steps == indices
    assert len(sampled) == 2
    for result in sampled:
        assert len(result["atom_types"]) == result["num_atoms"]
        assert all(1 <= value <= 100 for value in result["atom_types"])


class TestModelPackage:
    """Cover config -> model, package yaml -> build -> load -> sample."""

    def test_reference_bias_survives_loading_and_sampling(self, tmp_path):
        small_cfg = _tiny_backbone_cfg("diffcsp")
        _, package_dir = _write_model_package(tmp_path, "diffcsp", small_cfg)
        source = _make_backbone("diffcsp")
        native = _clone_state_dict(source)
        embedding = source.decoder.node_embedding
        bias = paddle.linspace(-0.1, 0.1, embedding.weight.shape[1])
        inputs = _make_denoise_inputs("diffcsp", source)
        hook = embedding.register_forward_post_hook(
            lambda layer, args, output: output + bias
        )
        expected = source._decode(*inputs)
        hook.remove()

        native["decoder.node_embedding.weight"] += bias
        checkpoint = package_dir / "checkpoints" / _WEIGHTS_NAME
        paddle.save(native, str(checkpoint))
        adapter = _make_adapter("diffcsp")
        save_load.load_pretrain(adapter, str(package_dir), weights_name=_WEIGHTS_NAME)
        actual = adapter.backbone()._decode(*inputs)
        for reference_value, actual_value in zip(expected, actual, strict=True):
            np.testing.assert_allclose(
                actual_value.numpy(), reference_value.numpy(), atol=1e-5, rtol=1e-5
            )
        sampled = adapter.sample(
            {
                "structure_array": {
                    "num_atoms": paddle.to_tensor([2], dtype="int64"),
                    "atom_types": paddle.to_tensor([14, 14], dtype="int64"),
                }
            },
            num_inference_steps=2,
        )
        assert len(sampled["result"]) == 1
        assert np.isfinite(sampled["result"][0]["frac_coords"]).all()

    @pytest.mark.parametrize("model_type", ["mattergen", "diffcsp"])
    def test_package_yaml_builds_and_loads(self, model_type, tmp_path):
        small_cfg = _tiny_backbone_cfg(model_type)
        if model_type == "diffcsp":
            small_cfg["__class_name__"] = "DiffCSP"
        package_name, package_dir = _write_model_package(
            tmp_path, model_type, small_cfg
        )
        resolved = resolve_model_package_dir(package_name, str(tmp_path))
        assert resolved == str(package_dir)

        package_config = _read_config(pathlib.Path(resolved) / f"{package_name}.yaml")
        model = build_model(package_config["Model"])
        assert isinstance(model, MatinventRL)
        assert type(model.backbone()).__name__ == _BACKBONE_CLASS[model_type].__name__
        expected_steps = package_config["Global"]["num_inference_steps"]
        assert model.num_inference_steps == expected_steps

        save_load.load_pretrain(model, resolved, weights_name=_WEIGHTS_NAME)
        saved = paddle.load(str(package_dir / "checkpoints" / _WEIGHTS_NAME))
        loaded = model.backbone().state_dict()
        key = next(iter(saved))
        assert np.allclose(saved[key].numpy(), loaded[key].numpy())

    def test_sampled_structures_from_package_model(self, tmp_path):
        package_name, package_dir = _write_model_package(
            tmp_path, "diffcsp", _tiny_backbone_cfg("diffcsp")
        )
        config = _read_config(package_dir / f"{package_name}.yaml")
        model = build_model(config["Model"])
        save_load.load_pretrain(model, str(package_dir), weights_name=_WEIGHTS_NAME)
        model.eval()

        sampler = MatInventSampler(
            model_type="diffcsp",
            batch_size=1,
            num_batches=2,
            num_inference_steps=2,
            build_structure_cfg={"format": "array", "niggli": False, "canocial": False},
        )
        np.random.seed(_DISPATCH_SEED)
        structures = sampler.generate(model)
        assert len(structures) == 2
        assert all(np.isfinite(s.lattice.matrix).all() for s in structures)
        assert all(np.isfinite(s.frac_coords).all() for s in structures)
        for error_type in (RuntimeError, IndexError):
            with mock.patch.object(
                model, "sample", side_effect=error_type("sample failed")
            ):
                with pytest.raises(error_type, match="sample failed"):
                    sampler.generate(model)


class TestMatterGenBackbone:
    """The MatInvent MatterGen variant must keep the mainline graph numerics."""

    def test_paddle_graph_keeps_mainline_numerics(self):
        mainline_cfg = copy.deepcopy(_tiny_backbone_cfg("mattergen"))
        mainline_cfg["__class_name__"] = "MatterGen"
        with paddle.utils.unique_name.guard():
            paddle.seed(_DISPATCH_SEED)
            mainline = build_model(mainline_cfg)
        with paddle.utils.unique_name.guard():
            paddle.seed(_DISPATCH_SEED)
            adapter = build_model(_tiny_backbone_cfg("mattergen"))
        assert type(adapter).__name__ == "MatinventMatterGen"

        adapter.set_state_dict(_clone_state_dict(mainline))
        mainline.eval()
        adapter.eval()

        inputs = _make_denoise_inputs("mattergen", adapter)
        expected = mainline._denoise(*inputs)
        actual = adapter._denoise(*inputs)
        for expected_value, actual_value in zip(
            _flatten_result(expected), _flatten_result(actual), strict=True
        ):
            np.testing.assert_allclose(
                np.asarray(actual_value), np.asarray(expected_value), atol=0, rtol=0
            )


class TestDataFlow:
    """Dataset -> backbone forward/loss, the RL training data path."""

    def test_qualification_graph_is_periodic(self):
        from ppmat.models.mattersim.m3gnet import M3GNet
        from ppmat.models.mattersim.m3gnet_graph_converter import M3GNetGraphConvertor

        structure = _make_structure(["Na", "Cl"], [[0, 0, 0], [-0.5, 0.5, 0.5]], a=5.64)
        equivalent = structure.copy()
        equivalent.translate_sites([1], [1, -1, 0], to_unit_cell=False)
        converter = M3GNetGraphConvertor()
        graphs = [converter(s).tensor() for s in (structure, equivalent)]
        model = M3GNet(num_blocks=1, units=8, force_key=None, stress_key=None)
        model.eval()
        predictions = [model.predict(graph)["energy"] for graph in graphs]
        np.testing.assert_allclose(predictions[0], predictions[1], atol=1e-6)
        for graph in graphs:
            assert float(graph.edge_feat["bond_dist"].max()) <= converter.cutoff

        prim_enabled = core._is_eager_prim_enabled()
        core.set_prim_eager_enabled(True)
        try:
            model.force_key = "force"
            model.stress_key = "stress"
            predictor = PotentialPredictor.__new__(PotentialPredictor)
            predictor.config = {
                "Global": {"label_names": ["energy", "force"]},
                "Model": {"__class_name__": "M3GNet"},
            }
            predictor.model = model
            predictor.graph_converter_fn = converter
            predictor.post_transforms = None
            predictor.eval_with_no_grad = False
            calculator = _MatterSimCalculator(predictor)
            assert isinstance(calculator, ASECalculator)
            atoms = AseAtomsAdaptor.get_atoms(structure)
            atoms.calc = calculator
            for displacement, scale in ((0.0, 1.0), (0.15, 1.02)):
                atoms.positions[1, 0] += displacement
                atoms.set_cell(atoms.cell * scale, scale_atoms=True)
                current = AseAtomsAdaptor.get_structure(atoms)
                reference = predictor._run_model(converter(current).tensor())
                np.testing.assert_allclose(
                    atoms.get_potential_energy(),
                    float(reference["energy"].item()) * len(atoms),
                    atol=1e-6,
                )
                np.testing.assert_allclose(
                    atoms.get_forces(), reference["force"], atol=1e-6
                )
                np.testing.assert_allclose(
                    atoms.get_stress(voigt=False),
                    reference["stress"] * units.GPa,
                    atol=1e-6,
                )
                assert atoms.get_potential_energy(
                    force_consistent=True
                ) == pytest.approx(atoms.get_potential_energy())
                assert "graph" not in atoms.info
                assert "graph" not in calculator.atoms.info
            nonfinite = {key: value.copy() for key, value in reference.items()}
            nonfinite["force"][0, 0] = np.nan
            atoms.positions[0, 0] += 0.1
            with mock.patch.object(model, "predict", return_value=nonfinite):
                for _ in range(2):
                    with pytest.raises(FloatingPointError, match="Nonfinite MatterSim"):
                        atoms.get_forces()
        finally:
            core.set_prim_eager_enabled(prim_enabled)

    def test_rl_score_target_and_descending_schedule(self):
        backbone = _make_backbone("diffcsp")
        loop = MatInvent.__new__(MatInvent)
        loop.has_atom_diffusion = False
        structures = [
            _make_structure(["Si", "Si"], [[0, 0, 0], [0.25, 0.25, 0.25]]),
            _make_structure(["C"], [[0.1, 0.2, 0.3]], a=4.0),
        ]
        batch = next(
            iter(
                create_rl_dataloader(
                    structures, np.ones(2, dtype="float32"), batch_size=2, shuffle=False
                )
            )
        )
        for step in (0, 99, backbone.num_train_timesteps - 1):
            noised = loop._add_noise_to_model(backbone, batch, step)
            noisy, _, times = noised
            assert (
                times.numpy().tolist() == [backbone.num_train_timesteps - 1 - step] * 2
            )
            per_atom = times.repeat_interleave(batch["structure_array"]["num_atoms"])
            sigmas = backbone.coord_scheduler.discrete_sigmas[per_atom][:, None]
            norm = backbone.coord_scheduler.discrete_sigmas_norm[per_atom][:, None]
            target = d_log_p_wrapped_normal(
                sigmas * noisy["rand_x"], sigmas
            ) / paddle.sqrt(norm)
            sigma_array = sigmas.numpy().astype("float64")
            offsets = (sigma_array * noisy["rand_x"].numpy())[..., None] + np.arange(
                -10, 11
            )
            variance = sigma_array[..., None] ** 2
            weights = np.exp(-(offsets**2) / (2 * variance))
            expected = (offsets * weights / variance).sum(-1) / weights.sum(-1)
            expected /= np.sqrt(norm.numpy())
            np.testing.assert_allclose(target.numpy(), expected, atol=1e-4, rtol=1e-4)
            prediction = target.detach().clone()
            prediction.stop_gradient = False
            with mock.patch.object(
                backbone, "_decode", return_value=(noisy["rand_l"], prediction)
            ) as denoise:
                loss, _ = loop._calc_sample_loss_from_model(backbone, noised)
            denoise.assert_called_once()
            np.testing.assert_allclose(loss.numpy(), np.zeros(2), atol=1e-7)
            loss.sum().backward()
            np.testing.assert_allclose(prediction.grad.numpy(), 0, atol=1e-7)
        with pytest.raises(ValueError, match="noise schedule"):
            loop._add_noise_to_model(backbone, batch, backbone.num_train_timesteps)

    @pytest.mark.parametrize("model_type", ["mattergen", "diffcsp"])
    def test_rl_loss_uses_model_denoise_boundary(self, model_type):
        backbone = _make_backbone(model_type, "cinn")
        backbone.train()
        loop = MatInvent.__new__(MatInvent)
        loop.has_atom_diffusion = model_type == "mattergen"
        batch = {"structure_array": _make_denoise_inputs("mattergen", backbone)[0]}
        noised = loop._add_noise_to_model(backbone, batch, 0)
        calls = []
        with mock.patch.object(
            type(backbone), "_run_runtime", _record_boundaries(calls)
        ):
            loss, _ = loop._calc_sample_loss_from_model(backbone, noised)
        assert calls == ["denoise_step"]
        assert bool(paddle.isfinite(loss).all())

    def test_dataset_to_forward_loss(self):
        joint_config = OmegaConf.load(
            _CONFIG_ROOT / "matinvent" / "matinvent_diffcsp_joint.yaml"
        )
        joint_config.Global.num_train_timesteps = 32
        resolved = OmegaConf.to_container(joint_config, resolve=True)
        params = resolved["Model"]["__init_params__"]
        backbone_params = params["backbone_cfg"]["__init_params__"]
        assert params["rl_cfg"]["finetune_cfg"]["timesteps"] == 32
        assert backbone_params["num_train_timesteps"] == 32
        for key in ("lattice_noise_scheduler_cfg", "coord_noise_scheduler_cfg"):
            assert backbone_params[key]["__init_params__"]["num_train_timesteps"] == 32
        sampler = MatInventSampler(
            model_type="diffcsp",
            batch_size=4096,
            num_batches=1,
            num_inference_steps=2,
            build_structure_cfg=params["build_structure_cfg"],
            atom_count_distribution=params["rl_cfg"]["sample_cfg"][
                "atom_count_distribution"
            ],
        )
        np.random.seed(_DISPATCH_SEED)
        backbone = _make_backbone("diffcsp")
        with mock.patch.object(
            backbone, "sample", return_value={"result": []}
        ) as sample:
            sampler.generate(backbone)
        counts = sample.call_args.args[0]["structure_array"]["num_atoms"].numpy()
        assert counts.min() == 1
        assert counts.max() == 20
        assert (
            resolved["Sample"]["model_sample_params"]["atom_count_distribution"]
            == sampler.atom_count_distribution
        )
        public_batch = next(iter(build_dataloader(resolved["Sample"]["data"])))
        original_counts = public_batch["structure_array"]["num_atoms"].clone()
        adapter = _make_adapter("diffcsp")
        with mock.patch.object(
            adapter.backbone(), "sample", return_value={"result": []}
        ) as sample:
            adapter.sample(public_batch, **resolved["Sample"]["model_sample_params"])
            sampled_counts = sample.call_args.args[0]["structure_array"]["num_atoms"]
            assert sampled_counts.shape == original_counts.shape
            assert bool(((sampled_counts >= 1) & (sampled_counts <= 20)).all())
            assert "atom_count_distribution" not in sample.call_args.kwargs
            np.testing.assert_array_equal(
                public_batch["structure_array"]["num_atoms"], original_counts
            )
            explicit = {"structure_array": {"num_atoms": paddle.to_tensor([4])}}
            adapter.sample(explicit, **resolved["Sample"]["model_sample_params"])
            assert sample.call_args.args[0] is explicit
            assert sample.call_args.args[0]["structure_array"]["num_atoms"].item() == 4
        structures = [
            _make_structure(["Si", "Si"], [[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]]),
            _make_structure(
                ["C", "O", "O"],
                [[0.0, 0.0, 0.0], [0.25, 0.25, 0.25], [0.5, 0.5, 0.5]],
                a=4.0,
            ),
        ]
        loader = create_rl_dataloader(
            structures,
            np.array([0.8, 0.3], dtype="float32"),
            batch_size=2,
            shuffle=False,
        )
        batch = next(iter(loader))
        structure_array = batch["structure_array"]

        assert structure_array["num_atoms"].numpy().tolist() == [2, 3]
        assert tuple(structure_array["frac_coords"].shape) == (5, 3)
        assert tuple(structure_array["reward"].shape) == (2,)

        result = backbone.forward(batch)
        loss = result["loss_dict"]["loss"]
        assert np.isfinite(float(loss))
        loss.backward()


class TestRewardScoring:
    """Reward scoring over sampled structures, including failed properties."""

    def test_properties_have_independent_result_files(self, tmp_path):
        structures = [_make_structure(["Si"], [[0, 0, 0]])]
        properties = [
            {
                "name": "density",
                "target": "ascending",
                "minv": 0,
                "maxv": 10,
                "calculator": {"__class_name__": "PyMatGen", "task": "density"},
            },
            {
                "name": "mcia",
                "target": "descending",
                "minv": 0,
                "maxv": 100,
                "calculator": {
                    "__class_name__": "PyMatGen",
                    "task": "mcia",
                    "substrate_path": str(
                        _REPO_ROOT
                        / "property_prediction/example_data/cifs/mp-18767-LiMnO2.cif"
                    ),
                },
            },
            {
                "name": "energy",
                "target": "descending",
                "minv": 0,
                "maxv": 10,
                "calculator": {"__class_name__": "DFTCalc", "task": "energy"},
            },
        ]
        reward = Reward(str(tmp_path), properties, reward_threshold=0.5)
        with pytest.raises(FileNotFoundError, match="DFT results are missing"):
            reward.scoring((structures, "valid"), label="batch")
        (energy_path,) = (tmp_path / "energy" / "batch").glob("*/0.cif")
        energy_path = energy_path.parent / "results.txt"
        np.savetxt(energy_path, [1.23])
        _, values, failed = reward.scoring((structures, "valid"), label="batch")
        assert not failed.any()
        assert np.isfinite(values["mcia"]).all() and values["mcia"][0] > 0
        np.testing.assert_allclose(values["energy"], [1.23])
        np.testing.assert_allclose(values["density"], [float(structures[0].density)])
        np.testing.assert_allclose(np.loadtxt(energy_path), 1.23)
        assert (tmp_path / "density" / "batch.txt").is_file()
        changed = [_make_structure(["Si"], [[0, 0, 0]], a=6.0)]
        with pytest.raises(FileNotFoundError, match="DFT results are missing"):
            reward.scoring((changed, "valid"), label="batch")

    def test_density_reward_and_failed_property(self, monkeypatch):
        structures = [
            _make_structure(["Si", "Si"], [[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]])
        ]

        class FailingCalculator:
            def __init__(self, root_dir, task=None):
                self.root_dir = root_dir

            def calc(self, samples, label="tmp"):
                return np.full(len(samples[0]), np.nan)

        with tempfile.TemporaryDirectory() as tmp:
            prop_cfg = [
                {
                    "name": "density",
                    "target": "ascending",
                    "minv": 0.0,
                    "maxv": 10.0,
                    "calculator": {
                        "__class_name__": "PyMatGen",
                        "task": "density",
                    },
                }
            ]
            reward = Reward(
                root_dir=tmp, prop_cfg=prop_cfg, reward_threshold=0.5, reduce="mean"
            )
            rewards, prop_dict, failed = reward.scoring((structures, "valid"))

            assert not failed.any()
            assert 0.0 <= float(rewards[0]) <= 1.0
            assert np.isfinite(prop_dict["density"][0])

            monkeypatch.setitem(CALCULATOR_REGISTRY, "Failing", FailingCalculator)
            prop_cfg[0]["calculator"]["__class_name__"] = "Failing"
            failing_reward = Reward(
                root_dir=tmp, prop_cfg=prop_cfg, reward_threshold=0.5, reduce="mean"
            )
            rewards, _, failed = failing_reward.scoring((structures, "failed"))

            assert failed.tolist() == [True]
            assert float(rewards[0]) == 0.0


class TestRLTraining:
    """The unified-entry hook: checkpointing and backend orchestration."""

    @pytest.mark.parametrize("accum_steps", [2, 3])
    def test_reference_replay_batch_and_time_labels(self, monkeypatch, accum_steps):
        cfg = _tiny_backbone_cfg("diffcsp")
        params = cfg["__init_params__"]
        params["decoder_cfg"].update({"pred_type": True, "smooth": True})
        params["atom_loss_weight"] = 20.0
        params["time_embedding_offset"] = 1
        params["lattice_noise_scheduler_cfg"]["__init_params__"].update(
            {"beta_clip_min": 1e-4, "beta_clip_max": 0.9999}
        )
        loop = MatInvent.__new__(MatInvent)
        loop.agent = build_model(cfg)
        loop.prior = build_model(cfg)
        loop.prior.set_state_dict(loop.agent.state_dict())
        for parameter in loop.prior.parameters():
            parameter.stop_gradient = True
        loop.has_type_diffusion = True
        loop.finetune_cfg = {
            "batch_size": 16,
            "epochs": 1,
            "timesteps": 4,
            "accum_steps": accum_steps,
            "sigma": 0.025,
            "optimizer_cfg": {
                "__class_name__": "Adam",
                "__init_params__": {"lr": 0.0001},
            },
        }
        structures = [
            _make_structure(["Si"], [[0.1, 0.2, 0.3]]),
            _make_structure(
                ["Si", "O", "O"], [[0, 0, 0], [0.3, 0.4, 0.2], [0.6, 0.7, 0.8]]
            ),
        ] * 9
        labels = []
        hook = loop.agent.time_embedding.register_forward_pre_hook(
            lambda layer, args: labels.append(args[0].numpy().copy())
        )
        steps = []
        optimizer_step = paddle.optimizer.Adam.step
        calc_loss = loop._calc_sample_loss_from_model
        scaled_gradients = []

        def checked_loss(model, noised_input):
            loss, predictions = calc_loss(model, noised_input)
            if model is loop.agent:
                noisy, clean, _ = noised_input
                counts = clean["structure_array"]["num_atoms"].numpy()
                boundaries = np.concatenate(([0], np.cumsum(counts)))
                lattice_error = np.square(
                    predictions["lattice"].numpy() - noisy["rand_l"].numpy()
                ).mean(axis=(1, 2))
                expected_losses = []
                for index, (start, end) in enumerate(
                    zip(boundaries[:-1], boundaries[1:])
                ):
                    coord_error = np.square(
                        predictions["coords"].numpy()[start:end]
                        - noisy["coord_target"].numpy()[start:end]
                    ).mean()
                    type_error = np.square(
                        predictions["atom_types"].numpy()[start:end]
                        - noisy["atom_type_target"].numpy()[start:end]
                    ).mean()
                    expected_losses.append(
                        model.lattice_loss_weight * lattice_error[index]
                        + model.coord_loss_weight * coord_error
                        + model.atom_loss_weight * type_error
                    )
                np.testing.assert_allclose(
                    loss.numpy(), expected_losses, atol=1e-5, rtol=1e-5
                )
                rewards = noised_input[1]["structure_array"]["reward"].numpy()
                expected = rewards / (len(structures) * accum_steps)

                def check_gradient(gradient):
                    np.testing.assert_allclose(gradient.numpy(), expected, rtol=1e-6)
                    scaled_gradients.append(gradient.numpy())

                loss.register_hook(check_gradient)
            return loss, predictions

        def step(optimizer):
            for name, parameter in loop.agent.named_parameters():
                if parameter.stop_gradient:
                    assert parameter.grad is None, name
                else:
                    assert parameter.grad is not None, name
                    assert bool(paddle.isfinite(parameter.grad).all()), name
            steps.append(1)
            return optimizer_step(optimizer)

        monkeypatch.setattr(paddle.optimizer.Adam, "step", step)
        monkeypatch.setattr(loop, "_calc_sample_loss_from_model", checked_loss)
        initial = _clone_state_dict(loop.agent)
        try:
            loop.ft_step(structures, np.linspace(0.1, 0.9, 18))
        finally:
            hook.remove()
        assert len(steps) == 2
        assert len(scaled_gradients) == 4
        assert len(labels) == 4
        for index, times in enumerate(labels):
            np.testing.assert_array_equal(times, np.full(18, 1000 - index))
        assert any(
            not paddle.equal_all(initial[key], value)
            for key, value in loop.agent.state_dict().items()
        )
        assert all(
            bool(paddle.isfinite(value).all())
            for value in loop.agent.state_dict().values()
        )

    def test_nonfinite_backward_stops_before_optimizer_update(self):
        loop = MatInvent.__new__(MatInvent)
        loop.agent = _make_backbone("diffcsp")
        loop.prior = _make_backbone("diffcsp")
        loop.prior.set_state_dict(loop.agent.state_dict())
        loop.finetune_cfg = {
            "batch_size": 1,
            "epochs": 1,
            "timesteps": 2,
            "accum_steps": 1,
            "sigma": 0.025,
            "optimizer_cfg": {
                "__class_name__": "Adam",
                "__init_params__": {"lr": 0.0001},
            },
        }
        initial = _clone_state_dict(loop.agent)
        parameter = loop.agent.decoder.lattice_out.weight
        hook = parameter.register_hook(lambda grad: grad * float("nan"))
        structure = _make_structure(["Si", "O"], [[0, 0, 0], [0.3, 0.4, 0.2]])
        try:
            with pytest.raises(FloatingPointError, match="Nonfinite RL gradient"):
                loop.ft_step([structure], np.ones(1))
        finally:
            hook.remove()
        for key, value in loop.agent.state_dict().items():
            np.testing.assert_array_equal(value.numpy(), initial[key].numpy())

    @pytest.mark.parametrize("mode", ["plain", "anchor", "qualification"])
    def test_before_train_writes_checkpoints(self, monkeypatch, tmp_path, mode):
        with_anchor = mode == "anchor"
        small_cfg = _tiny_backbone_cfg("diffcsp")
        package_name, package_dir = _write_model_package(tmp_path, "diffcsp", small_cfg)

        config = _load_config("diffcsp")
        config["Model"]["__init_params__"]["backbone_cfg"] = small_cfg
        output_dir = str(tmp_path / "output")
        config["Trainer"]["output_dir"] = output_dir
        rl_cfg = config["Model"]["__init_params__"]["rl_cfg"]
        rl_cfg["pretrained_model_name"] = package_name
        rl_cfg["rl_epoch"] = 1
        rl_cfg["save_freq"] = 1
        rl_cfg["replay_cfg"] = None
        rl_cfg["div_filter_cfg"]["enabled"] = False
        rl_cfg["sample_cfg"].update(
            {"batch_size": 1, "num_batches": 1, "num_inference_steps": 2}
        )
        rl_cfg["finetune_cfg"].update(
            {"batch_size": 2, "epochs": 1, "timesteps": 1, "accum_steps": 1}
        )
        structures = [_make_structure(["Si", "O"], [[0, 0, 0], [0.3, 0.4, 0.2]])]
        invalid = _make_structure(
            ["Na", "Cl", "Cl"], [[0, 0, 0], [0.3, 0.4, 0.2], [0.7, 0.7, 0.6]]
        )
        monkeypatch.setattr(
            "ppmat.models.matinvent.sampler.MatInventSampler.generate",
            lambda self, model: structures + [invalid],
        )
        if mode == "qualification":
            reference_structure = _make_structure(
                ["Si", "O"], [[0, 0, 0], [0.02, 0.02, 0.02]], a=20
            )
            reference_entries = [
                ComputedStructureEntry(_make_structure(["Si"], [[0, 0, 0]]), -1),
                ComputedStructureEntry(_make_structure(["O"], [[0, 0, 0]]), -1),
                ComputedStructureEntry(_make_structure(["Mg"], [[0, 0, 0]]), -1),
                ComputedStructureEntry(reference_structure, -2),
            ]
            database = tmp_path / "reference.lmdb"
            with lmdb.open(str(database), subdir=False, map_size=1048576) as env:
                with env.begin(write=True) as transaction:
                    systems = [e.composition.chemical_system for e in reference_entries]
                    transaction.put(b"chemical_systems", pickle.dumps(systems))
                    for entry, system in zip(reference_entries, systems):
                        formula = entry.composition.reduced_formula
                        values = {
                            f"{system}.reduced_formulas": [formula],
                            f"{system}.{formula}.length": 1,
                            f"{system}.{formula}.0": entry.as_dict(),
                        }
                        for key, value in values.items():
                            transaction.put(key.encode("ascii"), pickle.dumps(value))
            archive = tmp_path / "reference.gz"
            with gzip.open(archive, "wb") as target:
                target.write(database.read_bytes())
            rl_cfg["qualification_cfg"] = {
                "reference_path": str(archive),
                "cache_dir": str(tmp_path / "reference_cache"),
                "predictor_cfg": {},
            }
            rl_cfg["rl_epoch"] = 2
            rl_cfg["sample_cfg"]["max_num"] = 1
            rl_cfg["replay_cfg"] = {
                "buffer_size": 3,
                "sample_size": 1,
                "reward_cutoff": 0,
            }
            relaxed = structures[0].copy()
            relaxed.scale_lattice(relaxed.volume * 1.2)
            unconverged = structures[0].copy()
            unconverged.scale_lattice(unconverged.volume * 1.5)
            candidates = structures + [
                unconverged,
                structures[0].copy(),
                reference_structure,
            ]
            unstable = _make_structure(["Mg", "O"], [[0, 0, 0], [0.3, 0.4, 0.2]])
            candidates.append(unstable)
            monkeypatch.setattr(
                "ppmat.models.matinvent.sampler.MatInventSampler.generate",
                lambda self, model: candidates + [invalid],
            )

            executor = mock.MagicMock()
            executor.__enter__.return_value.submit.return_value.result.return_value = (
                [0, 2, 3, 4],
                [relaxed, relaxed.copy(), reference_structure, unstable],
                [-10, -10, -10, 10],
            )
            monkeypatch.setattr(
                "ppmat.models.matinvent.qualification.ProcessPoolExecutor",
                lambda **kwargs: executor,
            )
            original_loader = create_rl_dataloader

            def training_loader(**kwargs):
                assert all(s is structures[0] for s in kwargs["structures"])
                return original_loader(**kwargs)

            monkeypatch.setattr(
                "ppmat.models.matinvent.rl_loop.create_rl_dataloader", training_loader
            )
        if with_anchor:
            rl_cfg["finetune_cfg"]["timesteps"] = 6
            dataset = RLDataset(structures, np.ones(1))
            monkeypatch.setattr(
                "ppmat.datasets.mp20_dataset.MP20Dataset", lambda **kwargs: dataset
            )
            rl_cfg["anchor_cfg"] = {
                "enabled": True,
                "data_path": "anchor.csv",
                "build_structure_cfg": {"format": "cif_str"},
                "num_samples": 1,
                "loss_weight": 0.5,
                "every_n_steps": 1,
                "probe_cfg": {
                    "data_path": "validation.csv",
                    "build_structure_cfg": {"format": "cif_str"},
                    "num_samples": 1,
                },
            }
            rl_cfg["early_stop_cfg"] = {
                "enabled": True,
                "probe_every": 1,
                "probe_every_updates": 1,
                "probe_draws": 1,
                "max_loss_ratio": 1e-8,
            }
            probe_method = MatInvent._probe_anchor
            probe_calls = []

            def checked_probe(loop):
                assert loop.probe_batch is not loop.anchor_batch
                numpy_state = np.random.get_state()
                paddle_state = paddle.get_rng_state()
                expected = paddle.rand([3]).numpy()
                paddle.set_rng_state(paddle_state)
                inputs = []
                hook = loop.agent.register_forward_pre_hook(
                    lambda layer, args: inputs.append(args[0])
                )
                try:
                    result = probe_method(loop)
                finally:
                    hook.remove()
                assert inputs and all(batch is loop.probe_batch for batch in inputs)
                probe_calls.append(result)
                actual_numpy_state = np.random.get_state()
                np.testing.assert_array_equal(numpy_state[1], actual_numpy_state[1])
                assert numpy_state[2:] == actual_numpy_state[2:]
                np.testing.assert_array_equal(paddle.rand([3]).numpy(), expected)
                paddle.set_rng_state(paddle_state)
                return result

            monkeypatch.setattr(MatInvent, "_probe_anchor", checked_probe)

        monkeypatch.setitem(
            MODEL_REGISTRY, package_name, "http://localhost/package.zip"
        )
        monkeypatch.setattr(
            download, "get_weights_path_from_url", lambda url: str(tmp_path)
        )

        model = build_model(config["Model"])
        model.before_train(_FakeTrainer(output_dir))
        if with_anchor:
            assert len(probe_calls) == 3

        model_dir = os.path.join(output_dir, "models")
        initial = paddle.load(str(package_dir / "checkpoints" / _WEIGHTS_NAME))
        final = paddle.load(os.path.join(model_dir, "final", "model.pdparams"))
        for key in initial:
            if ".prop_mlp." in key:
                np.testing.assert_array_equal(initial[key].numpy(), final[key].numpy())
        assert any(not paddle.equal_all(initial[key], final[key]) for key in initial)
        assert all(bool(paddle.isfinite(value).all()) for value in final.values())
        assert os.path.isfile(os.path.join(model_dir, "final", "model.pdparams"))
        assert os.path.isfile(os.path.join(model_dir, "loop_0000", "model.pdparams"))
        assert os.path.isfile(
            os.path.join(output_dir, "samples", "long_term_memory.csv")
        )
        memory = pd.read_csv(
            os.path.join(output_dir, "samples", "long_term_memory.csv")
        )
        assert len(memory) == (2 if mode == "qualification" else 1)
        if mode == "qualification":
            scored = BuildStructure.build_one(
                memory["cif"][0], "cif_str", niggli=False, canocial=False
            )
            assert scored.volume == pytest.approx(relaxed.volume, rel=1e-6)
        assert memory["comp"].tolist() == ["SiO"] * len(memory)

    def test_trainer_rejects_unsupported_rl_resume(self, tmp_path):
        config = _load_config("diffcsp")
        config["Trainer"]["output_dir"] = str(tmp_path / "output")
        config["Trainer"]["resume_from_checkpoint"] = str(tmp_path / "checkpoint")
        config["Model"]["__init_params__"]["backbone_cfg"] = _tiny_backbone_cfg(
            "diffcsp"
        )
        model = build_model(config["Model"])
        trainer = BaseTrainer(config["Trainer"], model, train_dataloader=[None])
        with mock.patch("ppmat.models.matinvent.rl_finetune.run_rl_training") as run_rl:
            with pytest.raises(ValueError, match="does not support.*resume"):
                trainer.train()
            run_rl.assert_not_called()
        assert not (tmp_path / "output" / "models").exists()

    def test_before_train_applies_execution_backend(self, monkeypatch, tmp_path):
        config = _patched_rl_environment(monkeypatch, tmp_path, "diffcsp", "cinn")

        # Orchestration test only: no GPU compiler is involved.
        monkeypatch.setattr(
            DiffCSP, "validate_execution_backend", lambda self, **kwargs: None
        )
        names = []
        monkeypatch.setattr(DiffCSP, "_run_runtime", _record_boundaries(names))

        model = build_model(config["Model"])
        # Mirror what BaseTrainer does with the task ``Execution`` section.
        configure_execution_backend(
            model,
            config["Execution"]["backend"],
            init_params=config["Execution"]["__init_params__"],
            owner="test",
        )
        model.before_train(_FakeTrainer(config["Trainer"]["output_dir"]))

        model_dir = os.path.join(config["Trainer"]["output_dir"], "models")
        final_state = paddle.load(os.path.join(model_dir, "final", "model.pdparams"))
        # Structural only: the RL loop trains a separate agent and is stochastic.
        assert final_state.keys() == model.state_dict().keys()
        assert names, "sampling must dispatch through the denoise boundary"
        assert set(names) == {"denoise_step"}
        assert os.path.isfile(
            os.path.join(
                config["Trainer"]["output_dir"], "samples", "long_term_memory.csv"
            )
        )


class TestCinnDispatch:
    """CINN boundary dispatch: compiled numbers must match eager exactly."""

    @pytest.mark.parametrize("model_type", ["mattergen", "diffcsp"])
    def test_boundary_dispatch_matches_eager(self, model_type):
        backbone_cls = _BACKBONE_CLASS[model_type]
        method_name = _BACKBONE_METHOD[model_type]
        eager_backbone, cinn_backbone = _make_dispatch_pair(model_type)
        assert cinn_backbone.execution_backend == "cinn"

        inputs = _make_denoise_inputs(model_type, eager_backbone)
        expected = getattr(eager_backbone, method_name)(*inputs)

        names = []
        with mock.patch.object(backbone_cls, "_run_runtime", _record_boundaries(names)):
            actual = getattr(cinn_backbone, method_name)(*inputs)

        assert set(names) == {"denoise_step"}
        for expected_value, actual_value in zip(
            _flatten_result(expected), _flatten_result(actual), strict=True
        ):
            np.testing.assert_allclose(
                np.asarray(actual_value), np.asarray(expected_value), atol=0, rtol=0
            )
        assert cinn_backbone.state_dict().keys() == eager_backbone.state_dict().keys()

    def test_nested_adapter_samples_with_cinn_backend(self, monkeypatch):
        """The adapter forwards the backend; sampling hits the denoise boundary."""
        with paddle.utils.unique_name.guard():
            paddle.seed(_DISPATCH_SEED)
            adapter = _make_adapter("diffcsp", execution_backend="cinn")
        assert adapter.execution_backend == "cinn"
        keys_before = list(adapter.state_dict().keys())

        names = []
        monkeypatch.setattr(DiffCSP, "_run_runtime", _record_boundaries(names))
        result = adapter.sample(
            {
                "structure_array": {
                    "num_atoms": paddle.to_tensor([2], dtype="int64"),
                    "atom_types": paddle.to_tensor([14, 14], dtype="int64"),
                }
            },
            num_inference_steps=2,
        )

        assert len(result["result"]) == 1
        assert names and set(names) == {"denoise_step"}

        adapter.set_execution_backend("eager")
        assert list(adapter.state_dict().keys()) == keys_before


def _gate_gpu_cinn_rl_loop():
    if os.environ.get("PPMAT_RUN_CINN_WORKFLOW_TESTS") != "1":
        pytest.skip("Set PPMAT_RUN_CINN_WORKFLOW_TESTS=1 to run the GPU CINN test")
    if not paddle.is_compiled_with_cuda() or not paddle.base.is_compiled_with_cinn():
        pytest.skip("Paddle must be compiled with CUDA and CINN")
    paddle.set_device("gpu:0")


def test_gpu_cinn_rl_loop_matches_eager(monkeypatch):
    """Gated real-compile check: eager and cinn loops stay numerically close."""
    _gate_gpu_cinn_rl_loop()

    generate = MatInventSampler.generate

    def generate_training_structure(sampler, model):
        generate(sampler, model)
        return [_make_structure(["Na", "Cl"], [[0, 0, 0], [0.3, 0.4, 0.2]])]

    monkeypatch.setattr(MatInventSampler, "generate", generate_training_structure)

    with tempfile.TemporaryDirectory() as tmp_dir:
        final_states = {}
        for execution in ("eager", "cinn"):
            config = _patched_rl_environment(monkeypatch, tmp_dir, "diffcsp", execution)
            output_dir = os.path.join(tmp_dir, "output", execution)
            config["Trainer"]["output_dir"] = output_dir
            paddle.seed(_DISPATCH_SEED)
            np.random.seed(_DISPATCH_SEED)
            model = build_model(config["Model"])
            configure_execution_backend(model, execution, owner="test")
            runtimes = []
            original_run = MatInvent.run_rl

            def run_and_record(loop):
                original_run(loop)
                runtimes.append(loop.agent._runtime_cache.copy())

            with mock.patch.object(MatInvent, "run_rl", run_and_record):
                model.before_train(_FakeTrainer(output_dir))
            if execution == "cinn":
                assert ("cinn", "train", "denoise_step") in runtimes[0]
                assert ("cinn", "eval", "denoise_step") in runtimes[0]
            checkpoint = os.path.join(output_dir, "models", "final", "model.pdparams")
            final_states[execution] = paddle.load(checkpoint)

        for key, value in final_states["cinn"].items():
            np.testing.assert_allclose(
                value.numpy(),
                final_states["eager"][key].numpy(),
                atol=2e-5,
                rtol=2e-5,
            )


def test_gpu_cinn_mattergen_rl_loss_matches_eager():
    _gate_gpu_cinn_rl_loop()
    eager, cinn = _make_dispatch_pair("mattergen")
    batch = {"structure_array": _make_denoise_inputs("mattergen", eager)[0]}
    loop = MatInvent.__new__(MatInvent)
    loop.has_atom_diffusion = True
    noised = loop._add_noise_to_model(eager, batch, 0)
    outputs = []
    for model in (eager, cinn):
        model.train()
        loss, _ = loop._calc_sample_loss_from_model(model, noised)
        loss.mean().backward()
        gradients = {
            name: parameter.grad.numpy().copy()
            for name, parameter in model.named_parameters()
            if parameter.grad is not None
        }
        assert gradients and all(np.isfinite(grad).all() for grad in gradients.values())
        outputs.append((loss.numpy(), gradients))
    assert ("cinn", "train", "denoise_step") in cinn._runtime_cache
    np.testing.assert_allclose(outputs[0][0], outputs[1][0], atol=2e-5, rtol=2e-5)
    assert outputs[0][1].keys() == outputs[1][1].keys()
    for name, gradient in outputs[0][1].items():
        np.testing.assert_allclose(gradient, outputs[1][1][name], atol=2e-5, rtol=2e-5)

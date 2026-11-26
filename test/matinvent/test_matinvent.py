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

Covers the task configs, the diffusion backbones, the RL data and reward path,
registered weight packages, the round-based trainer and the CINN boundary.
"""

import copy
import csv
import importlib.util
import os
import pathlib
import tempfile
import unittest.mock as mock

import numpy as np
import paddle
import pytest
from ase import units
from omegaconf import OmegaConf
from paddle.framework import core
from pymatgen.core.structure import Lattice
from pymatgen.core.structure import Structure
from pymatgen.io.ase import AseAtomsAdaptor

from ppmat.calculator.ase import ASECalculator
from ppmat.datasets import build_dataloader
from ppmat.datasets.collate_fn import DefaultCollator
from ppmat.datasets.custom_data_type import ConcatData
from ppmat.datasets.transform import build_transforms
from ppmat.models import build_model
from ppmat.models.matinvent import MatinventDiffCSP
from ppmat.models.matinvent import MatinventMatterGen
from ppmat.models.matinvent import benchmark as matinvent_benchmark
from ppmat.models.matinvent.dataset import RLDataset
from ppmat.models.matinvent.dataset import to_tensor_batch
from ppmat.models.matinvent.qualification import _MatterSimCalculator
from ppmat.models.matinvent.rewards.reward import CALCULATOR_REGISTRY
from ppmat.models.matinvent.rewards.reward import Reward
from ppmat.models.matinvent.sampler import MatInventSampler
from ppmat.models.matinvent.training import MatInventTraining
from ppmat.optimizer import build_optimizer
from ppmat.predictor.potential_predictor import PotentialPredictor
from ppmat.trainer.base_trainer import BaseTrainer

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_CONFIG_ROOT = _REPO_ROOT / "structure_generation" / "configs" / "matinvent"
_TASK_CONFIG = {
    "mattergen": _CONFIG_ROOT / "matinvent_mattergen_mp20.yaml",
    "diffcsp": _CONFIG_ROOT / "matinvent_diffcsp_mp20.yaml",
    "diffcsp_joint": _CONFIG_ROOT / "matinvent_diffcsp_mp20_joint.yaml",
}
_MODEL_CLASS = {
    "mattergen": MatinventMatterGen,
    "diffcsp": MatinventDiffCSP,
    "diffcsp_joint": MatinventDiffCSP,
}
_WEIGHTS_NAME = "best.pdparams"
_SUBSTRATE_CIF = (
    _REPO_ROOT / "property_prediction/example_data/cifs/mp-18767-LiMnO2.cif"
)
_DISPATCH_SEED = 2026


def _read_config(path):
    return OmegaConf.to_container(OmegaConf.load(path), resolve=True)


def _load_config(model_type):
    return _read_config(_TASK_CONFIG[model_type])


def _make_structure(species, coords, a=5.43):
    return Structure(lattice=Lattice.cubic(a), species=species, coords=coords)


def _tiny_model_cfg(model_type):
    """Shrunk copy of the released architecture: same code path, fast numerics."""
    config = _load_config(model_type)
    model_cfg = copy.deepcopy(config["Model"])
    params = model_cfg["__init_params__"]
    if model_type != "mattergen":
        params["num_train_timesteps"] = 16
        for name in (
            "lattice_noise_scheduler_cfg",
            "coord_noise_scheduler_cfg",
            "atom_noise_scheduler_cfg",
        ):
            scheduler = params.get(name)
            if scheduler is not None:
                scheduler["__init_params__"]["num_train_timesteps"] = params[
                    "num_train_timesteps"
                ]
    if model_type == "mattergen":
        # Hidden and latent widths must stay coupled for the denoiser concat.
        params["decoder_cfg"]["hidden_dim"] = 32
        gemnet = params["decoder_cfg"]["gemnet_cfg"]
        gemnet["latent_dim"] = 32
        gemnet["atom_embedding_cfg"]["emb_size"] = 32
        gemnet["emb_size_atom"] = 32
        gemnet["emb_size_edge"] = 32
        gemnet["num_blocks"] = 1
        gemnet["max_neighbors"] = 12
        gemnet["cutoff"] = 5.0
    else:
        params["time_dim"] = 32
        decoder = params["decoder_cfg"]
        decoder["hidden_dim"] = 32
        decoder["latent_dim"] = 32
        decoder["num_layers"] = 1
        decoder["num_freqs"] = 6
    return model_cfg


def _make_model(model_type, execution_backend="eager"):
    with paddle.utils.unique_name.guard():
        paddle.seed(_DISPATCH_SEED)
        model = build_model(_tiny_model_cfg(model_type))
    model.set_execution_backend(execution_backend)
    return model


def _clone_state_dict(model):
    return {key: value.clone() for key, value in model.state_dict().items()}


def _batch():
    return {
        "structure_array": {
            "num_atoms": paddle.to_tensor([3, 2], dtype="int64"),
            "atom_types": paddle.to_tensor([6, 8, 8, 14, 14], dtype="int64"),
            "frac_coords": paddle.to_tensor(
                [
                    [0.1, 0.2, 0.3],
                    [0.4, 0.5, 0.6],
                    [0.7, 0.8, 0.9],
                    [0.2, 0.3, 0.4],
                    [0.5, 0.6, 0.7],
                ]
            ),
            "lattice": paddle.to_tensor(
                np.stack([np.eye(3) * 5.0, np.eye(3) * 4.0]).astype("float32")
            ),
        }
    }


def _write_package(directory, model_type, package_name, model_cfg=None):
    """Write a package with the registered archive layout."""
    package_dir = pathlib.Path(directory) / package_name
    (package_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    config = _load_config(model_type)
    config["Model"] = (
        model_cfg if model_cfg is not None else _tiny_model_cfg(model_type)
    )
    config["Model"]["__init_params__"]["training_cfg"]["qualification_cfg"] = None
    config["Trainer"]["pretrained_model_path"] = None
    with paddle.utils.unique_name.guard():
        paddle.seed(_DISPATCH_SEED)
        model = build_model(config["Model"])
    paddle.save(model.state_dict(), str(package_dir / "checkpoints" / _WEIGHTS_NAME))
    OmegaConf.save(OmegaConf.create(config), package_dir / f"{package_name}.yaml")
    return package_name, package_dir


def _denoise_inputs(model_type, model):
    num_atoms = paddle.to_tensor([2], dtype="int64")
    batch_idx = paddle.repeat_interleave(paddle.arange(1), repeats=num_atoms)
    lattice = paddle.eye(3).unsqueeze(0) * 5.43
    frac_coords = paddle.to_tensor(
        [[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]], dtype="float32"
    )
    atom_types = paddle.to_tensor([14, 14], dtype="int64")
    if model_type == "mattergen":
        noise_batch = {
            "frac_coords": frac_coords,
            "lattice": lattice,
            "atom_types": atom_types,
            "num_atoms": num_atoms,
            "batch": batch_idx,
        }
        return (noise_batch, paddle.full([1], 0.5, dtype="float32"))
    time_emb = model.time_embedding(
        paddle.full([1], model.num_train_timesteps // 2, dtype="int64")
    )
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


def _structures():
    return [
        _make_structure(
            ["Li", "Mn", "O"], [[0, 0, 0], [0.5, 0.5, 0.5], [0.25, 0.25, 0.25]]
        ),
        _make_structure(["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]], a=4.5),
    ]


def _make_loader(structures=None, rewards=None):
    structures = _structures() if structures is None else structures
    rewards = np.array([0.8, 0.4]) if rewards is None else rewards
    return paddle.io.DataLoader(
        RLDataset(structures, rewards),
        batch_size=len(structures),
        collate_fn=DefaultCollator(),
        shuffle=False,
    )


class _FakeSampler:
    def __init__(self, structures):
        self.structures = structures

    def generate(self, model):
        return [copy.deepcopy(structure) for structure in self.structures]


class _FakeReward:
    threshold = 0.5

    def scoring(self, structures, label):
        count = len(structures)
        return (
            np.linspace(0.9, 0.5, count),
            {"density": np.linspace(2.0, 1.0, count)},
            np.zeros(count, dtype=bool),
        )


def _round_config(model_type, tmp_path, package_name):
    config = _load_config(model_type)
    config["Model"] = _tiny_model_cfg(model_type)
    params = config["Model"]["__init_params__"]["training_cfg"]
    params["qualification_cfg"] = None
    params["topk_ratio"] = 0.5
    params["replay_cfg"] = None
    params["sample_cfg"] = {
        "data_cfg": {
            "dataset": {
                "__class_name__": "NumAtomsCrystalDataset",
                "__init_params__": {"total_num": 2, "dist_name": "MP_20"},
            },
            "sampler": {
                "__class_name__": "BatchSampler",
                "__init_params__": {"batch_size": 2, "shuffle": False},
            },
        },
        "num_inference_steps": 2,
    }
    params["finetune_cfg"] = {
        "batch_size": 2,
        "epochs": 1,
        "timesteps": 2,
        "accum_steps": 2,
        "sigma": 0.025,
    }
    params["anchor_cfg"] = {
        "num_samples": 2,
        "resample": False,
        "loss_weight": 1.0,
        "every_n_steps": 1,
    }
    params["early_stop_cfg"]["enabled"] = False
    config["Trainer"]["output_dir"] = str(pathlib.Path(tmp_path) / "output")
    config["Trainer"]["max_epochs"] = 2
    config["Trainer"]["gradient_accumulation_steps"] = 2
    return config


def _round_loader(config, tmp_path):
    path = pathlib.Path(tmp_path) / "anchors.csv"
    with path.open("w") as file:
        writer = csv.DictWriter(file, fieldnames=["cif"])
        writer.writeheader()
        writer.writerows(
            {"cif": structure.to(fmt="cif")} for structure in _structures()
        )
    data_cfg = copy.deepcopy(config["Dataset"]["train"])
    params = data_cfg["dataset"]["__init_params__"]
    params["path"] = str(path)
    params["cache_path"] = str(pathlib.Path(tmp_path) / "anchor_cache")
    params["build_structure_cfg"]["num_cpus"] = 1
    params["finetune_cfg"] = config["Model"]["__init_params__"]["training_cfg"][
        "finetune_cfg"
    ]
    return build_dataloader(data_cfg)


def _build_trainer(config, model, loader, monkeypatch, package_dir):
    optimizer, lr_scheduler = build_optimizer(
        config["Optimizer"], model, config["Trainer"]["max_epochs"], len(loader)
    )
    config["Trainer"]["pretrained_model_path"] = str(package_dir)
    config["Trainer"]["pretrained_weight_name"] = _WEIGHTS_NAME
    return BaseTrainer(
        config["Trainer"],
        model,
        train_dataloader=loader,
        val_dataloader=_make_loader(),
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
    )


def _make_round_env(monkeypatch, tmp_path, model_type, timesteps=2, epochs=1):
    package_name, package_dir = _write_package(
        tmp_path, model_type, f"test_{model_type}"
    )
    config = _round_config(model_type, tmp_path, package_name)
    params = config["Model"]["__init_params__"]["training_cfg"]
    params["finetune_cfg"].update(timesteps=timesteps, epochs=epochs)
    with paddle.utils.unique_name.guard():
        model = build_model(config["Model"])
    loader = _round_loader(config, tmp_path)
    original_init = MatInventTraining.__init__

    def initialize(context, trainer, cfg):
        original_init(context, trainer, cfg)
        context.loop.sampler = _FakeSampler(_structures())
        context.loop.reward = _FakeReward()

    monkeypatch.setattr(MatInventTraining, "__init__", initialize)
    trainer = _build_trainer(config, model, loader, monkeypatch, package_dir)
    return config, trainer, loader


class TestConfigContract:
    """The task configs follow the framework block and injection contract."""

    @pytest.mark.parametrize("model_type", list(_TASK_CONFIG))
    def test_blocks_and_injectable_params(self, model_type):
        config = _load_config(model_type)
        assert list(config) == [
            "Global",
            "Trainer",
            "Optimizer",
            "Dataset",
            "Model",
            "Sample",
        ]
        assert set(config["Dataset"]) == {"train", "val", "test"}
        assert "lr" in config["Optimizer"]["__init_params__"]
        assert "__class_name__" not in config["Trainer"]
        assert config["Sample"]["model_sample_params"]["num_inference_steps"] > 0
        assert config["Sample"]["build_structure_cfg"]["format"] == "array"
        model = build_model(config["Model"])
        assert isinstance(model, _MODEL_CLASS[model_type])
        if model_type == "mattergen":
            for block in model.model.gemnet.out_blocks:
                assert not {
                    "dense_rbf",
                    "scale_sum",
                    "layers",
                    "seq_energy",
                    "out_energy",
                } & set(block._sub_layers)
        else:
            assert not any(".prop_mlp." in name for name in model.state_dict())
        for split in ("train", "val", "test"):
            transforms = config["Dataset"][split]["dataset"]["__init_params__"][
                "transforms"
            ]
            if model_type == "mattergen":
                assert transforms[0]["__class_name__"] == "LatticePolarDecomposition"
            else:
                assert transforms == []

    @pytest.mark.parametrize("model_type", list(_TASK_CONFIG))
    def test_training_recipe_matches_dataset_and_accumulation(self, model_type):
        config = _load_config(model_type)
        recipe = config["Model"]["__init_params__"]["training_cfg"]
        assert "pretrained_model_name" not in recipe
        assert (
            config["Dataset"]["train"]["dataset"]["__init_params__"]["finetune_cfg"]
            == recipe["finetune_cfg"]
        )
        assert (
            config["Trainer"]["gradient_accumulation_steps"]
            == recipe["finetune_cfg"]["accum_steps"]
        )


class TestModelContracts:
    """Backbones share the forward/sample/denoise contract of the framework."""

    @pytest.mark.parametrize("model_type", list(_TASK_CONFIG))
    def test_prior_regularization_reuses_corruption(self, model_type):
        agent = _make_model(model_type)
        prior = copy.deepcopy(agent)
        prior.eval()
        for parameter in prior.parameters():
            parameter.stop_gradient = True
        agent.train()
        batch = _batch()["structure_array"]
        times = paddle.full(
            [2],
            0.5 if agent.continuous_timesteps else agent.num_train_timesteps // 2,
            dtype="float32" if agent.continuous_timesteps else "int64",
        )
        denoised = agent.denoise_with_time(batch, times)
        with paddle.no_grad():
            reference = prior.denoise_with_time(
                batch, times, noise_state=denoised["noise_state"]
            )
        assert reference["noise_state"] is denoised["noise_state"]
        for name, value in denoised["pred_dict"].items():
            np.testing.assert_array_equal(
                value.numpy(), reference["pred_dict"][name].numpy()
            )
        for name, value in denoised["loss_dict"].items():
            np.testing.assert_array_equal(
                value.numpy(), reference["loss_dict"][name].numpy()
            )
        loss = (
            agent.lattice_loss_weight * denoised["loss_dict"]["loss_lattice"].mean()
            + agent.coord_loss_weight * denoised["loss_dict"]["loss_coord"].mean()
        )
        if "loss_atom_type" in denoised["loss_dict"]:
            loss += (
                agent.atom_loss_weight * denoised["loss_dict"]["loss_atom_type"].mean()
            )
        loss.backward()
        active = [p for p in agent.parameters() if not p.stop_gradient]
        assert all(p.grad is not None for p in active)
        assert all(bool(paddle.isfinite(p.grad).all()) for p in active)
        assert all(p.grad is None for p in prior.parameters())

    @pytest.mark.parametrize("model_type", ["mattergen", "diffcsp", "diffcsp_joint"])
    def test_forward_composes_denoise_with_time(self, model_type):
        model = _make_model(model_type)
        batch = _batch()
        times = paddle.full(
            [2],
            0.5 if model.continuous_timesteps else model.num_train_timesteps // 2,
            dtype="float32" if model.continuous_timesteps else "int64",
        )
        paddle.seed(_DISPATCH_SEED)
        denoised = model.denoise_with_time(batch["structure_array"], times)
        loss_dict = denoised["loss_dict"]
        expected = model.lattice_loss_weight * float(
            loss_dict["loss_lattice"].mean()
        ) + model.coord_loss_weight * float(loss_dict["loss_coord"].mean())
        if "loss_atom_type" in loss_dict:
            expected += model.atom_loss_weight * float(
                loss_dict["loss_atom_type"].mean()
            )
        paddle.seed(_DISPATCH_SEED)
        if model.continuous_timesteps:
            sampler = mock.Mock(return_value=times)
            with mock.patch.object(model, "timestep_sampler", sampler):
                forward = model(batch)["loss_dict"]
        else:
            with mock.patch(
                "ppmat.models.matinvent.matinvent.uniform_sample_t",
                return_value=times,
            ):
                forward = model(batch)["loss_dict"]
        assert float(forward["loss"]) == pytest.approx(expected, rel=1e-6)
        for key in ("loss_lattice", "loss_coord"):
            assert tuple(loss_dict[key].shape) == (2,)
        assert denoised["pred_dict"]["lattice"].shape[0] == 2
        assert denoised["pred_dict"]["frac_coords"].shape[1] == 3

    def test_joint_type_channel_uses_atom_scheduler(self):
        model = _make_model("diffcsp_joint")
        batch = _batch()
        with mock.patch.object(
            model.atom_scheduler, "add_noise", wraps=model.atom_scheduler.add_noise
        ) as atom_add_noise, mock.patch.object(
            model.lattice_scheduler,
            "add_noise",
            wraps=model.lattice_scheduler.add_noise,
        ) as lattice_add_noise:
            model.denoise_with_time(
                batch["structure_array"], paddle.to_tensor([5, 5], dtype="int64")
            )
        atom_add_noise.assert_called_once()
        lattice_add_noise.assert_called_once()
        assert atom_add_noise.call_args.args[0].shape == (5, 100)

        with mock.patch.object(
            model.atom_scheduler, "step", wraps=model.atom_scheduler.step
        ) as atom_step:
            result = model.sample(
                {
                    "structure_array": {
                        "num_atoms": paddle.to_tensor([2], dtype="int64")
                    }
                },
                num_inference_steps=2,
            )
        assert atom_step.call_count == 2
        np.testing.assert_array_equal(
            model.atom_scheduler.timesteps.numpy(),
            model.lattice_scheduler.timesteps.numpy(),
        )
        assert len(result["result"]) == 1
        assert len(result["result"][0]["atom_types"]) == 2

    @pytest.mark.parametrize("model_type", ["diffcsp", "diffcsp_joint"])
    def test_sample_result_contract(self, model_type):
        # The MatterGen sampling loop is upstream code; MatInvent only adapts its
        # interaction graph, which TestMatterGenBackbone pins numerically.
        model = _make_model(model_type)
        inputs = {"num_atoms": paddle.to_tensor([2], dtype="int64")}
        if not model.supports_num_atoms_sampling:
            inputs["atom_types"] = paddle.to_tensor([14, 14], dtype="int64")
        result = model.sample({"structure_array": inputs}, num_inference_steps=2)
        (sample,) = result["result"]
        assert sample["num_atoms"] == len(sample["atom_types"])
        assert np.isfinite(sample["frac_coords"]).all()
        assert np.isfinite(sample["lattice"]).all()
        assert all(1 <= value <= 100 for value in sample["atom_types"])

    def test_timestep_regime_flags(self):
        assert _make_model("mattergen").continuous_timesteps is True
        assert _make_model("diffcsp").continuous_timesteps is False
        assert _make_model("diffcsp").supports_num_atoms_sampling is False
        assert _make_model("diffcsp_joint").supports_num_atoms_sampling is True

    def test_sample_rejects_unknown_kwargs(self):
        model = _make_model("diffcsp_joint")
        with pytest.raises(TypeError):
            model.sample(
                {
                    "structure_array": {
                        "num_atoms": paddle.to_tensor([2], dtype="int64")
                    }
                },
                num_inference_steps=1,
                unexpected_option=True,
            )


class TestDataFlow:
    """The RL data pipeline keeps the framework structure_array contract."""

    def test_rl_dataset_contract(self):
        dataset = RLDataset(_structures(), np.array([0.7, 0.2], dtype=float))
        item = dataset[1]
        structure_array = item["structure_array"]
        assert set(structure_array) == {
            "frac_coords",
            "atom_types",
            "lattice",
            "lengths",
            "angles",
            "num_atoms",
        }
        for key in ("frac_coords", "atom_types", "lattice", "lengths", "angles"):
            assert isinstance(structure_array[key], ConcatData)
        assert item["reward"] == pytest.approx(0.2)
        assert "reward" not in structure_array
        lattice = dataset.structures[1].lattice
        np.testing.assert_allclose(
            structure_array["lengths"].data, np.array(lattice.abc).reshape(1, 3)
        )
        np.testing.assert_allclose(
            structure_array["angles"].data, np.array(lattice.angles).reshape(1, 3)
        )

        batch = to_tensor_batch(DefaultCollator()([dataset[0], dataset[1]]))
        assert tuple(batch["reward"].shape) == (2,)
        assert batch["structure_array"]["num_atoms"].numpy().tolist() == [3, 2]
        assert tuple(batch["structure_array"]["lattice"].shape) == (2, 3, 3)

    def test_rl_dataset_applies_transforms(self):
        transforms = build_transforms(
            [{"__class_name__": "LatticePolarDecomposition", "__init_params__": {}}]
        )
        structure = _make_structure(["Si", "Si"], [[0, 0, 0], [0.3, 0.2, 0.1]], a=5.0)
        dataset = RLDataset([structure], np.array([0.5]), transforms=transforms)
        matrix = dataset[0]["structure_array"]["lattice"].data.reshape(3, 3)
        np.testing.assert_allclose(matrix, matrix.T, atol=1e-6)

    def test_dataset_to_forward_loss(self):
        model = _make_model("diffcsp_joint")
        loader = _make_loader(rewards=np.array([0.8, 0.3], dtype="float32"))
        batch = to_tensor_batch(next(iter(loader)))
        loss = model(batch)["loss_dict"]["loss"]
        assert np.isfinite(float(loss))
        loss.backward()
        assert model.decoder.node_embedding.weight.grad is not None


class TestRewardScoring:
    """Reward scoring and the calculator injection contract."""

    def test_calculator_protocol(self, tmp_path):
        structures = [_make_structure(["Si"], [[0, 0, 0]])]
        missing_params = [
            {
                "name": "density",
                "target": "ascending",
                "minv": 0,
                "maxv": 10,
                "calculator": {"__class_name__": "PyMatGen", "task": "density"},
            }
        ]
        reward = Reward(str(tmp_path), missing_params, reward_threshold=0.5)
        with pytest.raises(KeyError):
            reward.scoring(structures)

        unknown_key = copy.deepcopy(missing_params)
        unknown_key[0]["calculator"] = {
            "__class_name__": "PyMatGen",
            "__init_params__": {"task": "density"},
            "task": "density",
        }
        with pytest.raises(ValueError, match="Unexpected calculator keys"):
            Reward(str(tmp_path), unknown_key, reward_threshold=0.5).scoring(structures)

        owned_root = copy.deepcopy(missing_params)
        owned_root[0]["calculator"] = {
            "__class_name__": "PyMatGen",
            "__init_params__": {"task": "density", "root_dir": str(tmp_path)},
        }
        with pytest.raises(ValueError, match="root_dir is owned by Reward"):
            Reward(str(tmp_path), owned_root, reward_threshold=0.5).scoring(structures)

    def test_properties_have_independent_result_files(self, tmp_path):
        structures = [_make_structure(["Si"], [[0, 0, 0]])]
        properties = [
            {
                "name": "density",
                "target": "ascending",
                "minv": 0,
                "maxv": 10,
                "calculator": {
                    "__class_name__": "PyMatGen",
                    "__init_params__": {"task": "density"},
                },
            },
            {
                "name": "mcia",
                "target": "descending",
                "minv": 0,
                "maxv": 100,
                "calculator": {
                    "__class_name__": "PyMatGen",
                    "__init_params__": {
                        "task": "mcia",
                        "substrate_path": str(_SUBSTRATE_CIF),
                    },
                },
            },
            {
                "name": "energy",
                "target": "descending",
                "minv": 0,
                "maxv": 10,
                "calculator": {
                    "__class_name__": "DFTCalc",
                    "__init_params__": {"task": "energy"},
                },
            },
        ]
        reward = Reward(str(tmp_path), properties, reward_threshold=0.5)
        with pytest.raises(FileNotFoundError, match="DFT results are missing"):
            reward.scoring(structures, label="batch")
        (energy_path,) = (tmp_path / "energy" / "batch").glob("*/0.cif")
        energy_path = energy_path.parent / "results.txt"
        np.savetxt(energy_path, [1.23])
        _, values, failed = reward.scoring(structures, label="batch")
        assert not failed.any()
        assert np.isfinite(values["mcia"]).all() and values["mcia"][0] > 0
        np.testing.assert_allclose(values["energy"], [1.23])
        np.testing.assert_allclose(values["density"], [float(structures[0].density)])
        assert (tmp_path / "density" / "batch.txt").is_file()

    def test_density_reward_and_failed_property(self, monkeypatch):
        structures = [
            _make_structure(["Si", "Si"], [[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]])
        ]

        class FailingCalculator:
            def __init__(self, root_dir, task=None):
                self.root_dir = root_dir

            def calc(self, structures, label="tmp"):
                return np.full(len(structures), np.nan)

        with tempfile.TemporaryDirectory() as tmp:
            prop_cfg = [
                {
                    "name": "density",
                    "target": "ascending",
                    "minv": 0.0,
                    "maxv": 10.0,
                    "calculator": {
                        "__class_name__": "PyMatGen",
                        "__init_params__": {"task": "density"},
                    },
                }
            ]
            reward = Reward(
                root_dir=tmp, prop_cfg=prop_cfg, reward_threshold=0.5, reduce="mean"
            )
            rewards, prop_dict, failed = reward.scoring(structures)
            assert not failed.any()
            assert 0.0 <= float(rewards[0]) <= 1.0
            assert np.isfinite(prop_dict["density"][0])

            monkeypatch.setitem(CALCULATOR_REGISTRY, "Failing", FailingCalculator)
            prop_cfg[0]["calculator"]["__class_name__"] = "Failing"
            failing_reward = Reward(
                root_dir=tmp, prop_cfg=prop_cfg, reward_threshold=0.5, reduce="mean"
            )
            rewards, _, failed = failing_reward.scoring(structures)
            assert failed.tolist() == [True]
            assert float(rewards[0]) == 0.0


class TestSampler:
    """The RL sampler follows the model capability instead of a type string."""

    @pytest.mark.parametrize("model_type", list(_TASK_CONFIG))
    def test_atom_types_follow_capability(self, model_type):
        model = _make_model(model_type)
        sampler = MatInventSampler(
            data_cfg=_load_config(model_type)["Model"]["__init_params__"][
                "training_cfg"
            ]["sample_cfg"]["data_cfg"],
            num_inference_steps=1,
            build_structure_cfg={"format": "array", "niggli": False},
        )
        with mock.patch.object(model, "sample", return_value={"result": []}) as sample:
            sampler.generate(model)
        inputs = sample.call_args.args[0]["structure_array"]
        assert "num_atoms" in inputs
        assert ("atom_types" in inputs) is (not model.supports_num_atoms_sampling)


class TestModelPackage:
    """Registered packages hold native model weights and the task config."""

    def test_package_builds_loads_and_samples(self, tmp_path):
        package_name, package_dir = _write_package(tmp_path, "diffcsp_joint", "pkg")
        package_config = _read_config(package_dir / f"{package_name}.yaml")
        model = build_model(package_config["Model"])
        assert isinstance(model, MatinventDiffCSP)
        assert all(not key.startswith("backbone.") for key in model.state_dict())
        saved = paddle.load(str(package_dir / "checkpoints" / _WEIGHTS_NAME))
        model.set_state_dict(saved)
        sample = model.sample(
            {"structure_array": {"num_atoms": paddle.to_tensor([2], dtype="int64")}},
            num_inference_steps=2,
        )
        assert len(sample["result"]) == 1
        assert np.isfinite(sample["result"][0]["frac_coords"]).all()

    def test_package_yaml_matches_repo_schema(self, tmp_path):
        package_name, package_dir = _write_package(tmp_path, "mattergen", "pkg")
        package_config = _read_config(package_dir / f"{package_name}.yaml")
        for block in ("Global", "Trainer", "Optimizer", "Dataset", "Model", "Sample"):
            assert block in package_config
        assert "__class_name__" not in package_config["Trainer"]


class TestTrainer:
    """The unmodified BaseTrainer owns RL optimization and checkpointing."""

    @pytest.mark.parametrize("model_type", list(_TASK_CONFIG))
    @pytest.mark.parametrize("timesteps,epochs", [(2, 1), (3, 2)])
    def test_native_trainer_runs_rl_rounds(
        self, monkeypatch, tmp_path, model_type, timesteps, epochs
    ):
        config, trainer, loader = _make_round_env(
            monkeypatch, tmp_path, model_type, timesteps, epochs
        )
        assert type(trainer) is BaseTrainer
        assert len(loader) == epochs * ((timesteps + 1) // 2) * 2
        trainer.train()
        context = trainer.model._training_context
        assert trainer.state.epoch == 2
        assert trainer.state.global_step == 2 * epochs * ((timesteps + 1) // 2)
        assert context.loop.ltm.memory.shape[0] == 4
        assert context.metrics["reward_mean"] == pytest.approx(0.7)
        assert not context.stopped
        assert all(bool(paddle.isfinite(p).all()) for p in trainer.model.parameters())
        checkpoint = pathlib.Path(config["Trainer"]["output_dir"]) / "checkpoints"
        saved = paddle.load(str(checkpoint / "epoch_1.pdparams"))
        assert "_matinvent_training" in saved
        assert not list(checkpoint.glob("*.rlstate"))
        assert all(
            p.stop_gradient and p.grad is None for p in context.loop.prior.parameters()
        )

    def test_optimizer_and_backward_are_owned_by_base_trainer(
        self, monkeypatch, tmp_path
    ):
        _, trainer, _ = _make_round_env(monkeypatch, tmp_path, "diffcsp_joint")
        step = trainer.optimizer.step
        with mock.patch.object(trainer.optimizer, "step", wraps=step) as update:
            trainer.train()
        assert update.call_count == trainer.state.global_step == 2
        context = trainer.model._training_context
        assert "optimizer" not in vars(context.loop)

    def test_partial_accumulation_matches_explicit_optimizer_updates(
        self, monkeypatch, tmp_path
    ):
        config, trainer, _ = _make_round_env(
            monkeypatch, tmp_path, "diffcsp_joint", timesteps=3, epochs=2
        )
        trainer.max_epochs = 1
        optimizer_cfg = {"__class_name__": "SGD", "__init_params__": {"lr": 1e-4}}
        trainer.optimizer, _ = build_optimizer(optimizer_cfg, trainer.model, 1, 1)
        reference = copy.deepcopy(trainer.model)
        optimizer, _ = build_optimizer(optimizer_cfg, reference, 1, 1)
        trainer.model.before_train(trainer)
        context = trainer.model._training_context
        denoise = trainer.model.denoise_with_time
        corruptions = []

        def record_corruption(structure_array, times, noise_state=None):
            result = denoise(structure_array, times, noise_state=noise_state)
            corruptions.append((times, result["noise_state"]))
            return result

        monkeypatch.setattr(trainer.model, "denoise_with_time", record_corruption)
        trainer.train()
        recorded = iter(corruptions)
        reference_denoise = reference.denoise_with_time

        def replay_corruption(structure_array, times, noise_state=None):
            recorded_times, state = next(recorded)
            return reference_denoise(structure_array, recorded_times, noise_state=state)

        monkeypatch.setattr(reference, "denoise_with_time", replay_corruption)
        loop = copy.copy(context.loop)
        loop.agent = reference
        reference.train()
        for _ in range(2):
            for index in range(3):
                losses = loop.compute_loss(context.batch, index)
                (losses["objective"] / (2 if index < 2 else 1)).backward()
                if index in (1, 2):
                    optimizer.step()
                    optimizer.clear_grad()
        for name, actual in trainer.model.named_parameters():
            np.testing.assert_allclose(
                actual.numpy(),
                reference.state_dict()[name].numpy(),
                atol=1e-7,
                rtol=1e-6,
            )

    def test_resume_restores_model_prior_and_replay(self, monkeypatch, tmp_path):
        config, trainer, loader = _make_round_env(monkeypatch, tmp_path, "diffcsp")
        with paddle.utils.unique_name.guard():
            trainer.train()
        checkpoint = pathlib.Path(trainer.output_dir) / "checkpoints" / "epoch_1"
        saved = paddle.load(str(checkpoint) + ".pdparams")
        inference = build_model(config["Model"])
        assert inference.set_state_dict(saved) == ([], [])
        assert inference._training_context is None
        sampled = inference.sample(_batch(), num_inference_steps=2)
        assert len(sampled["result"]) == 2
        assert np.isfinite(sampled["result"][0]["frac_coords"]).all()
        config["Trainer"]["output_dir"] = str(tmp_path / "resumed")
        config["Trainer"]["resume_from_checkpoint"] = str(checkpoint)
        config["Trainer"]["max_epochs"] = 1
        with paddle.utils.unique_name.guard():
            model = build_model(config["Model"])
        resumed = _build_trainer(
            config, model, loader, monkeypatch, tmp_path / "test_diffcsp"
        )
        with paddle.utils.unique_name.guard():
            resumed.train()
        assert resumed.state.global_step == 1
        context = model._training_context
        assert context.loop.step == 0
        assert context.loop.ltm.memory.shape[0] == 2
        np.testing.assert_array_equal(
            model.coord_scheduler.discrete_sigmas_norm.numpy(),
            trainer.model.coord_scheduler.discrete_sigmas_norm.numpy(),
        )
        np.testing.assert_array_equal(
            context.loop.prior.coord_scheduler.discrete_sigmas_norm.numpy(),
            model.coord_scheduler.discrete_sigmas_norm.numpy(),
        )
        for name, parameter in model.named_parameters():
            np.testing.assert_array_equal(parameter.numpy(), saved[name].numpy())
        for name, value in context.loop.prior.state_dict().items():
            np.testing.assert_array_equal(
                value.numpy(),
                trainer.model._training_context.loop.prior.state_dict()[name].numpy(),
            )
        resumed.max_epochs = 2
        with paddle.utils.unique_name.guard():
            resumed.train()
        assert resumed.state.global_step == 2
        assert context.loop.ltm.memory.shape[0] == 4

    def test_early_stop_runs_after_update_and_saves_native_checkpoint(
        self, monkeypatch, tmp_path
    ):
        _, trainer, _ = _make_round_env(monkeypatch, tmp_path, "diffcsp", timesteps=4)
        trainer.model.before_train(trainer)
        context = trainer.model._training_context
        context.loop.early_stop_enabled = True
        context.loop.probe_initial = {
            "loss": 1.0,
            "loss_lattice": 1.0,
            "loss_coord": 1.0,
            "loss_atom_type": 1.0,
        }
        context.loop._probe_anchor = lambda: {
            "loss": 10.0,
            "loss_lattice": 10.0,
            "loss_coord": 10.0,
            "loss_atom_type": 10.0,
        }
        trainer.train()
        assert context.stopped
        assert trainer.state.global_step == 1
        assert trainer.state.epoch == 1
        checkpoint = pathlib.Path(trainer.output_dir) / "checkpoints"
        assert (checkpoint / "epoch_1.pdparams").is_file()
        assert not (checkpoint / "epoch_2.pdparams").exists()

    @pytest.mark.parametrize(
        "mutation,message",
        [
            ({"gradient_accumulation_steps": 1}, "accumulation"),
            ({"eval_freq_steps": 1}, "eval_freq_steps"),
            ({"save_freq_steps": 1}, "save_freq_steps"),
            ({"max_iter": 1}, "max_iter"),
        ],
    )
    def test_rejects_unsupported_training_config(
        self, monkeypatch, tmp_path, mutation, message
    ):
        _, trainer, _ = _make_round_env(monkeypatch, tmp_path, "diffcsp")
        trainer.config.update(mutation)
        trainer.gradient_accumulation_steps = trainer.config[
            "gradient_accumulation_steps"
        ]
        with pytest.raises(ValueError, match=message):
            trainer.train()

    def test_empty_population_does_not_update_optimizer(self, monkeypatch, tmp_path):
        _, trainer, _ = _make_round_env(monkeypatch, tmp_path, "diffcsp")
        trainer.model.before_train(trainer)
        trainer.model._training_context.loop.sampler = _FakeSampler([])
        with mock.patch.object(trainer.optimizer, "step") as update:
            with pytest.raises(RuntimeError, match="No scored structures"):
                trainer.train()
        update.assert_not_called()

    def test_evaluation_and_sampling_do_not_initialize_rl(self, tmp_path):
        config = _load_config("diffcsp")["Trainer"]
        config["output_dir"] = str(tmp_path)
        config["pretrained_model_path"] = None
        model = _make_model("diffcsp")
        trainer = BaseTrainer(config, model, val_dataloader=_make_loader())
        _, losses, _ = trainer.eval(trainer.val_dataloader)
        assert np.isfinite(losses["loss"].avg)
        assert model._training_context is None


class TestMatterGenBackbone:
    """The MatInvent MatterGen variant keeps the mainline graph numerics."""

    def test_paddle_graph_keeps_mainline_numerics(self):
        mainline_cfg = copy.deepcopy(_tiny_model_cfg("mattergen"))
        mainline_cfg["__class_name__"] = "MatterGen"
        mainline_cfg["__init_params__"].pop("training_cfg")
        with paddle.utils.unique_name.guard():
            paddle.seed(_DISPATCH_SEED)
            mainline = build_model(mainline_cfg)
        variant = _make_model("mattergen")
        full_state = mainline.state_dict()
        compact_keys = variant.state_dict().keys()
        assert variant.set_state_dict(
            {name: full_state[name].clone() for name in compact_keys}
        ) == ([], [])
        removed = full_state.keys() - compact_keys
        assert removed
        assert all(".out_blocks." in name for name in removed)
        mainline.eval()
        variant.eval()

        structure_array = _batch()["structure_array"]
        structure_array["batch"] = paddle.to_tensor([0, 0, 0, 1, 1], dtype="int64")
        times = paddle.full([2], 0.7, dtype="float32")
        expected = mainline._denoise(structure_array, times)
        actual = variant._denoise(structure_array, times)
        for key in expected:
            np.testing.assert_allclose(
                actual[key].numpy(), expected[key].numpy(), atol=0, rtol=0
            )
        paddle.seed(_DISPATCH_SEED)
        expected_loss = mainline(_batch())["loss_dict"]
        paddle.seed(_DISPATCH_SEED)
        actual_loss = variant(_batch())["loss_dict"]
        for name, loss in expected_loss.items():
            np.testing.assert_allclose(
                actual_loss[name].numpy(), loss.numpy(), atol=0, rtol=0
            )
        actual_loss["loss"].backward()
        active = [p for p in variant.parameters() if not p.stop_gradient]
        assert all(p.grad is not None for p in active)
        assert all(bool(paddle.isfinite(p.grad).all()) for p in active)
        optimizer = paddle.optimizer.Adam(parameters=active, learning_rate=1e-6)
        optimizer.step()
        assert all(bool(paddle.isfinite(p).all()) for p in active)


class TestEdgelessGraph:
    """Edgeless periodic graphs occur when a sampled cell is too large."""

    def test_mattergen_denoises_edgeless_structures(self):
        model = _make_model("mattergen")
        structure_array = {
            "num_atoms": paddle.to_tensor([2], dtype="int64"),
            "atom_types": paddle.to_tensor([6, 8], dtype="int64"),
            "frac_coords": paddle.to_tensor([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]),
            "lattice": paddle.to_tensor(np.eye(3, dtype="float32")[None] * 40.0),
        }
        graph = model._build_interaction_graph(structure_array)
        assert tuple(graph[0].shape) == (2, 0)
        assert int(graph[1].sum()) == 0
        structure_array["batch"] = paddle.to_tensor([0, 0], dtype="int64")
        predictions = model._denoise(
            structure_array, paddle.full([1], 0.5, dtype="float32")
        )
        assert all(bool(paddle.isfinite(value).all()) for value in predictions.values())
        denoised = model.denoise_with_time(
            structure_array, paddle.full([1], 0.5, dtype="float32")
        )
        for key, value in denoised["loss_dict"].items():
            assert np.isfinite(value.numpy()).all(), key
        for key, value in denoised["pred_dict"].items():
            assert np.isfinite(value.numpy()).all(), key


class TestQualification:
    """The stability qualification used by the joint round keeps a periodic graph."""

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
                assert "graph" not in atoms.info
                assert "graph" not in calculator.atoms.info
        finally:
            core.set_prim_eager_enabled(prim_enabled)


_BENCHMARK_ROOT = _REPO_ROOT / "benchmark" / "cinn"


def _load_benchmark_script(name):
    path = _BENCHMARK_ROOT / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"cinn_benchmark_{name}", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _cinn_gpu_available():
    return (
        paddle.is_compiled_with_cuda()
        and paddle.base.is_compiled_with_cinn()
        and paddle.device.cuda.device_count() > 0
    )


def _require_cinn_gpu():
    if not _cinn_gpu_available():
        pytest.skip("CINN execution needs a CUDA build, CINN support and a GPU device")
    paddle.set_device("gpu:0")


class TestCinnDispatch:
    """CINN boundary dispatch: compiled numbers must match eager exactly."""

    @pytest.mark.parametrize("model_type", ["mattergen", "diffcsp"])
    def test_boundary_dispatch_matches_eager(self, model_type):
        _require_cinn_gpu()
        eager = _make_model(model_type)
        cinn = _make_model(model_type, execution_backend="cinn")
        cinn.set_state_dict(_clone_state_dict(eager))
        eager.eval()
        cinn.eval()
        assert cinn.execution_backend == "cinn"

        inputs = _denoise_inputs(model_type, eager)
        boundary = "_denoise" if model_type == "mattergen" else "_decode"
        expected = getattr(eager, boundary)(*inputs)

        names = []
        original = type(cinn)._run_runtime

        def record(instance, name, function, *args, **kwargs):
            names.append(name)
            return original(instance, name, function, *args, **kwargs)

        with mock.patch.object(type(cinn), "_run_runtime", record):
            actual = getattr(cinn, boundary)(*inputs)
        assert names == ["denoise_step"]
        expected_values = list(_flatten_result(expected))
        actual_values = list(_flatten_result(actual))
        assert len(actual_values) == len(expected_values)
        for actual_value, expected_value in zip(actual_values, expected_values):
            # CINN may reassociate float32 operations, so only float noise remains.
            np.testing.assert_allclose(
                np.asarray(actual_value),
                np.asarray(expected_value),
                atol=1e-6,
                rtol=1e-6,
            )
        assert cinn.state_dict().keys() == eager.state_dict().keys()

    def test_sample_with_cinn_backend(self):
        _require_cinn_gpu()
        model = _make_model("diffcsp", execution_backend="cinn")
        model.eval()
        result = model.sample(
            {
                "structure_array": {
                    "num_atoms": paddle.to_tensor([2], dtype="int64"),
                    "atom_types": paddle.to_tensor([14, 14], dtype="int64"),
                }
            },
            num_inference_steps=2,
        )
        assert len(result["result"]) == 1
        assert ("cinn", "eval", "denoise_step") in model._runtime_cache


def _gate_gpu_cinn_rl_loop():
    if os.environ.get("PPMAT_RUN_CINN_WORKFLOW_TESTS") != "1":
        pytest.skip("Set PPMAT_RUN_CINN_WORKFLOW_TESTS=1 to run the GPU CINN test")
    if not _cinn_gpu_available():
        pytest.skip("CINN execution needs a CUDA build, CINN support and a GPU device")
    paddle.set_device("gpu:0")


def test_gpu_cinn_rl_loss_matches_eager(monkeypatch):
    _gate_gpu_cinn_rl_loop()
    eager = _make_model("mattergen")
    cinn = _make_model("mattergen", execution_backend="cinn")
    cinn.set_state_dict(_clone_state_dict(eager))
    batch = _batch()
    times = paddle.full([2], 0.6, dtype="float32")

    outputs = []
    for model in (eager, cinn):
        model.train()
        loss = model.denoise_with_time(batch["structure_array"], times)["loss_dict"][
            "loss_lattice"
        ].mean()
        loss.backward()
        gradients = {
            name: parameter.grad.numpy().copy()
            for name, parameter in model.named_parameters()
            if parameter.grad is not None
        }
        assert gradients and all(np.isfinite(g).all() for g in gradients.values())
        outputs.append((float(loss), gradients))
    assert ("cinn", "train", "denoise_step") in cinn._runtime_cache
    assert outputs[0][0] == pytest.approx(outputs[1][0], rel=2e-5, abs=2e-5)
    for name, gradient in outputs[0][1].items():
        np.testing.assert_allclose(gradient, outputs[1][1][name], atol=2e-5, rtol=2e-5)


class TestCinnBenchmarkWorkload:
    """The public CINN benchmark delegates the MatInvent workload to this package."""

    def test_family_name(self):
        comparison = _load_benchmark_script("compare_backends")
        assert comparison.family_name("matinvent_diffcsp_mp20") == "MatInvent"
        assert comparison.family_name("matinvent_mattergen_mp20") == "MatInvent"

    def test_build_workflow_returns_train_step(self, monkeypatch):
        cinn_benchmark = _load_benchmark_script("benchmark_registered")

        class _DummyModel(paddle.nn.Layer):
            def __init__(self):
                super().__init__()
                self.linear = paddle.nn.Linear(2, 2)

        monkeypatch.setattr(
            matinvent_benchmark,
            "build_model_from_name",
            lambda name: (_DummyModel(), {}),
        )
        monkeypatch.setattr(
            matinvent_benchmark,
            "configure_execution_backend",
            lambda *args, **kwargs: "eager",
        )

        workflow, call, scope = cinn_benchmark.build_workflow(
            "matinvent_diffcsp_mp20", "eager"
        )
        assert scope == matinvent_benchmark.TRAIN_SCOPE
        assert isinstance(workflow.model, _DummyModel)
        assert callable(call)

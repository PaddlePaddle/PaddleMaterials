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
from pathlib import Path

import numpy as np
import paddle
import pandas as pd
import pytest
from omegaconf import OmegaConf

from ppmat.datasets import build_dataloader
from ppmat.datasets.build_structure import BuildStructure
from ppmat.datasets.collate_fn import DefaultCollator
from ppmat.datasets.custom_data_type import ConcatData
from ppmat.metrics import StructGenMetric
from ppmat.models import build_model
from ppmat.schedulers import build_scheduler
from ppmat.trainer.base_trainer import BaseTrainer

_CONFIG_DIR = (
    Path(__file__).resolve().parents[2] / "structure_generation/configs/chemeleon2"
)


def _load_cfg(name, small=True):
    config = OmegaConf.load(_CONFIG_DIR / f"chemeleon2_mp20_{name}.yaml")
    if small:
        config.Global.vae_hidden_dim = 32
        config.Global.vae_num_layers = 1
        config.Global.vae_num_heads = 4
        config.Global.vae_feedforward_dim = 64
        if name != "vae":
            config.Model.__init_params__.vae_pretrained_model_name = None
            config.Global.dit_hidden_dim = 32
            config.Global.dit_num_layers = 1
            config.Global.dit_num_heads = 4
    return OmegaConf.to_container(config.Model, resolve=True)


def _structure_array(num_atoms_list):
    total = sum(num_atoms_list)
    return {
        "atom_types": paddle.randint(1, 90, [total]),
        "num_atoms": paddle.to_tensor(num_atoms_list),
        "frac_coords": paddle.rand([total, 3]),
        "lengths": paddle.to_tensor([[4.0, 5.0, 6.0] for _ in num_atoms_list]),
        "angles": paddle.to_tensor([[80.0, 90.0, 100.0] for _ in num_atoms_list]),
    }


def test_vae_training_pipeline():
    cfg = _load_cfg("vae")
    cfg["__init_params__"]["augmentation"] = {"translate": True, "rotate": True}
    cfg["__init_params__"]["noise"] = {"ratio": 0.4, "corruption_scale": 0.1}
    vae = build_model(cfg)
    batch = {"structure_array": _structure_array([3, 5])}
    converted = vae._convert_train_batch(batch)
    assert converted["cart_coords"].shape == [8, 3]
    optimizer = paddle.optimizer.Adam(learning_rate=1e-3, parameters=vae.parameters())
    initial = {key: value.clone() for key, value in vae.state_dict().items()}
    output = vae(batch)
    loss = output["loss_dict"]["loss"]
    assert paddle.isfinite(loss).item()
    loss.backward()
    for param in vae.parameters():
        assert param.grad is not None and paddle.isfinite(param.grad).all().item()
    optimizer.step()
    assert any(
        not paddle.equal_all(initial[key], value)
        for key, value in vae.state_dict().items()
    )
    assert all(
        paddle.isfinite(value).all().item() for value in vae.state_dict().values()
    )
    vae.eval()
    with paddle.no_grad():
        assert paddle.isfinite(vae(batch)["loss_dict"]["loss"]).item()


@pytest.mark.parametrize("learn_sigma", [True, False])
def test_ldm_training_and_sampling(learn_sigma):
    cfg = _load_cfg("ldm")
    params = cfg["__init_params__"]
    params["denoiser_cfg"]["__init_params__"]["learn_sigma"] = learn_sigma
    params["noise_scheduler_cfg"]["__init_params__"]["learn_sigma"] = learn_sigma
    ldm = build_model(cfg)
    optimizer = paddle.optimizer.Adam(learning_rate=1e-3, parameters=ldm.parameters())
    initial = ldm.denoiser.x_embedder.weight.clone()
    output = ldm({"structure_array": _structure_array([3, 5])})
    output["loss_dict"]["loss"].backward()
    for param in ldm.denoiser.parameters():
        assert param.grad is not None and paddle.isfinite(param.grad).all().item()
    assert all(
        param.stop_gradient and param.grad is None for param in ldm.vae.parameters()
    )
    optimizer.step()
    optimizer.clear_grad()
    assert not paddle.equal_all(initial, ldm.denoiser.x_embedder.weight)
    assert ldm.vae.training is False
    data = {"structure_array": {"num_atoms": paddle.to_tensor([3, 5])}}
    ldm.eval()
    for sampler in ("ddpm", "ddim"):
        with paddle.no_grad():
            result = ldm.sample(
                data, sampler=sampler, num_inference_steps=5, progress=False
            )
        assert len(result["result"]) == 2
        for count, structure in zip([3, 5], result["result"]):
            assert structure["num_atoms"] == count
            assert np.asarray(structure["frac_coords"]).shape == (count, 3)
            assert np.asarray(structure["lattice"]).shape == (3, 3)
            assert np.isfinite(structure["lattice"]).all()
    with pytest.raises(ValueError, match="by_chemical_formula"):
        ldm.sample({"structure_array": _structure_array([3])}, progress=False)
    with pytest.raises(ValueError):
        ldm.predict({"num_atoms": 3, "condition": "MgO"}, progress=False)
    with paddle.no_grad():
        result = ldm.predict(
            {"num_samples": 2, "num_atoms": [3, 5]},
            num_inference_steps=5,
            progress=False,
        )
    assert len(result["result"]) == 2

    diffusion = ldm.diffusion
    original = diffusion.scheduler.alphas_cumprod.numpy()
    sampling_cfg = copy.deepcopy(params["noise_scheduler_cfg"])
    sampling_cfg["__init_params__"]["timestep_respacing"] = "ddim50"
    spaced = build_scheduler(sampling_cfg)
    assert spaced.num_timesteps == spaced.scheduler.num_train_timesteps == 50
    np.testing.assert_allclose(
        spaced.scheduler.alphas_cumprod.numpy(),
        original[spaced.timestep_map],
        atol=1e-6,
    )
    x = paddle.full([1, 3, 8], 20.0)
    t = paddle.to_tensor([2])

    def zero_model(x, t, **kwargs):
        multiplier = 2 if learn_sigma else 1
        return paddle.zeros([1, multiplier * 3, 8])

    with paddle.no_grad():
        unbounded = spaced.p_sample(zero_model, x, t, clip_denoised=False)
        bounded = spaced.p_sample(zero_model, x, t, clip_denoised=True)
    assert unbounded["pred_xstart"].max().item() > 1
    assert bounded["pred_xstart"].max().item() == 1


def test_mp20_dataset_to_conditional_model(tmp_path):
    source = _CONFIG_DIR.parents[2] / "data/mp_20/train.csv"
    if not source.is_file():
        pytest.skip("Local MP20 data is required for the dataset integration test.")
    rows = pd.read_csv(source, nrows=20)
    selected = []
    counts = set()
    for _, row in rows.iterrows():
        structure = BuildStructure.build_one(
            row.cif, "cif_str", niggli=False, canocial=False
        )
        if len(structure) not in counts:
            selected.append(row)
            counts.add(len(structure))
        if len(selected) == 2:
            break
    assert len(selected) == 2
    path = tmp_path / "mp20_20261121204000.csv"
    pd.DataFrame(selected).to_csv(path, index=False)
    data_cfg = {
        "dataset": {
            "__class_name__": "MP20Dataset",
            "__init_params__": {
                "path": str(path),
                "property_names": ["band_gap"],
                "build_structure_cfg": {"format": "cif_str", "num_cpus": 1},
                "cache_path": str(tmp_path / "cache_20261121204000"),
            },
        },
        "sampler": {
            "__class_name__": "BatchSampler",
            "__init_params__": {"batch_size": 2},
        },
        "loader": {"num_workers": 0},
    }
    batch = next(iter(build_dataloader(data_cfg)))
    assert len(set(batch["structure_array"]["num_atoms"].tolist())) == 2
    cfg = _load_cfg("ldm")
    params = cfg["__init_params__"]
    params["condition_names"] = ["band_gap", "composition"]
    params["denoiser_cfg"]["__init_params__"]["condition_dim"] = 32
    params["property_embeddings_adapt_cfg"] = {
        "__class_name__": "Chemeleon2ConditionModule",
        "__init_params__": {
            "hidden_dim": 32,
            "p_unconditional": 0.1,
            "property_embeddings_adapt_cfg": {
                "band_gap": {"__class_name__": "ValueEncoder", "__init_params__": {}},
                "composition": {
                    "__class_name__": "CompositionEncoder",
                    "__init_params__": {"in_dim": 101},
                },
            },
        },
    }
    ldm = build_model(cfg)
    batch["composition"] = ["MgO", "FmO"]
    loss = ldm(batch)["loss_dict"]["loss"]
    loss.backward()
    assert paddle.isfinite(loss).item()
    assert all(param.grad is not None for param in ldm.condition_module.parameters())
    composition = ldm.condition_module.encoders["composition"]
    np.testing.assert_allclose(
        composition.preprocess(["MgO", "Mg2O", "FmO"]).sum(axis=1).numpy(), 1.0
    )
    ldm.eval()
    with paddle.no_grad():
        result = ldm.sample(
            {
                "structure_array": {"num_atoms": paddle.to_tensor([3])},
                "band_gap": 1.0,
                "composition": "MgO",
            },
            num_inference_steps=5,
            progress=False,
        )
    assert len(result["result"]) == 1


def test_generation_metrics_cross_batch():
    valid = {
        "atom_types": np.array([11, 11, 8]),
        "frac_coords": np.array([[0.1, 0.1, 0.1], [0.6, 0.1, 0.1], [0.6, 0.6, 0.1]]),
        "lattice": np.eye(3) * 5,
    }
    invalid = {**valid, "lattice": np.zeros((3, 3))}
    structure = BuildStructure.build_one(
        {**valid, "lattice": valid["lattice"].tolist()},
        "array",
        niggli=False,
        canocial=False,
    )
    metric = StructGenMetric(reference_structures=[structure])
    expected = {"validity": 2 / 3, "uniqueness": 0.5, "novelty": 0.0}
    assert metric([valid, valid, invalid]) == expected
    metric.update_step(result={"result": [valid]}, batch=None, stage="sample")
    metric.update_step(result={"result": [valid, invalid]}, batch=None, stage="sample")
    assert metric.compute_epoch(stage="sample") == expected
    metric.update_step(result={"result": [valid]}, batch=None, stage="eval")
    assert metric.compute_epoch(stage="eval")["validity"] == 1
    metric.reset()
    assert metric.compute_epoch(stage="sample")["validity"] == 0
    assert "novelty" not in StructGenMetric()([valid])


def test_ldm_trainer_resume_and_runtime(tmp_path, monkeypatch):
    from cinn_workflow_harness import patch_cinn_runtime
    from cinn_workflow_harness import trainer_config

    from ppmat.models import Chemeleon2LDMModule

    patch_cinn_runtime(monkeypatch, Chemeleon2LDMModule)
    batch = {"structure_array": _structure_array([3, 5])}
    loader = paddle.io.DataLoader(
        [
            {
                "structure_array": {
                    key: ConcatData(value.numpy())
                    for key, value in batch["structure_array"].items()
                }
            }
        ],
        batch_size=1,
        collate_fn=DefaultCollator(),
    )
    cfg = _load_cfg("ldm")
    with paddle.utils.unique_name.guard():
        model = build_model(cfg)
        optimizer = paddle.optimizer.Adam(
            learning_rate=1e-3, parameters=model.parameters()
        )
        trainer = BaseTrainer(
            trainer_config(tmp_path / "first", 1),
            model,
            train_dataloader=loader,
            val_dataloader=loader,
            optimizer=optimizer,
            execution_config={"backend": "cinn"},
        )
        trainer.train()
    checkpoint = tmp_path / "first/checkpoints/latest"
    for suffix in ("pdparams", "pdopt", "pdstates"):
        assert checkpoint.with_suffix("." + suffix).is_file()
    saved_model = paddle.load(str(checkpoint) + ".pdparams")
    saved_optimizer = paddle.load(str(checkpoint) + ".pdopt")
    with paddle.utils.unique_name.guard():
        resumed = build_model(cfg)
        resumed_optimizer = paddle.optimizer.Adam(
            learning_rate=1e-3, parameters=resumed.parameters()
        )
        config = trainer_config(tmp_path / "resumed", 2)
        resumed_trainer = BaseTrainer(
            config,
            resumed,
            train_dataloader=loader,
            val_dataloader=loader,
            optimizer=resumed_optimizer,
            execution_config={"backend": "cinn"},
        )
        original_step = resumed_optimizer.step
        checked = []

        def checked_step():
            if not checked:
                assert resumed_trainer.state.global_step == 2
                for key, value in resumed.state_dict().items():
                    np.testing.assert_array_equal(
                        value.numpy(), saved_model[key].numpy()
                    )
                restored = resumed_optimizer.state_dict()
                assert restored.keys() == saved_optimizer.keys()
                for key, value in saved_optimizer.items():
                    if isinstance(value, paddle.Tensor):
                        np.testing.assert_array_equal(
                            restored[key].numpy(), value.numpy()
                        )
                checked.append(True)
            original_step()

        monkeypatch.setattr(resumed_optimizer, "step", checked_step)
        resumed_trainer.train(resume_from_checkpoint=str(checkpoint))
    assert checked and resumed_trainer.state.global_step == 2
    assert resumed_trainer.state.epoch == 2
    assert any(
        not paddle.equal_all(value, saved_model[key])
        for key, value in resumed.state_dict().items()
    )
    assert all(
        paddle.isfinite(value).all().item() for value in resumed.state_dict().values()
    )


def test_ldm_cinn_dispatch_and_parity(monkeypatch):
    ldm = build_model(_load_cfg("ldm"))
    ldm.eval()
    data = {"structure_array": {"num_atoms": paddle.to_tensor([3, 5])}}
    keys = set(ldm.state_dict())
    paddle.seed(7)
    with paddle.no_grad():
        eager = ldm.sample(data, num_inference_steps=5, progress=False)
    boundaries = []

    def dispatch(name, function, *args, **kwargs):
        boundaries.append(name)
        return function(*args, **kwargs)

    monkeypatch.setattr(ldm.core, "_run_runtime", dispatch)
    ldm.core.set_execution_backend("cinn")
    assert set(ldm.state_dict()) == keys
    paddle.seed(7)
    with paddle.no_grad():
        cinn = ldm.sample(data, num_inference_steps=5, progress=False)
    assert boundaries and set(boundaries) == {"denoise_step"}
    for before, after in zip(eager["result"], cinn["result"]):
        for key in ("atom_types", "frac_coords", "lattice"):
            np.testing.assert_array_equal(before[key], after[key])


@pytest.mark.parametrize("stage", ["vae", "ldm"])
def test_released_weight_layout(stage, monkeypatch, tmp_path):
    from ppmat.models import build_model_from_name
    from ppmat.sampler import StructureSampler
    from ppmat.utils import download

    package = (
        Path.home()
        / ".paddlemat/weights"
        / f"chemeleon2_{stage}"
        / f"chemeleon2_{stage}"
    )
    checkpoint = package / "checkpoints/best.pdparams"
    if not checkpoint.is_file():
        pytest.skip("Cached registered weights are required for layout verification.")

    def cached_package(url):
        name = Path(url).stem
        cached = package.parents[1] / name / name
        if not cached.is_dir():
            pytest.skip(f"Cached package {name} is required.")
        return str(cached)

    monkeypatch.setattr(download, "get_weights_path_from_url", cached_package)
    state = paddle.load(str(checkpoint))
    model = build_model(_load_cfg(stage, small=False))
    assert model.state_dict().keys() == state.keys()
    for key, value in model.state_dict().items():
        assert value.shape == state[key].shape
    assert model.set_state_dict(state) == ([], [])
    del model
    legacy, _ = build_model_from_name(f"chemeleon2_{stage}")
    assert legacy.state_dict().keys() == state.keys()
    for key, value in legacy.state_dict().items():
        np.testing.assert_array_equal(value.numpy(), state[key].numpy())
    del legacy, state
    if stage == "ldm":
        sampler = StructureSampler(
            model_name="chemeleon2_ldm",
            config_overrides=["Sample.model_sample_params.num_inference_steps=5"],
        )
        paddle.seed(42)
        result = sampler.sample_by_num_atoms(20, save_path=str(tmp_path))
        assert len(result["result"]) == 1
        assert len(list(tmp_path.glob("*.cif"))) == 1


@pytest.mark.skipif(
    os.environ.get("PPMAT_RUN_CINN_WORKFLOW_TESTS") != "1",
    reason="Set PPMAT_RUN_CINN_WORKFLOW_TESTS=1 for GPU compilation.",
)
def test_ldm_cinn_gpu_compile_parity():
    if not (paddle.is_compiled_with_cuda() and paddle.base.is_compiled_with_cinn()):
        pytest.skip("CUDA Paddle with CINN is required.")
    previous_device = paddle.get_device()
    paddle.set_device("gpu")
    try:
        ldm = build_model(_load_cfg("ldm"))
        ldm.eval()
        x = paddle.randn([2, 5, 8])
        t = paddle.to_tensor([3, 3])
        mask = paddle.ones([2, 5], dtype="bool")
        with paddle.no_grad():
            eager = ldm._runtime_denoise(x, t, mask=mask)
            ldm.set_execution_backend("cinn")
            cinn = ldm._runtime_denoise(x, t, mask=mask)
        assert paddle.allclose(eager, cinn, atol=2e-5).item()
    finally:
        paddle.set_device(previous_device)


def test_rl_training_and_sampling():
    # Macro check: the full GRPO forward/backward and LDM-delegated sampling.
    paddle.seed(0)
    ldm_params = _load_cfg("ldm")["__init_params__"]
    model = build_model(
        {
            "__class_name__": "Chemeleon2",
            "__init_params__": {
                "stage": "rl",
                "denoiser_cfg": ldm_params["denoiser_cfg"],
                "noise_scheduler_cfg": ldm_params["noise_scheduler_cfg"],
                "vae_cfg": ldm_params["vae_cfg"],
                "rl_configs": {"num_group_samples": 2, "group_reward_norm": True},
                "reward_fn_cfg": {
                    "__class_name__": "ReinforceReward",
                    "__init_params__": {
                        "components": [
                            {
                                "__class_name__": "CreativityReward",
                                "__init_params__": {"weight": 1.0},
                            }
                        ]
                    },
                },
                "sampling_configs": {
                    "sampler": "ddim",
                    "sampling_steps": 5,
                    "eta": 1.0,
                    "cfg_scale": 2.0,
                },
            },
        }
    )
    output = model({"structure_array": _structure_array([3, 4])})
    loss = output["loss_dict"]["loss"]
    assert paddle.isfinite(loss).item()
    loss.backward()
    denoiser = model.core.ldm.denoiser
    vae = model.core.ldm.vae
    assert all(p.grad is not None for p in denoiser.parameters())
    assert all(p.stop_gradient and p.grad is None for p in vae.parameters())

    model.eval()
    with paddle.no_grad():
        result = model.sample(
            {"structure_array": {"num_atoms": paddle.to_tensor([3, 5])}},
            num_inference_steps=5,
            progress=False,
        )
    assert len(result["result"]) == 2
    for count, structure in zip((3, 5), result["result"]):
        assert structure["num_atoms"] == count
        assert np.asarray(structure["lattice"]).shape == (3, 3)


def test_unified_model_train_and_sample():
    # One Chemeleon2 object covers both training (forward loss) and sampling.
    paddle.seed(0)
    ldm = _load_cfg("ldm")["__init_params__"]
    model = build_model(
        {
            "__class_name__": "Chemeleon2",
            "__init_params__": {
                "stage": "diffusion",
                "denoiser_cfg": ldm["denoiser_cfg"],
                "noise_scheduler_cfg": ldm["noise_scheduler_cfg"],
                "vae_cfg": ldm["vae_cfg"],
            },
        }
    )
    loss = model({"structure_array": _structure_array([3, 4])})["loss_dict"]["loss"]
    assert paddle.isfinite(loss).item()
    loss.backward()
    assert any(p.grad is not None for p in model.parameters())
    model.eval()
    with paddle.no_grad():
        result = model.sample(
            {"structure_array": {"num_atoms": paddle.to_tensor([3, 5])}},
            num_inference_steps=5,
            progress=False,
        )
    assert len(result["result"]) == 2
    assert np.asarray(result["result"][0]["lattice"]).shape == (3, 3)

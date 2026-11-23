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
from ppmat.metrics import CrystalVUNMetric
from ppmat.metrics import build_metric
from ppmat.models import build_model
from ppmat.schedulers import build_scheduler
from ppmat.trainer.base_trainer import BaseTrainer
from ppmat.utils.execution import configure_execution_backend

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
        config.Model.__init_params__.vae_pretrained_model_name = None
        if name != "vae":
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


def _structure_loader(batch):
    return paddle.io.DataLoader(
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


def test_vae_training_pipeline(tmp_path):
    from cinn_workflow_harness import trainer_config

    cfg = _load_cfg("vae")
    params = cfg["__init_params__"]["vae_cfg"]["__init_params__"]
    params["augmentation"] = {"translate": True, "rotate": True}
    params["noise"] = {"ratio": 0.4, "corruption_scale": 0.1}
    vae = build_model(cfg)
    batch = {"structure_array": _structure_array([3, 5])}
    converted = vae.core._convert_train_batch(batch)
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
    loader = _structure_loader(batch)
    trainer = BaseTrainer(
        trainer_config(tmp_path, 1),
        vae,
        train_dataloader=loader,
        val_dataloader=loader,
        optimizer=optimizer,
    )
    trainer.train()
    assert trainer.state.global_step == 1
    ldm_config = _load_cfg("ldm")
    ldm_config["__init_params__"]["vae_ckpt_path"] = str(
        tmp_path / "checkpoints/latest.pdparams"
    )
    ldm = build_model(ldm_config)
    for key, value in vae.core.state_dict().items():
        np.testing.assert_array_equal(value.numpy(), ldm.vae.state_dict()[key].numpy())
    with pytest.raises(ValueError, match="only the eager"):
        configure_execution_backend(vae, "cinn", owner="test")


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
    metric = build_metric(
        {
            "__class_name__": "CrystalVUNMetric",
            "__init_params__": {"reference_structures": [structure]},
        }
    )
    expected = {"validity": 2 / 3, "uniqueness": 0.5, "novelty": 0.0}
    assert metric([valid, valid, invalid]) == expected
    metric.update_step(result={"result": [valid]}, batch=None, stage="sample")
    metric.update_step(result={"result": [valid, invalid]}, batch=None, stage="sample")
    assert metric.compute_epoch(stage="sample") == expected
    metric.update_step(result={"result": [valid]}, batch=None, stage="eval")
    assert metric.compute_epoch(stage="eval")["validity"] == 1
    metric.reset()
    assert metric.compute_epoch(stage="sample")["validity"] == 0
    assert "novelty" not in CrystalVUNMetric()([valid])


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "gpu",
            marks=pytest.mark.skipif(
                os.environ.get("PPMAT_RUN_CINN_WORKFLOW_TESTS") != "1",
                reason="Set PPMAT_RUN_CINN_WORKFLOW_TESTS=1 for GPU compilation.",
            ),
        ),
    ],
)
def test_ldm_trainer_resume_and_runtime(tmp_path, monkeypatch, device):
    from cinn_workflow_harness import patch_cinn_runtime
    from cinn_workflow_harness import trainer_config

    from ppmat.models import Chemeleon2LDMModule

    if device == "cpu":
        patch_cinn_runtime(monkeypatch, Chemeleon2LDMModule)
    else:
        paddle.set_device("gpu")
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
    monkeypatch.setattr(ldm.core, "validate_execution_backend", lambda **kw: None)
    assert configure_execution_backend(ldm, None, owner="test") == "cinn"
    assert (
        configure_execution_backend(
            ldm, "cinn", init_params={"full_graph": False}, owner="test"
        )
        == "cinn"
    )
    assert ldm.get_runtime_options("cinn")["full_graph"] is False
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
    config = _load_cfg(stage, small=False)
    config["__init_params__"]["vae_pretrained_model_name"] = None
    model = build_model(config)
    if stage == "vae":
        model = model.core
    assert model.state_dict().keys() == state.keys()
    for key, value in model.state_dict().items():
        assert value.shape == state[key].shape
    assert model.set_state_dict(state) == ([], [])
    del model
    legacy, _ = build_model_from_name(f"chemeleon2_{stage}")
    assert legacy.state_dict().keys() == state.keys()
    for key, value in legacy.state_dict().items():
        np.testing.assert_array_equal(value.numpy(), state[key].numpy())
    del legacy
    if stage == "vae":
        unified = build_model(_load_cfg(stage, small=False))
        for key, value in unified.core.state_dict().items():
            np.testing.assert_array_equal(value.numpy(), state[key].numpy())
        del unified
    del state
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
def test_ldm_cinn_gpu_compile_parity(tmp_path, monkeypatch):
    from cinn_workflow_harness import trainer_config

    from ppmat.sampler import StructureSampler

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
        runtimes = dict(ldm.core._runtime_cache)
        with paddle.no_grad():
            ldm._runtime_denoise(x, t, mask=mask)
        assert all(
            ldm.core._runtime_cache[key] is value for key, value in runtimes.items()
        )
        ldm.set_runtime_options({"cinn": {"full_graph": False, "dynamic_shape": False}})
        assert not ldm.core._runtime_cache

        batch = {"structure_array": _structure_array([3, 5])}
        trained = {}
        gradients = {}
        for backend in ("eager", "cinn"):
            paddle.seed(42)
            np.random.seed(42)
            model = build_model(_load_cfg("ldm"))
            optimizer = paddle.optimizer.SGD(
                learning_rate=1e-3, parameters=model.parameters()
            )
            original_step = optimizer.step

            def checked_step():
                for param in model.denoiser.parameters():
                    assert param.grad is not None
                    assert paddle.isfinite(param.grad).all().item()
                gradients[backend] = {
                    key: param.grad.numpy().copy()
                    for key, param in model.denoiser.named_parameters()
                }
                original_step()

            monkeypatch.setattr(optimizer, "step", checked_step)
            loader = _structure_loader(batch)
            trainer = BaseTrainer(
                trainer_config(tmp_path / backend / "training", 1),
                model,
                train_dataloader=loader,
                val_dataloader=loader,
                optimizer=optimizer,
                execution_config={
                    "backend": backend,
                    "__init_params__": {"full_graph": False},
                },
            )
            paddle.seed(123)
            np.random.seed(123)
            trainer.train()
            assert trainer.state.global_step == 1
            trained[backend] = {
                key: value.numpy().copy() for key, value in model.state_dict().items()
            }
            assert all(np.isfinite(value).all() for value in trained[backend].values())
        for key, value in gradients["eager"].items():
            np.testing.assert_allclose(
                value, gradients["cinn"][key], atol=2e-5, rtol=2e-5
            )
        assert trained["eager"].keys() == trained["cinn"].keys()
        for key, value in trained["eager"].items():
            np.testing.assert_allclose(
                value, trained["cinn"][key], atol=2e-5, rtol=2e-5
            )

        samples = {}
        for backend in ("eager", "cinn"):
            sampler = StructureSampler(
                model_name="chemeleon2_ldm",
                config_overrides=[
                    f"Execution.backend={backend}",
                    "Execution.__init_params__.full_graph=false",
                    "Sample.model_sample_params.progress=false",
                ],
            )
            assert sampler.execution_backend == backend
            paddle.seed(2024)
            np.random.seed(2024)
            output = tmp_path / backend / "samples"
            samples[backend] = sampler.sample_by_num_atoms(20, save_path=str(output))
            assert len(list(output.glob("*.cif"))) == 1
            assert (
                sampler.sample_config["model_sample_params"]["num_inference_steps"]
                == 50
            )
        before = samples["eager"]["result"][0]
        after = samples["cinn"]["result"][0]
        np.testing.assert_array_equal(before["atom_types"], after["atom_types"])
        for key in ("frac_coords", "lattice"):
            np.testing.assert_allclose(before[key], after[key], atol=2e-5, rtol=2e-5)
    finally:
        paddle.set_device(previous_device)


def test_rl_training_and_sampling(tmp_path):
    import gzip
    import pickle

    from ase.calculators.emt import EMT
    from cinn_workflow_harness import trainer_config
    from pymatgen.analysis.phase_diagram import PDEntry
    from pymatgen.analysis.phase_diagram import PhaseDiagram

    # Macro check: the full GRPO forward/backward and LDM-delegated sampling.
    paddle.seed(0)
    ldm_params = _load_cfg("ldm")["__init_params__"]
    references = [
        BuildStructure.build_one(
            {
                "atom_types": [28],
                "frac_coords": [[0, 0, 0]],
                "lattice": (np.eye(3) * 4).tolist(),
            },
            "array",
            niggli=False,
            canocial=False,
        )
    ]
    calculator = EMT()
    phase_path = tmp_path / "phase_diagram_20261122110000.pkl.gz"
    with gzip.open(phase_path, "wb") as handle:
        pickle.dump(PhaseDiagram([PDEntry("Ni", 0)]), handle)
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
                        "phase_diagram_path": str(phase_path),
                        "components": [
                            {
                                "__class_name__": "CreativityReward",
                                "__init_params__": {"weight": 1.0},
                            },
                            {
                                "__class_name__": "EnergyReward",
                                "__init_params__": {"weight": 0.1},
                            },
                        ],
                    },
                },
                "sampling_configs": {
                    "sampler": "ddim",
                    "sampling_steps": 5,
                    "eta": 1.0,
                    "cfg_scale": 2.0,
                },
            },
        },
        reference_structures=references,
        energy_calculator=calculator,
    )
    assert model.core.reward_fn.reference_structures is references
    provider = model.core.reward_fn.energy_provider
    assert provider.calculator is calculator
    assert np.isfinite(provider(references)).all()
    batch = {"structure_array": _structure_array([3, 4])}
    optimizer = paddle.optimizer.Adam(learning_rate=1e-4, parameters=model.parameters())
    initial = {key: value.clone() for key, value in model.state_dict().items()}
    output = model(batch)
    loss = output["loss_dict"]["loss"]
    assert paddle.isfinite(loss).item()
    loss.backward()
    denoiser = model.core.ldm.denoiser
    vae = model.core.ldm.vae
    assert all(p.grad is not None for p in denoiser.parameters())
    assert all(p.stop_gradient and p.grad is None for p in vae.parameters())
    optimizer.step()
    optimizer.clear_grad()
    assert all(paddle.isfinite(p).all().item() for p in model.parameters())
    assert any(
        not paddle.equal_all(value, initial[key])
        for key, value in model.state_dict().items()
    )
    loader = _structure_loader(batch)
    trainer = BaseTrainer(
        trainer_config(tmp_path, 1),
        model,
        train_dataloader=loader,
        val_dataloader=loader,
        optimizer=optimizer,
    )
    trainer.train()
    assert trainer.state.global_step == 1

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
    with pytest.raises(TypeError, match="misspelled_option"):
        build_model(
            {
                "__class_name__": "Chemeleon2",
                "__init_params__": {"misspelled_option": 1},
            }
        )
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

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
import copy
import os
import os.path as osp
from pathlib import Path
from unittest import mock

import numpy as np
import paddle
import pytest
from omegaconf import OmegaConf
from pymatgen.core import DummySpecies
from scipy.optimize import linear_sum_assignment

from ppmat.datasets import build_dataloader
from ppmat.datasets.build_structure import BuildStructure
from ppmat.datasets.collate_fn import DefaultCollator
from ppmat.datasets.custom_data_type import ConcatData
from ppmat.models import MODEL_REGISTRY
from ppmat.models import build_model
from ppmat.models import build_model_from_name
from ppmat.models import resolve_model_package_dir
from ppmat.models.xrd2mof.feature_extractor import Xrd2MofFeatureExtractor
from ppmat.models.xrd2mof.xrd2mof import Xrd2Mof
from ppmat.sampler import StructureSampler
from ppmat.trainer.base_trainer import BaseTrainer
from ppmat.utils import download
from ppmat.utils.crystal import lattice_params_to_matrix

CONFIG_PATH = (
    Path(__file__).resolve().parents[2]
    / "structure_generation"
    / "configs"
    / "xrd2mof"
    / "xrd2mof_cgmof50.yaml"
)

MODEL_NAME = "xrd2mof_cgmof50"
FEATURE_DIM = 16
TINY_NUM_TIMESTEPS = 50
TINY_DECODER_CFG = {
    "hidden_dim": 32,
    "feature_dim": FEATURE_DIM,
    "latent_dim": 16,
    "num_layers": 2,
    "num_atom_types": 2,
    "num_freqs": 4,
}
TINY_LATTICE_SCHEDULER_CFG = {
    "__class_name__": "DDPMScheduler",
    "__init_params__": {
        "beta_schedule": "squaredcos_cap_v2",
        "num_train_timesteps": TINY_NUM_TIMESTEPS,
        "clip_sample": False,
    },
}
TINY_COORD_SCHEDULER_CFG = {
    "__class_name__": "ScoreSdeVeSchedulerWrapped",
    "__init_params__": {
        "num_train_timesteps": TINY_NUM_TIMESTEPS,
        "sigma_min": 0.005,
        "sigma_max": 0.5,
        "snr": 1e-5,
    },
}


def load_config():
    config = OmegaConf.load(CONFIG_PATH)
    return OmegaConf.to_container(config, resolve=True)


def build_training_batch():
    config = load_config()
    loader = build_dataloader(config["Dataset"]["train"])
    return next(iter(loader))


def test_model_training_loss_and_step():
    config = load_config()
    model = build_model(config["Model"])
    batch = build_training_batch()

    outputs = model(batch)
    loss_dict = outputs["loss_dict"]
    loss = loss_dict["loss"]
    for key in ("loss", "loss_lattice", "loss_coord"):
        assert key in loss_dict
        assert bool(paddle.isfinite(loss_dict[key]))
    assert float(loss) > 0.0

    optimizer = paddle.optimizer.Adam(
        learning_rate=config["Optimizer"]["__init_params__"]["lr"]["__init_params__"][
            "learning_rate"
        ],
        parameters=model.parameters(),
    )
    before = [param.numpy().copy() for param in model.parameters()]
    loss.backward()
    for param in model.parameters():
        if not param.stop_gradient:
            assert param.grad is not None
            assert bool(paddle.isfinite(param.grad).all())
    optimizer.step()
    optimizer.clear_grad()
    after = [param.numpy() for param in model.parameters()]
    assert all(np.isfinite(param).all() for param in after)
    assert any(not np.allclose(prev, curr) for prev, curr in zip(before, after))


def test_val_dataset_end_to_end_forward():
    config = load_config()
    loader = build_dataloader(config["Dataset"]["val"])
    batch = next(iter(loader))
    structure_array = batch["structure_array"]
    assert set(structure_array) == {
        "num_atoms",
        "atom_types",
        "frac_coords",
        "lattice",
        "lengths",
        "angles",
    }
    assert batch["feature_embedding"].shape[0] == len(structure_array["num_atoms"])
    assert (
        int(structure_array["num_atoms"].sum())
        == structure_array["frac_coords"].shape[0]
    )

    model = build_model(config["Model"])
    outputs = model(batch)
    assert bool(paddle.isfinite(outputs["loss_dict"]["loss"]))


def test_sample_produces_valid_cg_structures():
    config = load_config()
    model = build_model(config["Model"])
    model.eval()
    loader = build_dataloader(config["Sample"]["data"])
    batch = next(iter(loader))

    with mock.patch.object(
        model.decoder, "gen_edges", wraps=model.decoder.gen_edges
    ) as gen_edges:
        outputs = model.sample(batch, num_inference_steps=5)
    assert gen_edges.call_count == 1

    assert len(outputs["result"]) == int(batch["structure_array"]["num_atoms"].shape[0])
    expected_species = np.asarray(["Xmetal", "Xlinker"])[
        batch["structure_array"]["atom_types"].numpy()
    ].tolist()
    assert [
        specie for item in outputs["result"] for specie in item["atom_types"]
    ] == expected_species
    for item in outputs["result"]:
        lattice = np.asarray(item["lattice"])
        frac_coords = np.asarray(item["frac_coords"])
        assert np.isfinite(lattice).all()
        assert np.isfinite(frac_coords).all()
        assert abs(np.linalg.det(lattice)) > 0.0
        assert frac_coords.min() >= 0.0 and frac_coords.max() < 1.0
        assert len(item["atom_types"]) == item["num_atoms"]
        assert set(item["atom_types"]).issubset({"Xmetal", "Xlinker"})
        structure = BuildStructure.build_one(
            item, **config["Sample"]["build_structure_cfg"]
        )
        assert len(structure) == item["num_atoms"]
        assert all(isinstance(site.specie, DummySpecies) for site in structure)
        assert [site.specie.symbol for site in structure] == item["atom_types"]
    assert [item["mof_id"] for item in outputs["result"]] == batch["mof_id"]


def build_feature_extractor_batch():
    batch_size, num_nodes, num_edges = 3, 9, 12
    return {
        "xrd": paddle.randn([batch_size, 5250]),
        "metal": paddle.randn([batch_size, 65]),
        "linker": {
            "x": paddle.randn([num_nodes, 37]),
            "edge_index": paddle.randint(0, num_nodes, [2, num_edges]),
            "batch": paddle.repeat_interleave(
                paddle.arange(batch_size), repeats=num_nodes // batch_size
            ),
        },
    }


def test_registered_model_full_chain():
    model, config = build_model_from_name(MODEL_NAME)
    extracted_path = download.get_weights_path_from_url(MODEL_REGISTRY[MODEL_NAME])

    assert type(model).__name__ == "Xrd2Mof"
    assert "Model" in config and "Sample" in config

    package_dir = resolve_model_package_dir(MODEL_NAME, extracted_path)
    assert (
        Path(package_dir, f"{MODEL_NAME}.yaml").read_bytes() == CONFIG_PATH.read_bytes()
    )
    checkpoint = paddle.load(osp.join(package_dir, "checkpoints", "best.pdparams"))
    assert checkpoint.keys() == model.state_dict().keys()
    for key, value in model.state_dict().items():
        assert checkpoint[key].shape == value.shape
    extractor = Xrd2MofFeatureExtractor()
    feature_checkpoint = paddle.load(
        osp.join(package_dir, "checkpoints", "feature_extractor.pdparams")
    )
    assert feature_checkpoint.keys() == extractor.state_dict().keys()
    for key, value in extractor.state_dict().items():
        assert feature_checkpoint[key].shape == value.shape
    extractor.set_state_dict(feature_checkpoint)
    extractor.eval()
    with paddle.no_grad():
        extractor_outputs = extractor(build_feature_extractor_batch())
    assert extractor_outputs["feature_embedding"].shape == [3, 256]
    assert bool(paddle.isfinite(extractor_outputs["feature_embedding"]).all())

    model.eval()
    sample_cfg = copy.deepcopy(config["Sample"]["data"])
    sample_cfg["sampler"]["__init_params__"]["batch_size"] = 3
    loader = build_dataloader(sample_cfg)
    batch = next(iter(loader))
    batch["feature_embedding"] = extractor_outputs["feature_embedding"]
    outputs = model.sample(batch, num_inference_steps=2)
    assert len(outputs["result"]) == 3
    with pytest.raises(ValueError, match="p_unconditional"):
        model.sample(batch, num_inference_steps=2, guidance_scale=2.0)


def test_sample_through_structure_sampler(tmp_path):
    sampler = StructureSampler(model_name=MODEL_NAME)

    sampler.sample_config["model_sample_params"]["num_inference_steps"] = 2
    results = sampler.sample_by_dataloader(save_path=tmp_path)

    assert len(results) == 16
    files = sorted(tmp_path.glob("*.cif"))
    assert len(files) == 16
    assert not list(tmp_path.glob("*.npz"))
    with pytest.warns(UserWarning, match="not a valid Element"):
        for index, item in enumerate(results, start=1):
            reference = BuildStructure.build_one(
                item, **sampler.sample_config["build_structure_cfg"]
            )
            formula = reference.formula.replace(" ", "-")
            restored = BuildStructure.build_one(
                str(tmp_path / f"{formula}_1_{index}.cif"),
                "cif_file",
                niggli=False,
                canocial=False,
            )
            assert all(isinstance(site.specie, DummySpecies) for site in restored)
            restored_species = [site.specie.symbol for site in restored]
            assert sorted(restored_species) == sorted(item["atom_types"])
            np.testing.assert_allclose(
                restored.lattice.parameters,
                reference.lattice.parameters,
                atol=1e-7,
                rtol=1e-7,
            )
            for symbol in set(item["atom_types"]):
                expected_coords = reference.frac_coords[
                    np.asarray(item["atom_types"]) == symbol
                ]
                restored_coords = restored.frac_coords[
                    np.asarray(restored_species) == symbol
                ]
                delta = expected_coords[:, None, :] - restored_coords[None, :, :]
                delta -= np.round(delta)
                rows, cols = linear_sum_assignment(np.sum(delta**2, axis=-1))
                np.testing.assert_allclose(delta[rows, cols], 0, atol=1e-4)

    with pytest.raises(ValueError, match="feature_embedding"):
        sampler.sample_by_chemical_formula("LiMnO2")


def _make_tiny_model(execution_backend="eager", p_unconditional=0.0):
    with paddle.utils.unique_name.guard():
        return Xrd2Mof(
            decoder_cfg=dict(TINY_DECODER_CFG),
            lattice_noise_scheduler_cfg=copy.deepcopy(TINY_LATTICE_SCHEDULER_CFG),
            coord_noise_scheduler_cfg=copy.deepcopy(TINY_COORD_SCHEDULER_CFG),
            num_train_timesteps=TINY_NUM_TIMESTEPS,
            lattice_loss_weight=0.1,
            coord_loss_weight=1.0,
            execution_backend=execution_backend,
            p_unconditional=p_unconditional,
        )


def test_classifier_free_training_and_sampling():
    model = _make_tiny_model(p_unconditional=0.5)
    loader = paddle.io.DataLoader(
        [_make_cg_sample(3 + i % 2, seed=i) for i in range(8)],
        batch_size=8,
        collate_fn=DefaultCollator(),
        return_list=True,
    )
    batch = next(iter(loader))
    original_embedding = batch["feature_embedding"].numpy().copy()
    paddle.seed(42)
    with mock.patch.object(
        model.decoder, "forward_with_edges", wraps=model.decoder.forward_with_edges
    ) as decoder:
        loss = model(batch)["loss_dict"]["loss"]
    embedding = decoder.call_args.args[0].numpy()
    dropped = np.all(embedding == 0, axis=1)
    assert dropped.any() and (~dropped).any()
    np.testing.assert_array_equal(embedding[~dropped], original_embedding[~dropped])
    optimizer = paddle.optimizer.Adam(learning_rate=1e-3, parameters=model.parameters())
    loss.backward()
    for parameter in model.parameters():
        assert parameter.grad is not None
        assert bool(paddle.isfinite(parameter.grad).all())
    optimizer.step()
    optimizer.clear_grad()
    assert all(
        bool(paddle.isfinite(parameter).all()) for parameter in model.parameters()
    )
    np.testing.assert_array_equal(
        batch["feature_embedding"].numpy(), original_embedding
    )

    model.eval()
    batch = _make_structure_batch()
    with mock.patch.object(
        model.decoder, "forward_with_edges", wraps=model.decoder.forward_with_edges
    ) as decoder:
        with paddle.no_grad():
            model(batch)
    np.testing.assert_array_equal(
        decoder.call_args.args[0].numpy(), batch["feature_embedding"].numpy()
    )
    paddle.seed(123)
    unconditional = model.sample(batch, num_inference_steps=2, guidance_scale=0.0)
    zero_batch = {
        **batch,
        "feature_embedding": paddle.zeros_like(batch["feature_embedding"]),
    }
    paddle.seed(123)
    expected = model.sample(zero_batch, num_inference_steps=2)
    assert unconditional == expected

    predictions = []
    decode = model._runtime_decode

    def record_predictions(*args):
        prediction = decode(*args)
        predictions.append(prediction)
        return prediction

    with mock.patch.object(
        model, "_runtime_decode", side_effect=record_predictions
    ), mock.patch.object(
        model.lattice_scheduler, "step", wraps=model.lattice_scheduler.step
    ) as lattice_step, mock.patch.object(
        model.coord_scheduler, "step_pred", wraps=model.coord_scheduler.step_pred
    ) as coord_step:
        guided = model.sample(batch, num_inference_steps=2, guidance_scale=2.0)
    assert len(guided["result"]) == 2
    assert len(predictions) == 8
    for step in range(2):
        conditional_l, conditional_x = predictions[4 * step + 2]
        unconditional_l, unconditional_x = predictions[4 * step + 3]
        np.testing.assert_allclose(
            lattice_step.call_args_list[step].args[0].numpy(),
            (unconditional_l + 2.0 * (conditional_l - unconditional_l)).numpy(),
        )
        np.testing.assert_allclose(
            coord_step.call_args_list[step].args[0].numpy(),
            (unconditional_x + 2.0 * (conditional_x - unconditional_x)).numpy(),
        )
    for item in guided["result"]:
        assert np.isfinite(item["frac_coords"]).all()
        assert np.isfinite(item["lattice"]).all()


def _clone_state_dict(model):
    return {key: value.clone() for key, value in model.state_dict().items()}


def _record_cinn_boundaries(names):
    def run_boundary(self, name, function, *args, **kwargs):
        names.append(name)
        return function(*args, **kwargs)

    return run_boundary


def _make_cg_sample(num_atoms, seed):
    rng = np.random.RandomState(seed)
    feature_embedding = rng.rand(FEATURE_DIM).astype("float32")
    atom_types = rng.randint(0, 2, size=num_atoms).astype("int64")
    frac_coords = rng.rand(num_atoms, 3).astype("float32")
    lengths = rng.rand(3).astype("float32") + 3.0
    angles = np.full(3, 90.0, dtype="float32")
    return {
        "feature_embedding": feature_embedding,
        "structure_array": {
            "num_atoms": ConcatData(np.array([num_atoms], dtype="int64")),
            "atom_types": ConcatData(atom_types),
            "frac_coords": ConcatData(frac_coords),
            "lattice": lattice_params_to_matrix(*lengths, *angles).astype("float32"),
            "lengths": lengths,
            "angles": angles,
        },
    }


def _make_structure_batch():
    return next(iter(_make_structure_loader()))


def _make_structure_loader():
    return paddle.io.DataLoader(
        [_make_cg_sample(3, seed=1), _make_cg_sample(4, seed=2)],
        batch_size=2,
        shuffle=False,
        collate_fn=DefaultCollator(),
        return_list=True,
    )


def _make_dispatch_pair():
    initial_state = _clone_state_dict(_make_tiny_model())
    paddle.seed(2026)
    eager_model = _make_tiny_model(p_unconditional=0.5)
    paddle.seed(2026)
    cinn_model = _make_tiny_model(execution_backend="cinn", p_unconditional=0.5)
    eager_model.set_state_dict(initial_state)
    cinn_model.set_state_dict(initial_state)
    eager_model.eval()
    cinn_model.eval()
    return eager_model, cinn_model


def test_cinn_dispatch_matches_eager():
    eager_model, cinn_model = _make_dispatch_pair()
    batch = _make_structure_batch()

    paddle.seed(42)
    np.random.seed(42)
    with paddle.no_grad():
        eager_loss = eager_model(batch)["loss_dict"]["loss"]

    names = []
    paddle.seed(42)
    np.random.seed(42)
    with mock.patch.object(Xrd2Mof, "_run_runtime", _record_cinn_boundaries(names)):
        with paddle.no_grad():
            cinn_loss = cinn_model(batch)["loss_dict"]["loss"]

    assert names == ["denoise_step"]
    np.testing.assert_allclose(cinn_loss.numpy(), eager_loss.numpy(), atol=0, rtol=0)
    assert cinn_model.state_dict().keys() == eager_model.state_dict().keys()

    paddle.seed(123)
    eager_result = eager_model.sample(batch, num_inference_steps=3, guidance_scale=2.0)[
        "result"
    ]

    names = []
    paddle.seed(123)
    with mock.patch.object(Xrd2Mof, "_run_runtime", _record_cinn_boundaries(names)):
        cinn_result = cinn_model.sample(
            batch, num_inference_steps=3, guidance_scale=2.0
        )["result"]

    assert names
    assert set(names) == {"denoise_step"}
    assert len(cinn_result) == len(eager_result)
    for eager_entry, cinn_entry in zip(eager_result, cinn_result):
        assert eager_entry["num_atoms"] == cinn_entry["num_atoms"]
        assert eager_entry["atom_types"] == cinn_entry["atom_types"]
        for key in ("frac_coords", "lattice"):
            np.testing.assert_allclose(
                cinn_entry[key], eager_entry[key], atol=0, rtol=0
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


@pytest.fixture
def mocked_cinn_runtime():
    boundaries = []
    with mock.patch.object(
        Xrd2Mof, "validate_execution_backend", lambda self, **kwargs: None
    ), mock.patch.object(Xrd2Mof, "_run_runtime", _record_cinn_boundaries(boundaries)):
        yield boundaries


def test_cinn_trainer_resume_workflow(mocked_cinn_runtime, tmp_path):
    initial_state = _clone_state_dict(_make_tiny_model())
    paddle.seed(2026)
    first_model = _make_tiny_model(execution_backend="cinn")
    first_model.set_state_dict(initial_state)
    first_trainer, first_optimizer = _build_trainer(first_model, tmp_path / "first", 1)
    with paddle.utils.unique_name.guard():
        first_trainer.train()

    checkpoint_dir = tmp_path / "first" / "checkpoints"
    for prefix in ("epoch_1", "latest", "best"):
        for suffix in ("pdparams", "pdopt", "pdstates"):
            assert (checkpoint_dir / f"{prefix}.{suffix}").is_file()
    saved_state = paddle.load(str(checkpoint_dir / "latest.pdparams"))
    saved_optimizer = paddle.load(str(checkpoint_dir / "latest.pdopt"))
    assert saved_state.keys() == first_model.state_dict().keys()

    paddle.seed(2026)
    resumed_model = _make_tiny_model(execution_backend="cinn")
    resumed_trainer, resumed_optimizer = _build_trainer(
        resumed_model, tmp_path / "resumed", 2
    )
    restored_steps = []

    def check_restored_state(model, inputs):
        if restored_steps:
            return
        assert resumed_trainer.state.global_step == first_trainer.state.global_step
        assert resumed_optimizer.get_lr() == first_optimizer.get_lr()
        for key, value in model.state_dict().items():
            np.testing.assert_array_equal(value.numpy(), saved_state[key].numpy())
        optimizer_state = resumed_optimizer.state_dict()
        assert optimizer_state.keys() == saved_optimizer.keys()
        for key, value in optimizer_state.items():
            np.testing.assert_array_equal(value.numpy(), saved_optimizer[key].numpy())
        restored_steps.append(resumed_trainer.state.global_step)

    resume_hook = resumed_model.register_forward_pre_hook(check_restored_state)
    with paddle.utils.unique_name.guard():
        resumed_trainer.train(resume_from_checkpoint=str(checkpoint_dir / "latest"))
    resume_hook.remove()
    assert restored_steps == [1]

    paddle.seed(2026)
    straight_model = _make_tiny_model(execution_backend="cinn")
    straight_model.set_state_dict(initial_state)
    straight_trainer, straight_optimizer = _build_trainer(
        straight_model, tmp_path / "straight", 2
    )
    with paddle.utils.unique_name.guard():
        straight_trainer.train()

    assert first_trainer.state.global_step == 1
    assert resumed_trainer.state.global_step == 2
    assert resumed_trainer.state.epoch == 2
    assert resumed_optimizer.get_lr() == straight_optimizer.get_lr()
    assert (
        resumed_optimizer.state_dict().keys() == straight_optimizer.state_dict().keys()
    )
    assert mocked_cinn_runtime
    assert set(mocked_cinn_runtime) == {"denoise_step"}
    for model in (resumed_model, straight_model):
        assert model.state_dict().keys() == first_model.state_dict().keys()
        for value in model.state_dict().values():
            assert bool(paddle.isfinite(value).all())


@pytest.mark.skipif(
    os.environ.get("PPMAT_RUN_CINN_WORKFLOW_TESTS") != "1",
    reason="Set PPMAT_RUN_CINN_WORKFLOW_TESTS=1 to run the GPU CINN compile test",
)
def test_gpu_cinn_matches_eager(tmp_path):
    if not paddle.is_compiled_with_cuda() or not paddle.base.is_compiled_with_cinn():
        pytest.skip("Paddle must be compiled with CUDA and CINN")
    paddle.set_device("gpu:0")

    initial_state = _clone_state_dict(_make_tiny_model())
    paddle.seed(2026)
    eager_model = _make_tiny_model(p_unconditional=0.5)
    eager_model.set_state_dict(initial_state)
    paddle.seed(2026)
    cinn_model = _make_tiny_model(execution_backend="cinn", p_unconditional=0.5)
    cinn_model.set_state_dict(initial_state)

    paddle.seed(2026)
    np.random.seed(2026)
    eager_trainer, _ = _build_trainer(
        eager_model, tmp_path / "eager", 1, backend="eager"
    )
    with paddle.utils.unique_name.guard():
        eager_trainer.train()

    paddle.seed(2026)
    np.random.seed(2026)
    cinn_trainer, _ = _build_trainer(cinn_model, tmp_path / "cinn", 1)
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

    cinn_model.set_state_dict(eager_model.state_dict())
    cinn_model.eval()
    eager_model.eval()
    batch = _make_structure_batch()
    with paddle.no_grad():
        paddle.seed(42)
        np.random.seed(42)
        eager_loss = eager_model(batch)["loss_dict"]["loss"]
        paddle.seed(42)
        np.random.seed(42)
        cinn_loss = cinn_model(batch)["loss_dict"]["loss"]
    np.testing.assert_allclose(
        cinn_loss.numpy(), eager_loss.numpy(), atol=2e-5, rtol=2e-5
    )

    paddle.seed(123)
    expected = eager_model.sample(batch, num_inference_steps=2, guidance_scale=2.0)[
        "result"
    ]
    paddle.seed(123)
    result = cinn_model.sample(batch, num_inference_steps=2, guidance_scale=2.0)[
        "result"
    ]
    assert len(result) == 2
    for eager_item, compiled_item in zip(expected, result):
        for key in ("frac_coords", "lattice"):
            np.testing.assert_allclose(
                compiled_item[key], eager_item[key], atol=2e-5, rtol=2e-5
            )

    sampler = StructureSampler(
        model_name=MODEL_NAME,
        config_overrides=[
            "Sample.data.sampler.__init_params__.batch_size=2",
            "Sample.model_sample_params.num_inference_steps=2",
        ],
    )
    batch = next(iter(build_dataloader(sampler.sample_config["data"])))
    state_shapes = {
        key: list(value.shape) for key, value in sampler.model.state_dict().items()
    }
    paddle.seed(42)
    np.random.seed(42)
    eager_result = sampler.sample(batch)["result"]
    sampler.model.set_execution_backend("cinn")
    paddle.seed(42)
    np.random.seed(42)
    compiled_result = sampler.sample(batch)["result"]
    runtime_cache = dict(sampler.model._runtime_cache)
    paddle.seed(42)
    np.random.seed(42)
    repeated_result = sampler.sample(batch)["result"]
    assert runtime_cache
    for key, runtime in runtime_cache.items():
        assert sampler.model._runtime_cache[key] is runtime
    for expected, actual, repeated in zip(
        eager_result, compiled_result, repeated_result
    ):
        assert expected["num_atoms"] == actual["num_atoms"] == repeated["num_atoms"]
        assert expected["atom_types"] == actual["atom_types"] == repeated["atom_types"]
        assert expected["mof_id"] == actual["mof_id"] == repeated["mof_id"]
        for key in ("frac_coords", "lattice"):
            assert np.isfinite(actual[key]).all()
            np.testing.assert_allclose(actual[key], expected[key], atol=2e-5, rtol=2e-5)
            np.testing.assert_allclose(repeated[key], actual[key], atol=2e-5, rtol=2e-5)
    assert state_shapes == {
        key: list(value.shape) for key, value in sampler.model.state_dict().items()
    }
    sampler.model.set_execution_backend("eager")
    assert not sampler.model._runtime_cache

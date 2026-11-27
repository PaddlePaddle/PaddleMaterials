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

"""GeoDiff's CINN runtime keeps one denoising boundary and moves no eager numbers.

The tests pin what the adaptation can silently break: which numerical function
is the boundary, that the eager numbers do not move when the backend is switched
on, that checkpoints keep their keys, and that the Trainer and the sampler select
the backend for GeoDiff through the ``Execution`` block.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import paddle
from cinn_workflow_harness import assert_checkpoint_keys_unchanged
from cinn_workflow_harness import gpu_cinn_or_skip
from cinn_workflow_harness import patch_cinn_runtime
from cinn_workflow_harness import requires_gpu_cinn
from cinn_workflow_harness import trainer_config
from omegaconf import OmegaConf

from ppmat.datasets import build_dataloader
from ppmat.models import GeoDiff
from ppmat.sampler import MolecularSampler
from ppmat.trainer.base_trainer import BaseTrainer

REPO_ROOT = Path(__file__).resolve().parents[1]
PRESET = (
    REPO_ROOT
    / "structure_generation"
    / "configs"
    / "geodiff"
    / "geodiff_qm9_default.yaml"
)
MODEL_PARAMS = {"hidden_dim": 16, "num_convs": 2, "num_convs_local": 2}


def _make_model(execution_backend="eager"):
    with paddle.utils.unique_name.guard():
        paddle.seed(2026)
        return GeoDiff(
            **MODEL_PARAMS,
            execution_backend=execution_backend,
            runtime_options={"cinn": {"full_graph": False}},
        )


def _loader(directory, batch_size, packed):
    return build_dataloader(
        {
            "dataset": {
                "__class_name__": "GeomConformerDataset",
                "__init_params__": {
                    "path": str(directory),
                    "split": "test",
                    "pack_by_molecule": packed,
                },
            },
            "sampler": {
                "__class_name__": "BatchSampler",
                "__init_params__": {"batch_size": batch_size, "shuffle": False},
            },
            "loader": {"num_workers": 0, "collate_fn": "GeomConformerCollator"},
        }
    )


def _directed_bonds(bonds, offset):
    """Expand molecule-local ``(i, j, type)`` bonds into both directions."""

    sources, targets, types = [], [], []
    for source, target, kind in bonds:
        sources += [source + offset, target + offset]
        targets += [target + offset, source + offset]
        types += [kind, kind]
    return np.array([sources, targets], dtype=np.int64), np.array(types, dtype=np.int64)


def _write_molecules(directory):
    """Write ``test_data_1k.npz``: two references of ethylene, one of ethanol."""

    ethylene = np.array(
        [
            [0.93, 0.92, 1.23],
            [0.0, 0.0, 0.67],
            [-0.93, -0.92, 1.23],
            [0.0, 0.0, -0.67],
            [0.93, -0.92, -1.23],
            [-0.93, 0.92, -1.23],
        ]
    )
    ethanol = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.54, 0.0, 0.0],
            [0.83, 1.24, 0.0],
            [-0.36, 0.62, 0.78],
            [-0.36, -0.47, -0.78],
            [-0.36, -0.16, 0.0],
            [1.9, -0.45, 0.9],
            [1.9, -0.45, -0.9],
            [0.6, 2.1, 0.3],
        ]
    )
    ethylene_bonds = [(1, 0, 1), (1, 3, 2), (1, 2, 1), (3, 4, 1), (3, 5, 1)]
    ethanol_bonds = [
        (0, 1, 1),
        (1, 2, 1),
        (0, 3, 1),
        (0, 4, 1),
        (0, 5, 1),
        (1, 6, 1),
        (1, 7, 1),
        (2, 8, 1),
    ]
    # Edges are molecule-local in every record; only the rows are concatenated.
    edges = [
        _directed_bonds(ethylene_bonds, 0),
        _directed_bonds(ethylene_bonds, 0),
        _directed_bonds(ethanol_bonds, 0),
    ]
    Path(directory).mkdir(parents=True, exist_ok=True)
    path = Path(directory) / "test_data_1k.npz"
    np.savez_compressed(
        path,
        atom_type=np.array([1, 6, 1, 6, 1, 1] * 2 + [6, 6, 8, 1, 1, 1, 1, 1, 1]),
        pos=np.concatenate([ethylene, ethylene + 0.02, ethanol]).astype(np.float32),
        edge_index=np.concatenate([index for index, _ in edges], axis=1),
        edge_type=np.concatenate([types for _, types in edges]),
        node_ptr=np.array([0, 6, 12, 21]),
        edge_ptr=np.array([0, 10, 20, 36]),
        smiles=np.array(["C=C", "C=C", "CCO"]),
    )
    return Path(directory)


def _denoise(model, batch):
    with paddle.no_grad():
        return model.denoise_step(
            atom_type=batch["atom_types"],
            pos=batch["cart_coords"],
            bond_index=batch["edge_index"],
            bond_type=batch["edge_type"],
            batch=batch["batch"],
            time_step=None,
            return_edges=True,
        )


def _to_numpy(values):
    return [value.numpy() for value in values]


def _assert_bitwise_equal(actual, expected):
    for left, right in zip(actual, expected):
        assert left.shape == right.shape
        np.testing.assert_array_equal(left, right)


def _training_step(model, batch):
    model.train()
    model.clear_gradients()
    paddle.seed(11)
    output = model(batch)
    output["loss_dict"]["loss"].backward()
    gradients = {name: parameter.grad for name, parameter in model.named_parameters()}
    return output["loss_dict"], gradients


def test_cinn_runs_one_denoise_step_boundary(monkeypatch, tmp_path):
    model = _make_model(execution_backend="cinn")
    boundary_names = []

    def run_boundary(self, name, function, *args, **kwargs):
        boundary_names.append(name)
        return function(*args, **kwargs)

    monkeypatch.setattr(GeoDiff, "_run_runtime", run_boundary)
    model.train()
    model(next(iter(_loader(_write_molecules(tmp_path), 2, False))))

    assert boundary_names == ["denoise_step"]

    model.eval()
    batch = next(iter(_loader(tmp_path, 1, True)))
    boundary_names.clear()
    _denoise(model, batch)

    assert boundary_names == ["denoise_step"]

    boundary_names.clear()
    model.sample(batch, num_confs="2x", n_steps=2)

    assert boundary_names == ["denoise_step", "denoise_step"]


def test_enabling_the_backend_does_not_move_eager_numbers(monkeypatch, tmp_path):
    """The boundary is a pass-through when it runs the eager callable.

    The same weights and inputs, once through the plain eager path and once
    through the dispatcher, have to agree bit-for-bit.
    """

    monkeypatch.setattr(
        GeoDiff,
        "_run_runtime",
        lambda self, name, function, *args, **kwargs: function(*args, **kwargs),
    )
    directory = _write_molecules(tmp_path)
    eager = _make_model("eager")
    dispatched = _make_model("cinn")
    dispatched.set_state_dict(eager.state_dict())

    eager.eval()
    dispatched.eval()
    batches = list(_loader(directory, 1, True))
    for batch in batches:
        _assert_bitwise_equal(
            _to_numpy(_denoise(dispatched, batch)), _to_numpy(_denoise(eager, batch))
        )

    paddle.seed(5)
    eager_coordinates = eager.sample(batches[0], num_confs="2x", n_steps=2)["pred"]
    paddle.seed(5)
    dispatched_coordinates = dispatched.sample(batches[0], num_confs="2x", n_steps=2)[
        "pred"
    ]
    for left, right in zip(dispatched_coordinates, eager_coordinates):
        _assert_bitwise_equal([left["cart_coords"]], [right["cart_coords"]])

    batch = next(iter(_loader(directory, 2, False)))
    eager_losses, eager_gradients = _training_step(eager, batch)
    dispatched_losses, dispatched_gradients = _training_step(dispatched, batch)
    for key in ("loss", "loss_global", "loss_local"):
        assert float(dispatched_losses[key]) == float(eager_losses[key])
    for name, gradient in eager_gradients.items():
        other = dispatched_gradients[name]
        assert (gradient is None) == (other is None), name
        if gradient is not None:
            _assert_bitwise_equal([other.numpy()], [gradient.numpy()])


def test_state_dict_keys_are_unchanged_by_the_backend():
    eager_keys = _make_model("eager").state_dict().keys()
    model = _make_model("cinn")
    dispatched_keys = model.state_dict().keys()

    assert dispatched_keys == eager_keys
    assert all(not key.startswith("model.") for key in dispatched_keys)
    model.set_execution_backend("eager")
    model.set_execution_backend("cinn")
    assert model.state_dict().keys() == eager_keys


def test_trainer_selects_the_backend_through_the_execution_block(tmp_path, monkeypatch):
    patch_cinn_runtime(monkeypatch, GeoDiff)
    model = _make_model("eager")
    directory = _write_molecules(tmp_path / "data")
    trainer = BaseTrainer(
        trainer_config(tmp_path / "trainer", 1),
        model,
        train_dataloader=_loader(directory, 2, False),
        val_dataloader=_loader(directory, 2, False),
        optimizer=paddle.optimizer.Adam(
            learning_rate=1e-3, parameters=model.parameters()
        ),
        execution_config={"backend": "cinn"},
    )

    trainer.train()

    assert model.execution_backend == "cinn"
    assert trainer.execution_backend == "cinn"
    assert_checkpoint_keys_unchanged(
        paddle.load(str(tmp_path / "trainer" / "checkpoints" / "latest.pdparams")),
        model,
    )


def test_sampler_selects_the_backend_through_the_execution_block(tmp_path, monkeypatch):
    patch_cinn_runtime(monkeypatch, GeoDiff)
    directory = _write_molecules(tmp_path / "data")
    config = OmegaConf.load(PRESET)
    config.Model.__init_params__.update(MODEL_PARAMS)
    config.Global.data_dir = str(directory)
    config_path = tmp_path / "geodiff.yaml"
    OmegaConf.save(config, config_path)
    checkpoint_path = tmp_path / "geodiff.pdparams"
    paddle.save(_make_model("eager").state_dict(), str(checkpoint_path))
    output_dir = tmp_path / "conformers"

    sampler = MolecularSampler(
        config_path=str(config_path),
        checkpoint_path=str(checkpoint_path),
        config_overrides=[
            "Execution.backend=cinn",
            "Dataset.train=null",
            "Metric=null",
            f"Sampler.output_dir={output_dir}",
            "Sampler.model_sample_params.num_confs=2x",
            "Sampler.model_sample_params.n_steps=2",
        ],
    )

    assert sampler.execution_backend == "cinn"
    batch = next(iter(sampler._sample_loader))
    (record,) = sampler.sample(batch)["pred"]

    assert record["cart_coords"].shape == (4, 6, 3)
    assert np.isfinite(record["cart_coords"]).all()


def test_atom_embedding_rows_are_clipped_like_paddle_max_norm():
    """The explicit row clipping replaces ``max_norm`` and must match it exactly."""

    def weights():
        paddle.seed(1)
        weight = paddle.randn([100, MODEL_PARAMS["hidden_dim"]])
        with paddle.no_grad():
            weight[3] = 5.0
            weight[1] = 1.0
            weight[7] = -4.0
        return weight

    index = paddle.to_tensor([3, 1, 3], dtype="int64")
    reference = paddle.nn.Embedding(100, MODEL_PARAMS["hidden_dim"], max_norm=10.0)
    reference.weight.set_value(weights())
    expected = reference(index).numpy()
    expected_weight = reference.weight.numpy()

    encoder = _make_model().encoder_global
    encoder.embedding.weight.set_value(weights())
    encoder.clip_embedding_norms(index)
    actual = encoder.embedding(index).numpy()

    np.testing.assert_array_equal(encoder.embedding.weight.numpy(), expected_weight)
    np.testing.assert_array_equal(actual, expected)
    looked_up, untouched = (np.linalg.norm(expected_weight[i]) for i in (3, 7))
    assert looked_up <= 10.0 < untouched


@requires_gpu_cinn
def test_gpu_cinn_matches_eager_for_sampling_and_training(tmp_path, monkeypatch):
    gpu_cinn_or_skip()
    monkeypatch.setenv("SOT_ENABLE_COMPILE_TIME_LIMIT", "0")
    directory = _write_molecules(tmp_path)
    eager = _make_model("eager")
    cinn = _make_model("cinn")
    cinn.set_state_dict(eager.state_dict())
    eager.eval()
    cinn.eval()
    ethylene, ethanol = list(_loader(directory, 1, True))

    for batch in (ethylene, ethanol, ethylene):
        eager_result = _to_numpy(_denoise(eager, batch))
        cinn_result = _to_numpy(_denoise(cinn, batch))
        _assert_bitwise_equal(cinn_result[2:5], eager_result[2:5])
        for left, right in zip(cinn_result[:2], eager_result[:2]):
            np.testing.assert_allclose(left, right, atol=1e-4, rtol=1e-4)

    paddle.seed(5)
    eager_sample = eager.sample(ethylene, num_confs="2x", n_steps=5)["pred"]
    paddle.seed(5)
    cinn_sample = cinn.sample(ethylene, num_confs="2x", n_steps=5)["pred"]
    np.testing.assert_allclose(
        cinn_sample[0]["cart_coords"],
        eager_sample[0]["cart_coords"],
        atol=1e-3,
        rtol=1e-3,
    )

    batch = next(iter(_loader(directory, 2, False)))
    eager_losses, eager_gradients = _training_step(eager, batch)
    cinn_losses, cinn_gradients = _training_step(cinn, batch)
    for key in ("loss", "loss_global", "loss_local"):
        np.testing.assert_allclose(
            float(cinn_losses[key]), float(eager_losses[key]), rtol=1e-4
        )
    for name, gradient in eager_gradients.items():
        other = cinn_gradients[name]
        assert (gradient is None) == (other is None), name
        if gradient is not None:
            scale = float(gradient.abs().max())
            np.testing.assert_allclose(
                other.numpy(), gradient.numpy(), rtol=1e-3, atol=1e-3 * scale
            )

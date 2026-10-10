# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

# Adapted from https://github.com/hspark1212/chemeleon

"""Chemeleon routes every decoder pass through one denoise_step boundary.

These tests pin the parts that restructuring can silently break: that the
boundary is unique and named denoise_step, that the eager numbers do not
move when the backend is switched on, and that the checkpoint keys are
identical across backends.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import numpy as np
import paddle

from ppmat.models.chemeleon.chemeleon import Chemeleon
from ppmat.models.chemeleon.diff_utils import prob_mask_like

TEXT_EMBED_DIM = 16
LOSS_KEYS = {"loss", "vb", "ce", "lattice", "coords"}


class _StubTextEncoder(paddle.nn.Layer):
    """Minimal text encoder honoring the construction-time contract.

    The model accepts any module exposing get_text_embeds(batch_text,
    cond_drop_prob); this stub encodes each text into a deterministic
    vector and applies the same null-substitution + text_emb tail as the
    bundled TextEncoder.
    """

    def __init__(self, text_embed_dim=TEXT_EMBED_DIM, text_dim=8):
        super().__init__()
        self.text_embed_dim = text_embed_dim
        self.text_emb = paddle.nn.Sequential(
            paddle.nn.Linear(text_embed_dim, text_embed_dim),
            paddle.nn.LayerNorm(text_embed_dim),
            paddle.nn.GELU(),
            paddle.nn.Linear(text_embed_dim, text_dim),
        )
        self.null_text_embeds = self.create_parameter(
            shape=[1, text_embed_dim],
            attr=paddle.ParamAttr(),
            dtype=paddle.get_default_dtype(),
            default_initializer=paddle.nn.initializer.Constant(0.0),
        )

    def get_text_embeds(self, batch_text, cond_drop_prob):
        batch_size = len(batch_text)
        vectors = []
        for text in batch_text:
            filled = (text * 8)[: self.text_embed_dim]
            vectors.append(
                [((i + 1) * (1 + ord(c) % 17)) / 103.0 for i, c in enumerate(filled)]
            )
        embeddings = paddle.to_tensor(vectors, dtype=self.null_text_embeds.dtype)
        keep_mask = prob_mask_like((batch_size,), 1.0 - cond_drop_prob)
        embeddings = paddle.where(
            keep_mask[:, None],
            embeddings,
            self.null_text_embeds.expand([batch_size, self.text_embed_dim]),
        )
        return self.text_emb(embeddings)


def _make_config():
    return {
        "hidden_dim": 8,
        "time_dim": 8,
        "text_dim": 8,
        "max_atoms": 16,
        "num_layers": 2,
        "act_fn": "silu",
        "dis_emb": "sin",
        "num_freqs": 4,
        "edge_style": "fc",
        "max_neighbors": 20,
        "cutoff": 6.0,
        "ln": True,
        "ip": True,
        "smooth": False,
        "pred_atom_types": True,
        "text_guide": True,
        "cond_drop_prob": 0.2,
        "beta_schedule": "cosine",
        "timesteps": 16,
        "d3pm_hybrid_coeff": 1.0,
        "cost_atom_types": 1.0,
        "cost_lattice": 1.0,
        "cost_coords": 1.0,
    }


def _make_model(execution_backend="eager"):
    with paddle.utils.unique_name.guard():
        paddle.seed(2026)
        return Chemeleon(
            _make_config(),
            text_encoder=_StubTextEncoder(),
            execution_backend=execution_backend,
            runtime_options={"cinn": {"full_graph": True}},
        )


def _make_batch():
    return {
        "structure_array": {
            "frac_coords": np.array(
                [
                    [0.10, 0.20, 0.30],
                    [0.40, 0.50, 0.60],
                    [0.12, 0.24, 0.36],
                    [0.48, 0.52, 0.60],
                    [0.71, 0.82, 0.93],
                ],
                dtype=np.float32,
            ),
            "atom_types": np.array([3, 8, 1, 6, 4], dtype=np.int64),
            "lattices": np.array(
                [
                    [[4.0, 0.0, 0.0], [0.0, 4.5, 0.0], [0.0, 0.0, 5.0]],
                    [[3.8, 0.1, 0.0], [0.1, 3.9, 0.0], [0.0, 0.1, 4.2]],
                ],
                dtype=np.float32,
            ),
            "natoms": np.array([2, 3], dtype=np.int64),
        },
        "text": ["Li2O", "CaTiO3"],
    }


def _deterministic_draws():
    """Fix every random draw site of forward (timestep, gumbel, lattice and
    coords noise, kept/dropped text vectors)."""

    num_nodes = 5
    batch_size = 2
    max_atoms = _make_config()["max_atoms"]
    return {
        "batched_t": np.array([5, 9], dtype=np.int64),
        "gumbel_u": np.random.RandomState(0)
        .random_sample((num_nodes, max_atoms))
        .astype(np.float32),
        "eps_lattice": np.random.RandomState(1)
        .standard_normal((batch_size, 3, 3))
        .astype(np.float32),
        "eps_coords": np.random.RandomState(2)
        .standard_normal((num_nodes, 3))
        .astype(np.float32),
        "text_vec_768": np.random.RandomState(3)
        .standard_normal((batch_size, TEXT_EMBED_DIM))
        .astype(np.float32),
        "text_keep_mask": np.array([True, False]),
    }


def test_denoise_step_is_the_single_runtime_boundary():
    source = Path(inspect.getsourcefile(Chemeleon))
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    names = [
        node.args[0].value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "runtime_boundary"
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and isinstance(node.args[0].value, str)
    ]
    assert names == ["denoise_step"]


def test_forward_dispatches_one_denoise_step(monkeypatch):
    model = _make_model(execution_backend="cinn")
    model.eval()
    boundary_names = []

    def run_boundary(self, name, function, *args, **kwargs):
        boundary_names.append(name)
        return function(*args, **kwargs)

    monkeypatch.setattr(Chemeleon, "_run_runtime", run_boundary)
    output = model.forward(_make_batch())

    assert boundary_names == ["denoise_step"]
    assert set(output) == {"loss_dict", "monitor_dict"}
    assert set(output["loss_dict"]) == LOSS_KEYS
    for value in output["loss_dict"].values():
        assert np.isfinite(float(value))


def test_enabling_the_backend_does_not_move_eager_numbers(monkeypatch):
    """The boundary must be a pass-through when it runs the eager callable.

    The same weights and the same deterministic draws, once through the
    plain eager path and once through the dispatcher, have to agree
    bit-for-bit on every loss and monitor tensor.
    """

    batch = _make_batch()
    draws = _deterministic_draws()

    eager_model = _make_model(execution_backend="eager")
    state = eager_model.state_dict()
    eager_model.eval()
    eager_output = eager_model.forward(batch, **draws)

    dispatched_model = _make_model(execution_backend="cinn")
    dispatched_model.set_state_dict(state)
    dispatched_model.eval()
    monkeypatch.setattr(
        Chemeleon,
        "_run_runtime",
        lambda self, name, function, *args, **kwargs: function(*args, **kwargs),
    )
    dispatched_output = dispatched_model.forward(batch, **draws)

    assert dispatched_output["loss_dict"].keys() == eager_output["loss_dict"].keys()
    assert dispatched_output["monitor_dict"].keys() == (
        eager_output["monitor_dict"].keys()
    )
    for section in ("loss_dict", "monitor_dict"):
        for key, reference in eager_output[section].items():
            np.testing.assert_allclose(
                np.asarray(dispatched_output[section][key]),
                np.asarray(reference),
                atol=0,
                rtol=0,
            )


def test_state_dict_keys_are_unchanged_by_the_backend():
    assert (
        _make_model(execution_backend="cinn").state_dict().keys()
        == _make_model(execution_backend="eager").state_dict().keys()
    )

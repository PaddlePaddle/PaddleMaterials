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

from pathlib import Path

import numpy as np
import paddle
import pytest

from ppmat.metrics.streaming_base import StreamingMetricBase
from ppmat.models.sgequidiff.diffusion_model import EquivariantDiffusionModel
from ppmat.models.sgequidiff.sgequidiff import SGEQuiDiff
from ppmat.models.sgequidiff.vocabs import build_embedding_tools
from ppmat.models.sgequidiff.wyckoff_geometry import build_wyckoff_geometry
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


def _make_sgequidiff():
    """Build a lightweight SGEQuiDiff for fast integration tests."""
    return SGEQuiDiff(
        dataset_name="mp_20",
        num_timesteps=2,
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
    def sampler(self):
        paddle.seed(0)
        return _make_sgequidiff()

    def test_sample_returns_valid_result_format(self, sampler):
        """Unconditional sampling yields crystals with self-consistent fields.

        Each generated structure must report a positive atom count whose
        atom_types / frac_coords match, plus 3 lattice lengths and angles.
        """
        batch_data = {
            "structure_array": {
                "num_atoms": paddle.to_tensor([2], dtype=paddle.int64),
            }
        }
        out = sampler.sample(batch_data)
        assert "result" in out
        for crystal in out["result"]:
            assert crystal["num_atoms"] > 0
            assert len(crystal["atom_types"]) == crystal["num_atoms"]
            assert len(crystal["frac_coords"]) == crystal["num_atoms"]
            assert len(crystal["lengths"]) == 3
            assert len(crystal["angles"]) == 3
            for pt in crystal["frac_coords"]:
                assert len(pt) == 3

    def test_forward_returns_loss_dict(self, sampler):
        """Full SGEQuiDiff forward pass returns a scalar, finite loss."""
        batch_data = _make_synthetic_batch()
        out = sampler(batch_data)
        assert "loss_dict" in out
        loss = out["loss_dict"]["loss"]
        assert loss.ndim == 0
        assert not paddle.isnan(loss).item()

    def test_sample_keeps_hydrogen_element(self, sampler, monkeypatch):
        """Regression: element_indices=0 (H) must survive the sample() filter.

        sample() maps 0-indexed element_indices through chemical_symbols,
        which is 1-indexed (chemical_symbols[0] == "X" placeholder). An
        off-by-one here drops H from every generated crystal.
        """
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
            if p.grad is not None and float(p.grad.abs().sum()) > 0:
                assert name.split(".")[0] in (
                    "atom_coord_diffusion_model",
                    "space_group_sampler",
                    "lattice_sampler",
                    "wyckoff_and_element_sampler",
                )

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

    def test_sampled_structure_validity(self, sampler):
        """Sampled crystals keep elements, coords and lattice params in range.

        Guards the sampling path against generating out-of-domain values
        (elements outside the encoding table, coords outside the unit cell,
        or lattice params outside the dataset bounds).
        """
        from ppmat.models.sgequidiff.sgequidiff_meta import ELEMENT_ENCODING_SIZE

        out = sampler.sample(
            {"structure_array": {"num_atoms": paddle.to_tensor([2, 3], dtype="int64")}}
        )
        assert len(out["result"]) > 0
        for crystal in out["result"]:
            n = crystal["num_atoms"]
            assert n > 0
            assert len(crystal["atom_types"]) == n
            assert len(crystal["frac_coords"]) == n
            for elem in crystal["atom_types"]:
                assert 1 <= elem <= ELEMENT_ENCODING_SIZE, f"elem out of range: {elem}"
            for coords in crystal["frac_coords"]:
                assert len(coords) == 3
                assert all(
                    0.0 <= c < 1.0 for c in coords
                ), f"coords out of cell: {coords}"
            assert len(crystal["lengths"]) == 3
            assert len(crystal["angles"]) == 3
            assert all(2.0 <= length <= 133.0 for length in crystal["lengths"])
            assert all(60.0 <= a <= 135.0 for a in crystal["angles"])


def _write_synthetic_mp20_npz(root_dir, split="train", num_crystals=8, name="mp_20"):
    """Write a small mp_20-style npz (packed + indices) for the dataset chain test.

    Field layout follows AsymmetricUnitDataset.read_data with NE=98:
    [n, sg, comp(98), lengths(3), angles(3), elems(n), wyckoffs(n),
     frac_coords(3n), wyckoff_shape(n)].
    """
    from ppmat.models.sgequidiff.sgequidiff_meta import ELEMENT_ENCODING_SIZE as NE

    rng = np.random.default_rng(0)
    packed_arrays = []
    indices = []
    offset = 0
    for i in range(num_crystals):
        n = int(rng.integers(1, 5))
        comp = np.zeros(NE, dtype=np.float32)
        comp[0] = 1.0
        lengths = np.array([5.0, 5.0, 5.0], dtype=np.float32)
        angles = np.array([90.0, 90.0, 90.0], dtype=np.float32)
        elems = np.full(n, 1 + int(rng.integers(0, 2)), dtype=np.float32)
        wycks = np.zeros(n, dtype=np.float32)  # P1 has a single wyckoff site
        fracs = rng.random([n, 3]).astype(np.float32)
        wsi = np.zeros(n, dtype=np.float32)
        flat = np.concatenate(
            [
                np.array([n], dtype=np.float32),
                np.array([1.0], dtype=np.float32),  # space group 1 (P1)
                comp,
                lengths,
                angles,
                elems,
                wycks,
                fracs.reshape(-1),
                wsi,
            ]
        ).astype(np.float32)
        packed_arrays.append(flat)
        offset += flat.shape[0]
        # indices holds the split points for the first num_crystals - 1
        # crystals only (matching the real mp_20 npz layout, where
        # np.split(packed, indices) yields exactly num_crystals segments).
        if i < num_crystals - 1:
            indices.append(offset)
    root = Path(root_dir) / name
    root.mkdir(parents=True, exist_ok=True)
    np.savez(
        root / f"{split}.npz",
        packed=np.concatenate(packed_arrays),
        indices=np.array(indices, dtype=np.int64),
    )
    return root


def test_dataset_collate_to_model_forward(tmp_path):
    """Data pipeline: npz -> build_dataloader -> collate -> model forward.

    Overall question: does a realistic dataset batch flow from disk through
    MP20ASUDataset / DefaultCollator into SGEQuiDiff.forward without
    key/shape mismatches?
    """
    from ppmat.datasets import build_dataloader

    _write_synthetic_mp20_npz(tmp_path, num_crystals=8)
    loader = build_dataloader(
        {
            "dataset": {
                "__class_name__": "MP20ASUDataset",
                "__init_params__": {
                    "path": str(tmp_path / "mp_20" / "train.npz"),
                },
            },
            "loader": {"num_workers": 0, "use_shared_memory": False},
            "sampler": {
                "__class_name__": "BatchSampler",
                "__init_params__": {
                    "batch_size": 4,
                    "shuffle": False,
                    "drop_last": False,
                },
            },
        }
    )
    batch = next(iter(loader))
    assert "n_atoms_per_asu" in batch

    sampler = _make_sgequidiff()
    sampler.train()
    out = sampler(batch)
    loss = out["loss_dict"]["loss"]
    assert loss.ndim == 0
    assert not paddle.isnan(loss).item()
    assert float(loss) > 0.0

    # Contract-8: label_dict must expose label keys identical to batch_data.
    label_keys = (
        "space_group_indices",
        "lattice_lengths",
        "lattice_angles",
        "n_atoms_per_asu",
        "element_indices",
        "wyckoff_indices",
        "wyckoff_shape_indices",
        "frac_coords",
    )
    assert "label_dict" in out
    for key in label_keys:
        assert key in out["label_dict"], f"missing label key: {key}"
        assert bool(
            paddle.equal(out["label_dict"][key], batch[key]).all().item()
        ), f"label_dict[{key}] mismatch with batch label"


def test_dataset_downloads_via_unified_pipeline(tmp_path, monkeypatch):
    """Download path: a missing explicit path falls back to the unified
    download pipeline (url + md5 + cache under ~/.paddlemat/datasets).

    Overall question: when ``path`` does not exist, does each concrete
    dataset subclass (``MP20ASUDataset`` / ``MPTS52ASUDataset``) resolve
    the npz as ``<extract_root>/<name>/<split>.npz`` using its own
    class-level ``name`` / ``url`` / ``md5``, while the abstract base
    class holds no download source?
    """
    import ppmat.utils.download as download
    from ppmat.datasets.asu_dataset import AsymmetricUnitDataset
    from ppmat.datasets.asu_dataset import MP20ASUDataset
    from ppmat.datasets.asu_dataset import MPTS52ASUDataset

    extract_root = tmp_path / "cache" / "mp_20_asu"
    _write_synthetic_mp20_npz(extract_root, num_crystals=4)

    calls = {}

    def fake_get_datasets_path_from_url(url, md5sum=None):
        calls["url"] = url
        calls["md5"] = md5sum
        return str(extract_root)

    monkeypatch.setattr(
        download, "get_datasets_path_from_url", fake_get_datasets_path_from_url
    )

    dataset = MP20ASUDataset(path=str(tmp_path / "custom" / "train.npz"))
    assert calls == {"url": MP20ASUDataset.url, "md5": MP20ASUDataset.md5}
    assert dataset.path == str(extract_root / "mp_20" / "train.npz")
    assert len(dataset) == 4

    # The base class holds no download source; subclasses declare their own.
    assert AsymmetricUnitDataset.url is None
    assert MP20ASUDataset.name == "mp_20"
    assert MPTS52ASUDataset.name == "mpts_52"
    assert MPTS52ASUDataset.url != MP20ASUDataset.url

    # Each subclass owns its own default path: building MPTS52 without an
    # explicit path must not fall back to the MP-20 default.
    _write_synthetic_mp20_npz(extract_root, num_crystals=4, name="mpts_52")
    calls.clear()
    mpts_dataset = MPTS52ASUDataset()
    assert calls == {"url": MPTS52ASUDataset.url, "md5": MPTS52ASUDataset.md5}
    assert mpts_dataset.path == str(extract_root / "mpts_52" / "train.npz")

    # Instantiating the base class without an existing path must fail fast.
    with pytest.raises(ValueError):
        AsymmetricUnitDataset(path=str(tmp_path / "missing" / "train.npz"))
    with pytest.raises(ValueError):
        AsymmetricUnitDataset(path=None)


def test_sgequidiff_metric(tmp_path):
    """Evaluation pipeline: generation-quality metric over synthetic structures.

    Overall question: does SGEQuiDiffMetric return the complete expected
    metric set (validity / uniqueness / novelty / coverage / distances)
    with values in valid ranges? Uses self-contained synthetic CIFs written
    to ``tmp_path``, so the test runs without external data files.
    """
    import pandas as pd
    from pymatgen.core import Lattice
    from pymatgen.core import Structure

    from ppmat.metrics.sgequidiff_metric import SGEQuiDiffMetric
    from ppmat.metrics.utils import get_crys_from_cif

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

    pred_dicts = [get_crys_from_cif(cif).dict for cif in cifs]
    # The first structure is repeated across "batches": it must be counted
    # once for uniqueness, so the streaming accumulation (which spreads the
    # duplicate over two steps) matches the batch interface.
    duplicated = pred_dicts + [pred_dicts[0]]
    metric = SGEQuiDiffMetric(gt_file_path=str(gt_csv))
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
    assert 0.0 <= result["validity"] <= 1.0
    assert 0.0 <= result["uniqueness"] <= 1.0
    assert 0.0 <= result["novelty"] <= 1.0

    # Streaming contract: per-step accumulation must match the batch interface.
    assert isinstance(metric, StreamingMetricBase)

    stream_metric = SGEQuiDiffMetric(gt_file_path=str(gt_csv))
    assert stream_metric.compute_epoch(stage="sample") == {}
    assert stream_metric.compute_epoch(stage="eval") == {}
    stream_metric.update_step(
        result={"samples": {"result": pred_dicts[:4]}}, batch=None, stage="sample"
    )
    stream_metric.update_step(
        result={"result": pred_dicts[4:] + [pred_dicts[0]]},
        batch=None,
        stage="sample",
    )
    stream_metric.update_step(result={"result": duplicated}, batch=None, stage="eval")
    streamed = stream_metric.compute_epoch(stage="sample")
    for key, value in result.items():
        assert streamed[key] == pytest.approx(value)
    stream_metric.reset()
    assert stream_metric.compute_epoch(stage="sample") == {}


def test_execution_backend_protocol():
    """Runtime protocol: backend selection is owned by the diffusion model.

    Overall question: does the SGEQuiDiff <-> EquivariantDiffusionModel
    delegation honor the framework execution-backend contract (backend
    propagation, eager fallback, invalid input rejection, AMP/world-size
    constraints) without changing any state_dict key?
    """
    model = _make_sgequidiff()
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


def _make_small_gnn_sgequidiff():
    """Tiny GNN SGEQuiDiff so CINN compilation stays fast in tests."""
    return SGEQuiDiff(
        dataset_name="mp_20",
        num_timesteps=2,
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
        paddle.is_compiled_with_cuda()
        and paddle.device.cuda.device_count() > 0
        and paddle.is_compiled_with_cinn()
    ),
    reason="requires a CUDA GPU and a CINN-enabled Paddle build",
)
def test_cinn_sampling_parity():
    """CINN workflow: compiled denoise_step matches eager sampling.

    Overall question: does sampling with ``execution_backend="cinn"`` reuse a
    single cached compiled runtime and reproduce the eager trajectory within
    GPU float32 tolerance?
    """
    paddle.set_device("gpu")
    try:
        model = _make_small_gnn_sgequidiff()
        batch = {"structure_array": {"num_atoms": paddle.to_tensor([1])}}

        def sample_frac_coords():
            paddle.seed(42)
            result = model.sample(dict(batch))["result"]
            return [crystal["frac_coords"] for crystal in result]

        model.set_execution_backend("eager")
        coords_eager = sample_frac_coords()

        model.set_execution_backend("cinn")
        model.set_runtime_options({"cinn": {"full_graph": False}})
        coords_cinn = sample_frac_coords()

        cache = model.atom_coord_diffusion_model._runtime_cache
        assert list(cache.keys()) == [("cinn", "train", "denoise_step")]

        max_diff = max(
            abs(a - b)
            for eager_cell, cinn_cell in zip(coords_eager, coords_cinn)
            for eager_row, cinn_row in zip(eager_cell, cinn_cell)
            for a, b in zip(eager_row, cinn_row)
        )
        assert max_diff < 2e-5, f"eager/CINN sampling mismatch: {max_diff}"
    finally:
        paddle.set_device("cpu")


def test_named_lr_groups_effective_learning_rates():
    """Optimizer contract: named_lr_groups scale the effective learning rate.

    Overall question: do ``lr_multiplier`` groups produce the configured
    effective learning rates under Paddle's parameter-group semantics (a
    float group ``learning_rate`` multiplies the optimizer's global rate,
    for both float and LRScheduler optimizers)? This pins the semantics that
    the SGEQuiDiff training recipe (sub-sampled lr for the autoregressive
    samplers) relies on.
    """
    from ppmat.optimizer.optimizer import AdamW

    class _TwoLayerModel(paddle.nn.Layer):
        def __init__(self):
            super().__init__()
            self.wyckoff_and_element_sampler = paddle.nn.Linear(2, 2, bias_attr=False)
            self.atom_coord_backbone = paddle.nn.Linear(2, 2, bias_attr=False)

    def _run(global_lr_factory, lr_multiplier):
        model = _TwoLayerModel()
        opt = AdamW(
            learning_rate=global_lr_factory(),
            weight_decay=0.0,
            named_lr_groups=[
                {
                    "name": "wyckoff_and_element_sampler",
                    "lr_multiplier": lr_multiplier,
                }
            ],
        )(model)
        before = {n: p.numpy().copy() for n, p in model.named_parameters()}
        # Every weight has grad = 1; with weight_decay=0 and first-step AdamW
        # bias correction, the parameter delta equals the effective lr.
        loss = paddle.zeros([1])
        for p in model.parameters():
            loss = loss + p.sum()
        loss.backward()
        opt.step()
        deltas = {
            n: float(np.abs(after - before[n]).max())
            for n, after in ((n, p.numpy()) for n, p in model.named_parameters())
        }
        return deltas

    # Case A: float global lr.
    deltas = _run(lambda: 2.0, 0.1)
    assert deltas["atom_coord_backbone.weight"] == pytest.approx(2.0, abs=1e-4)
    assert deltas["wyckoff_and_element_sampler.weight"] == pytest.approx(0.2, abs=1e-4)

    # Case B: LRScheduler global lr (ReduceOnPlateau-style recipes).
    scheduler_lr = 2.0
    deltas = _run(
        lambda: paddle.optimizer.lr.StepDecay(scheduler_lr, step_size=1000), 0.1
    )
    assert deltas["atom_coord_backbone.weight"] == pytest.approx(2.0, abs=1e-4)
    assert deltas["wyckoff_and_element_sampler.weight"] == pytest.approx(0.2, abs=1e-4)

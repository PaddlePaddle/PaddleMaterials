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

"""LiFlow dataset, sampling and collation tests.

The propagator dataset (``LiFlowDataset``, reference ``TimeDelayedPairDataset``)
and the corrector dataset (``NoisedFrameDataset``) are independent classes.
Both return plain PaddleMaterials ``Data`` objects that batch through the
unmodified ``DefaultCollator``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import paddle
import pytest
from ase import units

from ppmat.datasets.collate_fn import DefaultCollator
from ppmat.datasets.geometric_data_type.data import Data
from ppmat.datasets.liflow_dataset import LiFlowDataset
from ppmat.datasets.liflow_noised_frame_dataset import NoisedFrameDataset
from ppmat.models.liflow.prior import AdaptiveMaxwellBoltzmannPrior
from ppmat.models.liflow.prior import MaxwellBoltzmannPrior

DATASET_CLASSES = [LiFlowDataset, NoisedFrameDataset]


@pytest.fixture()
def mini_dataset_path() -> Path:
    return Path(__file__).resolve().parent / "fixtures" / "liflow" / "dataset_mini"


def _as_numpy(value):
    return value.numpy() if isinstance(value, paddle.Tensor) else np.asarray(value)


@pytest.mark.parametrize("dataset_cls", DATASET_CLASSES)
def test_dataset_returns_data_samples(mini_dataset_path, dataset_cls):
    sample = dataset_cls(path=mini_dataset_path, split="train", time_delay_steps=100)[0]
    assert isinstance(sample, Data)

    num_atoms = int(sample["start_positions"].shape[0])
    assert sample["start_positions"].shape == [num_atoms, 3]
    assert sample["end_positions"].shape == [num_atoms, 3]
    assert sample["prior"].shape == [num_atoms, 3]
    assert sample["velocity"].shape == [num_atoms, 3]
    assert sample["elements"].shape == [num_atoms]
    assert sample["atomic_numbers"].shape == [num_atoms]
    assert sample["elements"].dtype == paddle.int64
    assert sample["atomic_numbers"].dtype == paddle.int64
    assert sample["edge_index"].dtype == paddle.int64
    assert sample["edge_index"].shape[0] == 2
    assert sample["shifts"].shape == [sample["edge_index"].shape[1], 3]

    # Graph-level scalars are 0-dim so ``Batch.from_data_list`` promotes them.
    assert sample["temperature"].ndim == 0
    assert sample["flow_time"].ndim == 0
    assert float(sample["flow_time"]) >= 0.0
    assert float(sample["flow_time"]) <= 1.0

    expected_velocity = sample["end_positions"] - (
        sample["start_positions"] + sample["prior"]
    )
    np.testing.assert_allclose(
        _as_numpy(sample["velocity"]), _as_numpy(expected_velocity), atol=1e-6
    )


def test_propagator_endpoints_are_time_delayed_frames(mini_dataset_path):
    dataset = LiFlowDataset(path=mini_dataset_path, split="train", seed=3)
    for idx in range(3):
        sample = dataset[idx]
        row = dataset._df.iloc[idx]
        trajectory = np.asarray(
            dataset._get_positions(float(row["temp"]))[str(row["name"])]
        )
        start = _as_numpy(sample["start_positions"])
        end = _as_numpy(sample["end_positions"])
        start_idx = int(np.argmin(np.abs(trajectory - start).sum(axis=(1, 2))))
        end_idx = int(np.argmin(np.abs(trajectory - end).sum(axis=(1, 2))))
        np.testing.assert_allclose(trajectory[start_idx], start, atol=1e-5)
        np.testing.assert_allclose(trajectory[end_idx], end, atol=1e-5)
        assert end_idx - start_idx == dataset.time_delay_steps
        assert int(row["t_start"]) <= start_idx
        assert end_idx < int(row["t_end"])


def test_corrector_uses_clean_frame_and_noise_only(mini_dataset_path):
    dataset = NoisedFrameDataset(path=mini_dataset_path, split="train", seed=3)
    sample = dataset[0]
    row = dataset._df.iloc[0]
    trajectory = np.asarray(
        dataset._get_positions(float(row["temp"]))[str(row["name"])]
    )
    clean = _as_numpy(sample["end_positions"])
    noisy = _as_numpy(sample["start_positions"])
    prior = _as_numpy(sample["prior"])
    velocity = _as_numpy(sample["velocity"])

    # ``end`` is a real, untouched trajectory frame...
    dist = np.abs(trajectory - clean).sum(axis=(1, 2))
    frame_idx = int(np.argmin(dist))
    np.testing.assert_allclose(trajectory[frame_idx], clean, atol=1e-5)
    assert (
        int(row["t_start"]) + dataset.time_delay_steps <= frame_idx < int(row["t_end"])
    )

    # ...the noisy input is that frame plus noise...
    noise = noisy - clean
    assert np.abs(noise).max() > 0.0

    # ...and the flow-matching target is exactly ``-(noise + prior)``.
    np.testing.assert_allclose(velocity, -(noise + prior), atol=1e-5)


def test_corrector_noise_magnitude_follows_uniform_scale_prior(mini_dataset_path):
    dataset = NoisedFrameDataset(path=mini_dataset_path, split="train", seed=5)
    noises = []
    for idx in range(len(dataset)):
        sample = dataset[idx]
        noise = _as_numpy(sample["start_positions"]) - _as_numpy(
            sample["end_positions"]
        )
        noises.append(noise.reshape(-1))
    noise = np.concatenate(noises)
    # ``UniformScaleNormalPrior(0.25)`` draws per-atom sigma ~ U(0, 0.25), so the
    # pooled standard deviation is well below the upper bound and non-zero.
    assert 0.02 < float(noise.std()) < 0.25
    assert abs(float(noise.mean())) < 0.05


def test_train_sampling_is_dynamic(mini_dataset_path):
    for dataset_cls in DATASET_CLASSES:
        dataset = dataset_cls(path=mini_dataset_path, split="train")
        flow_times = {float(dataset[0]["flow_time"]) for _ in range(20)}
        starts = {float(dataset[0]["start_positions"].sum()) for _ in range(20)}
        # Time, window, prior and noise are resampled on every access; a frozen
        # cache would collapse both sets to a single value.
        assert len(flow_times) > 1
        assert len(starts) > 1


def test_train_sampling_follows_global_seed(mini_dataset_path):
    def draw():
        return _as_numpy(
            LiFlowDataset(path=mini_dataset_path, split="train")[0]["start_positions"]
        )

    np.random.seed(0)
    first = draw()
    np.random.seed(0)
    np.testing.assert_array_equal(first, draw())
    np.random.seed(1)
    assert not np.allclose(first, draw())


@pytest.mark.parametrize("dataset_cls", DATASET_CLASSES)
def test_validation_samples_are_deterministic(mini_dataset_path, dataset_cls):
    dataset = dataset_cls(path=mini_dataset_path, split="val", seed=42)
    assert len(dataset) == 2 * 11
    first = dataset[0]
    second = dataset_cls(path=mini_dataset_path, split="val", seed=42)[0]
    for key in (
        "start_positions",
        "end_positions",
        "prior",
        "flow_time",
        "edge_index",
        "shifts",
    ):
        np.testing.assert_array_equal(_as_numpy(first[key]), _as_numpy(second[key]))

    # All flow times of one row share the same (start, end, prior) pair.
    same_row = dataset[1]
    np.testing.assert_array_equal(
        _as_numpy(first["start_positions"]), _as_numpy(same_row["start_positions"])
    )
    np.testing.assert_array_equal(
        _as_numpy(first["prior"]), _as_numpy(same_row["prior"])
    )
    assert float(first["flow_time"]) == 0.0
    assert float(dataset[10]["flow_time"]) == pytest.approx(1.0)


def test_validation_seed_changes_the_draw(mini_dataset_path):
    first = LiFlowDataset(path=mini_dataset_path, split="val", seed=42)[0]
    second = LiFlowDataset(path=mini_dataset_path, split="val", seed=43)[0]
    assert not np.allclose(
        _as_numpy(first["start_positions"]), _as_numpy(second["start_positions"])
    )
    assert not np.allclose(_as_numpy(first["prior"]), _as_numpy(second["prior"]))


def test_splits_are_disjoint(mini_dataset_path):
    splits = {
        name: set(LiFlowDataset(path=mini_dataset_path, split=name).sample_ids)
        for name in ("train", "val", "test")
    }
    assert splits["train"].isdisjoint(splits["val"])
    assert splits["train"].isdisjoint(splits["test"])
    assert splits["val"].isdisjoint(splits["test"])


@pytest.mark.parametrize("dataset_cls", DATASET_CLASSES)
def test_default_collator_offsets_variable_length_graphs(
    mini_dataset_path, dataset_cls
):
    dataset = dataset_cls(path=mini_dataset_path, split="train", seed=0)
    batch = DefaultCollator()([dataset[0], dataset[2]])
    num_nodes = int(batch["start_positions"].shape[0])
    assert batch["start_positions"].shape == [num_nodes, 3]
    assert batch["elements"].dtype == paddle.int64
    assert int(batch["batch"].max()) == 1
    assert int(batch["batch"].min()) == 0
    assert int(batch["batch"].shape[0]) == num_nodes
    # Graph-level scalars become one value per graph.
    assert batch["temperature"].shape == [2]
    assert batch["flow_time"].shape == [2]
    # ``__inc__`` shifts the second graph's edges by the first graph's size.
    first_nodes = int(dataset[0]["start_positions"].shape[0])
    second_edge_index = _as_numpy(dataset[2]["edge_index"])
    assert _as_numpy(batch["edge_index"]).min() >= 0
    if second_edge_index.size:
        assert _as_numpy(batch["edge_index"]).max() >= first_nodes + (
            second_edge_index.min()
        )


def test_zero_edge_graph_batches(mini_dataset_path):
    dataset = LiFlowDataset(path=mini_dataset_path, split="train", seed=0)
    samples = [dataset[0], dataset[1]]
    empty_edges = [
        Data(
            num_nodes=int(s["start_positions"].shape[0]),
            start_positions=s["start_positions"],
            end_positions=s["end_positions"],
            prior=s["prior"],
            velocity=s["velocity"],
            elements=s["elements"],
            atomic_numbers=s["atomic_numbers"],
            edge_index=paddle.zeros([2, 0], dtype="int64"),
            shifts=paddle.zeros([0, 3], dtype="float32"),
            temperature=s["temperature"],
            flow_time=s["flow_time"],
        )
        for s in samples
    ]
    batch = DefaultCollator()(empty_edges)
    assert list(batch["edge_index"].shape) == [2, 0]
    assert list(batch["shifts"].shape) == [0, 3]
    assert batch["start_positions"].shape[0] == sum(
        int(s["start_positions"].shape[0]) for s in samples
    )


@pytest.mark.parametrize("dataset_cls", DATASET_CLASSES)
def test_short_window_is_rejected(tmp_path, dataset_cls):
    import pandas as pd

    source = Path(__file__).resolve().parent / "fixtures" / "liflow" / "dataset_mini"
    for name in ("atomic_numbers.npy", "element_index.npy", "lattice.npy"):
        (tmp_path / name).write_bytes((source / name).read_bytes())
    for name in ("positions_800K.npz",):
        (tmp_path / name).write_bytes((source / name).read_bytes())
    pd.DataFrame(
        {
            "name": ["train_a"],
            "temp": [800],
            "t_start": [0],
            "t_end": [100],
            "prior_Li": [1],
            "prior_frame": [0],
            "comp": ["mini"],
        }
    ).to_csv(tmp_path / "train.csv", index=False)
    dataset = dataset_cls(path=tmp_path, split="train", time_delay_steps=100)
    with pytest.raises(ValueError, match="shorter"):
        dataset[0]


def test_missing_directory_without_url_raises(tmp_path):
    for dataset_cls in DATASET_CLASSES:
        with pytest.raises(FileNotFoundError, match="does not exist"):
            dataset_cls(path=tmp_path / "missing", split="train")


def test_url_requires_md5(tmp_path):
    for dataset_cls in DATASET_CLASSES:
        with pytest.raises(ValueError, match="md5"):
            dataset_cls(
                path=tmp_path / "missing",
                split="train",
                url="https://example.invalid/liflow.zip",
            )


def test_no_default_download_url():
    # The LiFlow training archive is not hosted by PaddleMaterials: neither
    # dataset may advertise a default URL, otherwise a missing ``path`` would
    # silently try to fetch a non-existent archive.
    import inspect

    for dataset_cls in DATASET_CLASSES:
        signature = inspect.signature(dataset_cls.__init__)
        assert signature.parameters["url"].default is None
        assert signature.parameters["md5"].default is None


def test_adaptive_maxwell_boltzmann_prior_is_seeded():
    first = AdaptiveMaxwellBoltzmannPrior(scale=[[1.0, 10.0], [0.316, 3.16]], seed=42)
    second = AdaptiveMaxwellBoltzmannPrior(scale=[[1.0, 10.0], [0.316, 3.16]], seed=42)
    kwargs = {
        "temperature": 800.0,
        "atomic_numbers": np.array([3, 8], dtype=np.int64),
        "masses": np.array([6.94, 15.999], dtype=np.float64),
        "scale_Li_index": 1,
        "scale_frame_index": 0,
        "shape": (2, 3),
    }
    np.testing.assert_array_equal(first.sample(**kwargs), second.sample(**kwargs))


def test_maxwell_boltzmann_prior_statistics():
    temperature = 800.0
    masses = np.array([6.94, 15.999], dtype=np.float64)
    prior = MaxwellBoltzmannPrior(scale=0.1, seed=0)
    samples = np.stack(
        [
            prior.sample(
                temperature=temperature,
                masses=masses,
                shape=(2, 3),
                rng=np.random.default_rng(seed),
            )
            for seed in range(400)
        ]
    )
    expected_sigma = 0.1 * np.sqrt(units.kB * temperature / masses)
    np.testing.assert_allclose(samples.std(axis=(0, 2)), expected_sigma, rtol=0.1)
    assert abs(float(samples.mean())) < 1e-3


def test_prior_scales_differ_between_datasets(mini_dataset_path):
    propagator = LiFlowDataset(path=mini_dataset_path, split="val", seed=0)
    corrector = NoisedFrameDataset(path=mini_dataset_path, split="val", seed=0)
    assert isinstance(propagator.prior, AdaptiveMaxwellBoltzmannPrior)
    assert isinstance(corrector.prior, MaxwellBoltzmannPrior)
    assert corrector.prior.scale == pytest.approx(0.1)
    assert corrector.noise.scale == pytest.approx(0.25)


@pytest.mark.parametrize("dataset_cls", DATASET_CLASSES)
@pytest.mark.parametrize("num_workers", [0, 2])
def test_build_dataloader_transports_collated_graphs(
    mini_dataset_path, dataset_cls, num_workers
):
    """``build_dataloader`` must yield real batches for both loader modes.

    Paddle's data loader flattens each collated batch into dense tensors, which
    silently drops opaque ``Data`` / ``Batch`` containers when they are not
    exposed as dictionaries.  This is a regression guard for that transport.
    """
    from omegaconf import OmegaConf

    from ppmat.datasets import build_dataloader

    cfg = OmegaConf.create(
        {
            "dataset": {
                "__class_name__": dataset_cls.__name__,
                "__init_params__": {
                    "path": str(mini_dataset_path),
                    "split": "train",
                    "seed": 0,
                },
            },
            "loader": {
                "num_workers": num_workers,
                "use_shared_memory": True,
                "collate_fn": "DefaultCollator",
                "as_dict": True,
            },
            "sampler": {
                "__class_name__": "BatchSampler",
                "__init_params__": {
                    "batch_size": 2,
                    "shuffle": False,
                    "drop_last": False,
                },
            },
        }
    )
    loader = build_dataloader(cfg)
    batches = list(loader)
    assert len(batches) == 2
    first = batches[0]
    assert isinstance(first, dict)
    for key in ("start_positions", "end_positions", "prior", "velocity", "batch"):
        assert key in first
    assert int(first["batch"].max()) == 1


def test_build_dataloader_keeps_default_collator_for_other_datasets(tmp_path):
    """LiFlow's ``as_dict`` adapter must not leak into other datasets."""
    import numpy as np
    from omegaconf import OmegaConf

    from ppmat.datasets import build_dataloader

    data_dir = tmp_path / "md17"
    data_dir.mkdir(parents=True)
    (data_dir / "splits").mkdir()
    np.savez(
        data_dir / "md17_aspirin.npz",
        z=np.array([6, 1, 8, 1], dtype=np.int32),
        R=np.stack(
            [
                np.arange(12, dtype=np.float64).reshape(4, 3) * 0.01 + i * 0.01
                for i in range(4)
            ]
        ),
        E=np.array([-1.0, -2.0, -3.0, -4.0], dtype=np.float64),
        F=np.zeros((4, 4, 3), dtype=np.float64),
    )
    np.save(data_dir / "splits" / "aspirin_train_idx.npy", np.array([3, 1]))

    cfg = OmegaConf.create(
        {
            "dataset": {
                "__class_name__": "MD17Dataset",
                "__init_params__": {
                    "path": str(data_dir / "md17_aspirin.npz"),
                    "name": "aspirin",
                    "split": "train",
                    "cache_path": str(tmp_path / "cache"),
                    "build_molecule_cfg": {
                        "format": "dict",
                        "sanitize": False,
                        "add_hs": False,
                        "remove_hs": False,
                        "kekulize": False,
                        "num_cpus": 1,
                    },
                    "build_graph_cfg": {
                        "__class_name__": "RadiusGraphConverter",
                        "__init_params__": {
                            "cutoff": 5.0,
                            "return_triplet_indices": True,
                            "num_cpus": 1,
                        },
                    },
                },
            },
            "loader": {
                "num_workers": 0,
                "use_shared_memory": True,
                "collate_fn": "DefaultCollator",
            },
            "sampler": {
                "__class_name__": "BatchSampler",
                "__init_params__": {
                    "batch_size": 2,
                    "shuffle": False,
                    "drop_last": False,
                },
            },
        }
    )
    loader = build_dataloader(cfg)
    # The adapter is opt-in: without ``as_dict`` the loader keeps the upstream
    # collator object untouched.
    from ppmat.datasets import _FieldDictCollator

    assert not isinstance(loader.collate_fn, _FieldDictCollator)

    batches = list(loader)
    assert len(batches) == 1
    batch = batches[0]
    # Upstream behaviour for a non-LiFlow dataset: Paddle's transport already
    # hands back a field mapping, and its content is untouched by this PR.
    assert isinstance(batch, dict)
    for key in ("graph", "energy", "id"):
        assert key in batch
    assert len(batch["id"]) == 2

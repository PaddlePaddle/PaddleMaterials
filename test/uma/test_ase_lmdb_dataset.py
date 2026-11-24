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

import json
import shutil
import tarfile
import zipfile
import zlib
from pathlib import Path

import lmdb
import numpy as np
import paddle
import pytest
from omegaconf import OmegaConf

from ppmat import datasets
from ppmat.datasets.ase_lmdb_dataset import ASELMDBDataset
from ppmat.datasets.ase_lmdb_dataset import OMat24Dataset
from ppmat.datasets.ase_lmdb_dataset import UMAOC20Dataset
from ppmat.datasets.collate_fn import DefaultCollator
from ppmat.models import UMA
from ppmat.utils import download


@pytest.mark.parametrize("missing_path", [False, True])
def test_omat24_cached_split_archives_are_isolated(
    tmp_path, database_path, dataset_cache, graph_cfg, missing_path
):
    paths = []
    for split in ("train", "val"):
        cache_dir = dataset_cache / "OMat24" / split
        cache_dir.mkdir(parents=True)
        with tarfile.open(cache_dir / "rattled-500.tar.gz", "w:gz") as archive:
            directory = tarfile.TarInfo("rattled-500")
            directory.type = tarfile.DIRTYPE
            directory.mode = 0o755
            archive.addfile(directory)
            archive.add(database_path, arcname="rattled-500/sample.aselmdb")

        for _ in range(2):
            dataset = OMat24Dataset(
                build_graph_cfg=graph_cfg,
                split=split,
                path=tmp_path / "missing" / split if missing_path else None,
            )
            assert len(dataset) == 1
            assert dataset.split == split
            assert Path(dataset.path) == cache_dir / "rattled-500.tar" / "rattled-500"
            assert dataset[0]["energy"].item() == -1.0
            paths.append(dataset.path)
            del dataset
    assert paths[0] == paths[1]
    assert paths[2] == paths[3]
    assert paths[0] != paths[2]


@pytest.mark.parametrize("nested", [False, True])
def test_oc20_cached_archive_selects_each_split(
    database_path, dataset_cache, graph_cfg, nested
):
    cache_dir = dataset_cache
    cache_dir.mkdir(parents=True)
    prefix = "uma_datasets/" if nested else ""
    with zipfile.ZipFile(cache_dir / "uma_datasets.zip", "w") as archive:
        if nested:
            archive.writestr(prefix, b"")
        archive.writestr(f"{prefix}README.md", b"OC20 dataset package")
        for split in ("train", "val", "test"):
            archive.write(
                database_path, f"{prefix}oc20/uma_aselmdb/{split}/sample.aselmdb"
            )

    paths = []
    for split in ("train", "val", "test"):
        dataset = UMAOC20Dataset(build_graph_cfg=graph_cfg, split=split)
        assert len(dataset) == 1
        assert dataset.split == split
        assert Path(dataset.path).parts[-3:] == ("oc20", "uma_aselmdb", split)
        assert dataset[0]["energy"].item() == -1.0
        paths.append(dataset.path)
        del dataset
    assert len(set(paths)) == 3


@pytest.mark.parametrize("dataset_class", [OMat24Dataset, UMAOC20Dataset])
@pytest.mark.parametrize("directory", [False, True])
def test_existing_local_data_does_not_download(
    database_path, dataset_cache, graph_cfg, dataset_class, directory
):
    path = database_path.parent if directory else database_path
    dataset = dataset_class(build_graph_cfg=graph_cfg, path=path)
    assert dataset.path == str(path)
    assert len(dataset) == 1
    assert dataset[0]["graph"].num_nodes == 2
    assert not dataset_cache.exists()


@pytest.mark.parametrize(
    "dataset_class, split",
    [(OMat24Dataset, "test"), (UMAOC20Dataset, "invalid")],
)
def test_invalid_split_is_rejected_before_download(dataset_class, split, dataset_cache):
    with pytest.raises(ValueError, match="supports only"):
        dataset_class(build_graph_cfg={}, split=split)
    assert not dataset_cache.exists()


@pytest.mark.parametrize("missing_path", [False, True])
def test_unregistered_dataset_requires_local_data(tmp_path, missing_path):
    with pytest.raises(FileNotFoundError, match="no download URL configured"):
        ASELMDBDataset(
            build_graph_cfg={}, path=tmp_path / "missing" if missing_path else None
        )


def test_downloaded_package_requires_requested_split(tmp_path, monkeypatch):
    (tmp_path / "oc20" / "uma_aselmdb" / "train").mkdir(parents=True)
    monkeypatch.setattr(
        download, "get_datasets_path_from_url", lambda *args: str(tmp_path)
    )
    with pytest.raises(FileNotFoundError, match="val"):
        UMAOC20Dataset(build_graph_cfg={}, split="val")


@pytest.mark.parametrize(
    "filename", ["uma_omat24_r500_s2ef.yaml", "uma_oc20_200k_s2ef.yaml"]
)
def test_uma_config_uses_registered_dataset_with_local_path(filename):
    config_dir = (
        Path(__file__).resolve().parents[2]
        / "interatomic_potentials"
        / "configs"
        / "uma"
    )
    config = OmegaConf.to_container(OmegaConf.load(config_dir / filename), resolve=True)
    assert config["Model"]["__init_params__"]["cutoff"] == (
        config["Global"]["graph_converter"]["__init_params__"]["cutoff"]
    )
    assert config["Predict"]["graph_converter"] == config["Global"]["graph_converter"]
    for split, settings in config["Dataset"].items():
        dataset_cfg = settings["dataset"]
        dataset_class = getattr(datasets, dataset_cfg["__class_name__"])
        params = dataset_cfg["__init_params__"]
        assert issubclass(dataset_class, ASELMDBDataset)
        assert params["split"] == split
        assert params["property_names"] == ["energy", "forces"]
        assert split in Path(params["path"]).parts
        assert params["build_graph_cfg"] == config["Global"]["graph_converter"]
        if split != "train":
            assert settings["sampler"]["__init_params__"]["drop_last"] is False


def test_ase_lmdb_cache_is_lazy_and_reusable(
    tmp_path, database_path, graph_cfg, monkeypatch
):
    cache_root = tmp_path / "graphs"
    dataset = ASELMDBDataset(
        path=database_path, build_graph_cfg=graph_cfg, cache_path=cache_root
    )
    assert not cache_root.exists()
    sample = dataset[0]
    cached_paths = list(cache_root.rglob("*.pkl"))
    assert len(cached_paths) == 1
    assert len(list(cache_root.rglob("*"))) == 2

    def unexpected_read(*args):
        pytest.fail("Cached samples must not be decoded or rebuilt")

    monkeypatch.setattr(dataset, "read_data", unexpected_read)
    cached = dataset[0]
    np.testing.assert_array_equal(cached["energy"], sample["energy"])
    np.testing.assert_array_equal(cached["forces"], sample["forces"])
    np.testing.assert_array_equal(cached["graph"].edges, sample["graph"].edges)
    assert DefaultCollator()([cached, cached])["forces"].shape == (4, 3)


@pytest.mark.parametrize("change", ["graph", "properties", "structure", "label_key"])
def test_cache_namespace_tracks_data_configuration(
    tmp_path, database_path, graph_cfg, change
):
    params = dict(
        path=database_path, build_graph_cfg=graph_cfg, cache_path=tmp_path / "graphs"
    )
    dataset = ASELMDBDataset(**params)
    first_cache = dataset.cache_path
    del dataset
    if change == "graph":
        graph_cfg["__init_params__"]["cutoff"] = 3.0
    elif change == "properties":
        params["property_names"] = ["energy"]
    elif change == "structure":
        params["build_structure_cfg"] = {"format": "ase_atoms", "canocial": False}
    else:
        params["energy_key"] = "total_energy"
    dataset = ASELMDBDataset(**params)
    assert dataset.cache_path != first_cache


def test_cache_overwrite_rebuilds_requested_sample(
    tmp_path, database_path, graph_cfg, monkeypatch
):
    params = dict(
        path=database_path, build_graph_cfg=graph_cfg, cache_path=tmp_path / "graphs"
    )
    dataset = ASELMDBDataset(**params)
    dataset[0]
    del dataset
    dataset = ASELMDBDataset(**params, overwrite=True)
    reads = []
    original = dataset.read_data

    def read_data(path, index=None):
        reads.append(index)
        return original(path, index=index)

    monkeypatch.setattr(dataset, "read_data", read_data)
    dataset[0]
    assert reads == [0]


def test_cache_does_not_store_transform_results(tmp_path, database_path, graph_cfg):
    def shift_energy(data):
        data["energy"] += 10
        return data

    dataset = ASELMDBDataset(
        path=database_path,
        build_graph_cfg=graph_cfg,
        cache_path=tmp_path / "graphs",
        transforms=shift_energy,
    )
    assert dataset[0]["energy"].item() == 9
    assert dataset[0]["energy"].item() == 9


def test_dataset_without_graph_converter_returns_structure_arrays(database_path):
    dataset = ASELMDBDataset(path=database_path)
    batch = DefaultCollator()([dataset[0], dataset[0]])
    assert "graph" not in batch
    assert batch["structure_array"]["cart_coords"].shape == (4, 3)
    assert batch["structure_array"]["lattice"].shape == (2, 3, 3)
    np.testing.assert_array_equal(batch["structure_array"]["num_atoms"], [2, 2])


@pytest.mark.parametrize("compressed", [False, True])
@pytest.mark.parametrize("nested_labels", [False, True])
def test_ase_lmdb_decodes_array_records(
    database_path, graph_cfg, compressed, nested_labels
):
    database = lmdb.open(str(database_path), subdir=False)
    with database.begin(write=True) as transaction:
        record = json.loads(zlib.decompress(transaction.get(b"1")))
        expected_forces = np.asarray(record.pop("forces"), dtype="float32")
        labels = {
            "energy": record.pop("energy"),
            "forces": {
                "__ndarray__": [[2, 3], "float64", expected_forces.ravel().tolist()]
            },
        }
        for key, dtype in [
            ("numbers", "int64"),
            ("positions", "float64"),
            ("cell", "float64"),
        ]:
            value = np.asarray(record[key], dtype=dtype)
            record[key] = {
                "__ndarray__": [list(value.shape), dtype, value.ravel().tolist()]
            }
        if nested_labels:
            record["data"] = labels
        else:
            record.update(labels)
        payload = json.dumps(record).encode()
        transaction.put(b"1", zlib.compress(payload) if compressed else payload)
    database.close()
    dataset = ASELMDBDataset(path=database_path, build_graph_cfg=graph_cfg)
    sample = dataset[0]
    assert sample["energy"].dtype == sample["forces"].dtype == np.float32
    np.testing.assert_array_equal(sample["energy"], [-1.0])
    np.testing.assert_array_equal(sample["forces"], expected_forces)
    assert sample["graph"].num_nodes == 2


@pytest.mark.parametrize("property_name", ["energy", "forces"])
def test_missing_labels_preserve_nan_and_expected_shape(database_path, property_name):
    database = lmdb.open(str(database_path), subdir=False)
    with database.begin(write=True) as transaction:
        record = json.loads(zlib.decompress(transaction.get(b"1")))
        record.pop(property_name)
        transaction.put(b"1", zlib.compress(json.dumps(record).encode()))
    database.close()
    dataset = ASELMDBDataset(path=database_path, property_names=property_name)
    label = dataset[0][property_name]
    assert label.shape == ((1,) if property_name == "energy" else (2, 3))
    assert label.dtype == np.float32
    assert np.isnan(label).all()


@pytest.mark.parametrize("property_name", ["energy", "forces"])
def test_filtering_requires_only_selected_property(database_path, property_name):
    database = lmdb.open(str(database_path), subdir=False)
    with database.begin(write=True) as transaction:
        record = json.loads(zlib.decompress(transaction.get(b"1")))
        record.pop("forces" if property_name == "energy" else "energy")
        transaction.put(b"1", zlib.compress(json.dumps(record).encode()))
    database.close()
    dataset = ASELMDBDataset(
        path=database_path, property_names=property_name, filter_unvalid=True
    )
    assert len(dataset) == 1
    assert np.isfinite(dataset[0][property_name]).all()


def test_failed_cache_write_preserves_previous_sample(tmp_path, monkeypatch):
    cache_path = tmp_path / "graphs" / "sample.pkl"
    ASELMDBDataset.save_to_cache(str(cache_path), {"energy": -1.0})

    def fail_dump(data, stream):
        stream.write(b"incomplete")
        raise OSError("Cache write failed")

    monkeypatch.setattr("ppmat.datasets.ase_lmdb_dataset.pickle.dump", fail_dump)
    with pytest.raises(OSError, match="Cache write failed"):
        ASELMDBDataset.save_to_cache(str(cache_path), {"energy": 2.0})
    assert ASELMDBDataset.load_from_cache(str(cache_path)) == {"energy": -1.0}
    assert list(cache_path.parent.iterdir()) == [cache_path]


def test_filtering_and_subsetting_preserve_lmdb_row_ids(database_path, graph_cfg):
    database = lmdb.open(str(database_path), subdir=False)
    with database.begin(write=True) as transaction:
        record = json.loads(zlib.decompress(transaction.get(b"1")))
        record["energy"] = None
        transaction.put(b"2", zlib.compress(json.dumps(record).encode()))
        record["energy"] = -3.0
        transaction.put(b"3", zlib.compress(json.dumps(record).encode()))
        transaction.put(b"length", b"3")
    database.close()
    dataset = ASELMDBDataset(
        path=database_path, build_graph_cfg=graph_cfg, filter_unvalid=True
    )
    assert dataset.indices == [(0, 1), (0, 3)]
    assert len(dataset) == 2
    assert dataset[1]["energy"].item() == -3.0
    del dataset
    dataset = ASELMDBDataset(
        path=database_path,
        build_graph_cfg=graph_cfg,
        offset=1,
        limit=2,
        filter_unvalid=True,
    )
    assert len(dataset) == 1
    assert dataset[0]["energy"].item() == -3.0


def test_lmdb_shard_selection_stops_at_requested_limit(
    tmp_path, database_path, graph_cfg
):
    shards = tmp_path / "shards"
    shards.mkdir()
    for name in ("a.aselmdb", "b.aselmdb", "c.aselmdb"):
        shutil.copyfile(database_path, shards / name)
    dataset = ASELMDBDataset(path=shards, build_graph_cfg=graph_cfg, limit=1)
    assert dataset.indices == [(0, 1)]
    assert len(dataset.databases) == 1
    del dataset
    dataset = ASELMDBDataset(path=shards, build_graph_cfg=graph_cfg, offset=1, limit=2)
    assert dataset.indices == [(1, 1), (2, 1)]
    assert dataset[1]["energy"].item() == -1.0


def test_lmdb_without_length_sorts_numeric_record_keys(database_path, graph_cfg):
    database = lmdb.open(str(database_path), subdir=False)
    with database.begin(write=True) as transaction:
        record = json.loads(zlib.decompress(transaction.get(b"1")))
        transaction.delete(b"length")
        transaction.put(b"metadata", b"{}")
        for row_id in (2, 10):
            record["energy"] = -float(row_id)
            transaction.put(
                str(row_id).encode(), zlib.compress(json.dumps(record).encode())
            )
    database.close()
    dataset = ASELMDBDataset(
        path=database_path, build_graph_cfg=graph_cfg, offset=1, limit=2
    )
    assert dataset.indices == [(0, 2), (0, 10)]
    assert dataset[0]["energy"].item() == -2.0
    assert dataset[1]["energy"].item() == -10.0


@pytest.mark.parametrize("params", [{"offset": -1}, {"limit": -1}, {"limit": 0}])
def test_invalid_or_empty_record_selection(database_path, graph_cfg, params):
    with pytest.raises(ValueError):
        ASELMDBDataset(path=database_path, build_graph_cfg=graph_cfg, **params)


@pytest.mark.parametrize("compressed", [False, True])
def test_invalid_record_is_logged_and_raised(database_path, monkeypatch, compressed):
    database = lmdb.open(str(database_path), subdir=False)
    with database.begin(write=True) as transaction:
        payload = b"{invalid json"
        transaction.put(b"1", zlib.compress(payload) if compressed else payload)
    database.close()
    errors = []
    monkeypatch.setattr("ppmat.datasets.ase_lmdb_dataset.logger.error", errors.append)
    dataset = ASELMDBDataset(path=database_path)
    with pytest.raises(json.JSONDecodeError):
        dataset[0]
    assert len(errors) == 1
    assert str(database_path) in errors[0]


def test_missing_record_is_logged_and_raised(database_path, monkeypatch):
    database = lmdb.open(str(database_path), subdir=False)
    with database.begin(write=True) as transaction:
        transaction.put(b"length", b"2")
    database.close()
    errors = []
    monkeypatch.setattr("ppmat.datasets.ase_lmdb_dataset.logger.error", errors.append)
    dataset = ASELMDBDataset(path=database_path)
    with pytest.raises(KeyError, match="Missing row 2"):
        dataset[1]
    assert len(errors) == 1
    assert "Missing row 2" in errors[0]


def test_missing_split_is_logged_and_raised(tmp_path, monkeypatch):
    errors = []
    monkeypatch.setattr("ppmat.datasets.ase_lmdb_dataset.logger.error", errors.append)
    with pytest.raises(FileNotFoundError):
        ASELMDBDataset(path=tmp_path)
    assert len(errors) == 1
    assert str(tmp_path) in errors[0]


@pytest.mark.parametrize("num_workers", [0, 1])
def test_registered_dataloader_guarantees_label_tensors(
    database_path, graph_cfg, num_workers
):
    config = {
        "dataset": {
            "__class_name__": "OMat24Dataset",
            "__init_params__": {
                "path": str(database_path),
                "build_graph_cfg": graph_cfg,
            },
        },
        "loader": {
            "num_workers": num_workers,
            "use_shared_memory": False,
            "collate_fn": "DefaultCollator",
        },
        "sampler": {
            "__class_name__": "BatchSampler",
            "__init_params__": {
                "batch_size": 1,
                "shuffle": False,
                "drop_last": False,
            },
        },
    }
    batch = next(iter(datasets.build_dataloader(config)))
    for key, shape in (("energy", [1, 1]), ("forces", [2, 3])):
        assert isinstance(batch[key], paddle.Tensor)
        assert batch[key].dtype == paddle.float32
        assert list(batch[key].shape) == shape
    model = UMA(
        sphere_channels=16,
        edge_channels=16,
        hidden_channels=16,
        num_distance_basis=16,
        num_layers=1,
    )
    optimizer = paddle.optimizer.Adam(learning_rate=1e-4, parameters=model.parameters())
    loss = model(batch)["loss_dict"]["loss"]
    assert bool(paddle.isfinite(loss))
    loss.backward()
    assert all(
        bool(paddle.isfinite(parameter.grad).all())
        for parameter in model.parameters()
        if parameter.grad is not None
    )
    optimizer.step()
    assert all(
        bool(paddle.isfinite(parameter).all()) for parameter in model.parameters()
    )

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
import tarfile
import zipfile
import zlib
from pathlib import Path

import lmdb
import pytest
from omegaconf import OmegaConf

from ppmat import datasets
from ppmat.datasets.ase_lmdb_dataset import ASELMDBDataset
from ppmat.datasets.ase_lmdb_dataset import OMat24Dataset
from ppmat.datasets.ase_lmdb_dataset import UMAOC20Dataset
from ppmat.utils import download


@pytest.fixture
def graph_cfg():
    return {
        "__class_name__": "FindPointsInSpheres",
        "__init_params__": {"cutoff": 4.0, "max_neighbors": 8, "num_cpus": 1},
    }


@pytest.fixture
def database_path(tmp_path):
    path = tmp_path / "sample.aselmdb"
    record = {
        "numbers": [14, 14],
        "positions": [[0.0, 0.0, 0.0], [0.875, 0.875, 0.875]],
        "cell": [[3.5, 0.0, 0.0], [0.0, 3.5, 0.0], [0.0, 0.0, 3.5]],
        "pbc": [True, True, True],
        "energy": -1.0,
        "forces": [[0.1, 0.2, 0.3], [-0.1, -0.2, -0.3]],
    }
    database = lmdb.open(str(path), subdir=False, map_size=1024**2)
    with database.begin(write=True) as transaction:
        transaction.put(b"length", b"1")
        transaction.put(b"1", zlib.compress(json.dumps(record).encode()))
    database.close()
    return path


@pytest.fixture
def dataset_cache(monkeypatch, tmp_path):
    cache = tmp_path / "cache"
    monkeypatch.setattr(download, "DATASETS_HOME", str(cache))

    def unexpected_download(*args, **kwargs):
        pytest.fail("Unexpected network download")

    monkeypatch.setattr(download, "_download", unexpected_download)
    return cache


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


def test_downloaded_package_requires_requested_split(tmp_path):
    (tmp_path / "oc20" / "uma_aselmdb" / "train").mkdir(parents=True)
    dataset = UMAOC20Dataset.__new__(UMAOC20Dataset)
    with pytest.raises(FileNotFoundError, match="val"):
        dataset._get_downloaded_split_path(str(tmp_path), "val")


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
    for split, settings in config["Dataset"].items():
        dataset_cfg = settings["dataset"]
        dataset_class = getattr(datasets, dataset_cfg["__class_name__"])
        params = dataset_cfg["__init_params__"]
        assert issubclass(dataset_class, ASELMDBDataset)
        assert params["split"] == split
        assert split in Path(params["path"]).parts
        assert params["build_graph_cfg"] == config["Global"]["graph_converter"]

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
import zlib

import lmdb
import numpy as np
import pytest
from pymatgen.core import Lattice
from pymatgen.core import Structure

from ppmat.datasets.custom_data_type import ConcatNumpyWarper
from ppmat.models import UMA
from ppmat.models import FindPointsInSpheres
from ppmat.utils import download


@pytest.fixture
def graph_cfg():
    return {
        "__class_name__": "FindPointsInSpheres",
        "__init_params__": {"cutoff": 4.0, "max_neighbors": 8, "num_cpus": 1},
    }


@pytest.fixture
def small_uma():
    return UMA(
        sphere_channels=16,
        edge_channels=16,
        hidden_channels=16,
        num_distance_basis=16,
        num_layers=1,
    )


@pytest.fixture
def uma_samples():
    converter = FindPointsInSpheres(cutoff=4.0, max_neighbors=8, num_cpus=1)
    structures = [
        Structure(Lattice.cubic(3.5), ["Si", "Si"], [[0, 0, 0], [0.25, 0.25, 0.25]]),
        Structure(
            Lattice.cubic(4.0),
            ["C", "C", "C"],
            [[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5]],
        ),
    ]
    return [
        {
            "graph": converter(structure),
            "energy": np.zeros([1], dtype="float32"),
            "forces": ConcatNumpyWarper(np.zeros([len(structure), 3], dtype="float32")),
        }
        for structure in structures
    ]


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

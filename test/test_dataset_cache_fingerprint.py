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


"""A dataset cache is reused only for the data it was built from."""

from __future__ import annotations

import json
import os

import pytest
from pymatgen.core import Lattice
from pymatgen.core import Structure

from ppmat.datasets.cache_fingerprint import FINGERPRINT_FILE
from ppmat.datasets.mp2018_dataset import MP2018Dataset

STRUCTURE_CFG = {"format": "cif_str", "primitive": False, "niggli": True, "num_cpus": 1}


def _cubic_cif(num_atoms):
    lattice = Lattice.cubic(3.0 * num_atoms)
    coords = [[i / num_atoms, 0.0, 0.0] for i in range(num_atoms)]
    return Structure(lattice, ["Si"] * num_atoms, coords).to(fmt="cif")


def _write_mp2018_json(path, atom_counts):
    data = {
        "structure": {str(i): _cubic_cif(n) for i, n in enumerate(atom_counts)},
        "formation_energy_per_atom": {
            str(i): -0.1 * n for i, n in enumerate(atom_counts)
        },
    }
    path.write_text(json.dumps(data))
    return str(path)


def _load(path, cache_path):
    return MP2018Dataset(
        path=path,
        property_names=["formation_energy_per_atom"],
        build_structure_cfg=STRUCTURE_CFG,
        cache_path=str(cache_path),
    )


def _cached_atom_counts(dataset):
    return [len(dataset.load_from_cache(p)) for p in dataset.structures]


@pytest.fixture
def data(tmp_path):
    return {
        "a": _write_mp2018_json(tmp_path / "a.json", [1, 2, 3]),
        "b": _write_mp2018_json(tmp_path / "b.json", [4, 5, 6]),
        "c": _write_mp2018_json(tmp_path / "c.json", [4, 5]),
        "cache": tmp_path / "cache",
    }


def test_cache_is_rebuilt_when_data_changes(data):
    assert _cached_atom_counts(_load(data["a"], data["cache"])) == [1, 2, 3]
    # Same number of samples and the same build settings, different structures.
    assert _cached_atom_counts(_load(data["b"], data["cache"])) == [4, 5, 6]
    assert _cached_atom_counts(_load(data["c"], data["cache"])) == [4, 5]


def _mark_first_sample(dataset):
    """Replace the first cached structure so a reused cache can be recognized."""
    marker = Structure.from_str(_cubic_cif(7), fmt="cif")
    dataset.save_to_cache(dataset.structures[0], marker)


def test_cache_is_reused_for_unchanged_data(data):
    _mark_first_sample(_load(data["a"], data["cache"]))

    assert _cached_atom_counts(_load(data["a"], data["cache"])) == [7, 2, 3]


def test_legacy_cache_without_fingerprint(data):
    _load(data["a"], data["cache"])
    fingerprint = data["cache"] / FINGERPRINT_FILE
    stored = json.loads(fingerprint.read_text())
    os.remove(fingerprint)

    # A cache written before fingerprints existed is adopted if the sample count
    # matches, and the fingerprint is recorded for later checks.
    assert _cached_atom_counts(_load(data["a"], data["cache"])) == [1, 2, 3]
    assert json.loads(fingerprint.read_text()) == stored
    assert _cached_atom_counts(_load(data["b"], data["cache"])) == [4, 5, 6]

    # A legacy cache with a different sample count is rebuilt.
    os.remove(fingerprint)
    assert _cached_atom_counts(_load(data["c"], data["cache"])) == [4, 5]


def test_interrupted_rebuild_is_not_reused(data):
    fingerprint = data["cache"] / FINGERPRINT_FILE
    _mark_first_sample(_load(data["a"], data["cache"]))
    fingerprint.write_text(json.dumps({"version": 1, "status": "building"}))

    assert _cached_atom_counts(_load(data["a"], data["cache"])) == [1, 2, 3]

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

import json
import shutil
import zlib

import lmdb
import numpy as np
import paddle
import pytest

from ppmat import datasets
from ppmat.datasets.ase_lmdb_dataset import OMat24Dataset
from ppmat.datasets.ase_lmdb_dataset import UMAOC20Dataset
from ppmat.datasets.collate_fn import DefaultCollator
from ppmat.utils import download


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("split", ["train", "val", "test"])
def test_uma_dataset_resolves_only_requested_split(
    tmp_path, database_path, graph_cfg, monkeypatch, nested, split
):
    extracted_root = tmp_path / "uma_datasets" if nested else tmp_path
    requested_split = extracted_root / "oc20" / "uma_aselmdb" / split
    requested_split.mkdir(parents=True)
    shutil.copyfile(database_path, requested_split / "sample.aselmdb")
    monkeypatch.setattr(
        download, "get_datasets_path_from_url", lambda *args: str(tmp_path)
    )
    dataset = UMAOC20Dataset(build_graph_cfg=graph_cfg, split=split)

    assert dataset.path == str(requested_split)
    assert len(dataset) == 1


@pytest.mark.parametrize(
    "dataset_class, split",
    [
        (OMat24Dataset, "train"),
        (OMat24Dataset, "val"),
        (UMAOC20Dataset, "train"),
        (UMAOC20Dataset, "val"),
        (UMAOC20Dataset, "test"),
    ],
)
def test_uma_dataset_reads_local_split(tmp_path, dataset_class, split):
    database_path = tmp_path / f"{split}.aselmdb"
    database = lmdb.open(str(database_path), subdir=False, map_size=1024**2)
    record = {
        "numbers": [14, 14],
        "positions": [[0.0, 0.0, 0.0], [0.875, 0.875, 0.875]],
        "cell": [[3.5, 0.0, 0.0], [0.0, 3.5, 0.0], [0.0, 0.0, 3.5]],
        "pbc": [True, True, True],
        "energy": -1.0,
        "forces": [[0.1, 0.2, 0.3], [-0.1, -0.2, -0.3]],
    }
    with database.begin(write=True) as transaction:
        transaction.put(b"length", b"1")
        transaction.put(b"1", zlib.compress(json.dumps(record).encode()))
    database.close()

    assert getattr(datasets, dataset_class.__name__) is dataset_class
    dataset = dataset_class(
        path=str(database_path),
        split=split,
        build_graph_cfg={
            "__class_name__": "FindPointsInSpheres",
            "__init_params__": {"cutoff": 4.0, "max_neighbors": 8, "num_cpus": 1},
        },
    )
    sample = dataset[0]

    assert len(dataset) == 1
    assert sample["energy"].shape == (1,)
    assert sample["forces"].shape == (2, 3)
    assert sample["graph"].num_nodes == 2
    np.testing.assert_array_equal(sample["energy"], np.asarray([-1.0], dtype="float32"))
    expected_forces = np.asarray(record["forces"], dtype="float32")
    np.testing.assert_array_equal(sample["forces"], expected_forces)

    batch = DefaultCollator()([sample, dataset[0]])
    assert batch["graph"].num_graph == 2
    np.testing.assert_array_equal(batch["energy"], [[-1.0], [-1.0]])
    np.testing.assert_array_equal(batch["forces"], np.tile(expected_forces, (2, 1)))


def test_uma_forward_with_default_collator(uma_samples, small_uma):
    batch = DefaultCollator()(uma_samples)
    model = small_uma
    output = model(batch)

    assert output["pred_dict"]["energy"].shape == [2, 1]
    assert output["pred_dict"]["forces"].shape == [5, 3]
    output["loss_dict"]["loss"].backward()

    prediction = model.predict(batch["graph"])
    assert prediction["energy"].shape == (2, 1)
    assert prediction["forces"].shape == (5, 3)


@pytest.mark.parametrize(
    "return_loss, return_prediction", [(True, False), (False, True), (True, True)]
)
def test_uma_forward_return_flags(
    uma_samples, small_uma, return_loss, return_prediction
):
    batch = DefaultCollator()(uma_samples)
    if not return_loss:
        batch = {"graph": batch["graph"]}
    output = small_uma(
        batch, return_loss=return_loss, return_prediction=return_prediction
    )
    assert set(output) == {"loss_dict", "pred_dict"}
    assert bool(output["loss_dict"]) == return_loss
    assert bool(output["pred_dict"]) == return_prediction
    if return_loss:
        assert set(output["loss_dict"]) == {"energy", "forces", "loss"}


def test_uma_forward_requires_loss_or_prediction(small_uma):
    with pytest.raises(ValueError, match="At least one"):
        small_uma({}, return_loss=False, return_prediction=False)


def test_uma_predict_supports_lists_and_disables_gradients(
    uma_samples, small_uma, monkeypatch
):
    observed = []
    original = small_uma._forward

    def forward(data):
        observed.append(paddle.is_grad_enabled())
        output = original(data)
        assert all(value.stop_gradient for value in output.values())
        return output

    monkeypatch.setattr(small_uma, "_forward", forward)
    results = small_uma.predict([sample["graph"] for sample in uma_samples])
    assert isinstance(results, list)
    assert len(results) == 2
    for result, num_atoms in zip(results, [2, 3]):
        assert result["energy"].shape == (1, 1)
        assert result["forces"].shape == (num_atoms, 3)
    single = small_uma.predict(uma_samples[0]["graph"])
    assert isinstance(single, dict)
    assert single["forces"].shape == (2, 3)
    assert small_uma.predict([]) == []
    assert observed == [False, False, False]
    assert paddle.is_grad_enabled()


def test_uma_predict_preserves_batches_inside_lists(uma_samples, small_uma):
    batched_graph = DefaultCollator()(uma_samples)["graph"]
    results = small_uma.predict([batched_graph, uma_samples[0]["graph"]])
    assert len(results) == 2
    for result, num_graphs, num_atoms in zip(results, [2, 1], [5, 2]):
        assert result["energy"].shape == (num_graphs, 1)
        assert result["forces"].shape == (num_atoms, 3)


def test_uma_predict_restores_gradients_after_failure(small_uma, monkeypatch):
    def fail_forward(data):
        assert not paddle.is_grad_enabled()
        raise ValueError("Prediction failed")

    monkeypatch.setattr(small_uma, "_forward", fail_forward)
    with pytest.raises(ValueError, match="Prediction failed"):
        small_uma.predict(None)
    assert paddle.is_grad_enabled()


def test_uma_loss_preserves_structure_weighted_objective(uma_samples, small_uma):
    batch = DefaultCollator()(uma_samples)
    batch["graph"].tensor()
    predicted_energy = np.asarray([[4.0], [9.0]], dtype="float32")
    predicted_forces = np.asarray(
        [[3, 4, 0], [0, 0, 2], [0, 0, 1], [0, 2, 0], [3, 0, 0]],
        dtype="float32",
    )
    pred = {
        "energy": paddle.to_tensor(predicted_energy, stop_gradient=False),
        "forces": paddle.to_tensor(predicted_forces, stop_gradient=False),
    }
    losses = small_uma.loss_fn(pred, batch)
    expected_energy = (4 / 2 + 9 / 3) / 2
    expected_forces = ((5 + 2) / 2 + (1 + 2 + 3) / 3) / 2
    np.testing.assert_allclose(float(losses["energy"]), expected_energy)
    np.testing.assert_allclose(float(losses["forces"]), expected_forces)
    np.testing.assert_allclose(
        float(losses["loss"]), 10 * expected_energy + 30 * expected_forces
    )
    losses["loss"].backward()
    np.testing.assert_allclose(pred["energy"].grad.numpy(), [[2.5], [10 / 6]])
    assert bool(paddle.isfinite(pred["forces"].grad).all())

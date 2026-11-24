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

from pathlib import Path

import numpy as np
import paddle
import pytest
from omegaconf import OmegaConf
from pymatgen.core import Lattice
from pymatgen.core import Structure

from ppmat.datasets.build_grid import BuildGrid
from ppmat.datasets.collate_fn import DensityCollator
from ppmat.models import build_graph_converter
from ppmat.models import build_model
from ppmat.predictor import FieldPredictor
from ppmat.predictor import base as predictor_base


@pytest.mark.parametrize("config_name", ["infgcn_md17_ethane", "infgcn_mp"])
def test_original_infgcn_training_checkpoint_and_prediction(
    config_name, tmp_path, monkeypatch
):
    """Exercise both original decoders through training and FieldPredictor."""
    paddle.seed(42)
    root = Path(__file__).resolve().parents[1]
    config_path = root / "electronic_structure/configs/infgcn" / f"{config_name}.yaml"
    config = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    # Keep the released network dimensions; use an offline vocabulary for the
    # three elements in this small regression structure.
    vocab = {"atom": {"num_embeddings": 3, "atomic_number_to_id": {1: 0, 6: 1, 8: 2}}}
    structure = Structure(
        Lattice.cubic(4),
        ["C", "H", "H"],
        [[0.2, 0.3, 0.4], [1.1, 0.3, 0.4], [0.2, 1.2, 0.4]],
        coords_are_cartesian=True,
    )
    grid = BuildGrid(format="array")(
        {
            "shape": (2, 2, 2),
            "origin": np.array([0.5, 0.6, 0.7]),
            "voxel_vectors": np.eye(3) * 2,
        }
    )
    converter = build_graph_converter(config["Global"]["graph_converter"], vocab=vocab)
    samples = [
        {
            "graph": converter.from_structure(structure),
            "density": np.ones(8, dtype=np.float32),
            "grid_coord": grid.cartesian_coordinates().astype(np.float32),
            "info": {"cell": np.asarray(grid.cell_vectors, dtype=np.float32)},
        }
        for _ in range(2)
    ]
    loader = paddle.io.DataLoader(samples, batch_size=2, collate_fn=DensityCollator())
    batch = next(iter(loader))
    model = build_model(config["Model"], vocab=vocab)
    optimizer = paddle.optimizer.Adam(learning_rate=1e-4, parameters=model.parameters())
    model.train()
    output = model(batch)
    loss = output["loss_dict"]["loss"]
    assert bool(paddle.isfinite(loss))
    loss.backward()
    assert model.embedding.weight.grad is not None
    assert bool(paddle.isfinite(model.embedding.weight.grad).all())
    optimizer.step()
    optimizer.clear_grad()

    model.eval()
    with paddle.no_grad():
        full = model(batch, return_loss=False)["pred_dict"]["density"]
        chunked = model({**batch, "grid_batch_size": 2}, return_loss=False)[
            "pred_dict"
        ]["density"]
    np.testing.assert_allclose(chunked.numpy(), full.numpy(), rtol=1e-5, atol=1e-5)

    checkpoint = tmp_path / "best.pdparams"
    paddle.save(model.state_dict(), str(checkpoint))
    monkeypatch.setattr(predictor_base, "build_vocab", lambda name: vocab)
    predictor = FieldPredictor(
        config_path=str(config_path), checkpoint_path=str(checkpoint), device="cpu"
    )
    prediction = predictor.from_structures(structure, grid, grid_batch_size=2)[
        "density"
    ]
    assert list(prediction.shape) == [8]
    assert bool(paddle.isfinite(prediction).all())
    np.testing.assert_allclose(
        prediction.numpy(), full[0].numpy(), rtol=1e-5, atol=1e-5
    )

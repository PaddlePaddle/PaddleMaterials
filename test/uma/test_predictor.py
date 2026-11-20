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

import importlib.util
import sys
import zipfile
from pathlib import Path

import numpy as np
import paddle
import pandas as pd
import pytest
from omegaconf import OmegaConf
from pymatgen.core import Structure

from ppmat.models import MODEL_REGISTRY
from ppmat.predictor import PotentialPredictor
from ppmat.utils import download

ROOT = Path(__file__).resolve().parents[2]
CIF_FILE = ROOT / "interatomic_potentials/example_data/cifs/mp-18767-LiMnO2.cif"
MODEL_NAMES = ["uma_omat24_r500_s2ef", "uma_oc20_200k_s2ef"]


@pytest.fixture(autouse=True)
def cpu_device():
    previous = paddle.get_device()
    paddle.set_device("cpu")
    try:
        yield
    finally:
        paddle.set_device(previous)


@pytest.fixture
def cached_model_packages(tmp_path, small_uma, graph_cfg, monkeypatch):
    weights_cache = tmp_path / "weights"
    weights_cache.mkdir()
    monkeypatch.setattr(download, "WEIGHTS_HOME", str(weights_cache))

    def unexpected_download(*args, **kwargs):
        pytest.fail("The cached model package must not require a network download")

    monkeypatch.setattr(download, "_download", unexpected_download)
    packages = {}
    graph_cfg["__init_params__"]["cutoff"] = small_uma.cutoff
    with zipfile.ZipFile(weights_cache / "uma_checkpoints.zip", "w") as archive:
        for index, name in enumerate(MODEL_NAMES):
            assert MODEL_REGISTRY[name].endswith("/uma_checkpoints.zip")
            directory = tmp_path / name
            directory.mkdir()
            config = {
                "Model": {
                    "__class_name__": "UMA",
                    "__init_params__": {
                        "sphere_channels": 16,
                        "edge_channels": 16,
                        "hidden_channels": 16,
                        "num_distance_basis": 16,
                        "num_layers": 1,
                    },
                },
                "Predict": {"graph_converter": graph_cfg, "eval_with_no_grad": True},
            }
            config_path = directory / f"{name}.yaml"
            OmegaConf.save(OmegaConf.create(config), config_path)
            archive.write(config_path, f"{name}/{config_path.name}")
            states = {}
            for filename, offset in [("best.pdparams", 0), ("latest.pdparams", 10)]:
                state = {
                    key: value.clone() for key, value in small_uma.state_dict().items()
                }
                state["energy_head.4.bias"] = paddle.full_like(
                    state["energy_head.4.bias"], index + offset + 1
                )
                path = directory / filename
                paddle.save(state, str(path))
                archive.write(path, f"{name}/{filename}")
                states[filename] = state
            packages[name] = (directory, states)
    return packages


@pytest.mark.parametrize("model_name", MODEL_NAMES)
@pytest.mark.parametrize("weights_name", [None, "latest.pdparams"])
def test_registered_model_packages_select_their_own_weights(
    cached_model_packages, model_name, weights_name, tmp_path
):
    predictor = PotentialPredictor(
        model_name=model_name, weights_name=weights_name, device="cpu"
    )
    _, states = cached_model_packages[model_name]
    expected = states[weights_name or "best.pdparams"]
    for key, value in predictor.model.state_dict().items():
        np.testing.assert_array_equal(value.numpy(), expected[key].numpy())
    assert not predictor.model.training
    save_path = tmp_path / "prediction.csv"
    results = predictor.from_cif_file(str(CIF_FILE), str(save_path))
    assert len(results) == 1
    assert results[0]["energy"].shape == (1, 1)
    assert results[0]["forces"].shape == (len(Structure.from_file(CIF_FILE)), 3)
    assert all(np.isfinite(value).all() for value in results[0].values())
    assert pd.read_csv(save_path).columns.tolist() == ["cif_file", "energy", "forces"]


@pytest.fixture
def predict_cli():
    path = ROOT / "interatomic_potentials/predict.py"
    spec = importlib.util.spec_from_file_location("uma_predict_cli", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("source", ["registered", "local"])
def test_predict_cli_runs_with_cached_or_local_model(
    cached_model_packages, predict_cli, monkeypatch, tmp_path, source
):
    name = MODEL_NAMES[0]
    directory, _ = cached_model_packages[name]
    model_args = (
        ["--model_name", name]
        if source == "registered"
        else [
            "--config_path",
            str(directory / f"{name}.yaml"),
            "--checkpoint_path",
            str(directory / "best.pdparams"),
        ]
    )
    output = tmp_path / "results"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "predict.py",
            *model_args,
            "--device",
            "cpu",
            "--input_path",
            str(CIF_FILE),
            "--input_format",
            "cif",
            "--output_path",
            str(output),
            "Predict.eval_with_no_grad=True",
        ],
    )
    predict_cli.main()
    saved = pd.read_csv(output / "result.csv")
    assert len(saved) == 1
    assert saved["cif_file"].tolist() == [str(CIF_FILE)]


@pytest.mark.parametrize(
    "model_args",
    [
        ["--model_name", MODEL_NAMES[0], "--checkpoint_path", "best.pdparams"],
        ["--config_path", "model.yaml", "--weights_name", "best.pdparams"],
        ["--model_name", MODEL_NAMES[0], "--config_path", "model.yaml"],
    ],
)
def test_predict_cli_rejects_conflicting_model_sources(predict_cli, model_args):
    with pytest.raises(SystemExit):
        predict_cli.parse_args(
            [
                *model_args,
                "--input_path",
                "sample.cif",
                "--input_format",
                "cif",
            ]
        )

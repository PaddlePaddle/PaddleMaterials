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

"""Model package contract for the released LiFlow checkpoint.

The default tests are **offline**: they build a package with the released
layout (``liflow_universal.yaml`` + ``checkpoints/best.pdparams``) from the
in-repo ``LiFlow`` class and drive the real registry code path with
``get_weights_path_from_url`` redirected to that directory.  They therefore
need neither the BCE network nor a pre-populated ``~/.paddlemat/weights``.

Tests that touch the network or an externally provided package are opt-in:

* ``LIFLOW_NETWORK_TESTS=1`` — download the published package and strict-load it.
* ``LIFLOW_PACKAGE_DIR=<dir>`` — strict-load an already extracted package.
"""

from __future__ import annotations

import os
import zipfile
from pathlib import Path

import paddle
import pytest
from omegaconf import OmegaConf

import ppmat.models as models_mod
from ppmat.models import build_model
from ppmat.models import build_model_from_name
from ppmat.utils import download as download_mod

MODEL_NAME = "liflow_universal"
REPO_ROOT = Path(__file__).resolve().parents[1]
PROPAGATOR_CONFIG = (
    REPO_ROOT
    / "molecular_dynamics_integrator"
    / "liflow"
    / "configs"
    / "liflow_universal_propagator.yaml"
)
NETWORK_TESTS = os.environ.get("LIFLOW_NETWORK_TESTS") == "1"


@pytest.fixture(scope="module")
def offline_package(tmp_path_factory) -> Path:
    """Build a checkpoint package locally, using the released file layout."""
    config = OmegaConf.to_container(
        OmegaConf.load(PROPAGATOR_CONFIG)["Model"], resolve=True
    )
    package = tmp_path_factory.mktemp("liflow_package")
    (package / "checkpoints").mkdir()

    model = build_model(config)
    paddle.save(model.state_dict(), str(package / "checkpoints" / "best.pdparams"))
    OmegaConf.save({"Model": config}, package / f"{MODEL_NAME}.yaml")
    return package


@pytest.fixture
def local_package(monkeypatch, offline_package: Path) -> Path:
    monkeypatch.setattr(
        models_mod.download,
        "get_weights_path_from_url",
        lambda _: str(offline_package),
    )
    return offline_package


def test_liflow_package_contract(local_package: Path):
    model, config = build_model_from_name(MODEL_NAME, strict_weights=True)
    assert type(model).__name__ == "LiFlow"
    assert config["Model"]["__class_name__"] == "LiFlow"
    state = paddle.load(str(local_package / "checkpoints" / "best.pdparams"))
    assert set(model.state_dict()) == set(state)
    for name, expected in model.state_dict().items():
        assert tuple(state[name].shape) == tuple(expected.shape)


def test_liflow_package_arch_matches_config(local_package: Path):
    model, config = build_model_from_name(MODEL_NAME, strict_weights=True)
    params = config["Model"]["__init_params__"]
    assert model.num_features == params["num_features"]
    assert model.num_layers == params["num_layers"]
    assert model.atom_embedding.weight.shape[0] == params["num_elements"]


def test_liflow_registry_url_is_consistent():
    url = models_mod.MODEL_REGISTRY[MODEL_NAME]
    assert url.startswith("https://paddle-org.bj.bcebos.com/paddlematerials/")
    assert url.endswith(f"/{MODEL_NAME}.zip")
    assert "liflow_universal_propagator" not in models_mod.MODEL_REGISTRY
    assert "liflow_universal_corrector" not in models_mod.MODEL_REGISTRY


def test_liflow_offline_install_with_isolated_home(
    tmp_path, monkeypatch, offline_package: Path
):
    """Isolated ``$HOME``, no network: registry + download + Windows zip + load."""
    home = tmp_path / "home"
    weights = home / ".paddlemat" / "weights"
    weights.mkdir(parents=True)
    with zipfile.ZipFile(weights / f"{MODEL_NAME}.zip", "w") as archive:
        archive.writestr(
            f"{MODEL_NAME}/{MODEL_NAME}.yaml",
            (offline_package / f"{MODEL_NAME}.yaml").read_bytes(),
        )
        # Windows-made archives keep backslashes in the entry names.
        archive.writestr(
            f"{MODEL_NAME}/checkpoints\\best.pdparams",
            (offline_package / "checkpoints" / "best.pdparams").read_bytes(),
        )

    def no_network(*args, **kwargs):
        raise AssertionError("network access attempted by an offline test")

    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(download_mod, "WEIGHTS_HOME", str(weights))
    monkeypatch.setattr(download_mod.requests, "get", no_network)

    model, _ = build_model_from_name(MODEL_NAME, strict_weights=True)

    assert type(model).__name__ == "LiFlow"
    # The Windows entry ``checkpoints\best.pdparams`` must have been expanded
    # into a real nested file that the package resolver can find.
    found = sorted(weights.rglob("best.pdparams"))
    assert len(found) == 1
    assert found[0].parent.name == "checkpoints"
    assert found[0].parent.parent.name == MODEL_NAME


@pytest.mark.skipif(
    not NETWORK_TESTS,
    reason="set LIFLOW_NETWORK_TESTS=1 to download the published package",
)
def test_published_package_downloads_and_strict_loads():
    model, config = build_model_from_name(MODEL_NAME, strict_weights=True)
    params = config["Model"]["__init_params__"]
    assert len(model.state_dict()) == 45
    assert params["num_elements"] == 77
    assert params["num_layers"] == 3


@pytest.mark.skipif(
    not os.environ.get("LIFLOW_PACKAGE_DIR"),
    reason="set LIFLOW_PACKAGE_DIR to an extracted package to test it offline",
)
def test_provided_package_dir_strict_loads(monkeypatch):
    package = Path(os.environ["LIFLOW_PACKAGE_DIR"])
    if not package.exists():
        pytest.fail(f"LIFLOW_PACKAGE_DIR does not exist: {package}")
    monkeypatch.setattr(
        models_mod.download, "get_weights_path_from_url", lambda _: str(package)
    )
    model, _ = build_model_from_name(MODEL_NAME, strict_weights=True)
    assert len(model.state_dict()) == 45

"""Small, path-independent SevenNet integration tests."""

import subprocess
import sys
from pathlib import Path

import paddle
import pytest
from pymatgen.core import Lattice, Structure

from ppmat.models.sevennet import SevenNet, SevenNetGraphConverter
from ppmat.models.sevennet.parse_input import init_model_config
from ppmat.utils.eager_comp_setting import setting_eager_mode


def _model_and_graph():
    setting_eager_mode(True, ["concat_grad"])
    config = init_model_config(
        {
            "chemical_species": "Hf, O",
            "channel": 4,
            "cutoff": 3.0,
            "num_convolution_layer": 3,
            "conv_denominator": 1.0,
        }
    )
    graph = SevenNetGraphConverter(cutoff=3.0)(
        Structure(Lattice.cubic(8.0), ["Hf", "O"], [[0, 0, 0], [0.1, 0, 0]])
    )
    return SevenNet(config), graph


def _check_forward_predict_and_backward():
    model, graph = _model_and_graph()
    data = {
        "graph": graph,
        "energy_per_atom": paddle.zeros([1]),
        "force": paddle.zeros([2, 3]),
        "stress": paddle.zeros([1, 3, 3]),
    }
    result = model(data)
    assert set(result["pred_dict"]) == {"energy_per_atom", "force", "stress"}
    assert bool(paddle.isfinite(result["loss_dict"]["loss"]).item())

    result["loss_dict"]["loss"].backward()
    assert any(parameter.grad is not None for parameter in model.parameters())

    prediction = model.predict(graph)
    assert prediction["force"].shape == (2, 3)
    assert prediction["stress"].shape == (6,)


def test_chemical_species_configuration_is_deterministic():
    config = init_model_config({"chemical_species": "O, Hf, O"})
    assert config["chemical_species"] == ["Hf", "O"]
    assert config["_type_map"] == {72: 0, 8: 1}



def test_forward_predict_and_backward():
    # Prim settings are process-wide. Run this check in its own interpreter.
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import runpy, sys; runpy.run_path(sys.argv[1])"
            "['_check_forward_predict_and_backward']()",
            str(Path(__file__).resolve()),
        ],
        check=True,
        timeout=90,
    )


def test_parallel_mode_reports_unsupported():
    with pytest.raises(NotImplementedError, match="serial model"):
        SevenNet({}, parallel=True)

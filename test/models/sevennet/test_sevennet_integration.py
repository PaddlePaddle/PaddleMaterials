"""Small, path-independent SevenNet integration tests."""

import os
import subprocess
import sys
from pathlib import Path

import paddle
import pytest
from pymatgen.core import Lattice, Structure

from ppmat.models import SevenNet, SevenNetGraphConverter
from ppmat.models.sevennet.config import init_model_config
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
        env={**os.environ, "FLAGS_new_einsum": "0"},
        check=True,
        timeout=90,
    )


def test_parallel_mode_reports_unsupported():
    with pytest.raises(NotImplementedError, match="serial model"):
        SevenNet({}, parallel=True)



def test_pretrained_predictions_match_upstream():
    asset_path = os.environ.get("SEVENNET_REVIEW_ASSETS")
    if not asset_path:
        pytest.skip("Set SEVENNET_REVIEW_ASSETS to the extracted review archives")

    import numpy as np
    from ase.io import read
    from pymatgen.io.ase import AseAtomsAdaptor

    from ppmat.predictor import PotentialPredictor

    assets = Path(asset_path)
    model_dir = assets / "sevennet0"
    reference_dir = assets / "sevennet-validation-data" / "sevennet0_reference"
    predictor = PotentialPredictor(
        config_path=str(model_dir / "sevennet0.yaml"),
        checkpoint_path=str(model_dir / "checkpoints/latest.pdparams"),
        device="gpu:0",
    )
    tolerances = {"energy": 1e-4, "force": 1e-5, "stress": 1e-6}
    maximum = dict.fromkeys(tolerances, 0.0)
    structures = sorted(reference_dir.glob("*.extxyz"))
    assert len(structures) == 8, f"Expected eight references in {reference_dir}"
    for path in structures:
        structure = AseAtomsAdaptor.get_structure(read(path))
        predictions = predictor.from_structures(structure)
        with np.load(path.with_name(path.stem + "_torch.npz")) as reference:
            for name, tolerance in tolerances.items():
                prediction = np.asarray(predictions[name])
                target = reference[name].reshape(prediction.shape)
                assert np.isfinite(prediction).all(), f"{path.name}: nonfinite {name}"
                np.testing.assert_allclose(
                    prediction, target, atol=tolerance, rtol=0,
                    err_msg=f"{path.name}: {name}",
                )
                maximum[name] = max(maximum[name], float(np.max(np.abs(prediction - target))))
    print(f"{len(structures)} structures. Maximum absolute errors: {maximum}")

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

"""Benchmark one registered weight with eager or CINN execution."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import time
from pathlib import Path

import numpy as np
import paddle

from ppmat.datasets.build_image import BuildImage
from ppmat.datasets.build_molecule import BuildMolecule
from ppmat.datasets.build_structure import BuildStructure
from ppmat.predictor import PotentialPredictor
from ppmat.predictor import PropertyPredictor
from ppmat.predictor import SpectrumPredictor
from ppmat.sampler import MolecularSampler
from ppmat.sampler import StructureSampler

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = Path(__file__).with_name("fixtures")
CIF = ROOT / "property_prediction/example_data/cifs/mp-18767-LiMnO2.cif"
XYZ = ROOT / "property_prediction/example_data/molecules/isoguvacine.xyz"
BF_IMAGE = ROOT / "spectrum_enhancement/example_data/sfin_bf.png"
HAADF_IMAGE = ROOT / "spectrum_enhancement/example_data/sfin_haadf.png"
DIFFNMR_DATA = ROOT / "spectrum_elucidation/example_data/sample.csv"
SFIN_DATASETS = {
    "sfin_bf_detect": "bf_data_test",
    "sfin_bf_enhance": "bf_data_test",
    "sfin_haadf_detect": "haadf_data_test",
    "sfin_haadf_enhance": "haadf_data_test",
}
MD17_XYZ = {
    "spherenet_md17_aspirin": FIXTURES / "aspirin_31064.xyz",
    "spherenet_md17_benzene_old": FIXTURES / "benzene_old_394337.xyz",
    "spherenet_md17_ethanol": FIXTURES / "ethanol_355833.xyz",
    "spherenet_md17_malonaldehyde": FIXTURES / "malonaldehyde_872511.xyz",
    "spherenet_md17_naphthalene": FIXTURES / "naphthalene_53608.xyz",
    "spherenet_md17_salicylic": FIXTURES / "salicylic_4044.xyz",
    "spherenet_md17_toluene": FIXTURES / "toluene_156935.xyz",
    "spherenet_md17_uracil": FIXTURES / "uracil_131618.xyz",
}


def build_graph(predictor, input_format: str, input_path: Path):
    if input_format == "cif":
        sample = BuildStructure(
            format="cif_file",
            primitive=False,
            niggli=False,
            canocial=False,
        )(str(input_path))
    else:
        sample = BuildMolecule(format="xyz_file", sanitize=False)(str(input_path))
    return predictor.graph_converter(sample)


# Force and stress models differentiate inside the compiled boundary, where SOT
# capture can split paddle.grad from its input; they need full-graph capture.
FULL_GRAPH_PREFIXES = ("mattersim_", "chgnet_", "spherenet_md17_")


def build_workflow(model_name: str, backend: str):
    overrides = [f"Execution.backend={backend}"]
    if backend == "cinn" and model_name.startswith(FULL_GRAPH_PREFIXES):
        overrides.append("Execution.__init_params__.full_graph=true")
    if model_name.startswith(("spherenet_qm9_", "dimenetpp_qm9_")):
        predictor = PropertyPredictor(
            model_name=model_name,
            device="gpu:0",
            config_overrides=overrides,
        )
        graph = build_graph(predictor, "xyz", XYZ)
        return predictor, lambda: predictor._run_model(graph), "model_predict", graph

    if model_name.startswith(("comformer_", "megnet_", "dimenetpp_")):
        predictor = PropertyPredictor(
            model_name=model_name,
            device="gpu:0",
            config_overrides=overrides,
        )
        graph = build_graph(predictor, "cif", CIF)
        return predictor, lambda: predictor._run_model(graph), "model_predict", graph

    if model_name.startswith(("mattersim_", "chgnet_")):
        predictor = PotentialPredictor(
            model_name=model_name,
            device="gpu:0",
            config_overrides=overrides,
        )
        graph = build_graph(predictor, "cif", CIF)
        return predictor, lambda: predictor._run_model(graph), "model_predict", graph

    if model_name.startswith("spherenet_md17_"):
        predictor = PotentialPredictor(
            model_name=model_name,
            device="gpu:0",
            config_overrides=overrides,
        )
        graph = build_graph(predictor, "xyz", MD17_XYZ[model_name])
        return predictor, lambda: predictor._run_model(graph), "model_predict", graph

    if model_name.startswith("sfin_"):
        predictor = SpectrumPredictor(
            model_name=model_name,
            device="gpu:0",
            config_overrides=overrides,
        )
        image, _ = sfin_test_image(model_name, predictor.model.target_name)
        if image is None:
            image_path = HAADF_IMAGE if "haadf" in model_name else BF_IMAGE
            image = BuildImage(format="image_file", mode="L", dtype="float32")(
                str(image_path)
            )
        return predictor, lambda: predictor.from_image(image), "model_predict", image

    if model_name == "diffnmr_msdnmr_nless15":
        sampler = MolecularSampler(
            model_name=model_name,
            weights_name="best.pdparams",
            config_overrides=[
                *overrides,
                f"Sampler.data.dataset.__init_params__.path={DIFFNMR_DATA}",
                "Sampler.visual_num=0",
                "Sampler.chains_to_save=0",
            ],
        )
        batch = next(iter(sampler._sample_loader))
        return sampler, lambda: sampler.sample(batch), "full_reverse_diffusion", batch

    if model_name.startswith(("diffcsp_", "mattergen_")):
        sampler = StructureSampler(
            model_name=model_name,
            config_overrides=overrides,
        )
        condition_names = getattr(sampler.model, "condition_names", None) or []
        sample_params = {"num_inference_steps": 2}
        if condition_names:
            conditions = {
                name: "O"
                if name == "chemical_system"
                else 1
                if name == "space_group"
                else 1.0
                for name in condition_names
            }

            def call():
                return sampler.sample_by_condition(
                    8,
                    conditions,
                    sample_params=sample_params,
                )

        elif getattr(sampler.model, "supports_num_atoms_sampling", True):

            def call():
                return sampler.sample_by_num_atoms(
                    8,
                    sample_params=sample_params,
                )

        else:

            def call():
                return sampler.sample_by_chemical_formula(
                    "O8",
                    sample_params=sample_params,
                )

        return sampler, call, "two_step_structure_sampling", None

    raise ValueError(f"Unsupported registered CINN benchmark model: {model_name}")


def runtime_keys(workflow) -> list[str]:
    model = getattr(workflow, "model", None)
    cache = getattr(model, "_runtime_cache", {})
    return [":".join(map(str, key)) for key in sorted(cache)]


def as_float(value) -> float:
    """Return a Python float for one scalar loss or metric value."""

    if isinstance(value, paddle.Tensor):
        return float(value.detach().cpu().numpy().reshape(-1)[0])
    return float(value)


def collect_loss(workflow, model_name: str, graph) -> dict:
    """Return the weight package's own loss values for one graph.

    The loss needs a ground-truth label. Example inputs are unlabeled, so this
    only reports a value when the graph itself carries the label.
    """

    model = getattr(workflow, "model", None)
    forward = getattr(model, "forward", None)
    if model_name.startswith("sfin_") or graph is None or not callable(forward):
        return {}

    label = getattr(graph, "y", None)
    if label is None:
        return {}

    property_names = getattr(model, "property_names", None)
    if isinstance(property_names, str):
        property_names = [property_names]
    if not property_names or len(property_names) != 1:
        return {}

    batch = {"graph": graph, property_names[0]: paddle.to_tensor(label)}
    try:
        with paddle.no_grad():
            output = forward(batch, return_loss=True, return_prediction=False)
    except Exception:
        return {}
    if not isinstance(output, dict):
        return {}

    losses = {}
    for key, value in (output.get("loss_dict") or {}).items():
        if value is None:
            continue
        try:
            losses[str(key)] = as_float(value)
        except (TypeError, ValueError, IndexError):
            continue
    return losses


def sfin_test_image(model_name: str, target_name: str):
    """Return one paired noisy input and target image from the SFIN test set."""

    dataset_name = SFIN_DATASETS.get(model_name)
    if dataset_name is None or target_name not in {"gt_enhance", "gt_detect"}:
        return None, None
    from ppmat.utils import download

    root = download.get_datasets_path_from_url(
        "https://paddle-org.bj.bcebos.com/paddlematerials/datasets/SFIN/"
        f"{dataset_name}.zip"
    )
    dataset_root = Path(root) / dataset_name
    test_root = (
        dataset_root / "test" if (dataset_root / "test").is_dir() else dataset_root
    )
    noisy_dir = test_root / "noisy"
    target_dir = test_root / target_name
    if not noisy_dir.is_dir() or not target_dir.is_dir():
        return None, None

    target_by_name = {path.name: path for path in target_dir.glob("*.png")}
    build = BuildImage(format="image_file", mode="L", dtype="float32")
    for noisy in sorted(noisy_dir.glob("*.png")):
        target = target_by_name.get(noisy.name)
        if target is not None:
            return build(str(noisy)), build(str(target))
    return None, None


def collect_sfin_loss(workflow, model_name: str, prediction) -> dict:
    """Return the SFIN reconstruction loss for one prediction."""

    model = getattr(workflow, "model", None)
    if model is None or not isinstance(prediction, dict):
        return {}
    target_name = getattr(model, "target_name", None)
    if target_name not in prediction:
        return {}

    _, reference = sfin_test_image(model_name, target_name)
    if reference is None:
        return {}
    try:
        pred = paddle.to_tensor(prediction[target_name])
        reference_tensor = paddle.to_tensor(reference)
        with paddle.no_grad():
            loss = model.criterion(pred, reference_tensor) * float(
                getattr(model, "loss_weight", 1.0)
            )
    except Exception:
        return {}
    return {"loss": as_float(loss)}


def collect_sfin_metrics(workflow, model_name: str, prediction) -> dict:
    """Return PSNR and SSIM for one SFIN prediction."""

    model = getattr(workflow, "model", None)
    if model is None or not isinstance(prediction, dict):
        return {}
    target_name = getattr(model, "target_name", None)
    if target_name not in prediction:
        return {}

    _, reference = sfin_test_image(model_name, target_name)
    if reference is None:
        return {}
    try:
        pred = paddle.to_tensor(prediction[target_name])
        reference_tensor = paddle.to_tensor(reference)
        if pred.ndim == 3:
            pred = pred.unsqueeze(0)
        if reference_tensor.ndim == 3:
            reference_tensor = reference_tensor.unsqueeze(0)
        from ppmat.metrics.sfin_metric import calc_psnr
        from ppmat.metrics.sfin_metric import calc_ssim

        psnr = as_float(calc_psnr(pred, reference_tensor))
        ssim = as_float(calc_ssim(pred, reference_tensor))
    except Exception:
        return {}
    return {"psnr": psnr, "ssim": ssim}


def git_revision(name: str) -> str | None:
    """Resolve a Git revision without making the benchmark depend on Git."""

    try:
        process = subprocess.run(
            ["git", "rev-parse", name],
            cwd=ROOT,
            capture_output=True,
            check=False,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return process.stdout.strip() if process.returncode == 0 else None


def to_jsonable(value):
    """Convert a prediction to JSON-compatible nested lists."""

    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, paddle.Tensor):
        return value.detach().cpu().numpy().tolist()
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(f"Cannot serialize prediction value of type {type(value).__name__}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--backend", choices=("eager", "cinn"), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be at least 1")
    return args


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "model": args.model_name,
        "backend": args.backend,
        "source": "registered_model_package",
        "source_revision": git_revision("HEAD"),
        "origin_develop_revision": git_revision("origin/develop"),
        "status": "failed",
    }
    try:
        # Predictors accept an explicit device, but both samplers use Paddle's
        # process-wide default. Set it before constructing any workflow so every
        # family measures the GPU path selected by CUDA_VISIBLE_DEVICES.
        paddle.set_device("gpu:0")
        np.random.seed(2026)
        paddle.seed(2026)
        workflow, call, scope, loss_input = build_workflow(
            args.model_name, args.backend
        )

        def seeded_call():
            np.random.seed(2026)
            paddle.seed(2026)
            return call()

        paddle.device.synchronize()
        start = time.perf_counter()
        first_output = seeded_call()
        paddle.device.synchronize()
        first_seconds = time.perf_counter() - start
        compiled_runtime_keys = runtime_keys(workflow)
        active_backend = getattr(
            getattr(workflow, "model", None), "execution_backend", None
        )
        if active_backend != args.backend:
            raise RuntimeError(
                f"requested backend {args.backend!r}, but the model reports "
                f"{active_backend!r}"
            )
        if args.backend == "cinn" and not compiled_runtime_keys:
            raise RuntimeError(
                "CINN was selected but the measured call created no compiled "
                "runtime boundary"
            )
        prediction = None
        prediction_error = None
        try:
            prediction = to_jsonable(first_output)
        except (TypeError, ValueError, RuntimeError) as error:
            prediction_error = str(error)
        extra_loss = {}
        extra_metrics = {}
        try:
            if args.model_name.startswith("sfin_"):
                extra_loss = collect_sfin_loss(
                    workflow,
                    args.model_name,
                    first_output,
                )
                extra_metrics = collect_sfin_metrics(
                    workflow,
                    args.model_name,
                    first_output,
                )
            else:
                extra_loss = collect_loss(
                    workflow,
                    args.model_name,
                    loss_input,
                )
        except Exception:
            extra_loss = {}
            extra_metrics = {}
        warm_seconds = []
        for _ in range(args.repeats):
            start = time.perf_counter()
            seeded_call()
            paddle.device.synchronize()
            warm_seconds.append(time.perf_counter() - start)
        result.update(
            {
                "status": "passed",
                "scope": scope,
                "device": paddle.device.get_device(),
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "paddle_version": paddle.__version__,
                "compiled_with_cinn": bool(paddle.base.is_compiled_with_cinn()),
                "first_seconds": first_seconds,
                "warm_seconds": warm_seconds,
                "warm_median_seconds": statistics.median(warm_seconds),
                "runtime_keys": compiled_runtime_keys,
                "prediction": prediction,
                "prediction_error": prediction_error,
                "loss": extra_loss,
                "metrics": extra_metrics,
            }
        )
    except Exception as error:
        result.update(
            {
                "error_type": type(error).__name__,
                "error": str(error),
            }
        )
        output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result, indent=2))
        raise

    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

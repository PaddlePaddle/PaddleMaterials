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
import traceback
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


def build_workflow(model_name: str, backend: str):
    overrides = [f"Execution.backend={backend}"]
    if model_name.startswith(("comformer_", "megnet_", "dimenetpp_")):
        predictor = PropertyPredictor(
            model_name=model_name,
            device="gpu:0",
            config_overrides=overrides,
        )
        graph = build_graph(predictor, "cif", CIF)
        return predictor, lambda: predictor._run_model(graph), "model_predict"

    if model_name.startswith("spherenet_qm9_"):
        predictor = PropertyPredictor(
            model_name=model_name,
            device="gpu:0",
            config_overrides=overrides,
        )
        graph = build_graph(predictor, "xyz", XYZ)
        return predictor, lambda: predictor._run_model(graph), "model_predict"

    if model_name.startswith(("mattersim_", "chgnet_")):
        predictor = PotentialPredictor(
            model_name=model_name,
            device="gpu:0",
            config_overrides=overrides,
        )
        graph = build_graph(predictor, "cif", CIF)
        return predictor, lambda: predictor._run_model(graph), "model_predict"

    if model_name.startswith("spherenet_md17_"):
        predictor = PotentialPredictor(
            model_name=model_name,
            device="gpu:0",
            config_overrides=overrides,
        )
        graph = build_graph(predictor, "xyz", MD17_XYZ[model_name])
        return predictor, lambda: predictor._run_model(graph), "model_predict"

    if model_name.startswith("sfin_"):
        predictor = SpectrumPredictor(
            model_name=model_name,
            device="gpu:0",
            config_overrides=overrides,
        )
        image_path = HAADF_IMAGE if "haadf" in model_name else BF_IMAGE
        image = BuildImage(format="image_file", mode="L", dtype="float32")(
            str(image_path)
        )
        return predictor, lambda: predictor.from_image(image), "model_predict"

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
        return sampler, lambda: sampler.sample(batch), "full_reverse_diffusion"

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

        elif model_name.startswith("mattergen_"):

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

        return sampler, call, "two_step_structure_sampling"

    raise ValueError(f"Unsupported registered CINN benchmark model: {model_name}")


def runtime_keys(workflow) -> list[str]:
    model = getattr(workflow, "model", None)
    cache = getattr(model, "_runtime_cache", {})
    return [":".join(map(str, key)) for key in sorted(cache)]


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--backend", choices=("eager", "cinn"), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--repeats", type=int, default=10)
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
        workflow, call, scope = build_workflow(args.model_name, args.backend)

        def seeded_call():
            np.random.seed(2026)
            paddle.seed(2026)
            return call()

        paddle.device.synchronize()
        start = time.perf_counter()
        seeded_call()
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
            }
        )
    except Exception as error:
        result.update(
            {
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
            }
        )
        output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result, indent=2))
        raise

    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

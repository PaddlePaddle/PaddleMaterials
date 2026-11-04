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

import argparse
import json
import os
import os.path as osp
from datetime import datetime

import numpy as np
from omegaconf import OmegaConf


def _write_xyz(path: str, trajectory: np.ndarray, atomic_numbers: np.ndarray):
    from ase import Atoms
    from ase.io import write as ase_write

    frames = [
        Atoms(numbers=atomic_numbers, positions=positions, pbc=False)
        for positions in trajectory
    ]
    ase_write(path, frames, format="xyz")


def _build_registered(model_name: str):
    from ppmat.models import build_model_from_name

    model, _ = build_model_from_name(model_name, strict_weights=True)
    return model


def _build_side(predict_cfg: dict, prefix: str):
    """Build one model side from the registry or return ``None``.

    ``<prefix>_model`` names a registered weight package (auto-downloaded and
    strict-loaded).  ``<prefix>_config`` + ``<prefix>_checkpoint`` point at a
    local training output and are forwarded to ``IntegratorPredictor``.  Exactly
    one form may be configured; ``None`` for both means the side is disabled
    (used for the corrector, whose checkpoint is not published yet).
    """
    model_name = predict_cfg.get(f"{prefix}_model", None)
    config_path = predict_cfg.get(f"{prefix}_config", None)
    checkpoint_path = predict_cfg.get(f"{prefix}_checkpoint", None)
    has_local = config_path is not None or checkpoint_path is not None
    if model_name is not None and has_local:
        raise ValueError(
            f"Predict.{prefix}_model and Predict.{prefix}_config/"
            f"{prefix}_checkpoint are mutually exclusive."
        )
    if model_name is not None:
        return _build_registered(model_name)
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", type=str, required=True)
    parser.add_argument("--input", "-i", type=str, required=True)
    parser.add_argument(
        "--output_dir", "-o", type=str, default="output/liflow_prediction"
    )
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--flow_steps", type=int, default=None)
    parser.add_argument("--solver", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    config = OmegaConf.to_container(config, resolve=True)
    predict_cfg = config.get("Predict", {}) or {}
    overrides = {
        "temperature": args.temperature,
        "steps": args.steps,
        "flow_steps": args.flow_steps,
        "solver": args.solver,
        "seed": args.seed,
    }
    for key, value in overrides.items():
        if value is None:
            overrides[key] = predict_cfg.get(key, default_by_key(key))
    temperature = float(overrides["temperature"])

    data = np.load(args.input)
    positions = np.asarray(data["positions"], dtype=np.float32)
    atomic_numbers = np.asarray(data["atomic_numbers"], dtype=np.int64)
    lattice = np.asarray(data["lattice"], dtype=np.float32)

    # ``element_index`` maps atomic numbers to the contiguous embedding indices
    # used during training.  It may live in the input archive, in the configured
    # dataset directory, or be passed next to the input file.
    element_index = data["element_index"] if "element_index" in data else None
    if element_index is None:
        element_index = predict_cfg.get("element_index", None)
        if element_index is None:
            candidate = osp.join(
                osp.dirname(osp.abspath(args.input)), "element_index.npy"
            )
            if osp.exists(candidate):
                element_index = candidate
    if element_index is None:
        raise ValueError(
            "An `element_index` array is required to map atomic numbers to "
            "element embeddings.  Add it to the input .npz or set "
            "`Predict.element_index` in the config."
        )

    prior_Li = (
        int(data["prior_Li"])
        if "prior_Li" in data
        else int(predict_cfg.get("prior_Li", 0))
    )
    prior_frame = (
        int(data["prior_frame"])
        if "prior_frame" in data
        else int(predict_cfg.get("prior_frame", 0))
    )

    from ppmat.predictor import IntegratorPredictor

    propagator_model = _build_side(predict_cfg, "propagator")
    corrector_model = _build_side(predict_cfg, "corrector")
    propagator_config = predict_cfg.get("propagator_config", None)
    propagator_checkpoint = predict_cfg.get("propagator_checkpoint", None)
    if propagator_model is None and (
        propagator_config is None or propagator_checkpoint is None
    ):
        raise ValueError(
            "Set either Predict.propagator_model (registered weights) or both "
            "Predict.propagator_config and Predict.propagator_checkpoint."
        )

    predictor = IntegratorPredictor(
        propagator_model=propagator_model,
        corrector_model=corrector_model,
        propagator_config=propagator_config,
        propagator_checkpoint=propagator_checkpoint,
        corrector_config=predict_cfg.get("corrector_config", None),
        corrector_checkpoint=predict_cfg.get("corrector_checkpoint", None),
        cutoff=predict_cfg.get("cutoff", 5.0),
        periodic=predict_cfg.get("periodic", True),
        element_index=element_index,
        scale_Li_index=prior_Li,
        scale_frame_index=prior_frame,
        execution_backend=config.get("Execution", {}).get("backend", None),
    )

    trajectory = predictor.run(
        positions,
        atomic_numbers,
        lattice,
        temperature,
        steps=int(overrides["steps"]),
        flow_steps=int(overrides["flow_steps"]),
        solver=overrides["solver"],
        corrector_every=int(predict_cfg.get("corrector_every", 1)),
        fix_com=bool(predict_cfg.get("fix_com", True)),
        num_retries=int(predict_cfg.get("num_retries", 2)),
        seed=int(overrides["seed"]),
    )

    os.makedirs(args.output_dir, exist_ok=True)
    np.save(osp.join(args.output_dir, "trajectory.npy"), trajectory)
    _write_xyz(osp.join(args.output_dir, "trajectory.xyz"), trajectory, atomic_numbers)
    metadata = {
        "input": args.input,
        "temperature": temperature,
        "steps": int(overrides["steps"]),
        "flow_steps": int(overrides["flow_steps"]),
        "solver": overrides["solver"],
        "seed": int(overrides["seed"]),
        "trajectory_shape": list(trajectory.shape),
        "generated_at_utc": datetime.utcnow().isoformat() + "Z",
    }
    with open(osp.join(args.output_dir, "run_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved trajectory to {osp.join(args.output_dir, 'trajectory.npy')}")


def default_by_key(key: str):
    return {
        "temperature": 800.0,
        "steps": 25,
        "flow_steps": 10,
        "solver": "euler",
        "seed": 42,
    }[key]


if __name__ == "__main__":
    main()

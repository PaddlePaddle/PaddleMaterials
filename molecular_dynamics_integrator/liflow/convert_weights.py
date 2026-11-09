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

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any
from typing import Dict
from typing import Optional
from typing import Tuple

import numpy as np

try:
    from omegaconf import OmegaConf
except Exception:  # noqa: BLE001
    OmegaConf = None  # type: ignore

MODULE_PREFIXES = ("model.", "network.")

DUAL_PAINN_CFG_KEYS = (
    "num_features",
    "num_radial_basis",
    "num_layers",
    "num_elements",
    "r_max",
    "r_offset",
    "ref_temp",
)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_npz(data: Dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for key in sorted(data.keys()):
        digest.update(key.encode("utf-8"))
        digest.update(np.ascontiguousarray(data[key], dtype=np.float32).tobytes())
    return digest.hexdigest()


def _extract_model_cfg(ckpt: Dict[str, Any]) -> Dict[str, Any]:
    hp = ckpt.get("hyper_parameters")
    model_cfg = None
    if isinstance(hp, dict) and "cfg" in hp:
        raw = hp.get("cfg")
        getter = raw.get if hasattr(raw, "get") else None
        if getter is not None:
            model_cfg = getter("model")
    if model_cfg is not None and OmegaConf is not None:
        try:
            container = OmegaConf.to_container(model_cfg, resolve=True)
        except Exception:  # noqa: BLE001
            container = None
        if isinstance(container, dict):
            keep = {k: container[k] for k in DUAL_PAINN_CFG_KEYS if k in container}
            if len(keep) == len(DUAL_PAINN_CFG_KEYS):
                return keep
    return {
        "num_features": 64,
        "num_radial_basis": 20,
        "num_layers": 3,
        "num_elements": 77,
        "r_max": 5.0,
        "r_offset": 0.5,
        "ref_temp": 1000.0,
    }


def dump_torch_state(
    ckpt_path: str,
    state_npz_path: str,
    model_cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    import torch

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict) or "state_dict" not in ckpt:
        raise ValueError(
            f"{ckpt_path} is not a Lightning checkpoint with a 'state_dict' key"
        )
    raw = ckpt["state_dict"]

    arrays: Dict[str, np.ndarray] = {}
    for key, value in raw.items():
        try:
            arrays[key] = value.detach().cpu().numpy().astype(np.float32)
        except Exception as exc:  # noqa: BLE001
            raise TypeError(f"Unconvertible tensor for key {key}: {exc}") from exc

    if not model_cfg:
        model_cfg = _extract_model_cfg(ckpt)
    for k, v in model_cfg.items():
        arrays[f"_model_cfg/{k}"] = np.asarray(v)

    meta: Dict[str, Any] = {
        "source": os.path.abspath(ckpt_path),
        "source_sha256": _sha256_bytes(open(ckpt_path, "rb").read()),
        "state_dict_key_count": len(raw),
        "model_cfg": model_cfg,
    }
    if isinstance(ckpt.get("hyper_parameters"), dict):
        meta["has_hyper_parameters"] = True

    os.makedirs(os.path.dirname(os.path.abspath(state_npz_path)), exist_ok=True)
    np.savez_compressed(state_npz_path, **arrays)
    return meta


def load_state(npz_path: str) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    data = np.load(npz_path)
    arrays: Dict[str, np.ndarray] = {}
    model_cfg: Dict[str, Any] = {}
    for key in data.files:
        obj = data[key]
        if key.startswith("_model_cfg/"):
            val = obj
            if isinstance(val, np.ndarray):
                val = val[()] if val.shape == () else np.asarray(val).tolist()
            model_cfg[key[len("_model_cfg/") :]] = (
                int(val)
                if isinstance(val, (int, float)) and float(val).is_integer()
                else val
            )
        elif isinstance(obj, np.ndarray) and obj.dtype != np.object_:
            arrays[key] = obj
    return arrays, model_cfg


def _strip_prefix(name: str) -> str:
    for prefix in MODULE_PREFIXES:
        if name.startswith(prefix):
            return name[len(prefix) :]
    return name


def _json_safe(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def convert_state(
    arrays: Dict[str, np.ndarray],
    target_state_dict: Dict[str, np.ndarray],
) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    missing, unexpected, shape_mismatch = [], [], []

    stripped_target = {_strip_prefix(name) for name in target_state_dict}

    src_by_stripped: Dict[str, np.ndarray] = {}
    for key, value in arrays.items():
        src_by_stripped.setdefault(_strip_prefix(key), np.asarray(value))

    for name, expected in target_state_dict.items():
        core = _strip_prefix(name)
        src = src_by_stripped.get(core)
        if src is None:
            missing.append(name)
            continue
        expected = np.asarray(expected)

        should_transpose = (
            src.ndim == 2
            and core.endswith(".weight")
            and not core.startswith("atom_embedding")
        )
        if should_transpose and tuple(src.T.shape) == tuple(expected.shape):
            out[name] = {"array": np.ascontiguousarray(src.T), "transposed": True}
        elif tuple(src.shape) == tuple(expected.shape):
            out[name] = {"array": np.asarray(src).copy(), "transposed": False}
        else:
            shape_mismatch.append(
                f"{name}: src {tuple(src.shape)} vs target {tuple(expected.shape)}"
            )

    for key in arrays:
        stripped = _strip_prefix(key)
        if stripped not in stripped_target:
            unexpected.append(stripped)

    if missing or unexpected or shape_mismatch:
        details = []
        if missing:
            details.append(f"missing keys: {missing}")
        if unexpected:
            details.append(f"unexpected keys: {unexpected}")
        if shape_mismatch:
            details.append(f"shape mismatches: {shape_mismatch}")
        raise ValueError("Checkpoint conversion failed: " + "; ".join(details))

    return out


def _to_tensor(array: np.ndarray):
    import paddle

    return paddle.to_tensor(array, dtype=paddle.float32)


def convert_checkpoint(
    arrays: Dict[str, np.ndarray],
    model,
    output: str,
    audit: Optional[str] = None,
    model_cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    target = {name: np.asarray(v) for name, v in model.state_dict().items()}
    converted = convert_state(arrays, target)

    converted_arrays = {name: info["array"] for name, info in converted.items()}
    key_set_equal = set(converted_arrays) == set(target)
    shape_ok = all(converted_arrays[k].shape == target[k].shape for k in target)
    dtype_ok = all(converted_arrays[k].dtype == target[k].dtype for k in target)
    num_parameters_target = sum(int(np.asarray(v).size) for v in target.values())
    num_parameters_converted = sum(
        int(np.asarray(v).size) for v in converted_arrays.values()
    )

    if not (key_set_equal and shape_ok and dtype_ok):
        raise ValueError(
            f"Conversion audit failed (key_set_equal={key_set_equal}, "
            f"shape_ok={shape_ok}, dtype_ok={dtype_ok})"
        )

    paddle_state = {name: _to_tensor(arr) for name, arr in converted_arrays.items()}

    audit_dict: Dict[str, Any] = {
        "output": os.path.abspath(output),
        "num_parameters": num_parameters_converted,
        "num_buffers": int(sum(1 for k in target if target[k].ndim <= 1)),
        "num_transposed": int(sum(1 for v in converted.values() if v["transposed"])),
        "transposed_keys": [k for k, v in converted.items() if v["transposed"]],
        "mapping": {name: name for name in target},
        "key_set_equal": bool(key_set_equal),
        "shape_ok": bool(shape_ok),
        "dtype_ok": bool(dtype_ok),
        "num_parameters_target": num_parameters_target,
        "checksum": _sha256_npz(converted_arrays),
        "model_cfg": model_cfg or {},
    }

    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    import paddle

    paddle.save(paddle_state, output)

    if audit:
        audit_path = Path(audit)
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        with open(audit_path, "w") as f:
            json.dump(_json_safe(audit_dict), f, indent=2)
    return audit_dict


def _build_dual_painn(model_cfg: Dict[str, Any]):
    from ppmat.models.liflow.liflow import LiFlow

    cfg = {k: model_cfg[k] for k in DUAL_PAINN_CFG_KEYS}
    return LiFlow(**cfg)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert original LiFlow torch checkpoints to strict-loadable "
        "Paddle weights."
    )
    parser.add_argument("--input", help="Original Lightning .ckpt path.")
    parser.add_argument(
        "--state-npz",
        help="Source state_dict.npz (used instead of --input when torch is absent).",
    )
    parser.add_argument("--output", help="Destination .pdparams file.")
    parser.add_argument("--audit", help="Optional audit JSON path.")
    parser.add_argument(
        "--dump-state",
        action="store_true",
        help="Only dump the torch state_dict into --state-npz (requires torch).",
    )
    parser.add_argument(
        "--model-cfg", help="Optional model hyper-parameters as a JSON string."
    )
    args = parser.parse_args()

    model_cfg = json.loads(args.model_cfg) if args.model_cfg else {}

    if args.state_npz is None and args.input is None:
        parser.error("Provide --input (torch ckpt) or --state-npz.")

    if args.dump_state:
        if not args.input:
            parser.error("--dump-state requires --input.")
        meta = dump_torch_state(args.input, args.state_npz, model_cfg=model_cfg)
        print(json.dumps(meta, indent=2))
        return

    if not args.output:
        parser.error("--output is required when converting.")

    if args.state_npz:
        arrays, recovered_cfg = load_state(args.state_npz)
        if not model_cfg:
            model_cfg = recovered_cfg
    else:
        tmp = os.path.join(
            os.path.dirname(os.path.abspath(args.output)), "._dump_state.npz"
        )
        dump_torch_state(args.input, tmp, model_cfg=model_cfg)
        arrays, recovered_cfg = load_state(tmp)
        if not model_cfg:
            model_cfg = recovered_cfg

    if not model_cfg:
        model_cfg = {
            "num_features": 64,
            "num_radial_basis": 20,
            "num_layers": 3,
            "num_elements": 77,
            "r_max": 5.0,
            "r_offset": 0.5,
            "ref_temp": 1000.0,
        }

    model = _build_dual_painn(model_cfg)
    audit_result = convert_checkpoint(
        arrays, model, args.output, audit=args.audit, model_cfg=model_cfg
    )

    n = audit_result["num_parameters"]
    print(f"Converted {n} parameters -> {args.output}")
    if args.audit:
        print(f"Audit written to {args.audit}")


if __name__ == "__main__":
    main()

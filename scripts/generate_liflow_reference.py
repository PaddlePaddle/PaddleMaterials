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
import json
import os
import platform

import numpy as np
import torch

REFERENCE_COMMIT = "e6fc475361d046865f12cae1aee11c4f56c48d87"
SEED = 20260910
DTYPE = "float32"


def _load_layers(reference_root: str):
    import sys

    sys.path.insert(0, reference_root)
    import liflow.model.layers as L  # noqa: E402

    return L


def _state(layer):
    return {name: value.cpu().numpy() for name, value in layer.state_dict().items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="test/fixtures/liflow/reference_layers.npz")
    parser.add_argument(
        "--manifest", default="test/fixtures/liflow/reference_manifest.json"
    )
    parser.add_argument(
        "--reference-root",
        default=os.environ.get("LIFLOW_REFERENCE_ROOT", ""),
    )
    args = parser.parse_args()

    root = args.reference_root
    if not root:
        raise SystemExit(
            "Provide --reference-root (or set LIFLOW_REFERENCE_ROOT) of the liflow "
            f"reference checkout, pinned at {REFERENCE_COMMIT}."
        )

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    L = _load_layers(root)

    save = {}
    save["_seed"] = np.array(SEED)
    save["_dtype"] = np.array(DTYPE)
    save["_reference_commit"] = np.array(REFERENCE_COMMIT)

    gfb = L.GaussianFourierBasis(num_basis=8)
    gfb_x = torch.randn(6, 1, dtype=torch.float32)
    with torch.no_grad():
        gfb_out = gfb(gfb_x)
    save["gfb/num_basis"] = np.array(8)
    save["gfb/x"] = gfb_x.numpy()
    save["gfb/output"] = gfb_out.numpy()
    for name, arr in _state(gfb).items():
        save[f"gfb/state/{name}"] = arr

    bessel = L.BesselBasis(num_basis=4, r_max=5.0)
    bessel_x = torch.tensor([[0.5], [2.0]], dtype=torch.float32)
    with torch.no_grad():
        bessel_out = bessel(bessel_x)
    save["bessel/num_basis"] = np.array(4)
    save["bessel/r_max"] = np.array(5.0, dtype=np.float32)
    save["bessel/x"] = bessel_x.numpy()
    save["bessel/output"] = bessel_out.numpy()
    for name, arr in _state(bessel).items():
        save[f"bessel/state/{name}"] = arr

    cutoff = L.CosineCutoff(r_max=5.0)
    cutoff_x = torch.tensor([[1.0], [3.0], [6.0]], dtype=torch.float32)
    with torch.no_grad():
        cutoff_out = cutoff(cutoff_x)
    save["cutoff/r_max"] = np.array(5.0, dtype=np.float32)
    save["cutoff/x"] = cutoff_x.numpy()
    save["cutoff/output"] = cutoff_out.numpy()
    for name, arr in _state(cutoff).items():
        save[f"cutoff/state/{name}"] = arr

    f, r, n_nodes, n_edges = 8, 4, 3, 4
    dmb = L.DualMessageBlock(num_features=f, num_radial_basis=r)
    dmb.data = {
        "s": torch.randn(n_nodes, 1, f, dtype=torch.float32),
        "v": torch.randn(n_nodes, 3, f, dtype=torch.float32),
        "radial": torch.rand(n_edges, 1, r, dtype=torch.float32),
        "f_cut": torch.rand(n_edges, 1, dtype=torch.float32),
        "units": torch.nn.functional.normalize(
            torch.randn(n_edges, 3, dtype=torch.float32), dim=-1
        ),
        "edge_index": torch.tensor([[0, 1, 2, 0], [1, 2, 0, 2]], dtype=torch.long),
    }
    with torch.no_grad():
        dmb_out_s, dmb_out_v = dmb(
            dmb.data["s"],
            dmb.data["v"],
            dmb.data["radial"],
            dmb.data["radial"],
            dmb.data["f_cut"],
            dmb.data["f_cut"],
            dmb.data["units"],
            dmb.data["units"],
            dmb.data["edge_index"],
        )
    save["dmb/num_features"] = np.array(f)
    save["dmb/num_radial_basis"] = np.array(r)
    for key, val in dmb.data.items():
        save[f"dmb/{key}"] = val.numpy()
    save["dmb/output_s"] = dmb_out_s.numpy()
    save["dmb/output_v"] = dmb_out_v.numpy()
    for name, arr in _state(dmb).items():
        save[f"dmb/state/{name}"] = arr

    ub = L.UpdateBlock(num_features=f)
    ub_s = torch.randn(n_nodes, 1, f, dtype=torch.float32)
    ub_v = torch.randn(n_nodes, 3, f, dtype=torch.float32)
    with torch.no_grad():
        ub_out_s, ub_out_v = ub(ub_s, ub_v)
    save["ub/num_features"] = np.array(f)
    save["ub/s"] = ub_s.numpy()
    save["ub/v"] = ub_v.numpy()
    save["ub/output_s"] = ub_out_s.numpy()
    save["ub/output_v"] = ub_out_v.numpy()
    for name, arr in _state(ub).items():
        save[f"ub/state/{name}"] = arr

    geb = L.GatedEquivariantBlock(num_scalar_inputs=f, num_vector_inputs=4)
    eb_s = torch.randn(n_nodes, 1, f, dtype=torch.float32)
    eb_v = torch.randn(n_nodes, 3, 4, dtype=torch.float32)
    with torch.no_grad():
        eb_out = geb(eb_s, eb_v)
    save["geb/num_scalar_inputs"] = np.array(f)
    save["geb/num_vector_inputs"] = np.array(4)
    save["geb/s"] = eb_s.numpy()
    save["geb/v"] = eb_v.numpy()
    save["geb/output"] = eb_out.numpy()
    for name, arr in _state(geb).items():
        save[f"geb/state/{name}"] = arr

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    np.savez_compressed(args.output, **save)

    import torch_scatter

    manifest = {
        "reference": {
            "repository": "learningmatter-mit/liflow",
            "commit": REFERENCE_COMMIT,
            "url": f"https://github.com/learningmatter-mit/liflow/tree/{REFERENCE_COMMIT}",
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_scatter": torch_scatter.__version__,
            "numpy": np.__version__,
            "dtype": DTYPE,
            "seed": SEED,
        },
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.manifest)), exist_ok=True)
    with open(args.manifest, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    print(f"wrote {args.output}")
    print(f"wrote {args.manifest}")
    print(
        f"env: torch={manifest['environment']['torch']} "
        f"torch_scatter={manifest['environment']['torch_scatter']} "
        f"numpy={manifest['environment']['numpy']} "
        f"python={manifest['environment']['python']}"
    )


if __name__ == "__main__":
    main()

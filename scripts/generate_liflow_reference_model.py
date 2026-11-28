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
from torch_geometric.data import Data

REFERENCE_COMMIT = "e6fc475361d046865f12cae1aee11c4f56c48d87"
SEED = 20260910
DTYPE = "float32"

MODEL_CONFIG = dict(
    num_features=8,
    num_radial_basis=4,
    num_layers=2,
    num_elements=8,
    r_max=5.0,
    r_offset=0.0,
    ref_temp=1000.0,
)
N_ATOMS = 4


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="test/fixtures/liflow/reference_model.npz")
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

    import sys

    sys.path.insert(0, root)
    from liflow.model.models import DualPaiNN
    from liflow.utils.geometry import get_neighbor_list_batch  # noqa: E402

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    model = DualPaiNN(**MODEL_CONFIG)
    model.eval()

    lattice = np.eye(3, dtype=np.float64) * 8.0
    condition = np.random.uniform(2.0, 6.0, size=(N_ATOMS, 3)).astype(np.float64)
    flow = condition + np.random.uniform(-0.5, 0.5, size=(N_ATOMS, 3)).astype(
        np.float64
    )

    edge_index, shifts = get_neighbor_list_batch(
        [condition], lattice, cutoff=MODEL_CONFIG["r_max"], periodic=True
    )
    shift_cart = shifts.astype(np.float64)

    elements = np.random.randint(0, MODEL_CONFIG["num_elements"], size=(N_ATOMS,))
    node_time = np.random.uniform(0.0, 1.0, size=(N_ATOMS,)).astype(DTYPE)
    node_temp = np.random.uniform(600.0, 1200.0, size=(N_ATOMS,)).astype(DTYPE)

    data = Data(
        positions_1=torch.tensor(condition, dtype=torch.float32, requires_grad=True),
        positions_2=torch.tensor(flow, dtype=torch.float32, requires_grad=True),
        edge_index=torch.tensor(edge_index.astype(np.int64), dtype=torch.long),
        shifts=torch.tensor(shift_cart, dtype=torch.float32),
        elements=torch.tensor(elements, dtype=torch.long),
        time=torch.tensor(node_time, dtype=torch.float32),
        temp=torch.tensor(node_temp, dtype=torch.float32),
    )

    assert data.edge_index.shape[1] > 0, "neighbor list must contain edges"

    output = model(data)

    loss = torch.sum(output**2)
    input_grads = torch.autograd.grad(
        loss, (data.positions_1, data.positions_2), retain_graph=False
    )
    grad_condition = input_grads[0].detach().numpy()
    grad_flow = input_grads[1].detach().numpy()

    save = {
        "_seed": np.array(SEED),
        "_dtype": np.array(DTYPE),
        "_reference_commit": np.array(REFERENCE_COMMIT),
    }

    save["nb/positions_batch"] = condition[None].astype(DTYPE)
    save["nb/lattice"] = lattice.astype(DTYPE)
    save["nb/cutoff"] = np.array(MODEL_CONFIG["r_max"], dtype=DTYPE)
    save["nb/edge_index"] = edge_index.astype(np.int64)
    save["nb/shifts"] = shift_cart.astype(DTYPE)

    save["model/condition_positions"] = data.positions_1.detach().numpy()
    save["model/flow_positions"] = data.positions_2.detach().numpy()
    save["model/edge_index"] = data.edge_index.numpy()
    save["model/shifts"] = data.shifts.numpy()
    save["model/elements"] = data.elements.numpy()
    save["model/node_time"] = data.time.numpy()
    save["model/node_temperature"] = data.temp.numpy()
    save["model/output"] = output.detach().numpy()
    save["model/grad/condition_positions"] = grad_condition
    save["model/grad/flow_positions"] = grad_flow

    for key, val in MODEL_CONFIG.items():
        save[f"model/config/{key}"] = np.array(
            val, dtype=DTYPE if isinstance(val, float) else np.int64
        )

    for name, value in model.state_dict().items():
        save[f"model/state/{name}"] = value.cpu().numpy()

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    np.savez_compressed(args.output, **save)

    import torch_scatter

    print(f"wrote {args.output}")
    print(f"output[0]={output.detach().numpy()[0]}  edges={data.edge_index.shape[1]}")
    manifest = {
        "commit": REFERENCE_COMMIT,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_scatter": torch_scatter.__version__,
            "numpy": np.__version__,
            "dtype": DTYPE,
            "seed": SEED,
        },
    }
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

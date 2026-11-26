# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CINN benchmark workload for the MatInvent RL weights.

The public CINN benchmark only dispatches by model family; the input contract,
optimizer and compiled boundary a family needs stay in its own package.
"""

from types import SimpleNamespace

import numpy as np
import paddle

from ppmat.models import build_model_from_name
from ppmat.utils.execution import configure_execution_backend

TRAIN_SCOPE = "matinvent_train_step"


def build_train_workload(model_name: str, backend: str):
    """Return one MatInvent RL fine-tuning step as the timed workload.

    MatInvent fine-tunes a diffusion backbone, so the timed workload is one
    train-mode denoise step (forward + backward + optimizer update) rather than
    the inference sampling used by the other generators.
    """

    model, _ = build_model_from_name(model_name)
    configure_execution_backend(model, backend, owner="Benchmark")
    batch = {
        "structure_array": {
            "num_atoms": paddle.to_tensor([3, 2], dtype="int64"),
            "atom_types": paddle.to_tensor([6, 8, 8, 14, 14], dtype="int64"),
            "frac_coords": paddle.to_tensor(
                [
                    [0.1, 0.2, 0.3],
                    [0.4, 0.5, 0.6],
                    [0.7, 0.8, 0.9],
                    [0.2, 0.3, 0.4],
                    [0.5, 0.6, 0.7],
                ]
            ),
            "lattice": paddle.to_tensor(
                np.stack([np.eye(3) * 5.0, np.eye(3) * 4.0]).astype("float32")
            ),
        }
    }
    optimizer = paddle.optimizer.Adam(parameters=model.parameters(), learning_rate=0.0)

    def train_step():
        model.train()
        loss = model(batch)["loss_dict"]["loss"]
        loss.backward()
        optimizer.step()
        optimizer.clear_grad()
        return float(loss)

    return (
        SimpleNamespace(model=model),
        train_step,
        TRAIN_SCOPE,
    )

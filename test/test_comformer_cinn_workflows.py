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

import numpy as np
import paddle
import pgl

from ppmat.models.comformer.comformer import iComformer


def _make_graph():
    edges = np.asarray(
        [
            [source, target]
            for source in range(3)
            for target in range(3)
            if source != target
        ],
        dtype=np.int64,
    )
    node_feat = np.zeros([3, 92], dtype=np.float32)
    node_feat[np.arange(3), np.arange(3) + 1] = 1.0
    return pgl.Graph(
        edges,
        num_nodes=3,
        node_feat={"node_feat": node_feat},
        edge_feat={
            "r": np.ones([len(edges), 3], dtype=np.float32),
            "nei": np.tile(np.eye(3, dtype=np.float32), [len(edges), 1, 1]),
        },
    )


def _make_model(execution_backend):
    with paddle.utils.unique_name.guard():
        paddle.seed(2026)
        return iComformer(
            conv_layers=1,
            edge_layers=1,
            atom_input_features=92,
            edge_features=8,
            triplet_input_features=6,
            node_features=8,
            fc_features=8,
            output_features=1,
            execution_backend=execution_backend,
        )


def test_comformer_cinn_uses_the_public_forward(monkeypatch):
    eager_model = _make_model("eager")
    cinn_model = _make_model("cinn")
    cinn_model.set_state_dict(eager_model.state_dict())
    monkeypatch.setattr(
        iComformer,
        "_run_runtime",
        lambda self, name, function, *args, **kwargs: function(*args),
    )

    graph = _make_graph()
    expected = eager_model._forward({"graph": graph})
    actual = cinn_model._forward({"graph": graph})

    np.testing.assert_allclose(actual.numpy(), expected.numpy(), atol=2e-6, rtol=2e-6)
    assert cinn_model.state_dict().keys() == eager_model.state_dict().keys()

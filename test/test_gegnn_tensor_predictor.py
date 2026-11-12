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

import copy
import os
import unittest

import numpy as np
import paddle
import pgl

from ppmat.models.gegnn import GEGNNBinary
from ppmat.models.gegnn import GEGNNTensorPredictor


def batch():
    def graph():
        return pgl.Graph(
            num_nodes=2,
            edges=np.array([[0, 1], [1, 0], [0, 0], [1, 1]]),
            node_feat={"h": np.eye(2, 74, dtype="float32")},
        )

    return {
        "g1": pgl.Graph.batch([graph(), graph()]).tensor(),
        "g2": pgl.Graph.batch([graph(), graph()]).tensor(),
        "empty_solvsys": pgl.Graph.batch([graph(), graph()]).tensor(),
        "x1": paddle.to_tensor([[0.0], [1.0]]),
        "inter_hb": paddle.ones([2, 1]),
        "intra_hb1": paddle.zeros([2, 1]),
        "intra_hb2": paddle.ones([2, 1]),
    }


class TestGEGNNTensorPredictor(unittest.TestCase):
    def test_packing_preserves_graph_inputs(self):
        data = batch()
        inputs = GEGNNTensorPredictor.pack(data)
        self.assertTrue(
            all(isinstance(value, paddle.Tensor) for value in inputs.values())
        )
        np.testing.assert_array_equal(
            inputs["g1_edges"].numpy(), data["g1"].edges.numpy()
        )
        np.testing.assert_array_equal(
            inputs["g2_node_features"].numpy(), data["g2"].node_feat["h"].numpy()
        )
        self.assertEqual(inputs["empty_solvsys_node_features"].shape, [4, 1])

    def test_prediction_matches_both_checkpoint_architectures(self):
        data = batch()
        for edge_features in (1, 3):
            model = GEGNNBinary(hidden_dim=8, edge_in_feats=edge_features)
            model.eval()
            predictor = GEGNNTensorPredictor(model)
            predictor.eval()
            np.testing.assert_allclose(
                predictor(predictor.pack(data)).numpy(),
                model.predict(data)["gamma"].numpy(),
                rtol=0,
                atol=1e-6,
            )

    @unittest.skipUnless(
        os.environ.get("GEGNN_TEST_CINN") == "1", "Optional GPU CINN check"
    )
    def test_compiled_inputs_do_not_freeze_composition_or_features(self):
        paddle.set_device("gpu:0")
        model = GEGNNBinary(hidden_dim=8, mlp_activation="softplus")
        model.eval()
        eager = GEGNNTensorPredictor(model)
        eager.eval()
        compiled = paddle.jit.to_static(
            copy.deepcopy(eager), full_graph=True, backend="CINN"
        )
        inputs = eager.pack(batch())
        for variant in range(3):
            if variant == 1:
                inputs["x1"] = 1 - inputs["x1"]
            elif variant == 2:
                inputs["g1_node_features"] = paddle.roll(
                    inputs["g1_node_features"], 1, axis=1
                )
                inputs["inter_hb"] = inputs["inter_hb"] + 1
            np.testing.assert_allclose(
                compiled(inputs).numpy(), eager(inputs).numpy(), rtol=0, atol=1e-4
            )


if __name__ == "__main__":
    unittest.main()

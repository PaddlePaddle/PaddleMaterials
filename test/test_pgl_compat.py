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
import pytest

from ppmat.models.megnet.megnet import EdgeSet2Set
from ppmat.models.megnet.megnet import Set2Set
from ppmat.utils.execution import compiled_execution
from ppmat.utils.execution import in_compiled_execution
from ppmat.utils.pgl_compat import graph_count
from ppmat.utils.pgl_compat import segment_softmax
from ppmat.utils.pgl_compat import segment_sum

# Segment 1 is empty, and with num_segments=5 so is the trailing segment 4.
SEGMENT_IDS = [0, 0, 0, 2, 2, 3]
NUM_SEGMENTS = 5


def _ids():
    return paddle.to_tensor(SEGMENT_IDS, dtype="int64")


def _data(scale=1.0, seed=0):
    rng = np.random.default_rng(seed)
    data = rng.normal(scale=scale, size=(len(SEGMENT_IDS), 3)).astype("float32")
    return paddle.to_tensor(data, stop_gradient=False)


def _value_and_grad(function, data):
    out = function(data)
    weight = paddle.to_tensor(
        np.random.default_rng(1).normal(size=out.shape).astype("float32")
    )
    (grad,) = paddle.grad((out * weight).sum(), data)
    return out.numpy(), grad.numpy()


def _batch(num_nodes=(2, 3, 4)):
    graphs = [
        pgl.Graph(num_nodes=n, edges=[(i, i + 1) for i in range(n - 1)])
        for n in num_nodes
    ]
    return pgl.Graph.batch(graphs).tensor()


def test_compiled_execution_flag_is_scoped():
    assert not in_compiled_execution()
    with compiled_execution():
        with compiled_execution():
            assert in_compiled_execution()
        assert in_compiled_execution()
    assert not in_compiled_execution()
    with pytest.raises(RuntimeError):
        with compiled_execution():
            raise RuntimeError("inside")
    assert not in_compiled_execution()


def test_eager_segment_ops_are_the_pgl_ops():
    data = _data()
    expected_sum = pgl.math.segment_sum(data, _ids()).numpy()
    out_sum = segment_sum(data, _ids(), NUM_SEGMENTS).numpy()
    np.testing.assert_array_equal(out_sum[: expected_sum.shape[0]], expected_sum)
    np.testing.assert_array_equal(out_sum[expected_sum.shape[0] :], 0)
    np.testing.assert_array_equal(segment_sum(data, _ids()).numpy(), expected_sum)
    np.testing.assert_array_equal(
        segment_softmax(data, _ids(), NUM_SEGMENTS).numpy(),
        pgl.math.segment_softmax(data, _ids()).numpy(),
    )


def test_segment_sum_rejects_too_few_segments():
    with pytest.raises(ValueError, match="num_segments=3"):
        segment_sum(_data(), _ids(), 3)


def test_compiled_segment_sum_matches_pgl():
    def reference(data):
        out = pgl.math.segment_sum(data, _ids())
        pad = paddle.zeros([NUM_SEGMENTS - out.shape[0], out.shape[1]])
        return paddle.concat([out, pad], axis=0)

    expected = _value_and_grad(reference, _data())
    with compiled_execution():
        actual = _value_and_grad(
            lambda d: segment_sum(d, _ids(), NUM_SEGMENTS), _data()
        )
    for a, e in zip(actual, expected):
        np.testing.assert_allclose(a, e, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("scale", [1.0, 80.0])
def test_compiled_segment_softmax_matches_pgl(scale):
    expected = _value_and_grad(
        lambda d: pgl.math.segment_softmax(d, _ids()), _data(scale)
    )
    with compiled_execution():
        actual = _value_and_grad(
            lambda d: segment_softmax(d, _ids(), NUM_SEGMENTS), _data(scale)
        )
    for a, e in zip(actual, expected):
        np.testing.assert_allclose(a, e, rtol=1e-6, atol=1e-6)


def test_compiled_segment_ops_convert_with_ast():
    def function(data, ids):
        return segment_sum(
            segment_softmax(data, ids, NUM_SEGMENTS) * data, ids, NUM_SEGMENTS
        )

    data = _data().detach()
    expected = function(data, _ids()).numpy()
    static = paddle.jit.to_static(function, full_graph=True)
    np.testing.assert_allclose(
        static(data, _ids()).numpy(), expected, rtol=1e-6, atol=1e-6
    )


def test_graph_count_matches_num_graph():
    graph = _batch()
    assert graph_count(graph) == int(graph.num_graph) == 3


@pytest.mark.parametrize("layer_type", [Set2Set, EdgeSet2Set])
def test_megnet_set2set_compiled_path_matches_eager(layer_type):
    paddle.seed(7)
    graph = _batch()
    layer = layer_type(4, 3, 1)
    rows = graph.num_nodes if layer_type is Set2Set else graph.num_edges
    x = paddle.randn([int(rows), 4])
    expected = layer(graph, x).numpy()
    with compiled_execution():
        actual = layer(graph, x).numpy()
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)

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

"""MatterGen's triplet bilinear layer against the padded version it replaced."""

import numpy as np
import paddle
import pytest

from ppmat.models.mattergen.mattergen import EfficientInteractionBilinear
from ppmat.models.mattergen.mattergen import EfficientInteractionDownProjection

NUM_SPHERICAL = 7
NUM_RADIAL = 6
EMB_SIZE_CBF = 8
EMB_SIZE_TRIP = 16
UNITS_OUT = 12


def _layers():
    paddle.seed(0)
    down = EfficientInteractionDownProjection(NUM_SPHERICAL, NUM_RADIAL, EMB_SIZE_CBF)
    bilinear = EfficientInteractionBilinear(EMB_SIZE_TRIP, EMB_SIZE_CBF, UNITS_OUT)
    return down, bilinear


def _padded_bilinear(down, bilinear, rbf, sph, m, id_ca, id_ragged_idx):
    """The two layers before the rewrite, with the same weights.

    The triplets of every edge are padded to Kmax, the largest number of triplets
    of an edge, and the padded [E, S, Kmax] and [E, Kmax, emb] tensors multiplied.
    """
    num_edges = rbf.shape[1]
    rbf_w1 = paddle.matmul(rbf, down.weight).transpose([1, 2, 0])
    kmax = int(id_ragged_idx.max()) + 1
    sph2 = paddle.zeros([num_edges, kmax, NUM_SPHERICAL], dtype=sph.dtype)
    sph2[id_ca, id_ragged_idx] = sph
    sph2 = sph2.transpose([0, 2, 1])
    m2 = paddle.zeros([num_edges, kmax, EMB_SIZE_TRIP], dtype=m.dtype)
    m2[id_ca, id_ragged_idx] = m
    sum_k = paddle.matmul(sph2, m2)
    rbf_w1_sum_k = paddle.matmul(rbf_w1, sum_k)
    m_ca = paddle.matmul(rbf_w1_sum_k.transpose([2, 0, 1]), bilinear.weight)
    return m_ca.sum(axis=0)


def _bilinear(down, bilinear, rbf, sph, m, id_ca, id_ragged_idx):
    basis = down(rbf, sph, id_ca, id_ragged_idx)
    return bilinear(basis, m, id_ca, id_ragged_idx)


def _inputs(shuffle):
    """Inputs laid out as GemNetT.get_triplets builds them.

    id_ca is the edge of every triplet and id_ragged_idx its position among the
    triplets of that edge; some edges have no triplets.
    """
    rng = np.random.default_rng(3)
    counts = rng.integers(0, 6, size=40)
    id_ca = np.repeat(np.arange(len(counts)), counts)
    id_ragged_idx = np.concatenate([np.arange(c) for c in counts])
    num_triplets = len(id_ca)
    if shuffle:
        order = rng.permutation(num_triplets)
        id_ca, id_ragged_idx = id_ca[order], id_ragged_idx[order]

    def tensor(shape):
        data = rng.normal(size=shape).astype("float32")
        return paddle.to_tensor(data, stop_gradient=False)

    rbf = tensor([1, len(counts), NUM_RADIAL])
    sph = tensor([num_triplets, NUM_SPHERICAL])
    m = tensor([num_triplets, EMB_SIZE_TRIP])
    return rbf, sph, m, paddle.to_tensor(id_ca), paddle.to_tensor(id_ragged_idx)


def _value_and_grads(function, down, bilinear, rbf, sph, m, id_ca, id_ragged_idx):
    out = function(down, bilinear, rbf, sph, m, id_ca, id_ragged_idx)
    weight = paddle.to_tensor(
        np.random.default_rng(5).normal(size=out.shape).astype("float32")
    )
    inputs = [rbf, sph, m, down.weight, bilinear.weight]
    grads = paddle.grad((out * weight).sum(), inputs)
    return [out.numpy()] + [g.numpy() for g in grads]


@pytest.mark.parametrize("shuffle", [False, True])
def test_bilinear_matches_the_padded_version(shuffle):
    down, bilinear = _layers()
    args = (down, bilinear, *_inputs(shuffle))
    expected = _value_and_grads(_padded_bilinear, *args)
    actual = _value_and_grads(_bilinear, *args)
    # Output, then the gradients of rbf, sph, m and the two weights. The two
    # versions add the same products in a different order.
    for a, e in zip(actual, expected):
        assert a.shape == e.shape
        np.testing.assert_allclose(a, e, rtol=1e-5, atol=1e-5 * np.abs(e).max())


def test_bilinear_without_triplets_gives_zeros():
    # The padded version fails here: the max of an empty id_ragged_idx.
    down, bilinear = _layers()
    rbf = paddle.ones([1, 5, NUM_RADIAL])
    empty = paddle.zeros([0], dtype="int64")
    out = _bilinear(
        down,
        bilinear,
        rbf,
        paddle.zeros([0, NUM_SPHERICAL]),
        paddle.zeros([0, EMB_SIZE_TRIP]),
        empty,
        empty,
    )
    np.testing.assert_array_equal(out.numpy(), np.zeros([5, UNITS_OUT], "float32"))

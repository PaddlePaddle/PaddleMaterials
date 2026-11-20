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

"""MatterGen's PropertyEmbedding on missing (NaN) space-group labels."""

import contextlib

import numpy as np
import paddle
import pytest

from ppmat.models.mattergen.property_embeddings import PropertyEmbedding
from ppmat.utils.execution import compiled_execution

HIDDEN_DIM = 4
NAN = float("nan")


def _space_group_embedding():
    paddle.seed(0)
    return PropertyEmbedding(
        conditional_embedding_module_name="SpaceGroupEmbeddingVector",
        conditional_embedding_module_cfg={"hidden_dim": HIDDEN_DIM},
        unconditional_embedding_module_name="ZerosEmbedding",
        unconditional_embedding_module_cfg={"hidden_dim": HIDDEN_DIM},
        scaler_name="Identity",
        scaler_cfg={},
    )


# Rows with a missing label always use the unconditional embedding (SetEmbeddingType).
@pytest.mark.parametrize("compiled", [False, True])
def test_missing_space_groups_get_the_unconditional_embedding(compiled):
    embedding = _space_group_embedding()
    execution = compiled_execution() if compiled else contextlib.nullcontext()
    with execution:
        mixed = embedding(
            paddle.to_tensor([[225.0], [NAN], [1.0], [NAN]]),
            paddle.to_tensor([[False], [True], [False], [True]]),
        )
        all_missing = embedding(
            paddle.full([3, 1], NAN), paddle.ones([3, 1], dtype="bool")
        )

    expected = embedding.conditional_embedding_module(
        paddle.to_tensor([225.0, 1.0, 1.0, 1.0])
    ).numpy()
    expected[[1, 3]] = 0
    np.testing.assert_array_equal(mixed.numpy(), expected)
    np.testing.assert_array_equal(all_missing.numpy(), np.zeros([3, HIDDEN_DIM]))

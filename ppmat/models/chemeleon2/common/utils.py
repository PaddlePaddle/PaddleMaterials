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

import math

import paddle

__all__ = [
    "get_index_embedding",
    "to_dense_batch",
    "apply_augmentation",
    "apply_noise",
    "set_gelu_approx",
    "make_attn_mask",
]


def get_index_embedding(indices, emb_dim, max_len=2048):
    K = paddle.arange(emb_dim // 2)
    pos_embedding_sin = paddle.sin(
        indices.unsqueeze(-1) * math.pi / (max_len ** (2 * K / emb_dim))
    )
    pos_embedding_cos = paddle.cos(
        indices.unsqueeze(-1) * math.pi / (max_len ** (2 * K / emb_dim))
    )
    pos_embedding = paddle.concat([pos_embedding_sin, pos_embedding_cos], axis=-1)
    return pos_embedding


def to_dense_batch(x, batch_idx, max_num_nodes=None):
    batch_size = int(batch_idx.max().item()) + 1 if batch_idx.numel() > 0 else 1
    batch_idx = batch_idx.reshape([-1]).astype("int64")
    x = x.reshape([x.shape[0], -1])

    num_nodes = paddle.bincount(batch_idx, minlength=batch_size)
    if max_num_nodes is None:
        max_num_nodes = int(num_nodes.max().item())
    elif int(num_nodes.max().item()) > max_num_nodes:
        raise ValueError(
            "max_num_nodes must be >= the largest number of nodes in a graph"
        )

    feat_dim = x.shape[-1]
    total = x.shape[0]
    flat_size = batch_size * max_num_nodes

    # Graph-local node order keeps the dense layout deterministic
    starts = paddle.cumsum(num_nodes) - num_nodes
    local_idx = paddle.arange(total, dtype="int64") - starts[batch_idx]
    flat_idx = batch_idx * max_num_nodes + local_idx

    x_flat = paddle.zeros([flat_size, feat_dim], dtype=x.dtype)
    mask_flat = paddle.zeros([flat_size], dtype="bool")
    # Safe advanced-indexing setitem form (paddle_index_setitem_check rule)
    x_flat[flat_idx] = x
    mask_flat[flat_idx] = True
    x_dense = x_flat.reshape([batch_size, max_num_nodes, feat_dim])
    mask = mask_flat.reshape([batch_size, max_num_nodes])
    return x_dense, mask


def apply_augmentation(batch, translate=False, rotate=False):
    if not translate and not rotate:
        return batch
    batch_aug = batch.clone()
    if translate:
        batch_aug = _augmentation_translate(batch_aug)
    if rotate:
        batch_aug = _augmentation_rotate(batch_aug)
    return batch_aug


def _augmentation_translate(batch):
    lengths_mean = batch.lengths.mean(axis=0)
    lengths_std = batch.lengths.std(axis=0, unbiased=False)
    random_translate = (
        paddle.normal(
            mean=paddle.abs(lengths_mean),
            std=paddle.maximum(paddle.abs(lengths_std), paddle.to_tensor([1e-8])),
        )
        / 2
    )
    cart_coords_aug = batch.cart_coords + random_translate
    cell_per_node_inv = paddle.inverse(batch.lattices[batch.batch])
    frac_coords_aug = (
        paddle.einsum("bi,bij->bj", cart_coords_aug, cell_per_node_inv) % 1.0
    )
    batch.cart_coords = cart_coords_aug
    batch.frac_coords = frac_coords_aug
    return batch


def _augmentation_rotate(batch):
    rot_mat = _random_rotation_matrix()
    batch.cart_coords = paddle.matmul(batch.cart_coords, rot_mat.T)
    batch.lattices = paddle.matmul(batch.lattices, rot_mat.T)
    return batch


def apply_noise(batch, ratio=0.1, corruption_scale=0.1):
    if ratio <= 0:
        return batch
    batch_noise = batch.clone()
    total_num_atoms = batch_noise.num_nodes
    noise_num_atoms = int(total_num_atoms * ratio)
    noise_atom_types = batch_noise.atom_types.clone()
    type_noise_idx = paddle.randperm(total_num_atoms)[:noise_num_atoms]
    noise_atom_types[type_noise_idx] = 0
    noise_cart_coords = batch_noise.cart_coords.clone()
    coord_noise_idx = paddle.randperm(total_num_atoms)[:noise_num_atoms]
    noise_cart_coords[coord_noise_idx] += (
        paddle.randn([noise_num_atoms, 3]) * corruption_scale
    )
    cell_per_node_inv = paddle.inverse(batch.lattices[batch.batch])
    noise_frac_coords = (
        paddle.einsum("bi,bij->bj", noise_cart_coords, cell_per_node_inv) % 1.0
    )
    batch_noise.atom_types = noise_atom_types
    batch_noise.cart_coords = noise_cart_coords
    batch_noise.frac_coords = noise_frac_coords
    return batch_noise


def set_gelu_approx(transformer):
    if hasattr(transformer, "layers"):
        for layer in transformer.layers:
            if hasattr(layer, "activation"):
                layer.activation = paddle.nn.GELU(approximate="tanh")


def make_attn_mask(token_mask):
    if not token_mask.cast("bool").all():
        bsize, seq_len = token_mask.shape
        mask = paddle.zeros([bsize, 1, 1, seq_len], dtype="float32")
        return mask - 1e9 * (~token_mask).unsqueeze(1).unsqueeze(2).astype("float32")
    return None


def _random_rotation_matrix():
    q = paddle.randn([4])
    q = q / paddle.norm(q)
    rot_mat = paddle.to_tensor(
        [
            [
                1 - 2 * q[2] ** 2 - 2 * q[3] ** 2,
                2 * q[1] * q[2] - 2 * q[0] * q[3],
                2 * q[1] * q[3] + 2 * q[0] * q[2],
            ],
            [
                2 * q[1] * q[2] + 2 * q[0] * q[3],
                1 - 2 * q[1] ** 2 - 2 * q[3] ** 2,
                2 * q[2] * q[3] - 2 * q[0] * q[1],
            ],
            [
                2 * q[1] * q[3] - 2 * q[0] * q[2],
                2 * q[2] * q[3] + 2 * q[0] * q[1],
                1 - 2 * q[1] ** 2 - 2 * q[2] ** 2,
            ],
        ],
        dtype="float32",
    )
    return rot_mat

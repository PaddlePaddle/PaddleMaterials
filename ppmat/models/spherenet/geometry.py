# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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
"""
3D geometry utilities for spherical message-passing models.

SphereNet-specific — computes distance, angle, and torsion from 3D
atomic coordinates for spherical message passing.
"""

import math

import paddle

from ppmat.utils.scatter import scatter_argmin


def _cross_product(left, right):
    # paddle.linalg.cross lacks the second-order gradient required by force loss.
    return paddle.stack(
        (
            left[:, 1] * right[:, 2] - left[:, 2] * right[:, 1],
            left[:, 2] * right[:, 0] - left[:, 0] * right[:, 2],
            left[:, 0] * right[:, 1] - left[:, 1] * right[:, 0],
        ),
        axis=-1,
    )


def select_torsion_edges(pos, edge_index, triplet_indices):
    """Select the discrete q->j torsion edge for every cached triplet."""

    src, dst = edge_index
    idx_kj = triplet_indices["idx_kj"]
    idx_ji = triplet_indices["idx_ji"]
    edge_vector = pos[dst] - pos[src]
    axis = edge_vector[idx_ji].detach()
    reference = -edge_vector[idx_kj].detach()
    incoming_order = paddle.argsort(dst, stable=True)
    incoming_counts = paddle.bincount(dst, minlength=pos.shape[0])
    centers = src[idx_ji]
    i_atoms = dst[idx_ji]
    candidate_counts = incoming_counts[centers]
    candidate_triplet = paddle.repeat_interleave(
        paddle.arange(idx_kj.shape[0], dtype="int64"), candidate_counts
    )
    candidate_starts = paddle.cumsum(candidate_counts) - candidate_counts
    candidate_offsets = paddle.arange(
        candidate_triplet.shape[0], dtype="int64"
    ) - paddle.repeat_interleave(candidate_starts, candidate_counts)
    incoming_starts = paddle.cumsum(incoming_counts) - incoming_counts
    candidate_positions = (
        paddle.repeat_interleave(incoming_starts[centers], candidate_counts)
        + candidate_offsets
    )
    idx_qj = incoming_order[candidate_positions]
    valid_candidates = src[idx_qj] != i_atoms[candidate_triplet]
    idx_qj = idx_qj[valid_candidates]
    candidate_triplet = candidate_triplet[valid_candidates]

    candidate_axis = axis[candidate_triplet]
    candidate_reference = reference[candidate_triplet]
    candidate = -edge_vector.detach()[idx_qj]
    reference_plane = paddle.linalg.cross(candidate_axis, candidate_reference)
    candidate_plane = paddle.linalg.cross(candidate_axis, candidate)
    torsion_x = paddle.sum(reference_plane * candidate_plane, axis=-1)
    torsion_y = paddle.sum(
        paddle.linalg.cross(reference_plane, candidate_plane) * candidate_axis,
        axis=-1,
    ) / paddle.sqrt(paddle.sum(candidate_axis * candidate_axis, axis=-1))
    candidate_torsion = paddle.atan2(torsion_y, torsion_x)
    candidate_torsion = paddle.where(
        candidate_torsion <= 0,
        candidate_torsion + 2 * math.pi,
        candidate_torsion,
    )
    selected_candidates = scatter_argmin(
        candidate_torsion, candidate_triplet, idx_kj.shape[0]
    )
    return idx_qj[selected_candidates]


def compute_geometry(pos, edge_index, triplet_indices, idx_qj=None):
    """Compute differentiable SphereNet distance, angle, and torsion data."""
    src, dst = edge_index
    idx_kj = triplet_indices["idx_kj"]
    idx_ji = triplet_indices["idx_ji"]

    edge_vector = pos[dst] - pos[src]
    dist = paddle.sqrt(paddle.sum(edge_vector * edge_vector, axis=-1))

    axis = edge_vector[idx_ji]
    reference = -edge_vector[idx_kj]
    angle_cross = _cross_product(axis, reference)
    angle_sin = paddle.sqrt(paddle.sum(angle_cross * angle_cross, axis=-1) + 1e-8)
    angle_cos = paddle.sum(axis * reference, axis=-1)
    angle_norm = paddle.sqrt(angle_cos * angle_cos + angle_sin * angle_sin)
    angle_cos = angle_cos / angle_norm
    angle_sin = angle_sin / angle_norm

    # Match the original forward values while keeping force-loss gradients on
    # operations that support Paddle's second-order automatic differentiation.
    legacy_axis = axis.detach()
    legacy_reference = reference.detach()
    legacy_angle_cross = paddle.linalg.cross(legacy_axis, legacy_reference)
    legacy_angle = paddle.atan2(
        paddle.sqrt(paddle.sum(legacy_angle_cross * legacy_angle_cross, axis=-1)),
        paddle.sum(legacy_axis * legacy_reference, axis=-1),
    )
    angle_cos = angle_cos + (paddle.cos(legacy_angle) - angle_cos).detach()
    angle_sin = angle_sin + (paddle.sin(legacy_angle) - angle_sin).detach()

    if idx_qj is None:
        idx_qj = select_torsion_edges(pos, edge_index, triplet_indices)

    candidate_axis = legacy_axis
    candidate_reference = legacy_reference
    candidate = -edge_vector.detach()[idx_qj]
    reference_plane = paddle.linalg.cross(candidate_axis, candidate_reference)
    candidate_plane = paddle.linalg.cross(candidate_axis, candidate)
    torsion_x = paddle.sum(reference_plane * candidate_plane, axis=-1)
    torsion_y = paddle.sum(
        paddle.linalg.cross(reference_plane, candidate_plane) * candidate_axis,
        axis=-1,
    ) / paddle.sqrt(paddle.sum(candidate_axis * candidate_axis, axis=-1))
    legacy_torsion = paddle.atan2(torsion_y, torsion_x)
    legacy_torsion = paddle.where(
        legacy_torsion <= 0,
        legacy_torsion + 2 * math.pi,
        legacy_torsion,
    )

    candidate = -edge_vector[idx_qj]
    reference_plane = _cross_product(axis, reference)
    candidate_plane = _cross_product(axis, candidate)
    torsion_x = paddle.sum(reference_plane * candidate_plane, axis=-1)
    torsion_y = paddle.sum(
        _cross_product(reference_plane, candidate_plane) * axis,
        axis=-1,
    ) / paddle.sqrt(paddle.sum(axis * axis, axis=-1))
    torsion_squared_norm = torsion_x * torsion_x + torsion_y * torsion_y
    valid_torsion = torsion_squared_norm > 1e-12
    torsion_norm = paddle.sqrt(
        paddle.where(
            valid_torsion,
            torsion_squared_norm,
            paddle.ones_like(torsion_squared_norm),
        )
    )
    torsion_cos = paddle.where(
        valid_torsion, torsion_x / torsion_norm, paddle.ones_like(torsion_x)
    )
    torsion_sin = paddle.where(
        valid_torsion, torsion_y / torsion_norm, paddle.zeros_like(torsion_y)
    )
    torsion_cos = torsion_cos + (paddle.cos(legacy_torsion) - torsion_cos).detach()
    torsion_sin = torsion_sin + (paddle.sin(legacy_torsion) - torsion_sin).detach()

    return (
        dist,
        (angle_cos, angle_sin),
        (torsion_cos, torsion_sin),
        dst,
        src,
        idx_kj,
        idx_ji,
    )

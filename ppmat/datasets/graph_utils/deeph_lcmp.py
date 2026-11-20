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

"""Local-coordinate message-passing (LCMP) angular graph features."""

import numpy as np
from scipy.special import gammaln
from scipy.special import lpmv


def build_lcmp_subgraph(
    edge_index,
    edge_features,
    local_rotations,
    connected_atoms,
    connected_edges,
    num_l,
):
    """Build LCMP features using DeepH's real spherical-harmonic convention.

    The polar axis is x and negative m uses sin(|m| phi). SciPy supplies the
    associated Legendre polynomials, including their Condon-Shortley phase.
    Only connected edge/frame pairs are evaluated.
    """
    atom_indices = []
    edge_indices = []
    subgraph_indices = []
    subgraph_index = 0
    for current_edge in range(edge_features.shape[0]):
        for atom_index in edge_index[:, current_edge]:
            neighbor_edges = np.asarray(connected_edges[atom_index], dtype=np.int64)
            neighbor_atoms = connected_atoms[atom_index]
            atom_indices.append(
                np.stack(
                    [np.repeat(atom_index, len(neighbor_atoms)), neighbor_atoms],
                    axis=1,
                )
            )
            edge_indices.append(neighbor_edges)
            subgraph_indices.extend([subgraph_index] * len(neighbor_atoms))
            subgraph_index += 1

    edge_indices = np.concatenate(edge_indices).astype(np.int64)
    subgraph_indices = np.asarray(subgraph_indices, dtype=np.int64)
    vectors = edge_features[:, 1:4] - edge_features[:, 4:7]
    vectors = np.matmul(
        vectors[edge_indices, None, :],
        local_rotations[subgraph_indices // 2],
    ).reshape(-1, 3)
    theta = np.arctan2(np.sqrt(vectors[:, 1] ** 2 + vectors[:, 2] ** 2), vectors[:, 0])
    phi = np.arctan2(vectors[:, 2], vectors[:, 1])
    angular_features = []
    for degree in range(num_l):
        for order in range(-degree, degree + 1):
            abs_order = abs(order)
            norm = np.sqrt(
                (2 * degree + 1)
                / (4 * np.pi)
                * np.exp(
                    gammaln(degree - abs_order + 1) - gammaln(degree + abs_order + 1)
                )
            )
            value = norm * lpmv(abs_order, degree, np.cos(theta))
            if order > 0:
                value *= np.sqrt(2.0) * np.cos(order * phi)
            elif order < 0:
                value *= np.sqrt(2.0) * np.sin(abs_order * phi)
            angular_features.append(value)

    return {
        "subgraph_atom_idx": np.concatenate(atom_indices).astype(np.int64),
        "subgraph_edge_idx": edge_indices,
        "subgraph_edge_ang": np.stack(angular_features, axis=-1).astype(
            edge_features.dtype
        ),
        "subgraph_index": subgraph_indices,
    }

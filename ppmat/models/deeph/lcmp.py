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


def _semifactorial(value):
    result = 1.0
    for factor in range(value, 1, -2):
        result *= factor
    return result


def _pochhammer(value, count):
    result = float(value)
    for factor in range(value + 1, value + count):
        result *= factor
    return result


class _SphericalHarmonics:
    """Real spherical harmonics used by LCMP angular features."""

    def __init__(self):
        self.legendre = {}

    def _negative_lpmv(self, degree, order, value):
        if order < 0:
            value *= (-1) ** order / _pochhammer(degree + order + 1, -2 * order)
        return value

    def lpmv(self, degree, order, value):
        order_abs = abs(order)
        if (degree, order) in self.legendre:
            return self.legendre[(degree, order)]
        if order_abs > degree:
            return None
        if degree == 0:
            self.legendre[(degree, order)] = np.ones_like(value)
            return self.legendre[(degree, order)]
        if order_abs == degree:
            result = (-1) ** order_abs * _semifactorial(2 * order_abs - 1)
            result *= np.power(1 - value * value, order_abs / 2)
            self.legendre[(degree, order)] = self._negative_lpmv(degree, order, result)
            return self.legendre[(degree, order)]

        self.lpmv(degree - 1, order, value)
        result = (
            ((2 * degree - 1) / (degree - order_abs))
            * value
            * self.lpmv(degree - 1, order_abs, value)
        )
        if degree - order_abs > 1:
            result -= ((degree + order_abs - 1) / (degree - order_abs)) * (
                self.legendre[(degree - 2, order_abs)]
            )
        if order < 0:
            result = self._negative_lpmv(degree, order, result)
        self.legendre[(degree, order)] = result
        return result

    def get_element(self, degree, order, theta, phi):
        norm = np.sqrt((2 * degree + 1) / (4 * np.pi))
        legendre = self.lpmv(degree, abs(order), np.cos(theta))
        if order == 0:
            return norm * legendre
        angular = (
            np.cos(order * phi) * legendre
            if order > 0
            else np.sin(abs(order) * phi) * legendre
        )
        norm *= np.sqrt(2.0 / _pochhammer(degree - abs(order) + 1, 2 * abs(order)))
        return angular * norm

    def get(self, degree, theta, phi):
        self.legendre = {}
        return np.stack(
            [
                self.get_element(degree, order, theta, phi)
                for order in range(-degree, degree + 1)
            ],
            axis=-1,
        )


def _cartesian_to_spherical(cartesian):
    spherical = np.zeros(cartesian.shape[:-1] + (2,), dtype=cartesian.dtype)
    radius_yz = cartesian[..., 1] ** 2 + cartesian[..., 2] ** 2
    spherical[..., 0] = np.arctan2(np.sqrt(radius_yz), cartesian[..., 0])
    spherical[..., 1] = np.arctan2(cartesian[..., 2], cartesian[..., 1])
    return spherical


def build_lcmp_subgraph(
    edge_index,
    edge_features,
    local_rotations,
    connected_atoms,
    connected_edges,
    num_l,
):
    """Build DeepH LCMP indices and real spherical-harmonic edge features."""
    relative_vectors = edge_features[:, 1:4] - edge_features[:, 4:7]
    relative_vectors = np.matmul(
        relative_vectors[:, None, None, :],
        local_rotations[None, :, :, :],
    ).reshape(-1, 3)
    spherical = _cartesian_to_spherical(relative_vectors)
    harmonics = _SphericalHarmonics()
    angular_features = np.concatenate(
        [
            harmonics.get(degree, spherical[:, 0], spherical[:, 1])
            for degree in range(num_l)
        ],
        axis=-1,
    ).reshape(edge_features.shape[0], edge_features.shape[0], -1)

    atom_indices = []
    edge_indices = []
    edge_angles = []
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
            edge_angles.append(angular_features[neighbor_edges, current_edge, :])
            subgraph_indices.extend([subgraph_index] * len(neighbor_atoms))
            subgraph_index += 1

    return {
        "subgraph_atom_idx": np.concatenate(atom_indices).astype(np.int64),
        "subgraph_edge_idx": np.concatenate(edge_indices).astype(np.int64),
        "subgraph_edge_ang": np.concatenate(edge_angles).astype(edge_features.dtype),
        "subgraph_index": np.asarray(subgraph_indices, dtype=np.int64),
    }

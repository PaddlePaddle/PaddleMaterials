# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Paddle port of the ``mace.modules.utils`` helpers used for E/F/S.

The layout mirrors the official implementation so the numerical conventions
(especially the symmetric-displacement trick for the virial) stay identical.
"""

from typing import Optional
from typing import Tuple

import paddle

__all__ = [
    "det3x3",
    "get_edge_vectors_and_lengths",
    "get_symmetric_displacement",
    "compute_forces",
    "compute_forces_virials",
    "get_outputs",
    "prepare_graph",
]


def det3x3(matrix: paddle.Tensor) -> paddle.Tensor:
    """Determinant of batched 3x3 matrices.

    ``paddle.linalg.det`` cannot be differentiated twice, which training needs
    when forces/stresses are themselves differentiated, so expand the cofactor
    form explicitly.
    """
    a00, a01, a02 = matrix[:, 0, 0], matrix[:, 0, 1], matrix[:, 0, 2]
    a10, a11, a12 = matrix[:, 1, 0], matrix[:, 1, 1], matrix[:, 1, 2]
    a20, a21, a22 = matrix[:, 2, 0], matrix[:, 2, 1], matrix[:, 2, 2]
    return (
        a00 * (a11 * a22 - a12 * a21)
        - a01 * (a10 * a22 - a12 * a20)
        + a02 * (a10 * a21 - a11 * a20)
    )


def get_edge_vectors_and_lengths(
    positions: paddle.Tensor,  # [n_nodes, 3]
    edge_index: paddle.Tensor,  # [2, n_edges]
    shifts: paddle.Tensor,  # [n_edges, 3]
    normalize: bool = False,
    eps: float = 1e-9,
) -> Tuple[paddle.Tensor, paddle.Tensor]:
    sender = edge_index[0]
    receiver = edge_index[1]
    vectors = positions[receiver] - positions[sender] + shifts  # [n_edges, 3]
    # sqrt(sum(r^2)) instead of paddle.linalg.norm: the norm gradient has no
    # second-order implementation. Clipping the squared length keeps sqrt
    # differentiable; zero-length edges do not occur (self edges are excluded).
    lengths = paddle.sqrt(
        paddle.clip(paddle.sum(vectors * vectors, axis=-1, keepdim=True), min=eps)
    )  # [n_edges, 1]
    if normalize:
        vectors = vectors / lengths
    return vectors, lengths


def get_symmetric_displacement(
    positions: paddle.Tensor,  # [n_nodes, 3]
    unit_shifts: paddle.Tensor,  # [n_edges, 3]
    cell: Optional[paddle.Tensor],  # [n_graphs, 3, 3]
    edge_index: paddle.Tensor,  # [2, n_edges]
    num_graphs: int,
    batch: paddle.Tensor,  # [n_nodes]
    displacement: Optional[paddle.Tensor] = None,  # [n_graphs, 3, 3]
) -> Tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]:
    if cell is None:
        cell = paddle.zeros(
            [num_graphs * 3, 3], dtype=positions.dtype
        )
    sender = edge_index[0]
    if displacement is None:
        displacement = paddle.zeros([num_graphs, 3, 3], dtype=positions.dtype)
        # Keep displacement connected to the autograd graph of positions so the
        # virial is always defined, exactly like the official implementation.
        displacement = displacement + positions.sum() * 0.0
    symmetric_displacement = 0.5 * (
        displacement + displacement.transpose([0, 2, 1])
    )  # [n_graphs, 3, 3]
    # r' = r + r @ eps, written with bmm because Paddle's einsum gradient has no
    # second-order implementation.
    positions = (
        positions
        + paddle.bmm(
            positions.unsqueeze(-2), symmetric_displacement[batch]
        ).squeeze(-2)
    )
    cell = cell.reshape([-1, 3, 3])
    cell = cell + paddle.matmul(cell, symmetric_displacement)
    shifts = paddle.bmm(
        unit_shifts.unsqueeze(-2), cell[batch[sender]]
    ).squeeze(-2)
    return positions, shifts, displacement


def compute_forces(
    energy: paddle.Tensor, positions: paddle.Tensor, training: bool = True
) -> paddle.Tensor:
    grad_outputs = [paddle.ones_like(energy)]
    gradient = paddle.grad(
        outputs=[energy],
        inputs=[positions],
        grad_outputs=grad_outputs,
        retain_graph=training,
        create_graph=training,
        allow_unused=True,
    )[0]
    if gradient is None:
        return paddle.zeros_like(positions)
    return -1 * gradient


def compute_forces_virials(
    energy: paddle.Tensor,
    positions: paddle.Tensor,
    displacement: paddle.Tensor,
    cell: paddle.Tensor,
    training: bool = True,
    compute_stress: bool = False,
) -> Tuple[paddle.Tensor, paddle.Tensor, Optional[paddle.Tensor]]:
    grad_outputs = [paddle.ones_like(energy)]
    forces, virials = paddle.grad(
        outputs=[energy],
        inputs=[positions, displacement],
        grad_outputs=grad_outputs,
        retain_graph=training,
        create_graph=training,
        allow_unused=True,
    )
    stress = paddle.zeros_like(displacement)
    if compute_stress and virials is not None:
        cell = cell.reshape([-1, 3, 3])
        volume = paddle.abs(det3x3(cell)).unsqueeze(-1)  # [n_graphs, 1]
        stress = virials / volume.reshape([-1, 1, 1])
        stress = paddle.where(
            paddle.abs(stress) < 1e10, stress, paddle.zeros_like(stress)
        )
    if forces is None:
        forces = paddle.zeros_like(positions)
    if virials is None:
        virials = paddle.zeros([1, 3, 3], dtype=positions.dtype)
    return -1 * forces, -1 * virials, stress


def get_outputs(
    energy: paddle.Tensor,
    positions: paddle.Tensor,
    displacement: Optional[paddle.Tensor],
    cell: paddle.Tensor,
    vectors: Optional[paddle.Tensor] = None,
    training: bool = False,
    compute_force: bool = True,
    compute_virials: bool = True,
    compute_stress: bool = True,
):
    forces = None
    virials = None
    stress = None
    if (compute_virials or compute_stress) and displacement is not None:
        forces, virials, stress = compute_forces_virials(
            energy=energy,
            positions=positions,
            displacement=displacement,
            cell=cell,
            training=training,
            compute_stress=compute_stress,
        )
    elif compute_force:
        forces = compute_forces(energy=energy, positions=positions, training=training)
    return forces, virials, stress


def prepare_graph(
    data,
    compute_virials: bool = False,
    compute_stress: bool = False,
    compute_displacement: bool = False,
):
    """Prepare positions/vectors/lengths for the forward pass.

    ``data`` is a dict of paddle tensors with the keys ``positions``,
    ``cell``, ``edge_index``, ``unit_shifts``, ``batch``, ``num_graphs`` and
    ``node_heads``. Only the subset of the official ``GraphContext`` needed for
    E/F/S is returned.
    """
    positions = data["positions"]
    positions.stop_gradient = False
    cell = data.get("cell", None)
    edge_index = data["edge_index"]
    num_graphs = int(data["num_graphs"])
    batch = data["batch"]
    node_heads = data.get("node_heads", None)
    if node_heads is None:
        node_heads = paddle.zeros([positions.shape[0]], dtype="int64")

    displacement = data.get("displacement", None)
    if compute_virials or compute_stress or compute_displacement:
        positions, shifts, displacement = get_symmetric_displacement(
            positions=positions,
            unit_shifts=data["unit_shifts"],
            cell=cell,
            edge_index=edge_index,
            num_graphs=num_graphs,
            batch=batch,
            displacement=displacement,
        )
    else:
        # The official implementation expects precomputed cartesian shifts in
        # this branch; accept them directly and fall back to ``unit_shifts @ cell``.
        shifts = data.get("shifts", None)
        if shifts is None:
            if cell is None:
                shifts = paddle.zeros_like(data["unit_shifts"])
            else:
                shifts = paddle.bmm(
                    data["unit_shifts"].unsqueeze(-2), cell[batch[edge_index[0]]]
                ).squeeze(-2)

    vectors, lengths = get_edge_vectors_and_lengths(
        positions=positions,
        edge_index=edge_index,
        shifts=shifts,
    )
    return {
        "num_graphs": num_graphs,
        "num_atoms_arange": paddle.arange(positions.shape[0], dtype="int64"),
        "displacement": displacement,
        "positions": positions,
        "vectors": vectors,
        "lengths": lengths,
        "cell": cell,
        "node_heads": node_heads,
    }
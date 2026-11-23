# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the License);
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an AS IS BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import copy

import numpy as np
import paddle
from pymatgen.core.periodic_table import Element

from ppmat.utils import paddle_aux  # noqa: F401
from ppmat.utils.paddle_aux import dim2perm

OFFSET_LIST = [
    [-1, -1, -1],
    [-1, -1, 0],
    [-1, -1, 1],
    [-1, 0, -1],
    [-1, 0, 0],
    [-1, 0, 1],
    [-1, 1, -1],
    [-1, 1, 0],
    [-1, 1, 1],
    [0, -1, -1],
    [0, -1, 0],
    [0, -1, 1],
    [0, 0, -1],
    [0, 0, 0],
    [0, 0, 1],
    [0, 1, -1],
    [0, 1, 0],
    [0, 1, 1],
    [1, -1, -1],
    [1, -1, 0],
    [1, -1, 1],
    [1, 0, -1],
    [1, 0, 0],
    [1, 0, 1],
    [1, 1, -1],
    [1, 1, 0],
    [1, 1, 1],
]


def normalize_coordinate_unit(coordinate_unit: str) -> str:
    """Validate and normalize a supported coordinate unit."""

    if not isinstance(coordinate_unit, str):
        raise TypeError("coordinate_unit must be a string.")
    coordinate_unit = coordinate_unit.strip().lower()
    if coordinate_unit not in {"angstrom", "bohr"}:
        raise ValueError(
            "coordinate_unit must be either 'angstrom' or 'bohr', but got "
            f"{coordinate_unit!r}."
        )
    return coordinate_unit


def atomic_number_from_symbol(symbol: str) -> int:
    """Return the atomic number, accepting cvve CUBE ``X{n}`` placeholders."""

    symbol = str(symbol)
    if symbol.startswith("X") and symbol[1:].isdigit():
        return int(symbol[1:])
    return int(Element(symbol).Z)


def lattice_params_to_matrix_paddle(lengths, angles):
    """Batched paddle version to compute lattice matrix from params.

    lengths: paddle.Tensor of shape (N, 3), unit A
    angles: paddle.Tensor of shape (N, 3), unit degree
    """
    angles_r = paddle.deg2rad(x=angles)
    coses = paddle.cos(x=angles_r)
    sins = paddle.sin(x=angles_r)
    val = (coses[:, 0] * coses[:, 1] - coses[:, 2]) / (sins[:, 0] * sins[:, 1])
    val = paddle.clip(x=val, min=-1.0, max=1.0)
    gamma_star = paddle.acos(x=val)
    vector_a = paddle.stack(
        x=[
            lengths[:, 0] * sins[:, 1],
            paddle.zeros(shape=lengths.shape[0]),
            lengths[:, 0] * coses[:, 1],
        ],
        axis=1,
    )
    vector_b = paddle.stack(
        x=[
            -lengths[:, 1] * sins[:, 0] * paddle.cos(x=gamma_star),
            lengths[:, 1] * sins[:, 0] * paddle.sin(x=gamma_star),
            lengths[:, 1] * coses[:, 0],
        ],
        axis=1,
    )
    vector_c = paddle.stack(
        x=[
            paddle.zeros(shape=lengths.shape[0]),
            paddle.zeros(shape=lengths.shape[0]),
            lengths[:, 2],
        ],
        axis=1,
    )
    return paddle.stack(x=[vector_a, vector_b, vector_c], axis=1)


def lattices_to_params_shape_paddle(lattices):
    lengths = paddle.sqrt(x=paddle.sum(x=lattices**2, axis=-1))
    angles = paddle.zeros_like(x=lengths)
    for i in range(3):
        j = (i + 1) % 3
        k = (i + 2) % 3
        angles[..., i] = paddle.clip(
            x=paddle.sum(x=lattices[..., j, :] * lattices[..., k, :], axis=-1)
            / (lengths[..., j] * lengths[..., k]),
            min=-1.0,
            max=1.0,
        )
    angles = paddle.acos(x=angles) * 180.0 / np.pi
    return lengths, angles


def lattices_to_params_shape_numpy(lattices):
    lengths = np.sum(lattices**2, axis=-1) ** 0.5
    angles = np.zeros_like(lengths)
    for i in range(3):
        j = (i + 1) % 3
        k = (i + 2) % 3
        angles[..., i] = np.clip(
            np.sum(lattices[..., j, :] * lattices[..., k, :], axis=-1)
            / (lengths[..., j] * lengths[..., k]),
            a_min=-1.0,
            a_max=1.0,
        )
    angles = np.arccos(angles) * 180.0 / np.pi
    return lengths, angles


def lattices_to_params_shape(lattices):
    if isinstance(lattices, np.ndarray):
        return lattices_to_params_shape_numpy(lattices)
    elif isinstance(lattices, paddle.Tensor):
        return lattices_to_params_shape_paddle(lattices)
    else:
        raise TypeError(f"Unsupported type {type(lattices)}.")


def lattice_params_to_matrix(a, b, c, alpha, beta, gamma):
    """Converts lattice from abc, angles to matrix.
    https://github.com/materialsproject/pymatgen/blob/b789d74639aa851d7e5ee427a765d9fd5a8d1079/pymatgen/core/lattice.py#L311
    """
    angles_r = np.radians([alpha, beta, gamma])
    cos_alpha, cos_beta, cos_gamma = np.cos(angles_r)
    sin_alpha, sin_beta, sin_gamma = np.sin(angles_r)

    val = (cos_alpha * cos_beta - cos_gamma) / (sin_alpha * sin_beta)
    val = abs_cap(val)
    gamma_star = np.arccos(val)

    vector_a = [a * sin_beta, 0.0, a * cos_beta]
    vector_b = [
        -b * sin_alpha * np.cos(gamma_star),
        b * sin_alpha * np.sin(gamma_star),
        b * cos_alpha,
    ]
    vector_c = [0.0, 0.0, float(c)]
    return np.array([vector_a, vector_b, vector_c])


def abs_cap(val, max_abs_val=1):
    """
    Returns the value with its absolute value capped at max_abs_val.
    Particularly useful in passing values to trignometric functions where
    numerical errors may result in an argument > 1 being passed in.
    https://github.com/materialsproject/pymatgen/blob/b789d74639aa851d7e5ee427a765d9fd5a8d1079/pymatgen/util/num.py#L15
    Args:
        val (float): Input value.
        max_abs_val (float): The maximum absolute value for val. Defaults to 1.
    Returns:
        val if abs(val) < 1 else sign of val * max_abs_val.
    """
    return max(min(val, max_abs_val), -max_abs_val)


def get_pbc_distances(
    coords,
    edge_index,
    lattice,
    to_jimages,
    num_atoms,
    num_bonds,
    coord_is_cart=False,
    return_offsets=False,
    return_distance_vec=False,
):
    # lattice = lattice_params_to_matrix_paddle(lengths, angles)
    if coord_is_cart:
        pos = coords
    else:
        lattice_nodes = paddle.repeat_interleave(x=lattice, repeats=num_atoms, axis=0)
        pos = paddle.bmm(coords.unsqueeze(1), lattice_nodes).squeeze(1)
    j_index, i_index = edge_index
    distance_vectors = pos[j_index] - pos[i_index]
    lattice_edges = paddle.repeat_interleave(x=lattice, repeats=num_bonds, axis=0)
    offsets = paddle.bmm(
        to_jimages.astype(dtype="float32").unsqueeze(1), lattice_edges
    ).squeeze(1)
    distance_vectors += offsets
    distances = distance_vectors.norm(axis=-1)
    out = {"edge_index": edge_index, "distances": distances}
    if return_distance_vec:
        out["distance_vec"] = distance_vectors
    if return_offsets:
        out["offsets"] = offsets
    return out


def radius_graph_pbc(
    cart_coords,
    lattice,
    num_atoms,
    radius,
    max_num_neighbors_threshold,
    device,
    topk_per_pair=None,
):
    """Computes pbc graph edges under pbc."""
    batch_size = len(num_atoms)
    atom_pos = cart_coords
    num_atoms_per_image = num_atoms
    num_atoms_per_image_sqr = (num_atoms_per_image**2).astype(dtype="int64")
    index_offset = paddle.cumsum(x=num_atoms_per_image, axis=0) - num_atoms_per_image
    index_offset_expand = paddle.repeat_interleave(
        x=index_offset, repeats=num_atoms_per_image_sqr
    )
    num_atoms_per_image_expand = paddle.repeat_interleave(
        x=num_atoms_per_image, repeats=num_atoms_per_image_sqr
    )
    num_atom_pairs = paddle.sum(x=num_atoms_per_image_sqr)
    index_sqr_offset = (
        paddle.cumsum(x=num_atoms_per_image_sqr, axis=0) - num_atoms_per_image_sqr
    )
    index_sqr_offset = paddle.repeat_interleave(
        x=index_sqr_offset, repeats=num_atoms_per_image_sqr
    )
    atom_count_sqr = paddle.arange(end=num_atom_pairs) - index_sqr_offset
    index1 = (atom_count_sqr // num_atoms_per_image_expand).astype(
        dtype="int64"
    ) + index_offset_expand
    index2 = (atom_count_sqr % num_atoms_per_image_expand).astype(
        dtype="int64"
    ) + index_offset_expand
    pos1 = paddle.index_select(x=atom_pos, axis=0, index=index1)
    pos2 = paddle.index_select(x=atom_pos, axis=0, index=index2)
    unit_cell = paddle.to_tensor(data=OFFSET_LIST, place=device).astype(dtype="float32")
    num_cells = len(unit_cell)
    unit_cell_per_atom = unit_cell.view(1, num_cells, 3).repeat(len(index2), 1, 1)
    x = unit_cell
    perm_1 = list(range(x.ndim))
    perm_1[0] = 1
    perm_1[1] = 0
    unit_cell = paddle.transpose(x=x, perm=perm_1)
    unit_cell_batch = unit_cell.view(1, 3, num_cells).expand(shape=[batch_size, -1, -1])

    x = lattice
    perm_2 = list(range(x.ndim))
    perm_2[1] = 2
    perm_2[2] = 1
    data_cell = paddle.transpose(x=x, perm=perm_2)
    pbc_offsets = paddle.bmm(x=data_cell, y=unit_cell_batch)
    pbc_offsets_per_atom = paddle.repeat_interleave(
        x=pbc_offsets, repeats=num_atoms_per_image_sqr, axis=0
    )
    pos1 = pos1.view(-1, 3, 1).expand(shape=[-1, -1, num_cells])
    pos2 = pos2.view(-1, 3, 1).expand(shape=[-1, -1, num_cells])
    index1 = index1.view(-1, 1).repeat(1, num_cells).view(-1)
    index2 = index2.view(-1, 1).repeat(1, num_cells).view(-1)
    pos2 = pos2 + pbc_offsets_per_atom
    atom_distance_sqr = paddle.sum(x=(pos1 - pos2) ** 2, axis=1)
    if topk_per_pair is not None:
        assert topk_per_pair.shape[0] == num_atom_pairs
        atom_distance_sqr_sort_index = paddle.argsort(x=atom_distance_sqr, axis=1)
        assert tuple(atom_distance_sqr_sort_index.shape) == (num_atom_pairs, num_cells)
        atom_distance_sqr_sort_index = (
            atom_distance_sqr_sort_index
            + paddle.arange(end=num_atom_pairs)[:, None] * num_cells
        ).view(-1)
        topk_mask = paddle.arange(end=num_cells)[None, :] < topk_per_pair[:, None]
        topk_mask = topk_mask.view(-1)
        topk_indices = atom_distance_sqr_sort_index.masked_select(mask=topk_mask)
        topk_mask = paddle.zeros(shape=num_atom_pairs * num_cells)
        topk_mask.put_along_axis_(axis=0, indices=topk_indices, values=1.0)
        topk_mask = topk_mask.astype(dtype="bool")
    atom_distance_sqr = atom_distance_sqr.view(-1)
    mask_within_radius = paddle.less_equal(
        x=atom_distance_sqr, y=paddle.to_tensor(radius * radius, dtype="float32")
    )
    mask_not_same = paddle.greater_than(
        x=atom_distance_sqr, y=paddle.to_tensor(0.0001, dtype="float32")
    )
    mask = paddle.logical_and(x=mask_within_radius, y=mask_not_same)
    index1 = paddle.masked_select(x=index1, mask=mask)
    index2 = paddle.masked_select(x=index2, mask=mask)
    unit_cell = paddle.masked_select(
        x=unit_cell_per_atom.view(-1, 3), mask=mask.view(-1, 1).expand(shape=[-1, 3])
    )
    unit_cell = unit_cell.view(-1, 3)
    if topk_per_pair is not None:
        topk_mask = paddle.masked_select(x=topk_mask, mask=mask)
    num_neighbors = paddle.zeros(shape=len(cart_coords))
    num_neighbors.index_add_(axis=0, index=index1, value=paddle.ones(shape=len(index1)))
    num_neighbors = num_neighbors.astype(dtype="int64")
    max_num_neighbors = paddle.max(x=num_neighbors).astype(dtype="int64")
    _max_neighbors = copy.deepcopy(num_neighbors)
    _max_neighbors[
        _max_neighbors > max_num_neighbors_threshold
    ] = max_num_neighbors_threshold
    _num_neighbors = paddle.zeros(shape=len(cart_coords) + 1).astype(dtype="int64")
    _natoms = paddle.zeros(shape=tuple(num_atoms.shape)[0] + 1).astype(dtype="int64")
    _num_neighbors[1:] = paddle.cumsum(x=_max_neighbors, axis=0)
    _natoms[1:] = paddle.cumsum(x=num_atoms, axis=0)
    num_neighbors_image = _num_neighbors[_natoms[1:]] - _num_neighbors[_natoms[:-1]]
    if (
        max_num_neighbors <= max_num_neighbors_threshold
        or max_num_neighbors_threshold <= 0
    ):
        if topk_per_pair is None:
            return paddle.stack(x=(index2, index1)), unit_cell, num_neighbors_image
        else:
            return (
                paddle.stack(x=(index2, index1)),
                unit_cell,
                num_neighbors_image,
                topk_mask,
            )
    atom_distance_sqr = paddle.masked_select(x=atom_distance_sqr, mask=mask)
    distance_sort = paddle.zeros(shape=len(cart_coords) * max_num_neighbors).fill_(
        value=radius * radius + 1.0
    )
    index_neighbor_offset = paddle.cumsum(x=num_neighbors, axis=0) - num_neighbors
    index_neighbor_offset_expand = paddle.repeat_interleave(
        x=index_neighbor_offset, repeats=num_neighbors
    )
    index_sort_map = (
        index1 * max_num_neighbors
        + paddle.arange(end=len(index1))
        - index_neighbor_offset_expand
    )
    distance_sort.scatter_(index_sort_map, atom_distance_sqr)
    distance_sort = distance_sort.view(len(cart_coords), max_num_neighbors)
    distance_sort, index_sort = paddle.sort(x=distance_sort, axis=1), paddle.argsort(
        x=distance_sort, axis=1
    )
    distance_sort = distance_sort[:, :max_num_neighbors_threshold]
    index_sort = index_sort[:, :max_num_neighbors_threshold]
    index_sort = index_sort + index_neighbor_offset.view(-1, 1).expand(
        shape=[-1, max_num_neighbors_threshold]
    )
    mask_within_radius = paddle.less_equal(
        x=distance_sort, y=paddle.to_tensor(radius * radius, dtype="float32")
    )
    index_sort = paddle.masked_select(x=index_sort, mask=mask_within_radius)
    mask_num_neighbors = paddle.zeros(shape=len(index1)).astype(dtype="bool")
    mask_num_neighbors.index_fill_(axis=0, index=index_sort, value=True)
    index1 = paddle.masked_select(x=index1, mask=mask_num_neighbors)
    index2 = paddle.masked_select(x=index2, mask=mask_num_neighbors)
    unit_cell = paddle.masked_select(
        x=unit_cell.view(-1, 3),
        mask=mask_num_neighbors.view(-1, 1).expand(shape=[-1, 3]),
    )
    unit_cell = unit_cell.view(-1, 3)
    if topk_per_pair is not None:
        topk_mask = paddle.masked_select(x=topk_mask, mask=mask_num_neighbors)
    edge_index = paddle.stack(x=(index2, index1))
    if topk_per_pair is None:
        return edge_index, unit_cell, num_neighbors_image
    else:
        return edge_index, unit_cell, num_neighbors_image, topk_mask


def radius_graph_pbc_wrapper(
    frac_coords, lattices, num_atoms, radius, max_num_neighbors_threshold, device
):
    cart_coords = frac_to_cart_coords(
        frac_coords, num_atoms=num_atoms, lattices=lattices
    )
    return radius_graph_pbc(
        cart_coords, lattices, num_atoms, radius, max_num_neighbors_threshold, device
    )


def frac_to_cart_coords(
    frac_coords, num_atoms, lengths=None, angles=None, lattices=None
):
    assert (lengths is not None and angles is not None) or lattices is not None
    if lattices is None:
        lattices = lattice_params_to_matrix_paddle(lengths, angles)
    lattice_nodes = paddle.repeat_interleave(x=lattices, repeats=num_atoms, axis=0)
    pos = paddle.bmm(frac_coords.unsqueeze(1), lattice_nodes).squeeze(1)
    return pos


def frac_to_cart_coords_with_lattice(
    frac_coords: paddle.Tensor, num_atoms: paddle.Tensor, lattice: paddle.Tensor
) -> paddle.Tensor:
    lattice_nodes = paddle.repeat_interleave(x=lattice, repeats=num_atoms, axis=0)
    pos = paddle.bmm(frac_coords.unsqueeze(1), lattice_nodes).squeeze(1)
    return pos


def cart_to_frac_coords(
    cart_coords, num_atoms, lengths=None, angles=None, lattices=None
):
    assert (lengths is not None and angles is not None) or lattices is not None
    if lattices is None:
        lattices = lattice_params_to_matrix_paddle(lengths, angles)
    inv_lattice = paddle.linalg.pinv(x=lattices)
    inv_lattice_nodes = paddle.repeat_interleave(
        x=inv_lattice, repeats=num_atoms, axis=0
    )
    frac_coords = paddle.bmm(cart_coords.unsqueeze(1), inv_lattice_nodes).squeeze(1)
    return frac_coords % 1.0


def polar_decomposition(x):
    vecU, vals, vecV = paddle.linalg.svd(x)
    P = (
        vecV.transpose([0, 2, 1]).multiply(vals.view([vals.shape[0], 1, vals.shape[1]]))
        @ vecV
    )
    U = vecU @ vecV
    return U, P


def compute_lattice_polar_decomposition(lattice_matrix: paddle.Tensor) -> paddle.Tensor:
    W, S, V_transp = paddle.linalg.svd(full_matrices=True, x=lattice_matrix)
    S_square = paddle.diag_embed(input=S)
    V = V_transp.transpose(perm=dim2perm(V_transp.ndim, 1, 2))
    U = W @ V_transp
    P = V @ S_square @ V_transp
    P_prime = U @ P @ U.transpose(perm=dim2perm(U.ndim, 1, 2))
    symm_lattice_matrix = P_prime
    return symm_lattice_matrix

# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
ASU / Wyckoff math helpers: hull test, ASU wrapping, diffusion score computation.
"""
from typing import Dict
from typing import NamedTuple
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
import paddle
from pymatgen.core import Lattice
from pymatgen.core import Structure

from ppmat.models.sgequidiff.wyckoff_geometry import WyckoffGeometry
from ppmat.utils.crystal import OFFSET_LIST
from ppmat.utils.crystal import frac_to_cart_coords
from ppmat.utils.crystal import lattice_params_to_matrix_paddle
from ppmat.utils.scatter import scatter


def _segment_arg_extremum(
    src: paddle.Tensor,
    index: paddle.Tensor,
    segment_op,
    sentinel: float,
):
    """Per-group extreme value and its source position over 1-D segment ids.

    Float weights encode source positions; taking the extremum over the kept
    positions resolves ties deterministically (max -> last occurrence,
    min -> first occurrence). float32 represents integers exactly up to 2**24;
    larger inputs would lose arg-position precision.

    Returns (seg_size, empty_group_mask, per-group extreme, arg positions).
    """
    order = paddle.argsort(index, stable=True)
    sorted_index = index[order]
    sorted_src = src[order]
    seg_size = int(sorted_index.max().item()) + 1
    empty_mask = ~paddle.bincount(sorted_index, minlength=seg_size).cast(paddle.bool)

    extreme = segment_op(sorted_src, sorted_index)
    weights = paddle.arange(sorted_src.shape[0], dtype=paddle.float32)
    is_extreme = sorted_src == extreme[sorted_index]
    extreme_weights = paddle.where(is_extreme, weights, paddle.to_tensor(sentinel))
    arg_sorted = segment_op(extreme_weights, sorted_index)
    arg = order[arg_sorted.cast(paddle.int64)]
    return seg_size, empty_mask, extreme, arg


def _scatter_argmax(
    src: paddle.Tensor,
    index: paddle.Tensor,
    dim_size: int,
) -> paddle.Tensor:
    """Return the source index of the maximum value in each group.

    ``src`` and ``index`` must be one-dimensional with equal length. Empty
    groups are assigned ``0``. Ties resolve to the last occurrence in ``src``.
    """
    if src.ndim != 1 or index.ndim != 1 or src.shape[0] != index.shape[0]:
        raise ValueError("src and index must be one-dimensional with equal length")

    seg_size, empty_mask, _, argmax = _segment_arg_extremum(
        src, index, paddle.geometric.segment_max, -float("inf")
    )
    argmax = paddle.where(empty_mask, paddle.zeros_like(argmax), argmax)
    out = paddle.zeros([dim_size], dtype="int64")
    out[:seg_size] = argmax
    return out


def _scatter_min_indices(
    ov_row: paddle.Tensor,
    ov_col: paddle.Tensor,
    n_total: int,
) -> paddle.Tensor:
    """Map each row to its minimum overlapping column; rows without overlaps
    keep their own index."""
    if ov_row.shape[0] == 0:
        return paddle.arange(n_total, dtype=paddle.int64)
    # paddle.geometric.segment_min requires sorted segment ids; sort first.
    order = paddle.argsort(ov_row, stable=True)
    sorted_row = ov_row[order]
    sorted_col = ov_col[order]
    min_per_row = paddle.geometric.segment_min(
        sorted_col.cast(paddle.float32), sorted_row
    )
    unique_rows = paddle.unique(ov_row)
    result = paddle.arange(n_total, dtype=paddle.float32)
    result = paddle.scatter(
        result, unique_rows, min_per_row[unique_rows].cast(paddle.float32)
    )
    return result.cast(paddle.int64)


def primitive_lattice_matrix_from_conventional_lattice_params(
    space_group_indices: paddle.Tensor,
    wyckoff_geometry: "WyckoffGeometry",
    conventional_lattice_lengths: Optional[paddle.Tensor] = None,
    conventional_lattice_angles: Optional[paddle.Tensor] = None,
    conventional_lattice_matrix: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    """Compute primitive lattice matrix from conventional lattice params."""
    if conventional_lattice_matrix is None:
        conventional_lattice_matrix = lattice_params_to_matrix_paddle(
            conventional_lattice_lengths, conventional_lattice_angles
        )
    P_matrices = wyckoff_geometry.conventional_to_primitive_P_matrices[
        space_group_indices
    ]
    return paddle.bmm(P_matrices, conventional_lattice_matrix)


def _deduplicate_orbit_coords(
    coords: paddle.Tensor,
    orbit_sizes: paddle.Tensor,
    return_inverse: bool = False,
):
    orbit_sizes_sqr = (orbit_sizes**2).cast(paddle.int64)
    first_idx = paddle.cumsum(orbit_sizes, axis=0) - orbit_sizes
    first_expand = paddle.repeat_interleave(first_idx, orbit_sizes_sqr)
    orbit_expand = paddle.repeat_interleave(orbit_sizes, orbit_sizes_sqr)

    offset = paddle.repeat_interleave(
        paddle.cumsum(orbit_sizes_sqr, axis=0) - orbit_sizes_sqr, orbit_sizes_sqr
    )
    # ``arange`` accepts a tensor ``end`` directly: no GPU sync is needed.
    pair_ids = paddle.arange(end=orbit_sizes_sqr.sum(), dtype=paddle.int64) - offset
    row_ids = paddle.floor_divide(pair_ids, orbit_expand) + first_expand
    col_ids = pair_ids % orbit_expand + first_expand

    row_coords = coords[row_ids]
    col_coords = coords[col_ids]
    overlapping = paddle.all(
        paddle.abs((row_coords - col_coords + 0.5) % 1.0 - 0.5) < 1e-6, axis=1
    )
    ov_col = col_ids[overlapping]
    ov_row = row_ids[overlapping]

    min_col_per_row = _scatter_min_indices(ov_row, ov_col, coords.shape[0])
    if return_inverse:
        return paddle.unique(min_col_per_row, return_inverse=True)
    return paddle.unique(min_col_per_row), None


class _ASUToPrimitiveResult(NamedTuple):
    """Named result of batched ASU-to-primitive conversion."""

    coords: paddle.Tensor
    element_indices: paddle.Tensor
    wyckoff_indices: paddle.Tensor
    num_nodes_per_crystal: paddle.Tensor
    lattice_matrix: paddle.Tensor
    map_prim_to_asu: paddle.Tensor
    node_is_original: Optional[paddle.Tensor]
    asu_frac_coords_of_prim_atoms: paddle.Tensor


class _SpaceGroupOpsResult(NamedTuple):
    """Named result of space-group operation expansion.

    Note: ``map_conventional_to_asu_atom`` maps the **raw** (pre-dedup)
    conventional atoms to ASU atoms; ``conventional_frac_coords`` and the
    other per-atom fields are already restricted to
    ``unique_non_overlapping_atom_indices``. Consumers that need a mapping
    aligned with the deduplicated atoms must index this field by
    ``unique_non_overlapping_atom_indices`` first.
    """

    A_inv_ops: paddle.Tensor
    inverse_indices: paddle.Tensor
    map_conventional_to_asu_atom: paddle.Tensor
    conventional_wyckoff_indices: paddle.Tensor
    conventional_element_indices: paddle.Tensor
    conventional_frac_coords: paddle.Tensor
    unique_non_overlapping_atom_indices: paddle.Tensor


def batched_convert_asu_frac_coords_to_primitive_cartesian_coords(
    asu_frac_coords: paddle.Tensor,
    asu_element_indices: paddle.Tensor,
    asu_wyckoff_indices: paddle.Tensor,
    n_coords_per_asu: paddle.Tensor,
    conventional_lattice_matrix: paddle.Tensor,
    space_group_indices: paddle.Tensor,
    wyckoff_geometry: "WyckoffGeometry",
    return_cartesian_coords: bool = True,
    return_node_is_original: bool = False,
    map_frac_coords_to_0_1_unit_cell: bool = True,
    get_primitive_cell: bool = True,
) -> _ASUToPrimitiveResult:
    """
    Batch expand ASU fractional coords to primitive cell Cartesian coords.
    """
    batch_size = n_coords_per_asu.shape[0]
    num_asu_nodes_per_crystal = n_coords_per_asu

    conventional_to_primitive_transformations = (
        wyckoff_geometry.conventional_to_primitive_invP_matrices[space_group_indices]
    )

    padded_gc_mats = wyckoff_geometry.padded_general_wyckoff_matrices[
        space_group_indices
    ]
    padded_gc_trans = wyckoff_geometry.padded_general_wyckoff_translations[
        space_group_indices
    ]
    padded_gc_mask = wyckoff_geometry.padded_general_wyckoff_ops_mask[
        space_group_indices
    ]
    general_wyckoff_multiplicity_per_crystal = padded_gc_mask.cast(paddle.int64).sum(
        axis=1
    )

    def _repeat_interleave_along_asu(tensor_B_X):
        if tensor_B_X.dtype == paddle.bool:
            return paddle.repeat_interleave(
                tensor_B_X.cast(paddle.int32), num_asu_nodes_per_crystal, axis=0
            ).cast(paddle.bool)
        return paddle.repeat_interleave(tensor_B_X, num_asu_nodes_per_crystal, axis=0)

    padded_gc_mats = _repeat_interleave_along_asu(padded_gc_mats)
    padded_gc_trans = _repeat_interleave_along_asu(padded_gc_trans)
    padded_gc_mask = _repeat_interleave_along_asu(padded_gc_mask)

    stacked_gc_mats = padded_gc_mats[padded_gc_mask].reshape([-1, 3, 3])
    stacked_gc_trans = padded_gc_trans[
        padded_gc_mask.unsqueeze(-1).unsqueeze(-1).expand_as(padded_gc_trans)
    ].reshape([-1, 1, 3])

    # orbit ASU atoms to conventional cell
    asu_multiplicity_per_asu_atom = (
        general_wyckoff_multiplicity_per_crystal.repeat_interleave(
            num_asu_nodes_per_crystal, axis=0
        )
    )
    asu_frac_coords_repeated = asu_frac_coords.repeat_interleave(
        asu_multiplicity_per_asu_atom, axis=0
    ).unsqueeze(1)

    conventional_frac_coords = (
        paddle.bmm(asu_frac_coords_repeated, stacked_gc_mats) + stacked_gc_trans
    )
    if map_frac_coords_to_0_1_unit_cell:
        conventional_frac_coords = conventional_frac_coords % 1.0

    if get_primitive_cell:
        stacked_c2p = paddle.repeat_interleave(
            conventional_to_primitive_transformations,
            num_asu_nodes_per_crystal * general_wyckoff_multiplicity_per_crystal,
            axis=0,
        )
        primitive_frac_coords = paddle.bmm(
            conventional_frac_coords, stacked_c2p
        ).squeeze(1)
    else:
        primitive_frac_coords = conventional_frac_coords.reshape([-1, 3])

    if map_frac_coords_to_0_1_unit_cell:
        primitive_frac_coords = primitive_frac_coords % 1.0

    map_node_to_crystal = paddle.arange(batch_size).repeat_interleave(
        num_asu_nodes_per_crystal * general_wyckoff_multiplicity_per_crystal, axis=0
    )

    if return_node_is_original:
        first_asu_atom_index_per_orbit = (
            paddle.cumsum(asu_multiplicity_per_asu_atom, axis=0)
            - asu_multiplicity_per_asu_atom
        )
        node_is_original = paddle.zeros(
            [primitive_frac_coords.shape[0]], dtype=paddle.bool
        )
        node_is_original[first_asu_atom_index_per_orbit] = True

    unique_non_overlapping_atom_indices, _ = _deduplicate_orbit_coords(
        primitive_frac_coords, asu_multiplicity_per_asu_atom
    )
    primitive_frac_coords = primitive_frac_coords[unique_non_overlapping_atom_indices]
    map_node_to_crystal = map_node_to_crystal[unique_non_overlapping_atom_indices]
    if return_node_is_original:
        node_is_original = node_is_original[unique_non_overlapping_atom_indices]

    num_prim_nodes_per_crystal = scatter(
        src=paddle.ones([map_node_to_crystal.shape[0]], dtype=paddle.int64),
        index=map_node_to_crystal,
        dim_size=batch_size,
        reduce="sum",
    )

    map_prim_to_asu = paddle.arange(asu_frac_coords.shape[0]).repeat_interleave(
        asu_multiplicity_per_asu_atom, axis=0
    )[unique_non_overlapping_atom_indices]
    primitive_wyckoff_indices = asu_wyckoff_indices[map_prim_to_asu]
    primitive_element_indices = asu_element_indices[map_prim_to_asu]
    asu_frac_coords_of_prim_atoms = asu_frac_coords[map_prim_to_asu]

    primitive_lattice_matrix = None
    if return_cartesian_coords or get_primitive_cell:
        primitive_lattice_matrix = (
            primitive_lattice_matrix_from_conventional_lattice_params(
                space_group_indices=space_group_indices,
                wyckoff_geometry=wyckoff_geometry,
                conventional_lattice_matrix=conventional_lattice_matrix,
            )
        )

    if return_cartesian_coords:
        out_coords = frac_to_cart_coords(
            primitive_frac_coords,
            num_prim_nodes_per_crystal,
            lattices=primitive_lattice_matrix,
        )
    else:
        out_coords = primitive_frac_coords

    return _ASUToPrimitiveResult(
        coords=out_coords,
        element_indices=primitive_element_indices,
        wyckoff_indices=primitive_wyckoff_indices,
        num_nodes_per_crystal=num_prim_nodes_per_crystal,
        lattice_matrix=primitive_lattice_matrix,
        map_prim_to_asu=map_prim_to_asu,
        node_is_original=node_is_original if return_node_is_original else None,
        asu_frac_coords_of_prim_atoms=asu_frac_coords_of_prim_atoms,
    )


def atoms_are_in_hull(
    supercell_frac_coords: paddle.Tensor,
    hull_equations: paddle.Tensor,
    epsilon: float = -1e-5,
) -> paddle.Tensor:
    """
    Check if each point in supercell is inside Wyckoff shape hull.
    """
    n_atoms, n_images, _ = supercell_frac_coords.shape
    n_shapes = hull_equations.shape[1]
    n_bounds = hull_equations.shape[2]

    coords_flat = supercell_frac_coords.reshape([-1, 3])
    hulls_expanded = (
        hull_equations.unsqueeze(1)
        .expand([n_atoms, n_images, n_shapes, n_bounds, 4])
        .reshape([n_atoms * n_images, n_shapes, n_bounds, 4])
    )

    normals = hulls_expanded[..., :3]
    offsets = hulls_expanded[..., 3]

    # (n*img, 1, 1, 3) @ (n*img, n_shapes, n_bounds, 3) -> sum
    dots = (coords_flat[:, None, None, :] * normals).sum(axis=-1)

    inside = (dots < -offsets - epsilon).all(axis=-1)
    return inside.reshape([n_atoms, n_images, n_shapes])


@paddle.no_grad()
def wrap_frac_coords_into_asu(
    frac_coords: paddle.Tensor,
    wyckoff_indices: paddle.Tensor,
    space_group_indices: paddle.Tensor,
    num_atoms_per_asu: paddle.Tensor,
    hull_equations: paddle.Tensor,
    hull_equations_mask: paddle.Tensor,
    wyckoff_geometry: "WyckoffGeometry",
) -> Tuple[paddle.Tensor, paddle.Tensor]:
    """
    Map noisy fractional coordinates back to canonical ASU via group operations.
    """
    n_asu_atoms = frac_coords.shape[0]
    frac_coords = frac_coords % 1.0

    conversion_result = batched_convert_asu_frac_coords_to_primitive_cartesian_coords(
        asu_frac_coords=frac_coords,
        asu_element_indices=paddle.zeros_like(wyckoff_indices),
        asu_wyckoff_indices=wyckoff_indices,
        n_coords_per_asu=num_atoms_per_asu,
        conventional_lattice_matrix=paddle.zeros(
            [space_group_indices.shape[0], 3, 3], dtype=paddle.float32
        ),
        space_group_indices=space_group_indices,
        wyckoff_geometry=wyckoff_geometry,
        return_cartesian_coords=False,
        get_primitive_cell=False,
    )
    conventional_cell_frac_coords = conversion_result.coords
    map_conventional_to_asu_coord = conversion_result.map_prim_to_asu

    supercell_frac_translations = paddle.to_tensor(OFFSET_LIST, dtype=paddle.float32)
    supercell_frac_coords = conventional_cell_frac_coords.unsqueeze(
        1
    ) + supercell_frac_translations.unsqueeze(0)
    hull_eq_expanded = hull_equations[map_conventional_to_asu_coord]
    wyckoff_shape_exists = hull_equations_mask[map_conventional_to_asu_coord]

    supercell_in_wyckoff = atoms_are_in_hull(
        supercell_frac_coords, hull_eq_expanded, epsilon=-1e-5
    ) & wyckoff_shape_exists.unsqueeze(1)

    supercell_in_any_wyckoff = supercell_in_wyckoff.any(axis=-1)
    conv_atom_has_image_in_asu = supercell_in_any_wyckoff.any(axis=-1)

    indices_of_conv_atoms_in_asu = _scatter_argmax(
        src=conv_atom_has_image_in_asu.cast(paddle.float32),
        index=map_conventional_to_asu_coord,
        dim_size=n_asu_atoms,
    )

    supercell_frac_coords_in_asu = supercell_frac_coords[indices_of_conv_atoms_in_asu]
    asu_atom_is_inside = supercell_in_wyckoff[indices_of_conv_atoms_in_asu]
    supercell_in_any_in_asu = supercell_in_any_wyckoff[indices_of_conv_atoms_in_asu]

    lattice_translation_into_asu_idx = paddle.argmax(
        supercell_in_any_in_asu.cast(paddle.float32), axis=-1
    )

    atom_indices = paddle.arange(n_asu_atoms)
    wrapped_asu_frac_coords = supercell_frac_coords_in_asu[
        atom_indices, lattice_translation_into_asu_idx
    ]
    wrapped_asu_wyckoff_shape_indices = paddle.argmax(
        asu_atom_is_inside[atom_indices, lattice_translation_into_asu_idx].cast(
            paddle.float32
        ),
        axis=-1,
    )

    return wrapped_asu_frac_coords, wrapped_asu_wyckoff_shape_indices


@paddle.no_grad()
def p_asu_wrapped_normal(
    noisy_frac_coord: paddle.Tensor,
    conventional_frac_coords: paddle.Tensor,
    map_conventional_to_asu_frac_coords: paddle.Tensor,
    n_lattice_translations: int = 5,
    sigma: Union[float, paddle.Tensor] = 1.0,
) -> paddle.Tensor:
    """Compute isotropic Gaussian sum over equivalent positions in ASU.
    Without prefactor 1/(2pi*sigma).
    """
    t = paddle.arange(
        -n_lattice_translations, n_lattice_translations + 1, dtype=paddle.float32
    )
    translations = paddle.stack(
        paddle.meshgrid(t, t, t, indexing="ij"), axis=-1
    ).reshape([-1, 3])

    noisy_x_minus_gt = noisy_frac_coord[map_conventional_to_asu_frac_coords].unsqueeze(
        1
    ) - (conventional_frac_coords.unsqueeze(1) + translations.unsqueeze(0))

    diff_sq = (noisy_x_minus_gt**2).sum(axis=-1)
    gaussian = paddle.exp(-diff_sq / (2 * sigma**2))
    p = gaussian.sum(axis=1)

    p_asu = scatter(
        src=p,
        index=map_conventional_to_asu_frac_coords,
        dim=0,
        dim_size=noisy_frac_coord.shape[0],
        reduce="sum",
    )
    return p_asu


@paddle.no_grad()
def d_log_p_asu_wrapped_normal(
    noisy_frac_coord: paddle.Tensor,
    conventional_frac_coords: paddle.Tensor,
    map_conventional_to_asu_frac_coords: paddle.Tensor,
    n_lattice_translations: int = 5,
    sigma: Union[float, paddle.Tensor] = 1.0,
) -> paddle.Tensor:
    """
    Compute gradient of log ASU-wrapped normal probability (i.e. ground truth score).
    """
    if not isinstance(sigma, paddle.Tensor):
        sigma_conv = paddle.full([conventional_frac_coords.shape[0], 1], sigma)
    else:
        sigma_conv = sigma[map_conventional_to_asu_frac_coords].unsqueeze(1)

    t = paddle.arange(
        -n_lattice_translations, n_lattice_translations + 1, dtype=paddle.float32
    )
    translations = paddle.stack(
        paddle.meshgrid(t, t, t, indexing="ij"), axis=-1
    ).reshape([-1, 3])

    noisy_x_minus_gt = noisy_frac_coord[map_conventional_to_asu_frac_coords].unsqueeze(
        1
    ) - (conventional_frac_coords.unsqueeze(1) + translations.unsqueeze(0))

    diff_sq = (noisy_x_minus_gt**2).sum(axis=-1, keepdim=True)
    gaussian = paddle.exp(-diff_sq / (2 * sigma_conv.unsqueeze(-1) ** 2))
    numerator = -(gaussian * noisy_x_minus_gt).sum(axis=1)

    n_asu_atoms = noisy_frac_coord.shape[0]
    flat_idx = (
        map_conventional_to_asu_frac_coords.unsqueeze(1) * 3
        + paddle.arange(3, dtype=paddle.int64)
    ).reshape([-1])
    numerator_asu = scatter(
        src=numerator.reshape([-1]),
        index=flat_idx,
        dim=0,
        dim_size=n_asu_atoms * 3,
        reduce="sum",
    ).reshape([n_asu_atoms, 3])

    denominator = (
        p_asu_wrapped_normal(
            noisy_frac_coord,
            conventional_frac_coords,
            map_conventional_to_asu_frac_coords,
            n_lattice_translations,
            sigma_conv,
        )
        * sigma**2
    )
    return numerator_asu / denominator.unsqueeze(-1)


def get_space_group_ops_and_conventional_atoms(
    frac_coords: paddle.Tensor,
    element_indices: paddle.Tensor,
    wyckoff_indices: paddle.Tensor,
    space_group_indices: paddle.Tensor,
    n_atoms_per_xtal: paddle.Tensor,
    wyckoff_geometry: "WyckoffGeometry",
) -> tuple:
    """Get space group operations (mod lattice translation) and
    de-duplicated conventional cell atoms.
    """
    padded_gc_mats = wyckoff_geometry.padded_general_wyckoff_matrices[
        space_group_indices
    ]
    padded_gc_inv_mats = wyckoff_geometry.padded_inverse_general_wyckoff_matrices[
        space_group_indices
    ]
    padded_gc_trans = wyckoff_geometry.padded_general_wyckoff_translations[
        space_group_indices
    ]
    padded_gc_mask = wyckoff_geometry.padded_general_wyckoff_ops_mask[
        space_group_indices
    ]

    general_wyckoff_multiplicity = padded_gc_mask.cast(paddle.int64).sum(axis=1)

    padded_gc_mats = paddle.repeat_interleave(padded_gc_mats, n_atoms_per_xtal, axis=0)
    padded_gc_inv_mats = paddle.repeat_interleave(
        padded_gc_inv_mats, n_atoms_per_xtal, axis=0
    )
    padded_gc_trans = paddle.repeat_interleave(
        padded_gc_trans, n_atoms_per_xtal, axis=0
    )
    padded_gc_mask = paddle.repeat_interleave(
        padded_gc_mask.cast(paddle.int32), n_atoms_per_xtal, axis=0
    ).cast(paddle.bool)

    stacked_mats = padded_gc_mats[padded_gc_mask].reshape([-1, 3, 3])
    stacked_inv_mats = padded_gc_inv_mats[padded_gc_mask].reshape([-1, 3, 3])
    stacked_trans = padded_gc_trans[
        padded_gc_mask.unsqueeze(-1).unsqueeze(-1).expand_as(padded_gc_trans)
    ].reshape([-1, 1, 3])

    mult_per_asu_atom = general_wyckoff_multiplicity.repeat_interleave(
        n_atoms_per_xtal, axis=0
    )

    frac_coords_repeated = frac_coords.repeat_interleave(
        mult_per_asu_atom, axis=0
    ).unsqueeze(1)

    conventional_frac_coords_with_dupes = (
        paddle.bmm(frac_coords_repeated, stacked_mats) + stacked_trans
    ) % 1.0
    conventional_frac_coords = conventional_frac_coords_with_dupes.reshape([-1, 3])

    unique_non_overlapping_atom_indices, inverse_indices = _deduplicate_orbit_coords(
        conventional_frac_coords, mult_per_asu_atom, return_inverse=True
    )
    conventional_frac_coords = conventional_frac_coords[
        unique_non_overlapping_atom_indices
    ]

    map_conv_to_asu_with_dupes = paddle.arange(frac_coords.shape[0]).repeat_interleave(
        mult_per_asu_atom, axis=0
    )
    map_conventional_to_asu_atom = map_conv_to_asu_with_dupes[
        unique_non_overlapping_atom_indices
    ]
    conventional_wyckoff_indices = wyckoff_indices[map_conventional_to_asu_atom]
    conventional_element_indices = element_indices[map_conventional_to_asu_atom]

    return _SpaceGroupOpsResult(
        A_inv_ops=stacked_inv_mats,
        inverse_indices=inverse_indices,
        map_conventional_to_asu_atom=map_conv_to_asu_with_dupes,
        conventional_wyckoff_indices=conventional_wyckoff_indices,
        conventional_element_indices=conventional_element_indices,
        conventional_frac_coords=conventional_frac_coords,
        unique_non_overlapping_atom_indices=unique_non_overlapping_atom_indices,
    )


def get_wyckoff_projected_gaussian_noise(
    space_group_indices: paddle.Tensor,
    wyckoff_indices: paddle.Tensor,
    wyckoff_shape_indices: paddle.Tensor,
    n_atoms_per_xtal: paddle.Tensor,
    sigma: Union[float, paddle.Tensor],
    wyckoff_geometry: "WyckoffGeometry",
) -> paddle.Tensor:
    """
    Generate Gaussian noise projected onto Wyckoff subspace.
    """
    n_asu_atoms = wyckoff_indices.shape[0]
    unprojected_noise = sigma * paddle.randn([n_asu_atoms, 3])
    sg_per_atom = space_group_indices.repeat_interleave(n_atoms_per_xtal, axis=0)
    projection_matrices = wyckoff_geometry.noise_projection_matrices[
        sg_per_atom, wyckoff_indices, wyckoff_shape_indices
    ]
    projected_noise = paddle.bmm(
        unprojected_noise.unsqueeze(1), projection_matrices
    ).reshape([-1, 3])
    return projected_noise


def get_orbit_from_single_asu_position(
    asu_frac_coord: paddle.Tensor,
    space_group_number: int,
    wyckoff_geometry: "WyckoffGeometry",
) -> paddle.Tensor:
    """Expand one ASU fractional coordinate into its full Wyckoff orbit.

    Args:
        asu_frac_coord: shape (3,) fractional coordinate inside the ASU.
        space_group_number: 1-indexed space group number in [1, 230].

    Returns:
        shape (multiplicity, 3) orbit coordinates. Special (non-general)
        Wyckoff positions automatically collapse to their multiplicity via
        periodic de-duplication.
    """
    sg_idx = space_group_number - 1
    rotations = wyckoff_geometry.padded_general_wyckoff_matrices[sg_idx]
    translations = wyckoff_geometry.padded_general_wyckoff_translations[sg_idx].reshape(
        [-1, 3]
    )
    ops_mask = wyckoff_geometry.padded_general_wyckoff_ops_mask[sg_idx]
    rotations = rotations[ops_mask]
    translations = translations[ops_mask]

    coord = asu_frac_coord % 1.0
    orbit = (
        coord.unsqueeze(0).unsqueeze(0) @ rotations + translations.unsqueeze(1)
    ).reshape([-1, 3]) % 1.0

    # De-duplicate orbits generated for special (non-general) Wyckoff
    # positions: the general-position orbit contains repeated sites that map
    # onto the same atom under the special position constraints.
    d = paddle.abs((orbit.unsqueeze(1) - orbit.unsqueeze(0) + 0.5) % 1.0 - 0.5)
    duplicates_adjacency = paddle.all(d < 1e-4, axis=-1)
    first_representative = paddle.argmax(
        duplicates_adjacency.cast(paddle.float32), axis=1
    )
    unique_atom_idxs = paddle.unique(first_representative)
    return orbit[unique_atom_idxs]


def asu_to_pymatgen_structure(
    asu: Dict, wyckoff_geometry: "WyckoffGeometry"
) -> Structure:
    """Expand an ASU crystal into a full conventional-cell pymatgen Structure."""
    lengths = asu["conventional_lattice_lengths"].reshape([1, 3])
    angles = asu["conventional_lattice_angles"].reshape([1, 3])
    lattice_matrix = lattice_params_to_matrix_paddle(lengths, angles)[0]
    pmg_lattice = Lattice(matrix=lattice_matrix.numpy())

    conventional_frac_coords = []
    conventional_atomic_numbers = []
    for frac_coord, atom_type in zip(
        asu["conventional_frac_coords"], asu["element_indices"]
    ):
        orbit = get_orbit_from_single_asu_position(
            frac_coord,
            space_group_number=int(asu["space_group_number"]),
            wyckoff_geometry=wyckoff_geometry,
        )
        conventional_frac_coords.append(orbit.numpy())
        conventional_atomic_numbers.extend([int(atom_type) + 1] * orbit.shape[0])

    structure = Structure(
        lattice=pmg_lattice,
        species=conventional_atomic_numbers,
        coords=np.concatenate(conventional_frac_coords, axis=0),
        to_unit_cell=True,
        coords_are_cartesian=False,
    )
    return structure

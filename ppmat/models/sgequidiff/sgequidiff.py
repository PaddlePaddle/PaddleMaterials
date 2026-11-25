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

"""Space-group equivariant crystal generation and its numerical components."""

from __future__ import annotations

import hashlib
import json
import math
import pickle
from typing import Any
from typing import Dict
from typing import List
from typing import NamedTuple
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import pgl
from paddle.distribution import Categorical
from tqdm import tqdm

from ppmat.datasets.build_asu import ELEMENT_ENCODING_SIZE
from ppmat.datasets.build_asu import MAX_WYCKOFF_POSITIONS
from ppmat.datasets.build_asu import NUM_CRYSTALLOGRAPHIC_SPACE_GROUPS
from ppmat.datasets.build_asu import NUM_LATTICE_PARAMS
from ppmat.datasets.build_asu import EmbeddingTools
from ppmat.datasets.build_asu import WyckoffGeometry
from ppmat.datasets.build_asu import build_embedding_tools
from ppmat.datasets.build_asu import build_wyckoff_geometry
from ppmat.datasets.build_asu import chemical_symbols
from ppmat.datasets.build_asu import get_data_directory
from ppmat.datasets.build_asu import lattice_parameter_ranges
from ppmat.datasets.build_asu import load_wyckoff_shapes
from ppmat.datasets.build_asu import max_atoms_per_dataset
from ppmat.datasets.build_asu import save_asu_cache
from ppmat.models.common.activation import ScaledSiLU as Swish
from ppmat.models.common.radial_basis import GaussianSmearing
from ppmat.models.common.runtime import RuntimeMixin
from ppmat.models.common.runtime import runtime_boundary
from ppmat.models.common.time_embedding import SinusoidalTimeEmbeddings
from ppmat.schedulers import build_scheduler
from ppmat.schedulers.scheduling_asu_ve_sde import build_ve_sigma_grid
from ppmat.utils import logger
from ppmat.utils.crystal import OFFSET_LIST
from ppmat.utils.crystal import frac_to_cart_coords
from ppmat.utils.crystal import get_pbc_distances
from ppmat.utils.crystal import lattice_params_to_matrix_paddle
from ppmat.utils.scatter import scatter
from ppmat.utils.scatter import scatter_min


def sample_point_in_asu_wyckoff_site(
    space_group_numbers: List[str],
    wyckoff_letters: List[str],
    dictionary_of_wyckoffs_in_asu: dict,
    dictionary_of_wyckoff_shape_decompositions: dict,
    hull_equations_3d: paddle.Tensor,
    n_samples_per_wyckoff: int = 1,
) -> Tuple[paddle.Tensor, paddle.Tensor]:
    """Uniformly sample points in ASU Wyckoff sites."""
    random_samples_in_wyckoffs = []
    sampled_wyckoff_shape_indices = []
    for space_group_number, wyckoff_letter in zip(space_group_numbers, wyckoff_letters):
        wyckoff_position_dict = dictionary_of_wyckoffs_in_asu[space_group_number][
            wyckoff_letter
        ]
        wyckoff_dof = int(wyckoff_position_dict["dim"])

        if wyckoff_dof == 1:
            lengths = paddle.to_tensor(
                dictionary_of_wyckoff_shape_decompositions[space_group_number][
                    wyckoff_letter
                ]["volumes"],
                dtype=paddle.float32,
            )
            vertices = paddle.to_tensor(
                wyckoff_position_dict["vertices"].astype("float32")
            )
            sampled_line_index = paddle.multinomial(
                lengths, num_samples=n_samples_per_wyckoff, replacement=True
            )
            vertices = vertices[sampled_line_index]
            sampled_wyckoff_shape_index = sampled_line_index

        elif wyckoff_dof == 2:
            wyckoff_shapes_decomp_dict = dictionary_of_wyckoff_shape_decompositions[
                space_group_number
            ][wyckoff_letter]
            facet_areas = paddle.to_tensor(
                wyckoff_shapes_decomp_dict["volumes"], dtype=paddle.float32
            )
            sampled_facet_idxs = paddle.multinomial(
                facet_areas, num_samples=n_samples_per_wyckoff, replacement=True
            )
            max_num_triangles_per_facet = wyckoff_shapes_decomp_dict[
                "max_triangles_per_facet"
            ]
            _sampled_facet_triangle_areas = [
                paddle.to_tensor(
                    wyckoff_shapes_decomp_dict["facet_triangle_areas"][int(fid)],
                    dtype=paddle.float32,
                )
                for fid in sampled_facet_idxs
            ]
            sampled_facet_triangle_areas = paddle.zeros(
                [n_samples_per_wyckoff, max_num_triangles_per_facet]
            )
            for j in range(n_samples_per_wyckoff):
                n = _sampled_facet_triangle_areas[j].shape[0]
                sampled_facet_triangle_areas[j, :n] = _sampled_facet_triangle_areas[j]
            sampled_triangle_idxs = paddle.multinomial(
                sampled_facet_triangle_areas, num_samples=1
            ).squeeze(axis=1)
            vertices = paddle.stack(
                [
                    paddle.to_tensor(
                        wyckoff_shapes_decomp_dict["facet_triangles"][int(fid)][
                            int(tid)
                        ],
                        dtype=paddle.float32,
                    )
                    for fid, tid in zip(sampled_facet_idxs, sampled_triangle_idxs)
                ],
                axis=0,
            )
            sampled_wyckoff_shape_index = sampled_facet_idxs

        else:
            vertices = wyckoff_position_dict["vertices_tensor"]
            sampled_wyckoff_shape_index = paddle.to_tensor(
                [0], dtype=paddle.int64
            ).expand([n_samples_per_wyckoff])

        h_eq = hull_equations_3d[int(space_group_number) - 1]
        sample_in_wyckoff = uniformly_sample_point_in_convex_shape(
            vertices=vertices,
            wyckoff_site_dimensionality=wyckoff_dof,
            n_samples=n_samples_per_wyckoff,
            hull_equations=h_eq,
        )

        random_samples_in_wyckoffs.append(sample_in_wyckoff)
        sampled_wyckoff_shape_indices.append(sampled_wyckoff_shape_index)

    random_samples_in_wyckoffs = paddle.stack(random_samples_in_wyckoffs, axis=0)
    sampled_wyckoff_shape_indices = paddle.stack(sampled_wyckoff_shape_indices, axis=0)
    return random_samples_in_wyckoffs, sampled_wyckoff_shape_indices


def is_inside(
    xs: paddle.Tensor,
    hull_equations: paddle.Tensor,
    epsilon: float = 1e-6,
) -> paddle.Tensor:
    """Check if points are inside a convex polytope."""
    results = xs @ hull_equations[:, :3].T < -hull_equations[:, 3][None, :] - epsilon
    return results.all(axis=-1)


def uniformly_sample_point_in_convex_shape(
    vertices: paddle.Tensor,
    wyckoff_site_dimensionality: int,
    n_samples: int = 1,
    hull_equations: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    """Uniformly sample points inside a convex shape."""
    if wyckoff_site_dimensionality == 0:
        if list(vertices.shape) != [1, 3]:
            raise ValueError(
                f"0D site expects vertices of shape [1, 3], got {vertices.shape}"
            )
        return vertices.expand([n_samples, 3])
    elif wyckoff_site_dimensionality == 1:
        if list(vertices.shape) != [n_samples, 2, 3]:
            raise ValueError(
                f"1D site expects vertices of shape [n, 2, 3], got {vertices.shape}"
            )
        samples = paddle.rand([n_samples, 1])
        ep1, ep2 = vertices[:, 0], vertices[:, 1]
        return samples * (ep2 - ep1) + ep1
    elif wyckoff_site_dimensionality == 2:
        if list(vertices.shape) != [n_samples, 3, 3]:
            raise ValueError(
                f"2D site expects vertices of shape [n, 3, 3], got {vertices.shape}"
            )
        r1_sqrt, r2 = paddle.rand([n_samples, 1]).sqrt(), paddle.rand([n_samples, 1])
        return (
            (1.0 - r1_sqrt) * vertices[:, 0]
            + r1_sqrt * (1.0 - r2) * vertices[:, 1]
            + r1_sqrt * r2 * vertices[:, 2]
        )
    elif wyckoff_site_dimensionality == 3:
        if hull_equations is None:
            raise ValueError("hull_equations required for 3D sampling")
        box_lower_left = paddle.min(vertices, axis=0)
        box_top_right = paddle.max(vertices, axis=0)
        accepted = []
        total = 0
        # Empirical rejection-sampling budget; exhaustion raises the
        # diagnostic RuntimeError below.
        max_iters = 1000
        batch_size = max(3 * n_samples, 1)
        for _ in range(max_iters):
            candidate = (
                paddle.rand([batch_size, 3]) * (box_top_right - box_lower_left)
                + box_lower_left
            )
            mask = is_inside(candidate, hull_equations)
            accepted.append(candidate[mask])
            total += mask.cast(paddle.int64).sum().item()
            if total >= n_samples:
                return paddle.concat(accepted, axis=0)[:n_samples]
            batch_size *= 2
        raise RuntimeError(
            f"Rejection sampling failed to collect {n_samples} points inside the "
            f"3D hull after {max_iters} iterations (collected {total}); "
            "check hull_equations / bounding box validity"
        )
    raise ValueError(f"Invalid dimensionality: {wyckoff_site_dimensionality}")


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


def expand_asu_frac_coords_to_conventional_coords(
    asu_frac_coords: paddle.Tensor,
    n_coords_per_asu: paddle.Tensor,
    space_group_indices: paddle.Tensor,
    wyckoff_geometry: "WyckoffGeometry",
) -> Tuple[paddle.Tensor, paddle.Tensor]:
    """Expand ASU fractional coordinates into de-duplicated conventional-cell
    fractional coordinates, returning them with the conventional-to-ASU atom map.
    """
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
                tensor_B_X.cast(paddle.int32), n_coords_per_asu, axis=0
            ).cast(paddle.bool)
        return paddle.repeat_interleave(tensor_B_X, n_coords_per_asu, axis=0)

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
            n_coords_per_asu, axis=0
        )
    )
    asu_frac_coords_repeated = asu_frac_coords.repeat_interleave(
        asu_multiplicity_per_asu_atom, axis=0
    ).unsqueeze(1)

    conventional_frac_coords = (
        paddle.bmm(asu_frac_coords_repeated, stacked_gc_mats) + stacked_gc_trans
    ) % 1.0
    conventional_frac_coords = conventional_frac_coords.reshape([-1, 3])

    unique_non_overlapping_atom_indices, _ = _deduplicate_orbit_coords(
        conventional_frac_coords, asu_multiplicity_per_asu_atom
    )
    conventional_frac_coords = conventional_frac_coords[
        unique_non_overlapping_atom_indices
    ]

    map_conventional_to_asu = paddle.arange(asu_frac_coords.shape[0]).repeat_interleave(
        asu_multiplicity_per_asu_atom, axis=0
    )[unique_non_overlapping_atom_indices]
    return conventional_frac_coords, map_conventional_to_asu


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

    conversion_result = expand_asu_frac_coords_to_conventional_coords(
        asu_frac_coords=frac_coords,
        n_coords_per_asu=num_atoms_per_asu,
        space_group_indices=space_group_indices,
        wyckoff_geometry=wyckoff_geometry,
    )
    conventional_cell_frac_coords, map_conventional_to_asu_coord = conversion_result

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


def expand_asu(asu: Dict, wyckoff_geometry: WyckoffGeometry) -> Dict:
    """Expand Wyckoff orbits into the conventional-cell array format."""
    coords = []
    atom_types = []
    for frac_coord, atom_type in zip(
        asu["conventional_frac_coords"], asu["element_indices"]
    ):
        orbit = get_orbit_from_single_asu_position(
            frac_coord,
            space_group_number=int(asu["space_group_number"]),
            wyckoff_geometry=wyckoff_geometry,
        )
        coords.append(orbit.numpy())
        atom_types.extend([int(atom_type) + 1] * orbit.shape[0])
    return {
        "num_atoms": len(atom_types),
        "atom_types": atom_types,
        "frac_coords": np.concatenate(coords, axis=0).tolist(),
        "lengths": asu["conventional_lattice_lengths"].reshape([-1]).tolist(),
        "angles": asu["conventional_lattice_angles"].reshape([-1]).tolist(),
    }


class SGEQuiDiffGraphConverter:
    """Build minimum-image PGL graphs while retaining symmetry-related ties."""

    def __call__(self, structure):
        return self.build_graph(
            paddle.to_tensor(structure.cart_coords, dtype="float32"),
            paddle.to_tensor(structure.lattice.matrix[None], dtype="float32"),
            paddle.to_tensor([len(structure)], dtype="int64"),
        )

    @staticmethod
    def build_graph(
        cart_coords: paddle.Tensor,
        lattice: paddle.Tensor,
        num_atoms: paddle.Tensor,
    ) -> pgl.Graph:
        """Build a periodic-boundary graph over all atom pairs and supercell images.

        For every ordered atom pair inside each crystal, edges to all 26
        supercell image translations are materialized; per (dst, src) node pair
        only the minimum-distance image is kept (periodic self-images of an atom
        with itself are kept when their distance exceeds a numerical tolerance).
        Symmetry-equivalent images whose distance ties within a tolerance are
        all kept, which preserves the Wyckoff symmetry required by the GNN
        drift module.

        Args:
            cart_coords: Packed Cartesian coordinates, shape ``[n_atoms, 3]``.
            lattice: Lattice matrices, shape ``[n_crystals, 3, 3]``.
            num_atoms: Atoms per crystal, shape ``[n_crystals]``.

        Returns:
            PGL graph with source-to-destination edges and periodic image offsets.
        """
        batch_size = num_atoms.shape[0]
        atom_pos = cart_coords

        num_atoms_per_crystal_sqr = (num_atoms**2).cast(paddle.int64)

        first_node_index_per_crystal = paddle.cumsum(num_atoms, axis=0) - num_atoms
        first_node_index_per_crystal_expand = paddle.repeat_interleave(
            first_node_index_per_crystal, num_atoms_per_crystal_sqr
        )
        num_atoms_per_crystal_expand = paddle.repeat_interleave(
            num_atoms, num_atoms_per_crystal_sqr
        )

        num_atom_pairs = paddle.sum(num_atoms_per_crystal_sqr)
        index_sqr_offset = (
            paddle.cumsum(num_atoms_per_crystal_sqr, axis=0) - num_atoms_per_crystal_sqr
        )
        index_sqr_offset = paddle.repeat_interleave(
            index_sqr_offset, num_atoms_per_crystal_sqr
        )
        atom_count_sqr = paddle.arange(end=num_atom_pairs, dtype=paddle.int64) - (
            index_sqr_offset
        )

        destination_index = (
            paddle.floor_divide(atom_count_sqr, num_atoms_per_crystal_expand)
            + first_node_index_per_crystal_expand
        )
        source_index = (
            atom_count_sqr % num_atoms_per_crystal_expand
            + first_node_index_per_crystal_expand
        )
        map_edge_to_crystal = paddle.arange(batch_size).repeat_interleave(
            num_atoms_per_crystal_sqr, axis=0
        )

        n_edges_per_crystal_before_masking = num_atoms_per_crystal_sqr

        source_position = atom_pos[source_index]
        destination_position = atom_pos[destination_index]

        num_supercell_images = len(OFFSET_LIST)
        supercell_frac_offsets = paddle.to_tensor(OFFSET_LIST, dtype=paddle.float32)
        batch_supercell_frac_offsets = supercell_frac_offsets.unsqueeze(0).expand(
            [batch_size, num_supercell_images, 3]
        )
        pbc_frac_offsets_per_source_atom = paddle.repeat_interleave(
            batch_supercell_frac_offsets, n_edges_per_crystal_before_masking, axis=0
        )

        pbc_cart_offsets_per_source_atom = paddle.bmm(
            pbc_frac_offsets_per_source_atom,
            paddle.repeat_interleave(
                lattice, n_edges_per_crystal_before_masking, axis=0
            ),
        )

        destination_position = destination_position.unsqueeze(1).expand(
            [-1, num_supercell_images, -1]
        )
        source_position = (
            source_position.unsqueeze(1).expand([-1, num_supercell_images, -1])
            + pbc_cart_offsets_per_source_atom
        )

        source_index = source_index.unsqueeze(1).expand([-1, num_supercell_images])
        destination_index = destination_index.unsqueeze(1).expand(
            [-1, num_supercell_images]
        )
        map_edge_to_crystal = map_edge_to_crystal.unsqueeze(1).expand(
            [-1, num_supercell_images]
        )

        inter_atom_distances = (source_position - destination_position).norm(axis=-1)

        mask = paddle.logical_or(
            source_index != destination_index,
            inter_atom_distances > 1e-5,
        )

        flat_mask = mask.reshape([-1])
        destination_index = destination_index[mask]
        source_index = source_index[mask]
        map_edge_to_crystal = map_edge_to_crystal[mask]
        pbc_frac_offsets = pbc_frac_offsets_per_source_atom.reshape([-1, 3])[flat_mask]
        pbc_cart_offsets = pbc_cart_offsets_per_source_atom.reshape([-1, 3])[flat_mask]

        source_position_flat = atom_pos[source_index]
        destination_position_flat = atom_pos[destination_index]
        inter_atom_distances = (
            destination_position_flat - source_position_flat - pbc_cart_offsets
        ).norm(axis=-1)

        (
            source_index,
            destination_index,
            source_node_image_offsets,
            num_edges_per_crystal,
        ) = _get_smallest_edge_per_node_pair(
            dst_idx=destination_index,
            src_idx=source_index,
            map_edge_to_crystal=map_edge_to_crystal,
            inter_atom_distances=inter_atom_distances,
            pbc_frac_offsets_per_source_atom=pbc_frac_offsets,
            num_atoms=num_atoms,
        )
        return pgl.Graph(
            edges=paddle.stack([source_index, destination_index], axis=-1),
            num_nodes=cart_coords.shape[0],
            node_feat={"cart_coords": cart_coords},
            edge_feat={"pbc_offset": source_node_image_offsets},
            _num_graph=num_atoms.shape[0],
            _graph_node_index=paddle.concat(
                [paddle.zeros([1], dtype="int64"), paddle.cumsum(num_atoms)]
            ),
            _graph_edge_index=paddle.concat(
                [paddle.zeros([1], dtype="int64"), paddle.cumsum(num_edges_per_crystal)]
            ),
        )


def _get_smallest_edge_per_node_pair(
    dst_idx: paddle.Tensor,
    src_idx: paddle.Tensor,
    map_edge_to_crystal: paddle.Tensor,
    inter_atom_distances: paddle.Tensor,
    pbc_frac_offsets_per_source_atom: paddle.Tensor,
    num_atoms: paddle.Tensor,
) -> Tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor]:
    """Keep only the minimum-distance image edge per (dst, src) node pair.

    Edges whose distance is within a ``1e-4`` tolerance of the pair minimum
    are all kept (they correspond to symmetry-equivalent images).
    """
    edges = paddle.stack([dst_idx, src_idx], axis=-1)
    unique_edges, map_edge_to_unique = paddle.unique(edges, axis=0, return_inverse=True)

    min_inter_atom_distances = scatter_min(
        inter_atom_distances,
        map_edge_to_unique,
        dim=0,
        dim_size=unique_edges.shape[0],
    )

    batch_size = num_atoms.shape[0]

    cutoff_distances = 1e-4 + min_inter_atom_distances[map_edge_to_unique]
    keep_edge_mask = inter_atom_distances < cutoff_distances
    indices_to_keep = paddle.nonzero(keep_edge_mask).reshape([-1])

    edges = edges[indices_to_keep]
    pbc_frac_offsets_per_source_atom = pbc_frac_offsets_per_source_atom[indices_to_keep]

    num_edges_per_crystal = scatter(
        src=paddle.ones([edges.shape[0]], dtype=paddle.float32),
        index=map_edge_to_crystal[indices_to_keep],
        dim_size=batch_size,
        reduce="sum",
    ).cast(paddle.int64)

    dst_idx = edges[:, 0]
    src_idx = edges[:, 1]
    return src_idx, dst_idx, pbc_frac_offsets_per_source_atom, num_edges_per_crystal


class SpaceGroupEncoder(nn.Layer):
    """Feature encoder for space group indices."""

    def __init__(
        self,
        embedding_tools: "EmbeddingTools",
        hidden_channels: int = 256,
        space_group_embedding_dim: int = 128,
    ):
        super().__init__()
        self.embedding_tools = embedding_tools
        self.net = nn.Sequential(
            nn.Linear(embedding_tools.space_group_embedding_length, hidden_channels),
            Swish(),
            nn.Linear(hidden_channels, space_group_embedding_dim),
            Swish(),
        )

    def forward(self, space_group_indices):
        return self.net(
            self.embedding_tools.get_space_group_embedding(space_group_indices)
        )


class FourierLinear(nn.Layer):
    """Fourier feature encoding for 3D points."""

    def __init__(
        self,
        input_dim: int,
        num_fourier_frequencies: int,
        scale: float,
        output_dim: int,
        num_layers: int = 1,
        use_bias: bool = True,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}.")
        self.scale = scale
        self.num_layers = num_layers

        if self.scale > 0:
            self.fourier_freqs = paddle.create_parameter(
                shape=[input_dim, num_fourier_frequencies],
                dtype="float32",
                default_initializer=nn.initializer.Normal(std=scale),
            )
            self.fourier_freqs.stop_gradient = True
            in_dim = input_dim + 2 * num_fourier_frequencies
            self.layer = nn.Linear(in_dim, output_dim, bias_attr=use_bias)
        else:
            in_dim = input_dim
            self.layer = nn.Linear(in_dim, output_dim, bias_attr=use_bias)
        if num_layers > 1:
            self.layers = nn.LayerList(
                [nn.Linear(output_dim, output_dim) for _ in range(num_layers - 1)]
            )

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        if self.scale > 0:
            with paddle.no_grad():
                v = 2 * math.pi * x @ self.fourier_freqs
                v = paddle.concat([x, v.sin(), v.cos()], axis=-1)
        else:
            v = x
        v = self.layer(v)
        if self.num_layers > 1:
            for layer in self.layers:
                v = nn.functional.silu(layer(v)) + v
        return v


class VariancePreservingAggregation(nn.Layer):
    """Variance preserving aggregation: vpa(X) = sum(X) / sqrt(|X|)."""

    def forward(
        self,
        src: paddle.Tensor,
        index: paddle.Tensor,
        dim_size: int,
    ) -> paddle.Tensor:
        sum_agg = scatter(src, index, dim=0, dim_size=dim_size, reduce="sum")
        counts = scatter(
            paddle.ones([src.shape[0]], dtype=src.dtype),
            index,
            dim=0,
            dim_size=dim_size,
            reduce="sum",
        )
        return paddle.nan_to_num(sum_agg / paddle.sqrt(counts).unsqueeze(-1))


class GraphNorm(nn.Layer):
    """Graph normalization layer."""

    def __init__(self, in_channels: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = self.create_parameter(
            [in_channels],
            default_initializer=nn.initializer.Constant(1.0),
        )
        self.bias = self.create_parameter(
            [in_channels],
            default_initializer=nn.initializer.Constant(0.0),
        )
        self.mean_scale = self.create_parameter(
            [in_channels],
            default_initializer=nn.initializer.Constant(1.0),
        )

    def forward(
        self,
        x: paddle.Tensor,
        map_node_to_graph: paddle.Tensor,
        num_graphs: int,
    ) -> paddle.Tensor:
        mean = scatter(x, map_node_to_graph, dim=0, dim_size=num_graphs, reduce="mean")

        out = x - mean[map_node_to_graph] * self.mean_scale

        var = scatter(
            out**2, map_node_to_graph, dim=0, dim_size=num_graphs, reduce="mean"
        )

        std = (var + self.eps).sqrt()[map_node_to_graph].clip(min=1.0)
        return self.weight * out / std + self.bias


class SinusoidsEmbedding(nn.Layer):
    def __init__(self, n_frequencies=10, n_space=3):
        super().__init__()
        self.n_frequencies = n_frequencies
        self.n_space = n_space
        self.frequencies = 2 * math.pi * paddle.arange(end=self.n_frequencies)
        self.dim = self.n_frequencies * 2 * self.n_space

    def forward(self, x):
        emb = x.unsqueeze(axis=-1) * self.frequencies[None, None, :]
        emb = emb.reshape([-1, self.n_frequencies * self.n_space])
        emb = paddle.concat(x=(emb.sin(), emb.cos()), axis=-1)
        return emb


class CSPLayer(nn.Layer):
    """Message passing layer for CSPNet.

    Args:
        hidden_dim: Hidden dimension.
        act_fn: Activation function.
        dis_emb: Distance embedding module (optional).
        ln: Whether to use LayerNorm.
        use_lattice_ip: If True, use 9-dim lattice inner products (DiffCSP).
                        If False, use 6-dim lattice representation (SGEquiDiff).
        lattice_dim: Dimension of lattice representation
                     (9 for inner products, 6 for lengths+angles).
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        act_fn: nn.Layer = None,
        dis_emb: nn.Layer = None,
        ln: bool = False,
        use_lattice_ip: bool = True,
        lattice_dim: int = 9,
    ):
        super().__init__()
        self.dis_dim = 3
        self.dis_emb = dis_emb
        self.use_lattice_ip = use_lattice_ip
        self.lattice_dim = lattice_dim
        if dis_emb is not None:
            self.dis_dim = dis_emb.dim

        self.edge_mlp = nn.Sequential(
            nn.Linear(
                in_features=hidden_dim * 2 + self.lattice_dim + self.dis_dim,
                out_features=hidden_dim,
            ),
            act_fn,
            nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
            act_fn,
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(in_features=hidden_dim * 2, out_features=hidden_dim),
            act_fn,
            nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
            act_fn,
        )

        self.ln = ln
        if self.ln:
            self.layer_norm = nn.LayerNorm(normalized_shape=hidden_dim)

    def edge_model(
        self,
        node_features: paddle.Tensor,
        frac_coords: paddle.Tensor,
        lattices: paddle.Tensor,
        edge_index: paddle.Tensor,
        edge2graph: paddle.Tensor,
        frac_diff: paddle.Tensor = None,
    ) -> paddle.Tensor:
        hi, hj = node_features[edge_index[0]], node_features[edge_index[1]]
        if frac_diff is None:
            xi, xj = frac_coords[edge_index[0]], frac_coords[edge_index[1]]
            frac_diff = (xj - xi) % 1.0
        if self.dis_emb is not None:
            frac_diff = self.dis_emb(frac_diff)

        if self.use_lattice_ip:
            # DiffCSP: 9-dim lattice inner products
            x = lattices
            perm_0 = list(range(x.ndim))
            perm_0[-1] = -2
            perm_0[-2] = -1
            lattice_ips = x @ x.transpose(perm=perm_0)
        else:
            # SGEquiDiff: 6-dim lattice representation (lengths + angles)
            lattice_ips = lattices

        lattice_ips_flatten = lattice_ips.reshape([-1, self.lattice_dim])
        lattice_ips_flatten_edges = lattice_ips_flatten[edge2graph]
        edges_input = paddle.concat(
            x=[hi, hj, lattice_ips_flatten_edges, frac_diff], axis=1
        )
        edge_features = self.edge_mlp(edges_input)
        return edge_features

    def node_model(
        self,
        node_features: paddle.Tensor,
        edge_features: paddle.Tensor,
        edge_index: paddle.Tensor,
    ) -> paddle.Tensor:
        agg = paddle.geometric.segment_mean(edge_features, edge_index[0])
        agg = paddle.concat(x=[node_features, agg], axis=1)
        out = self.node_mlp(agg)
        return out

    def forward(
        self,
        node_features: paddle.Tensor,
        frac_coords: paddle.Tensor,
        lattices: paddle.Tensor,
        edge_index: paddle.Tensor,
        edge2graph: paddle.Tensor,
        frac_diff: paddle.Tensor = None,
    ) -> paddle.Tensor:
        node_input = node_features
        if self.ln:
            node_features = self.layer_norm(node_input)
        edge_features = self.edge_model(
            node_features, frac_coords, lattices, edge_index, edge2graph, frac_diff
        )
        node_output = self.node_model(node_features, edge_features, edge_index)
        return node_input + node_output


# Plane-wave frequency sampling: draw `num_freqs * _SAMPLES_PER_FREQ_MULT`
# candidates per (kx, ky, kz) to keep the unique-selection loop bounded.
_SAMPLES_PER_FREQ_MULT: int = 2
# Hard cap on rejection-sampling iterations for plane-wave frequency draws.
_PLANE_WAVE_MAX_ITERS: int = 100


def get_plane_wave_frequencies(
    num_freqs: int,
    max_freq: int = 512,
    fourier_scale: float = 1.0,
    isotropic_plane_waves: bool = False,
) -> paddle.Tensor:
    """Return (3, num_freqs) plane wave frequency tensor."""
    if isotropic_plane_waves:
        plane_wave_freqs = (
            paddle.linspace(1, num_freqs, num_freqs).unsqueeze(0).expand([3, num_freqs])
        )
    else:
        freqs_1d_grid = paddle.linspace(-max_freq, max_freq, 1 + 2 * max_freq)
        freqs_1d_grid = freqs_1d_grid[freqs_1d_grid != 0.0]
        normal = paddle.distribution.Normal(
            paddle.to_tensor([0.0]), paddle.to_tensor([fourier_scale])
        )
        probs = normal.log_prob(freqs_1d_grid).exp()

        plane_wave_freqs = paddle.empty([3, 0])
        samples_per_iter = _SAMPLES_PER_FREQ_MULT * num_freqs
        max_iters = _PLANE_WAVE_MAX_ITERS
        iteration = 0
        while plane_wave_freqs.shape[-1] < num_freqs and iteration <= max_iters:
            iteration += 1
            kx = freqs_1d_grid[
                paddle.multinomial(
                    probs, num_samples=samples_per_iter, replacement=True
                )
            ]
            ky = freqs_1d_grid[
                paddle.multinomial(
                    probs, num_samples=samples_per_iter, replacement=True
                )
            ]
            kz = freqs_1d_grid[
                paddle.multinomial(
                    probs, num_samples=samples_per_iter, replacement=True
                )
            ]
            plane_wave_freqs = paddle.concat(
                [plane_wave_freqs, paddle.stack([kx, ky, kz])], axis=-1
            )
            _, unique_idx = paddle.unique(plane_wave_freqs, axis=-1, return_index=True)
            plane_wave_freqs = plane_wave_freqs[:, unique_idx]
        if plane_wave_freqs.shape[-1] < num_freqs:
            plane_wave_freqs = get_plane_wave_frequencies(
                num_freqs, max_freq, fourier_scale, isotropic_plane_waves=True
            )
    return plane_wave_freqs[:, :num_freqs]


def plane_wave_fourier_features(
    x: paddle.Tensor, plane_wave_freqs: paddle.Tensor
) -> paddle.Tensor:
    """Plane wave Fourier features for 3D points."""
    v = 2 * math.pi * x @ plane_wave_freqs
    return paddle.concat([v.sin(), v.cos()], axis=-1)


class TorusMLP(nn.Layer):
    """Simple MLP drift model based on Fourier features."""

    def __init__(
        self,
        time_embedder: SinusoidalTimeEmbeddings,
        num_plane_wave_freqs: int,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.time_embedder = time_embedder
        self.num_plane_wave_freqs = num_plane_wave_freqs
        plane_wave_freqs = get_plane_wave_frequencies(num_freqs=num_plane_wave_freqs)
        self.register_buffer("plane_wave_freqs", plane_wave_freqs)
        self.layers = nn.Sequential(
            nn.Linear(2 * self.num_plane_wave_freqs + time_embedder.dim, hidden_dim),
            nn.Silu(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Silu(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Silu(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Silu(),
            nn.Linear(hidden_dim, 3),
        )

    def construct_graph_inputs(
        self,
        frac_coords: paddle.Tensor,
        n_atoms_per_xtal: paddle.Tensor,
        lattice_matrices: paddle.Tensor,
        lattice_lengths: paddle.Tensor,
        lattice_angles: paddle.Tensor,
    ) -> tuple:
        """Protocol placeholder: the MLP backbone builds no graph.

        Kept for uniformity with GNN/CSPNet so the compiled runtime
        boundary stays backbone-agnostic; the whole MLP forward is
        tensor-only and compiled in one piece.
        """
        return ()

    def forward_with_graph(
        self,
        frac_coords: paddle.Tensor,
        element_indices: paddle.Tensor = None,
        time_embeddings: paddle.Tensor = None,
        n_atoms_per_xtal: paddle.Tensor = None,
        graph_inputs: tuple = (),
    ) -> paddle.Tensor:
        """Pure numerical core (the whole MLP is tensor-only)."""
        return self.layers(
            paddle.concat(
                [
                    plane_wave_fourier_features(frac_coords, self.plane_wave_freqs),
                    time_embeddings,
                ],
                axis=-1,
            )
        )


def custom_he_orthogonal_(weight: paddle.Tensor, gain: float = 1.0) -> paddle.Tensor:
    """He initialization + orthogonalization."""
    with paddle.no_grad():
        fan_in = weight.shape[1]
        if fan_in <= 1:
            raise ValueError(f"custom_he_orthogonal_ requires fan_in > 1, got {fan_in}")

        nn.initializer.Orthogonal()(weight)
        eps = 1e-6
        mean = weight.mean(axis=1, keepdim=True)
        var = weight.std(axis=1, keepdim=True)
        result = gain * math.sqrt(1 / fan_in) * ((weight - mean) / (var + eps).sqrt())
        paddle.assign(result, weight)
    return weight


class NodeAndEdgeEmbedder(nn.Layer):
    """Node and edge initial embedding module."""

    def __init__(
        self,
        edge_hidden_dim: int,
        atom_hidden_dim: int,
        fourier_frac_edge_dim: int,
        gaussian_cart_edge_dim: int,
        time_emb_dim: int,
        activation: nn.Layer,
        use_frac_coords_in_node_emb: bool,
        embedding_tools: "EmbeddingTools",
    ):
        super().__init__()
        self.act = activation
        self.use_frac_coords_in_node_emb = use_frac_coords_in_node_emb
        self.embedding_tools = embedding_tools

        if use_frac_coords_in_node_emb:
            self.frac_pos_emb = nn.Linear(fourier_frac_edge_dim, atom_hidden_dim)
            self.ele_emb = nn.Linear(
                embedding_tools.element_embedding_length, atom_hidden_dim
            )
            self.atom_emb1 = nn.Linear(2 * atom_hidden_dim, atom_hidden_dim)
        else:
            self.atom_emb1 = nn.Linear(
                embedding_tools.element_embedding_length, atom_hidden_dim
            )
        self.atom_emb2 = nn.Linear(atom_hidden_dim + time_emb_dim, atom_hidden_dim)

        self.edge_emb1 = nn.Linear(
            fourier_frac_edge_dim + gaussian_cart_edge_dim + NUM_LATTICE_PARAMS,
            edge_hidden_dim,
            bias_attr=False,
        )
        self.edge_emb2 = nn.Linear(edge_hidden_dim, edge_hidden_dim, bias_attr=False)
        self._reset_parameters()

    def _reset_parameters(self):
        custom_he_orthogonal_(self.atom_emb1.weight, gain=2.0)
        paddle.assign(paddle.zeros_like(self.atom_emb1.bias), self.atom_emb1.bias)
        custom_he_orthogonal_(self.atom_emb2.weight, gain=2.0)
        paddle.assign(paddle.zeros_like(self.atom_emb2.bias), self.atom_emb2.bias)
        custom_he_orthogonal_(self.edge_emb1.weight, gain=2.0)
        custom_he_orthogonal_(self.edge_emb2.weight, gain=2.0)

    def forward(
        self,
        element_indices: paddle.Tensor,
        time_embeddings: paddle.Tensor,
        fourier_relative_frac_pos: paddle.Tensor,
        gaussian_cart_dists: paddle.Tensor,
        normed_lattice_params: paddle.Tensor,
        fourier_atom_frac_pos: Optional[paddle.Tensor] = None,
    ) -> dict:
        e = self.edge_emb1(
            paddle.concat(
                [fourier_relative_frac_pos, gaussian_cart_dists, normed_lattice_params],
                axis=-1,
            )
        )
        e = self.act(e)
        e = self.act(self.edge_emb2(e))

        if self.use_frac_coords_in_node_emb:
            frac_pos_emb = self.act(self.frac_pos_emb(fourier_atom_frac_pos))
            ele_emb_out = self.act(
                self.ele_emb(
                    self.embedding_tools.get_element_embedding(1 + element_indices)
                )
            )
            h = self.atom_emb1(paddle.concat([frac_pos_emb, ele_emb_out], axis=-1))
        else:
            h = self.atom_emb1(
                self.embedding_tools.get_element_embedding(1 + element_indices)
            )
        h = self.act(h)
        h = self.act(self.atom_emb2(paddle.concat([h, time_embeddings], axis=-1)))
        return {"h": h, "e": e}


class InteractionBlock(nn.Layer):
    """Custom message passing GNN layer."""

    def __init__(
        self,
        hidden_channels: int,
        edge_hidden_dim: int,
        activation: nn.Layer,
        graph_norm: bool = True,
        use_vpa: bool = True,
    ):
        super().__init__()
        self.act = activation
        self.use_graph_norm = graph_norm
        self.use_vpa = use_vpa

        if use_vpa:
            self.aggregator = VariancePreservingAggregation()
        if graph_norm:
            self.graph_norm = GraphNorm(hidden_channels)

        self.lin_geom = nn.Linear(
            edge_hidden_dim + 2 * hidden_channels, hidden_channels, bias_attr=False
        )
        self.lin_h = nn.Linear(hidden_channels, hidden_channels)
        self.out_layer = nn.Linear(hidden_channels, hidden_channels)
        self.skipinit_gain = self.create_parameter(
            [], default_initializer=nn.initializer.Constant(0.0)
        )
        self._reset_parameters()

    def _reset_parameters(self):
        custom_he_orthogonal_(self.lin_geom.weight, gain=4.0)
        custom_he_orthogonal_(self.out_layer.weight, gain=3.0)
        paddle.assign(paddle.zeros_like(self.out_layer.bias), self.out_layer.bias)
        custom_he_orthogonal_(self.lin_h.weight, gain=3.0)
        paddle.assign(paddle.zeros_like(self.lin_h.bias), self.lin_h.bias)

    def forward(
        self,
        h: paddle.Tensor,
        edge_index: paddle.Tensor,
        e: paddle.Tensor,
        map_node_to_graph: Optional[paddle.Tensor] = None,
        num_graphs: Optional[int] = None,
    ) -> paddle.Tensor:
        """Message passing + aggregation."""
        src_ids = edge_index[0]
        dst_ids = edge_index[1]

        e_full = paddle.concat([e, h[src_ids], h[dst_ids]], axis=1)
        e_full = self.act(self.lin_geom(e_full))

        messages = h[src_ids] * e_full

        n_nodes = h.shape[0]
        if self.use_vpa:
            h_agg = self.aggregator(messages, dst_ids, dim_size=n_nodes)
        else:
            h_agg = scatter(messages, dst_ids, dim=0, dim_size=n_nodes, reduce="sum")

        if self.use_graph_norm:
            h_agg = self.graph_norm(h_agg, map_node_to_graph, num_graphs)
            h_agg = self.act(h_agg)
        h_agg = self.act(self.lin_h(h_agg))
        h_agg = self.act(self.out_layer(h_agg))

        return self.skipinit_gain * h_agg


class GNN(nn.Layer):
    """GNN non-equivariant drift module."""

    def __init__(
        self,
        time_embedder: SinusoidalTimeEmbeddings,
        embedding_tools: "EmbeddingTools",
        num_plane_wave_freqs: int = 96,
        num_cartesian_distance_gaussians: int = 96,
        edge_hidden_dim: int = 128,
        atom_hidden_dim: int = 256,
        use_vpa: bool = True,
        use_graph_norm: bool = True,
        num_msg_pass_steps: int = 5,
        cutoff: float = 10.0,
        use_frac_coords_in_node_emb: bool = True,
        dataset_name: str = "mp_20",
        graph_converter_cfg: Optional[dict] = None,
    ):
        super().__init__()
        from ppmat.models import build_graph_converter

        if graph_converter_cfg is None:
            graph_converter_cfg = {
                "__class_name__": "SGEQuiDiffGraphConverter",
                "__init_params__": {},
            }
        self.graph_converter = build_graph_converter(graph_converter_cfg)
        self.time_embedder = time_embedder
        self.use_frac_coords_in_node_emb = use_frac_coords_in_node_emb
        self.dataset_name = dataset_name

        plane_wave_freqs = get_plane_wave_frequencies(num_freqs=num_plane_wave_freqs)
        self.register_buffer("plane_wave_freqs", plane_wave_freqs)

        self.gaussian_smearing = GaussianSmearing(
            0.0, cutoff, num_cartesian_distance_gaussians
        )
        self.activation = Swish()
        self.embed_block = NodeAndEdgeEmbedder(
            edge_hidden_dim,
            atom_hidden_dim,
            2 * num_plane_wave_freqs,
            num_cartesian_distance_gaussians,
            self.time_embedder.dim,
            self.activation,
            use_frac_coords_in_node_emb,
            embedding_tools,
        )
        self.interaction_blocks = nn.LayerList(
            [
                InteractionBlock(
                    hidden_channels=atom_hidden_dim,
                    edge_hidden_dim=edge_hidden_dim,
                    activation=self.activation,
                    graph_norm=use_graph_norm,
                    use_vpa=use_vpa,
                )
                for _ in range(num_msg_pass_steps)
            ]
        )
        self.mlp_skip_co = nn.Linear(
            (num_msg_pass_steps + 1) * atom_hidden_dim,
            atom_hidden_dim,
        )
        self.mlp_out = nn.Linear(atom_hidden_dim, 3)

    def construct_graph_inputs(
        self,
        frac_coords: paddle.Tensor,
        n_atoms_per_xtal: paddle.Tensor,
        lattice_matrices: paddle.Tensor,
        lattice_lengths: paddle.Tensor,
        lattice_angles: paddle.Tensor,
    ) -> tuple:
        """Graph construction and edge features (data-dependent shapes, kept
        eager outside the compiled runtime boundary)."""
        (
            map_atom_to_xtal,
            edge_index,
            relative_fractional_positions,
            cartesian_distances,
            num_edges_per_crystal,
        ) = self.construct_graphs(frac_coords, n_atoms_per_xtal, lattice_matrices)
        fourier_relative_frac_pos = plane_wave_fourier_features(
            relative_fractional_positions, self.plane_wave_freqs
        )
        gaussian_smeared_cart_dists = self.gaussian_smearing(cartesian_distances)
        normed_lattice_params = self.norm_lattice_params(
            lattice_lengths, lattice_angles
        ).repeat_interleave(num_edges_per_crystal, axis=0)
        return (
            map_atom_to_xtal,
            edge_index,
            fourier_relative_frac_pos,
            gaussian_smeared_cart_dists,
            normed_lattice_params,
        )

    def forward_with_graph(
        self,
        frac_coords: paddle.Tensor,
        element_indices: paddle.Tensor,
        time_embeddings: paddle.Tensor,
        n_atoms_per_xtal: paddle.Tensor,
        graph_inputs: tuple,
    ) -> paddle.Tensor:
        """Pure numerical core: embedding, message passing, output head."""
        (
            map_atom_to_xtal,
            edge_index,
            fourier_relative_frac_pos,
            gaussian_smeared_cart_dists,
            normed_lattice_params,
        ) = graph_inputs

        if self.use_frac_coords_in_node_emb:
            fourier_atom_frac_pos = plane_wave_fourier_features(
                frac_coords, self.plane_wave_freqs
            )
        else:
            fourier_atom_frac_pos = None

        embed_out = self.embed_block(
            element_indices,
            time_embeddings,
            fourier_relative_frac_pos,
            gaussian_smeared_cart_dists,
            normed_lattice_params,
            fourier_atom_frac_pos,
        )
        node_latents = embed_out["h"]
        edge_latents = embed_out["e"]

        skip_connection_elements = []
        for interaction in self.interaction_blocks:
            skip_connection_elements.append(node_latents)
            node_latents = node_latents + interaction(
                node_latents,
                edge_index,
                edge_latents,
                map_atom_to_xtal,
                n_atoms_per_xtal.shape[0],
            )

        skip_connection_elements.append(node_latents)
        node_latents = self.mlp_skip_co(paddle.concat(skip_connection_elements, axis=1))
        out_vectors = self.mlp_out(node_latents)
        return out_vectors

    def construct_graphs(
        self,
        frac_coords: paddle.Tensor,
        n_atoms_per_xtal: paddle.Tensor,
        lattice_matrices: paddle.Tensor,
    ) -> tuple:
        """Construct PBC graph structure."""
        n_crystals = lattice_matrices.shape[0]
        n_nodes = frac_coords.shape[0]
        atom_counts = n_atoms_per_xtal.cast("int64").reshape([-1])
        cumulative = paddle.cumsum(atom_counts, axis=0)
        node_ids = paddle.arange(n_nodes, dtype=cumulative.dtype).reshape([-1, 1])
        map_atom_to_xtal = (
            (node_ids >= cumulative.reshape([1, -1])).cast("int64").sum(axis=1)
        )
        map_atom_to_xtal = paddle.clip(map_atom_to_xtal, min=0, max=n_crystals - 1)

        cart_coords = frac_to_cart_coords(
            frac_coords, n_atoms_per_xtal, lattices=lattice_matrices
        )
        graph = self.graph_converter.build_graph(
            cart_coords=cart_coords,
            lattice=lattice_matrices,
            num_atoms=n_atoms_per_xtal,
        )
        edge_index = graph.edges.transpose([1, 0])
        source_ids, destination_ids = edge_index[0], edge_index[1]
        source_node_image_offsets = graph.edge_feat["pbc_offset"]
        num_edges_per_crystal = paddle.bincount(
            graph.graph_edge_id, minlength=n_atoms_per_xtal.shape[0]
        )
        out = get_pbc_distances(
            cart_coords,
            edge_index,
            lattice_matrices,
            source_node_image_offsets,
            num_atoms=n_atoms_per_xtal,
            num_bonds=num_edges_per_crystal,
            coord_is_cart=True,
        )
        cartesian_distances = out["distances"]
        edge_index = out["edge_index"]
        relative_fractional_positions = (
            frac_coords[source_ids] - frac_coords[destination_ids]
        )
        return (
            map_atom_to_xtal,
            edge_index,
            relative_fractional_positions,
            cartesian_distances,
            num_edges_per_crystal,
        )

    @paddle.no_grad()
    def norm_lattice_params(
        self, lattice_lengths: paddle.Tensor, lattice_angles: paddle.Tensor
    ) -> paddle.Tensor:
        """Normalize lattice parameters to [-1, 1]."""
        param_ranges = lattice_parameter_ranges[self.dataset_name]
        min_len = param_ranges["min_lattice_length"]
        max_len = param_ranges["max_lattice_length"]
        min_ang = param_ranges["min_lattice_angle"]
        max_ang = param_ranges["max_lattice_angle"]

        normed_lengths = 2.0 * (lattice_lengths - min_len) / (max_len - min_len) - 1.0
        normed_angles = 2.0 * (lattice_angles - min_ang) / (max_ang - min_ang) - 1.0
        return paddle.concat([normed_lengths, normed_angles], axis=-1)


class CSPNet(nn.Layer):
    """CSPNet from DiffCSP architecture.

    Uses the shared ``CSPLayer`` with the 6-dim lattice representation
    (lengths + angles) specific to SGEquiDiff.
    """

    def __init__(
        self,
        time_embedder: nn.Layer,
        hidden_dim: int = 256,
        num_msg_pass_steps: int = 6,
        ln: bool = False,
        act_fn: str = "silu",
        dis_emb: str = "sin",
        num_freqs: int = 128,
        dense: bool = False,
    ):
        super().__init__()
        latent_dim = time_embedder.dim
        num_layers = num_msg_pass_steps
        max_atoms = ELEMENT_ENCODING_SIZE

        self.node_embedding = nn.Embedding(max_atoms, hidden_dim)
        self.atom_latent_emb = nn.Linear(hidden_dim + latent_dim, hidden_dim)

        if act_fn == "silu":
            self.act_fn = nn.Silu()
        else:
            raise ValueError(f"Unsupported act_fn: {act_fn}")
        if dis_emb == "sin":
            self.dis_emb = SinusoidsEmbedding(n_frequencies=num_freqs, n_space=3)
        elif dis_emb == "none":
            self.dis_emb = None
        else:
            raise ValueError(f"Unsupported dis_emb: {dis_emb}")

        for i in range(num_layers):
            self.add_sublayer(
                f"csp_layer_{i}",
                CSPLayer(
                    hidden_dim=hidden_dim,
                    act_fn=self.act_fn,
                    dis_emb=self.dis_emb,
                    ln=ln,
                    use_lattice_ip=False,
                    lattice_dim=NUM_LATTICE_PARAMS,
                ),
            )
        self.num_layers = num_layers
        self.dense = dense

        hidden_dim_before_out = hidden_dim
        if self.dense:
            hidden_dim_before_out = hidden_dim_before_out * (num_layers + 1)

        self.coord_out = nn.Linear(hidden_dim_before_out, 3, bias_attr=False)
        self.ln = ln
        if self.ln:
            self.final_layer_norm = nn.LayerNorm(hidden_dim)

    def gen_edges(
        self, num_atoms: paddle.Tensor, frac_coords: paddle.Tensor
    ) -> Tuple[paddle.Tensor, paddle.Tensor]:
        """Generate fully-connected edges."""
        lis = [
            paddle.ones([n, n], dtype=paddle.float32)
            for n in num_atoms.numpy().tolist()
        ]
        fc_graph = paddle.block_diag(lis)
        fc_edges = paddle.nonzero(fc_graph).T
        frac_diff = (frac_coords[fc_edges[1]] - frac_coords[fc_edges[0]]) % 1.0
        return fc_edges, frac_diff

    def construct_graph_inputs(
        self,
        frac_coords: paddle.Tensor,
        n_atoms_per_xtal: paddle.Tensor,
        lattice_lengths: paddle.Tensor,
        lattice_angles: paddle.Tensor,
    ) -> tuple:
        """Fully-connected edge construction (data-dependent shapes, kept
        eager outside the compiled runtime boundary)."""
        n_crystals = n_atoms_per_xtal.shape[0]
        node2graph = paddle.arange(n_crystals).repeat_interleave(
            n_atoms_per_xtal, axis=0
        )
        lattices = paddle.concat([lattice_lengths, lattice_angles], axis=-1)
        edges, frac_diff = self.gen_edges(n_atoms_per_xtal, frac_coords)
        return (edges, frac_diff, node2graph, lattices)

    def forward_with_graph(
        self,
        frac_coords: paddle.Tensor,
        element_indices: paddle.Tensor,
        time_embeddings: paddle.Tensor,
        n_atoms_per_xtal: paddle.Tensor,
        graph_inputs: tuple,
    ) -> paddle.Tensor:
        """Pure numerical core: embedding, message passing, coordinate head."""
        (edges, frac_diff, node2graph, lattices) = graph_inputs
        edge2graph = node2graph[edges[0]]
        node_features = self.node_embedding(element_indices)
        node_features = paddle.concat([node_features, time_embeddings], axis=-1)
        node_features = self.atom_latent_emb(node_features)

        h_list = [node_features]
        for i in range(self.num_layers):
            # self.sublayers(name) returns all sublayer list in Paddle,
            # cannot index by name; use getattr
            node_features = getattr(self, f"csp_layer_{i}")(
                node_features,
                frac_coords,
                lattices,
                edges,
                edge2graph,
                frac_diff=frac_diff,
            )
            if i != self.num_layers - 1:
                h_list.append(node_features)

        if self.ln:
            node_features = self.final_layer_norm(node_features)
        h_list.append(node_features)

        if self.dense:
            node_features = paddle.concat(h_list, axis=-1)

        return self.coord_out(node_features)


def lattice_transform_and_log_prob_mask(
    spacegroup: int,
) -> Tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor]:
    if 1 <= spacegroup <= 2:
        length_matrix = paddle.eye(3)
        angle_matrix = paddle.eye(3)
        angle_vector = paddle.zeros([3])
        log_prob_mask = paddle.ones([NUM_LATTICE_PARAMS])
    elif 3 <= spacegroup <= 15:
        length_matrix = paddle.eye(3)
        angle_matrix = paddle.to_tensor(
            [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]
        )
        angle_vector = paddle.to_tensor([90.0, 0.0, 90.0])
        log_prob_mask = paddle.to_tensor([1.0, 1.0, 1.0, 0.0, 1.0, 0.0])
    elif 16 <= spacegroup <= 74:
        length_matrix = paddle.eye(3)
        angle_matrix = paddle.zeros([3, 3])
        angle_vector = paddle.to_tensor([90.0, 90.0, 90.0])
        log_prob_mask = paddle.to_tensor([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    elif 75 <= spacegroup <= 142:
        length_matrix = paddle.to_tensor(
            [[1.0, 1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
        )
        angle_matrix = paddle.zeros([3, 3])
        angle_vector = paddle.to_tensor([90.0, 90.0, 90.0])
        log_prob_mask = paddle.to_tensor([1.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    elif 143 <= spacegroup <= 194:
        length_matrix = paddle.to_tensor(
            [[1.0, 1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
        )
        angle_matrix = paddle.zeros([3, 3])
        angle_vector = paddle.to_tensor([90.0, 90.0, 120.0])
        log_prob_mask = paddle.to_tensor([1.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    elif 195 <= spacegroup <= NUM_CRYSTALLOGRAPHIC_SPACE_GROUPS:
        length_matrix = paddle.to_tensor(
            [[1.0, 1.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
        )
        angle_matrix = paddle.zeros([3, 3])
        angle_vector = paddle.to_tensor([90.0, 90.0, 90.0])
        log_prob_mask = paddle.to_tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    else:
        raise ValueError(f"Invalid space group: {spacegroup}")
    return length_matrix, angle_matrix, angle_vector, log_prob_mask


class TelescopingDiscreteLatticeSampler(nn.Layer):
    def __init__(
        self,
        embedding_tools: "EmbeddingTools",
        input_dimension: int = 128,
        hidden_dimension: int = 256,
        min_lattice_length: float = 2.0,
        max_lattice_length: float = 133.0,
        min_lattice_angle: float = 60.0,
        max_lattice_angle: float = 135.0,
        gradient_attenuation_factor: float = 1.0,
        n_bins: int = 100,
        n_telescopes: int = 2,
        lattice_length_bin_embedder_fourier_scale: float = 2.0,
        lattice_angle_bin_embedder_fourier_scale: float = 1.0,
        lattice_length_embedder_fourier_scale: float = 5.0,
        lattice_angle_embedder_fourier_scale: float = 1.0,
        lattice_param_dim: int = 32,
        n_emb_layers: int = 2,
        space_group_encoder_hidden_channels: int = 256,
        num_fourier_frequencies: int = 128,
        length_fourier_output_dim: int = 512,
        angle_fourier_output_dim: int = 256,
        bin_fourier_output_dim: int = 256,
    ):
        super().__init__()
        self.MAX_LATTICE_LENGTH = max_lattice_length
        self.MIN_LATTICE_LENGTH = min_lattice_length
        self.MAX_LATTICE_ANGLE = max_lattice_angle
        self.MIN_LATTICE_ANGLE = min_lattice_angle
        self.gradient_attenuation_factor = gradient_attenuation_factor

        bravais_data = [lattice_transform_and_log_prob_mask(sg) for sg in range(1, 231)]
        bravais_length_transforms = paddle.stack([d[0] for d in bravais_data], axis=0)
        bravais_angle_transforms = paddle.stack([d[1] for d in bravais_data], axis=0)
        bravais_angle_offsets = paddle.stack([d[2] for d in bravais_data], axis=0)
        bravais_log_prob_masks = paddle.stack([d[3] for d in bravais_data], axis=0)
        self.register_buffer("bravais_length_transforms", bravais_length_transforms)
        self.register_buffer("bravais_angle_transforms", bravais_angle_transforms)
        self.register_buffer("bravais_angle_offsets", bravais_angle_offsets)
        self.register_buffer("bravais_log_prob_masks", bravais_log_prob_masks)

        self.n_bins = n_bins
        self.n_telescopes = n_telescopes
        # Normalized-parameter bin domain; wider than the [-1, 1] normalized
        # parameter range (empirical choice inherited from the upstream
        # implementation) so telescoping refinement has margin.
        self.min_bin_edge = -4.0
        self.max_bin_edge = 4.0

        self.space_group_encoder = SpaceGroupEncoder(
            embedding_tools=embedding_tools,
            hidden_channels=space_group_encoder_hidden_channels,
            space_group_embedding_dim=input_dimension,
        )

        self.lattice_param_dim = lattice_param_dim

        self.lattice_length_embedder = nn.Sequential(
            FourierLinear(
                input_dim=1,
                num_fourier_frequencies=num_fourier_frequencies,
                scale=lattice_length_embedder_fourier_scale,
                num_layers=n_emb_layers,
                output_dim=length_fourier_output_dim,
                use_bias=True,
            ),
            nn.Linear(length_fourier_output_dim, self.lattice_param_dim),
            nn.Silu(),
        )

        self.lattice_angle_embedder = nn.Sequential(
            FourierLinear(
                input_dim=1,
                num_fourier_frequencies=num_fourier_frequencies,
                scale=lattice_angle_embedder_fourier_scale,
                num_layers=n_emb_layers,
                output_dim=angle_fourier_output_dim,
                use_bias=True,
            ),
            nn.Linear(angle_fourier_output_dim, self.lattice_param_dim),
            nn.Silu(),
        )

        self.length_bin_embedder = nn.Sequential(
            FourierLinear(
                input_dim=2,
                num_fourier_frequencies=num_fourier_frequencies,
                scale=lattice_length_bin_embedder_fourier_scale,
                output_dim=bin_fourier_output_dim,
                num_layers=n_emb_layers,
                use_bias=True,
            ),
            nn.Linear(bin_fourier_output_dim, hidden_dimension),
            nn.Silu(),
        )

        self.angle_bin_embedder = nn.Sequential(
            FourierLinear(
                input_dim=2,
                num_fourier_frequencies=num_fourier_frequencies,
                scale=lattice_angle_bin_embedder_fourier_scale,
                output_dim=bin_fourier_output_dim,
                num_layers=n_emb_layers,
                use_bias=True,
            ),
            nn.Linear(bin_fourier_output_dim, hidden_dimension),
            nn.Silu(),
        )

        self.bin_conditioning_info_dim = (
            input_dimension
            + NUM_LATTICE_PARAMS * self.lattice_param_dim
            + NUM_LATTICE_PARAMS
        )

        self.bin_logit_head = nn.Sequential(
            nn.Linear(
                hidden_dimension + self.bin_conditioning_info_dim,
                hidden_dimension,
            ),
            nn.Silu(),
            nn.Linear(hidden_dimension, NUM_LATTICE_PARAMS),
        )

        self.register_buffer("grid_pts", paddle.linspace(0, 1, self.n_bins + 1))

        angle_offsets = self.bravais_angle_offsets.clone()
        unconstrained_angle_mask = angle_offsets == 0.0
        angle_offsets[unconstrained_angle_mask] = self.MIN_LATTICE_ANGLE
        _normed_params = self._get_normed_lattice_parameters(
            self.MIN_LATTICE_LENGTH * paddle.ones_like(self.bravais_angle_offsets),
            angle_offsets,
            self.min_bin_edge,
            self.max_bin_edge,
        )
        _discretized_normed_params = self.get_discretized_normed_lattice_params(
            _normed_params, angle_offsets[:, :2]
        )
        _discretized_normed_angle_offsets = _discretized_normed_params[:, 3:].clone()
        _discretized_normed_angle_offsets[unconstrained_angle_mask] = 0.0
        self.register_buffer(
            "discretized_normed_bravais_angle_offsets",
            _discretized_normed_angle_offsets,
        )

    @paddle.no_grad()
    def _get_normed_lattice_parameters(
        self,
        lattice_lengths,
        lattice_angles,
        min_normed_param: float = -1.0,
        max_normed_param: float = 1.0,
    ):
        normed_param_range = max_normed_param - min_normed_param
        normed_lattice_lengths = (
            normed_param_range
            * (
                (lattice_lengths - self.MIN_LATTICE_LENGTH)
                / (self.MAX_LATTICE_LENGTH - self.MIN_LATTICE_LENGTH)
            )
            + min_normed_param
        )

        min_gamma_angle = self.MIN_LATTICE_ANGLE
        max_gamma_angle = self.MAX_LATTICE_ANGLE

        normed_lattice_angles = paddle.concat(
            [
                normed_param_range
                * (
                    (lattice_angles[:, :2] - self.MIN_LATTICE_ANGLE)
                    / (self.MAX_LATTICE_ANGLE - self.MIN_LATTICE_ANGLE)
                )
                + min_normed_param,
                normed_param_range
                * (
                    (lattice_angles[:, -1] - min_gamma_angle)
                    / (max_gamma_angle - min_gamma_angle)
                ).unsqueeze(-1)
                + min_normed_param,
            ],
            axis=1,
        )
        return paddle.concat([normed_lattice_lengths, normed_lattice_angles], axis=1)

    def get_valid_gamma_angle_interval(self, alpha_and_beta_angles):
        cos_alpha = paddle.cos(alpha_and_beta_angles[:, 0] * math.pi / 180.0)
        cos_beta = paddle.cos(alpha_and_beta_angles[:, 1] * math.pi / 180.0)
        cos_alpha_sq = cos_alpha**2
        cos_beta_sq = cos_beta**2
        term1 = cos_alpha * cos_beta
        inner = 4 * cos_alpha_sq * cos_beta_sq - 4 * (cos_alpha_sq + cos_beta_sq - 1)
        inner = paddle.clip(inner, min=0.0)
        term2 = 0.5 * paddle.sqrt(inner)
        gamma_min = (
            paddle.acos(paddle.clip(term1 + term2, min=-1.0, max=1.0)) * 180.0 / math.pi
        )
        gamma_max = (
            paddle.acos(paddle.clip(term1 - term2, min=-1.0, max=1.0)) * 180.0 / math.pi
        )
        return (
            paddle.clip(
                gamma_min, min=self.MIN_LATTICE_ANGLE, max=self.MAX_LATTICE_ANGLE
            ),
            paddle.clip(
                gamma_max, min=self.MIN_LATTICE_ANGLE, max=self.MAX_LATTICE_ANGLE
            ),
        )

    def _angle_to_normed(self, angle: paddle.Tensor) -> paddle.Tensor:
        """Map an angle (degrees) in [MIN_LATTICE_ANGLE, MAX_LATTICE_ANGLE]
        to normed bin coords."""
        return (self.max_bin_edge - self.min_bin_edge) * (
            (angle - self.MIN_LATTICE_ANGLE)
            / (self.MAX_LATTICE_ANGLE - self.MIN_LATTICE_ANGLE)
        ) + self.min_bin_edge

    def _build_bin_edges(
        self, min_bin_edge: paddle.Tensor, max_bin_edge: paddle.Tensor
    ) -> paddle.Tensor:
        """Build per-sample bin edges from [min, max] range over self.grid_pts."""
        return min_bin_edge.unsqueeze(-1) + (max_bin_edge - min_bin_edge).unsqueeze(
            -1
        ) * self.grid_pts.unsqueeze(0)

    @paddle.no_grad()
    def get_discretized_normed_lattice_params(
        self, normed_lattice_parameters, raw_alpha_and_beta_angles
    ):
        batch_size = normed_lattice_parameters.shape[0]
        _batch_idxs = paddle.arange(batch_size)
        discretized_normed_lattice_parameters = paddle.zeros_like(
            normed_lattice_parameters
        )
        for i in range(NUM_LATTICE_PARAMS):
            min_bin_edge = self.min_bin_edge * paddle.ones([batch_size])
            max_bin_edge = self.max_bin_edge * paddle.ones([batch_size])
            x = normed_lattice_parameters[:, i]
            for _ in range(self.n_telescopes):
                bin_edges = self._build_bin_edges(min_bin_edge, max_bin_edge)
                bins = paddle.stack([bin_edges[:, :-1], bin_edges[:, 1:]], axis=-1)
                normalized_x = (x - min_bin_edge) / (max_bin_edge - min_bin_edge)
                bin_idxs = paddle.bucketize(normalized_x, self.grid_pts[1:])
                bin_idxs = paddle.clip(bin_idxs, max=self.n_bins - 1)
                chosen_bins = bins[_batch_idxs, bin_idxs]
                min_bin_edge = chosen_bins[:, 0]
                max_bin_edge = chosen_bins[:, 1]
            discretized_normed_lattice_parameters[:, i] = paddle.mean(
                chosen_bins, axis=-1
            )

        min_gamma, max_gamma = self.get_valid_gamma_angle_interval(
            raw_alpha_and_beta_angles
        )
        min_normed_gamma = self._angle_to_normed(min_gamma)
        max_normed_gamma = self._angle_to_normed(max_gamma)
        gammas_lt_min = discretized_normed_lattice_parameters[:, -1] < min_normed_gamma
        gammas_gt_max = discretized_normed_lattice_parameters[:, -1] > max_normed_gamma

        if paddle.any(gammas_lt_min | gammas_gt_max):
            bin_edges = (self.max_bin_edge - self.min_bin_edge) * paddle.linspace(
                0, 1, self.n_telescopes * self.n_bins + 1
            ) + self.min_bin_edge
            bin_midpoints = paddle.mean(
                paddle.stack([bin_edges[:-1], bin_edges[1:]], axis=-1), axis=-1
            )
            if paddle.any(gammas_lt_min):
                valid_bin_indices = paddle.argmax(
                    (
                        bin_midpoints.unsqueeze(0)
                        > min_normed_gamma[gammas_lt_min].unsqueeze(-1)
                    ).cast("int32"),
                    axis=-1,
                )
                discretized_normed_lattice_parameters[
                    gammas_lt_min, -1
                ] = bin_midpoints[valid_bin_indices]
            if paddle.any(gammas_gt_max):
                valid_bin_indices = paddle.argmax(
                    (
                        bin_midpoints.unsqueeze(0)
                        < max_normed_gamma[gammas_gt_max].unsqueeze(-1)
                    ).cast("int32"),
                    axis=-1,
                )
                discretized_normed_lattice_parameters[
                    gammas_gt_max, -1
                ] = bin_midpoints[valid_bin_indices]
        return discretized_normed_lattice_parameters

    def log_prob(
        self,
        lattice_lengths: paddle.Tensor,
        lattice_angles: paddle.Tensor,
        space_group_indices: paddle.Tensor,
        noisy_lattice_lengths: paddle.Tensor = None,
        noisy_lattice_angles: paddle.Tensor = None,
    ) -> paddle.Tensor:
        """Log forward probability of sampling the given lattice parameters."""
        batch_size = lattice_lengths.shape[0]
        noisy_lattice_lengths = (
            noisy_lattice_lengths
            if self.training and noisy_lattice_lengths is not None
            else lattice_lengths
        )
        noisy_lattice_angles = (
            noisy_lattice_angles
            if self.training and noisy_lattice_angles is not None
            else lattice_angles
        )
        expected_shape = (batch_size, 3)
        if not (
            lattice_lengths.shape
            == lattice_angles.shape
            == noisy_lattice_lengths.shape
            == noisy_lattice_angles.shape
            == expected_shape
        ):
            raise ValueError(
                "lattice_lengths / lattice_angles / noisy_* must all have shape "
                f"{expected_shape}, got lengths={lattice_lengths.shape}, "
                f"angles={lattice_angles.shape}, "
                f"noisy_lengths={noisy_lattice_lengths.shape}, "
                f"noisy_angles={noisy_lattice_angles.shape}"
            )

        sg_features = self.space_group_encoder(space_group_indices)
        normed_lattice_parameters = self._get_normed_lattice_parameters(
            lattice_lengths,
            lattice_angles,
            self.min_bin_edge,
            self.max_bin_edge,
        )
        normed_noisy_lattice_parameters = (
            self._get_normed_lattice_parameters(
                noisy_lattice_lengths,
                noisy_lattice_angles,
                self.min_bin_edge,
                self.max_bin_edge,
            )
            if self.training
            else normed_lattice_parameters
        )
        _normed_gt_noisy_params = self.get_discretized_normed_lattice_params(
            paddle.concat(
                [normed_lattice_parameters, normed_noisy_lattice_parameters],
            ),
            # get_valid_gamma_angle_interval expects alpha/beta angles [:, :2].
            # DEV NOTE (intentional divergence from legacy): legacy code passed
            # lattice_angles[:, 1:] (beta/gamma), computing a wrong gamma
            # interval in log_prob(); forward() always used [:, :2]. Sampling
            # is unaffected.
            paddle.concat([lattice_angles[:, :2], noisy_lattice_angles[:, :2]], axis=0),
        )
        normed_lattice_parameters = _normed_gt_noisy_params[:batch_size]
        normed_noisy_lattice_parameters = _normed_gt_noisy_params[batch_size:]

        log_pfs = []
        lattice_mask = paddle.ones([NUM_LATTICE_PARAMS])
        current_lattice_embedding = paddle.concat(
            [
                self.lattice_length_embedder(
                    normed_noisy_lattice_parameters[:, :3].reshape([-1, 3, 1])
                ).reshape([batch_size, -1]),
                self.lattice_angle_embedder(
                    normed_noisy_lattice_parameters[:, 3:].reshape([-1, 3, 1])
                ).reshape([batch_size, -1]),
            ],
            axis=-1,
        )
        for i in range(NUM_LATTICE_PARAMS - 1, -1, -1):
            next_normed_lattice_parameter = normed_lattice_parameters[:, i]
            current_lattice_embedding[
                :, self.lattice_param_dim * i : self.lattice_param_dim * (i + 1)
            ] = 0.0
            lattice_mask[i] = 0.0
            current_state_features = paddle.concat(
                [
                    sg_features,
                    current_lattice_embedding,
                    lattice_mask[None, :].expand([batch_size, NUM_LATTICE_PARAMS]),
                ],
                axis=1,
            )[:, None, :].expand([-1, self.n_bins, -1])
            _, log_prob = self._sample_and_log_prob(
                lattice_param_index=i,
                bin_embedder=self.length_bin_embedder
                if i < 3
                else self.angle_bin_embedder,
                batch_size=batch_size,
                x=next_normed_lattice_parameter,
                z=current_state_features,
            )
            log_pfs.append(log_prob)
        log_pfs = paddle.stack(log_pfs, axis=1)
        column_indices_reversed = paddle.arange(
            start=log_pfs.shape[1] - 1, end=-1, step=-1, dtype=paddle.int64
        )
        log_pfs = log_pfs[:, column_indices_reversed]
        log_pf_masks = self.bravais_log_prob_masks[space_group_indices]
        log_pfs = log_pfs * log_pf_masks
        if self.gradient_attenuation_factor != 1.0:
            log_pfs_detach = log_pfs.detach()
            log_pfs = (
                self.gradient_attenuation_factor * log_pfs
                - self.gradient_attenuation_factor * log_pfs_detach
                + log_pfs_detach
            )
        return log_pfs

    @paddle.no_grad()
    def forward(self, space_group_indices):
        """
        Sample 6 lattice parameters constrained by the Bravais lattices.

        Args:
            space_group_indices: (batch_size,) int64, 0-indexed

        Returns:
            lengths: (batch_size, 3)
            angles: (batch_size, 3)
        """
        batch_size = space_group_indices.shape[0]
        sg_features = self.space_group_encoder(space_group_indices)

        num_lengths = 3
        num_angles = 3
        num_lattice_parameters = NUM_LATTICE_PARAMS

        lattice_params_transform = paddle.concat(
            (
                paddle.to_tensor(
                    [self.MAX_LATTICE_LENGTH - self.MIN_LATTICE_LENGTH]
                ).expand([batch_size, num_lengths]),
                paddle.to_tensor(
                    [self.MAX_LATTICE_ANGLE - self.MIN_LATTICE_ANGLE]
                ).expand([batch_size, num_angles]),
            ),
            axis=1,
        )

        lattice_params_offset = paddle.concat(
            (
                paddle.to_tensor([self.MIN_LATTICE_LENGTH]).expand(
                    [batch_size, num_lengths]
                ),
                paddle.to_tensor([self.MIN_LATTICE_ANGLE]).expand(
                    [batch_size, num_angles]
                ),
            ),
            axis=1,
        )

        normed_lattice_parameters = paddle.zeros([batch_size, num_lattice_parameters])
        current_lattice_embedding = paddle.zeros(
            [batch_size, NUM_LATTICE_PARAMS * self.lattice_param_dim]
        )
        lattice_mask = paddle.zeros([NUM_LATTICE_PARAMS])
        alpha_and_beta_angles = None
        bravais_length_transforms = self.bravais_length_transforms[space_group_indices]
        bravais_angle_transforms = self.bravais_angle_transforms[space_group_indices]
        discretized_normed_bravais_angle_offsets = (
            self.discretized_normed_bravais_angle_offsets[space_group_indices]
        )

        for i in range(num_lattice_parameters):
            if i > 0:
                if i < 4:
                    current_lattice_embedding[
                        :, self.lattice_param_dim * (i - 1) : self.lattice_param_dim * i
                    ] = self.lattice_length_embedder(
                        normed_lattice_parameters[:, i - 1].unsqueeze(-1)
                    )
                else:
                    current_lattice_embedding[
                        :, self.lattice_param_dim * (i - 1) : self.lattice_param_dim * i
                    ] = self.lattice_angle_embedder(
                        normed_lattice_parameters[:, i - 1].unsqueeze(-1)
                    )

            current_state_features = paddle.concat(
                [
                    sg_features,
                    current_lattice_embedding,
                    lattice_mask.unsqueeze(0).expand([batch_size, NUM_LATTICE_PARAMS]),
                ],
                axis=1,
            )[:, None, :].expand([batch_size, self.n_bins, -1])

            if i == 5:
                alpha_and_beta_angles = (
                    (normed_lattice_parameters[:, 3:5] - self.min_bin_edge)
                    / (self.max_bin_edge - self.min_bin_edge)
                ) * lattice_params_transform[:, 3:5] + lattice_params_offset[:, 3:5]

            sample, _ = self._sample_and_log_prob(
                lattice_param_index=i,
                bin_embedder=self.length_bin_embedder
                if i < 3
                else self.angle_bin_embedder,
                batch_size=batch_size,
                z=current_state_features,
                alpha_and_beta_angles=alpha_and_beta_angles,
                enforce_gamma_bounds=(i == (num_lattice_parameters - 1)),
            )

            normed_lattice_parameters[:, i] = sample
            lattice_mask[i] = 1.0

            if i < 3:
                normed_lattice_parameters[:, :3] = paddle.bmm(
                    normed_lattice_parameters[:, :3].unsqueeze(1),
                    bravais_length_transforms,
                ).squeeze(1)
            else:
                normed_lattice_parameters[:, 3:] = (
                    paddle.bmm(
                        normed_lattice_parameters[:, 3:].unsqueeze(1),
                        bravais_angle_transforms,
                    ).squeeze(1)
                    + discretized_normed_bravais_angle_offsets
                )

        lattice_parameters = (
            (normed_lattice_parameters - self.min_bin_edge)
            / (self.max_bin_edge - self.min_bin_edge)
        ) * lattice_params_transform + lattice_params_offset

        lengths = lattice_parameters[:, :num_lengths]
        angles = lattice_parameters[:, num_lengths:]
        return lengths, angles

    def _sample_and_log_prob(
        self,
        lattice_param_index: int,
        bin_embedder,
        batch_size: int,
        z=None,
        x=None,
        alpha_and_beta_angles=None,
        enforce_gamma_bounds: bool = False,
    ):
        if not 0 <= lattice_param_index < NUM_LATTICE_PARAMS:
            raise ValueError(
                f"lattice_param_index must be in [0, {NUM_LATTICE_PARAMS}), "
                f"got {lattice_param_index}"
            )
        if z is None:
            z = paddle.zeros([batch_size, self.n_bins, self.bin_conditioning_info_dim])

        if enforce_gamma_bounds:
            if alpha_and_beta_angles is None:
                raise ValueError(
                    "alpha_and_beta_angles is required when "
                    "enforce_gamma_bounds=True"
                )
            min_gamma, max_gamma = self.get_valid_gamma_angle_interval(
                alpha_and_beta_angles
            )
            min_gamma = min_gamma.unsqueeze(-1)
            max_gamma = max_gamma.unsqueeze(-1)

            min_normed_gamma = self._angle_to_normed(min_gamma)
            max_normed_gamma = self._angle_to_normed(max_gamma)

        _batch_idxs = paddle.arange(batch_size)
        min_bin_edge = self.min_bin_edge * paddle.ones([batch_size])
        max_bin_edge = self.max_bin_edge * paddle.ones([batch_size])
        log_probs = paddle.zeros([batch_size])

        for j in range(self.n_telescopes):
            bin_edges = self._build_bin_edges(min_bin_edge, max_bin_edge)
            bins = paddle.stack([bin_edges[:, :-1], bin_edges[:, 1:]], axis=-1)

            # bin_embedder: (batch_size, n_bins, 2) -> (batch_size, n_bins, hidden_dim)
            bin_emb = bin_embedder(bins.reshape([-1, 2])).reshape(
                [batch_size, self.n_bins, -1]
            )

            bin_logits = self.bin_logit_head(paddle.concat([bin_emb, z], axis=-1))[
                :, :, lattice_param_index
            ]

            if enforce_gamma_bounds:
                if j < self.n_telescopes - 1:
                    epsilon = (bin_edges[0, 1] - bin_edges[0, 0]) / self.n_bins
                    zero_prob_bins_mask = (
                        bins[:, :, 1] - epsilon < min_normed_gamma
                    ) | (bins[:, :, 0] + epsilon > max_normed_gamma)
                else:
                    bin_midpoints = paddle.mean(bins, axis=-1)
                    zero_prob_bins_mask = (bin_midpoints < min_normed_gamma) | (
                        bin_midpoints > max_normed_gamma
                    )

                if paddle.any(paddle.all(zero_prob_bins_mask, axis=-1)):
                    raise RuntimeError(
                        "All bins were masked out by the gamma feasibility "
                        "interval; the noisy lattice parameters are outside "
                        "the valid angle domain"
                    )

                bin_logits = bin_logits + paddle.where(
                    zero_prob_bins_mask,
                    paddle.to_tensor(-float("inf")),
                    paddle.to_tensor(0.0),
                )

            if x is None:
                dist = Categorical(logits=bin_logits)
                bin_idxs = dist.sample([1]).squeeze(0)
            else:
                normalized_x = (x - min_bin_edge) / (max_bin_edge - min_bin_edge)
                bin_idxs = paddle.bucketize(normalized_x, self.grid_pts[1:])
                log_probs = log_probs + paddle.take_along_axis(
                    F.log_softmax(bin_logits, axis=-1),
                    bin_idxs.unsqueeze(-1),
                    axis=-1,
                ).squeeze(-1)

            chosen_bins = bins[_batch_idxs, bin_idxs]
            min_bin_edge = chosen_bins[:, 0]
            max_bin_edge = chosen_bins[:, 1]

        samples = paddle.mean(chosen_bins, axis=-1)
        return samples, log_probs


class MultiheadAttention(nn.MultiHeadAttention):
    """Paddle-native MHA with the call convention used by the model.

    ``paddle.nn.MultiHeadAttention`` exposes the QKV/out projections and the
    scaled dot-product attention, but its native forward lacks two behaviours
    this model relies on: a separate ``key_padding_mask`` argument and a
    NaN-to-zero fallback for fully-masked rows. This subclass keeps the native
    projection layers and re-runs the attention math with the masking
    convention used by the model (bool ``attn_mask`` with ``True`` = masked).

    Note: the native attention-prob dropout path is not re-implemented here,
    so ``dropout_rate`` only applies to the residual/FFN dropouts of
    ``PreLNDecoderLayer``.
    """

    def __init__(self, embed_dim, num_heads, weights_only=False):
        super().__init__(embed_dim=embed_dim, num_heads=num_heads)
        self.weights_only = weights_only
        if weights_only:
            # Released checkpoints retain these unused projection parameters.
            for layer in (self.v_proj, self.out_proj):
                for parameter in layer.parameters():
                    parameter.stop_gradient = True

    def forward(
        self,
        query,
        key,
        value,
        need_weights=False,
        attn_mask=None,
        key_padding_mask=None,
    ):
        batch_size = query.shape[0]
        seq_q = query.shape[1]
        seq_k = key.shape[1]

        q = self.q_proj(query)
        k = self.k_proj(key)

        q = q.reshape([batch_size, seq_q, self.num_heads, self.head_dim]).transpose(
            [0, 2, 1, 3]
        )
        k = k.reshape([batch_size, seq_k, self.num_heads, self.head_dim]).transpose(
            [0, 2, 1, 3]
        )

        scale = math.sqrt(self.head_dim)
        attn_weights = paddle.matmul(q, k.transpose([0, 1, 3, 2])) / scale

        if attn_mask is not None:
            if attn_mask.dtype == paddle.bool:
                attn_weights = paddle.where(
                    attn_mask.unsqueeze(1) if attn_mask.dim() == 3 else attn_mask,
                    paddle.full_like(attn_weights, float("-inf")),
                    attn_weights,
                )
            else:
                attn_weights = attn_weights + attn_mask

        if key_padding_mask is not None:
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            attn_weights = paddle.where(
                mask,
                paddle.full_like(attn_weights, float("-inf")),
                attn_weights,
            )

        attn_weights = F.softmax(attn_weights, axis=-1)
        attn_weights = paddle.where(
            paddle.isnan(attn_weights),
            paddle.zeros_like(attn_weights),
            attn_weights,
        )

        if self.weights_only:
            return None, attn_weights.mean(axis=1)

        v = (
            self.v_proj(value)
            .reshape([batch_size, seq_k, self.num_heads, self.head_dim])
            .transpose([0, 2, 1, 3])
        )
        attn_output = paddle.matmul(attn_weights, v)
        attn_output = attn_output.transpose([0, 2, 1, 3]).reshape(
            [batch_size, seq_q, self.embed_dim]
        )

        output = self.out_proj(attn_output)
        return output, attn_weights.mean(axis=1) if need_weights else None


class SpaceGroupAndLatticeEncoder(nn.Layer):
    """Encode space group index and lattice parameters."""

    def __init__(
        self,
        hidden_dim: int,
        dataset_name: str,
        embedding_tools: "EmbeddingTools",
        lattice_fourier_num_frequencies: int = 64,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.dataset_name = dataset_name
        lattice_param_range_dict = lattice_parameter_ranges[self.dataset_name]
        self.lattice_length_range = (
            lattice_param_range_dict["max_lattice_length"]
            - lattice_param_range_dict["min_lattice_length"]
        )
        self.lattice_angle_range = (
            lattice_param_range_dict["max_lattice_angle"]
            - lattice_param_range_dict["min_lattice_angle"]
        )
        self.space_group_encoder = SpaceGroupEncoder(
            embedding_tools=embedding_tools,
            hidden_channels=self.hidden_dim,
            space_group_embedding_dim=math.floor(self.hidden_dim / 2),
        )
        self.lattice_encoder = FourierLinear(
            input_dim=NUM_LATTICE_PARAMS,
            num_fourier_frequencies=lattice_fourier_num_frequencies,
            scale=1.0,
            output_dim=math.ceil(self.hidden_dim / 2),
            use_bias=True,
        )

    def forward(self, space_group_indices, lattice_lengths, lattice_angles):
        normed_lattice_params = paddle.concat(
            [
                lattice_lengths / self.lattice_length_range,
                lattice_angles / self.lattice_angle_range,
            ],
            axis=-1,
        )
        emb = paddle.concat(
            [
                self.space_group_encoder(space_group_indices),
                self.lattice_encoder(normed_lattice_params),
            ],
            axis=-1,
        )
        return emb


class PreLNDecoderLayer(nn.Layer):
    """Pre-LayerNorm transformer decoder block (self-attention only).

    Named to avoid confusion with ``paddle.nn.TransformerDecoderLayer`` (a
    post-LN layer with cross-attention and different mask semantics).
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        num_heads: int = 4,
        dropout_rate: float = 0.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.layernorm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.mha = MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
        )
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.activation = nn.GELU()

    def forward(self, x, attn_mask=None, key_padding_mask=None, **kwargs):
        x_norm = self.layernorm(x)
        attn_out = self.mha(
            query=x_norm,
            key=x_norm,
            value=x_norm,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
        )[0]
        x = x + attn_out
        x = self.dropout(x)
        x = x + self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout(x)


class WyckoffElementTransformer(nn.Layer):
    """Weight layout matches the released checkpoint format exactly."""

    def __init__(
        self,
        wyckoff_geometry: "WyckoffGeometry",
        embedding_tools: "EmbeddingTools",
        hidden_dim: int = 256,
        dataset_name: str = "mp_20",
        num_heads: int = 2,
        num_hidden_layers: int = 4,
        dropout_rate: float = 0.1,
        lattice_fourier_num_frequencies: int = 64,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_hidden_layers = num_hidden_layers
        self.dropout_rate = dropout_rate
        self.dataset_name = dataset_name

        self.embedding_tools = embedding_tools
        self.wyckoff_geometry = wyckoff_geometry

        if self.hidden_dim % 2 != 0:
            raise ValueError(
                f"hidden_dim must be even (wyckoff/element split), "
                f"got {self.hidden_dim}."
            )
        half_dim = int(self.hidden_dim / 2)
        self.wyckoff_dim = half_dim
        self.element_dim = half_dim

        self.wyckoff_emb = nn.Linear(
            self.embedding_tools.wyckoff_embedding_length, self.wyckoff_dim
        )
        self.element_emb = nn.Linear(
            self.embedding_tools.element_embedding_length, self.element_dim
        )

        nn.initializer.XavierUniform()(self.element_emb.weight)

        self.stop_key = paddle.create_parameter(
            shape=[1, 1, self.wyckoff_dim],
            dtype="float32",
            default_initializer=nn.initializer.Normal(),
        )
        self.seed_wyckoff_embedding = paddle.create_parameter(
            shape=[1, 1, self.wyckoff_dim],
            dtype="float32",
            default_initializer=nn.initializer.Normal(),
        )
        self.seed_element_embedding = paddle.create_parameter(
            shape=[1, 1, self.element_dim],
            dtype="float32",
            default_initializer=nn.initializer.Normal(),
        )

        self.space_group_and_lattice_emb = SpaceGroupAndLatticeEncoder(
            self.hidden_dim,
            dataset_name,
            lattice_fourier_num_frequencies=lattice_fourier_num_frequencies,
            embedding_tools=embedding_tools,
        )

        self.atom_embedder = nn.Sequential(
            nn.Linear(
                self.hidden_dim + self.wyckoff_dim + self.element_dim,
                self.hidden_dim,
            ),
            Swish(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            Swish(),
        )

        self.hidden_layers = nn.LayerList(
            [
                PreLNDecoderLayer(
                    hidden_dim=self.hidden_dim,
                    num_heads=self.num_heads,
                    dropout_rate=self.dropout_rate,
                )
                for _ in range(self.num_hidden_layers)
            ]
        )

        self.wyckoff_mha = MultiheadAttention(
            embed_dim=half_dim,
            num_heads=1,
            weights_only=True,
        )

        d = half_dim + self.wyckoff_dim
        self.mix_xtal_and_wyckoff_mlp = nn.Sequential(
            nn.Linear(d, half_dim),
            Swish(),
            nn.Linear(half_dim, half_dim),
            Swish(),
            nn.Linear(half_dim, half_dim),
            Swish(),
        )

        self.element_keys_mlp = nn.Sequential(
            nn.Linear(self.element_dim, self.element_dim),
            Swish(),
            nn.Linear(self.element_dim, self.element_dim),
            Swish(),
            nn.Linear(self.element_dim, self.element_dim),
            Swish(),
        )

        self.element_mha = MultiheadAttention(
            embed_dim=half_dim,
            num_heads=1,
            weights_only=True,
        )

        _valid_mask = paddle.zeros(
            [NUM_CRYSTALLOGRAPHIC_SPACE_GROUPS, MAX_WYCKOFF_POSITIONS], dtype="bool"
        )
        _zero_dim_mask = paddle.zeros(
            [NUM_CRYSTALLOGRAPHIC_SPACE_GROUPS, MAX_WYCKOFF_POSITIONS], dtype="bool"
        )
        for sg_num in range(1, NUM_CRYSTALLOGRAPHIC_SPACE_GROUPS + 1):
            sg_dict = self.wyckoff_geometry.asu_wyckoff_dict[str(sg_num)]
            sg_wyckoff_letters = sg_dict["ordered_wyckoff_letters"]
            _valid_mask[sg_num - 1, : len(sg_wyckoff_letters)] = True
            for wyckoff_index, wyckoff_letter in enumerate(sg_wyckoff_letters):
                if sg_dict[wyckoff_letter]["dim"] == 0:
                    _zero_dim_mask[sg_num - 1, wyckoff_index] = True
        self.register_buffer("valid_wyckoff_positions_mask", _valid_mask)
        self.register_buffer("zero_dimensional_wyckoff_mask", _zero_dim_mask)

    @paddle.no_grad()
    def sample(
        self,
        lattice_lengths,
        lattice_angles,
        space_group_indices,
        temperature: float = 1.0,
    ):
        """Autoregressively sample Wyckoff positions and elements."""
        if temperature <= 0.0:
            raise ValueError(f"temperature must be positive, got {temperature}.")
        max_allowed_atoms = max_atoms_per_dataset[self.dataset_name]
        n_crystals = space_group_indices.shape[0]

        global_context = self.space_group_and_lattice_emb(
            space_group_indices, lattice_lengths, lattice_angles
        )[:, None, :]

        padded_wyckoff_embeddings = self.seed_wyckoff_embedding.expand(
            [n_crystals, 1, -1]
        )
        padded_element_embeddings = self.seed_element_embedding.expand(
            [n_crystals, 1, -1]
        )

        atom_tokens = paddle.concat(
            [
                global_context,
                padded_wyckoff_embeddings,
                padded_element_embeddings,
            ],
            axis=-1,
        )

        wyckoff_exists = self.valid_wyckoff_positions_mask[space_group_indices]

        # Every space group has at least one Wyckoff site ("a"), so the
        # nonzero of ``wyckoff_exists`` is never empty.
        _existing_positions = paddle.nonzero(wyckoff_exists)
        _xtal_indices = _existing_positions[:, 0]
        _existing_wyckoff_indices = _existing_positions[:, 1]
        wyckoff_keys = self.wyckoff_emb(
            self.embedding_tools.get_wyckoff_embedding(
                wyckoff_index=_existing_wyckoff_indices,
                space_group_index=space_group_indices[_xtal_indices],
            )
        )

        padded_wyckoff_keys = paddle.zeros(
            [n_crystals, MAX_WYCKOFF_POSITIONS, self.wyckoff_dim]
        )
        padded_wyckoff_keys[_xtal_indices, _existing_wyckoff_indices] = wyckoff_keys

        padded_wyckoff_keys = paddle.concat(
            [self.stop_key.expand([n_crystals, 1, -1]), padded_wyckoff_keys],
            axis=1,
        )

        wyckoff_padding_mask = paddle.concat(
            [
                paddle.zeros([n_crystals, 1], dtype="bool"),
                ~wyckoff_exists,
            ],
            axis=1,
        )

        _element_keys = self.element_keys_mlp(
            self.element_emb(
                self.embedding_tools.get_element_embedding(
                    atomic_number=1
                    + paddle.arange(ELEMENT_ENCODING_SIZE, dtype="int64")
                )
            )
        )[None, ...]

        wyckoff_attn_mask = paddle.zeros(
            [n_crystals, 1, 1 + MAX_WYCKOFF_POSITIONS], dtype="bool"
        )
        wyckoff_attn_mask[:, :, 0] = True

        crystal_is_complete = paddle.zeros([n_crystals], dtype="bool")
        n_asu_atoms_per_xtal = paddle.zeros([n_crystals], dtype="int64")
        padded_wyckoff_indices = paddle.full(
            [n_crystals, max_allowed_atoms], fill_value=-1, dtype="int64"
        )
        padded_element_indices = paddle.full(
            [n_crystals, max_allowed_atoms], fill_value=-1, dtype="int64"
        )

        iteration = 0
        arange_n_crystals = paddle.arange(n_crystals, dtype="int64")

        while iteration < max_allowed_atoms and not crystal_is_complete.all():
            atom_tokens = self.atom_embedder(atom_tokens)

            seq_len = atom_tokens.shape[1]
            causal_mask = paddle.triu(
                paddle.ones([seq_len, seq_len], dtype="bool"), diagonal=1
            )

            for layer in self.hidden_layers:
                atom_tokens = layer(
                    atom_tokens,
                    attn_mask=causal_mask,
                )

            incomplete_mask = ~crystal_is_complete
            n_incomplete = int(incomplete_mask.sum())
            _idx = paddle.arange(n_incomplete, dtype="int64")

            last_pos = n_asu_atoms_per_xtal[incomplete_mask]

            atom_tokens_flat = atom_tokens[_idx, last_pos]

            atom_z_wyckoff = atom_tokens_flat[:, : self.wyckoff_dim]
            atom_z_element = atom_tokens_flat[:, self.wyckoff_dim :]

            wyckoff_keys_batch = padded_wyckoff_keys[incomplete_mask]
            wyckoff_padding_batch = wyckoff_padding_mask[incomplete_mask]

            wyckoff_and_stop_probs = self.wyckoff_mha(
                query=atom_z_wyckoff[:, None, :],
                key=wyckoff_keys_batch,
                value=wyckoff_keys_batch,
                need_weights=True,
                key_padding_mask=wyckoff_padding_batch,
                attn_mask=wyckoff_attn_mask,
            )[1]

            if temperature != 1.0:
                wyckoff_and_stop_probs = F.softmax(
                    paddle.log(wyckoff_and_stop_probs + 1e-12) / temperature, axis=-1
                )

            wyckoff_or_stop_sample = paddle.multinomial(
                wyckoff_and_stop_probs.squeeze(1), num_samples=1
            ).squeeze(-1)

            sampled_stop_token = wyckoff_or_stop_sample == 0
            sampled_wyckoff_indices = wyckoff_or_stop_sample[~sampled_stop_token] - 1

            idxs_of_xtals_to_update = arange_n_crystals[incomplete_mask][
                ~sampled_stop_token
            ]
            if idxs_of_xtals_to_update.shape[0] > 0:
                padded_wyckoff_indices[
                    idxs_of_xtals_to_update, iteration
                ] = sampled_wyckoff_indices
                n_asu_atoms_per_xtal[idxs_of_xtals_to_update] += 1

            iteration += 1
            crystal_is_complete[incomplete_mask] = sampled_stop_token

            if (~crystal_is_complete).sum() > 0:
                # At least one crystal is still incomplete, so at least one
                # of them sampled an atom (not a stop token); hence
                # ``sampled_wyckoff_indices`` is non-empty here.
                sampled_wyckoff_embeddings = self.wyckoff_emb(
                    self.embedding_tools.get_wyckoff_embedding(
                        wyckoff_index=sampled_wyckoff_indices,
                        space_group_index=space_group_indices[idxs_of_xtals_to_update],
                    )
                )

                atom_z_element_mixed = self.mix_xtal_and_wyckoff_mlp(
                    paddle.concat(
                        [
                            atom_z_element[~sampled_stop_token],
                            sampled_wyckoff_embeddings,
                        ],
                        axis=-1,
                    )
                )

                element_keys = _element_keys.expand(
                    [atom_z_element_mixed.shape[0], -1, -1]
                )

                (
                    wyckoff_attn_mask,
                    element_attn_mask,
                ) = self.get_lexicographic_attn_masks(
                    space_group_indices[idxs_of_xtals_to_update],
                    padded_wyckoff_indices[
                        idxs_of_xtals_to_update, iteration - 2 : iteration
                    ].reshape([-1])
                    if iteration > 1
                    else sampled_wyckoff_indices,
                    padded_element_indices[
                        idxs_of_xtals_to_update, iteration - 2 : iteration
                    ].reshape([-1])
                    if iteration > 1
                    else paddle.zeros_like(sampled_wyckoff_indices),
                    paddle.full(
                        [
                            sampled_wyckoff_indices.shape[0],
                            2 if iteration > 1 else 1,
                        ],
                        fill_value=True,
                        dtype="bool",
                    ),
                    2 * paddle.ones_like(sampled_wyckoff_indices)
                    if iteration > 1
                    else paddle.ones_like(sampled_wyckoff_indices),
                )
                wyckoff_attn_mask = wyckoff_attn_mask[:, -1, :].unsqueeze(1)
                element_attn_mask = element_attn_mask[:, -1, :].unsqueeze(1)

                element_probs = self.element_mha(
                    query=atom_z_element_mixed[:, None, :],
                    key=element_keys,
                    value=element_keys,
                    need_weights=True,
                    attn_mask=element_attn_mask,
                )[1]

                if temperature != 1.0:
                    element_probs = F.softmax(
                        paddle.log(element_probs + 1e-12) / temperature, axis=-1
                    )

                sampled_element_indices = paddle.multinomial(
                    element_probs.squeeze(1), num_samples=1
                ).squeeze(-1)

                padded_element_indices[
                    idxs_of_xtals_to_update, iteration - 1
                ] = sampled_element_indices

                sampled_element_embeddings = self.element_emb(
                    self.embedding_tools.get_element_embedding(
                        atomic_number=1 + sampled_element_indices
                    )
                )

                padded_sampled_wyckoff = paddle.zeros([n_crystals, self.wyckoff_dim])
                padded_sampled_element = paddle.zeros([n_crystals, self.element_dim])
                padded_sampled_wyckoff[
                    idxs_of_xtals_to_update
                ] = sampled_wyckoff_embeddings
                padded_sampled_element[
                    idxs_of_xtals_to_update
                ] = sampled_element_embeddings

                padded_wyckoff_embeddings = paddle.concat(
                    [
                        padded_wyckoff_embeddings,
                        padded_sampled_wyckoff[:, None, :],
                    ],
                    axis=1,
                )
                padded_element_embeddings = paddle.concat(
                    [
                        padded_element_embeddings,
                        padded_sampled_element[:, None, :],
                    ],
                    axis=1,
                )

                incomplete_mask_new = ~crystal_is_complete
                if incomplete_mask_new.sum() > 0:
                    global_context_expanded = global_context[
                        incomplete_mask_new
                    ].expand([-1, 1 + iteration, -1])
                    atom_tokens = paddle.concat(
                        [
                            global_context_expanded,
                            padded_wyckoff_embeddings[incomplete_mask_new],
                            padded_element_embeddings[incomplete_mask_new],
                        ],
                        axis=-1,
                    )

        atom_mask = (
            paddle.arange(max_allowed_atoms, dtype="int64")[None, :]
            < n_asu_atoms_per_xtal[:, None]
        )

        element_indices = padded_element_indices[atom_mask]
        wyckoff_indices = padded_wyckoff_indices[atom_mask]
        return element_indices, wyckoff_indices, n_asu_atoms_per_xtal

    def forward(
        self,
        space_group_indices: paddle.Tensor,
        lattice_lengths: paddle.Tensor,
        lattice_angles: paddle.Tensor,
        wyckoff_indices: paddle.Tensor,
        element_indices: paddle.Tensor,
        n_asu_atoms_per_xtal: paddle.Tensor,
        atom_mask: paddle.Tensor,
    ):
        """Teacher-forced parallel forward: predict (stop, Wyckoff, element)
        for every position.

        Returns:
            wyckoff_and_stop_probs: (n_crystals, 1+max_atoms, 1+max_wyckoffs)
            element_probs: (n_crystals, max_atoms, num_elements)
        """
        max_atoms = atom_mask.shape[-1]
        batch_size = space_group_indices.shape[0]
        atom_padding_mask = paddle.concat(
            [paddle.zeros([batch_size, 1], dtype="bool"), ~atom_mask], axis=1
        )
        global_context = self.space_group_and_lattice_emb(
            space_group_indices, lattice_lengths, lattice_angles
        )[:, None, :].expand([-1, 1 + max_atoms, -1])
        space_group_indices_per_atom = paddle.repeat_interleave(
            space_group_indices, n_asu_atoms_per_xtal, axis=0
        )
        wyckoff_embeddings = self.wyckoff_emb(
            self.embedding_tools.get_wyckoff_embedding(
                wyckoff_index=wyckoff_indices,
                space_group_index=space_group_indices_per_atom,
            )
        )
        element_embeddings = self.element_emb(
            self.embedding_tools.get_element_embedding(
                atomic_number=1 + element_indices
            )
        )
        padded_wyckoff_embeddings = paddle.zeros(
            [batch_size, max_atoms, wyckoff_embeddings.shape[-1]]
        )
        padded_element_embeddings = paddle.zeros(
            [batch_size, max_atoms, element_embeddings.shape[-1]]
        )
        padded_wyckoff_embeddings[atom_mask] = wyckoff_embeddings
        padded_element_embeddings[atom_mask] = element_embeddings
        padded_wyckoff_embeddings = paddle.concat(
            [
                self.seed_wyckoff_embedding.expand([batch_size, 1, -1]),
                padded_wyckoff_embeddings,
            ],
            axis=1,
        )
        padded_element_embeddings = paddle.concat(
            [
                self.seed_element_embedding.expand([batch_size, 1, -1]),
                padded_element_embeddings,
            ],
            axis=1,
        )
        atom_tokens = paddle.concat(
            [global_context, padded_wyckoff_embeddings, padded_element_embeddings],
            axis=-1,
        )
        atom_tokens = self.atom_embedder(atom_tokens)
        seq_len = atom_tokens.shape[1]
        causal_mask = paddle.triu(
            paddle.ones([seq_len, seq_len], dtype="bool"), diagonal=1
        )
        for layer in self.hidden_layers:
            atom_tokens = layer(
                atom_tokens,
                key_padding_mask=atom_padding_mask,
                attn_mask=causal_mask,
            )
        atom_z_wyckoff = atom_tokens[..., : self.wyckoff_dim]
        atom_z_element = atom_tokens[..., self.wyckoff_dim :]

        stop_key = self.stop_key.expand([batch_size, -1, -1])
        wyckoff_exists = self.valid_wyckoff_positions_mask[space_group_indices]
        wyckoff_padding_mask = paddle.concat(
            [paddle.zeros([batch_size, 1], dtype="bool"), ~wyckoff_exists], axis=1
        )
        _existing_positions = paddle.nonzero(wyckoff_exists)
        _xtal_indices = _existing_positions[:, 0]
        _existing_wyckoff_indices = _existing_positions[:, 1]
        wyckoff_keys = self.wyckoff_emb(
            self.embedding_tools.get_wyckoff_embedding(
                wyckoff_index=_existing_wyckoff_indices,
                space_group_index=space_group_indices[_xtal_indices],
            )
        )
        padded_wyckoff_keys = paddle.zeros(
            [batch_size, MAX_WYCKOFF_POSITIONS, wyckoff_keys.shape[-1]]
        )
        padded_wyckoff_keys[wyckoff_exists] = wyckoff_keys
        padded_wyckoff_keys = paddle.concat([stop_key, padded_wyckoff_keys], axis=1)

        wyckoff_attn_mask, element_attn_mask = self.get_lexicographic_attn_masks(
            space_group_indices,
            wyckoff_indices,
            element_indices,
            atom_mask,
            n_asu_atoms_per_xtal,
        )
        wyckoff_and_stop_probs = self.wyckoff_mha(
            query=atom_z_wyckoff,
            key=padded_wyckoff_keys,
            value=padded_wyckoff_keys,
            need_weights=True,
            key_padding_mask=wyckoff_padding_mask,
            attn_mask=wyckoff_attn_mask,
        )[1]

        element_keys = self.element_keys_mlp(
            self.element_emb(
                self.embedding_tools.get_element_embedding(
                    atomic_number=1
                    + paddle.arange(ELEMENT_ENCODING_SIZE, dtype="int64")
                )
            )
        )[None, ...].expand([batch_size, -1, -1])
        atom_z_element = paddle.concat(
            [atom_z_element[:, :-1, :], padded_wyckoff_embeddings[:, 1:, :]], axis=-1
        )
        atom_z_element = self.mix_xtal_and_wyckoff_mlp(atom_z_element)
        element_probs = self.element_mha(
            query=atom_z_element,
            key=element_keys,
            value=element_keys,
            need_weights=True,
            key_padding_mask=None,
            attn_mask=element_attn_mask,
        )[1]
        return wyckoff_and_stop_probs, element_probs

    def get_lexicographic_attn_masks(
        self,
        space_group_indices: paddle.Tensor,
        wyckoff_indices: paddle.Tensor,
        element_indices: paddle.Tensor,
        atom_mask: paddle.Tensor,
        n_asu_atoms_per_xtal: paddle.Tensor,
    ):
        """Boolean attention masks enforcing lexicographic sampling order."""
        max_atoms = atom_mask.shape[-1]
        batch_size = space_group_indices.shape[0]
        wyckoff_exists_mask = self.valid_wyckoff_positions_mask[space_group_indices]
        wyck_attn_is_allowed = paddle.zeros(
            [batch_size, max_atoms, MAX_WYCKOFF_POSITIONS], dtype="bool"
        )
        _arange = paddle.arange(MAX_WYCKOFF_POSITIONS)[None, :]
        wyckoff_is_0d = self.zero_dimensional_wyckoff_mask[
            space_group_indices.repeat_interleave(n_asu_atoms_per_xtal)
        ]
        wyck_attn_is_allowed[atom_mask] = (_arange > wyckoff_indices[:, None]) | (
            (_arange == wyckoff_indices[:, None]) & ~wyckoff_is_0d
        )
        wyck_attn_is_allowed = paddle.concat(
            [
                paddle.ones([batch_size, 1, MAX_WYCKOFF_POSITIONS], dtype="bool"),
                wyck_attn_is_allowed,
            ],
            axis=1,
        )
        wyck_attn_is_allowed = wyck_attn_is_allowed & wyckoff_exists_mask[:, None, :]
        atom_can_attend_to_stop_token = paddle.ones(
            [batch_size, 1 + max_atoms, 1], dtype="bool"
        )
        atom_can_attend_to_stop_token[:, 0, :] = False
        wyck_attn_is_allowed = paddle.concat(
            [atom_can_attend_to_stop_token, wyck_attn_is_allowed], axis=-1
        )
        wyck_attn_mask = ~wyck_attn_is_allowed

        ele_attn_is_allowed = paddle.zeros(
            [batch_size, max_atoms, ELEMENT_ENCODING_SIZE], dtype="bool"
        )
        ele_attn_is_allowed[atom_mask] = True
        padded_wyckoff_indices = -1 * paddle.ones(
            [batch_size, max_atoms], dtype="int64"
        )
        padded_element_indices = -1 * paddle.ones(
            [batch_size, max_atoms], dtype="int64"
        )
        padded_wyckoff_indices[atom_mask] = wyckoff_indices
        padded_element_indices[atom_mask] = element_indices
        wyckoff_is_tied_with_previous = (
            padded_wyckoff_indices[:, 1:] == padded_wyckoff_indices[:, :-1]
        )
        ele_attn_is_allowed[:, 1:, :] = ~wyckoff_is_tied_with_previous[
            ..., None
        ] | wyckoff_is_tied_with_previous[..., None] & (
            paddle.arange(ELEMENT_ENCODING_SIZE)[None, None, :]
            >= padded_element_indices[:, :-1][..., None]
        )
        ele_attn_mask = ~ele_attn_is_allowed
        return wyck_attn_mask, ele_attn_mask

    def log_prob(
        self,
        element_indices: paddle.Tensor,
        wyckoff_indices: paddle.Tensor,
        n_asu_atoms_per_xtal: paddle.Tensor,
        lattice_lengths: paddle.Tensor,
        lattice_angles: paddle.Tensor,
        space_group_indices: paddle.Tensor,
        max_atoms: int,
    ):
        """Log probabilities of the given elements and Wyckoff positions.

        Args:
            max_atoms: Static padding width for the teacher-forced forward.
                Passing the dataset-wide maximum keeps forward shapes static
                (no GPU->CPU sync, CINN-friendly). Callers must guarantee
                ``n_asu_atoms_per_xtal <= max_atoms`` (the bundled ASU
                datasets satisfy this by construction).
        """
        n_crystals = n_asu_atoms_per_xtal.shape[0]
        n_asu_atoms = element_indices.shape[0]
        atom_mask = (
            paddle.arange(max_atoms, dtype="int64")[None, :]
            < n_asu_atoms_per_xtal[:, None]
        )
        padded_wyckoff_and_stop_probs, padded_element_probs = self(
            space_group_indices,
            lattice_lengths,
            lattice_angles,
            wyckoff_indices,
            element_indices,
            n_asu_atoms_per_xtal,
            atom_mask,
        )
        _atom_idxs = paddle.arange(n_asu_atoms)
        wyckoff_probs = padded_wyckoff_and_stop_probs[
            paddle.arange(1 + max_atoms, dtype="int64")[None, :]
            < n_asu_atoms_per_xtal[:, None]
        ]
        wyckoff_probs = wyckoff_probs[_atom_idxs, 1 + wyckoff_indices]
        _xtal_idxs = paddle.arange(n_crystals)
        termination_probs = padded_wyckoff_and_stop_probs[
            _xtal_idxs, n_asu_atoms_per_xtal, 0
        ]
        element_probs = padded_element_probs[atom_mask]
        element_probs = element_probs[_atom_idxs, element_indices]
        elements_log_prob = paddle.log(1e-12 + element_probs)
        wyckoffs_log_prob = paddle.log(1e-12 + wyckoff_probs)
        termination_log_prob = paddle.log(1e-12 + termination_probs)
        return elements_log_prob, wyckoffs_log_prob, termination_log_prob


def _sigma_norm_cache_path(sigmas, geometry, num_lattice_translations, num_samples):
    identity = {
        "version": 1,
        "geometry": geometry.cache_key,
        "sigmas": hashlib.sha256(sigmas.numpy().tobytes()).hexdigest(),
        "timesteps": sigmas.shape[0],
        "translations": num_lattice_translations,
        "samples": num_samples,
    }
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    return get_data_directory() / f"score_norms-{digest}.pkl"


@paddle.no_grad()
def compute_sigma_norms(
    num_timesteps: int,
    wyckoff_geometry: "WyckoffGeometry",
    sigma_min: float = 0.002,
    sigma_max: float = 0.5,
    num_lattice_translations: int = 3,
    num_monte_carlo_samples: int = 2_500,
) -> paddle.Tensor:
    """Compute the sigma-norm table of shape ``[230, MAX_WYCKOFF, num_timesteps]``.

    Args:
        num_timesteps: Number of diffusion timesteps (the table's last dim).
        sigma_min / sigma_max: Diffusion sigma range.
        num_lattice_translations: Lattice-translation neighbors for the
            wrapped-normal expectation.
        num_monte_carlo_samples: MC samples per (space group, Wyckoff site).

    Returns:
        Table of shape ``[NUM_CRYSTALLOGRAPHIC_SPACE_GROUPS, MAX_WYCKOFF_POSITIONS,
        num_timesteps]``.  The caller prepends a leading ``ones`` column to
        align with the scheduler's time index.
    """
    sigmas = build_ve_sigma_grid(num_timesteps, sigma_min, sigma_max)
    cache_path = _sigma_norm_cache_path(
        sigmas, wyckoff_geometry, num_lattice_translations, num_monte_carlo_samples
    )
    if cache_path.is_file():
        try:
            with cache_path.open("rb") as file_obj:
                cached = pickle.load(file_obj)
            expected_shape = (
                NUM_CRYSTALLOGRAPHIC_SPACE_GROUPS,
                MAX_WYCKOFF_POSITIONS,
                num_timesteps,
            )
            if cached.shape != expected_shape or not np.isfinite(cached).all():
                raise ValueError("Invalid cached score-normalization table.")
            return paddle.to_tensor(cached, dtype=paddle.float32)
        except (AttributeError, EOFError, OSError, pickle.UnpicklingError, ValueError):
            logger.warning(
                f"Rebuilding invalid score-normalization cache: {cache_path}"
            )
    logger.info(f"Computing score-normalization cache: {cache_path}")
    wyckoff_shape_decomp_dict = load_wyckoff_shapes(wyckoff_geometry)

    asu_wyckoff_dict = wyckoff_geometry.asu_wyckoff_dict
    sigma_norms = paddle.zeros(
        [NUM_CRYSTALLOGRAPHIC_SPACE_GROUPS, MAX_WYCKOFF_POSITIONS, num_timesteps],
        dtype=paddle.float32,
    )

    # Reverse iteration order is kept from the upstream implementation so the
    # global RNG draw sequence (and thus recomputed tables) stays reproducible
    # against previously cached .pdparams tables.
    for sg_num in range(NUM_CRYSTALLOGRAPHIC_SPACE_GROUPS, 0, -1):
        sg_dict = asu_wyckoff_dict[str(sg_num)]
        wyckoff_letters = sg_dict["ordered_wyckoff_letters"]

        x0s, wsi = sample_point_in_asu_wyckoff_site(
            space_group_numbers=[str(sg_num)] * len(wyckoff_letters),
            wyckoff_letters=wyckoff_letters,
            dictionary_of_wyckoffs_in_asu=asu_wyckoff_dict,
            dictionary_of_wyckoff_shape_decompositions=wyckoff_shape_decomp_dict,
            hull_equations_3d=wyckoff_geometry.asu_hull_equations,
            n_samples_per_wyckoff=num_monte_carlo_samples,
        )

        space_group_idx = paddle.to_tensor([sg_num - 1], dtype=paddle.int64)

        for i in range(len(wyckoff_letters)):
            _wyckoff_idx = paddle.to_tensor([i], dtype=paddle.int64)
            _wyckoff_idx_expanded = _wyckoff_idx.expand([num_monte_carlo_samples])

            sg_ops = get_space_group_ops_and_conventional_atoms(
                x0s[i],
                paddle.zeros_like(_wyckoff_idx_expanded),
                _wyckoff_idx_expanded,
                space_group_idx,
                n_atoms_per_xtal=paddle.to_tensor(
                    [num_monte_carlo_samples], dtype=paddle.int64
                ),
                wyckoff_geometry=wyckoff_geometry,
            )
            map_conv_to_asu = sg_ops.map_conventional_to_asu_atom
            orbited_x = sg_ops.conventional_frac_coords
            unique_indices = sg_ops.unique_non_overlapping_atom_indices
            map_unique_conv_to_asu = map_conv_to_asu[unique_indices]

            # Cap the chunk count so each chunked sigma batch bounds peak
            # memory of the MC estimate.
            _chunks = min(200, num_timesteps)
            for sigma_idxs in paddle.chunk(
                paddle.arange(num_timesteps), chunks=_chunks
            ):
                batch_sigmas = sigmas[sigma_idxs]
                norms = []
                for sigma in batch_sigmas.tolist():
                    noise = paddle.randn([num_monte_carlo_samples, 3]) * sigma
                    xts = x0s[i] + noise

                    scores = d_log_p_asu_wrapped_normal(
                        xts,
                        orbited_x,
                        map_unique_conv_to_asu,
                        num_lattice_translations,
                        sigma,
                    )
                    norm_t = ((scores**2).sum(axis=-1)).sqrt().mean()
                    norms.append(norm_t.item())

                sigma_norms[sg_num - 1, i, sigma_idxs] = paddle.to_tensor(
                    norms, dtype=paddle.float32
                )

    save_asu_cache(cache_path, sigma_norms.numpy())
    logger.info(f"Saved sigma norms to: {cache_path}")
    return sigma_norms


class EquivariantDiffusionModel(RuntimeMixin, nn.Layer):
    """Space-group equivariant diffusion model modeling crystal coords in ASU."""

    def __init__(
        self,
        model_type: str = "gnn",
        num_lattice_translations: int = 3,
        noise_scheduler_num_monte_carlo_samples: int = 2_500,
        num_timesteps: int = 1000,
        sigma_min: float = 0.002,
        sigma_max: float = 0.5,
        time_emb_dim: int = 128,
        num_plane_wave_freqs: int = 96,
        mlp_hidden_dim: int = 128,
        gnn_config: Optional[dict] = None,
        cspnet_config: Optional[dict] = None,
        noise_scheduler_cfg: Optional[dict] = None,
        wyckoff_geometry: "WyckoffGeometry" = None,
        embedding_tools: "EmbeddingTools" = None,
        execution_backend: str = "eager",
        runtime_options: Optional[dict] = None,
    ):
        super().__init__()
        self._init_runtime(execution_backend, runtime_options)
        if sigma_min <= 0.0:
            raise ValueError(f"sigma_min must be positive, got {sigma_min}")
        if not isinstance(time_emb_dim, int) or time_emb_dim <= 0:
            raise ValueError(f"time_emb_dim must be a positive int, got {time_emb_dim}")
        self.num_lattice_translations = num_lattice_translations

        if noise_scheduler_cfg is None:
            noise_scheduler_cfg = {
                "__class_name__": "ASUVESDEScheduler",
                "__init_params__": {
                    "num_timesteps": num_timesteps,
                    "sigma_min": sigma_min,
                    "sigma_max": sigma_max,
                },
            }
        self.noise_scheduler = build_scheduler(noise_scheduler_cfg)
        sigma_norms = compute_sigma_norms(
            num_timesteps=self.noise_scheduler.num_timesteps,
            wyckoff_geometry=wyckoff_geometry,
            sigma_min=self.noise_scheduler.sigma_min,
            sigma_max=self.noise_scheduler.sigma_max,
            num_lattice_translations=num_lattice_translations,
            num_monte_carlo_samples=noise_scheduler_num_monte_carlo_samples,
        )
        self.register_buffer(
            "sigma_norms",
            paddle.concat(
                [paddle.ones(list(sigma_norms.shape[:-1]) + [1]), sigma_norms],
                axis=-1,
            ),
        )
        self.time_embedder = SinusoidalTimeEmbeddings(dim=time_emb_dim)
        self.wyckoff_geometry = wyckoff_geometry
        self.embedding_tools = embedding_tools
        if model_type == "mlp":
            self.non_equivariant_drift_model = TorusMLP(
                self.time_embedder, num_plane_wave_freqs, mlp_hidden_dim
            )
        elif model_type == "gnn":
            self.non_equivariant_drift_model = GNN(
                time_embedder=self.time_embedder,
                embedding_tools=embedding_tools,
                **(gnn_config or {}),
            )
        elif model_type == "cspnet":
            self.non_equivariant_drift_model = CSPNet(
                time_embedder=self.time_embedder, **(cspnet_config or {})
            )
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        padded_hull_equations = self.wyckoff_geometry.padded_hull_equations
        mask_padded_hull_equations = self.wyckoff_geometry.padded_hull_equations_mask

        self.register_buffer("padded_hull_equations", padded_hull_equations)
        self.register_buffer("padded_hull_equations_mask", mask_padded_hull_equations)

        self._wyckoff_shape_decomposition_dict = None

    @property
    def wyckoff_shape_decomposition_dict(self) -> dict:
        """Load shape decompositions for this model's geometry on demand."""
        if self._wyckoff_shape_decomposition_dict is None:
            self._wyckoff_shape_decomposition_dict = load_wyckoff_shapes(
                self.wyckoff_geometry
            )
        return self._wyckoff_shape_decomposition_dict

    def compute_loss(
        self,
        asu_frac_coords: paddle.Tensor,
        element_indices: paddle.Tensor,
        wyckoff_indices: paddle.Tensor,
        space_group_indices: paddle.Tensor,
        n_atoms_per_xtal: paddle.Tensor,
        wyckoff_shape_indices: paddle.Tensor,
        lattice_matrices: paddle.Tensor,
        lattice_lengths: paddle.Tensor,
        lattice_angles: paddle.Tensor,
    ) -> paddle.Tensor:
        """Compute training loss (score matching)."""
        sampled_timesteps = self.noise_scheduler.uniform_sample_timestep(
            batch_size=space_group_indices.shape[0]
        )
        time_embeddings = self.time_embedder(sampled_timesteps.cast(paddle.float32))

        sampled_timesteps = sampled_timesteps.repeat_interleave(
            n_atoms_per_xtal, axis=0
        )
        time_embeddings = time_embeddings.repeat_interleave(n_atoms_per_xtal, axis=0)
        sigmas = self.noise_scheduler.sigmas[sampled_timesteps].detach()

        _sg_per_asu_atom = space_group_indices.repeat_interleave(
            n_atoms_per_xtal, axis=0
        )

        with paddle.no_grad():
            projected_noise = get_wyckoff_projected_gaussian_noise(
                space_group_indices,
                wyckoff_indices,
                wyckoff_shape_indices,
                n_atoms_per_xtal,
                sigmas.unsqueeze(1),
                self.wyckoff_geometry,
            )
            noisy_asu_frac_coords = asu_frac_coords.detach() + projected_noise

            (noisy_asu_frac_coords, wyckoff_shape_indices,) = wrap_frac_coords_into_asu(
                noisy_asu_frac_coords,
                space_group_indices,
                n_atoms_per_xtal,
                self.padded_hull_equations[_sg_per_asu_atom, wyckoff_indices].clone(),
                self.padded_hull_equations_mask[
                    _sg_per_asu_atom, wyckoff_indices
                ].clone(),
                self.wyckoff_geometry,
            )

            sg_ops = get_space_group_ops_and_conventional_atoms(
                asu_frac_coords,
                element_indices,
                wyckoff_indices,
                space_group_indices,
                n_atoms_per_xtal,
                self.wyckoff_geometry,
            )
            map_conventional_to_asu_atom = sg_ops.map_conventional_to_asu_atom
            conventional_frac_coords = sg_ops.conventional_frac_coords
            unique_non_overlapping_atom_indices = (
                sg_ops.unique_non_overlapping_atom_indices
            )
            map_unique_conventional_to_asu = map_conventional_to_asu_atom[
                unique_non_overlapping_atom_indices
            ]

            ground_truth_scores = d_log_p_asu_wrapped_normal(
                noisy_asu_frac_coords,
                conventional_frac_coords,
                map_unique_conventional_to_asu,
                self.num_lattice_translations,
                sigmas,
            )

        # Avoid Paddle's AssignOutGradNode issue with shared computation graphs
        noisy_asu_frac_coords = noisy_asu_frac_coords.detach()
        wyckoff_shape_indices = wyckoff_shape_indices.detach()
        ground_truth_scores = ground_truth_scores.detach()
        _sg_per_asu_atom = _sg_per_asu_atom.detach()
        sampled_timesteps = sampled_timesteps.detach()

        predicted_scores = self.predict_equivariant_vectors(
            time_embeddings,
            noisy_asu_frac_coords,
            element_indices,
            wyckoff_indices,
            space_group_indices,
            n_atoms_per_xtal,
            lattice_matrices=lattice_matrices,
            lattice_lengths=lattice_lengths,
            lattice_angles=lattice_angles,
        )

        score_norms = self.sigma_norms[
            _sg_per_asu_atom, wyckoff_indices, sampled_timesteps
        ]
        loss = F.mse_loss(
            predicted_scores,
            ground_truth_scores / (score_norms.unsqueeze(1) + 1e-8),
        )
        return loss

    @paddle.no_grad()
    def sample(
        self,
        wyckoff_indices: paddle.Tensor,
        element_indices: paddle.Tensor,
        space_group_indices: paddle.Tensor,
        n_atoms_per_xtal: paddle.Tensor,
        lattice_matrices: paddle.Tensor,
        lattice_lengths: paddle.Tensor,
        lattice_angles: paddle.Tensor,
        snr: float = 0.4,
        max_step_size: float = 1e6,
    ) -> paddle.Tensor:
        """Variance-exploding Predictor-Corrector SDE sampling."""
        num_asu_atoms: int = wyckoff_indices.shape[0]
        time_start: int = self.noise_scheduler.num_timesteps

        space_group_indices_per_atom = space_group_indices.repeat_interleave(
            n_atoms_per_xtal, axis=0
        )

        space_group_numbers = [
            str(1 + int(sg_idx)) for sg_idx in space_group_indices_per_atom.tolist()
        ]
        wyckoff_letters = [
            self.wyckoff_geometry.asu_wyckoff_dict[sg_num]["ordered_wyckoff_letters"][
                int(w_idx)
            ]
            for sg_num, w_idx in zip(space_group_numbers, wyckoff_indices.tolist())
        ]

        x_T, wyckoff_shape_indices = sample_point_in_asu_wyckoff_site(
            space_group_numbers=space_group_numbers,
            wyckoff_letters=wyckoff_letters,
            dictionary_of_wyckoffs_in_asu=self.wyckoff_geometry.asu_wyckoff_dict,
            dictionary_of_wyckoff_shape_decompositions=self.wyckoff_shape_decomposition_dict,
            hull_equations_3d=self.wyckoff_geometry.asu_hull_equations,
            n_samples_per_wyckoff=1,
        )
        x_T = x_T.squeeze(1)
        wyckoff_shape_indices = wyckoff_shape_indices.squeeze(1)

        wyckoff_dims = self.wyckoff_geometry.wyckoff_dimension_tensor[
            space_group_indices_per_atom, wyckoff_indices
        ]
        proj_matrices = self.wyckoff_geometry.noise_projection_matrices[
            space_group_indices_per_atom, wyckoff_indices, wyckoff_shape_indices
        ]

        def _predict_and_project(x, sigma_norm, t):
            te = self.time_embedder(paddle.to_tensor([t], dtype=paddle.float32)).expand(
                [num_asu_atoms, -1]
            )
            score = sigma_norm.unsqueeze(1) * self.predict_equivariant_vectors(
                te,
                x,
                element_indices,
                wyckoff_indices,
                space_group_indices,
                n_atoms_per_xtal,
                lattice_matrices=lattice_matrices,
                lattice_lengths=lattice_lengths,
                lattice_angles=lattice_angles,
            )
            return paddle.bmm(score.unsqueeze(1), proj_matrices).reshape([-1, 3])

        def _wyckoff_noise():
            return get_wyckoff_projected_gaussian_noise(
                space_group_indices,
                wyckoff_indices,
                wyckoff_shape_indices,
                n_atoms_per_xtal,
                1.0,
                self.wyckoff_geometry,
            )

        x_t_plus_1 = x_T
        for t in tqdm(
            range(time_start - 1, 0, -1),
            desc="reverse diffusion",
            disable=time_start <= 2,
        ):
            sn_t1 = self.sigma_norms[
                space_group_indices_per_atom, wyckoff_indices, t + 1
            ]
            sn_t = self.sigma_norms[space_group_indices_per_atom, wyckoff_indices, t]

            score_pred = _predict_and_project(x_t_plus_1, sn_t1, t)
            x_t = self.noise_scheduler.step_pred(
                x_t_plus_1, score_pred, t, _wyckoff_noise()
            )

            score_corr = _predict_and_project(x_t, sn_t, t)
            x_t = self.noise_scheduler.step_correct(
                x_t, score_corr, _wyckoff_noise(), snr, max_step_size
            )

            x_t = self.project_point_onto_wyckoff_shape(
                x_t,
                space_group_indices_per_atom,
                wyckoff_indices,
                wyckoff_shape_indices,
                wyckoff_dims,
                self.wyckoff_geometry,
            )
            x_t_plus_1 = x_t

        x_final = x_t_plus_1 % 1.0
        x_final, _ = wrap_frac_coords_into_asu(
            x_final,
            space_group_indices,
            n_atoms_per_xtal,
            self.padded_hull_equations[space_group_indices_per_atom, wyckoff_indices],
            self.padded_hull_equations_mask[
                space_group_indices_per_atom, wyckoff_indices
            ],
            self.wyckoff_geometry,
        )
        return x_final.unsqueeze(0)

    def forward(self, batch: Dict) -> Dict:
        """Training entry: unpack batch, compute loss, return
        BaseTrainer-compatible dict."""
        if "lattice_matrices" in batch:
            lattice_matrices = batch["lattice_matrices"]
        else:
            lattice_matrices = lattice_params_to_matrix_paddle(
                batch["lattice_lengths"], batch["lattice_angles"]
            )
        loss = self.compute_loss(
            asu_frac_coords=batch["frac_coords"],
            element_indices=batch["element_indices"],
            wyckoff_indices=batch["wyckoff_indices"],
            space_group_indices=batch["space_group_indices"],
            n_atoms_per_xtal=batch["n_atoms_per_asu"],
            wyckoff_shape_indices=batch["wyckoff_shape_indices"],
            lattice_matrices=lattice_matrices,
            lattice_lengths=batch["lattice_lengths"],
            lattice_angles=batch["lattice_angles"],
        )
        return {"loss_dict": {"loss": loss}}

    def predict_equivariant_vectors(
        self,
        time_embeddings: paddle.Tensor,
        frac_coords: paddle.Tensor,
        element_indices: paddle.Tensor,
        wyckoff_indices: paddle.Tensor,
        space_group_indices: paddle.Tensor,
        n_atoms_per_xtal: paddle.Tensor,
        lattice_matrices: Optional[paddle.Tensor] = None,
        lattice_lengths: Optional[paddle.Tensor] = None,
        lattice_angles: Optional[paddle.Tensor] = None,
    ) -> paddle.Tensor:
        """Equivariant vector field: mean(A^-1 @ f(Ax + t))."""
        frac_coords = frac_coords % 1.0

        sg_ops = get_space_group_ops_and_conventional_atoms(
            frac_coords,
            element_indices,
            wyckoff_indices,
            space_group_indices,
            n_atoms_per_xtal,
            self.wyckoff_geometry,
        )
        map_conventional_to_asu_atom = sg_ops.map_conventional_to_asu_atom
        conventional_wyckoff_indices = sg_ops.conventional_wyckoff_indices
        conventional_element_indices = sg_ops.conventional_element_indices
        frac_coords_of_conv_atoms = sg_ops.conventional_frac_coords
        unique_non_overlapping_atom_indices = sg_ops.unique_non_overlapping_atom_indices

        map_asu_atom_to_xtal = paddle.arange(
            space_group_indices.shape[0]
        ).repeat_interleave(n_atoms_per_xtal, axis=0)
        map_unique_conventional_to_asu = map_conventional_to_asu_atom[
            unique_non_overlapping_atom_indices
        ]

        n_conv_atoms_per_xtal = scatter(
            src=paddle.ones(
                [conventional_wyckoff_indices.shape[0]], dtype=paddle.int64
            ),
            index=map_asu_atom_to_xtal[map_unique_conventional_to_asu],
            dim=0,
            dim_size=space_group_indices.shape[0],
            reduce="sum",
        )

        drift = self.non_equivariant_drift_model
        with paddle.no_grad():
            graph_inputs = drift.construct_graph_inputs(
                frac_coords=frac_coords_of_conv_atoms,
                n_atoms_per_xtal=n_conv_atoms_per_xtal,
                lattice_matrices=lattice_matrices,
                lattice_lengths=lattice_lengths,
                lattice_angles=lattice_angles,
            )
        return self._runtime_denoise_step(
            frac_coords=frac_coords_of_conv_atoms,
            element_indices=conventional_element_indices,
            time_embeddings=time_embeddings[map_unique_conventional_to_asu],
            n_atoms_per_xtal=n_conv_atoms_per_xtal,
            graph_inputs=graph_inputs,
            A_inv_ops=sg_ops.A_inv_ops,
            inverse_indices=sg_ops.inverse_indices,
            map_conventional_to_asu_atom=map_conventional_to_asu_atom,
            n_asu_atoms=frac_coords.shape[0],
        )

    @runtime_boundary("denoise_step")
    def _runtime_denoise_step(
        self,
        frac_coords: paddle.Tensor,
        element_indices: paddle.Tensor,
        time_embeddings: paddle.Tensor,
        n_atoms_per_xtal: paddle.Tensor,
        graph_inputs: tuple,
        A_inv_ops: paddle.Tensor,
        inverse_indices: paddle.Tensor,
        map_conventional_to_asu_atom: paddle.Tensor,
        n_asu_atoms: int,
    ) -> paddle.Tensor:
        """Compiled numerical core of one coordinate-denoise step."""
        non_equivariant_output = self.non_equivariant_drift_model.forward_with_graph(
            frac_coords=frac_coords,
            element_indices=element_indices,
            time_embeddings=time_embeddings,
            n_atoms_per_xtal=n_atoms_per_xtal,
            graph_inputs=graph_inputs,
        )

        src = paddle.bmm(
            non_equivariant_output[inverse_indices].unsqueeze(1),
            A_inv_ops,
        ).squeeze(1)

        vector_field = scatter(
            src=src,
            index=map_conventional_to_asu_atom,
            dim=0,
            dim_size=n_asu_atoms,
            reduce="mean",
        )
        return vector_field

    @staticmethod
    def project_point_onto_wyckoff_shape(
        points: paddle.Tensor,
        space_group_indices_repeated: paddle.Tensor,
        wyckoff_indices: paddle.Tensor,
        wyckoff_shape_indices: paddle.Tensor,
        wyckoff_dims: paddle.Tensor,
        wyckoff_geometry: "WyckoffGeometry",
    ) -> paddle.Tensor:
        """Project points to Wyckoff subspace (1D->line, 2D->plane)."""

        def project_to_lines(pts, p0s, dirs):
            dirs = dirs / (dirs.norm(axis=-1, keepdim=True) + 1e-12)
            v = pts - p0s
            t = (v * dirs).sum(axis=-1, keepdim=True)
            return p0s + t * dirs

        def project_to_planes(pts, p0s, normals):
            normals = normals / (normals.norm(axis=-1, keepdim=True) + 1e-12)
            v = pts - p0s
            dist = (v * normals).sum(axis=-1, keepdim=True)
            return pts - dist * normals

        wyckoff_dim_is_1 = wyckoff_dims == 1
        wyckoff_dim_is_2 = wyckoff_dims == 2

        if wyckoff_dim_is_1.any():
            mask = wyckoff_dim_is_1
            pts_1d = project_to_lines(
                points[mask],
                wyckoff_geometry.point_per_1d_wyckoff_line[
                    space_group_indices_repeated[mask],
                    wyckoff_indices[mask],
                    wyckoff_shape_indices[mask],
                ],
                wyckoff_geometry.line_directions_of_1d_wyckoffs[
                    space_group_indices_repeated[mask],
                    wyckoff_indices[mask],
                    wyckoff_shape_indices[mask],
                ],
            )
            points = paddle.scatter(
                points, paddle.where(mask)[0], pts_1d, overwrite=True
            )

        if wyckoff_dim_is_2.any():
            mask = wyckoff_dim_is_2
            pts_2d = project_to_planes(
                points[mask],
                wyckoff_geometry.point_per_2d_wyckoff_plane[
                    space_group_indices_repeated[mask],
                    wyckoff_indices[mask],
                    wyckoff_shape_indices[mask],
                ],
                wyckoff_geometry.plane_normals_of_2d_wyckoffs[
                    space_group_indices_repeated[mask],
                    wyckoff_indices[mask],
                    wyckoff_shape_indices[mask],
                ],
            )
            points = paddle.scatter(
                points, paddle.where(mask)[0], pts_2d, overwrite=True
            )

        return points


def _as_tensor(x, dtype=None):
    """Convert numpy inputs to tensors; pass tensors through without copying."""
    if isinstance(x, paddle.Tensor):
        return x.cast(dtype) if dtype is not None else x
    return paddle.to_tensor(x, dtype=dtype)


class SpaceGroupSampler(nn.Layer):
    def __init__(self):
        super().__init__()
        self.marginal_space_group_logits = paddle.create_parameter(
            shape=[NUM_CRYSTALLOGRAPHIC_SPACE_GROUPS],
            dtype="float32",
            default_initializer=nn.initializer.Constant(1.0),
        )

    def sample(self, batch_size: int = 1, temperature: float = 1.0) -> paddle.Tensor:
        log_probs = F.log_softmax(
            self.marginal_space_group_logits / temperature, axis=-1
        )
        log_probs = log_probs.unsqueeze(0).expand([batch_size, -1])
        return Categorical(logits=log_probs).sample([1]).squeeze(0)

    def log_prob(self, space_group_indices):
        normed_logits = F.log_softmax(self.marginal_space_group_logits, axis=-1)
        return normed_logits[space_group_indices]


class SGEQuiDiff(RuntimeMixin, nn.Layer):
    """Full crystal sampler combining all submodules.

    This is the unified model entry: ``forward(batch)`` returns the
    training loss dict (delegating to the internal coordinate diffusion model),
    and ``sample(batch_data)`` returns structures compatible with
    ``structure_generation/sample.py``.

    SGEQuiDiff is an **unconditional** generator:
    ``sample_by_num_atoms`` raises ``NotImplementedError`` (it declares
    ``supports_num_atoms_sampling = False``), and ``sample`` rejects atom types
    passed by ``sample_by_chemical_formula``; use
    ``sample_by_dataloader`` instead.

    Args:
        dataset_name: Dataset name, e.g. ``"mp_20"``.
        diffusion_snr: SNR for the diffusion sampling step.
        temperature: Sampling temperature.
        num_timesteps: Number of diffusion timesteps.
        noise_scheduler_num_monte_carlo_samples: MC samples for sigma norms.
        num_lattice_translations: Lattice-translation neighbors for the
            wrapped-normal noise model.
        sigma_min: Lower bound of the VE-SDE noise schedule.
        sigma_max: Upper bound of the VE-SDE noise schedule.
        model_type: Coordinate drift backbone type (``"gnn"``/``"mlp"``/``"cspnet"``).
        time_emb_dim: Time embedding dimension.
        num_plane_wave_freqs: Number of plane wave frequencies.
        gnn_config: GNN backbone overrides (plain dict of constructor kwargs).
        cspnet_config: CSPNet backbone overrides (plain dict of constructor kwargs).
        noise_scheduler_cfg: Noise scheduler config dict.
        vocab: Prebuilt ``sgequidiff`` vocabulary dict. ``None`` (default)
            resolves it internally via ``ppmat.vocab.build_vocab``, with no
            framework-level wiring required.
        lattice_length_noise: Noise added to lattice lengths during training.
        lattice_angle_noise: Noise added to lattice angles during training.
        space_group_grad_weight: Loss weight for the space-group log-prob.
        lattice_grad_weight: Loss weight for the lattice log-prob.
        wyckoff_element_grad_weight: Loss weight for the wyckoff/element log-prob.
        frac_coord_grad_weight: Loss weight for the coordinate score matching.
        lattice_sampler_config: Overrides for the lattice sampler constructor
            fields. Keys must be valid constructor parameter names; unknown
            keys raise a ``TypeError``. ``None`` uses the defaults, which
            match the released checkpoint structure.
        wyckoff_element_transformer_config: Overrides for the
            Wyckoff-element transformer constructor fields, same semantics as
            ``lattice_sampler_config``.
        execution_backend: Numerical execution backend, ``"eager"`` or
            ``"cinn"``. Forwarded to the coordinate diffusion model, which
            owns the compiled runtime. Defaults to ``"eager"``.
        runtime_options: Per-backend runtime options forwarded to the
            coordinate diffusion model.
        lattice_lr_multiplier: Learning-rate multiplier for lattice parameters.
        wyckoff_element_lr_multiplier: Learning-rate multiplier for the
            autoregressive element and Wyckoff parameters.
    """

    supports_num_atoms_sampling = False

    def __init__(
        self,
        dataset_name: str = "mp_20",
        diffusion_snr: float = 0.4,
        temperature: float = 1.0,
        num_timesteps: int = 1000,
        noise_scheduler_num_monte_carlo_samples: int = 2500,
        num_lattice_translations: int = 3,
        sigma_min: float = 0.002,
        sigma_max: float = 0.5,
        model_type: str = "gnn",
        time_emb_dim: int = 128,
        num_plane_wave_freqs: int = 96,
        gnn_config: Any = None,
        cspnet_config: Any = None,
        noise_scheduler_cfg: Optional[dict] = None,
        lattice_length_noise: Optional[float] = 0.0,
        lattice_angle_noise: Optional[float] = 0.0,
        space_group_grad_weight: Optional[float] = 1.0,
        lattice_grad_weight: Optional[float] = 1.0,
        wyckoff_element_grad_weight: Optional[float] = 1.0,
        frac_coord_grad_weight: Optional[float] = 1.0,
        lattice_sampler_config: Optional[Dict[str, Any]] = None,
        wyckoff_element_transformer_config: Optional[Dict[str, Any]] = None,
        lattice_lr_multiplier: float = 0.1,
        wyckoff_element_lr_multiplier: float = 0.1,
        execution_backend: str = "eager",
        runtime_options: Optional[dict] = None,
        vocab: Optional[dict] = None,
    ):
        super().__init__()
        self.dataset_name = dataset_name
        self.diffusion_snr = diffusion_snr
        self.temperature = temperature

        lr = lattice_parameter_ranges[dataset_name]

        if gnn_config is None:
            gnn_config = {"dataset_name": dataset_name}

        lattice_kwargs = {
            "min_lattice_length": lr["min_lattice_length"],
            "max_lattice_length": lr["max_lattice_length"],
            "min_lattice_angle": lr["min_lattice_angle"],
            "max_lattice_angle": lr["max_lattice_angle"],
        }
        if lattice_sampler_config is not None:
            lattice_kwargs.update(lattice_sampler_config)
        we_kwargs = {"dataset_name": dataset_name}
        if wyckoff_element_transformer_config is not None:
            we_kwargs.update(wyckoff_element_transformer_config)

        self.lattice_length_noise = lattice_length_noise
        self.lattice_angle_noise = lattice_angle_noise
        self.space_group_grad_weight = space_group_grad_weight
        self.lattice_grad_weight = lattice_grad_weight
        self.wyckoff_element_grad_weight = wyckoff_element_grad_weight
        self.frac_coord_grad_weight = frac_coord_grad_weight

        # Build static resources explicitly (no global singletons).
        self.wyckoff_geometry = build_wyckoff_geometry(vocab)
        self.embedding_tools = build_embedding_tools(vocab)

        self.atom_coord_diffusion_model = EquivariantDiffusionModel(
            model_type=model_type,
            num_timesteps=num_timesteps,
            noise_scheduler_num_monte_carlo_samples=noise_scheduler_num_monte_carlo_samples,
            num_lattice_translations=num_lattice_translations,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            time_emb_dim=time_emb_dim,
            num_plane_wave_freqs=num_plane_wave_freqs,
            gnn_config=gnn_config,
            cspnet_config=cspnet_config,
            noise_scheduler_cfg=noise_scheduler_cfg,
            wyckoff_geometry=self.wyckoff_geometry,
            embedding_tools=self.embedding_tools,
            execution_backend=execution_backend,
            runtime_options=runtime_options,
        )
        self.space_group_sampler = SpaceGroupSampler()
        self.lattice_sampler = TelescopingDiscreteLatticeSampler(
            embedding_tools=self.embedding_tools,
            **lattice_kwargs,
        )
        self.wyckoff_and_element_sampler = WyckoffElementTransformer(
            wyckoff_geometry=self.wyckoff_geometry,
            embedding_tools=self.embedding_tools,
            **we_kwargs,
        )
        for sampler, multiplier in (
            (self.lattice_sampler, lattice_lr_multiplier),
            (self.wyckoff_and_element_sampler, wyckoff_element_lr_multiplier),
        ):
            if multiplier <= 0:
                raise ValueError("Sampler learning-rate multipliers must be positive.")
            for parameter in sampler.parameters():
                parameter.optimize_attr["learning_rate"] = multiplier

        logger.info(
            f"[SGEQuiDiff] Initialized: dataset={dataset_name}, "
            f"snr={diffusion_snr}, temperature={temperature}"
        )

    @property
    def execution_backend(self) -> str:
        """Active numerical execution backend (owned by the diffusion model)."""
        return self.atom_coord_diffusion_model.execution_backend

    def set_execution_backend(self, backend: str) -> None:
        self.atom_coord_diffusion_model.set_execution_backend(backend)

    def set_runtime_options(self, runtime_options: dict) -> None:
        self.atom_coord_diffusion_model.set_runtime_options(runtime_options)

    def validate_execution_backend(
        self, use_amp: bool = False, world_size: int = 1
    ) -> None:
        self.atom_coord_diffusion_model.validate_execution_backend(
            use_amp=use_amp, world_size=world_size
        )

    def forward(self, batch) -> Dict:
        """Training entry: MLE on discrete variables + score matching on coords.

        Returns:
            loss_dict : total loss (``loss_dict["loss"]`` is backpropagated).
            pred_dict : detached per-component log-probs / losses, tracked in
                training logs only. SGEQuiDiff is an unconditional generative
                model: its training forward has no per-attribute predictions
                aligned with batch labels, so **non-streaming
                ``compute_metric_func_dict`` metrics are not applicable** here.
                Generations are scored with the streaming
                ``SGEQuiDiffMetric`` (``stage == "sample"``).
            label_dict : supervision fields passed through with keys identical
                to the corresponding ``batch`` labels, so per-key
                pred/label lookups stay aligned when a streaming metric needs
                them.
        """
        loss, artifacts = self.compute_loss(batch)
        label_keys = (
            "space_group_indices",
            "lattice_lengths",
            "lattice_angles",
            "n_atoms_per_asu",
            "element_indices",
            "wyckoff_indices",
            "wyckoff_shape_indices",
            "frac_coords",
        )
        label_dict = {k: batch[k] for k in label_keys if k in batch}
        return {
            "loss_dict": {"loss": loss},
            "pred_dict": artifacts,
            "label_dict": label_dict,
        }

    def compute_loss(self, batch):
        """MLE on discrete variables, score matching on atom coordinates.

        Returns:
            loss: scalar tensor.
            artifacts: dict of detached log-probs / losses for logging.
        """
        space_group_indices = _as_tensor(batch["space_group_indices"])
        lattice_lengths = _as_tensor(batch["lattice_lengths"], paddle.float32)
        lattice_angles = _as_tensor(batch["lattice_angles"], paddle.float32)
        if "lattice_matrices" in batch:
            lattice_matrices = batch["lattice_matrices"]
        else:
            lattice_matrices = lattice_params_to_matrix_paddle(
                lattice_lengths, lattice_angles
            )
        lattice_matrices = _as_tensor(lattice_matrices, paddle.float32)
        element_indices = _as_tensor(batch["element_indices"], paddle.int64)
        wyckoff_indices = _as_tensor(batch["wyckoff_indices"], paddle.int64)
        n_asu_atoms_per_xtal = _as_tensor(batch["n_atoms_per_asu"], paddle.int64)
        wyckoff_shape_indices = _as_tensor(batch["wyckoff_shape_indices"], paddle.int64)
        asu_frac_coords = _as_tensor(batch["frac_coords"], paddle.float32)

        if self.training and (
            self.lattice_length_noise > 0 or self.lattice_angle_noise > 0
        ):
            (
                noisy_lattice_lengths,
                noisy_lattice_angles,
            ) = self.get_noisy_lattice_lengths_and_angles(
                lattice_lengths, lattice_angles, space_group_indices
            )
        else:
            noisy_lattice_lengths = lattice_lengths
            noisy_lattice_angles = lattice_angles

        space_group_log_prob = self.space_group_sampler.log_prob(space_group_indices)
        lattice_log_prob = self.lattice_sampler.log_prob(
            lattice_lengths,
            lattice_angles,
            space_group_indices,
            noisy_lattice_lengths,
            noisy_lattice_angles,
        )
        (
            elements_log_prob,
            wyckoffs_log_prob,
            termination_log_prob,
        ) = self.wyckoff_and_element_sampler.log_prob(
            element_indices,
            wyckoff_indices,
            n_asu_atoms_per_xtal,
            noisy_lattice_lengths,
            noisy_lattice_angles,
            space_group_indices,
            # Static pad width keeps shapes CINN-friendly.
            max_atoms=max_atoms_per_dataset[self.dataset_name],
        )
        score_matching_loss = self.atom_coord_diffusion_model.compute_loss(
            asu_frac_coords,
            element_indices,
            wyckoff_indices,
            space_group_indices,
            n_asu_atoms_per_xtal,
            wyckoff_shape_indices,
            lattice_matrices,
            lattice_lengths,
            lattice_angles,
        )
        (
            space_group_log_prob,
            lattice_log_prob,
            termination_log_prob,
            elements_log_prob,
            wyckoffs_log_prob,
            score_matching_loss,
            detached_space_group_log_prob,
            detached_lattice_log_prob,
            detached_termination_log_prob,
            detached_elements_log_prob,
            detached_wyckoffs_log_prob,
            detached_score_matching_loss,
        ) = self._get_rebalanced_grads(
            space_group_log_prob,
            lattice_log_prob,
            termination_log_prob,
            elements_log_prob,
            wyckoffs_log_prob,
            score_matching_loss,
        )
        artifacts = {
            "space_group_log_prob": detached_space_group_log_prob,
            "lattice_log_prob": detached_lattice_log_prob,
            "elements_log_prob": detached_elements_log_prob,
            "wyckoffs_log_prob": detached_wyckoffs_log_prob,
            "termination_log_prob": detached_termination_log_prob,
            "score_matching_loss": detached_score_matching_loss,
        }
        lattice_log_prob_nonzero = lattice_log_prob[lattice_log_prob != 0.0]
        if lattice_log_prob_nonzero.numel() > 0:
            lattice_nll = lattice_log_prob_nonzero.mean()
        else:
            lattice_nll = paddle.zeros([1], dtype=lattice_log_prob.dtype)
        nll = -1.0 * (
            (space_group_log_prob + termination_log_prob).mean()
            + lattice_nll
            + (elements_log_prob + wyckoffs_log_prob).mean()
        )
        return nll + score_matching_loss, artifacts

    def get_noisy_lattice_lengths_and_angles(
        self,
        lattice_lengths: paddle.Tensor,
        lattice_angles: paddle.Tensor,
        space_group_indices: paddle.Tensor,
    ):
        """Rejection-sample noisy lattice parameters under Bravais constraints.

        NOTE: only invoked when ``lattice_length_noise`` or ``lattice_angle_noise``
        is > 0 (see ``compute_loss``); the default config (0.0) skips this path.
        The while-loop checks ``.item()`` every iteration, forcing a GPU sync;
        this matches the upstream algorithm and is kept for parity.
        """
        batch_size = space_group_indices.shape[0]
        lattice_sampler = self.lattice_sampler
        MIN_LATTICE_LENGTH = lattice_sampler.MIN_LATTICE_LENGTH
        MAX_LATTICE_LENGTH = lattice_sampler.MAX_LATTICE_LENGTH
        MIN_LATTICE_ANGLE = lattice_sampler.MIN_LATTICE_ANGLE
        MAX_LATTICE_ANGLE = lattice_sampler.MAX_LATTICE_ANGLE
        length_transforms = lattice_sampler.bravais_length_transforms[
            space_group_indices
        ]
        angle_transforms = lattice_sampler.bravais_angle_transforms[space_group_indices]
        angle_offsets = lattice_sampler.bravais_angle_offsets[space_group_indices]
        lattice_angle_bounds = paddle.to_tensor(
            [MIN_LATTICE_ANGLE, MAX_LATTICE_ANGLE], dtype=paddle.float32
        )
        lattice_noise_magnitude = paddle.to_tensor(
            [
                self.lattice_length_noise,
                self.lattice_length_noise,
                self.lattice_length_noise,
                self.lattice_angle_noise,
                self.lattice_angle_noise,
                self.lattice_angle_noise,
            ],
            dtype=paddle.float32,
        )
        num_samples_per_iter = 2
        max_iters = 100
        iteration = 0
        lattice_param_is_done = paddle.zeros(
            [batch_size, NUM_LATTICE_PARAMS], dtype="bool"
        )
        accepted_noisy_lattice_parameters = paddle.full(
            [batch_size, NUM_LATTICE_PARAMS], fill_value=-1.0, dtype=paddle.float32
        )
        # NOTE: .item() per iteration forces a GPU->CPU sync; only active when
        # lattice noise > 0, inherited from the upstream rejection-sampling loop.
        while not paddle.all(lattice_param_is_done).item():
            iteration += 1
            if iteration > max_iters:
                raise RuntimeError(
                    "Failed to sample valid noisy lattice parameters after "
                    f"{max_iters} iterations: "
                    f"{int((~lattice_param_is_done).all(axis=-1).sum())} "
                    f"of {batch_size} lattices still incomplete"
                )
            lattice_is_done = paddle.all(lattice_param_is_done, axis=-1)
            unfinished_lengths = lattice_lengths[~lattice_is_done]
            unfinished_angles = lattice_angles[~lattice_is_done]
            num_lattices_left_to_noise = unfinished_angles.shape[0]
            noise = lattice_noise_magnitude[None, None, :] * (
                2.0
                * paddle.rand(
                    [
                        num_lattices_left_to_noise,
                        num_samples_per_iter,
                        NUM_LATTICE_PARAMS,
                    ]
                )
                - 1.0
            )
            noisy_lengths = noise[:, :, :3] + unfinished_lengths[:, None, :]
            noisy_angles = noise[:, :, 3:] + unfinished_angles[:, None, :]
            noisy_lengths = paddle.bmm(
                noisy_lengths, length_transforms[~lattice_is_done]
            )
            noisy_angles = (
                paddle.bmm(noisy_angles, angle_transforms[~lattice_is_done])
                + angle_offsets[~lattice_is_done][:, None, :]
            )
            lengths_are_valid = (noisy_lengths >= MIN_LATTICE_LENGTH) & (
                noisy_lengths <= MAX_LATTICE_LENGTH
            )
            (
                min_gamma_angle,
                max_gamma_angle,
            ) = lattice_sampler.get_valid_gamma_angle_interval(
                alpha_and_beta_angles=noisy_angles[:, :, :2].reshape([-1, 2])
            )
            min_gamma_angle = min_gamma_angle + 0.01
            max_gamma_angle = max_gamma_angle - 0.01
            min_allowed_angles = paddle.concat(
                [
                    lattice_angle_bounds[0]
                    .reshape([1, 1, 1])
                    .expand([num_lattices_left_to_noise, num_samples_per_iter, 2]),
                    min_gamma_angle.reshape(
                        [num_lattices_left_to_noise, num_samples_per_iter, 1]
                    ),
                ],
                axis=-1,
            )
            max_allowed_angles = paddle.concat(
                [
                    lattice_angle_bounds[1]
                    .reshape([1, 1, 1])
                    .expand([num_lattices_left_to_noise, num_samples_per_iter, 2]),
                    max_gamma_angle.reshape(
                        [num_lattices_left_to_noise, num_samples_per_iter, 1]
                    ),
                ],
                axis=-1,
            )
            angles_are_valid = (noisy_angles >= min_allowed_angles) & (
                noisy_angles <= max_allowed_angles
            )
            for i in range(3):
                sample_is_valid = lengths_are_valid[:, :, i]
                any_sample_is_valid = paddle.any(sample_is_valid, axis=-1)
                if paddle.any(any_sample_is_valid).item():
                    update_lattice_params_mask = (
                        paddle.nn.functional.one_hot(
                            paddle.to_tensor([i], dtype="int64"),
                            num_classes=NUM_LATTICE_PARAMS,
                        )
                        .cast("bool")
                        .tile([batch_size, 1])
                    )
                    update_lattice_params_mask[lattice_is_done] = False
                    accept_sample_mask = (
                        any_sample_is_valid
                        & ~lattice_param_is_done[~lattice_is_done][:, i]
                    )
                    update_lattice_params_mask[
                        ~lattice_is_done[:, None] & update_lattice_params_mask
                    ] = accept_sample_mask
                    first_accepted_length_idx = paddle.argmax(
                        sample_is_valid.cast("float32"), axis=-1
                    )[accept_sample_mask][:, None]
                    accepted_noisy_lattice_parameters[update_lattice_params_mask] = (
                        noisy_lengths[:, :, i][accept_sample_mask]
                        .take_along_axis(first_accepted_length_idx, axis=1)
                        .reshape([-1])
                    )
                    lattice_param_is_done[update_lattice_params_mask] = True
            sample_is_valid = paddle.all(angles_are_valid, axis=-1)
            any_sample_is_valid = paddle.any(sample_is_valid, axis=-1)
            update_lattice_angles_mask = paddle.to_tensor(
                [[False, False, False, True, True, True]], dtype="bool"
            ).tile([batch_size, 1])
            update_lattice_angles_mask[lattice_is_done] = False
            accept_sample_mask = any_sample_is_valid & paddle.all(
                ~lattice_param_is_done[~lattice_is_done][:, 3:], axis=-1
            )
            update_lattice_angles_mask[
                ~lattice_is_done[:, None] & update_lattice_angles_mask
            ] = (accept_sample_mask[:, None].tile([1, 3]).reshape([-1]))
            first_accepted_angles_idx = paddle.argmax(
                sample_is_valid.cast("float32"), axis=-1
            )[accept_sample_mask]
            _lattice_idx = paddle.arange(first_accepted_angles_idx.shape[0])
            accepted_noisy_lattice_parameters[
                update_lattice_angles_mask
            ] = noisy_angles[accept_sample_mask][
                _lattice_idx, first_accepted_angles_idx
            ].reshape(
                [-1]
            )
            lattice_param_is_done[update_lattice_angles_mask] = True
        noisy_lattice_lengths = accepted_noisy_lattice_parameters[:, :3]
        noisy_lattice_angles = accepted_noisy_lattice_parameters[:, 3:]
        return noisy_lattice_lengths, noisy_lattice_angles

    def _get_rebalanced_grads(
        self,
        space_group_log_prob: paddle.Tensor,
        lattice_log_prob: paddle.Tensor,
        termination_log_prob: paddle.Tensor,
        elements_log_prob: paddle.Tensor,
        wyckoffs_log_prob: paddle.Tensor,
        score_matching_loss: paddle.Tensor,
    ):
        """Apply straight-through estimators to rebalance gradients."""
        detached_space_group_log_prob = space_group_log_prob.detach()
        detached_lattice_log_prob = lattice_log_prob.detach()
        detached_termination_log_prob = termination_log_prob.detach()
        detached_elements_log_prob = elements_log_prob.detach()
        detached_wyckoffs_log_prob = wyckoffs_log_prob.detach()
        detached_score_matching_loss = score_matching_loss.detach()
        space_group_log_prob = (
            self.space_group_grad_weight * space_group_log_prob
            - self.space_group_grad_weight * detached_space_group_log_prob
            + detached_space_group_log_prob
        )
        lattice_log_prob = (
            self.lattice_grad_weight * lattice_log_prob
            - self.lattice_grad_weight * detached_lattice_log_prob
            + detached_lattice_log_prob
        )
        termination_log_prob = (
            self.wyckoff_element_grad_weight * termination_log_prob
            - self.wyckoff_element_grad_weight * detached_termination_log_prob
            + detached_termination_log_prob
        )
        elements_log_prob = (
            self.wyckoff_element_grad_weight * elements_log_prob
            - self.wyckoff_element_grad_weight * detached_elements_log_prob
            + detached_elements_log_prob
        )
        wyckoffs_log_prob = (
            self.wyckoff_element_grad_weight * wyckoffs_log_prob
            - self.wyckoff_element_grad_weight * detached_wyckoffs_log_prob
            + detached_wyckoffs_log_prob
        )
        score_matching_loss = (
            self.frac_coord_grad_weight * score_matching_loss
            - self.frac_coord_grad_weight * detached_score_matching_loss
            + detached_score_matching_loss
        )
        return (
            space_group_log_prob,
            lattice_log_prob,
            termination_log_prob,
            elements_log_prob,
            wyckoffs_log_prob,
            score_matching_loss,
            detached_space_group_log_prob,
            detached_lattice_log_prob,
            detached_termination_log_prob,
            detached_elements_log_prob,
            detached_wyckoffs_log_prob,
            detached_score_matching_loss,
        )

    @paddle.no_grad()
    def sample_crystal(
        self,
        batch_size: int,
        diffusion_snr: float = 0.4,
        temperature: float = 1.0,
    ) -> List[Dict]:
        """Full crystal sampling pipeline."""
        space_group_indices = self.space_group_sampler.sample(batch_size, temperature)
        lattice_lengths, lattice_angles = self.lattice_sampler(space_group_indices)
        lattice_matrices = lattice_params_to_matrix_paddle(
            lattice_lengths, lattice_angles
        )

        (
            element_indices,
            wyckoff_indices,
            n_asu_atoms_per_xtal,
        ) = self.wyckoff_and_element_sampler.sample(
            lattice_lengths, lattice_angles, space_group_indices, temperature
        )

        frac_coords = self.atom_coord_diffusion_model.sample(
            wyckoff_indices,
            element_indices,
            space_group_indices,
            n_asu_atoms_per_xtal,
            lattice_matrices,
            lattice_lengths,
            lattice_angles,
            snr=diffusion_snr,
            max_step_size=1e6,
        )[0]

        asu_crystals = []
        atom_offsets = paddle.concat(
            [
                paddle.cumsum(n_asu_atoms_per_xtal, axis=0) - n_asu_atoms_per_xtal,
                n_asu_atoms_per_xtal.sum().unsqueeze(0),
            ],
            axis=0,
        )
        for i in range(n_asu_atoms_per_xtal.shape[0]):
            asu_crystals.append(
                {
                    "space_group_number": 1 + space_group_indices[i],
                    "conventional_lattice_lengths": lattice_lengths[i],
                    "conventional_lattice_angles": lattice_angles[i],
                    "element_indices": element_indices[
                        atom_offsets[i] : atom_offsets[i + 1]
                    ],
                    "wyckoff_indices": wyckoff_indices[
                        atom_offsets[i] : atom_offsets[i + 1]
                    ],
                    "conventional_frac_coords": frac_coords[
                        atom_offsets[i] : atom_offsets[i + 1]
                    ],
                }
            )
        return asu_crystals

    @paddle.no_grad()
    def sample(self, batch_data: Dict) -> Dict:
        """Sample crystals compatible with ``structure_generation/sample.py``.

        Args:
            batch_data: Dict with key ``"structure_array"`` containing at
                least ``"num_atoms"`` (a 1-D int tensor of batch sizes).
                SGEQuiDiff is an **unconditional** generator, so the actual
                atom counts in *batch_data* are ignored -- they only
                determine the *batch size*.

        Returns:
            Dict with key ``"result"`` mapping to a list of dicts, each
            containing the **full conventional cell** (ASU expanded via the
            space-group symmetry operations):
            - ``"num_atoms"``: int
            - ``"atom_types"``: list[int]  (1-indexed atomic numbers)
            - ``"frac_coords"``: list[list[float]]
            - ``"lengths"``: list[float]  (a, b, c in Angstrom)
            - ``"angles"``: list[float]   (alpha, beta, gamma in degrees)
        """
        structure_array = batch_data.get("structure_array")
        if structure_array is not None:
            if "atom_types" in structure_array:
                raise NotImplementedError(
                    "SGEQuiDiff does not support atom-type or chemical formula "
                    "conditioning. Use sample_by_dataloader instead."
                )
            num_atoms_tensor = structure_array["num_atoms"]
        else:
            # sample_by_dataloader passes an ASU dataset batch without
            # structure_array; SGEQuiDiff is unconditional so the number of
            # generated crystals is simply the batch size.
            num_atoms_tensor = paddle.ones(
                [batch_data["n_atoms_per_asu"].shape[0]], dtype=paddle.int64
            )
        batch_size = num_atoms_tensor.shape[0]

        crystals = self.sample_crystal(
            batch_size=batch_size,
            diffusion_snr=self.diffusion_snr,
            temperature=self.temperature,
        )

        # Expand each ASU crystal into a full conventional cell, so the result
        # is a physically complete structure (not just the asymmetric unit).
        result = []
        for crystal in crystals:
            valid_mask = []
            for i, idx in enumerate(crystal["element_indices"].tolist()):
                if 0 <= idx < ELEMENT_ENCODING_SIZE:
                    elem = chemical_symbols[idx + 1]
                    if elem != "X":
                        valid_mask.append(i)

            if len(valid_mask) == 0:
                logger.warning("Generated crystal has no valid elements, skipping.")
                continue

            mask = paddle.to_tensor(valid_mask, dtype=paddle.int64)
            filtered_crystal = {
                "space_group_number": crystal["space_group_number"],
                "conventional_lattice_lengths": crystal["conventional_lattice_lengths"],
                "conventional_lattice_angles": crystal["conventional_lattice_angles"],
                "element_indices": crystal["element_indices"][mask],
                "wyckoff_indices": crystal["wyckoff_indices"][mask],
                "conventional_frac_coords": crystal["conventional_frac_coords"][mask],
            }
            result.append(expand_asu(filtered_crystal, self.wyckoff_geometry))

        return {"result": result}

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

"""Precompute the space-group-aware sigma-norm table for SGEquiDiff.

The score-matching objective normalizes ground-truth scores by a per
(space group, Wyckoff site, timestep) expected-norm table.  The table is
estimated by Monte Carlo over the ASU-wrapped normal and cached on disk,
keyed by (sigma range, num_timesteps, MC samples, lattice translations).

This module lives in the model package because the estimate is bound to
the ASU / Wyckoff / space-group semantics of SGEquiDiff; the generic
VE-SDE scheduler only consumes the finished table.
"""

import pickle

import paddle

from ppmat.models.sgequidiff.asu_crystal import sample_point_in_asu_wyckoff_site
from ppmat.models.sgequidiff.asu_math import d_log_p_asu_wrapped_normal
from ppmat.models.sgequidiff.asu_math import get_space_group_ops_and_conventional_atoms
from ppmat.models.sgequidiff.sgequidiff_meta import MAX_WYCKOFF_POSITIONS
from ppmat.models.sgequidiff.sgequidiff_meta import NUM_CRYSTALLOGRAPHIC_SPACE_GROUPS
from ppmat.models.sgequidiff.wyckoff_geometry import WyckoffGeometry
from ppmat.models.sgequidiff.wyckoff_shape_decomp import ensure_wyckoff_shape_decomp
from ppmat.models.sgequidiff.wyckoff_shape_decomp import get_data_directory
from ppmat.models.sgequidiff.wyckoff_shape_decomp import get_shape_decomp_dict_path
from ppmat.schedulers.scheduling_asu_ve_sde import build_ve_sigma_grid
from ppmat.utils import logger


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
    return _sigma_norm_asu_wrapped(
        sigmas,
        wyckoff_geometry,
        num_lattice_translations,
        num_monte_carlo_samples,
    )


@paddle.no_grad()
def _sigma_norm_asu_wrapped(
    sigmas: paddle.Tensor,
    wyckoff_geometry: "WyckoffGeometry",
    num_lattice_translations: int,
    num_monte_carlo_samples: int = 2_500,
) -> paddle.Tensor:
    """Monte Carlo estimate of expected score L2 norm for ASU-wrapped normal."""
    num_timesteps = sigmas.shape[0]

    cache_name = (
        f"expected_score_norms_minSigma{float(sigmas[0]):0.3f}"
        f"_maxSigma{float(sigmas[-1]):0.3f}"
        f"_T{num_timesteps}_{num_monte_carlo_samples}MCsamples"
        f"_{num_lattice_translations}LatticeTranslations.pdparams"
    )
    cache_path = get_data_directory() / cache_name
    logger.info(f"Checking cache: {cache_path}")
    logger.info(f"Cache exists: {cache_path.exists()}")
    if cache_path.exists():
        logger.info(f"Loading sigma norms from: {cache_path}")
        sigma_norms = paddle.load(str(cache_path))
        logger.info(f"Successfully loaded sigma_norms with shape: {sigma_norms.shape}")
        return sigma_norms
    logger.info("Cache not found, computing sigma_norms...")

    ensure_wyckoff_shape_decomp()
    with open(str(get_shape_decomp_dict_path()), "rb") as f:
        wyckoff_shape_decomp_dict = pickle.load(f)

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
            return_sampled_wyckoff_shape_indices=True,
        )

        space_group_idx = paddle.to_tensor([sg_num - 1], dtype=paddle.int64)

        for i, letter in enumerate(wyckoff_letters):
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

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    paddle.save(sigma_norms, str(cache_path))
    logger.info(f"Saved sigma norms to: {cache_path}")
    return sigma_norms

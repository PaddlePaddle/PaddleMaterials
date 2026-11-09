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

"""Telescoping discrete lattice parameter sampler."""
import math
from typing import Tuple

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.distribution import Categorical

from ppmat.models.sgequidiff.sgequidiff_meta import NUM_LATTICE_PARAMS
from ppmat.models.sgequidiff.shared import FourierLinear
from ppmat.models.sgequidiff.shared import SpaceGroupEncoder
from ppmat.models.sgequidiff.vocabs import EmbeddingTools


def lattice_transform_and_log_prob_mask(
    spacegroup: int,
) -> Tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor]:
    if 1 <= spacegroup <= 2:
        length_matrix = paddle.eye(3)
        angle_matrix = paddle.eye(3)
        angle_vector = paddle.zeros([3])
        log_prob_mask = paddle.ones([6])
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
    elif 195 <= spacegroup <= 230:
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
            norm_gamma_separately=False,
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
        norm_gamma_separately: bool = True,
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

        if norm_gamma_separately:
            min_gamma_angle, max_gamma_angle = self.get_valid_gamma_angle_interval(
                alpha_and_beta_angles=lattice_angles[:, :2]
            )
        else:
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
            for j in range(self.n_telescopes):
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
            norm_gamma_separately=False,
        )
        normed_noisy_lattice_parameters = (
            self._get_normed_lattice_parameters(
                noisy_lattice_lengths,
                noisy_lattice_angles,
                self.min_bin_edge,
                self.max_bin_edge,
                norm_gamma_separately=False,
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
            log_pfs: (batch_size,)
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
        log_pfs = []
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

            sample, log_prob = self._sample_and_log_prob(
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
            log_pfs.append(log_prob)

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

        log_pfs = paddle.stack(log_pfs, axis=1)

        lattice_parameters = (
            (normed_lattice_parameters - self.min_bin_edge)
            / (self.max_bin_edge - self.min_bin_edge)
        ) * lattice_params_transform + lattice_params_offset

        lengths = lattice_parameters[:, :num_lengths]
        angles = lattice_parameters[:, num_lengths:]

        log_pf_masks = self.bravais_log_prob_masks[space_group_indices]
        log_pfs = (log_pfs * log_pf_masks).sum(axis=1)

        return lengths, angles, log_pfs

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
                F.log_softmax(bin_logits, axis=-1), bin_idxs.unsqueeze(-1), axis=-1
            ).squeeze(-1)

            chosen_bins = bins[_batch_idxs, bin_idxs]
            min_bin_edge = chosen_bins[:, 0]
            max_bin_edge = chosen_bins[:, 1]

        samples = paddle.mean(chosen_bins, axis=-1)
        return samples, log_probs

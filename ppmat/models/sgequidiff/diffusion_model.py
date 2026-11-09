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

"""EquivariantDiffusionModel and NoiseScheduler."""
from contextlib import nullcontext
from typing import Dict
from typing import Optional

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from tqdm import tqdm

from ppmat.models.common.runtime import RuntimeMixin
from ppmat.models.common.runtime import runtime_boundary
from ppmat.models.common.time_embedding import SinusoidalTimeEmbeddings
from ppmat.models.sgequidiff.asu_crystal import sample_point_in_asu_wyckoff_site
from ppmat.models.sgequidiff.asu_math import d_log_p_asu_wrapped_normal
from ppmat.models.sgequidiff.asu_math import get_space_group_ops_and_conventional_atoms
from ppmat.models.sgequidiff.asu_math import get_wyckoff_projected_gaussian_noise
from ppmat.models.sgequidiff.asu_math import wrap_frac_coords_into_asu
from ppmat.models.sgequidiff.drift_modules import GNN
from ppmat.models.sgequidiff.drift_modules import CSPNet
from ppmat.models.sgequidiff.drift_modules import TorusMLP
from ppmat.models.sgequidiff.sigma_norm import compute_sigma_norms
from ppmat.models.sgequidiff.vocabs import EmbeddingTools
from ppmat.models.sgequidiff.wyckoff_geometry import WyckoffGeometry
from ppmat.models.sgequidiff.wyckoff_shape_decomp import ensure_wyckoff_shape_decomp
from ppmat.models.sgequidiff.wyckoff_shape_decomp import get_shape_decomp_dict_path
from ppmat.schedulers import build_scheduler
from ppmat.utils.crystal import lattice_params_to_matrix_paddle
from ppmat.utils.scatter import scatter as paddle_scatter


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
        self.model_type = model_type
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
            num_timesteps=num_timesteps,
            wyckoff_geometry=wyckoff_geometry,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
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
        """Lazy-load wyckoff_shape_decomposition.pkl."""
        if self._wyckoff_shape_decomposition_dict is None:
            import pickle

            ensure_wyckoff_shape_decomp()
            with open(str(get_shape_decomp_dict_path()), "rb") as f:
                self._wyckoff_shape_decomposition_dict = pickle.load(f)
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
                wyckoff_indices,
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
            return_sampled_wyckoff_shape_indices=True,
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
            wyckoff_indices,
            space_group_indices,
            n_atoms_per_xtal,
            self.padded_hull_equations[space_group_indices_per_atom, wyckoff_indices],
            self.padded_hull_equations_mask[
                space_group_indices_per_atom, wyckoff_indices
            ],
            self.wyckoff_geometry,
        )
        return x_final.unsqueeze(0)

    def forward(self, batch_data: Dict) -> Dict:
        """Training entry: unpack batch, compute loss, return
        BaseTrainer-compatible dict."""
        if "lattice_matrices" in batch_data:
            lattice_matrices = batch_data["lattice_matrices"]
        else:
            lattice_matrices = lattice_params_to_matrix_paddle(
                batch_data["lattice_lengths"], batch_data["lattice_angles"]
            )
        loss = self.compute_loss(
            asu_frac_coords=batch_data["frac_coords"],
            element_indices=batch_data["element_indices"],
            wyckoff_indices=batch_data["wyckoff_indices"],
            space_group_indices=batch_data["space_group_indices"],
            n_atoms_per_xtal=batch_data["n_atoms_per_asu"],
            wyckoff_shape_indices=batch_data["wyckoff_shape_indices"],
            lattice_matrices=lattice_matrices,
            lattice_lengths=batch_data["lattice_lengths"],
            lattice_angles=batch_data["lattice_angles"],
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
        differentiate_graph_construction: bool = False,
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

        n_conv_atoms_per_xtal = paddle_scatter(
            src=paddle.ones(
                [conventional_wyckoff_indices.shape[0]], dtype=paddle.int64
            ),
            index=map_asu_atom_to_xtal[map_unique_conventional_to_asu],
            dim=0,
            dim_size=space_group_indices.shape[0],
            reduce="sum",
        )

        drift = self.non_equivariant_drift_model
        cm = paddle.no_grad() if not differentiate_graph_construction else nullcontext()
        with cm:
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

        vector_field = paddle_scatter(
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

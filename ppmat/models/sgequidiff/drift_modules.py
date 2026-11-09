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

"""Non-equivariant drift backbones: GNN (PBC message passing), CSPNet (from
DiffCSP), and TorusMLP."""
import math
from contextlib import nullcontext
from typing import Optional
from typing import Tuple

import paddle
import paddle.nn as nn

from ppmat.models.common.activation import ScaledSiLU as Swish
from ppmat.models.common.radial_basis import GaussianSmearing
from ppmat.models.common.time_embedding import SinusoidalTimeEmbeddings
from ppmat.models.sgequidiff.pbc_graph import build_pbc_graph
from ppmat.models.sgequidiff.sgequidiff_csp_layer import CSPLayer
from ppmat.models.sgequidiff.sgequidiff_csp_layer import SinusoidsEmbedding
from ppmat.models.sgequidiff.sgequidiff_meta import ELEMENT_ENCODING_SIZE
from ppmat.models.sgequidiff.sgequidiff_meta import lattice_parameter_ranges
from ppmat.models.sgequidiff.shared import GraphNorm
from ppmat.models.sgequidiff.shared import VariancePreservingAggregation
from ppmat.models.sgequidiff.vocabs import EmbeddingTools
from ppmat.utils.crystal import frac_to_cart_coords
from ppmat.utils.crystal import get_pbc_distances
from ppmat.utils.scatter import scatter as paddle_scatter

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

    def forward(
        self,
        frac_coords: paddle.Tensor,
        time_embeddings: paddle.Tensor,
        *args,
        **kwargs,
    ) -> paddle.Tensor:
        return self.forward_with_graph(
            frac_coords=frac_coords,
            time_embeddings=time_embeddings,
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
        num_cartesian_distance_gaussians: int,
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
            fourier_frac_edge_dim + gaussian_cart_edge_dim + 6,
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
        self.hidden_channels = hidden_channels
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
            h_agg = paddle_scatter(
                messages, dst_ids, dim=0, dim_size=n_nodes, reduce="sum"
            )

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
    ):
        super().__init__()
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
            num_cartesian_distance_gaussians,
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

    def forward(
        self,
        frac_coords: paddle.Tensor,
        element_indices: paddle.Tensor,
        n_atoms_per_xtal: paddle.Tensor,
        lattice_matrices: paddle.Tensor,
        lattice_lengths: paddle.Tensor,
        lattice_angles: paddle.Tensor,
        time_embeddings: paddle.Tensor,
        differentiate_graph_construction: bool = False,
    ) -> paddle.Tensor:
        """GNN forward pass, returns (n_atoms, 3)."""
        cm = paddle.no_grad() if not differentiate_graph_construction else nullcontext()
        with cm:
            graph_inputs = self.construct_graph_inputs(
                frac_coords=frac_coords,
                n_atoms_per_xtal=n_atoms_per_xtal,
                lattice_matrices=lattice_matrices,
                lattice_lengths=lattice_lengths,
                lattice_angles=lattice_angles,
            )
        return self.forward_with_graph(
            frac_coords=frac_coords,
            element_indices=element_indices,
            time_embeddings=time_embeddings,
            n_atoms_per_xtal=n_atoms_per_xtal,
            graph_inputs=graph_inputs,
        )

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

    @staticmethod
    def construct_graphs(
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
        (
            source_ids,
            destination_ids,
            source_node_image_offsets,
            num_edges_per_crystal,
        ) = build_pbc_graph(
            cart_coords=cart_coords,
            lattice=lattice_matrices,
            num_atoms=n_atoms_per_xtal,
        )
        edge_index = paddle.stack([source_ids, destination_ids], axis=0)
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
        if dis_emb == "sin":
            self.dis_emb = SinusoidsEmbedding(n_frequencies=num_freqs, n_space=3)
        elif dis_emb == "none":
            self.dis_emb = None

        for i in range(num_layers):
            self.add_sublayer(
                f"csp_layer_{i}",
                CSPLayer(
                    hidden_dim=hidden_dim,
                    prop_dim=0,
                    act_fn=self.act_fn,
                    dis_emb=self.dis_emb,
                    ln=ln,
                    use_lattice_ip=False,
                    lattice_dim=6,
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

    def forward(
        self,
        frac_coords: paddle.Tensor,
        element_indices: paddle.Tensor,
        n_atoms_per_xtal: paddle.Tensor,
        lattice_matrices: paddle.Tensor,
        lattice_lengths: paddle.Tensor,
        lattice_angles: paddle.Tensor,
        time_embeddings: paddle.Tensor,
        *args,
        **kwargs,
    ) -> paddle.Tensor:
        """CSPNet forward pass."""
        graph_inputs = self.construct_graph_inputs(
            frac_coords=frac_coords,
            n_atoms_per_xtal=n_atoms_per_xtal,
            lattice_lengths=lattice_lengths,
            lattice_angles=lattice_angles,
        )
        return self.forward_with_graph(
            frac_coords=frac_coords,
            element_indices=element_indices,
            time_embeddings=time_embeddings,
            n_atoms_per_xtal=n_atoms_per_xtal,
            graph_inputs=graph_inputs,
        )

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

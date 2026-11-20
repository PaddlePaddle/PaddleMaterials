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

"""Property-guided molecule generation with geometry-complete flow matching.

Reference: https://www.nature.com/articles/s43588-025-00946-y

The model owns the numerical runtime (see ``PropMolFlowVectorField.forward``) and
follows the ``forward``/``sample`` contracts of PaddleMaterials: the training
forward returns ``{"loss_dict": ...}`` and sampling returns ``{"pred": [...]}``
consumed by ``MolecularSampler``.
"""

from __future__ import annotations

import copy
import os
import os.path as osp

import numpy as np
import paddle
import paddle.nn as nn
from scipy.optimize import linear_sum_assignment

from ppmat.models.common.runtime import RuntimeMixin
from ppmat.models.common.runtime import runtime_boundary
from ppmat.models.propmolflow.gvp import GVP
from ppmat.models.propmolflow.gvp import GVPConv
from ppmat.models.propmolflow.gvp import StableSiLU
from ppmat.models.propmolflow.gvp import norm_no_nan
from ppmat.schedulers import build_scheduler
from ppmat.utils import logger
from ppmat.utils.scatter import scatter_mean
from ppmat.utils.scatter import scatter_sum

FEATURE_NAMES = ["positions", "atom_types", "atom_charges", "bond_types"]
MASK_ATOM_SYMBOL = "Se"


def build_batched_complete_graph(num_atoms: np.ndarray) -> dict:
    """Build complete molecular graphs with upper edges before reverse edges."""
    num_atoms = np.asarray(num_atoms, dtype="int64")
    edge_idxs_list = []
    for n in num_atoms:
        upper = np.stack(np.triu_indices(int(n), k=1))
        edge_idxs_list.append(np.concatenate((upper, upper[::-1]), axis=1))
    num_edges = np.array([e.shape[1] for e in edge_idxs_list], dtype="int64")

    node_offsets = np.concatenate([[0], np.cumsum(num_atoms)[:-1]])
    src = np.concatenate([e[0] for e in edge_idxs_list]) + np.repeat(
        node_offsets, num_edges
    )
    dst = np.concatenate([e[1] for e in edge_idxs_list]) + np.repeat(
        node_offsets, num_edges
    )

    upper_edge_mask = np.zeros(int(num_edges.sum()), dtype=bool)
    offset = 0
    for n_edges in num_edges:
        upper_edge_mask[offset : offset + n_edges // 2] = True
        offset += n_edges

    return {
        "src": src,
        "dst": dst,
        "upper_edge_mask": upper_edge_mask,
        "node_batch_idx": np.repeat(np.arange(len(num_atoms)), num_atoms),
        "edge_batch_idx": np.repeat(np.arange(len(num_atoms)), num_edges),
    }


def _rbf(D, D_max, D_count, D_min=0.0):
    """Radial basis function embedding of pairwise distances."""
    D_mu = paddle.linspace(D_min, D_max, D_count)
    D_mu = D_mu.reshape([1, -1])
    D_sigma = (D_max - D_min) / D_count
    D_expand = paddle.unsqueeze(D, -1)
    return paddle.exp(-(((D_expand - D_mu) / D_sigma) ** 2))


class GaussianExpansion(nn.Layer):
    """Expansion layer using a set of Gaussian functions.

    Reference: https://github.com/atomistic-machine-learning/cG-SchNet
    """

    def __init__(self, start, stop, n_gaussians):
        super().__init__()
        offset = paddle.linspace(start, stop, n_gaussians)
        widths = (offset[1] - offset[0]) * paddle.ones_like(offset)
        self.register_buffer("widths", widths)
        self.register_buffer("offsets", offset)

    def forward(self, values: paddle.Tensor) -> paddle.Tensor:
        coeff = -0.5 / paddle.pow(self.widths, 2)[None, :]
        diff = values - self.offsets[None, :]
        return paddle.exp(coeff * paddle.pow(diff, 2))


class PropertyEmbedder(nn.Layer):
    """Embed scalar property values through a Gaussian expansion and an MLP."""

    def __init__(self, start: float, stop: float, embedding_dim: int, n_gaussians: int):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.mlp = nn.Sequential(
            nn.Linear(n_gaussians, embedding_dim),
            StableSiLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )
        self.gaussian_expansion = GaussianExpansion(
            start=start, stop=stop, n_gaussians=n_gaussians
        )

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        expanded_x = self.gaussian_expansion(x)
        return self.mlp(expanded_x)


def centered_normal_prior(n: int, d: int, std: float) -> paddle.Tensor:
    """Sample a zero-mean Gaussian prior for one molecule."""
    prior_feat = paddle.randn([n, d]) * std
    return prior_feat - prior_feat.mean(axis=0, keepdim=True)


def centered_normal_prior_batched(
    num_nodes: int, node_batch_idx: paddle.Tensor, std: float
) -> paddle.Tensor:
    """Sample a zero-mean Gaussian prior for every molecule in a batch."""
    prior_sample = paddle.randn([num_nodes, 3]) * std
    return (
        prior_sample - scatter_mean(prior_sample, node_batch_idx, dim=0)[node_batch_idx]
    )


def ctmc_masked_prior(n: int, d: int) -> paddle.Tensor:
    """Sample a CTMC masked prior where every sample is the mask token at t=0."""
    p = paddle.full([n], fill_value=d, dtype="int64")
    return paddle.nn.functional.one_hot(p, num_classes=d + 1).astype("float32")


def rigid_alignment(x_0: paddle.Tensor, x_1: paddle.Tensor) -> paddle.Tensor:
    """Align two point clouds with the Kabsch algorithm.

    Reference: https://en.wikipedia.org/wiki/Kabsch_algorithm
    """
    if x_0.shape != x_1.shape:
        raise ValueError("x_0 and x_1 must have the same shape")

    x_0_mean = x_0.mean(axis=0, keepdim=True)
    x_1_mean = x_1.mean(axis=0, keepdim=True)
    x_0_c = x_0 - x_0_mean
    x_1_c = x_1 - x_1_mean

    h = paddle.matmul(x_0_c, x_1_c, transpose_x=True)
    u, _, vh = paddle.linalg.svd(h, full_matrices=False)
    v = vh.transpose([1, 0])
    r = paddle.matmul(v, u, transpose_y=True)
    t = x_1_mean - paddle.matmul(x_0_mean, r, transpose_y=True)

    x_0_aligned = paddle.matmul(x_0_c, r, transpose_y=True)
    x_0_aligned = x_0_aligned + x_0_mean
    return x_0_aligned + t


def align_prior(
    prior_feat: paddle.Tensor,
    dst_feat: paddle.Tensor,
    permutation: bool = False,
    rigid_body: bool = False,
) -> paddle.Tensor:
    """Align a prior feature to a destination feature."""
    if permutation:
        cost_mat = paddle.cdist(dst_feat, prior_feat, p=2)
        _, prior_idx = linear_sum_assignment(cost_mat.numpy())
        prior_feat = prior_feat[paddle.to_tensor(prior_idx, dtype="int64")]

    if rigid_body:
        prior_feat = rigid_alignment(prior_feat, dst_feat)

    return prior_feat


class NodePositionUpdate(nn.Layer):
    """Update node positions with the vectors predicted by a stack of GVPs."""

    def __init__(self, n_scalars, n_vec_channels, n_gvps: int, n_cp_feats: int):
        super().__init__()
        gvps = []
        for i in range(n_gvps):
            if i == n_gvps - 1:
                vectors_out = 1
                vectors_activation = nn.Identity()
            else:
                vectors_out = n_vec_channels
                vectors_activation = nn.Sigmoid()

            gvps.append(
                GVP(
                    dim_feats_in=n_scalars,
                    dim_feats_out=n_scalars,
                    dim_vectors_in=n_vec_channels,
                    dim_vectors_out=vectors_out,
                    n_cp_feats=n_cp_feats,
                    vectors_activation=vectors_activation,
                )
            )
        self.gvps = nn.LayerList(gvps)

    def forward(self, scalars, positions, vectors):
        for gvp in self.gvps:
            scalars, vectors = gvp(scalars, vectors)
        return positions + vectors.squeeze(1)


class EdgeUpdate(nn.Layer):
    """Update edge features with source/destination scalars and distances."""

    def __init__(self, n_node_scalars, n_edge_feats, update_edge_w_distance, rbf_dim):
        super().__init__()
        self.update_edge_w_distance = update_edge_w_distance

        input_dim = n_node_scalars * 2 + n_edge_feats
        if update_edge_w_distance:
            input_dim += rbf_dim

        self.edge_update_fn = nn.Sequential(
            nn.Linear(input_dim, n_edge_feats),
            StableSiLU(),
            nn.Linear(n_edge_feats, n_edge_feats),
            StableSiLU(),
        )
        self.edge_norm = nn.LayerNorm(n_edge_feats)

    def forward(self, node_scalars, edge_feats, d, src, dst):
        mlp_inputs = [node_scalars[src], node_scalars[dst], edge_feats]
        if self.update_edge_w_distance:
            mlp_inputs.append(d)
        edge_feats = self.edge_norm(
            edge_feats + self.edge_update_fn(paddle.concat(mlp_inputs, axis=-1))
        )
        return edge_feats


def _categorical_sample(probs: paddle.Tensor) -> paddle.Tensor:
    """Draw one category per row; ``probs`` must be normalized probabilities."""
    return paddle.multinomial(probs, num_samples=1).squeeze(-1)


def _purity_sampling(
    xt,
    x1_probs,
    unmask_prob,
    mask_index,
    batch_size,
    node_batch_idx,
    hc_thresh,
):
    masked_nodes = xt == mask_index
    purities = paddle.max(x1_probs, axis=-1)

    hc_mask = purities >= hc_thresh
    hc_mask = hc_mask & masked_nodes

    hc_nodes_per_graph = scatter_sum(
        hc_mask.astype("float32"), node_batch_idx, dim=0, dim_size=batch_size
    )
    masked_nodes_per_graph = scatter_sum(
        masked_nodes.astype("float32"), node_batch_idx, dim=0, dim_size=batch_size
    )

    ph_max = unmask_prob * masked_nodes_per_graph / hc_nodes_per_graph
    ph_max = paddle.where(
        hc_nodes_per_graph == 0, paddle.full_like(ph_max, float("inf")), ph_max
    )
    ph = paddle.minimum(ph_max, paddle.ones_like(ph_max))
    pl = (unmask_prob * masked_nodes_per_graph - ph * hc_nodes_per_graph) / (
        masked_nodes_per_graph - hc_nodes_per_graph
    )

    node_unmask_prob = paddle.zeros_like(xt, dtype="float32")
    node_unmask_prob[hc_mask] = ph[node_batch_idx[hc_mask]]
    lc_mask = (purities < hc_thresh) & masked_nodes
    node_unmask_prob[lc_mask] = pl[node_batch_idx[lc_mask]]

    return paddle.rand(xt.shape) < node_unmask_prob


class PropMolFlowVectorField(RuntimeMixin, nn.Layer):
    """Continuous-time Markov chain vector field for molecular flow matching.

    Categorical features (atom types, charges, bond types) follow CTMC
    probability paths, positions follow linear interpolation paths. The
    endpoint is predicted by a GVP message passing network.
    """

    def __init__(
        self,
        n_atom_types: int,
        feature_names: list,
        interpolant_scheduler,
        property_embedding_dim: int,
        properties_handle_method: str,
        property_embedder,
        n_charges: int,
        n_bond_types: int,
        charge_offset: int,
        n_vec_channels: int = 16,
        n_cp_feats: int = 4,
        n_hidden_scalars: int = 256,
        n_hidden_edge_feats: int = 128,
        n_recycles: int = 1,
        n_molecule_updates: int = 8,
        convs_per_update: int = 1,
        n_message_gvps: int = 3,
        n_update_gvps: int = 3,
        n_position_update_gvps: int = 3,
        separate_mol_updaters: bool = True,
        message_norm=100,
        update_edge_w_distance: bool = True,
        rbf_dmax: float = 14,
        rbf_dim: int = 32,
        stochasticity: float = 10.0,
        high_confidence_threshold: float = 0.0,
        cat_temperature: float = 0.05,
        execution_backend: str = "eager",
        runtime_options: dict = None,
    ):
        super().__init__()
        self._init_runtime(execution_backend, runtime_options)

        if n_recycles < 1 or n_molecule_updates < 1 or convs_per_update < 1:
            raise ValueError(
                "n_recycles, n_molecule_updates and convs_per_update must be positive"
            )
        if n_molecule_updates == 1 and convs_per_update == 1:
            raise ValueError(
                "The released update schedule requires at least two convolutions "
                "to update molecular coordinates"
            )

        if properties_handle_method not in [
            "concatenate",
            "sum",
            "multiply",
            "concatenate_sum",
            "concatenate_multiply",
        ]:
            raise ValueError(
                "properties_handle_method must be one of 'concatenate', 'sum', "
                "'multiply', 'concatenate_sum', 'concatenate_multiply', got "
                f"{properties_handle_method}"
            )
        if property_embedder is None:
            raise ValueError("property_embedder must be provided")
        if properties_handle_method in ["sum", "multiply", "concatenate_multiply"] and (
            property_embedding_dim != n_hidden_scalars
        ):
            raise ValueError(
                f"properties_handle_method '{properties_handle_method}' requires "
                "property_embedding_dim == n_hidden_scalars, got "
                f"{property_embedding_dim} and {n_hidden_scalars}"
            )

        self.n_atom_types = n_atom_types
        self.charge_offset = charge_offset
        self.n_vec_channels = n_vec_channels
        self.rbf_dmax = rbf_dmax
        self.rbf_dim = rbf_dim
        self.convs_per_update = convs_per_update
        self.n_recycles = n_recycles
        self.separate_mol_updaters = separate_mol_updaters
        self.interpolant_scheduler = interpolant_scheduler
        self.feature_names = feature_names
        self.properties_handle_method = properties_handle_method
        self.property_embedder = property_embedder

        self.stochasticity = stochasticity
        self.high_confidence_threshold = high_confidence_threshold
        self.cat_temperature = cat_temperature

        n_mask_feats = 1

        self.n_cat_feats = {
            "atom_types": n_atom_types,
            "atom_charges": n_charges,
            "bond_types": n_bond_types,
        }

        self.scalar_embedding_uncond = nn.Sequential(
            nn.Linear(
                n_atom_types + n_charges + 1 + 2 * n_mask_feats, n_hidden_scalars
            ),
            StableSiLU(),
            nn.Linear(n_hidden_scalars, n_hidden_scalars),
            StableSiLU(),
            nn.LayerNorm(n_hidden_scalars),
        )

        self.scalar_embedding_cond = nn.Sequential(
            nn.Linear(n_hidden_scalars + property_embedding_dim, n_hidden_scalars),
            StableSiLU(),
            nn.Linear(n_hidden_scalars, n_hidden_scalars),
        )

        self.edge_embedding = nn.Sequential(
            nn.Linear(n_bond_types + n_mask_feats, n_hidden_edge_feats),
            StableSiLU(),
            nn.Linear(n_hidden_edge_feats, n_hidden_edge_feats),
            StableSiLU(),
            nn.LayerNorm(n_hidden_edge_feats),
        )

        conv_layers = []
        for _ in range(convs_per_update * n_molecule_updates):
            conv_layers.append(
                GVPConv(
                    scalar_size=n_hidden_scalars,
                    vector_size=n_vec_channels,
                    n_cp_feats=n_cp_feats,
                    edge_feat_size=n_hidden_edge_feats,
                    n_message_gvps=n_message_gvps,
                    n_update_gvps=n_update_gvps,
                    message_norm=message_norm,
                    rbf_dim=rbf_dim,
                )
            )
        self.conv_layers = nn.LayerList(conv_layers)

        n_updaters = n_molecule_updates if separate_mol_updaters else 1
        self.node_position_updaters = nn.LayerList(
            [
                NodePositionUpdate(
                    n_hidden_scalars,
                    n_vec_channels,
                    n_gvps=n_position_update_gvps,
                    n_cp_feats=n_cp_feats,
                )
                for _ in range(n_updaters)
            ]
        )
        self.edge_updaters = nn.LayerList(
            [
                EdgeUpdate(
                    n_hidden_scalars,
                    n_hidden_edge_feats,
                    update_edge_w_distance=update_edge_w_distance,
                    rbf_dim=rbf_dim,
                )
                for _ in range(n_updaters)
            ]
        )

        # Frozen checkpoint-only layers preserve the released parameter layout.
        checkpoint_only_layers = []
        if properties_handle_method not in ["concatenate", "concatenate_sum"]:
            checkpoint_only_layers.append(self.scalar_embedding_cond)
        if separate_mol_updaters and convs_per_update == 1:
            checkpoint_only_layers.extend(
                [self.node_position_updaters[0], self.edge_updaters[0]]
            )
        for layer in checkpoint_only_layers:
            for parameter in layer.parameters():
                parameter.stop_gradient = True

        self.node_output_head = nn.Sequential(
            nn.Linear(n_hidden_scalars, n_hidden_scalars),
            StableSiLU(),
            nn.Linear(n_hidden_scalars, n_atom_types + n_charges),
        )

        self.to_edge_logits = nn.Sequential(
            nn.Linear(n_hidden_edge_feats, n_hidden_edge_feats),
            StableSiLU(),
            nn.Linear(n_hidden_edge_feats, n_bond_types),
        )

    def precompute_distances(self, node_positions, src, dst):
        """Precompute normalized pairwise displacement and RBF distances."""
        x_diff = node_positions[src] - node_positions[dst]
        dij = norm_no_nan(x_diff, keepdims=True) + 1e-8
        x_diff = x_diff / dij
        d = _rbf(dij.squeeze(1), D_max=self.rbf_dmax, D_count=self.rbf_dim)
        return x_diff, d

    def _velocity(self, x_t, x_1, alpha_t, alpha_t_prime):
        return alpha_t_prime / (1 - alpha_t) * (x_1 - x_t)

    def _apply_property_conditioning(self, node_scalar_features, prop_emb):
        if self.properties_handle_method == "concatenate_sum":
            intermediate = paddle.concat([node_scalar_features, prop_emb], axis=-1)
            intermediate = self.scalar_embedding_cond(intermediate)
            return node_scalar_features + intermediate
        if self.properties_handle_method == "concatenate":
            intermediate = paddle.concat([node_scalar_features, prop_emb], axis=-1)
            return self.scalar_embedding_cond(intermediate)
        if self.properties_handle_method == "sum":
            return node_scalar_features + prop_emb
        # multiply and concatenate_multiply share the same gating: the upstream
        # scalar_embedding_cond call of concatenate_multiply is dead (its output
        # is discarded), so it is dropped here, numerically equal.
        gate = paddle.nn.functional.sigmoid(prop_emb) + 0.5
        return node_scalar_features * gate

    def forward(
        self,
        state: dict,
        t: paddle.Tensor,
        node_batch_idx: paddle.Tensor,
        upper_edge_mask: paddle.Tensor,
        apply_softmax: bool = False,
        remove_com: bool = False,
        properties: paddle.Tensor = None,
    ) -> dict:
        """Predict the endpoint of the trajectory given the state at time t."""
        return self._runtime_denoise_step(
            state,
            t,
            node_batch_idx,
            upper_edge_mask,
            apply_softmax=apply_softmax,
            remove_com=remove_com,
            properties=properties,
        )

    @runtime_boundary("denoise_step")
    def _runtime_denoise_step(
        self,
        state: dict,
        t: paddle.Tensor,
        node_batch_idx: paddle.Tensor,
        upper_edge_mask: paddle.Tensor,
        apply_softmax: bool = False,
        remove_com: bool = False,
        properties: paddle.Tensor = None,
    ) -> dict:
        time_tensor = t[:, 0] if t.ndim > 1 else t
        node_scalar_features = paddle.concat(
            [
                state["atom_types"],
                time_tensor[node_batch_idx].unsqueeze(-1),
                state["atom_charges"],
            ],
            axis=-1,
        )
        node_scalar_features = self.scalar_embedding_uncond(node_scalar_features)

        if t.ndim > 1:
            prop_emb = t[:, 1:][node_batch_idx]
        else:
            if properties is None:
                raise ValueError("properties must be provided during sampling")
            prop_emb = self.property_embedder(properties.reshape([-1, 1]))
            prop_emb = prop_emb[node_batch_idx]
        node_scalar_features = self._apply_property_conditioning(
            node_scalar_features, prop_emb
        )

        node_positions = state["positions"]
        num_nodes = node_positions.shape[0]
        node_vec_features = paddle.zeros(
            [num_nodes, self.n_vec_channels, 3], dtype=node_positions.dtype
        )

        edge_features = self.edge_embedding(state["bond_types"])

        src, dst = state["src"], state["dst"]
        x_diff, d = self.precompute_distances(node_positions, src, dst)

        for _ in range(self.n_recycles):
            for conv_idx, conv in enumerate(self.conv_layers):
                node_scalar_features, node_vec_features = conv(
                    scalar_feats=node_scalar_features,
                    vec_feats=node_vec_features,
                    src=src,
                    dst=dst,
                    num_nodes=num_nodes,
                    edge_feats=edge_features,
                    x_diff=x_diff,
                    d=d,
                )

                if conv_idx != 0 and (conv_idx + 1) % self.convs_per_update == 0:
                    if self.separate_mol_updaters:
                        updater_idx = conv_idx // self.convs_per_update
                    else:
                        updater_idx = 0

                    node_positions = self.node_position_updaters[updater_idx](
                        node_scalar_features, node_positions, node_vec_features
                    )
                    x_diff, d = self.precompute_distances(node_positions, src, dst)
                    edge_features = self.edge_updaters[updater_idx](
                        node_scalar_features, edge_features, d=d, src=src, dst=dst
                    )

        node_scalar_features = self.node_output_head(node_scalar_features)
        atom_type_logits = node_scalar_features[:, : self.n_atom_types]
        atom_charge_logits = node_scalar_features[:, self.n_atom_types :]

        ue_feats = edge_features[upper_edge_mask]
        le_feats = edge_features[~upper_edge_mask]
        edge_logits = self.to_edge_logits(ue_feats + le_feats)

        if remove_com:
            node_positions = (
                node_positions
                - scatter_mean(node_positions, node_batch_idx, dim=0)[node_batch_idx]
            )

        output = {
            "positions": node_positions,
            "atom_types": atom_type_logits,
            "atom_charges": atom_charge_logits,
            "bond_types": edge_logits,
        }

        if apply_softmax:
            for feat in ["atom_types", "atom_charges", "bond_types"]:
                output[feat] = paddle.nn.functional.softmax(output[feat], axis=-1)

        return output

    def sample_conditional_path(
        self,
        batch: dict,
        t: paddle.Tensor,
        prior_positions: paddle.Tensor,
        node_batch_idx: paddle.Tensor,
        edge_batch_idx: paddle.Tensor,
        upper_edge_mask: paddle.Tensor,
    ) -> dict:
        """Sample the CTMC probability path between the prior and the endpoint."""
        _, alpha_t = self.interpolant_scheduler.interpolant_weights(t)

        state = {
            "src": batch["src"],
            "dst": batch["dst"],
            "upper_edge_mask": upper_edge_mask,
            "node_batch_idx": node_batch_idx,
            "edge_batch_idx": edge_batch_idx,
            "batch_size": batch["batch_size"],
        }

        positions_idx = self.feature_names.index("positions")
        dst_weight = alpha_t[:, positions_idx][node_batch_idx].unsqueeze(-1)
        state["positions"] = (1 - dst_weight) * prior_positions + dst_weight * batch[
            "positions"
        ]

        for feat in ["atom_types", "atom_charges"]:
            feat_idx = self.feature_names.index(feat)
            tokens = batch[feat].argmax(-1)
            alpha_feat = alpha_t[:, feat_idx][node_batch_idx]
            mask = paddle.rand(tokens.shape, dtype=alpha_feat.dtype) < (1 - alpha_feat)
            tokens = paddle.where(
                mask,
                paddle.full_like(tokens, self.n_cat_feats[feat]),
                tokens,
            )
            state[feat] = paddle.nn.functional.one_hot(
                tokens, num_classes=self.n_cat_feats[feat] + 1
            ).astype("float32")

        bond_idx = self.feature_names.index("bond_types")
        alpha_bond = alpha_t[:, bond_idx][edge_batch_idx][upper_edge_mask]
        tokens = batch["bond_types"][upper_edge_mask].argmax(-1)
        mask = paddle.rand(tokens.shape, dtype=alpha_bond.dtype) < (1 - alpha_bond)
        tokens = paddle.where(
            mask,
            paddle.full_like(tokens, self.n_cat_feats["bond_types"]),
            tokens,
        )
        upper_states = paddle.nn.functional.one_hot(
            tokens, num_classes=self.n_cat_feats["bond_types"] + 1
        ).astype("float32")
        bond_states = paddle.zeros(
            [batch["bond_types"].shape[0], self.n_cat_feats["bond_types"] + 1],
            dtype=batch["bond_types"].dtype,
        )
        bond_states[upper_edge_mask] = upper_states
        bond_states[~upper_edge_mask] = upper_states
        state["bond_types"] = bond_states
        return state

    @paddle.no_grad()
    def integrate(
        self,
        state: dict,
        node_batch_idx: paddle.Tensor,
        upper_edge_mask: paddle.Tensor,
        n_timesteps: int,
        properties: paddle.Tensor = None,
        stochasticity: float = None,
        high_confidence_threshold: float = None,
    ) -> dict:
        """Integrate the molecule trajectories from the prior to the endpoint."""
        if n_timesteps < 2:
            raise ValueError("num_inference_steps must be at least 2")
        t = paddle.linspace(0, 1, n_timesteps)
        alpha_t = self.interpolant_scheduler.alpha_t(t)
        alpha_t_prime = self.interpolant_scheduler.alpha_t_prime(t)

        for s_idx in range(1, t.shape[0]):
            s_i = t[s_idx]
            t_i = t[s_idx - 1]
            alpha_t_i = alpha_t[s_idx - 1]
            alpha_s_i = alpha_t[s_idx]
            alpha_t_prime_i = alpha_t_prime[s_idx - 1]
            last_step = s_idx == t.shape[0] - 1

            state = self.step(
                state,
                s_i,
                t_i,
                alpha_t_i,
                alpha_s_i,
                alpha_t_prime_i,
                node_batch_idx,
                upper_edge_mask,
                properties=properties,
                stochasticity=stochasticity,
                high_confidence_threshold=high_confidence_threshold,
                last_step=last_step,
            )

        return state

    def step(
        self,
        state: dict,
        s_i: paddle.Tensor,
        t_i: paddle.Tensor,
        alpha_t_i: paddle.Tensor,
        alpha_s_i: paddle.Tensor,
        alpha_t_prime_i: paddle.Tensor,
        node_batch_idx: paddle.Tensor,
        upper_edge_mask: paddle.Tensor,
        properties: paddle.Tensor = None,
        stochasticity: float = None,
        high_confidence_threshold: float = None,
        last_step: bool = False,
    ) -> dict:
        eta = self.stochasticity if stochasticity is None else stochasticity
        hc_thresh = (
            self.high_confidence_threshold
            if high_confidence_threshold is None
            else high_confidence_threshold
        )

        output = self(
            state,
            t=paddle.full([state["batch_size"]], t_i),
            node_batch_idx=node_batch_idx,
            upper_edge_mask=upper_edge_mask,
            apply_softmax=True,
            remove_com=True,
            properties=properties,
        )

        dt = s_i - t_i

        x_1 = output["positions"]
        x_t = state["positions"]
        vf = self._velocity(x_t, x_1, alpha_t_i[0], alpha_t_prime_i[0])
        state["positions"] = x_t + dt * vf

        for feat_idx, feat in enumerate(self.feature_names):
            if feat == "positions":
                continue

            xt = state[feat].argmax(-1)
            if feat == "bond_types":
                xt = xt[upper_edge_mask]

            p_s_1 = output[feat]
            p_s_1 = paddle.nn.functional.softmax(
                paddle.log(p_s_1) / self.cat_temperature, axis=-1
            )

            xt = self.campbell_step(
                p_1_given_t=p_s_1,
                xt=xt,
                stochasticity=eta,
                hc_thresh=hc_thresh,
                alpha_t=alpha_t_i[feat_idx],
                alpha_t_prime=alpha_t_prime_i[feat_idx],
                dt=dt,
                batch_size=state["batch_size"],
                n_classes=self.n_cat_feats[feat] + 1,
                mask_index=self.n_cat_feats[feat],
                last_step=last_step,
                batch_idx=(
                    state["edge_batch_idx"][upper_edge_mask]
                    if feat == "bond_types"
                    else node_batch_idx
                ),
            )

            if feat == "bond_types":
                bond_states = paddle.zeros_like(state["bond_types"])
                bond_states[upper_edge_mask] = xt
                bond_states[~upper_edge_mask] = xt
                xt = bond_states

            state[feat] = xt

        return state

    def campbell_step(
        self,
        p_1_given_t: paddle.Tensor,
        xt: paddle.Tensor,
        stochasticity: float,
        hc_thresh: float,
        alpha_t: paddle.Tensor,
        alpha_t_prime: paddle.Tensor,
        dt: paddle.Tensor,
        batch_size: int,
        n_classes: int,
        mask_index: int,
        last_step: bool,
        batch_idx: paddle.Tensor,
    ):
        x1 = _categorical_sample(p_1_given_t)

        unmask_prob = dt * (alpha_t_prime + stochasticity * alpha_t) / (1 - alpha_t)
        mask_prob = dt * stochasticity
        unmask_prob = paddle.clip(unmask_prob, min=0, max=1)
        mask_prob = paddle.clip(mask_prob, min=0, max=1)

        if hc_thresh > 0:
            will_unmask = _purity_sampling(
                xt=xt,
                x1_probs=p_1_given_t,
                unmask_prob=unmask_prob,
                mask_index=mask_index,
                batch_size=batch_size,
                node_batch_idx=batch_idx,
                hc_thresh=hc_thresh,
            )
        else:
            will_unmask = paddle.rand(xt.shape) < unmask_prob
            will_unmask = will_unmask & (xt == mask_index)

        if not last_step:
            will_mask = paddle.rand(xt.shape) < mask_prob
            will_mask = will_mask & (xt != mask_index)
            xt = paddle.where(will_mask, paddle.full_like(xt, mask_index), xt)

        xt = paddle.where(will_unmask, x1, xt)
        return paddle.nn.functional.one_hot(xt, num_classes=n_classes).astype("float32")


class PropMolFlow(RuntimeMixin, nn.Layer):
    """Property-guided molecule generation with geometry-complete flow matching.

    The numerical runtime is owned by :class:`PropMolFlowVectorField`; this class
    forwards the execution backend to it and keeps the dataset-level contracts.
    """

    def __init__(
        self,
        atom_type_map: list,
        total_loss_weights: dict,
        interpolant_scheduler_config: dict,
        vector_field_config: dict,
        prior_config: dict,
        property_embedding_dim: int,
        properties_handle_method: str,
        gaussian_expansion: bool,
        n_gaussians: int,
        gaussian_start: float = None,
        gaussian_stop: float = None,
        execution_backend: str = "eager",
        runtime_options: dict = None,
    ):
        super().__init__()

        self.atom_type_map = list(atom_type_map)
        self.n_atom_types = len(self.atom_type_map)
        self.prior_config = prior_config
        self.feature_names = FEATURE_NAMES

        for feat in self.feature_names:
            if feat not in total_loss_weights:
                raise ValueError(
                    f"total_loss_weights must provide a weight for feature {feat}"
                )
        self.total_loss_weights = total_loss_weights

        if prior_config["positions"]["type"] != "centered-normal":
            raise ValueError(
                "prior_config['positions']['type'] must be 'centered-normal', got "
                f"{prior_config['positions']['type']}"
            )
        for feat in ["atom_types", "atom_charges", "bond_types"]:
            if prior_config[feat]["type"] != "ctmc":
                raise ValueError(
                    f"prior_config['{feat}']['type'] must be 'ctmc', got "
                    f"{prior_config[feat]['type']}"
                )

        if gaussian_expansion:
            if gaussian_start is None or gaussian_stop is None:
                raise ValueError(
                    "gaussian_start and gaussian_stop must be provided when "
                    "gaussian_expansion is enabled"
                )
            self.property_embedder = PropertyEmbedder(
                start=gaussian_start,
                stop=gaussian_stop,
                embedding_dim=property_embedding_dim,
                n_gaussians=n_gaussians,
            )
        else:
            self.property_embedder = nn.Sequential(
                nn.Linear(1, property_embedding_dim),
                StableSiLU(),
                nn.Linear(property_embedding_dim, property_embedding_dim),
                nn.LayerNorm(property_embedding_dim),
            )

        scheduler_cfg = copy.deepcopy(interpolant_scheduler_config)
        scheduler_cfg["__init_params__"]["feature_names"] = self.feature_names
        self.interpolant_scheduler = build_scheduler(scheduler_cfg)

        self.vector_field = PropMolFlowVectorField(
            n_atom_types=self.n_atom_types,
            feature_names=self.feature_names,
            interpolant_scheduler=self.interpolant_scheduler,
            property_embedding_dim=property_embedding_dim,
            properties_handle_method=properties_handle_method,
            property_embedder=self.property_embedder,
            execution_backend=execution_backend,
            runtime_options=runtime_options,
            **vector_field_config,
        )

        self.loss_fn_positions = nn.MSELoss(reduction="mean")
        self.loss_fn_categorical = nn.CrossEntropyLoss(
            reduction="mean", ignore_index=-100
        )

    @property
    def execution_backend(self):
        """Execution backend of the numerical vector field."""
        return self.vector_field.execution_backend

    def set_execution_backend(self, backend):
        self.vector_field.set_execution_backend(backend)

    def set_runtime_options(self, runtime_options):
        self.vector_field.set_runtime_options(runtime_options)

    def validate_execution_backend(self, *, use_amp=False, world_size=1):
        self.vector_field.validate_execution_backend(
            use_amp=use_amp, world_size=world_size
        )

    def _prepare_training_batch(self, batch: dict) -> dict:
        structure_array = batch["structure_array"]
        place = next(iter(self.parameters())).place
        num_atoms = paddle.to_tensor(
            structure_array["num_atoms"], dtype="int64", place=place
        ).reshape([-1])
        num_atoms_np = num_atoms.numpy()
        graph = build_batched_complete_graph(num_atoms_np)
        prepared = {
            key: paddle.to_tensor(value, place=place) for key, value in graph.items()
        }
        prepared["structure_array"] = {"num_atoms": num_atoms}
        prepared["batch_size"] = len(num_atoms_np)
        for key in ("positions", "atom_types", "atom_charges"):
            prepared[key] = paddle.to_tensor(
                structure_array[key], dtype="float32", place=place
            )
        upper_bonds = paddle.to_tensor(
            structure_array["bond_types"], dtype="float32", place=place
        )
        bond_parts = []
        offset = 0
        for n in num_atoms_np:
            n_edges = int(n * (n - 1) // 2)
            bonds = upper_bonds[offset : offset + n_edges]
            bond_parts.extend((bonds, bonds))
            offset += n_edges
        prepared["bond_types"] = paddle.concat(bond_parts, axis=0)
        prepared["prop"] = paddle.to_tensor(batch["prop"], dtype="float32", place=place)
        if "prior_positions" in batch:
            prepared["prior_positions"] = paddle.to_tensor(
                batch["prior_positions"], dtype="float32", place=place
            )
        return prepared

    def _sample_prior_positions(self, batch: dict) -> paddle.Tensor:
        """Sample and align the position prior of every molecule."""
        place = batch["positions"].place
        num_atoms = batch["structure_array"]["num_atoms"].numpy()

        prior_list = []
        node_offset = 0
        for n in num_atoms:
            n = int(n)
            positions = batch["positions"][node_offset : node_offset + n]
            prior = centered_normal_prior(
                n, 3, **self.prior_config["positions"]["kwargs"]
            ).to(place)
            if self.prior_config["positions"]["align"]:
                prior = align_prior(prior, positions, permutation=True, rigid_body=True)
            prior_list.append(prior)
            node_offset += n
        return paddle.concat(prior_list, axis=0)

    def _build_targets(self, batch: dict, state: dict, upper_edge_mask) -> dict:
        """Build the loss targets; already unmasked states are ignored."""
        targets = {"positions": batch["positions"]}
        for feat in ["atom_types", "atom_charges"]:
            target = batch[feat].argmax(-1)
            target[state[feat].argmax(-1) != self.vector_field.n_cat_feats[feat]] = -100
            targets[feat] = target

        target = batch["bond_types"][upper_edge_mask].argmax(-1)
        bond_states = state["bond_types"][upper_edge_mask].argmax(-1)
        target[bond_states != self.vector_field.n_cat_feats["bond_types"]] = -100
        targets["bond_types"] = target
        return targets

    def forward(self, batch: dict) -> dict:
        """Compute the flow matching losses of a training batch.

        The ``pred_dict`` is not produced because the model predicts trajectory
        endpoints rather than per-attribute labels, so ``compute_metric_func_dict``
        does not apply to this model.
        """
        batch = self._prepare_training_batch(batch)
        batch_size = batch["batch_size"]
        node_batch_idx = batch["node_batch_idx"]
        upper_edge_mask = batch["upper_edge_mask"]

        t = paddle.rand([batch_size])
        prop = batch["prop"].astype("float32").reshape([batch_size, 1])
        prop_emb = self.property_embedder(prop)
        t_combined = paddle.concat([t.unsqueeze(-1), prop_emb], axis=-1)

        prior_positions = batch.get("prior_positions")
        if prior_positions is None:
            prior_positions = self._sample_prior_positions(batch)

        state = self.vector_field.sample_conditional_path(
            batch,
            t,
            prior_positions,
            node_batch_idx,
            batch["edge_batch_idx"],
            upper_edge_mask,
        )
        output = self.vector_field(
            state,
            t_combined,
            node_batch_idx=node_batch_idx,
            upper_edge_mask=upper_edge_mask,
        )

        targets = self._build_targets(batch, state, upper_edge_mask)
        losses = {
            "positions": self.loss_fn_positions(
                output["positions"], targets["positions"]
            ),
            "atom_types": self.loss_fn_categorical(
                output["atom_types"], targets["atom_types"]
            ),
            "atom_charges": self.loss_fn_categorical(
                output["atom_charges"], targets["atom_charges"]
            ),
            "bond_types": self.loss_fn_categorical(
                output["bond_types"], targets["bond_types"]
            ),
        }

        total_loss = None
        for feat in self.feature_names:
            term = self.total_loss_weights[feat] * losses[feat]
            total_loss = term if total_loss is None else total_loss + term

        loss_dict = {"loss": total_loss}
        for feat in self.feature_names:
            loss_dict[f"loss_{feat}"] = losses[feat]
        return {"loss_dict": loss_dict}

    def _build_sampling_state(self, num_atoms: paddle.Tensor, place) -> dict:
        """Build the initial sampling state from the prior distributions."""
        num_atoms_np = num_atoms.numpy().astype("int64")
        graph = build_batched_complete_graph(num_atoms_np)
        state = {
            "src": paddle.to_tensor(graph["src"], dtype="int64", place=place),
            "dst": paddle.to_tensor(graph["dst"], dtype="int64", place=place),
            "upper_edge_mask": paddle.to_tensor(graph["upper_edge_mask"], place=place),
            "node_batch_idx": paddle.to_tensor(
                graph["node_batch_idx"], dtype="int64", place=place
            ),
            "edge_batch_idx": paddle.to_tensor(
                graph["edge_batch_idx"], dtype="int64", place=place
            ),
            "batch_size": len(num_atoms_np),
        }

        num_nodes = int(state["node_batch_idx"].shape[0])
        num_edges = int(state["upper_edge_mask"].shape[0])
        state["positions"] = centered_normal_prior_batched(
            num_nodes,
            state["node_batch_idx"],
            **self.prior_config["positions"]["kwargs"],
        ).to(place)
        state["atom_types"] = ctmc_masked_prior(
            num_nodes, self.vector_field.n_cat_feats["atom_types"]
        ).to(place)
        state["atom_charges"] = ctmc_masked_prior(
            num_nodes, self.vector_field.n_cat_feats["atom_charges"]
        ).to(place)
        state["bond_types"] = ctmc_masked_prior(
            num_edges, self.vector_field.n_cat_feats["bond_types"]
        ).to(place)
        return state

    @paddle.no_grad()
    def sample(
        self,
        batch_data: dict,
        num_inference_steps: int = 100,
        stochasticity: float = None,
        high_confidence_threshold: float = None,
    ) -> dict:
        """Sample molecules conditioned on the properties in ``batch_data``.

        Args:
            batch_data (dict): A batch containing ``structure_array.num_atoms``
                and the normalized property values under the ``properties`` key.
            num_inference_steps (int): Number of integration steps.
            stochasticity (float): CTMC stochasticity; uses the configured value
                when None.
            high_confidence_threshold (float): Purity sampling threshold; uses
                the configured value when None.
        """
        num_atoms = batch_data["structure_array"]["num_atoms"].reshape([-1])
        batch_size = num_atoms.shape[0]
        place = next(iter(self.parameters())).place

        properties = batch_data["properties"].astype("float32")
        properties = properties.reshape([batch_size, 1]).to(place)

        state = self._build_sampling_state(num_atoms, place)
        state = self.vector_field.integrate(
            state,
            state["node_batch_idx"],
            state["upper_edge_mask"],
            n_timesteps=num_inference_steps,
            properties=properties,
            stochasticity=stochasticity,
            high_confidence_threshold=high_confidence_threshold,
        )

        return {"pred": self._decode_samples(state, num_atoms)}

    def save_samples(self, molecules: list, output_dir: str) -> None:
        """Persist sampled molecules into an SDF file under ``output_dir``."""
        from rdkit import Chem

        from ppmat.models.propmolflow.molecule_builder import build_rdkit_mol

        os.makedirs(output_dir, exist_ok=True)
        sdf_path = osp.join(output_dir, "propmolflow_samples.sdf")
        writer = Chem.SDWriter(sdf_path)
        writer.SetKekulize(False)
        for molecule in molecules:
            writer.write(build_rdkit_mol(molecule))
        writer.close()
        logger.info(f"Saved {len(molecules)} molecules to {sdf_path}")

    def _decode_samples(self, state: dict, num_atoms: paddle.Tensor) -> list:
        """Convert the sampled state into per-molecule dictionaries."""
        positions = state["positions"].numpy()
        atom_types = state["atom_types"].argmax(-1).numpy()
        charges = (
            state["atom_charges"].argmax(-1).numpy() - self.vector_field.charge_offset
        )
        src = state["src"].numpy()[state["upper_edge_mask"].numpy()]
        dst = state["dst"].numpy()[state["upper_edge_mask"].numpy()]
        bond_types = (
            state["bond_types"].argmax(-1).numpy()[state["upper_edge_mask"].numpy()]
        )

        results = []
        node_offset = 0
        edge_offset = 0
        for n in num_atoms.numpy().astype("int64"):
            n = int(n)
            n_edges = n * (n - 1) // 2
            node_slice = slice(node_offset, node_offset + n)
            edge_slice = slice(edge_offset, edge_offset + n_edges)

            molecule_src = src[edge_slice] - node_offset
            molecule_dst = dst[edge_slice] - node_offset
            molecule_bonds = bond_types[edge_slice]
            n_bond_types = self.vector_field.n_cat_feats["bond_types"]
            keep = (molecule_bonds != 0) & (molecule_bonds < n_bond_types)

            molecule_atom_types = atom_types[node_slice]
            symbols = [
                self.atom_type_map[idx] if idx < self.n_atom_types else MASK_ATOM_SYMBOL
                for idx in molecule_atom_types
            ]

            results.append(
                {
                    "num_atoms": n,
                    "atom_types": symbols,
                    "positions": positions[node_slice].tolist(),
                    "charges": charges[node_slice].tolist(),
                    "bonds": [
                        [int(s), int(d), int(b)]
                        for s, d, b in zip(
                            molecule_src[keep],
                            molecule_dst[keep],
                            molecule_bonds[keep],
                        )
                    ],
                }
            )
            node_offset += n
            edge_offset += n_edges

        return results

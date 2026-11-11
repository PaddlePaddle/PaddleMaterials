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

"""CTMC vector field numerical core of the FlowMol model.

Categorical features (atom types, charges, bond types) follow CTMC
probability paths, positions follow linear interpolation paths. The
endpoint is predicted by a GVP message passing network. Ported from the
upstream PyTorch implementation (vector_field.py, ctmc_vector_field.py,
self_conditioning.py, priors.py); DGL message passing is replaced by direct
edge indexing plus ppmat scatter reductions.
"""

from __future__ import annotations

import math

import paddle
import paddle.nn as nn

from ppmat.models.common.runtime import RuntimeMixin
from ppmat.models.common.runtime import runtime_boundary
from ppmat.models.flowmol.gvp import GVP
from ppmat.models.flowmol.gvp import GVPConv
from ppmat.models.flowmol.gvp import norm_no_nan
from ppmat.utils.scatter import scatter_mean
from ppmat.utils.scatter import scatter_sum


def _rbf(dist, d_max, d_count, d_min=0.0):
    """Radial basis function embedding of pairwise distances."""
    d_mu = paddle.linspace(d_min, d_max, d_count)
    d_mu = d_mu.reshape([1, -1])
    d_sigma = (d_max - d_min) / d_count
    d_expand = paddle.unsqueeze(dist, -1)
    return paddle.exp(-(((d_expand - d_mu) / d_sigma) ** 2))


def get_time_embedding(timesteps, embedding_dim=256, max_positions=1000):
    """Sinusoidal time embedding of a 1-D tensor of timesteps."""
    if len(timesteps.shape) != 1:
        raise ValueError("timesteps must be a 1-D tensor")
    timesteps = timesteps * max_positions
    half_dim = embedding_dim // 2
    emb = math.log(max_positions) / (half_dim - 1)
    emb = paddle.exp(paddle.arange(half_dim, dtype="float32") * -emb)
    emb = timesteps.astype("float32").unsqueeze(1) * emb.unsqueeze(0)
    emb = paddle.concat([paddle.sin(emb), paddle.cos(emb)], axis=1)
    if embedding_dim % 2 == 1:
        emb = paddle.nn.functional.pad(emb, [0, 1, 0, 0])
    return emb


def centered_normal_prior_batched(
    num_nodes: int, node_batch_idx: paddle.Tensor, std: float
) -> paddle.Tensor:
    """Sample a zero-mean Gaussian prior for every molecule in a batch."""
    prior_sample = paddle.randn([num_nodes, 3]) * std
    return (
        prior_sample - scatter_mean(prior_sample, node_batch_idx, dim=0)[node_batch_idx]
    )


def ctmc_masked_prior(n: int, n_classes: int) -> paddle.Tensor:
    """Sample a CTMC masked prior where every sample is the mask token at t=0.

    The mask token is appended after the real classes, so its index equals
    ``n_classes``.
    """
    tokens = paddle.full([n], fill_value=n_classes, dtype="int64")
    return paddle.nn.functional.one_hot(tokens, num_classes=n_classes + 1).astype(
        "float32"
    )


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
    """Sample nodes to unmask, favoring high-confidence predictions."""
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

    return paddle.rand([xt.shape[0]]) < node_unmask_prob


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
            nn.SiLU(),
            nn.Linear(n_edge_feats, n_edge_feats),
            nn.SiLU(),
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


class SelfConditioningResidualLayer(nn.Layer):
    """Residual conditioning on a previously predicted trajectory endpoint."""

    def __init__(
        self,
        n_atom_types,
        n_charges,
        n_bond_types,
        node_embedding_dim,
        edge_embedding_dim,
        rbf_dim,
        rbf_dmax,
    ):
        super().__init__()

        self.rbf_dim = rbf_dim
        self.rbf_dmax = rbf_dmax

        self.node_residual_mlp = nn.Sequential(
            nn.Linear(
                node_embedding_dim + n_atom_types + n_charges + rbf_dim,
                node_embedding_dim,
            ),
            nn.SiLU(),
            nn.Linear(node_embedding_dim, node_embedding_dim),
            nn.SiLU(),
        )

        self.edge_residual_mlp = nn.Sequential(
            nn.Linear(edge_embedding_dim + n_bond_types + rbf_dim, edge_embedding_dim),
            nn.SiLU(),
            nn.Linear(edge_embedding_dim, edge_embedding_dim),
            nn.SiLU(),
        )

    def forward(
        self,
        src,
        dst,
        s_t,
        x_t,
        v_t,
        e_t,
        dst_dict,
        upper_edge_mask,
    ):
        d_node = norm_no_nan(x_t - dst_dict["positions"])
        d_node = _rbf(d_node, d_max=self.rbf_dmax, d_count=self.rbf_dim)

        node_residual_inputs = [
            s_t,
            dst_dict["atom_types"],
            dst_dict["atom_charges"],
            d_node,
        ]
        node_residual = self.node_residual_mlp(
            paddle.concat(node_residual_inputs, axis=-1)
        )

        d_edge_t = self.edge_distances(src, dst, node_positions=x_t)
        d_edge_1 = self.edge_distances(src, dst, node_positions=dst_dict["positions"])

        d_edge_t = d_edge_t[upper_edge_mask]
        d_edge_1 = d_edge_1[upper_edge_mask]

        edge_residual_inputs = [
            e_t[upper_edge_mask],
            dst_dict["bond_types"],
            d_edge_1 - d_edge_t,
        ]
        edge_residual = self.edge_residual_mlp(
            paddle.concat(edge_residual_inputs, axis=-1)
        )

        node_feats_out = s_t + node_residual
        positions_out = x_t
        vectors_out = v_t

        edge_feats_out = paddle.zeros_like(e_t)
        one_triangle_output = e_t[upper_edge_mask] + edge_residual
        edge_feats_out[upper_edge_mask] = one_triangle_output
        edge_feats_out[~upper_edge_mask] = one_triangle_output

        return node_feats_out, positions_out, vectors_out, edge_feats_out

    def edge_distances(self, src, dst, node_positions=None):
        """Compute the RBF embedding of every edge length."""
        x_diff = node_positions[src] - node_positions[dst]
        dij = norm_no_nan(x_diff, keepdims=True) + 1e-8
        return _rbf(dij.squeeze(1), d_max=self.rbf_dmax, d_count=self.rbf_dim)


class CTMCVectorField(RuntimeMixin, nn.Layer):
    """Continuous-time Markov chain vector field for molecular flow matching.

    Only the ``campbell`` discrete-flow type is ported; the upstream
    configuration of every released checkpoint and the QM9 reference config
    keeps the default value.
    """

    def __init__(
        self,
        n_atom_types: int,
        feature_names: list,
        interpolant_scheduler,
        n_charges: int,
        n_bond_types: int,
        charge_offset: int,
        n_vec_channels: int = 16,
        n_cp_feats: int = 0,
        n_hidden_scalars: int = 64,
        n_hidden_edge_feats: int = 64,
        n_recycles: int = 1,
        n_molecule_updates: int = 2,
        convs_per_update: int = 2,
        n_message_gvps: int = 3,
        n_update_gvps: int = 3,
        n_position_update_gvps: int = 3,
        separate_mol_updaters: bool = False,
        message_norm=100,
        update_edge_w_distance: bool = False,
        rbf_dmax: float = 20,
        rbf_dim: int = 16,
        stochasticity: float = 0.0,
        high_confidence_threshold: float = 0.0,
        cat_temperature: float = 0.05,
        time_embedding_dim: int = 1,
        a_token_dim: int = 0,
        c_token_dim: int = 0,
        e_token_dim: int = 0,
        self_conditioning: bool = False,
        scprop: float = 0.5,
        execution_backend: str = "eager",
        runtime_options: dict = None,
    ):
        super().__init__()
        self._init_runtime(execution_backend, runtime_options)

        self.n_atom_types = n_atom_types
        self.n_charges = n_charges
        self.n_bond_types = n_bond_types
        self.charge_offset = charge_offset
        self.n_vec_channels = n_vec_channels
        self.rbf_dmax = rbf_dmax
        self.rbf_dim = rbf_dim
        self.convs_per_update = convs_per_update
        self.n_recycles = n_recycles
        self.n_position_update_gvps = n_position_update_gvps
        self.separate_mol_updaters = separate_mol_updaters
        self.interpolant_scheduler = interpolant_scheduler
        self.feature_names = feature_names
        self.positions_idx = feature_names.index("positions")

        self.stochasticity = stochasticity
        self.high_confidence_threshold = high_confidence_threshold
        self.cat_temperature = cat_temperature
        self.time_embedding_dim = time_embedding_dim
        self.self_conditioning = self_conditioning
        self.scprop = scprop

        if n_vec_channels < 3:
            raise ValueError("n_vec_channels must be >= 3")

        self.n_cat_feats = {
            "atom_types": n_atom_types,
            "atom_charges": n_charges,
            "bond_types": n_bond_types,
        }

        self.mask_idxs = {
            "atom_types": n_atom_types,
            "atom_charges": n_charges,
            "bond_types": n_bond_types,
        }

        n_mask_feats = 1

        self.token_dims = {
            "atom_types": a_token_dim,
            "atom_charges": c_token_dim,
            "bond_types": e_token_dim,
        }
        self.token_embeddings = nn.LayerDict()
        for feat, token_dim in self.token_dims.items():
            if token_dim == 0:
                self.token_embeddings[feat] = None
            else:
                self.token_embeddings[feat] = nn.Embedding(
                    self.n_cat_feats[feat] + n_mask_feats, token_dim
                )

        for modality, token_dim in self.token_dims.items():
            if token_dim == 0:
                self.token_dims[modality] = self.n_cat_feats[modality] + n_mask_feats

        self.scalar_embedding = nn.Sequential(
            nn.Linear(
                self.token_dims["atom_types"]
                + self.token_dims["atom_charges"]
                + self.time_embedding_dim,
                n_hidden_scalars,
            ),
            nn.SiLU(),
            nn.Linear(n_hidden_scalars, n_hidden_scalars),
            nn.SiLU(),
            nn.LayerNorm(n_hidden_scalars),
        )

        self.edge_embedding = nn.Sequential(
            nn.Linear(self.token_dims["bond_types"], n_hidden_edge_feats),
            nn.SiLU(),
            nn.Linear(n_hidden_edge_feats, n_hidden_edge_feats),
            nn.SiLU(),
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
                    rbf_dmax=rbf_dmax,
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

        self.node_output_head = nn.Sequential(
            nn.Linear(n_hidden_scalars, n_hidden_scalars),
            nn.SiLU(),
            nn.Linear(n_hidden_scalars, n_atom_types + n_charges),
        )

        self.to_edge_logits = nn.Sequential(
            nn.Linear(n_hidden_edge_feats, n_hidden_edge_feats),
            nn.SiLU(),
            nn.Linear(n_hidden_edge_feats, n_bond_types),
        )

        if self.self_conditioning:
            self.self_conditioning_residual_layer = SelfConditioningResidualLayer(
                n_atom_types=n_atom_types,
                n_charges=n_charges,
                n_bond_types=n_bond_types,
                node_embedding_dim=n_hidden_scalars,
                edge_embedding_dim=n_hidden_edge_feats,
                rbf_dim=rbf_dim,
                rbf_dmax=rbf_dmax,
            )

    def precompute_distances(self, node_positions, src, dst):
        """Precompute normalized pairwise displacement and RBF distances."""
        x_diff = node_positions[src] - node_positions[dst]
        dij = norm_no_nan(x_diff, keepdims=True) + 1e-8
        x_diff = x_diff / dij
        d = _rbf(dij.squeeze(1), d_max=self.rbf_dmax, d_count=self.rbf_dim)
        return x_diff, d

    def _velocity(self, x_t, x_1, alpha_t, alpha_t_prime):
        return alpha_t_prime / (1 - alpha_t) * (x_1 - x_t)

    def forward(
        self,
        state: dict,
        t: paddle.Tensor,
        node_batch_idx: paddle.Tensor,
        upper_edge_mask: paddle.Tensor,
        apply_softmax: bool = False,
        remove_com: bool = False,
        prev_dst_dict: dict = None,
    ) -> dict:
        """Predict the endpoint of the trajectory given the state at time t."""
        return self._runtime_denoise_step(
            state,
            t,
            node_batch_idx,
            upper_edge_mask,
            self_condition_now=self._should_self_condition(t, prev_dst_dict),
            apply_softmax=apply_softmax,
            remove_com=remove_com,
            prev_dst_dict=prev_dst_dict,
        )

    def _should_self_condition(self, t: paddle.Tensor, prev_dst_dict: dict) -> bool:
        """Decide whether to run the self-conditioning pass.

        The decision draws a random trigger and reads host scalars, so it must
        stay outside the compiled runtime boundary.
        """
        if not self.self_conditioning or prev_dst_dict is not None:
            return False
        if self.training:
            return bool((paddle.rand([1]) > self.scprop).item())
        return bool((t == 0).all().item())

    @runtime_boundary("denoise_step")
    def _runtime_denoise_step(
        self,
        state: dict,
        t: paddle.Tensor,
        node_batch_idx: paddle.Tensor,
        upper_edge_mask: paddle.Tensor,
        self_condition_now: bool = False,
        apply_softmax: bool = False,
        remove_com: bool = False,
        prev_dst_dict: dict = None,
    ) -> dict:
        src, dst = state["src"], state["dst"]
        num_nodes = state["positions"].shape[0]

        node_scalar_features = []
        if self.token_embeddings["atom_types"] is None:
            node_scalar_features.append(state["atom_types"])
            node_scalar_features.append(state["atom_charges"])
        else:
            node_scalar_features.append(
                self.token_embeddings["atom_types"](state["atom_types"].argmax(axis=-1))
            )
            node_scalar_features.append(
                self.token_embeddings["atom_charges"](
                    state["atom_charges"].argmax(axis=-1)
                )
            )

        if self.time_embedding_dim == 1:
            node_scalar_features.append(t[node_batch_idx].unsqueeze(-1))
        else:
            t_emb = get_time_embedding(t, embedding_dim=self.time_embedding_dim)
            node_scalar_features.append(t_emb[node_batch_idx])

        node_scalar_features = paddle.concat(node_scalar_features, axis=-1)
        node_scalar_features = self.scalar_embedding(node_scalar_features)

        node_positions = state["positions"]

        node_vec_features = paddle.zeros(
            [num_nodes, self.n_vec_channels, 3], dtype=node_positions.dtype
        )

        if self.token_embeddings["bond_types"] is None:
            edge_features = state["bond_types"]
        else:
            edge_features = self.token_embeddings["bond_types"](
                state["bond_types"].argmax(axis=-1)
            )
        edge_features = self.edge_embedding(edge_features)

        if self_condition_now:
            with paddle.no_grad():
                prev_dst_dict = self.denoise_graph(
                    state,
                    node_scalar_features.clone(),
                    node_vec_features.clone(),
                    node_positions.clone(),
                    edge_features.clone(),
                    src,
                    dst,
                    num_nodes,
                    node_batch_idx,
                    upper_edge_mask,
                    apply_softmax=True,
                    remove_com=False,
                )

        if self.self_conditioning and prev_dst_dict is not None:
            (
                node_scalar_features,
                node_positions,
                node_vec_features,
                edge_features,
            ) = self.self_conditioning_residual_layer(
                src,
                dst,
                node_scalar_features,
                node_positions,
                node_vec_features,
                edge_features,
                prev_dst_dict,
                upper_edge_mask,
            )

        dst_dict = self.denoise_graph(
            state,
            node_scalar_features,
            node_vec_features,
            node_positions,
            edge_features,
            src,
            dst,
            num_nodes,
            node_batch_idx,
            upper_edge_mask,
            apply_softmax=apply_softmax,
            remove_com=remove_com,
        )
        return dst_dict

    def denoise_graph(
        self,
        state,
        node_scalar_features,
        node_vec_features,
        node_positions,
        edge_features,
        src,
        dst,
        num_nodes,
        node_batch_idx,
        upper_edge_mask,
        apply_softmax: bool = False,
        remove_com: bool = False,
    ):
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
            com = scatter_mean(node_positions, node_batch_idx, dim=0)
            node_positions = node_positions - com[node_batch_idx]

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

        positions_idx = self.positions_idx
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
        stochasticity: float = None,
        high_confidence_threshold: float = None,
    ) -> dict:
        """Integrate the molecule trajectories from the prior to the endpoint."""
        t = paddle.linspace(0, 1, n_timesteps)
        alpha_t = self.interpolant_scheduler.alpha_t(t)
        alpha_t_prime = self.interpolant_scheduler.alpha_t_prime(t)

        num_nodes = state["positions"].shape[0]
        # categorical features start from the mask prior
        state["atom_types"] = ctmc_masked_prior(
            num_nodes, self.n_cat_feats["atom_types"]
        )
        state["atom_charges"] = ctmc_masked_prior(
            num_nodes, self.n_cat_feats["atom_charges"]
        )
        num_edges = state["upper_edge_mask"].shape[0]
        state["bond_types"] = ctmc_masked_prior(
            num_edges, self.n_cat_feats["bond_types"]
        )

        dst_dict = None
        for s_idx in range(1, t.shape[0]):
            s_i = t[s_idx]
            t_i = t[s_idx - 1]
            alpha_t_i = alpha_t[s_idx - 1]
            alpha_s_i = alpha_t[s_idx]
            alpha_t_prime_i = alpha_t_prime[s_idx - 1]
            last_step = s_idx == t.shape[0] - 1

            state, dst_dict = self.step(
                state,
                s_i,
                t_i,
                alpha_t_i,
                alpha_s_i,
                alpha_t_prime_i,
                node_batch_idx,
                upper_edge_mask,
                stochasticity=stochasticity,
                high_confidence_threshold=high_confidence_threshold,
                last_step=last_step,
                prev_dst_dict=dst_dict,
            )

        state["atom_types_1"] = state["atom_types"]
        state["atom_charges_1"] = state["atom_charges"]
        state["bond_types_1"] = state["bond_types"]
        state["positions_1"] = state["positions"]
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
        stochasticity: float = None,
        high_confidence_threshold: float = None,
        last_step: bool = False,
        prev_dst_dict: dict = None,
    ) -> dict:
        eta = self.stochasticity if stochasticity is None else stochasticity
        hc_thresh = (
            self.high_confidence_threshold
            if high_confidence_threshold is None
            else high_confidence_threshold
        )

        output = self(
            state,
            t=paddle.full([state["batch_size"]], float(t_i)),
            node_batch_idx=node_batch_idx,
            upper_edge_mask=upper_edge_mask,
            apply_softmax=True,
            remove_com=True,
            prev_dst_dict=prev_dst_dict,
        )

        dt = s_i - t_i

        x_1 = output["positions"]
        x_t = state["positions"]
        alpha_x = alpha_t_i[self.positions_idx]
        alpha_x_prime = alpha_t_prime_i[self.positions_idx]
        vf = self._velocity(x_t, x_1, alpha_x, alpha_x_prime)
        state["positions"] = x_t + dt * vf

        for feat_idx, feat in enumerate(self.feature_names):
            if feat == "positions":
                continue

            if feat == "bond_types":
                data_state, data_output = state["bond_types"], output[feat]
                xt = data_state.argmax(-1)[upper_edge_mask]
            else:
                data_state, data_output = state[feat], output[feat]
                xt = data_state.argmax(-1)

            p_s_1 = paddle.nn.functional.softmax(
                paddle.log(data_output) / self.cat_temperature, axis=-1
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
                mask_index=self.mask_idxs[feat],
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
                state["bond_types"] = xt
            else:
                state[feat] = xt

        return state, output

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
            will_unmask = paddle.rand([xt.shape[0]]) < unmask_prob
            will_unmask = will_unmask & (xt == mask_index)

        if not last_step:
            will_mask = paddle.rand([xt.shape[0]]) < mask_prob
            will_mask = will_mask & (xt != mask_index)
            xt = paddle.where(will_mask, paddle.full_like(xt, mask_index), xt)

        xt = paddle.where(will_unmask, x1, xt)
        return paddle.nn.functional.one_hot(xt, num_classes=n_classes).astype("float32")

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

"""Wyckoff and Element Transformer: autoregressive sampling of Wyckoff
positions and elements.
"""
import math
from typing import Optional

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from ppmat.models.common.activation import ScaledSiLU as Swish
from ppmat.models.sgequidiff.sgequidiff_meta import ELEMENT_ENCODING_SIZE
from ppmat.models.sgequidiff.sgequidiff_meta import MAX_WYCKOFF_POSITIONS
from ppmat.models.sgequidiff.sgequidiff_meta import NUM_CRYSTALLOGRAPHIC_SPACE_GROUPS
from ppmat.models.sgequidiff.sgequidiff_meta import NUM_LATTICE_PARAMS
from ppmat.models.sgequidiff.sgequidiff_meta import lattice_parameter_ranges
from ppmat.models.sgequidiff.sgequidiff_meta import max_atoms_per_dataset
from ppmat.models.sgequidiff.shared import FourierLinear
from ppmat.models.sgequidiff.shared import SpaceGroupEncoder
from ppmat.models.sgequidiff.vocabs import EmbeddingTools
from ppmat.models.sgequidiff.wyckoff_geometry import WyckoffGeometry


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

    def __init__(self, embed_dim, num_heads):
        super().__init__(embed_dim=embed_dim, num_heads=num_heads)

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
        v = self.v_proj(value)

        q = q.reshape([batch_size, seq_q, self.num_heads, self.head_dim]).transpose(
            [0, 2, 1, 3]
        )
        k = k.reshape([batch_size, seq_k, self.num_heads, self.head_dim]).transpose(
            [0, 2, 1, 3]
        )
        v = v.reshape([batch_size, seq_k, self.num_heads, self.head_dim]).transpose(
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
        self.half_dim = half_dim
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
    def sample_and_log_prob(
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
        padded_wyckoff_probs = paddle.full(
            [n_crystals, max_allowed_atoms], fill_value=-1.0
        )
        padded_element_probs = paddle.full(
            [n_crystals, max_allowed_atoms], fill_value=-1.0
        )
        termination_probs = paddle.zeros([n_crystals])

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

            if (~sampled_stop_token).sum() > 0:
                sampled_wyckoff_probs = paddle.take_along_axis(
                    wyckoff_and_stop_probs[~sampled_stop_token].squeeze(1),
                    (1 + sampled_wyckoff_indices[:, None]),
                    axis=1,
                ).squeeze(-1)
            else:
                sampled_wyckoff_probs = paddle.zeros([0])

            termination_probs[incomplete_mask] = (
                sampled_stop_token.cast("float32") * wyckoff_and_stop_probs[:, 0, 0]
            )

            idxs_of_xtals_to_update = arange_n_crystals[incomplete_mask][
                ~sampled_stop_token
            ]
            if idxs_of_xtals_to_update.shape[0] > 0:
                padded_wyckoff_indices[
                    idxs_of_xtals_to_update, iteration
                ] = sampled_wyckoff_indices
                padded_wyckoff_probs[
                    idxs_of_xtals_to_update, iteration
                ] = sampled_wyckoff_probs
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
                sampled_element_probs = paddle.take_along_axis(
                    element_probs.squeeze(1),
                    sampled_element_indices[:, None],
                    axis=1,
                ).squeeze(-1)

                padded_element_indices[
                    idxs_of_xtals_to_update, iteration - 1
                ] = sampled_element_indices
                padded_element_probs[
                    idxs_of_xtals_to_update, iteration - 1
                ] = sampled_element_probs

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
        _element_probs = padded_element_probs[atom_mask]
        _wyckoff_probs = padded_wyckoff_probs[atom_mask]

        elements_log_prob = paddle.log(_element_probs + 1e-12)
        wyckoffs_log_prob = paddle.log(_wyckoff_probs + 1e-12)
        termination_log_prob = paddle.log(termination_probs + 1e-12)

        return (
            element_indices,
            wyckoff_indices,
            n_asu_atoms_per_xtal,
            elements_log_prob,
            wyckoffs_log_prob,
            termination_log_prob,
        )

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
        max_atoms: Optional[int] = None,
    ):
        """Log probabilities of the given elements and Wyckoff positions.

        Args:
            max_atoms: Static padding width for the teacher-forced forward.
                Passing the dataset-wide maximum keeps forward shapes static
                (no GPU->CPU sync, CINN-friendly). When omitted, the batch
                maximum is computed with ``.max()`` which forces a GPU->CPU
                sync every call. Callers must guarantee
                ``n_asu_atoms_per_xtal <= max_atoms`` (the bundled ASU
                datasets satisfy this by construction).
        """
        if max_atoms is None:
            max_atoms = int(n_asu_atoms_per_xtal.max())
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

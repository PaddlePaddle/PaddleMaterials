# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math

import paddle
import paddle.nn as nn

from ..common import get_index_embedding as get_pos_embedding


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TimestepEmbedder(nn.Layer):
    def __init__(self, hidden_dim, frequency_embedding_dim=256):
        super().__init__()
        self.frequency_embedding_dim = frequency_embedding_dim
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_dim, hidden_dim, bias_attr=True),
            nn.Silu(),
            nn.Linear(hidden_dim, hidden_dim, bias_attr=True),
        )

    def forward(self, t):
        return self.mlp(self._timestep_embedding(t, self.frequency_embedding_dim))

    def _timestep_embedding(self, t, dim, max_period=10000):
        half = dim // 2
        freqs = paddle.exp(
            -math.log(max_period)
            * paddle.arange(start=0, end=half, dtype=paddle.get_default_dtype())
            / half
        )
        args = t.astype(paddle.get_default_dtype())[:, None] * freqs[None]
        embedding = paddle.concat([paddle.cos(args), paddle.sin(args)], axis=-1)
        if dim % 2:
            embedding = paddle.concat(
                [embedding, paddle.zeros_like(embedding[:, :1])], axis=-1
            )
        return embedding


class _MLP(nn.Layer):
    def __init__(self, in_features, hidden_features, act_layer=None):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer() if act_layer else nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class FinalLayer(nn.Layer):
    def __init__(self, hidden_dim, out_dim):
        super().__init__()
        self.norm_final = nn.LayerNorm(
            hidden_dim, epsilon=1e-6, weight_attr=False, bias_attr=False
        )
        self.linear = nn.Linear(hidden_dim, out_dim, bias_attr=True)
        self.adaLN_modulation = nn.Sequential(
            nn.Silu(), nn.Linear(hidden_dim, 2 * hidden_dim, bias_attr=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, axis=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class DiTBlock(nn.Layer):
    def __init__(self, hidden_dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(
            hidden_dim, epsilon=1e-6, weight_attr=False, bias_attr=False
        )
        self.attn = nn.MultiHeadAttention(hidden_dim, num_heads, dropout=0.0)
        self.norm2 = nn.LayerNorm(
            hidden_dim, epsilon=1e-6, weight_attr=False, bias_attr=False
        )
        mlp_hidden_dim = int(hidden_dim * mlp_ratio)

        self.mlp = _MLP(
            in_features=hidden_dim,
            hidden_features=mlp_hidden_dim,
            act_layer=lambda: nn.GELU(approximate=True),
        )
        self.adaLN_modulation = nn.Sequential(
            nn.Silu(), nn.Linear(hidden_dim, 6 * hidden_dim, bias_attr=True)
        )

    def forward(self, x, c, mask):
        (
            shift_msa,
            scale_msa,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = self.adaLN_modulation(c).chunk(6, axis=1)

        norm_x = self.norm1(x)
        modulated_x = modulate(norm_x, shift_msa, scale_msa)

        # mask is already inverted from valid_mask: True means padding
        attn_mask = None
        if mask is not None:
            attn_mask = mask.unsqueeze([1, 2])
            attn_mask = paddle.cast(attn_mask, dtype="float32") * -1e9

        attn_out = self.attn(modulated_x, modulated_x, modulated_x, attn_mask=attn_mask)
        x = x + gate_msa.unsqueeze(1) * attn_out

        norm_x = self.norm2(x)
        modulated_x = modulate(norm_x, shift_mlp, scale_mlp)
        mlp_out = self.mlp(modulated_x)
        x = x + gate_mlp.unsqueeze(1) * mlp_out

        return x


class Chemeleon2DiT(nn.Layer):
    def __init__(
        self,
        input_dim=8,
        hidden_dim=768,
        num_heads=12,
        num_layers=12,
        mlp_ratio=4.0,
        condition_dim=None,
        learn_sigma=True,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.learn_sigma = learn_sigma
        self.latent_dim = input_dim

        self.x_embedder = nn.Linear(input_dim, hidden_dim, bias_attr=True)
        self.t_embedder = TimestepEmbedder(hidden_dim)

        if condition_dim is not None:
            self.y_embedder = nn.Linear(condition_dim, hidden_dim, bias_attr=True)
        else:
            self.y_embedder = None

        self.blocks = nn.LayerList(
            [
                DiTBlock(hidden_dim, num_heads, mlp_ratio=mlp_ratio)
                for _ in range(num_layers)
            ]
        )

        out_dim = input_dim * 2 if learn_sigma else input_dim
        self.final_layer = FinalLayer(hidden_dim, out_dim)

    def forward(self, x, t, mask=None, y=None, apply_mask=True):
        if mask is not None:
            token_indices = paddle.cumsum(mask.astype("int64"), axis=-1) - 1
            pos_emb = get_pos_embedding(token_indices, self.hidden_dim)
        else:
            pos_emb = 0

        x = self.x_embedder(x) + pos_emb
        c = self.t_embedder(t)

        if y is not None and self.y_embedder is not None:
            c = c + self.y_embedder(y)

        mask_inverted = paddle.logical_not(mask) if mask is not None else None
        for block in self.blocks:
            x = block(x, c, mask_inverted)

        x = self.final_layer(x, c)

        x = x.reshape([x.shape[0], 2 * x.shape[1], self.latent_dim])

        if mask is not None and apply_mask:
            x = x * mask.repeat(1, 2).unsqueeze(-1).astype(x.dtype)

        return x

    def forward_with_cfg(self, x, t, mask, y, cfg_scale):
        half_x = x[: x.shape[0] // 2]
        combined_x = paddle.concat([half_x, half_x], axis=0)
        model_out = self.forward(combined_x, t, mask, y)

        cond_eps, uncond_eps = paddle.split(model_out, 2, axis=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = paddle.concat([half_eps, half_eps], axis=0)
        return eps

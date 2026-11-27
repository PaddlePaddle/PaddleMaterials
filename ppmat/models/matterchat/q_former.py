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

"""Query-only Q-Former with cross-attention for MatterChat."""

import math
from dataclasses import dataclass
from typing import Optional

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from ppmat.models.matterchat.utils.activations import get_activation
from ppmat.models.matterchat.utils.outputs import SubscriptableOutput
from ppmat.models.matterchat.utils.weight_init import default_init_weights


@dataclass
class BertConfig:
    """Typed settings for the MatterChat Q-Former."""

    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    hidden_act: str = "gelu"
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-12
    max_position_embeddings: int = 512
    add_cross_attention: bool = False
    cross_attention_freq: int = 2
    encoder_width: int = 768
    query_length: int = 32


@dataclass
class QFormerOutput(SubscriptableOutput):
    last_hidden_state: Optional[paddle.Tensor] = None


class BertEmbeddings(nn.Layer):
    """Normalize the learnable query tokens before the encoder."""

    def __init__(self, config):
        super().__init__()
        self.LayerNorm = nn.LayerNorm(config.hidden_size, epsilon=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # Released checkpoints store this constant buffer under the embeddings prefix.
        self.register_buffer(
            "position_ids",
            paddle.arange(config.max_position_embeddings).reshape([1, -1]),
        )

    def forward(self, query_embeds):
        embeddings = self.LayerNorm(query_embeds)
        return self.dropout(embeddings)


class BertSelfAttention(nn.Layer):
    def __init__(self, config, is_cross_attention):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads)
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        encoder_width = (
            config.encoder_width if is_cross_attention else config.hidden_size
        )
        self.key = nn.Linear(encoder_width, self.all_head_size)
        self.value = nn.Linear(encoder_width, self.all_head_size)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.shape[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.reshape(new_x_shape)
        return x.transpose([0, 2, 1, 3])

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
    ):
        if encoder_hidden_states is not None:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))

        query_layer = self.transpose_for_scores(self.query(hidden_states))
        attention_scores = paddle.matmul(query_layer, key_layer.transpose([0, 1, 3, 2]))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        attention_probs = self.dropout(F.softmax(attention_scores, axis=-1))
        context_layer = paddle.matmul(attention_probs, value_layer)
        context_layer = context_layer.transpose([0, 2, 1, 3])
        new_context_layer_shape = context_layer.shape[:-2] + (self.all_head_size,)
        return context_layer.reshape(new_context_layer_shape)


class BertSelfOutput(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, epsilon=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return self.LayerNorm(hidden_states + input_tensor)


class BertAttention(nn.Layer):
    def __init__(self, config, is_cross_attention=False):
        super().__init__()
        self.self = BertSelfAttention(config, is_cross_attention)
        self.output = BertSelfOutput(config)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
    ):
        self_output = self.self(
            hidden_states,
            attention_mask,
            encoder_hidden_states,
            encoder_attention_mask,
        )
        return self.output(self_output, hidden_states)


class BertIntermediate(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = get_activation(config.hidden_act)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        return self.intermediate_act_fn(hidden_states)


class BertOutput(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, epsilon=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return self.LayerNorm(hidden_states + input_tensor)


class BertLayer(nn.Layer):
    def __init__(self, config, layer_num):
        super().__init__()
        self.attention = BertAttention(config)
        self.has_cross_attention = (
            config.add_cross_attention and layer_num % config.cross_attention_freq == 0
        )
        if self.has_cross_attention:
            self.crossattention = BertAttention(config, is_cross_attention=True)
        self.intermediate_query = BertIntermediate(config)
        self.output_query = BertOutput(config)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
    ):
        attention_output = self.attention(hidden_states, attention_mask)

        if self.has_cross_attention:
            if encoder_hidden_states is None:
                raise ValueError(
                    "encoder_hidden_states must be given for cross-attention layers"
                )
            attention_output = self.crossattention(
                attention_output,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
            )

        intermediate_output = self.intermediate_query(attention_output)
        return self.output_query(intermediate_output, attention_output)


class BertEncoder(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.layer = nn.LayerList(
            [BertLayer(config, i) for i in range(config.num_hidden_layers)]
        )

    def forward(
        self,
        hidden_states,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
    ):
        for layer_module in self.layer:
            hidden_states = layer_module(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
            )
        return hidden_states


class BertModel(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        default_init_weights(module, self.config.initializer_range)

    def forward(
        self, query_embeds, encoder_hidden_states, encoder_attention_mask=None
    ) -> QFormerOutput:
        embedding_output = self.embeddings(query_embeds)

        if encoder_attention_mask is not None:
            encoder_attention_mask = encoder_attention_mask[:, None, None, :]
            encoder_attention_mask = encoder_attention_mask.cast(paddle.float32)
            encoder_attention_mask = (1.0 - encoder_attention_mask) * paddle.finfo(
                paddle.float32
            ).min

        hidden_states = self.encoder(
            embedding_output,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
        )
        return QFormerOutput(last_hidden_state=hidden_states)


class QFormer(nn.Layer):
    """Query encoder keeping the published ``Qformer.bert`` weight prefix."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.bert = BertModel(config)

    def forward(self, query_embeds, encoder_hidden_states, encoder_attention_mask=None):
        return self.bert(
            query_embeds=query_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
        )

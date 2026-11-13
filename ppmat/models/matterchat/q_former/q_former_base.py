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

"""Q-Former (BERT variant with cross-attention) for MatterChat."""

import contextlib
import math
from dataclasses import dataclass
from typing import Optional
from typing import Tuple

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from ppmat.models.matterchat.material_encoder import MatterChatEncoder
from ppmat.models.matterchat.utils.activations import get_activation as _get_activation
from ppmat.models.matterchat.utils.weight_init import default_init_weights


@dataclass
class BertConfig:
    """Typed settings for the MatterChat Q-Former."""

    vocab_size: int = 30522
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    hidden_act: str = "gelu"
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    max_position_embeddings: int = 512
    type_vocab_size: int = 2
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-12
    pad_token_id: int = 0
    chunk_size_feed_forward: int = 0
    add_cross_attention: bool = False
    cross_attention_freq: int = 2
    encoder_width: int = 768
    query_length: int = 32
    output_attentions: bool = False
    output_hidden_states: bool = False
    use_return_dict: bool = True
    gradient_checkpointing: bool = False


class _SubscriptableOutput:
    """Base class for model outputs that support tuple-like indexing."""

    def __getitem__(self, index):
        fields = [f for f in self.__dataclass_fields__ if not f.startswith("_")]
        return getattr(self, fields[index])

    def __len__(self):
        return len([f for f in self.__dataclass_fields__ if not f.startswith("_")])


@dataclass
class BaseModelOutputWithPastAndCrossAttentions(_SubscriptableOutput):
    last_hidden_state: paddle.Tensor = None
    past_key_values: Optional[Tuple] = None
    hidden_states: Optional[Tuple] = None
    attentions: Optional[Tuple] = None
    cross_attentions: Optional[Tuple] = None


@dataclass
class BaseModelOutputWithPoolingAndCrossAttentions(_SubscriptableOutput):
    last_hidden_state: paddle.Tensor = None
    pooler_output: Optional[paddle.Tensor] = None
    past_key_values: Optional[Tuple] = None
    hidden_states: Optional[Tuple] = None
    attentions: Optional[Tuple] = None
    cross_attentions: Optional[Tuple] = None


@dataclass
class CausalLMOutputWithCrossAttentions(_SubscriptableOutput):
    loss: Optional[paddle.Tensor] = None
    logits: paddle.Tensor = None
    past_key_values: Optional[Tuple] = None
    hidden_states: Optional[Tuple] = None
    attentions: Optional[Tuple] = None
    cross_attentions: Optional[Tuple] = None


@dataclass
class MaskedLMOutput:
    loss: Optional[paddle.Tensor] = None
    logits: paddle.Tensor = None
    hidden_states: Optional[Tuple] = None
    attentions: Optional[Tuple] = None


def apply_chunking_to_forward(forward_fn, chunk_size, chunk_dim, *input_tensors):
    """Apply forward function. Simplified from transformers version."""
    return forward_fn(*input_tensors)


def find_pruneable_heads_and_indices(heads, n_heads, head_size, already_pruned_heads):
    """Find pruneable heads and their indices."""
    mask = paddle.ones([n_heads, head_size])
    heads = set(heads) - already_pruned_heads
    for head in heads:
        mask[head] = 0
    index = paddle.nonzero(mask.reshape([-1]) == 1).squeeze(-1)
    index = index[: len(heads) * head_size]
    return heads, index


def prune_linear_layer(layer, index, dim=0):
    """Prune linear layer (stub)."""
    return layer


class BertEmbeddings(nn.Layer):
    """Construct the embeddings from word and position embeddings."""

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            padding_idx=config.pad_token_id,
        )
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings,
            config.hidden_size,
        )
        self.LayerNorm = nn.LayerNorm(config.hidden_size, epsilon=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.register_buffer(
            "position_ids",
            paddle.arange(config.max_position_embeddings).reshape([1, -1]),
        )
        self.position_embedding_type = getattr(
            config, "position_embedding_type", "absolute"
        )
        self.config = config

    def forward(
        self,
        input_ids=None,
        position_ids=None,
        query_embeds=None,
        past_key_values_length=0,
    ):
        if input_ids is not None:
            seq_length = input_ids.shape[1]
        else:
            seq_length = 0

        if position_ids is None:
            position_ids = self.position_ids[
                :, past_key_values_length : seq_length + past_key_values_length
            ].clone()

        if input_ids is not None:
            embeddings = self.word_embeddings(input_ids)
            if self.position_embedding_type == "absolute":
                position_embeddings = self.position_embeddings(position_ids)
                embeddings = embeddings + position_embeddings
            if query_embeds is not None:
                embeddings = paddle.concat((query_embeds, embeddings), axis=1)
        else:
            embeddings = query_embeds

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertSelfAttention(nn.Layer):
    def __init__(self, config, is_cross_attention):
        super().__init__()
        self.config = config

        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(
            config, "embedding_size"
        ):
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads)
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)

        if is_cross_attention:
            self.key = nn.Linear(config.encoder_width, self.all_head_size)
            self.value = nn.Linear(config.encoder_width, self.all_head_size)
        else:
            self.key = nn.Linear(config.hidden_size, self.all_head_size)
            self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = getattr(
            config, "position_embedding_type", "absolute"
        )
        if (
            self.position_embedding_type == "relative_key"
            or self.position_embedding_type == "relative_key_query"
        ):
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(
                2 * config.max_position_embeddings - 1, self.attention_head_size
            )

        self.save_attention = False

    def save_attn_gradients(self, attn_gradients):
        self.attn_gradients = attn_gradients

    def get_attn_gradients(self):
        return self.attn_gradients

    def save_attention_map(self, attention_map):
        self.attention_map = attention_map

    def get_attention_map(self):
        return self.attention_map

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
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = paddle.concat([past_key_value[0], key_layer], axis=2)
            value_layer = paddle.concat([past_key_value[1], value_layer], axis=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))

        mixed_query_layer = self.query(hidden_states)
        query_layer = self.transpose_for_scores(mixed_query_layer)
        past_key_value = (key_layer, value_layer)

        attention_scores = paddle.matmul(query_layer, key_layer.transpose([0, 1, 3, 2]))

        if (
            self.position_embedding_type == "relative_key"
            or self.position_embedding_type == "relative_key_query"
        ):
            seq_length = hidden_states.shape[1]
            position_ids_l = paddle.arange(seq_length, dtype=paddle.int64).reshape(
                [-1, 1]
            )
            position_ids_r = paddle.arange(seq_length, dtype=paddle.int64).reshape(
                [1, -1]
            )
            distance = position_ids_l - position_ids_r
            positional_embedding = self.distance_embedding(
                distance + self.max_position_embeddings - 1
            )
            positional_embedding = positional_embedding.cast(query_layer.dtype)

            if self.position_embedding_type == "relative_key":
                relative_position_scores = paddle.einsum(
                    "bhld,lrd->bhlr", query_layer, positional_embedding
                )
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = paddle.einsum(
                    "bhld,lrd->bhlr", query_layer, positional_embedding
                )
                relative_position_scores_key = paddle.einsum(
                    "bhrd,lrd->bhlr", key_layer, positional_embedding
                )
                attention_scores = (
                    attention_scores
                    + relative_position_scores_query
                    + relative_position_scores_key
                )

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        attention_probs = F.softmax(attention_scores, axis=-1)

        if is_cross_attention and self.save_attention:
            self.save_attention_map(attention_probs)

        attention_probs_dropped = self.dropout(attention_probs)

        if head_mask is not None:
            attention_probs_dropped = attention_probs_dropped * head_mask

        context_layer = paddle.matmul(attention_probs_dropped, value_layer)
        context_layer = context_layer.transpose([0, 2, 1, 3])
        new_context_layer_shape = context_layer.shape[:-2] + (self.all_head_size,)
        context_layer = context_layer.reshape(new_context_layer_shape)

        outputs = (
            (context_layer, attention_probs) if output_attentions else (context_layer,)
        )
        outputs = outputs + (past_key_value,)
        return outputs


class BertSelfOutput(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, epsilon=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Layer):
    def __init__(self, config, is_cross_attention=False):
        super().__init__()
        self.self = BertSelfAttention(config, is_cross_attention)
        self.output = BertSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads,
            self.self.num_attention_heads,
            self.self.attention_head_size,
            self.pruned_heads,
        )
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = (
            self.self.attention_head_size * self.self.num_attention_heads
        )
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]
        return outputs


class BertIntermediate(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = _get_activation(config.hidden_act)
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, epsilon=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(nn.Layer):
    def __init__(self, config, layer_num):
        super().__init__()
        self.config = config
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = BertAttention(config)
        self.layer_num = layer_num
        if (
            self.config.add_cross_attention
            and layer_num % self.config.cross_attention_freq == 0
        ):
            self.crossattention = BertAttention(
                config, is_cross_attention=self.config.add_cross_attention
            )
            self.has_cross_attention = True
        else:
            self.has_cross_attention = False

        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)
        self.intermediate_query = BertIntermediate(config)
        self.output_query = BertOutput(config)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
        query_length=0,
    ):
        self_attn_past_key_value = (
            past_key_value[:2] if past_key_value is not None else None
        )
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:-1]
        present_key_value = self_attention_outputs[-1]

        if query_length > 0:
            query_attention_output = attention_output[:, :query_length, :]

            if self.has_cross_attention:
                assert (
                    encoder_hidden_states is not None
                ), "encoder_hidden_states must be given for cross-attention layers"
                cross_attention_outputs = self.crossattention(
                    query_attention_output,
                    attention_mask,
                    head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    output_attentions=output_attentions,
                )
                query_attention_output = cross_attention_outputs[0]
                outputs = outputs + cross_attention_outputs[1:-1]

            layer_output = apply_chunking_to_forward(
                self.feed_forward_chunk_query,
                self.chunk_size_feed_forward,
                self.seq_len_dim,
                query_attention_output,
            )
            if attention_output.shape[1] > query_length:
                layer_output_text = apply_chunking_to_forward(
                    self.feed_forward_chunk,
                    self.chunk_size_feed_forward,
                    self.seq_len_dim,
                    attention_output[:, query_length:, :],
                )
                layer_output = paddle.concat([layer_output, layer_output_text], axis=1)
        else:
            layer_output = apply_chunking_to_forward(
                self.feed_forward_chunk,
                self.chunk_size_feed_forward,
                self.seq_len_dim,
                attention_output,
            )
        outputs = (layer_output,) + outputs
        outputs = outputs + (present_key_value,)
        return outputs

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output

    def feed_forward_chunk_query(self, attention_output):
        intermediate_output = self.intermediate_query(attention_output)
        layer_output = self.output_query(intermediate_output, attention_output)
        return layer_output


class BertEncoder(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.LayerList(
            [BertLayer(config, i) for i in range(config.num_hidden_layers)]
        )

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
        query_length=0,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = (
            () if output_attentions and self.config.add_cross_attention else None
        )
        next_decoder_cache = () if use_cache else None

        for i in range(self.config.num_hidden_layers):
            layer_module = self.layer[i]
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            layer_outputs = layer_module(
                hidden_states,
                attention_mask,
                layer_head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                past_key_value,
                output_attentions,
                query_length,
            )

            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )


class BertPooler(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertPredictionHeadTransform(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = _get_activation(config.hidden_act)
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = nn.LayerNorm(config.hidden_size, epsilon=config.layer_norm_eps)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertLMPredictionHead(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.transform = BertPredictionHeadTransform(config)
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias_attr=False)
        self.bias = self.create_parameter(
            shape=[config.vocab_size],
            is_bias=True,
            default_initializer=nn.initializer.Constant(0.0),
        )
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


class BertOnlyMLMHead(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.predictions = BertLMPredictionHead(config)

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


class BertPreTrainedModel(nn.Layer):
    config_class = BertConfig
    base_model_prefix = "bert"

    def __init__(self, config):
        super().__init__()
        self.config = config

    def _init_weights(self, module):
        default_init_weights(module, self.config.initializer_range)

    def init_weights(self):
        self.apply(self._init_weights)

    def get_head_mask(self, head_mask, num_hidden_layers):
        if head_mask is not None:
            if isinstance(head_mask, paddle.Tensor):
                if len(head_mask.shape) == 1:
                    head_mask = (
                        head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                    )
                    head_mask = head_mask.expand([num_hidden_layers, -1, -1, -1, -1])
            return head_mask
        return [None] * num_hidden_layers

    def invert_attention_mask(self, encoder_attention_mask):
        if isinstance(encoder_attention_mask, paddle.Tensor):
            if len(encoder_attention_mask.shape) == 2:
                encoder_extended_attention_mask = encoder_attention_mask[
                    :, None, None, :
                ]
            elif len(encoder_attention_mask.shape) == 3:
                encoder_extended_attention_mask = encoder_attention_mask[:, None, :, :]
            else:
                encoder_extended_attention_mask = encoder_attention_mask
            encoder_extended_attention_mask = encoder_extended_attention_mask.cast(
                paddle.float32
            )
            encoder_extended_attention_mask = (
                1.0 - encoder_extended_attention_mask
            ) * paddle.finfo(paddle.float32).min
            return encoder_extended_attention_mask
        return encoder_attention_mask


class BertModel(BertPreTrainedModel):
    def __init__(self, config, add_pooling_layer=False):
        super().__init__(config)
        self.config = config
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config) if add_pooling_layer else None
        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def get_extended_attention_mask(
        self,
        attention_mask,
        input_shape,
        is_decoder,
        has_query=False,
    ):
        if len(attention_mask.shape) == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif len(attention_mask.shape) == 2:
            if is_decoder:
                batch_size, seq_length = input_shape
                seq_ids = paddle.arange(seq_length)
                causal_mask = (
                    seq_ids[None, None, :].expand([batch_size, seq_length, seq_length])
                    <= seq_ids[None, :, None]
                )
                causal_mask = causal_mask.cast(attention_mask.dtype)

                if causal_mask.shape[1] < attention_mask.shape[1]:
                    prefix_seq_len = attention_mask.shape[1] - causal_mask.shape[1]
                    if has_query:
                        causal_mask = paddle.concat(
                            [
                                paddle.zeros(
                                    [batch_size, prefix_seq_len, seq_length],
                                    dtype=causal_mask.dtype,
                                ),
                                causal_mask,
                            ],
                            axis=1,
                        )
                    causal_mask = paddle.concat(
                        [
                            paddle.ones(
                                [batch_size, causal_mask.shape[1], prefix_seq_len],
                                dtype=causal_mask.dtype,
                            ),
                            causal_mask,
                        ],
                        axis=-1,
                    )
                extended_attention_mask = (
                    causal_mask[:, None, :, :] * attention_mask[:, None, None, :]
                )
            else:
                extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(
                "Wrong shape for attention_mask (shape {})".format(attention_mask.shape)
            )

        extended_attention_mask = extended_attention_mask.cast(paddle.float32)
        extended_attention_mask = (1.0 - extended_attention_mask) * paddle.finfo(
            paddle.float32
        ).min
        return extended_attention_mask

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        head_mask=None,
        query_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        is_decoder=False,
    ):
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if input_ids is None:
            assert (
                query_embeds is not None
            ), "You have to specify query_embeds when input_ids is None"

        past_key_values_length = (
            past_key_values[0][0].shape[2] - self.config.query_length
            if past_key_values is not None
            else 0
        )
        query_length = query_embeds.shape[1] if query_embeds is not None else 0

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            query_embeds=query_embeds,
            past_key_values_length=past_key_values_length,
        )

        input_shape = embedding_output.shape[:-1]
        batch_size, seq_length = input_shape

        if attention_mask is None:
            attention_mask = paddle.ones(
                [batch_size, seq_length + past_key_values_length]
            )

        if is_decoder:
            extended_attention_mask = self.get_extended_attention_mask(
                attention_mask,
                input_ids.shape if input_ids is not None else (batch_size, seq_length),
                is_decoder,
                has_query=(query_embeds is not None),
            )
        else:
            extended_attention_mask = self.get_extended_attention_mask(
                attention_mask, input_shape, is_decoder
            )

        if encoder_hidden_states is not None:
            if isinstance(encoder_hidden_states, list):
                encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states[
                    0
                ].shape
            else:
                (
                    encoder_batch_size,
                    encoder_sequence_length,
                    _,
                ) = encoder_hidden_states.shape
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)

            if isinstance(encoder_attention_mask, list):
                encoder_extended_attention_mask = [
                    self.invert_attention_mask(mask) for mask in encoder_attention_mask
                ]
            elif encoder_attention_mask is None:
                encoder_attention_mask = paddle.ones(encoder_hidden_shape)
                encoder_extended_attention_mask = self.invert_attention_mask(
                    encoder_attention_mask
                )
            else:
                encoder_extended_attention_mask = self.invert_attention_mask(
                    encoder_attention_mask
                )
        else:
            encoder_extended_attention_mask = None

        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            query_length=query_length,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = (
            self.pooler(sequence_output) if self.pooler is not None else None
        )

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values if return_dict else None,
            hidden_states=encoder_outputs.hidden_states if return_dict else None,
            attentions=encoder_outputs.attentions if return_dict else None,
            cross_attentions=encoder_outputs.cross_attentions if return_dict else None,
        )


class BertLMHeadModel(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config, add_pooling_layer=False)
        self.cls = BertOnlyMLMHead(config)
        self.init_weights()

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        head_mask=None,
        query_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        past_key_values=None,
        use_cache=True,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        return_logits=False,
        is_decoder=True,
        reduction="mean",
    ):
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        if labels is not None:
            use_cache = False
        if past_key_values is not None:
            query_embeds = None

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            query_embeds=query_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            is_decoder=is_decoder,
        )

        sequence_output = outputs[0]
        if query_embeds is not None:
            sequence_output = outputs[0][:, query_embeds.shape[1] :, :]

        prediction_scores = self.cls(sequence_output)

        if return_logits:
            return prediction_scores[:, :-1, :]

        lm_loss = None
        if labels is not None:
            shifted_prediction_scores = prediction_scores[:, :-1, :]
            labels = labels[:, 1:]
            loss_fct = nn.CrossEntropyLoss(reduction=reduction)
            lm_loss = loss_fct(
                shifted_prediction_scores.reshape([-1, self.config.vocab_size]),
                labels.reshape([-1]),
            )
            if reduction == "none":
                lm_loss = lm_loss.reshape([prediction_scores.shape[0], -1]).sum(1)

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((lm_loss,) + output) if lm_loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=lm_loss,
            logits=prediction_scores,
            past_key_values=outputs.past_key_values
            if hasattr(outputs, "past_key_values")
            else None,
            hidden_states=outputs.hidden_states
            if hasattr(outputs, "hidden_states")
            else None,
            attentions=outputs.attentions if hasattr(outputs, "attentions") else None,
            cross_attentions=outputs.cross_attentions
            if hasattr(outputs, "cross_attentions")
            else None,
        )

    def prepare_inputs_for_generation(
        self, input_ids, query_embeds, past=None, attention_mask=None, **model_kwargs
    ):
        if attention_mask is None:
            attention_mask = paddle.ones(input_ids.shape)
        query_mask = paddle.ones(query_embeds.shape[:-1])
        attention_mask = paddle.concat([query_mask, attention_mask], axis=-1)

        if past is not None:
            input_ids = input_ids[:, -1:]

        return {
            "input_ids": input_ids,
            "query_embeds": query_embeds,
            "attention_mask": attention_mask,
            "past_key_values": past,
            "encoder_hidden_states": model_kwargs.get("encoder_hidden_states", None),
            "encoder_attention_mask": model_kwargs.get("encoder_attention_mask", None),
            "is_decoder": True,
        }

    def _reorder_cache(self, past, beam_idx):
        reordered_past = ()
        for layer_past in past:
            reordered_past += (
                tuple(
                    paddle.index_sample(past_state, beam_idx)
                    for past_state in layer_past
                ),
            )
        return reordered_past


class BertForMaskedLM(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config, add_pooling_layer=False)
        self.cls = BertOnlyMLMHead(config)
        self.init_weights()

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        head_mask=None,
        query_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        return_logits=False,
        is_decoder=False,
    ):
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            query_embeds=query_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            is_decoder=is_decoder,
        )

        if query_embeds is not None:
            sequence_output = outputs[0][:, query_embeds.shape[1] :, :]
        else:
            sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)

        if return_logits:
            return prediction_scores

        masked_lm_loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            masked_lm_loss = loss_fct(
                prediction_scores.reshape([-1, self.config.vocab_size]),
                labels.reshape([-1]),
            )

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return (
                ((masked_lm_loss,) + output) if masked_lm_loss is not None else output
            )

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states
            if hasattr(outputs, "hidden_states")
            else None,
            attentions=outputs.attentions if hasattr(outputs, "attentions") else None,
        )


class LayerNorm(nn.LayerNorm):
    """Subclass of paddle LayerNorm to support mixed precision."""

    def forward(self, x):
        orig_dtype = x.dtype
        x = super().forward(x.cast(paddle.float32))
        return x.cast(orig_dtype)


class Blip2Base(nn.Layer):
    """Base class for Blip2 models."""

    @classmethod
    def init_Qformer(cls, num_query_token, vision_width, qformer_config=None):
        settings = {} if qformer_config is None else dict(qformer_config)
        derived = {
            "encoder_width": vision_width,
            "query_length": num_query_token,
            "add_cross_attention": True,
        }
        for name, value in derived.items():
            if name in settings and settings[name] != value:
                raise ValueError(f"{name} must match the model-derived value {value}.")
        config = BertConfig(**dict(settings, **derived))
        if config.num_hidden_layers < 1 or config.cross_attention_freq < 1:
            raise ValueError(
                "Q-Former depth and cross-attention frequency must be positive."
            )
        if config.num_attention_heads < 1 or config.hidden_size < 1:
            raise ValueError(
                "Q-Former hidden size and attention heads must be positive."
            )
        if config.hidden_size % config.num_attention_heads:
            raise ValueError(
                "Q-Former hidden size must be divisible by attention heads."
            )

        Qformer = BertLMHeadModel(config)

        query_tokens = paddle.create_parameter(
            shape=[1, num_query_token, config.hidden_size],
            dtype=paddle.float32,
            default_initializer=nn.initializer.Normal(
                mean=0.0, std=config.initializer_range
            ),
        )
        return Qformer, query_tokens

    def init_material_encoder(self, config=None):
        """Initialize CHGNet material encoder without pretrained weights."""
        return MatterChatEncoder(config)

    def maybe_autocast(self, dtype=paddle.float16):
        """Context manager for mixed precision."""
        if not paddle.is_compiled_with_cuda():
            return contextlib.nullcontext()
        return paddle.amp.auto_cast(dtype=dtype)

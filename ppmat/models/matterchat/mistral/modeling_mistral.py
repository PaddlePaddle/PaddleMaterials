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

"""PaddlePaddle Mistral-7B model implementation."""

import math
from dataclasses import dataclass
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.nn.attention import SDPBackend
from paddle.nn.attention import sdpa_kernel

from ppmat.models.common.runtime import RuntimeMixin
from ppmat.models.common.runtime import runtime_boundary
from ppmat.models.matterchat.mistral.configuration_mistral import MistralConfig
from ppmat.models.matterchat.utils.activations import get_activation
from ppmat.models.matterchat.utils.weight_init import default_init_weights
from ppmat.utils import logger

# Retain the existing beam search's GNMT length normalization.
_BEAM_LENGTH_PENALTY_OFFSET = 5.0


class _SubscriptableOutput:
    """Base class for model outputs that support tuple-like indexing."""

    def __getitem__(self, index):
        fields = [f for f in self.__dataclass_fields__ if not f.startswith("_")]
        return getattr(self, fields[index])

    def __len__(self):
        return len([f for f in self.__dataclass_fields__ if not f.startswith("_")])


@dataclass
class BaseModelOutputWithPast(_SubscriptableOutput):
    last_hidden_state: paddle.Tensor = None
    past_key_values: Optional[Tuple] = None
    hidden_states: Optional[Tuple] = None
    attentions: Optional[Tuple] = None


@dataclass
class CausalLMOutputWithPast(_SubscriptableOutput):
    loss: Optional[paddle.Tensor] = None
    logits: paddle.Tensor = None
    past_key_values: Optional[Tuple] = None
    hidden_states: Optional[Tuple] = None
    attentions: Optional[Tuple] = None


class DynamicCache:
    """Simple dynamic KV cache for autoregressive generation."""

    def __init__(self):
        self._seen_tokens = 0
        self.key_cache: List[paddle.Tensor] = []
        self.value_cache: List[paddle.Tensor] = []

    def __getitem__(self, layer_idx):
        if layer_idx < len(self.key_cache):
            return (self.key_cache[layer_idx], self.value_cache[layer_idx])
        raise IndexError(f"layer_idx {layer_idx} out of range")

    def update(
        self,
        key_states: paddle.Tensor,
        value_states: paddle.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[dict] = None,
    ):
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]

        if len(self.key_cache) <= layer_idx:
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
        else:
            self.key_cache[layer_idx] = paddle.concat(
                [self.key_cache[layer_idx], key_states], axis=-2
            )
            self.value_cache[layer_idx] = paddle.concat(
                [self.value_cache[layer_idx], value_states], axis=-2
            )

        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def get_seq_length(self, layer_idx=0):
        if len(self.key_cache) <= layer_idx:
            return 0
        return self.key_cache[layer_idx].shape[-2]

    def get_max_length(self):
        return None

    @classmethod
    def from_legacy_cache(cls, past_key_values):
        cache = cls()
        if past_key_values is not None:
            for layer_past in past_key_values:
                cache.key_cache.append(layer_past[0])
                cache.value_cache.append(layer_past[1])
                cache._seen_tokens = layer_past[0].shape[-2]
        return cache

    def to_legacy_cache(self):
        legacy_cache = ()
        for layer_idx in range(len(self.key_cache)):
            legacy_cache += ((self.key_cache[layer_idx], self.value_cache[layer_idx]),)
        return legacy_cache


class MistralRMSNorm(nn.Layer):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = self.create_parameter(
            shape=[hidden_size],
            default_initializer=nn.initializer.Constant(1.0),
        )
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.cast(paddle.float32)
        variance = paddle.pow(hidden_states, 2).mean(axis=-1, keepdim=True)
        hidden_states = hidden_states * paddle.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.cast(input_dtype)


class MistralRotaryEmbedding(nn.Layer):
    def __init__(self, dim, max_position_embeddings=2048, base=10000):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (
            self.base
            ** (
                paddle.arange(0, self.dim, 2, dtype=paddle.int64).cast(paddle.float32)
                / self.dim
            )
        )
        self.register_buffer("inv_freq", inv_freq, persistable=False)

    @paddle.no_grad()
    def forward(self, x, position_ids):
        inv_freq_expanded = (
            self.inv_freq[None, :, None]
            .cast(paddle.float32)
            .expand([position_ids.shape[0], -1, 1])
        )
        position_ids_expanded = position_ids[:, None, :].cast(paddle.float32)
        freqs = (
            inv_freq_expanded.cast(paddle.float32)
            @ position_ids_expanded.cast(paddle.float32)
        ).transpose([0, 2, 1])
        emb = paddle.concat((freqs, freqs), axis=-1)
        cos = emb.cos()
        sin = emb.sin()
        return cos.cast(x.dtype), sin.cast(x.dtype)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return paddle.concat((-x2, x1), axis=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    cos = cos.unsqueeze(axis=unsqueeze_dim)
    sin = sin.unsqueeze(axis=unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class MistralMLP(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(
            self.hidden_size, self.intermediate_size, bias_attr=False
        )
        self.up_proj = nn.Linear(
            self.hidden_size, self.intermediate_size, bias_attr=False
        )
        self.down_proj = nn.Linear(
            self.intermediate_size, self.hidden_size, bias_attr=False
        )
        self.act_fn = get_activation(config.hidden_act)

    def forward(self, hidden_state):
        return self.down_proj(
            self.act_fn(self.gate_proj(hidden_state)) * self.up_proj(hidden_state)
        )


def repeat_kv(hidden_states: paddle.Tensor, n_rep: int) -> paddle.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        [batch, num_key_value_heads, n_rep, slen, head_dim]
    )
    return hidden_states.reshape([batch, num_key_value_heads * n_rep, slen, head_dim])


class MistralAttention(nn.Layer):
    def __init__(self, config: MistralConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning(
                f"Instantiating {self.__class__.__name__} without passing a "
                "`layer_idx` is not recommended."
            )

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                "hidden_size must be divisible by num_heads "
                f"(got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias_attr=False
        )
        self.k_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias_attr=False
        )
        self.v_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias_attr=False
        )
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias_attr=False)

        self.rotary_emb = MistralRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )

    def forward(
        self,
        hidden_states: paddle.Tensor,
        attention_mask: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.Tensor] = None,
        past_key_value: Optional[DynamicCache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[paddle.Tensor] = None,
    ) -> Tuple[paddle.Tensor, Optional[paddle.Tensor], Optional[Tuple[paddle.Tensor]]]:
        bsz, q_len, _ = hidden_states.shape

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.reshape(
            [bsz, q_len, self.num_heads, self.head_dim]
        ).transpose([0, 2, 1, 3])
        key_states = key_states.reshape(
            [bsz, q_len, self.num_key_value_heads, self.head_dim]
        ).transpose([0, 2, 1, 3])
        value_states = value_states.reshape(
            [bsz, q_len, self.num_key_value_heads, self.head_dim]
        ).transpose([0, 2, 1, 3])

        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        if attention_mask is not None:
            attention_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_output, attn_weights = self._compute_attention(
            query_states,
            key_states,
            value_states,
            attention_mask,
            output_attentions,
        )

        if attn_output.shape != [bsz, self.num_heads, q_len, self.head_dim]:
            raise ValueError(
                "`attn_output` should be of size "
                f"{[bsz, self.num_heads, q_len, self.head_dim]}, but is"
                f" {attn_output.shape}"
            )

        attn_output = attn_output.transpose([0, 2, 1, 3])
        attn_output = attn_output.reshape([bsz, q_len, -1])
        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights, past_key_value

    def _compute_attention(
        self, query_states, key_states, value_states, attention_mask, output_attentions
    ):
        attn_weights = paddle.matmul(
            query_states, key_states.transpose([0, 1, 3, 2])
        ) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        attn_weights = F.softmax(attn_weights, axis=-1, dtype=paddle.float32).cast(
            query_states.dtype
        )
        attn_weights = F.dropout(
            attn_weights, p=self.attention_dropout, training=self.training
        )
        attn_output = paddle.matmul(attn_weights, value_states)
        return attn_output, attn_weights if output_attentions else None


class MistralSdpaAttention(MistralAttention):
    """Mistral attention using paddle.nn.functional.scaled_dot_product_attention."""

    def _compute_attention(
        self, query_states, key_states, value_states, attention_mask, output_attentions
    ):
        if output_attentions:
            return super()._compute_attention(
                query_states,
                key_states,
                value_states,
                attention_mask,
                output_attentions,
            )
        # Avoid memory-efficient FP32 kernels that reject valid GPU input shapes.
        with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.MATH]):
            attn_output = F.scaled_dot_product_attention(
                query_states.transpose([0, 2, 1, 3]),
                key_states.transpose([0, 2, 1, 3]),
                value_states.transpose([0, 2, 1, 3]),
                attn_mask=attention_mask,
                dropout_p=self.attention_dropout if self.training else 0.0,
                is_causal=False,
                training=self.training,
            )
        return attn_output.transpose([0, 2, 1, 3]), None


MISTRAL_ATTENTION_CLASSES = {
    "eager": MistralAttention,
    "sdpa": MistralSdpaAttention,
}


class MistralDecoderLayer(nn.Layer):
    def __init__(self, config: MistralConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        attn_cls = MISTRAL_ATTENTION_CLASSES[config.attn_implementation]
        self.self_attn = attn_cls(config=config, layer_idx=layer_idx)
        self.mlp = MistralMLP(config)
        self.input_layernorm = MistralRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = MistralRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        hidden_states: paddle.Tensor,
        attention_mask: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.Tensor] = None,
        past_key_value: Optional[DynamicCache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[paddle.Tensor] = None,
        **kwargs,
    ) -> Tuple[paddle.Tensor, Optional[Tuple[paddle.Tensor, paddle.Tensor]]]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        if use_cache:
            outputs += (present_key_value,)

        return outputs


class MistralPreTrainedModel(nn.Layer):
    config_class = MistralConfig
    base_model_prefix = "model"

    def __init__(self, config: MistralConfig):
        super().__init__()
        self.config = config

    def _init_weights(self, module):
        default_init_weights(module, self.config.initializer_range)

    def init_weights(self):
        self.apply(self._init_weights)


def _apply_repetition_penalty(logits, gen_ids, penalty):
    """Vectorized repetition penalty following HF GenerationMixin."""
    if gen_ids is None or len(gen_ids) == 0:
        return logits
    unique_ids = paddle.unique(gen_ids)
    gathered = logits[:, unique_ids]
    penalized = paddle.where(gathered > 0, gathered / penalty, gathered * penalty)
    logits = logits.clone()
    logits[:, unique_ids] = penalized
    return logits


def _create_causal_mask(
    attention_mask: Optional[paddle.Tensor],
    input_tensor: paddle.Tensor,
    cache_position: paddle.Tensor,
    past_key_values: Optional[DynamicCache],
):
    dtype = input_tensor.dtype
    min_dtype = paddle.finfo(dtype).min
    sequence_length = input_tensor.shape[1]

    past_seen_tokens = cache_position[0] if past_key_values is not None else 0
    target_length = (
        attention_mask.shape[-1]
        if attention_mask is not None and isinstance(attention_mask, paddle.Tensor)
        else past_seen_tokens + sequence_length + 1
    )

    if attention_mask is not None and len(attention_mask.shape) == 4:
        causal_mask = attention_mask
    else:
        causal_mask = paddle.full(
            [sequence_length, target_length], fill_value=min_dtype, dtype=dtype
        )
        exclude_mask = paddle.arange(target_length) > cache_position.reshape([-1, 1])
        sliding_window = getattr(input_tensor, "_sliding_window", None)
        if sliding_window is not None:
            if sequence_length > sliding_window:
                exclude_mask = paddle.bitwise_or(
                    exclude_mask,
                    paddle.arange(target_length)
                    <= (cache_position.reshape([-1, 1]) - sliding_window),
                )
        causal_mask = causal_mask * exclude_mask.cast(dtype)
        causal_mask = causal_mask[None, None, :, :].expand(
            [input_tensor.shape[0], 1, -1, -1]
        )
        if attention_mask is not None:
            causal_mask = causal_mask.clone()
            if len(attention_mask.shape) == 2:
                mask_length = attention_mask.shape[-1]
                attention_mask_float = attention_mask.cast(dtype)
                padding_mask = (
                    causal_mask[:, :, :, :mask_length]
                    + attention_mask_float[:, None, None, :]
                )
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[
                    :, :, :, :mask_length
                ].masked_fill(padding_mask, min_dtype)

    return causal_mask


class MistralModel(RuntimeMixin, MistralPreTrainedModel):
    def __init__(
        self, config: MistralConfig, execution_backend="eager", runtime_options=None
    ):
        super().__init__(config)
        self._init_runtime(execution_backend, runtime_options)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=self.padding_idx
        )
        self.layers = nn.LayerList(
            [
                MistralDecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = MistralRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.init_weights()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    @runtime_boundary("forward")
    def forward(
        self,
        input_ids: paddle.Tensor = None,
        attention_mask: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.Tensor] = None,
        past_key_values: Optional[Union[DynamicCache, List[paddle.Tensor]]] = None,
        inputs_embeds: Optional[paddle.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[paddle.Tensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
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
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same "
                "time, and must specify either one"
            )

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        return_legacy_cache = False
        if use_cache and not isinstance(past_key_values, DynamicCache):
            if past_key_values is not None:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            else:
                past_key_values = DynamicCache()
            return_legacy_cache = True

        if cache_position is None:
            past_seen_tokens = (
                past_key_values.get_seq_length() if past_key_values is not None else 0
            )
            cache_position = paddle.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1]
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = _create_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values
        )

        hidden_states = inputs_embeds

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
            )

            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]
            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if return_legacy_cache and next_cache is not None:
            if isinstance(next_cache, DynamicCache):
                next_cache = next_cache.to_legacy_cache()

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, next_cache, all_hidden_states, all_self_attns]
                if v is not None
            )
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class MistralForCausalLM(MistralPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.model = MistralModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias_attr=False)
        self.init_weights()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def forward(
        self,
        input_ids: paddle.Tensor = None,
        attention_mask: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.Tensor] = None,
        past_key_values: Optional[Union[DynamicCache, List[paddle.Tensor]]] = None,
        inputs_embeds: Optional[paddle.Tensor] = None,
        labels: Optional[paddle.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[paddle.Tensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
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

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0] if not return_dict else outputs.last_hidden_state
        logits = self.lm_head(hidden_states)
        logits = logits.cast(paddle.float32)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :]
            shift_labels = labels[..., 1:]
            shift_logits = shift_logits.reshape([-1, self.config.vocab_size])
            shift_labels = shift_labels.reshape([-1])
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        past_kv = (
            outputs.past_key_values if hasattr(outputs, "past_key_values") else None
        )
        hidden = outputs.hidden_states if hasattr(outputs, "hidden_states") else None
        attns = outputs.attentions if hasattr(outputs, "attentions") else None

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=past_kv,
            hidden_states=hidden,
            attentions=attns,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        **kwargs,
    ):
        if past_key_values is not None:
            if inputs_embeds is not None:
                input_ids = input_ids[:, -cache_position.shape[0] :]
            elif input_ids.shape[1] != cache_position.shape[0]:
                input_ids = input_ids[:, cache_position]

        if attention_mask is not None and position_ids is None:
            position_ids = attention_mask.cast(paddle.int64).cumsum(axis=-1) - 1
            position_ids = position_ids.masked_fill(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        if inputs_embeds is not None and cache_position[0] == 0:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    def _generate_beam_search(
        self,
        inputs_embeds,
        attention_mask,
        max_length=256,
        min_length=1,
        num_beams=5,
        repetition_penalty=1.5,
        length_penalty=1.0,
        temperature=1.0,
        _suppress_token_id=None,
    ):
        """Search active beams and retain completed responses at every step."""
        seq_len = inputs_embeds.shape[1]
        eos_token_id = self.config.eos_token_id
        vocab_size = self.config.vocab_size
        max_new_tokens = max_length - seq_len

        def normalized_score(score, length):
            penalty = (
                (_BEAM_LENGTH_PENALTY_OFFSET + length)
                / (_BEAM_LENGTH_PENALTY_OFFSET + 1)
            ) ** length_penalty
            return score / penalty

        past_key_values = DynamicCache()
        outputs = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
            cache_position=paddle.arange(seq_len),
        )
        logits = self.lm_head(outputs[0][:, -1:, :])[:, 0, :]
        beam_tokens = paddle.empty([1, 0], dtype="int64")
        beam_scores = paddle.zeros([1], dtype="float32")
        finished_hypos = []

        for step in range(max_new_tokens):
            logits = logits / temperature
            if _suppress_token_id is not None:
                logits[:, _suppress_token_id] = float("-inf")
            if repetition_penalty != 1.0 and step:
                logits = paddle.concat(
                    [
                        _apply_repetition_penalty(
                            logits[index : index + 1], tokens, repetition_penalty
                        )
                        for index, tokens in enumerate(beam_tokens)
                    ],
                    axis=0,
                )
            if step < min_length:
                logits[:, eos_token_id] = float("-inf")

            candidate_scores = beam_scores.unsqueeze(-1) + F.log_softmax(
                logits, axis=-1
            )
            flat_scores = candidate_scores.reshape([-1])
            topk_scores, topk_indices = paddle.topk(
                flat_scores, min(2 * num_beams, flat_scores.shape[0])
            )
            beam_indices = topk_indices // vocab_size
            token_indices = topk_indices % vocab_size
            new_tokens, new_scores, beam_map = [], [], []
            for rank in range(topk_scores.shape[0]):
                score = float(topk_scores[rank])
                if not math.isfinite(score):
                    continue
                parent = int(beam_indices[rank])
                token = int(token_indices[rank])
                sequence = paddle.concat(
                    [beam_tokens[parent], token_indices[rank : rank + 1]]
                )
                if token == eos_token_id:
                    if rank < num_beams:
                        finished_hypos.append(
                            (normalized_score(score, step + 1), sequence)
                        )
                    continue
                new_tokens.append(sequence)
                new_scores.append(score)
                beam_map.append(parent)
                if len(new_tokens) == num_beams:
                    break

            finished_hypos.sort(key=lambda item: item[0], reverse=True)
            finished_hypos = finished_hypos[:num_beams]
            if not new_tokens:
                beam_tokens = None
                break
            beam_tokens = paddle.stack(new_tokens)
            beam_scores = paddle.to_tensor(new_scores, dtype="float32")
            bound_length = max_new_tokens if length_penalty > 0 else step + 1
            best_possible_score = normalized_score(max(new_scores), bound_length)
            if (
                len(finished_hypos) == num_beams
                and finished_hypos[-1][0] >= best_possible_score
            ):
                beam_tokens = None
                break
            if step + 1 == max_new_tokens:
                break

            reorder_idx = paddle.to_tensor(beam_map, dtype="int64")
            for layer_idx in range(len(past_key_values.key_cache)):
                past_key_values.key_cache[layer_idx] = paddle.index_select(
                    past_key_values.key_cache[layer_idx], reorder_idx, axis=0
                )
                past_key_values.value_cache[layer_idx] = paddle.index_select(
                    past_key_values.value_cache[layer_idx], reorder_idx, axis=0
                )
            attention_mask = paddle.index_select(attention_mask, reorder_idx, axis=0)
            attention_mask = paddle.concat(
                [
                    attention_mask,
                    paddle.ones([len(new_tokens), 1], dtype="int64"),
                ],
                axis=-1,
            )
            outputs = self.model(
                input_ids=beam_tokens[:, -1:],
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
                cache_position=paddle.arange(seq_len + step, seq_len + step + 1),
            )
            logits = self.lm_head(outputs[0][:, -1:, :])[:, 0, :]

        if beam_tokens is not None:
            finished_hypos.extend(
                (normalized_score(float(score), tokens.shape[0]), tokens)
                for score, tokens in zip(beam_scores, beam_tokens)
            )
        if not finished_hypos:
            raise ValueError("No finite beam-search candidates remain.")
        return max(finished_hypos, key=lambda item: item[0])[1].unsqueeze(0)

    @staticmethod
    def _next_token(logits, do_sample, top_p):
        if not do_sample:
            return paddle.argmax(logits, axis=-1, keepdim=True)
        probabilities = F.softmax(logits, axis=-1)
        indices = paddle.argsort(probabilities, axis=-1, descending=True)
        probabilities = paddle.take_along_axis(probabilities, indices, axis=-1)
        previous_mass = paddle.cumsum(probabilities, axis=-1) - probabilities
        probabilities = paddle.where(
            previous_mass < top_p, probabilities, paddle.zeros_like(probabilities)
        )
        selected = paddle.multinomial(probabilities, num_samples=1)
        return paddle.take_along_axis(indices, selected, axis=-1)

    def generate(
        self,
        input_ids=None,
        inputs_embeds=None,
        attention_mask=None,
        max_length=256,
        min_length=1,
        num_beams=1,
        do_sample=False,
        top_p=0.9,
        temperature=1.0,
        repetition_penalty=1.5,
        length_penalty=1.0,
        num_return_sequences=1,
    ):
        """Autoregressive generation loop."""
        if num_return_sequences != 1:
            raise ValueError("MatterChat generates one response per call.")
        if not 0 < top_p <= 1 or temperature <= 0 or repetition_penalty <= 0:
            raise ValueError("Invalid generation probability or penalty settings.")
        if num_beams < 1 or min_length < 0:
            raise ValueError("num_beams must be positive and min_length nonnegative.")
        if do_sample and num_beams != 1:
            raise ValueError("Nucleus sampling requires num_beams=1.")
        if inputs_embeds is not None:
            hidden = inputs_embeds
        else:
            hidden = self.model.embed_tokens(input_ids)

        batch_size = hidden.shape[0]
        seq_len = hidden.shape[1]
        if batch_size != 1:
            raise ValueError("MatterChat generation accepts one crystal per call.")
        if max_length <= seq_len or min_length > max_length - seq_len:
            raise ValueError("Generation length must exceed the input and minimum.")

        if attention_mask is None:
            attention_mask = paddle.ones([batch_size, seq_len], dtype=paddle.int64)

        _suppress_token_id = getattr(self.config, "pad_token_id", None)

        if num_beams > 1:
            return self._generate_beam_search(
                inputs_embeds=hidden,
                attention_mask=attention_mask,
                max_length=max_length,
                min_length=min_length,
                num_beams=num_beams,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                temperature=temperature,
                _suppress_token_id=_suppress_token_id,
            )

        # Greedy / sampling path (num_beams == 1)
        past_key_values = DynamicCache()
        cache_position = paddle.arange(seq_len)

        outputs = self.model(
            inputs_embeds=hidden,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
            cache_position=cache_position,
        )
        logits = self.lm_head(outputs[0][:, -1:, :])[:, 0, :]
        logits = logits / max(temperature, 1e-8)

        if _suppress_token_id is not None:
            logits[:, _suppress_token_id] = float("-inf")

        if repetition_penalty != 1.0:
            logits = _apply_repetition_penalty(logits, None, repetition_penalty)

        if min_length > 0:
            logits[:, self.config.eos_token_id] = float("-inf")
        next_tokens = self._next_token(logits, do_sample, top_p)

        generated = [next_tokens]
        gen_ids_tensor = next_tokens.reshape([-1])

        max_new_tokens = max_length - seq_len
        eos_token_id = getattr(self.config, "eos_token_id", 2)

        for step in range(1, max_new_tokens):
            if next_tokens.reshape([-1])[0].item() == eos_token_id:
                break
            attention_mask = paddle.concat(
                [
                    attention_mask,
                    paddle.ones([batch_size, 1], dtype=paddle.int64),
                ],
                axis=-1,
            )

            past_len = seq_len + step
            cache_position = paddle.arange(past_len - 1, past_len)

            step_outputs = self.model(
                input_ids=next_tokens,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
                cache_position=cache_position,
            )

            logits = self.lm_head(step_outputs[0][:, -1:, :])[:, 0, :]
            logits = logits / max(temperature, 1e-8)

            if _suppress_token_id is not None:
                logits[:, _suppress_token_id] = float("-inf")

            if repetition_penalty != 1.0:
                logits = _apply_repetition_penalty(
                    logits, gen_ids_tensor, repetition_penalty
                )

            if step < min_length:
                logits[:, eos_token_id] = float("-inf")
            next_tokens = self._next_token(logits, do_sample, top_p)

            generated.append(next_tokens)
            gen_ids_tensor = paddle.concat([gen_ids_tensor, next_tokens.reshape([-1])])

            if next_tokens.reshape([-1])[0].item() == eos_token_id:
                break

        output_ids = paddle.concat(generated, axis=-1)
        return output_ids

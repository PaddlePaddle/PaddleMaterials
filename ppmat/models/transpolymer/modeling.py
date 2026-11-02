import math

import paddle
import paddle.nn as nn
import paddle.nn.functional as F


def _normal_init(layer, initializer_range):
    if isinstance(layer, nn.Linear):
        nn.initializer.Normal(mean=0.0, std=initializer_range)(layer.weight)
        if layer.bias is not None:
            nn.initializer.Constant(0.0)(layer.bias)
    elif isinstance(layer, nn.Embedding):
        nn.initializer.Normal(mean=0.0, std=initializer_range)(layer.weight)
        padding_idx = getattr(layer, "_padding_idx", None)
        if padding_idx is not None:
            with paddle.no_grad():
                weight = layer.weight.numpy()
                weight[padding_idx] = 0
                layer.weight.set_value(weight)
    elif isinstance(layer, nn.LayerNorm):
        nn.initializer.Constant(1.0)(layer.weight)
        nn.initializer.Constant(0.0)(layer.bias)


def init_weights(layer, initializer_range):
    for sublayer in [layer] + list(layer.sublayers()):
        _normal_init(sublayer, initializer_range)


def create_position_ids_from_input_ids(input_ids, padding_idx):
    mask = paddle.cast(input_ids != padding_idx, "int64")
    incremental_indices = paddle.cumsum(mask, axis=1) * mask
    return incremental_indices + padding_idx


class RobertaEmbeddings(nn.Layer):
    def __init__(
        self,
        vocab_size,
        hidden_size,
        max_position_embeddings,
        type_vocab_size,
        hidden_dropout_prob,
        layer_norm_eps,
        pad_token_id,
        position_embedding_type,
    ):
        super().__init__()
        # Paddle masks padding_idx outputs to zero at runtime, while HuggingFace
        # PyTorch returns the loaded padding row. Keep runtime behavior aligned
        # with the source checkpoint and handle padding only in the attention mask.
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(
            max_position_embeddings,
            hidden_size,
            padding_idx=pad_token_id,
        )
        self.token_type_embeddings = nn.Embedding(type_vocab_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, epsilon=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.padding_idx = pad_token_id
        self.position_embedding_type = position_embedding_type

    def forward(
        self,
        input_ids=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
    ):
        if position_ids is None:
            if input_ids is not None:
                position_ids = create_position_ids_from_input_ids(
                    input_ids, self.padding_idx
                )
            else:
                input_shape = inputs_embeds.shape[:-1]
                seq_length = input_shape[1]
                position_ids = paddle.arange(
                    self.padding_idx + 1,
                    seq_length + self.padding_idx + 1,
                    dtype="int64",
                )
                position_ids = position_ids.unsqueeze(0).expand(input_shape)

        input_shape = (
            input_ids.shape if input_ids is not None else inputs_embeds.shape[:-1]
        )
        if token_type_ids is None:
            token_type_ids = paddle.zeros(input_shape, dtype="int64")
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        embeddings = inputs_embeds + self.token_type_embeddings(token_type_ids)
        if self.position_embedding_type == "absolute":
            embeddings = embeddings + self.position_embeddings(position_ids)
        embeddings = self.LayerNorm(embeddings)
        return self.dropout(embeddings)


class RobertaSelfAttention(nn.Layer):
    def __init__(self, hidden_size, num_attention_heads, attention_probs_dropout_prob):
        super().__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads
        self.all_head_size = num_attention_heads * self.attention_head_size
        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)
        self.dropout = nn.Dropout(attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_shape = list(x.shape[:-1]) + [
            self.num_attention_heads,
            self.attention_head_size,
        ]
        x = x.reshape(new_shape)
        return x.transpose([0, 2, 1, 3])

    def forward(self, hidden_states, attention_mask=None):
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))

        attention_scores = paddle.matmul(query_layer, key_layer, transpose_y=True)
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        attention_probs = self.dropout(F.softmax(attention_scores, axis=-1))

        context_layer = paddle.matmul(attention_probs, value_layer)
        context_layer = context_layer.transpose([0, 2, 1, 3])
        return context_layer.reshape(
            [
                context_layer.shape[0],
                context_layer.shape[1],
                self.all_head_size,
            ]
        )


class RobertaSelfOutput(nn.Layer):
    def __init__(self, hidden_size, hidden_dropout_prob, layer_norm_eps):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, epsilon=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dropout(self.dense(hidden_states))
        return self.LayerNorm(hidden_states + input_tensor)


class RobertaAttention(nn.Layer):
    def __init__(
        self,
        hidden_size,
        num_attention_heads,
        hidden_dropout_prob,
        attention_probs_dropout_prob,
        layer_norm_eps,
    ):
        super().__init__()
        self.self = RobertaSelfAttention(
            hidden_size,
            num_attention_heads,
            attention_probs_dropout_prob,
        )
        self.output = RobertaSelfOutput(
            hidden_size,
            hidden_dropout_prob,
            layer_norm_eps,
        )

    def forward(self, hidden_states, attention_mask=None):
        attention_output = self.self(hidden_states, attention_mask)
        return self.output(attention_output, hidden_states)


class RobertaIntermediate(nn.Layer):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, intermediate_size)

    def forward(self, hidden_states):
        return F.gelu(self.dense(hidden_states))


class RobertaOutput(nn.Layer):
    def __init__(
        self,
        hidden_size,
        intermediate_size,
        hidden_dropout_prob,
        layer_norm_eps,
    ):
        super().__init__()
        self.dense = nn.Linear(intermediate_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, epsilon=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dropout(self.dense(hidden_states))
        return self.LayerNorm(hidden_states + input_tensor)


class RobertaLayer(nn.Layer):
    def __init__(
        self,
        hidden_size,
        intermediate_size,
        num_attention_heads,
        hidden_dropout_prob,
        attention_probs_dropout_prob,
        layer_norm_eps,
    ):
        super().__init__()
        self.attention = RobertaAttention(
            hidden_size,
            num_attention_heads,
            hidden_dropout_prob,
            attention_probs_dropout_prob,
            layer_norm_eps,
        )
        self.intermediate = RobertaIntermediate(hidden_size, intermediate_size)
        self.output = RobertaOutput(
            hidden_size,
            intermediate_size,
            hidden_dropout_prob,
            layer_norm_eps,
        )

    def forward(self, hidden_states, attention_mask=None):
        attention_output = self.attention(hidden_states, attention_mask)
        layer_output = self.intermediate(attention_output)
        return self.output(layer_output, attention_output)


class RobertaEncoder(nn.Layer):
    def __init__(
        self,
        hidden_size,
        intermediate_size,
        num_attention_heads,
        num_hidden_layers,
        hidden_dropout_prob,
        attention_probs_dropout_prob,
        layer_norm_eps,
    ):
        super().__init__()
        self.layer = nn.LayerList(
            [
                RobertaLayer(
                    hidden_size,
                    intermediate_size,
                    num_attention_heads,
                    hidden_dropout_prob,
                    attention_probs_dropout_prob,
                    layer_norm_eps,
                )
                for _ in range(num_hidden_layers)
            ]
        )

    def forward(self, hidden_states, attention_mask=None):
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
        return hidden_states


class RobertaPooler(nn.Layer):
    def __init__(self, hidden_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        return self.activation(self.dense(hidden_states[:, 0]))


class RobertaModel(nn.Layer):
    """RoBERTa backbone used by :class:`TransPolymerRegressor`.

    Model parameters are supplied by the outer TransPolymer model from the project
    YAML or a converted checkpoint's ``config.json``. This module intentionally
    owns no checkpoint/configuration serialization API.
    """

    def __init__(
        self,
        vocab_size=50265,
        hidden_size=768,
        num_hidden_layers=6,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=514,
        type_vocab_size=1,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        pad_token_id=1,
        position_embedding_type="absolute",
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.initializer_range = initializer_range
        self.embeddings = RobertaEmbeddings(
            vocab_size,
            hidden_size,
            max_position_embeddings,
            type_vocab_size,
            hidden_dropout_prob,
            layer_norm_eps,
            pad_token_id,
            position_embedding_type,
        )
        self.encoder = RobertaEncoder(
            hidden_size,
            intermediate_size,
            num_attention_heads,
            num_hidden_layers,
            hidden_dropout_prob,
            attention_probs_dropout_prob,
            layer_norm_eps,
        )
        self.pooler = RobertaPooler(hidden_size)
        init_weights(self, initializer_range)

    def resize_token_embeddings(self, new_num_tokens):
        old_embeddings = self.embeddings.word_embeddings
        old_num_tokens, embedding_dim = old_embeddings.weight.shape
        if new_num_tokens == old_num_tokens:
            return old_embeddings
        new_embeddings = nn.Embedding(new_num_tokens, embedding_dim)
        _normal_init(new_embeddings, self.initializer_range)
        num_to_copy = min(old_num_tokens, new_num_tokens)
        with paddle.no_grad():
            weight = new_embeddings.weight.numpy()
            weight[:num_to_copy] = old_embeddings.weight.numpy()[:num_to_copy]
            new_embeddings.weight.set_value(weight)
        self.embeddings.word_embeddings = new_embeddings
        self.vocab_size = new_num_tokens
        return new_embeddings

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
    ):
        input_shape = (
            input_ids.shape if input_ids is not None else inputs_embeds.shape[:-1]
        )
        if attention_mask is None:
            attention_mask = paddle.ones(input_shape, dtype="int64")

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = paddle.cast(
            extended_attention_mask,
            paddle.get_default_dtype(),
        )
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        embedding_output = self.embeddings(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
        )
        return self.encoder(embedding_output, attention_mask=extended_attention_mask)

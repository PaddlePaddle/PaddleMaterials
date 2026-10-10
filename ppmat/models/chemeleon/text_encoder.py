# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

# Adapted from https://github.com/hspark1212/chemeleon

"""Text condition encoder.

Pipeline: tokenizer -> transformer forward -> optional projection ->
classifier-free-guidance null substitution at the embedding dim ->
text_emb MLP down to the model text_dim.

The transformer forward is injected as a module (any object exposing
``forward(input_ids, attention_mask) -> namespace(last_hidden_state)``,
e.g. MinimalBert); an optional ``text_proj`` callable applies a projection
stage on top of the class-token embedding.
"""

from typing import List
from typing import Optional

import paddle

from ppmat.models.chemeleon.diff_utils import prob_mask_like


class TextEncoder(paddle.nn.Layer):
    def __init__(
        self,
        text_embed_dim: int = 768,
        max_text_len: int = 256,
        text_dim: int = 512,
        tokenizer=None,
        text_transformer=None,
        text_proj: Optional[paddle.nn.Layer] = None,
    ):
        super(TextEncoder, self).__init__()
        # configs
        self.text_embed_dim = text_embed_dim
        self.max_text_len = max_text_len
        self.text_dim = text_dim

        # embedding layers
        self.text_emb = paddle.nn.Sequential(
            paddle.nn.Linear(self.text_embed_dim, self.text_embed_dim),
            paddle.nn.LayerNorm(self.text_embed_dim),
            paddle.nn.GELU(),
            paddle.nn.Linear(self.text_embed_dim, self.text_dim),
        )
        # follows the default dtype so the parameter matches the rest of
        # the model
        self.null_text_embeds = self.create_parameter(
            shape=[1, self.text_embed_dim],
            attr=paddle.ParamAttr(),
            dtype=paddle.get_default_dtype(),
            default_initializer=paddle.nn.initializer.Normal(),
        )

        # injected modules: tokenizer + transformer forward (+ optional
        # projection); None means no text conditioning at construction time
        self.tokenizer = tokenizer
        self.text_encoder = text_transformer
        self.text_proj = text_proj

    def text_encode(self, batch_text: List[str]) -> paddle.Tensor:
        tokenized = self.tokenizer.batch_encode_plus(
            batch_text,
            padding="longest",
            max_length=self.max_text_len,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = tokenized.input_ids
        attention_mask = tokenized.attention_mask

        # embeddings are the class token embeddings
        outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = outputs.last_hidden_state[:, 0, :]  # [B, D]

        # projection layer when using contrastive learning model
        if self.text_proj is not None:  # contrastive learning model
            embeddings = self.text_proj(embeddings)

        return embeddings

    def get_text_embeds(self, batch_text: List[str], cond_drop_prob: float):
        batch_size = len(batch_text)
        # get text embeddings from text_encoder
        text_embeds = self.text_encode(batch_text)  # [B, D]
        # drop embeddings with cond_drop_prob
        text_keep_mask = prob_mask_like((batch_size), 1.0 - cond_drop_prob)  # [B]
        text_embeds = paddle.where(
            text_keep_mask[:, None],
            text_embeds,
            self.null_text_embeds.expand([batch_size, self.text_embed_dim]),
        )  # [B, D]
        # last embedding to match dimension of hidden_dim
        text_embeds = self.text_emb(text_embeds)  # [B, hidden_dim]
        return text_embeds

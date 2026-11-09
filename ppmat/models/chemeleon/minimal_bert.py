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

"""Minimal BERT-base forward implementation and pure-python WordPiece tokenizer.

Parameter naming matches the HuggingFace BertModel state_dict layout
(BERT-base: 12 layers, 12 heads, hidden 768, intermediate 3072, vocab 31116
for MatTPUSciBERT):

    embeddings.word_embeddings.weight
    embeddings.position_embeddings.weight
    embeddings.token_type_embeddings.weight
    embeddings.LayerNorm.{weight,bias}
    encoder.layer.{i}.attention.self.{query,key,value}.{weight,bias}
    encoder.layer.{i}.attention.output.dense.{weight,bias}
    encoder.layer.{i}.attention.output.LayerNorm.{weight,bias}
    encoder.layer.{i}.intermediate.dense.{weight,bias}
    encoder.layer.{i}.output.dense.{weight,bias}
    encoder.layer.{i}.output.LayerNorm.{weight,bias}

No pooler, no pretraining heads, no dropout layers: the intended use is the
frozen eval-mode encoder feeding the CLS token to the diffusion model.
Use ``.astype(dtype)`` to switch the module between float32 and float64.

The tokenizer reproduces BertTokenizer behavior (slow,
do_lower_case=False) with the batching options used by the model:
padding="longest", truncation=True, max_length, and tensor return.
"""

import collections
import unicodedata
from pathlib import Path

import paddle
import paddle.nn.functional as F

VOCAB_PATH = Path(__file__).resolve().parent / "resources" / "vocab.txt"


def load_vocab(vocab_path):
    """Read a standard BERT vocab.txt into an ordered token->id mapping."""
    vocab = collections.OrderedDict()
    with open(vocab_path, "r", encoding="utf-8") as reader:
        for index, line in enumerate(reader):
            token = line.rstrip("\n")
            vocab[token] = index
    return vocab


def whitespace_tokenize(text):
    text = text.strip()
    if not text:
        return []
    return text.split()


def _is_whitespace(char):
    if char in (" ", "\t", "\n", "\r"):
        return True
    cat = unicodedata.category(char)
    if cat == "Zs":
        return True
    return False


def _is_control(char):
    if char in ("\t", "\n", "\r"):
        return False
    cat = unicodedata.category(char)
    if cat.startswith("C"):
        return True
    return False


def _is_punctuation(char):
    cp = ord(char)
    # treat all non-letter/number ASCII as punctuation
    if (33 <= cp <= 47) or (58 <= cp <= 64) or (91 <= cp <= 96) or (123 <= cp <= 126):
        return True
    cat = unicodedata.category(char)
    if cat.startswith("P"):
        return True
    return False


class BasicTokenizer:
    """Runs basic tokenization (punctuation splitting, lower casing, CJK)."""

    def __init__(self, do_lower_case=False, tokenize_chinese_chars=True):
        self.do_lower_case = do_lower_case
        self.tokenize_chinese_chars = tokenize_chinese_chars

    def tokenize(self, text):
        text = self._clean_text(text)
        if self.tokenize_chinese_chars:
            text = self._tokenize_chinese_chars(text)
        orig_tokens = whitespace_tokenize(text)
        split_tokens = []
        for token in orig_tokens:
            if self.do_lower_case:
                token = token.lower()
                token = self._run_strip_accents(token)
            split_tokens.extend(self._run_split_on_punc(token))
        output_tokens = whitespace_tokenize(" ".join(split_tokens))
        return output_tokens

    def _run_strip_accents(self, text):
        text = unicodedata.normalize("NFD", text)
        output = []
        for char in text:
            cat = unicodedata.category(char)
            if cat == "Mn":
                continue
            output.append(char)
        return "".join(output)

    def _run_split_on_punc(self, text):
        chars = list(text)
        i = 0
        start_new_word = True
        output = []
        while i < len(chars):
            char = chars[i]
            if _is_punctuation(char):
                output.append([char])
                start_new_word = True
            else:
                if start_new_word:
                    output.append([])
                start_new_word = False
                output[-1].append(char)
            i += 1
        return ["".join(x) for x in output]

    def _tokenize_chinese_chars(self, text):
        output = []
        for char in text:
            cp = ord(char)
            if self._is_chinese_char(cp):
                output.append(" ")
                output.append(char)
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)

    def _is_chinese_char(self, cp):
        # CJK Unicode block ranges
        if (
            (0x4E00 <= cp <= 0x9FFF)
            or (0x3400 <= cp <= 0x4DBF)
            or (0x20000 <= cp <= 0x2A6DF)
            or (0x2A700 <= cp <= 0x2B73F)
            or (0x2B740 <= cp <= 0x2B81F)
            or (0x2B820 <= cp <= 0x2CEAF)
            or (0xF900 <= cp <= 0xFAFF)
            or (0x2F800 <= cp <= 0x2FA1F)
        ):
            return True
        return False

    def _clean_text(self, text):
        output = []
        for char in text:
            cp = ord(char)
            if cp == 0 or cp == 0xFFFD or _is_control(char):
                continue
            if _is_whitespace(char):
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)


class WordpieceTokenizer:
    """Runs WordPiece tokenization (greedy longest-match-first)."""

    def __init__(self, vocab, unk_token, max_input_chars_per_word=100):
        self.vocab = vocab
        self.unk_token = unk_token
        self.max_input_chars_per_word = max_input_chars_per_word

    def tokenize(self, text):
        output_tokens = []
        for token in whitespace_tokenize(text):
            chars = list(token)
            if len(chars) > self.max_input_chars_per_word:
                output_tokens.append(self.unk_token)
                continue

            is_bad = False
            start = 0
            sub_tokens = []
            while start < len(chars):
                end = len(chars)
                cur_substr = None
                while start < end:
                    substr = "".join(chars[start:end])
                    if start > 0:
                        substr = "##" + substr
                    if substr in self.vocab:
                        cur_substr = substr
                        break
                    end -= 1
                if cur_substr is None:
                    is_bad = True
                    break
                sub_tokens.append(cur_substr)
                start = end

            if is_bad:
                output_tokens.append(self.unk_token)
            else:
                output_tokens.extend(sub_tokens)
        return output_tokens


class BatchEncoding(dict):
    """Dict with attribute access (HF BatchEncoding style)."""

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError:
            raise AttributeError(item)


class BertWordPieceTokenizer:
    """Pure-python WordPiece tokenizer matching BertTokenizer behavior.

    Supports the subset used by the model: batch_encode_plus with
    padding="longest", truncation to max_length (longest-first, special
    tokens accounted for) and paddle tensor return via return_tensors="pt".
    """

    def __init__(self, vocab_file, do_lower_case=False):
        self.vocab = load_vocab(vocab_file)
        self.ids_to_tokens = collections.OrderedDict(
            [(ids, tok) for tok, ids in self.vocab.items()]
        )
        self.basic_tokenizer = BasicTokenizer(do_lower_case=do_lower_case)
        self.wordpiece_tokenizer = WordpieceTokenizer(
            vocab=self.vocab, unk_token="[UNK]"
        )
        self.pad_token = "[PAD]"
        self.cls_token = "[CLS]"
        self.sep_token = "[SEP]"
        self.pad_token_id = self.vocab[self.pad_token]
        self.cls_token_id = self.vocab[self.cls_token]
        self.sep_token_id = self.vocab[self.sep_token]

    def tokenize(self, text):
        split_tokens = []
        for token in self.basic_tokenizer.tokenize(text):
            for sub_token in self.wordpiece_tokenizer.tokenize(token):
                split_tokens.append(sub_token)
        return split_tokens

    def convert_tokens_to_ids(self, tokens):
        return [self.vocab.get(token, self.vocab["[UNK]"]) for token in tokens]

    def encode(self, text, max_length=None):
        """Tokenize one text and add [CLS]/[SEP]; truncate when max_length given."""
        ids = self.convert_tokens_to_ids(self.tokenize(text))
        ids = [self.cls_token_id] + ids + [self.sep_token_id]
        if max_length is not None and len(ids) > max_length:
            ids = ids[: max_length - 1] + [self.sep_token_id]
        return ids

    def batch_encode_plus(
        self,
        batch_text,
        padding="longest",
        max_length=None,
        truncation=True,
        return_tensors=None,
    ):
        if padding != "longest":
            raise NotImplementedError("only padding='longest' is supported")
        encoded = [
            self.encode(text, max_length if truncation else None) for text in batch_text
        ]
        longest = max(len(ids) for ids in encoded)
        input_ids, attention_mask = [], []
        for ids in encoded:
            n_pad = longest - len(ids)
            input_ids.append(ids + [self.pad_token_id] * n_pad)
            attention_mask.append([1] * len(ids) + [0] * n_pad)
        output = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if return_tensors is not None:
            output = {key: paddle.to_tensor(value) for key, value in output.items()}
        return BatchEncoding(output)


class BertSelfAttention(paddle.nn.Layer):
    def __init__(self, hidden_size, num_attention_heads):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads
        self.query = paddle.nn.Linear(hidden_size, hidden_size)
        self.key = paddle.nn.Linear(hidden_size, hidden_size)
        self.value = paddle.nn.Linear(hidden_size, hidden_size)

    def _shape(self, x):
        # [B, L, H, Dh] -> [B, H, L, Dh]
        new_shape = x.shape[:2] + [self.num_attention_heads, self.attention_head_size]
        return x.reshape(new_shape).transpose([0, 2, 1, 3])

    def forward(self, hidden_states, attention_mask):
        query = self._shape(self.query(hidden_states))
        key = self._shape(self.key(hidden_states))
        value = self._shape(self.value(hidden_states))

        scores = paddle.matmul(query, key, transpose_y=True)
        scores = scores / (self.attention_head_size**0.5)
        scores = scores + attention_mask
        probs = paddle.nn.functional.softmax(scores, axis=-1)

        context = paddle.matmul(probs, value)
        context = context.transpose([0, 2, 1, 3])
        context = context.reshape(context.shape[:2] + [-1])
        return context


class BertSelfOutput(paddle.nn.Layer):
    def __init__(self, hidden_size, layer_norm_eps):
        super().__init__()
        self.dense = paddle.nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = paddle.nn.LayerNorm(hidden_size, epsilon=layer_norm_eps)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(paddle.nn.Layer):
    def __init__(self, hidden_size, num_attention_heads, layer_norm_eps):
        super().__init__()
        self.self = BertSelfAttention(hidden_size, num_attention_heads)
        self.output = BertSelfOutput(hidden_size, layer_norm_eps)

    def forward(self, hidden_states, attention_mask):
        self_output = self.self(hidden_states, attention_mask)
        attention_output = self.output(self_output, hidden_states)
        return attention_output


class BertIntermediate(paddle.nn.Layer):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.dense = paddle.nn.Linear(hidden_size, intermediate_size)

    def forward(self, hidden_states):
        return F.gelu(self.dense(hidden_states))


class BertOutput(paddle.nn.Layer):
    def __init__(self, hidden_size, intermediate_size, layer_norm_eps):
        super().__init__()
        self.dense = paddle.nn.Linear(intermediate_size, hidden_size)
        self.LayerNorm = paddle.nn.LayerNorm(hidden_size, epsilon=layer_norm_eps)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(paddle.nn.Layer):
    def __init__(
        self, hidden_size, num_attention_heads, intermediate_size, layer_norm_eps
    ):
        super().__init__()
        self.attention = BertAttention(hidden_size, num_attention_heads, layer_norm_eps)
        self.intermediate = BertIntermediate(hidden_size, intermediate_size)
        self.output = BertOutput(hidden_size, intermediate_size, layer_norm_eps)

    def forward(self, hidden_states, attention_mask):
        attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class BertEncoder(paddle.nn.Layer):
    def __init__(
        self,
        num_hidden_layers,
        hidden_size,
        num_attention_heads,
        intermediate_size,
        layer_norm_eps,
    ):
        super().__init__()
        self.layer = paddle.nn.LayerList(
            [
                BertLayer(
                    hidden_size, num_attention_heads, intermediate_size, layer_norm_eps
                )
                for _ in range(num_hidden_layers)
            ]
        )

    def forward(self, hidden_states, attention_mask):
        for layer in self.layer:
            hidden_states = layer(hidden_states, attention_mask)
        return hidden_states


class BertEmbeddings(paddle.nn.Layer):
    def __init__(
        self,
        vocab_size,
        hidden_size,
        max_position_embeddings,
        type_vocab_size,
        layer_norm_eps,
    ):
        super().__init__()
        self.word_embeddings = paddle.nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = paddle.nn.Embedding(
            max_position_embeddings, hidden_size
        )
        self.token_type_embeddings = paddle.nn.Embedding(type_vocab_size, hidden_size)
        self.LayerNorm = paddle.nn.LayerNorm(hidden_size, epsilon=layer_norm_eps)

    def forward(self, input_ids, token_type_ids=None):
        batch_size, seq_length = input_ids.shape
        if token_type_ids is None:
            token_type_ids = paddle.zeros([batch_size, seq_length], dtype="int64")
        position_ids = paddle.arange(seq_length, dtype="int64").expand(
            [batch_size, seq_length]
        )
        embeddings = (
            self.word_embeddings(input_ids)
            + self.position_embeddings(position_ids)
            + self.token_type_embeddings(token_type_ids)
        )
        return self.LayerNorm(embeddings)


class MinimalBert(paddle.nn.Layer):
    """BERT-base encoder returning last_hidden_state via a HF-style namespace.

    forward(input_ids, attention_mask) -> object with .last_hidden_state
    [B, L, hidden]; the model takes the CLS position (index 0) downstream.
    """

    def __init__(
        self,
        vocab_size=31116,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        max_position_embeddings=512,
        type_vocab_size=2,
        layer_norm_eps=1e-12,
    ):
        super().__init__()
        self.embeddings = BertEmbeddings(
            vocab_size,
            hidden_size,
            max_position_embeddings,
            type_vocab_size,
            layer_norm_eps,
        )
        self.encoder = BertEncoder(
            num_hidden_layers,
            hidden_size,
            num_attention_heads,
            intermediate_size,
            layer_norm_eps,
        )

    def forward(self, input_ids, attention_mask=None):
        if attention_mask is None:
            attention_mask = paddle.ones_like(input_ids)
        # additive attention mask
        extended_attention_mask = (1.0 - attention_mask.astype("float32")) * -10000.0
        extended_attention_mask = extended_attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.astype(
            self.embeddings.word_embeddings.weight.dtype
        )
        hidden_states = self.embeddings(input_ids)
        hidden_states = self.encoder(hidden_states, extended_attention_mask)
        return BatchEncoding({"last_hidden_state": hidden_states})

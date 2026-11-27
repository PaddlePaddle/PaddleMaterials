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

"""Tokenizer wrapper for MatterChat training and inference.

Implements the subset of the HuggingFace ``tokenizers`` file format that
MatterChat uses, without depending on the ``tokenizers`` package:

- Metaspace pre-tokenization + BPE with byte fallback and unknown fusion
  (the released MatterChat checkpoint).
- Whitespace pre-tokenization + WordLevel (unit-test fixtures).
- TemplateProcessing that prepends special tokens.
- Left/right padding and truncation, and the released decoder chain.

Unsupported components raise ``NotImplementedError`` instead of silently
producing ids that differ from the reference implementation.
"""

import json
import os
import re
import unicodedata
from dataclasses import dataclass
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

import numpy as np
import paddle


@dataclass
class EncodingResult:
    """Token ids and attention mask returned by the wrapper."""

    input_ids: paddle.Tensor
    attention_mask: paddle.Tensor


_BYTE_TOKEN_RE = re.compile(r"^<0x([0-9A-Fa-f]{2})>$")


def _is_word_char(char: str) -> bool:
    """Match the Rust regex ``\\w`` used by the tokenizers Whitespace pre-tokenizer.

    Rust defines ``\\w`` as Alphabetic + Mark + Nd + Nl + Pc + Join_Control.
    Python's ``re`` uses a narrower definition, so classify explicitly.
    """
    category = unicodedata.category(char)
    return (
        category[0] in ("L", "M")
        or category in ("Nd", "Nl", "Pc")
        or char in ("‌", "‍")
    )


def _is_whitespace(char: str) -> bool:
    """Match the Rust regex ``\\s`` (Unicode White_Space) used by tokenizers.

    Python additionally reports the four C0 separators as whitespace.
    """
    return char.isspace() and char not in "\x1c\x1d\x1e\x1f"


def _split_whitespace(text: str) -> List[str]:
    pieces = []
    current: List[str] = []
    current_is_word = None
    for char in text:
        if _is_whitespace(char):
            if current:
                pieces.append("".join(current))
                current, current_is_word = [], None
            continue
        is_word = _is_word_char(char)
        if current and is_word != current_is_word:
            pieces.append("".join(current))
            current = []
        current.append(char)
        current_is_word = is_word
    if current:
        pieces.append("".join(current))
    return pieces


class _Trie:
    """Leftmost-longest matcher for added token contents."""

    _END = "\x00"

    def __init__(self, contents: List[str]):
        self.root: Dict[str, dict] = {}
        for content in contents:
            if not content or self._END in content:
                continue
            node = self.root
            for char in content:
                node = node.setdefault(char, {})
            node[self._END] = content

    def split(self, text: str):
        """Yield ``(is_added, chunk)`` pairs covering ``text`` in order."""
        chunks = []
        start = 0
        index = 0
        size = len(text)
        while index < size:
            node = self.root
            cursor = index
            match = None
            match_end = index
            while cursor < size and text[cursor] in node:
                node = node[text[cursor]]
                cursor += 1
                if self._END in node:
                    match = node[self._END]
                    match_end = cursor
            if match is None:
                index += 1
                continue
            if index > start:
                chunks.append((False, text[start:index]))
            chunks.append((True, match))
            index = match_end
            start = match_end
        if start < size:
            chunks.append((False, text[start:]))
        return chunks


class _BPEModel:
    """BPE model with byte fallback, matching ``tokenizers`` semantics."""

    def __init__(self, model_cfg: dict):
        self.vocab = dict(model_cfg["vocab"])
        self.unk_token = model_cfg.get("unk_token")
        self.unk_id = self.vocab.get(self.unk_token) if self.unk_token else None
        self.byte_fallback = bool(model_cfg.get("byte_fallback", False))
        self.fuse_unk = bool(model_cfg.get("fuse_unk", False))
        if model_cfg.get("continuing_subword_prefix"):
            raise NotImplementedError("continuing_subword_prefix is not supported.")
        if model_cfg.get("end_of_word_suffix"):
            raise NotImplementedError("end_of_word_suffix is not supported.")
        if model_cfg.get("ignore_merges"):
            raise NotImplementedError("ignore_merges is not supported.")
        merges = model_cfg.get("merges") or []
        self.ranks: Dict[tuple, int] = {}
        for rank, merge in enumerate(merges):
            if isinstance(merge, str):
                first, second = merge.split(" ", 1)
            else:
                first, second = merge
            self.ranks[(first, second)] = rank
        self._cache: Dict[str, List[str]] = {}

    def _initial_symbols(self, text: str) -> List[str]:
        symbols = []
        for char in text:
            if char in self.vocab:
                symbols.append(char)
            elif self.byte_fallback:
                for byte in char.encode("utf-8"):
                    symbols.append(f"<0x{byte:02X}>")
            else:
                symbols.append(self.unk_token)
        return symbols

    def _merge(self, symbols: List[str]) -> List[str]:
        if len(symbols) < 2:
            return symbols
        parts = symbols
        while True:
            best = None
            best_rank = None
            for index in range(len(parts) - 1):
                rank = self.ranks.get((parts[index], parts[index + 1]))
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank = rank
                    best = (parts[index], parts[index + 1])
            if best is None:
                return parts
            merged = []
            index = 0
            last = len(parts) - 1
            while index < len(parts):
                if (
                    index < last
                    and parts[index] == best[0]
                    and parts[index + 1] == best[1]
                ):
                    merged.append(best[0] + best[1])
                    index += 2
                else:
                    merged.append(parts[index])
                    index += 1
            parts = merged

    def tokenize(self, text: str) -> List[Optional[int]]:
        if not text:
            return []
        cached = self._cache.get(text)
        if cached is None:
            cached = self._merge(self._initial_symbols(text))
            self._cache[text] = cached
        ids = []
        unknown_pending = False
        for symbol in cached:
            token_id = self.vocab.get(symbol)
            if token_id is None:
                if self.fuse_unk and unknown_pending:
                    continue
                ids.append(self.unk_id)
                unknown_pending = True
            else:
                ids.append(token_id)
                unknown_pending = False
        return ids


class _WordLevelModel:
    """WordLevel model: every pre-token maps to one vocabulary entry."""

    def __init__(self, model_cfg: dict):
        self.vocab = dict(model_cfg["vocab"])
        unk_token = model_cfg.get("unk_token")
        self.unk_id = self.vocab.get(unk_token) if unk_token else None

    def tokenize(self, text: str) -> List[Optional[int]]:
        if not text:
            return []
        return [self.vocab.get(text, self.unk_id)]


class _MetaspacePreTokenizer:
    def __init__(self, cfg: dict):
        self.replacement = cfg.get("replacement", "_")
        self.prepend_scheme = cfg.get("prepend_scheme", "always")
        self.split = bool(cfg.get("split", True))
        if self.split:
            raise NotImplementedError("Metaspace split=True is not supported.")

    def pre_tokenize(self, text: str, is_first: bool) -> List[str]:
        if not text:
            return []
        normalized = text.replace(" ", self.replacement)
        # "first" only prepends to the first chunk of the whole text.
        if self.prepend_scheme == "always" or (
            self.prepend_scheme == "first" and is_first
        ):
            if not normalized.startswith(self.replacement):
                normalized = self.replacement + normalized
        return [normalized]


class _WhitespacePreTokenizer:
    def pre_tokenize(self, text: str, is_first: bool) -> List[str]:
        return _split_whitespace(text)


class _TemplateProcessor:
    """TemplateProcessing for the single-sequence case."""

    def __init__(self, cfg: Optional[dict]):
        self.single = None
        self.special_ids: Dict[str, List[int]] = {}
        if cfg is None:
            return
        if cfg.get("type") != "TemplateProcessing":
            raise NotImplementedError(f"Unsupported post_processor: {cfg.get('type')}")
        self.single = cfg.get("single")
        if self.single is None:
            raise NotImplementedError("TemplateProcessing requires a single template.")
        for token, spec in (cfg.get("special_tokens") or {}).items():
            self.special_ids[token] = list(spec["ids"])

    def process(self, ids: List[int]):
        if self.single is None:
            return ids, [0] * len(ids)
        out_ids: List[int] = []
        mask: List[int] = []
        for item in self.single:
            if "SpecialToken" in item:
                token = item["SpecialToken"]["id"]
                if token not in self.special_ids:
                    raise ValueError(f"Unknown special token in template: {token}")
                special_ids = self.special_ids[token]
                out_ids.extend(special_ids)
                mask.extend([1] * len(special_ids))
            elif "Sequence" in item:
                out_ids.extend(ids)
                mask.extend([0] * len(ids))
            else:
                raise NotImplementedError(f"Unsupported template item: {item}")
        return out_ids, mask


class _Decoder:
    """Sequence decoder used by the released MatterChat tokenizer."""

    def __init__(self, cfg: Optional[dict]):
        self.steps = []
        if cfg is None:
            return
        if cfg.get("type") == "Sequence":
            decoders = cfg.get("decoders", [])
        else:
            decoders = [cfg]
        for item in decoders:
            kind = item.get("type")
            if kind == "Replace":
                self.steps.append(
                    ("replace", item["pattern"]["String"], item["content"])
                )
            elif kind == "ByteFallback":
                self.steps.append(("byte_fallback",))
            elif kind == "Fuse":
                self.steps.append(("fuse",))
            elif kind == "Strip":
                self.steps.append(
                    (
                        "strip",
                        item.get("content", " "),
                        item.get("start", 0),
                        item.get("stop", 0),
                    )
                )
            else:
                raise NotImplementedError(f"Unsupported decoder: {kind}")

    def decode(self, tokens: List[str]) -> str:
        if not self.steps:
            return " ".join(tokens)
        parts = list(tokens)
        for step in self.steps:
            if step[0] == "replace":
                pattern, content = step[1], step[2]
                parts = [part.replace(pattern, content) for part in parts]
            elif step[0] == "byte_fallback":
                parts = self._byte_fallback(parts)
            elif step[0] == "fuse":
                parts = ["".join(parts)]
            elif step[0] == "strip":
                content, start, stop = step[1], step[2], step[3]
                stripped = []
                for part in parts:
                    begin = 0
                    remaining = start
                    while remaining > 0 and part[begin:].startswith(content):
                        begin += len(content)
                        remaining -= 1
                    end = len(part)
                    remaining = stop
                    while remaining > 0 and part[:end].endswith(content):
                        end -= len(content)
                        remaining -= 1
                    stripped.append(part[begin:end])
                parts = stripped
        return "".join(parts)

    @staticmethod
    def _byte_fallback(tokens: List[str]) -> List[str]:
        out = []
        pending = bytearray()
        for token in tokens:
            match = _BYTE_TOKEN_RE.match(token)
            if match:
                pending.append(int(match.group(1), 16))
                continue
            if pending:
                out.append(bytes(pending).decode("utf-8", errors="replace"))
                pending = bytearray()
            out.append(token)
        if pending:
            out.append(bytes(pending).decode("utf-8", errors="replace"))
        return out


class MistralTokenizerWrapper:
    """Tokenizer API shared by MatterChat training and inference."""

    VOCAB_SIZE = 32768
    PAD_TOKEN_ID = 32768

    def __init__(self, tokenizer_path: str):
        if not os.path.exists(tokenizer_path):
            raise FileNotFoundError(f"Tokenizer file not found: {tokenizer_path}")

        with open(tokenizer_path, "r", encoding="utf-8") as handle:
            self._config = json.load(handle)

        model_cfg = self._config["model"]
        model_type = model_cfg.get("type")
        if model_type == "BPE":
            self.model = _BPEModel(model_cfg)
        elif model_type == "WordLevel":
            self.model = _WordLevelModel(model_cfg)
        else:
            raise NotImplementedError(f"Unsupported tokenizer model: {model_type}")

        if self._config.get("normalizer") is not None:
            raise NotImplementedError("Only null normalizers are supported.")

        pre_cfg = self._config.get("pre_tokenizer")
        if pre_cfg is None:
            raise NotImplementedError("A pre_tokenizer is required.")
        if pre_cfg.get("type") == "Metaspace":
            self.pre_tokenizer = _MetaspacePreTokenizer(pre_cfg)
        elif pre_cfg.get("type") == "Whitespace":
            self.pre_tokenizer = _WhitespacePreTokenizer()
        else:
            raise NotImplementedError(
                f"Unsupported pre_tokenizer: {pre_cfg.get('type')}"
            )

        self.post_processor = _TemplateProcessor(self._config.get("post_processor"))
        self.decoder = _Decoder(self._config.get("decoder"))

        added = self._config.get("added_tokens", [])
        for entry in added:
            if entry.get("single_word") or entry.get("lstrip") or entry.get("rstrip"):
                raise NotImplementedError(
                    "Added tokens with single_word/lstrip/rstrip are not supported."
                )
        self.added_ids = {entry["content"]: entry["id"] for entry in added}
        self.special_ids = {
            entry["id"] for entry in added if entry.get("special", False)
        }
        self._added_matcher = _Trie(list(self.added_ids))

        self.padding_side = "right"
        self.truncation_side = "right"

        self.bos_token = "<s>"
        self.eos_token = "</s>"
        self.pad_token = "[PAD]"

        self._id_to_token_cache: Dict[int, str] = {}

    def _encode_text(self, text: str, add_special_tokens: bool):
        ids: List[int] = []
        special_mask: List[int] = []
        for chunk_index, (is_added, chunk) in enumerate(
            self._added_matcher.split(text)
        ):
            if is_added:
                token_id = self.added_ids[chunk]
                ids.append(token_id)
                special_mask.append(1 if token_id in self.special_ids else 0)
                continue
            for piece in self.pre_tokenizer.pre_tokenize(chunk, chunk_index == 0):
                for token_id in self.model.tokenize(piece):
                    if token_id is None:
                        continue
                    ids.append(token_id)
                    special_mask.append(0)
        if add_special_tokens:
            return self.post_processor.process(ids)
        return ids, special_mask

    @staticmethod
    def _truncate(ids: List[int], mask: List[int], max_length: int, direction: str):
        if len(ids) <= max_length:
            return ids, mask
        if direction == "right":
            return ids[:max_length], mask[:max_length]
        reserve = sum(mask) if max_length > sum(mask) else max_length
        special_index = [i for i, flag in enumerate(mask) if flag]
        kept = special_index[:reserve]
        budget = max_length - len(kept)
        if budget > 0:
            non_special = [i for i, flag in enumerate(mask) if not flag]
            kept = sorted(kept + non_special[-budget:])
        else:
            kept = kept[:max_length]
        return [ids[i] for i in kept], [mask[i] for i in kept]

    def __call__(
        self,
        text: Union[str, List[str]],
        padding: Union[bool, str] = False,
        truncation: bool = False,
        max_length: Optional[int] = None,
        add_special_tokens: bool = True,
    ) -> EncodingResult:
        """Encode text to token IDs and attention mask."""
        if not isinstance(text, list):
            text = [text]

        encoded = []
        for item in text:
            ids, special_mask = self._encode_text(item, add_special_tokens)
            if truncation and max_length is not None:
                ids, special_mask = self._truncate(
                    ids, special_mask, max_length, self.truncation_side
                )
            encoded.append((ids, [1] * len(ids)))

        if padding and encoded:
            width = max(len(ids) for ids, _ in encoded)
            padded_ids, padded_mask = [], []
            for ids, mask in encoded:
                fill = width - len(ids)
                pad_ids = [self.PAD_TOKEN_ID] * fill
                pad_mask = [0] * fill
                if self.padding_side == "left":
                    padded_ids.append(pad_ids + ids)
                    padded_mask.append(pad_mask + mask)
                else:
                    padded_ids.append(ids + pad_ids)
                    padded_mask.append(mask + pad_mask)
        else:
            padded_ids = [ids for ids, _ in encoded]
            padded_mask = [mask for _, mask in encoded]
            width = max((len(ids) for ids in padded_ids), default=0)

        input_ids = paddle.to_tensor(
            np.array(padded_ids, dtype=np.int64).reshape(len(padded_ids), width),
            dtype="int64",
        )
        attention_mask = paddle.to_tensor(
            np.array(padded_mask, dtype=np.int64).reshape(len(padded_mask), width),
            dtype="int64",
        )
        return EncodingResult(input_ids=input_ids, attention_mask=attention_mask)

    def decode(
        self,
        token_ids: Union[List[int], np.ndarray, paddle.Tensor],
        skip_special_tokens: bool = True,
    ) -> str:
        """Decode token IDs back to text."""
        if isinstance(token_ids, paddle.Tensor):
            token_ids = token_ids.numpy().tolist()
        elif isinstance(token_ids, np.ndarray):
            token_ids = token_ids.tolist()
        if token_ids and isinstance(token_ids[0], list):
            token_ids = token_ids[0]

        # The added pad token is outside the tokenizer vocabulary.
        ids = [int(t) for t in token_ids]
        ids = [t for t in ids if t < self.VOCAB_SIZE]
        if skip_special_tokens:
            ids = [t for t in ids if t not in self.special_ids]

        tokens = [token for token in (self._id_to_token(i) for i in ids) if token]
        return self.decoder.decode(tokens)

    def _id_to_token(self, token_id: int) -> Optional[str]:
        if not self._id_to_token_cache:
            for token, index in self.model.vocab.items():
                self._id_to_token_cache.setdefault(index, token)
        return self._id_to_token_cache.get(token_id)

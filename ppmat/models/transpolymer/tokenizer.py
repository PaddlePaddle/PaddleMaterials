import json
import os
import shutil
import urllib.request
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import numpy as np
import regex as re


VOCAB_FILES_NAMES = {
    "vocab_file": "vocab.json",
    "merges_file": "merges.txt",
}

PRETRAINED_VOCAB_FILES_MAP = {
    "roberta-base": {
        "vocab_file": [
            "https://huggingface.co/roberta-base/resolve/main/vocab.json",
            "https://hf-mirror.com/roberta-base/resolve/main/vocab.json",
        ],
        "merges_file": [
            "https://huggingface.co/roberta-base/resolve/main/merges.txt",
            "https://hf-mirror.com/roberta-base/resolve/main/merges.txt",
        ],
    }
}


@lru_cache()
def bytes_to_unicode():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("隆"), ord("卢") + 1)) + list(range(ord("庐"), ord("每") + 1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


def get_pairs(word):
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


class PolymerSmilesTokenizer:
    """Standalone Paddle-friendly port of the original TransPolymer tokenizer.

    The SMILES-aware regex and byte-level BPE logic are intentionally kept in sync
    with the PyTorch/HuggingFace implementation used by the official project.
    """

    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file,
        merges_file,
        errors="replace",
        bos_token="<s>",
        eos_token="</s>",
        sep_token="</s>",
        cls_token="<s>",
        unk_token="<unk>",
        pad_token="<pad>",
        mask_token="<mask>",
        add_prefix_space=False,
        max_len=None,
        **kwargs,
    ):
        with open(vocab_file, encoding="utf-8") as vocab_handle:
            self.encoder: Dict[str, int] = json.load(vocab_handle)
        self.decoder = {v: k for k, v in self.encoder.items()}

        with open(merges_file, encoding="utf-8") as merges_handle:
            bpe_merges = merges_handle.read().split("\n")[1:-1]
        bpe_merges = [tuple(merge.split()) for merge in bpe_merges]
        self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))

        self.errors = errors
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        self.cache = {}
        self.add_prefix_space = add_prefix_space
        self.max_len = max_len

        self.bos_token = bos_token
        self.eos_token = eos_token
        self.sep_token = sep_token
        self.cls_token = cls_token
        self.unk_token = unk_token
        self.pad_token = pad_token
        self.mask_token = mask_token

        self.added_tokens_encoder: Dict[str, int] = {}
        self.added_tokens_decoder: Dict[int, str] = {}
        self._added_tokens_pattern = None

        smi_regex_pattern = r"(\-?[0-9]+\.?[0-9]*|\[|\]|SELF|Li|Be|Na|Mg|Al|K|Ca|Co|Zn|Ga|Ge|As|Se|Sn|Te|N|O|P|H|I|b|c|n|o|s|p|Br?|Cl?|Fe?|Ni?|Si?|\||\(|\)|\^|=|#|-|\+|\\|\/|@|\*|\.|\%|\$)"
        self.pat = re.compile(smi_regex_pattern)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, max_len=None, **kwargs):
        if os.path.isdir(pretrained_model_name_or_path):
            vocab_file = os.path.join(pretrained_model_name_or_path, VOCAB_FILES_NAMES["vocab_file"])
            merges_file = os.path.join(pretrained_model_name_or_path, VOCAB_FILES_NAMES["merges_file"])
        elif pretrained_model_name_or_path in PRETRAINED_VOCAB_FILES_MAP:
            cache_dir = kwargs.pop(
                "cache_dir",
                os.path.join(os.path.expanduser("~"), ".cache", "transpolymer", pretrained_model_name_or_path),
            )
            os.makedirs(cache_dir, exist_ok=True)
            vocab_file = os.path.join(cache_dir, VOCAB_FILES_NAMES["vocab_file"])
            merges_file = os.path.join(cache_dir, VOCAB_FILES_NAMES["merges_file"])
            for key, path in (("vocab_file", vocab_file), ("merges_file", merges_file)):
                if not os.path.exists(path):
                    cls._download_with_fallback(
                        PRETRAINED_VOCAB_FILES_MAP[pretrained_model_name_or_path][key],
                        path,
                    )
        else:
            raise ValueError(f"Cannot locate tokenizer files from {pretrained_model_name_or_path!r}")

        tokenizer = cls(vocab_file, merges_file, max_len=max_len, **kwargs)
        added_tokens_file = os.path.join(pretrained_model_name_or_path, "added_tokens.json")
        if os.path.isdir(pretrained_model_name_or_path) and os.path.exists(added_tokens_file):
            with open(added_tokens_file, "r", encoding="utf-8") as f:
                tokenizer.add_tokens(json.load(f))
        return tokenizer

    @staticmethod
    def _download_with_fallback(urls, output_path):
        """Download an artifact atomically, trying official and mirror endpoints."""
        errors = []
        temporary_path = output_path + ".tmp"
        for url in urls:
            try:
                with urllib.request.urlopen(url, timeout=30) as response:
                    with open(temporary_path, "wb") as output_file:
                        shutil.copyfileobj(response, output_file)
                os.replace(temporary_path, output_path)
                return
            except Exception as error:
                errors.append(f"{url}: {error}")
                if os.path.exists(temporary_path):
                    os.remove(temporary_path)
        raise OSError(
            "Unable to download tokenizer file. Tried: " + "; ".join(errors)
        )

    @property
    def vocab_size(self):
        return len(self.encoder)

    @property
    def cls_token_id(self):
        return self.convert_tokens_to_ids(self.cls_token)

    @property
    def sep_token_id(self):
        return self.convert_tokens_to_ids(self.sep_token)

    @property
    def pad_token_id(self):
        return self.convert_tokens_to_ids(self.pad_token)

    @property
    def mask_token_id(self):
        return self.convert_tokens_to_ids(self.mask_token)

    @property
    def unk_token_id(self):
        return self.convert_tokens_to_ids(self.unk_token)

    def __len__(self):
        return len(self.encoder) + len(self.added_tokens_encoder)

    def get_vocab(self):
        vocab = dict(self.encoder)
        vocab.update(self.added_tokens_encoder)
        return vocab

    def add_tokens(self, new_tokens):
        if isinstance(new_tokens, str):
            new_tokens = [new_tokens]
        added = 0
        for token in new_tokens:
            if token is None:
                continue
            token = str(token)
            if token in self.encoder or token in self.added_tokens_encoder:
                continue
            index = len(self.encoder) + len(self.added_tokens_encoder)
            self.added_tokens_encoder[token] = index
            self.added_tokens_decoder[index] = token
            added += 1
        if added:
            self._refresh_added_tokens_pattern()
        return added

    def _refresh_added_tokens_pattern(self):
        tokens = sorted(self.added_tokens_encoder, key=len, reverse=True)
        self._added_tokens_pattern = re.compile("|".join(re.escape(token) for token in tokens)) if tokens else None

    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token)
        pairs = get_pairs(word)
        if not pairs:
            return token

        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                except ValueError:
                    new_word.extend(word[i:])
                    break
                new_word.extend(word[i:j])
                i = j
                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = tuple(new_word)
            if len(word) == 1:
                break
            pairs = get_pairs(word)
        word = " ".join(word)
        self.cache[token] = word
        return word

    def _tokenize(self, text):
        bpe_tokens = []
        for token in re.findall(self.pat, text):
            token = "".join(self.byte_encoder[b] for b in token.encode("utf-8"))
            bpe_tokens.extend(bpe_token for bpe_token in self.bpe(token).split(" "))
        return bpe_tokens

    def tokenize(self, text):
        if not self._added_tokens_pattern:
            return self._tokenize(text)

        tokens = []
        last_end = 0
        for match in self._added_tokens_pattern.finditer(text):
            if match.start() > last_end:
                tokens.extend(self._tokenize(text[last_end : match.start()]))
            tokens.append(match.group(0))
            last_end = match.end()
        if last_end < len(text):
            tokens.extend(self._tokenize(text[last_end:]))
        return tokens

    def convert_tokens_to_ids(self, tokens):
        if isinstance(tokens, list):
            return [self.convert_tokens_to_ids(token) for token in tokens]
        if tokens in self.added_tokens_encoder:
            return self.added_tokens_encoder[tokens]
        return self.encoder.get(tokens, self.encoder.get(self.unk_token))

    def convert_ids_to_tokens(self, ids):
        if isinstance(ids, list):
            return [self.convert_ids_to_tokens(index) for index in ids]
        return self.added_tokens_decoder.get(ids, self.decoder.get(ids, self.unk_token))

    def convert_tokens_to_string(self, tokens):
        text = "".join(tokens)
        return bytearray([self.byte_decoder[c] for c in text]).decode("utf-8", errors=self.errors)

    def build_inputs_with_special_tokens(self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None):
        if token_ids_1 is None:
            return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
        return [self.cls_token_id] + token_ids_0 + [self.sep_token_id, self.sep_token_id] + token_ids_1 + [self.sep_token_id]

    def create_token_type_ids_from_sequences(self, token_ids_0, token_ids_1=None):
        return [0] * len(self.build_inputs_with_special_tokens(token_ids_0, token_ids_1))

    def get_special_tokens_mask(self, token_ids_0, token_ids_1=None, already_has_special_tokens=False):
        if already_has_special_tokens:
            return [1 if token_id in {self.cls_token_id, self.sep_token_id, self.pad_token_id} else 0 for token_id in token_ids_0]
        if token_ids_1 is None:
            return [1] + ([0] * len(token_ids_0)) + [1]
        return [1] + ([0] * len(token_ids_0)) + [1, 1] + ([0] * len(token_ids_1)) + [1]

    def encode(self, text, add_special_tokens=True, max_length=None, truncation=False):
        token_ids = self.convert_tokens_to_ids(self.tokenize(text))
        if max_length is None:
            max_length = self.max_len
        if add_special_tokens and max_length is not None and truncation:
            token_ids = token_ids[: max(0, max_length - 2)]
        elif max_length is not None and truncation:
            token_ids = token_ids[:max_length]
        if add_special_tokens:
            token_ids = self.build_inputs_with_special_tokens(token_ids)
        return token_ids

    def __call__(
        self,
        text,
        add_special_tokens=True,
        max_length=None,
        return_token_type_ids=False,
        padding=False,
        truncation=False,
        return_attention_mask=True,
        return_tensors=None,
        **kwargs,
    ):
        if max_length is None:
            max_length = self.max_len
        input_ids = self.encode(text, add_special_tokens=add_special_tokens, max_length=max_length, truncation=truncation)
        attention_mask = [1] * len(input_ids)

        if padding == "max_length" and max_length is not None:
            pad_len = max_length - len(input_ids)
            if pad_len > 0:
                input_ids = input_ids + [self.pad_token_id] * pad_len
                attention_mask = attention_mask + [0] * pad_len
            elif pad_len < 0 and truncation:
                input_ids = input_ids[:max_length]
                attention_mask = attention_mask[:max_length]

        output = {"input_ids": input_ids}
        if return_attention_mask:
            output["attention_mask"] = attention_mask
        if return_token_type_ids:
            output["token_type_ids"] = [0] * len(input_ids)

        if return_tensors is not None:
            arrays = {k: np.asarray([v], dtype="int64") for k, v in output.items()}
            if return_tensors in ("np", "numpy"):
                return arrays
            if return_tensors == "pd":
                import paddle

                return {k: paddle.to_tensor(v, dtype="int64") for k, v in arrays.items()}
            if return_tensors == "pt":
                import torch

                return {k: torch.tensor(v, dtype=torch.long) for k, v in arrays.items()}
            raise ValueError(f"Unsupported return_tensors={return_tensors!r}")
        return output

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str, str]:
        os.makedirs(save_directory, exist_ok=True)
        vocab_file = os.path.join(save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"])
        merge_file = os.path.join(save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["merges_file"])
        with open(vocab_file, "w", encoding="utf-8") as f:
            json.dump(self.encoder, f, ensure_ascii=False)
        with open(merge_file, "w", encoding="utf-8") as writer:
            writer.write("#version: 0.2\n")
            for bpe_tokens, _ in sorted(self.bpe_ranks.items(), key=lambda kv: kv[1]):
                writer.write(" ".join(bpe_tokens) + "\n")
        if self.added_tokens_encoder:
            with open(os.path.join(save_directory, "added_tokens.json"), "w", encoding="utf-8") as f:
                json.dump(list(self.added_tokens_encoder.keys()), f, ensure_ascii=False, indent=2)
        return vocab_file, merge_file

    def save_pretrained(self, save_directory):
        return self.save_vocabulary(save_directory)

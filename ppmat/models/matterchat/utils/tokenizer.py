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

"""Tokenizer wrapper for MatterChat inference."""

import os
from dataclasses import dataclass
from typing import List
from typing import Optional
from typing import Union

import numpy as np
import paddle
from tokenizers import Tokenizer


@dataclass
class EncodingResult:
    """Mimics HuggingFace BatchEncoding for compatibility."""

    input_ids: paddle.Tensor
    attention_mask: paddle.Tensor


class MistralTokenizerWrapper:
    """Tokenizer API shared by MatterChat training and inference."""

    VOCAB_SIZE = 32768
    PAD_TOKEN_ID = 32768

    def __init__(self, tokenizer_path: str):
        if not os.path.exists(tokenizer_path):
            raise FileNotFoundError(f"Tokenizer file not found: {tokenizer_path}")

        self._tokenizer = Tokenizer.from_file(tokenizer_path)
        self._tokenizer.enable_padding(direction="right")
        self.padding_side = "right"
        self.truncation_side = "right"

        self.bos_token_id = 1
        self.eos_token_id = 2
        self.pad_token_id = self.PAD_TOKEN_ID

        self.bos_token = "<s>"
        self.eos_token = "</s>"
        self.pad_token = "[PAD]"

    def __call__(
        self,
        text: Union[str, List[str]],
        return_tensors: str = "pd",
        padding: Union[bool, str] = False,
        truncation: bool = False,
        max_length: Optional[int] = None,
        add_special_tokens: bool = True,
        **kwargs,
    ) -> EncodingResult:
        """Encode text to token IDs and attention mask."""
        is_batched = isinstance(text, list)
        if not is_batched:
            text = [text]

        if truncation and max_length is not None:
            self._tokenizer.enable_truncation(
                max_length, direction=self.truncation_side
            )
        else:
            self._tokenizer.no_truncation()

        if padding and padding != "do_not_pad":
            dir_map = {"right": "right", "left": "left"}
            direction = dir_map.get(self.padding_side, "right")
            self._tokenizer.enable_padding(
                direction=direction,
                pad_id=self.PAD_TOKEN_ID,
                pad_token=self.pad_token,
            )
        else:
            self._tokenizer.no_padding()

        encodings = self._tokenizer.encode_batch(
            text, add_special_tokens=add_special_tokens
        )

        all_ids = []
        all_masks = []
        for enc in encodings:
            all_ids.append(enc.ids)
            all_masks.append(enc.attention_mask)

        input_ids = paddle.to_tensor(np.array(all_ids, dtype=np.int64))
        attention_mask = paddle.to_tensor(np.array(all_masks, dtype=np.int64))

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

        # Filter out the added pad token (32768) which the base tokenizer doesn't know
        token_ids = [t for t in token_ids if t < self.VOCAB_SIZE]

        return self._tokenizer.decode(
            token_ids, skip_special_tokens=skip_special_tokens
        )

    def batch_decode(
        self,
        sequences: Union[List, paddle.Tensor],
        skip_special_tokens: bool = True,
    ) -> List[str]:
        """Decode a batch of token ID sequences."""
        if isinstance(sequences, paddle.Tensor):
            sequences = sequences.numpy().tolist()

        results = []
        for seq in sequences:
            if isinstance(seq, (list, np.ndarray)):
                token_ids = list(seq)
            else:
                token_ids = [int(seq)]
            token_ids = [t for t in token_ids if t < self.VOCAB_SIZE]
            results.append(
                self._tokenizer.decode(
                    token_ids, skip_special_tokens=skip_special_tokens
                )
            )
        return results

    def get_vocab_size(self) -> int:
        return self.VOCAB_SIZE + 1

    @property
    def vocab_size(self) -> int:
        return self.VOCAB_SIZE + 1

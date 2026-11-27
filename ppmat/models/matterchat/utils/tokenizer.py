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

"""Tokenizer wrapper for MatterChat training and inference."""

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
    """Token ids and attention mask returned by the wrapper."""

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
        self.padding_side = "right"
        self.truncation_side = "right"

        self.bos_token = "<s>"
        self.eos_token = "</s>"
        self.pad_token = "[PAD]"

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

        if truncation and max_length is not None:
            self._tokenizer.enable_truncation(
                max_length, direction=self.truncation_side
            )
        else:
            self._tokenizer.no_truncation()

        if padding:
            self._tokenizer.enable_padding(
                direction=self.padding_side,
                pad_id=self.PAD_TOKEN_ID,
                pad_token=self.pad_token,
            )
        else:
            self._tokenizer.no_padding()

        encodings = self._tokenizer.encode_batch(
            text, add_special_tokens=add_special_tokens
        )
        input_ids = paddle.to_tensor(
            np.array([enc.ids for enc in encodings], dtype=np.int64)
        )
        attention_mask = paddle.to_tensor(
            np.array([enc.attention_mask for enc in encodings], dtype=np.int64)
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
        token_ids = [t for t in token_ids if t < self.VOCAB_SIZE]
        return self._tokenizer.decode(
            token_ids, skip_special_tokens=skip_special_tokens
        )

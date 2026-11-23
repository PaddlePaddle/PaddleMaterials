# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import annotations

import functools
import json

from ppmat.utils import download

VOCAB_REGISTRY = {
    "infgcn_md17": "https://paddle-org.bj.bcebos.com/paddlematerials/assets/vocabs/infgcn_md17.json",
    "infgcn_qm9": "https://paddle-org.bj.bcebos.com/paddlematerials/assets/vocabs/infgcn_qm9.json",
    "infgcn_mp": "https://paddle-org.bj.bcebos.com/paddlematerials/assets/vocabs/infgcn_mp.json",
    "infgcn_omol25": "https://paddle-org.bj.bcebos.com/paddlematerials/assets/vocabs/infgcn_omol25.json",
    "diffnmr_msdnmr_nless15": "https://paddle-org.bj.bcebos.com/paddlematerials/assets/vocabs/diffnmr_msdnmr_nless15.json",
}

VOCAB_MD5_REGISTRY = {
    "infgcn_md17": "1bcbc1e4e67c3da8e59ff8eeb49380eb",
    "infgcn_qm9": "ec80669b0653c6c03d9019481d8a53e0",
    "infgcn_mp": "43be5eaad8204401be0264ca513d78f2",
    "infgcn_omol25": "e72c2df22f3dc79ea9c62b9d7418180b",
    "diffnmr_msdnmr_nless15": "1a7e77ea68380ed44ae2dc16b544ec50",
}


def build_vocab(
    name: str | None,
) -> dict | None:
    """Build a registered vocabulary package."""

    if name is None:
        return None
    return _build_vocab_from_name(name)


@functools.lru_cache(maxsize=None)
def _build_vocab_from_name(name: str) -> dict:
    path = download.get_datasets_path_from_url(
        VOCAB_REGISTRY[name],
        VOCAB_MD5_REGISTRY[name],
    )
    with open(path, encoding="utf-8") as file_obj:
        vocabularies = json.load(file_obj)

    result = {}
    for role, vocab_cfg in vocabularies.items():
        vocab_cfg = dict(vocab_cfg)
        tokens = vocab_cfg["tokens"]
        if vocab_cfg["type"] == "element":
            symbols = [token["value"] for token in tokens]
            atomic_numbers = [int(token["atomic_number"]) for token in tokens]
            vocab_cfg["tokens"] = symbols
            vocab_cfg["token_to_id"] = {
                symbol: index for index, symbol in enumerate(symbols)
            }
            vocab_cfg["id_to_token"] = dict(enumerate(symbols))
            vocab_cfg["atomic_number_to_id"] = {
                atomic_number: index
                for index, atomic_number in enumerate(atomic_numbers)
            }
            vocab_cfg["id_to_atomic_number"] = dict(enumerate(atomic_numbers))
        else:
            vocab_cfg["token_to_id"] = {
                token: index for index, token in enumerate(tokens)
            }
            vocab_cfg["id_to_token"] = dict(enumerate(tokens))
        result[role] = vocab_cfg
    return result

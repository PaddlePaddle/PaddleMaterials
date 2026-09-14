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

"""Generic read-only LMDB utilities.

Provides the common read-only LMDB access patterns shared by datasets:

- ``open_lmdb``: open an LMDB environment with safe read-only defaults;
- ``lmdb_keys``: list all keys, optionally filtering metadata keys;
- ``lmdb_get``: fetch a single record by key;
- ``decode_payload``: multi-format payload decoding (zlib -> pickle ->
  json -> ast).

Only the read path is abstracted; writing/building LMDB files stays with the
code that owns the data format, and format-specific value decoding (e.g. an
``__ndarray__`` payload convention) stays in the consuming dataset.
"""

import ast
import json
import os
import pickle
import zlib
from typing import Any
from typing import Callable
from typing import List
from typing import Optional

import lmdb


def open_lmdb(
    file_path: str,
    *,
    subdir: Optional[bool] = None,
    max_readers: int = 126,
) -> lmdb.Environment:
    """Open an LMDB environment read-only with memory-safe defaults.

    Args:
        file_path: Path to the LMDB file or directory.
        subdir: Whether ``file_path`` is a directory. When None (default),
            it is auto-detected from the path.
        max_readers: Maximum number of concurrent readers.

    Returns:
        A read-only :class:`lmdb.Environment`.
    """
    if subdir is None:
        subdir = os.path.isdir(file_path)
    return lmdb.open(
        file_path,
        subdir=subdir,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=max_readers,
    )


def lmdb_keys(
    env: lmdb.Environment,
    *,
    skip_meta: bool = True,
) -> List[str]:
    """List all keys in the environment.

    Args:
        env: An open LMDB environment.
        skip_meta: Whether to skip metadata keys (starting with "__").

    Returns:
        List of keys (decoded to str).
    """
    with env.begin() as txn:
        all_keys = [
            k.decode("ascii") if isinstance(k, bytes) else k
            for k in txn.cursor().iternext(values=False)
        ]
    if skip_meta:
        all_keys = [k for k in all_keys if not k.startswith("__")]
    return all_keys


def lmdb_get(
    env: lmdb.Environment, key: str, *, decoder: Optional[Callable] = None
) -> Any:
    """Fetch the record stored under ``key``.

    Args:
        env: An open LMDB environment.
        key: Record key.
        decoder: Optional payload decoder; defaults to ``decode_payload``.

    Returns:
        The deserialized object.

    Raises:
        KeyError: If ``key`` does not exist in the environment.
    """
    key_bytes = key.encode("ascii") if isinstance(key, str) else key
    with env.begin() as txn:
        value = txn.get(key_bytes)
    if value is None:
        raise KeyError(f"Key {key} not found in LMDB.")
    decoder = decoder or decode_payload
    return decoder(value)


def decode_payload(payload: bytes) -> Any:
    """Decode a raw LMDB payload with progressive formats.

    Tries, in order: zlib decompress -> pickle -> json -> ast literal. Returns
    the first successful decode; raises ``ValueError`` if all fail.
    """
    raw = payload
    try:
        raw = zlib.decompress(payload)
    except Exception:
        pass
    try:
        return pickle.loads(raw)
    except Exception:
        pass
    try:
        return json.loads(raw.decode("utf-8"))
    except Exception:
        pass
    try:
        return ast.literal_eval(raw.decode("utf-8"))
    except Exception:
        raise ValueError("Failed to decode LMDB payload with zlib/pickle/json/ast.")

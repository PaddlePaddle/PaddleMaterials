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

"""Tie a dataset cache to the raw data it was built from.

Dataset classes cache the structures, molecules and graphs they build under
``cache_path`` and, on later runs, reuse them when the build settings match. The
settings alone do not identify the data: pointing a dataset at a different file while
``cache_path`` still holds another dataset's cache silently pairs the new labels with
the old graphs. The helpers below store a hash of the cached inputs next to the cache
and check it before reuse.

Usage in a dataset (rank 0 builds the cache, the other ranks wait at the barrier)::

    fingerprint = compute_data_fingerprint(raw_inputs)
    if cache_exists and not overwrite:
        if not cache_matches_data(cache_path, fingerprint, ["structures", "graphs"]):
            overwrite = True
    if overwrite or not cache_exists:
        if dist.get_rank() == 0:
            mark_cache_rebuilding(cache_path)
            ...  # build and save the cache
        if dist.is_initialized():
            dist.barrier()
    if dist.get_rank() == 0:
        save_data_fingerprint(cache_path, fingerprint)
"""

from __future__ import annotations

import hashlib
import json
import os
import os.path as osp
import pickle
from typing import Iterable
from typing import Optional

from ppmat.utils import logger

FINGERPRINT_FILE = "data_fingerprint.json"
FINGERPRINT_VERSION = 1


def _payload(item) -> bytes:
    if isinstance(item, str):
        return item.encode("utf-8")
    if isinstance(item, bytes):
        return item
    return pickle.dumps(item, protocol=4)


def compute_data_fingerprint(items: Iterable, extra=None) -> dict:
    """Hash the raw inputs that are converted into cached objects, in cache order.

    Args:
        items: The per-sample raw inputs (CIF strings, structure dicts, molecule
            records, ...), in the order they are written to the cache.
        extra: Optional object that is merged into every cached sample (for example
            supplementary per-atom data). It is hashed once after the samples.

    Returns:
        dict: ``{"version", "num_samples", "sha256"}``.
    """
    digest = hashlib.sha256()
    num_samples = 0
    for item in items:
        payload = _payload(item)
        digest.update(len(payload).to_bytes(8, "little"))
        digest.update(payload)
        num_samples += 1
    if extra is not None:
        payload = _payload(extra)
        digest.update(b"extra")
        digest.update(len(payload).to_bytes(8, "little"))
        digest.update(payload)
    return {
        "version": FINGERPRINT_VERSION,
        "num_samples": num_samples,
        "sha256": digest.hexdigest(),
    }


def _count_cached_files(cache_dir: str) -> Optional[int]:
    if not osp.isdir(cache_dir):
        return None
    return sum(1 for name in os.listdir(cache_dir) if name.endswith(".pkl"))


def cache_matches_data(
    cache_path: str, fingerprint: dict, sample_dirs: Iterable[str]
) -> bool:
    """Return whether the cache under ``cache_path`` was built from the given data.

    A cache with a stored fingerprint is reused only if the fingerprint is identical.
    A cache written before fingerprints existed is reused only if every sample directory
    holds exactly ``fingerprint["num_samples"]`` files. The caller then records the
    current fingerprint with :func:`save_data_fingerprint`, so later changes of the
    data are detected. This function does not write anything.

    Args:
        cache_path: Root directory of the cache.
        fingerprint: Result of :func:`compute_data_fingerprint` for the current data.
        sample_dirs: Sub-directories (relative to ``cache_path``) holding one ``.pkl``
            file per sample, e.g. ``["structures", "graphs"]``.

    Returns:
        bool: ``True`` if the cache can be reused, ``False`` if it must be rebuilt.
    """
    fingerprint_path = osp.join(cache_path, FINGERPRINT_FILE)
    if osp.exists(fingerprint_path):
        try:
            with open(fingerprint_path, "r") as f:
                stored = json.load(f)
        except (OSError, ValueError) as e:
            logger.warning(
                f"Unreadable cache fingerprint {fingerprint_path} ({e}). "
                "Will rebuild the cache."
            )
            return False
        if not isinstance(stored, dict):
            stored = {}
        if stored == fingerprint:
            return True
        if stored.get("status") == "building":
            logger.warning(
                f"The cache in {cache_path} was not completely built. "
                "Will rebuild the cache."
            )
            return False
        logger.warning(
            f"The cache in {cache_path} was built from different data "
            f"({stored.get('num_samples')} samples, sha256 "
            f"{str(stored.get('sha256'))[:12]}) than the current input "
            f"({fingerprint['num_samples']} samples, sha256 "
            f"{fingerprint['sha256'][:12]}). Will rebuild the cache."
        )
        return False

    expected = fingerprint["num_samples"]
    for sub_dir in sample_dirs:
        count = _count_cached_files(osp.join(cache_path, sub_dir))
        if count != expected:
            logger.warning(
                f"The cache in {cache_path} has no data fingerprint and "
                f"'{sub_dir}' holds {count} files for {expected} samples. "
                "Will rebuild the cache."
            )
            return False
    logger.warning(
        f"The cache in {cache_path} has no data fingerprint (it was written by an "
        "older version). Its sample count matches the current input, so it is reused "
        "and the current fingerprint is recorded. If the input data changed without "
        "changing the number of samples, delete the cache or set overwrite=True."
    )
    return True


def save_data_fingerprint(cache_path: str, fingerprint: dict) -> None:
    """Write the fingerprint of the data a cache was built from."""
    os.makedirs(cache_path, exist_ok=True)
    tmp_path = osp.join(cache_path, FINGERPRINT_FILE + ".tmp")
    with open(tmp_path, "w") as f:
        json.dump(fingerprint, f, indent=1)
    os.replace(tmp_path, osp.join(cache_path, FINGERPRINT_FILE))


def mark_cache_rebuilding(cache_path: str) -> None:
    """Replace the stored fingerprint with a marker before the cache is rebuilt.

    The marker never matches real data, so a rebuild that is interrupted leaves a cache
    that is rebuilt again on the next run instead of being reused half-written.
    """
    save_data_fingerprint(
        cache_path, {"version": FINGERPRINT_VERSION, "status": "building"}
    )

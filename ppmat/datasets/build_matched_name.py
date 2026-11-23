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

import copy
import os.path as osp
from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence
from typing import Union


def build_matched_name_samples(cfg: Dict):
    """Build sample matcher from config."""
    if cfg is None:
        raise ValueError("Sample matcher config must not be None.")

    cfg = copy.deepcopy(cfg)

    match_mode = cfg.pop("match_mode", "indexed")
    if match_mode in ("indexed", "index", "BuildIndexedNameSamples"):
        return BuildIndexedNameSamples(**cfg)
    if match_mode in ("matched", "same_name", "BuildMatchedNameSamples"):
        return BuildMatchedNameSamples(**cfg)

    raise ValueError(
        f"Unsupported match_mode: {match_mode}. "
        "Expected one of {'indexed', 'matched'}."
    )


def build_prediction_samples(
    noisy_files: Sequence[str],
    noisy_key: str = "noisy",
    name_key: str = "name",
) -> List[Dict[str, str]]:
    return [{noisy_key: file_name, name_key: file_name} for file_name in noisy_files]


def _pair_files_by_name(
    noisy_files: Sequence[str],
    target_files: Sequence[str],
    noisy_file_key: str,
    target_file_key: str,
) -> List[Dict[str, str]]:
    """Pair noisy and target files by exactly matched file names.

    This helper only prepares file-name pairs for the sample builder. It does
    not read image contents or depend on a dataset instance.
    """
    noisy_file_set = set(noisy_files)
    target_file_set = set(target_files)
    missing_target = sorted(noisy_file_set - target_file_set)
    missing_noisy = sorted(target_file_set - noisy_file_set)
    if missing_target or missing_noisy:
        raise FileNotFoundError(
            "Noisy and target images are not paired. "
            f"Missing target files: {missing_target[:10]}, "
            f"missing noisy files: {missing_noisy[:10]}."
        )
    return [
        {
            noisy_file_key: file_name,
            target_file_key: file_name,
        }
        for file_name in sorted(noisy_file_set & target_file_set)
    ]


def _collect_indexed_file_map(
    file_names: Sequence[str],
    root: Optional[str],
    file_suffix: Optional[str],
) -> Dict[int, str]:
    """Validate strict integer file stems and map index to file name.

    Files such as ``0.png`` and ``00.png`` are treated as the same integer
    index. Invalid stems and duplicate integer indices are rejected here so the
    builder can fail before producing ambiguous noisy/target pairs.
    """
    index_map = {}
    invalid_files = []
    duplicate_files = []
    for file_name in file_names:
        stem = osp.splitext(file_name)[0]
        if not stem.isdigit():
            invalid_files.append((root, file_name))
            continue
        index = int(stem)
        if index in index_map:
            duplicate_files.append((root, index_map[index], file_name))
            continue
        index_map[index] = file_name

    if invalid_files:
        expected_name = f"0{file_suffix}" if file_suffix is not None else "0.*"
        raise ValueError(
            "Strict indexed naming requires files named like "
            f"'{expected_name}'. Invalid files: {invalid_files[:10]}."
        )
    if duplicate_files:
        raise ValueError(
            "Strict indexed naming requires one file per integer index. "
            f"Duplicate indexed files: {duplicate_files[:10]}."
        )
    return index_map


def _pair_files_by_index(
    noisy_files: Sequence[str],
    target_files: Sequence[str],
    noisy_file_key: str,
    target_file_key: str,
    noisy_root: Optional[str] = None,
    target_root: Optional[str] = None,
    file_suffix: Optional[str] = None,
) -> List[Dict[str, str]]:
    """Pair noisy and target files by integer file stems.

    The returned dictionaries are still passed through ``BuildIndexedNameSamples``
    so the existing per-sample validation remains the final source of truth.
    """
    noisy_map = _collect_indexed_file_map(noisy_files, noisy_root, file_suffix)
    target_map = _collect_indexed_file_map(target_files, target_root, file_suffix)
    common_indices = sorted(set(noisy_map.keys()) & set(target_map.keys()))
    missing_target = sorted(set(noisy_map.keys()) - set(target_map.keys()))
    missing_noisy = sorted(set(target_map.keys()) - set(noisy_map.keys()))
    if missing_target or missing_noisy:
        raise FileNotFoundError(
            "Noisy and target images are not paired. "
            f"Missing target indices: {missing_target[:10]}, "
            f"missing noisy indices: {missing_noisy[:10]}."
        )
    return [
        {
            noisy_file_key: noisy_map[idx],
            target_file_key: target_map[idx],
        }
        for idx in common_indices
    ]


class BuildMatchedNameSamples:
    """Match noisy and target samples by identical file names."""

    def __init__(
        self,
        noisy_file_key: str = "noisy_file",
        target_file_key: str = "target_file",
        file_key: str = "file_name",
        noisy_key: str = "noisy",
        target_key: str = "target",
        name_key: str = "name",
    ):
        self.noisy_file_key = noisy_file_key
        self.target_file_key = target_file_key
        self.file_key = file_key
        self.noisy_key = noisy_key
        self.target_key = target_key
        self.name_key = name_key

    @staticmethod
    def build_one(
        file_data: Union[Dict[str, str], str],
        noisy_file_key: str,
        target_file_key: str,
        file_key: str,
        noisy_key: str,
        target_key: str,
        name_key: str,
    ) -> Dict[str, str]:
        if isinstance(file_data, dict):
            if file_key in file_data:
                noisy_file = file_data.get(file_key)
                target_file = noisy_file
            else:
                noisy_file = file_data.get(noisy_file_key)
                target_file = file_data.get(target_file_key)
        else:
            noisy_file = file_data
            target_file = file_data
        if not isinstance(noisy_file, str) or not noisy_file:
            raise ValueError(
                f"Expected non-empty noisy file name, but got {noisy_file}."
            )
        if not isinstance(target_file, str) or not target_file:
            raise ValueError(
                f"Expected non-empty target file name, but got {target_file}."
            )
        if noisy_file != target_file:
            raise ValueError(
                "Matched-name samples require identical noisy and target file names, "
                f"but got {noisy_file} and {target_file}."
            )
        return {
            noisy_key: noisy_file,
            target_key: target_file,
            name_key: noisy_file,
        }

    def __call__(
        self,
        file_names: Union[
            Sequence[Union[Dict[str, str], str]],
            Dict[str, str],
            str,
        ],
        target_file_names: Optional[Sequence[str]] = None,
        noisy_root: Optional[str] = None,
        target_root: Optional[str] = None,
        file_suffix: Optional[str] = None,
    ) -> Union[List[Dict[str, str]], Dict[str, str]]:
        if target_file_names is not None:
            file_names = _pair_files_by_name(
                file_names,
                target_file_names,
                self.noisy_file_key,
                self.target_file_key,
            )

        if isinstance(file_names, (list, tuple)):
            if len(file_names) == 0:
                return []
            return [
                BuildMatchedNameSamples.build_one(
                    file_name,
                    self.noisy_file_key,
                    self.target_file_key,
                    self.file_key,
                    self.noisy_key,
                    self.target_key,
                    self.name_key,
                )
                for file_name in file_names
            ]
        return BuildMatchedNameSamples.build_one(
            file_names,
            self.noisy_file_key,
            self.target_file_key,
            self.file_key,
            self.noisy_key,
            self.target_key,
            self.name_key,
        )


class BuildIndexedNameSamples:
    """Match noisy and target samples by integer file stem."""

    def __init__(
        self,
        noisy_file_key: str = "noisy_file",
        target_file_key: str = "target_file",
        noisy_key: str = "noisy",
        target_key: str = "target",
        name_key: str = "name",
    ):
        self.noisy_file_key = noisy_file_key
        self.target_file_key = target_file_key
        self.noisy_key = noisy_key
        self.target_key = target_key
        self.name_key = name_key

    @staticmethod
    def build_one(
        sample_data: Dict[str, str],
        noisy_file_key: str,
        target_file_key: str,
        noisy_key: str,
        target_key: str,
        name_key: str,
    ) -> Dict[str, str]:
        if not isinstance(sample_data, dict):
            raise TypeError(
                f"Indexed sample data must be a dict, but got {type(sample_data)}."
            )
        noisy_file = sample_data.get(noisy_file_key)
        target_file = sample_data.get(target_file_key)
        if not isinstance(noisy_file, str) or not noisy_file:
            raise ValueError(
                f"Expected non-empty noisy file name, but got {noisy_file}."
            )
        if not isinstance(target_file, str) or not target_file:
            raise ValueError(
                f"Expected non-empty target file name, but got {target_file}."
            )
        noisy_stem = osp.splitext(noisy_file)[0]
        target_stem = osp.splitext(target_file)[0]
        if not noisy_stem.isdigit() or not target_stem.isdigit():
            raise ValueError(
                "Indexed-name samples require integer file stems, but got "
                f"{noisy_file} and {target_file}."
            )
        if int(noisy_stem) != int(target_stem):
            raise ValueError(
                "Indexed-name samples require matching integer file stems, but got "
                f"{noisy_file} and {target_file}."
            )
        return {
            noisy_key: noisy_file,
            target_key: target_file,
            name_key: noisy_file,
        }

    def __call__(
        self,
        sample_data_list: Union[
            Sequence[Dict[str, str]],
            Dict[str, str],
        ],
        target_file_names: Optional[Sequence[str]] = None,
        noisy_root: Optional[str] = None,
        target_root: Optional[str] = None,
        file_suffix: Optional[str] = None,
    ) -> Union[List[Dict[str, str]], Dict[str, str]]:
        if target_file_names is not None:
            sample_data_list = _pair_files_by_index(
                sample_data_list,
                target_file_names,
                self.noisy_file_key,
                self.target_file_key,
                noisy_root,
                target_root,
                file_suffix,
            )

        if isinstance(sample_data_list, (list, tuple)):
            if len(sample_data_list) == 0:
                return []
            return [
                BuildIndexedNameSamples.build_one(
                    sample_data,
                    self.noisy_file_key,
                    self.target_file_key,
                    self.noisy_key,
                    self.target_key,
                    self.name_key,
                )
                for sample_data in sample_data_list
            ]
        return BuildIndexedNameSamples.build_one(
            sample_data_list,
            self.noisy_file_key,
            self.target_file_key,
            self.noisy_key,
            self.target_key,
            self.name_key,
        )

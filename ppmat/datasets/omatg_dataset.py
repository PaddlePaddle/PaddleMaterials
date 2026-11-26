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

"""OMatG structure dataset: reads LMDB/CSV/parquet into plain dict samples."""

import ast
import json
import os
import pickle
import zlib
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

import lmdb
import numpy as np
import paddle
import pandas as pd

from ppmat.datasets.custom_data_type import ConcatData


def _open_lmdb(file_path: str) -> lmdb.Environment:
    """Open an LMDB file or directory read-only with memory-safe defaults."""
    return lmdb.open(
        file_path,
        subdir=os.path.isdir(file_path),
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
    )


def _lmdb_keys(env: lmdb.Environment) -> List[str]:
    """List the data keys, excluding metadata keys (starting with "__")."""
    with env.begin() as txn:
        all_keys = [
            k.decode("ascii") if isinstance(k, bytes) else k
            for k in txn.cursor().iternext(values=False)
        ]
    return [k for k in all_keys if not k.startswith("__")]


def _lmdb_get(env: lmdb.Environment, key: str) -> Any:
    """Fetch and decode the record stored under ``key``."""
    key_bytes = key.encode("ascii") if isinstance(key, str) else key
    with env.begin() as txn:
        value = txn.get(key_bytes)
    if value is None:
        raise KeyError(f"Key {key} not found in LMDB.")
    return _decode_payload(value)


def _decode_payload(payload: bytes) -> Any:
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


class _OMATGStructure:
    """Single-crystal container with coordinate conversion, used to build samples."""

    def __init__(
        self,
        cell: paddle.Tensor,
        atomic_numbers: paddle.Tensor,
        pos: paddle.Tensor,
        pos_is_fractional: bool = False,
    ) -> None:
        if cell.shape != (3, 3):
            raise ValueError(f"cell must be 3x3, got {cell.shape}")
        if atomic_numbers.dim() != 1:
            raise ValueError(f"atomic_numbers must be 1D, got {atomic_numbers.dim()}D")
        if pos.shape[0] != len(atomic_numbers) or pos.shape[1] != 3:
            raise ValueError(f"pos must be (N, 3), got {pos.shape}")

        self._cell = cell
        self._atomic_numbers = atomic_numbers
        self._pos = pos
        self._fractional = pos_is_fractional

    @property
    def cell(self) -> paddle.Tensor:
        return self._cell

    @property
    def atomic_numbers(self) -> paddle.Tensor:
        return self._atomic_numbers

    @property
    def pos(self) -> paddle.Tensor:
        return self._pos

    @property
    def pos_is_fractional(self) -> bool:
        return self._fractional

    def niggli_reduce(self) -> None:
        from pymatgen.core import Element
        from pymatgen.core import Structure as PmgStructure

        species = [Element.from_Z(int(z)).symbol for z in self._atomic_numbers.numpy()]
        pmg = PmgStructure(
            lattice=self._cell.numpy(),
            species=species,
            coords=self._pos.numpy(),
            coords_are_cartesian=not self._fractional,
        )
        reduced = pmg.get_reduced_structure(reduction_algo="niggli")
        self._cell = paddle.to_tensor(reduced.lattice.matrix, dtype=self._cell.dtype)
        self._pos = paddle.to_tensor(reduced.frac_coords, dtype=self._pos.dtype)
        self._fractional = True

    def convert_to_fractional(self) -> None:
        if not self._fractional:
            with paddle.no_grad():
                self._pos = paddle.remainder(
                    paddle.linalg.solve(self._cell, self._pos, left=False), 1.0
                )
            self._fractional = True


class OMATGStructureDataset(paddle.io.Dataset):
    """Read crystal structures from LMDB/CSV/parquet; lazy LMDB supported."""

    def __init__(
        self,
        path: Optional[str] = None,
        lazy_storage: bool = True,
        convert_to_fractional: bool = True,
        niggli_reduce: bool = False,
        *,
        file_path: Optional[str] = None,
    ) -> None:
        self._env = None
        if path is not None and file_path is not None:
            raise ValueError("Specify path or the legacy file_path alias, not both.")
        path = path if path is not None else file_path
        if path is None:
            raise ValueError("A dataset path must be provided.")
        self.path = path
        self.lazy_storage = lazy_storage
        self.convert_to_fractional = convert_to_fractional
        self.niggli_reduce = niggli_reduce
        file_format = Path(path).suffix.lower()

        if file_format == ".lmdb":
            self._init_from_lmdb()
        elif file_format == ".csv":
            self._init_from_csv()
        elif file_format == ".parquet":
            self._init_from_parquet()
        else:
            raise ValueError(f"Unsupported file format: {file_format}")

    @property
    def file_path(self):
        return self.path

    @property
    def env(self):
        if self._env is None:
            self._env = _open_lmdb(self.file_path)
        return self._env

    def _init_from_lmdb(self) -> None:
        temp_env = _open_lmdb(self.file_path)

        try:
            self.keys = _lmdb_keys(temp_env)

            if not self.lazy_storage:
                self.structures = []
                for key in self.keys:
                    data = _lmdb_get(temp_env, key)
                    self.structures.append(self._create_structure(data))
        finally:
            temp_env.close()

    def _build_sample(
        self,
        cell: paddle.Tensor,
        atomic_numbers: paddle.Tensor,
        pos: paddle.Tensor,
        pos_is_fractional: bool = False,
    ) -> Dict[str, Any]:
        structure = _OMATGStructure(
            cell=cell,
            atomic_numbers=atomic_numbers,
            pos=pos,
            pos_is_fractional=pos_is_fractional,
        )
        if self.convert_to_fractional:
            structure.convert_to_fractional()
        if self.niggli_reduce:
            structure.niggli_reduce()

        # Wrap fields in ConcatData so DefaultCollator concatenates atom-level
        # fields and stacks sample-level fields (same pattern as MP20Dataset).
        n = len(structure.atomic_numbers)
        return {
            "n_atoms": ConcatData(np.array([n], dtype="int64")),
            "species": ConcatData(structure.atomic_numbers.numpy()),
            "cell": ConcatData(structure.cell.numpy().reshape(1, 3, 3)),
            "pos": ConcatData(structure.pos.numpy()),
            "pos_is_fractional": ConcatData(
                np.array([structure.pos_is_fractional], dtype="bool")
            ),
        }

    def _init_from_csv(self) -> None:
        from ppmat.datasets.build_structure import BuildStructure

        df = pd.read_csv(self.file_path)
        if "cif" not in df.columns:
            raise KeyError(
                f"CSV file does not contain 'cif' column. "
                f"Available columns: {list(df.columns)}"
            )

        self.structures = []
        for _, row in df.iterrows():
            pmg = BuildStructure.build_one(
                row["cif"], "cif_str", niggli=False, canocial=False
            )

            cell = paddle.to_tensor(pmg.lattice.matrix, dtype="float32")
            atomic_numbers = paddle.to_tensor(pmg.atomic_numbers, dtype="int64")
            pos = paddle.to_tensor(pmg.frac_coords, dtype="float32")

            self.structures.append(
                self._build_sample(cell, atomic_numbers, pos, pos_is_fractional=True)
            )

        self.keys = list(range(len(self.structures)))
        self.lazy_storage = False

    def _init_from_parquet(self) -> None:
        df = pd.read_parquet(self.file_path)
        required_cols = ["positions", "cell", "atomic_numbers"]
        for col in required_cols:
            if col not in df.columns:
                raise KeyError(
                    f"Parquet file missing '{col}'. "
                    f"Available columns: {list(df.columns)}"
                )

        self.structures = []
        for _, row in df.iterrows():
            atomic_numbers = paddle.to_tensor(
                np.asarray(row["atomic_numbers"], dtype=np.int64), dtype="int64"
            )
            pos = paddle.to_tensor(np.stack(row["positions"]), dtype="float32")
            cell = paddle.to_tensor(np.stack(row["cell"]), dtype="float32")

            self.structures.append(self._build_sample(cell, atomic_numbers, pos))

        self.keys = list(range(len(self.structures)))
        self.lazy_storage = False

    def _create_structure(self, data: Dict[str, Any]) -> Dict[str, Any]:
        cell = paddle.to_tensor(data["cell"], dtype="float32")
        atomic_numbers = paddle.to_tensor(data["atomic_numbers"], dtype="int64")
        pos = paddle.to_tensor(data["pos"], dtype="float32")

        return self._build_sample(cell, atomic_numbers, pos)

    def __len__(self) -> int:
        if self.lazy_storage:
            return len(self.keys)
        else:
            return len(self.structures)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if self.lazy_storage:
            data = _lmdb_get(self.env, self.keys[idx])
            return self._create_structure(data)
        return self.structures[idx]

    def __del__(self) -> None:
        if self._env is not None:
            self._env.close()
            self._env = None


__all__ = [
    "OMATGStructureDataset",
]

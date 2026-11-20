# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import ast
import datetime
import gzip
import hashlib
import json
import lzma
import os
import os.path as osp
import shutil
import tempfile
from contextlib import contextmanager
from contextlib import nullcontext
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Iterator
from typing import List
from typing import Optional
from typing import TextIO

import numpy as np

if TYPE_CHECKING:
    from cvve import GridSpec


def open_text(path: str | os.PathLike[str], mode: str = "rt") -> TextIO:
    """Open a plain, gzip, xz, or lz4 text file based on its suffix."""

    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".lz4":
        import lz4.frame

        return lz4.frame.open(path, mode=mode)
    if suffix == ".xz":
        return lzma.open(path, mode=mode)
    if suffix == ".gz":
        return gzip.open(path, mode=mode)
    return path.open(mode=mode)


@contextmanager
def materialize_text_path(
    source: str | os.PathLike[str] | TextIO,
) -> Iterator[Path]:
    """Materialize compressed input or a text stream for path-only parsers."""

    if isinstance(source, (str, os.PathLike)):
        path = Path(source).expanduser()
        if path.suffix.lower() not in {".gz", ".xz", ".lz4"}:
            yield path
            return
        file_context = open_text(path)
    else:
        file_context = nullcontext(source)

    temporary_path: Path | None = None
    try:
        with file_context as file_obj:
            with tempfile.NamedTemporaryFile(
                mode="wt",
                encoding="utf-8",
                delete=False,
            ) as temporary_file:
                temporary_path = Path(temporary_file.name)
                shutil.copyfileobj(file_obj, temporary_file)
        yield temporary_path
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def write_cube(
    destination: str | os.PathLike[str],
    atom_numbers,
    atom_coord,
    density,
    grid: "GridSpec",
) -> None:
    """Write one scalar Gaussian CUBE file with ASE.

    Input geometry follows ``grid.length_unit``. ASE serializes CUBE
    geometry in Bohr, while field values are written without unit scaling.

    Args:
        destination: Plain CUBE output path.
        atom_numbers: Atomic numbers with shape ``[num_atoms]``.
        atom_coord: Atomic coordinates with shape ``[num_atoms, 3]``.
        density: Flattened scalar values matching ``grid.shape``.
        grid: Grid geometry and coordinate-unit metadata.
    """

    from ase import Atoms
    from ase.io import write as ase_write
    from ase.units import Bohr

    from ppmat.utils.crystal import normalize_coordinate_unit

    destination = Path(destination)
    if destination.suffix.lower() in {".gz", ".xz", ".lz4"}:
        raise ValueError("ASE CUBE output requires an uncompressed path.")
    destination.parent.mkdir(parents=True, exist_ok=True)

    coordinate_unit = normalize_coordinate_unit(grid.length_unit)
    to_angstrom = 1.0 if coordinate_unit == "angstrom" else Bohr

    atom_numbers = np.asarray(atom_numbers, dtype=np.int64).reshape(-1)
    atom_coord = np.asarray(atom_coord, dtype=float)
    density = np.asarray(density, dtype=float).reshape(grid.shape)

    atoms = Atoms(
        numbers=atom_numbers,
        positions=atom_coord * to_angstrom,
        cell=grid.cell_vectors * to_angstrom,
    )
    ase_write(
        destination,
        atoms,
        format="cube",
        data=density,
        origin=grid.origin * to_angstrom,
    )


def count_samples_json_lines(path: str):
    """Fast count of samples in a line-delimited JSON file."""
    with open(path, "r") as f:
        return sum(1 for _ in f)


def read_json_lines(path):
    """
    Read all lines from a line-delimited JSON file,
    extracting all properties into a dictionary of lists.
    """
    property_data = {}

    with open(path, "r") as f:
        for idx, line in enumerate(f):
            content = ast.literal_eval(line.strip())
            # if idx == 301:
            #     break
            if idx == 0:
                all_property_names = list(content.keys())
                # print("all_property_names:", all_property_names)
                property_data = {name: [] for name in all_property_names}

            for property_name in all_property_names:
                if property_name not in content:
                    raise ValueError(
                        f"'{property_name}' not found in line {idx + 1} of file"
                    )
                property_data[property_name].append(content[property_name])
    return property_data


def read_json(path):
    """ """
    if not path.endswith(".json"):
        raise UserWarning(f"Path {path} is not a json-path.")
    with open(path, "r") as f:
        content = json.load(f)
    return content


def list_files_by_suffix(path: str, suffix: str) -> List[str]:
    """List files under path with the given suffix."""
    if not osp.isdir(path):
        raise FileNotFoundError(f"Directory not found: {path}")
    file_names = sorted(
        file_name for file_name in os.listdir(path) if file_name.endswith(suffix)
    )
    if not file_names:
        raise FileNotFoundError(f"No files ending with {suffix} found under {path}.")
    return file_names


def update_json(path, data):
    """ """
    if not path.endswith(".json"):
        raise UserWarning(f"Path {path} is not a json-path.")
    content = read_json(path)
    content.update(data)
    write_json(path, content)


def write_json(path, data):
    """ """
    if not path.endswith(".json"):
        raise UserWarning(f"Path {path} is not a json-path.")

    def handler(obj: object) -> (int | object):
        """Convert numpy int64 to int.

        Fixes TypeError: Object of type int64 is not JSON serializable
        reported in https://github.com/CederGroupHub/chgnet/issues/168.

        Returns:
            int | object: object for serialization
        """
        if isinstance(obj, np.integer):
            return int(obj)
        return obj

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4, default=handler)


def read_value_json(path, key):
    """ """
    content = read_json(path)
    if key in content.keys():
        return content[key]
    else:
        return None


def calc_md5(fullname):
    md5 = hashlib.md5()
    fullname = os.path.expanduser(fullname)
    with open(fullname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            md5.update(chunk)
    calc_md5sum = md5.hexdigest()

    return calc_md5sum


def append_timestamp_to_output_dir(
    config,
    now: Optional[datetime.datetime] = None,
):
    seed = config["Trainer"].get("seed", 42)
    timestamp = (now or datetime.datetime.now()).strftime("%Y%m%d_%H%M%S")
    base_output_dir = config["Trainer"]["output_dir"]
    config["Trainer"]["output_dir"] = f"{base_output_dir}_t_{timestamp}_s_{seed}"
    return config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate MD5 hash of a file")
    parser.add_argument("filename", help="Path to the file to hash")
    args = parser.parse_args()

    md5 = calc_md5(args.filename)
    print(md5)

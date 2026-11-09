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


def load_deeph_structure(folder: str | os.PathLike[str]):
    """Load a DeepH processed structure without changing its lattice or coordinates."""
    from ase import Atoms
    from pymatgen.io.ase import AseAtomsAdaptor

    folder = os.fspath(folder)
    lattice = np.loadtxt(osp.join(folder, "lat.dat")).T
    atomic_numbers = np.loadtxt(osp.join(folder, "element.dat")).astype(np.int64)
    cart_coords = np.loadtxt(osp.join(folder, "site_positions.dat")).T
    atoms = Atoms(
        numbers=atomic_numbers.reshape(-1),
        positions=cart_coords,
        cell=lattice,
        pbc=True,
    )
    return AseAtomsAdaptor.get_structure(atoms)


def load_deeph_orbital_counts(path: str | os.PathLike[str]):
    """Return the number of atomic orbitals described on each input line."""
    with open(path) as file:
        orbital_types = [list(map(int, line.split())) for line in file if line.strip()]
    return [
        sum(2 * orbital + 1 for orbital in atom_orbitals)
        for atom_orbitals in orbital_types
    ]


def _load_deeph_npz_matrices(path, dtype):
    matrices = {}
    with np.load(path) as matrix_file:
        for key_string, value in matrix_file.items():
            translation_x, translation_y, translation_z, atom_i, atom_j = json.loads(
                key_string
            )
            matrices[
                (translation_x, translation_y, translation_z, atom_i - 1, atom_j - 1)
            ] = np.asarray(value, dtype=dtype)
    return matrices


def load_deeph_matrices(folder, dtype, include_hamiltonian=True):
    """Load orbital counts, local rotations, and optional Hamiltonian blocks."""
    orbital_counts = load_deeph_orbital_counts(osp.join(folder, "orbital_types.dat"))
    rotations = _load_deeph_npz_matrices(osp.join(folder, "rc.npz"), dtype)
    hamiltonians = None
    if include_hamiltonian:
        hamiltonians = _load_deeph_npz_matrices(osp.join(folder, "rh.npz"), dtype)
    return orbital_counts, rotations, hamiltonians


def build_deeph_hamiltonian_blocks(prediction, model_input, orbital, spinful=False):
    """Restore edge predictions to DeepH ``rh.npz`` Hamiltonian blocks."""
    import paddle

    if spinful:
        raise NotImplementedError("Spinful DeepH Hamiltonian export is not enabled.")

    def to_numpy(value):
        return value.numpy() if paddle.is_tensor(value) else np.asarray(value)

    folder = model_input["structure_folder"]
    if isinstance(folder, (list, tuple)):
        if len(folder) != 1:
            raise ValueError("DeepH Hamiltonian export expects batch_size=1.")
        folder = folder[0]
    if not isinstance(folder, str):
        raise TypeError("DeepH Hamiltonian export requires structure_folder.")

    orbital_counts = load_deeph_orbital_counts(osp.join(folder, "orbital_types.dat"))
    prediction = to_numpy(prediction)
    edge_index = to_numpy(model_input["edge_index"]).astype(np.int64)
    if edge_index.ndim == 2 and edge_index.shape[0] != 2:
        edge_index = edge_index.T
    edge_attr = to_numpy(model_input["edge_attr"])
    atomic_numbers = to_numpy(model_input["structure_atomic_numbers"]).astype(np.int64)
    lattice = to_numpy(model_input["structure_lattice"]).reshape(-1, 3, 3)[0]
    inverse_lattice = np.linalg.inv(lattice)
    lattice_shifts = np.rint(
        edge_attr[:, 4:7] @ inverse_lattice - edge_attr[:, 7:10] @ inverse_lattice
    ).astype(int)

    hamiltonian = {}
    for edge_id in range(edge_attr.shape[0]):
        atom_i, atom_j = edge_index[:, edge_id]
        block = np.zeros(
            (orbital_counts[atom_i], orbital_counts[atom_j]),
            dtype=prediction.dtype,
        )
        number_i = int(atomic_numbers[atom_i])
        number_j = int(atomic_numbers[atom_j])
        for output_index, orbital_dict in enumerate(orbital):
            for species_pair, orbital_pair in orbital_dict.items():
                expected_i, expected_j = map(int, species_pair.split())
                if number_i == expected_i and number_j == expected_j:
                    orbital_i, orbital_j = orbital_pair
                    block[orbital_i, orbital_j] = prediction[edge_id, output_index]
        key = json.dumps(
            [
                int(lattice_shifts[edge_id, 0]),
                int(lattice_shifts[edge_id, 1]),
                int(lattice_shifts[edge_id, 2]),
                int(atom_i) + 1,
                int(atom_j) + 1,
            ]
        )
        hamiltonian[key] = block
    return hamiltonian


def save_deeph_hamiltonian(path, prediction, model_input, orbital, spinful=False):
    """Write restored DeepH Hamiltonian blocks to a compressed NPZ file."""
    hamiltonian = build_deeph_hamiltonian_blocks(
        prediction, model_input, orbital, spinful=spinful
    )
    path = osp.abspath(path)
    os.makedirs(osp.dirname(path), exist_ok=True)
    np.savez(path, **hamiltonian)


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

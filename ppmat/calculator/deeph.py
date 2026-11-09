# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

import glob
import importlib
import json
import os
import re
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.calculators.calculator import all_changes
from ase.data import atomic_numbers
from ase.io import write
from omegaconf import OmegaConf
from pymatgen.io.ase import AseAtomsAdaptor

from ppmat.utils import logger


def _import_h5py():
    try:
        import h5py
    except ImportError as exc:
        raise ImportError(
            "OpenMX overlap processing requires h5py. Install the dependencies "
            "from requirements.txt."
        ) from exc
    return h5py


def _parse_openmx_metadata(output_path):
    lines = Path(output_path).read_text().splitlines()
    orbital_by_element = {}
    lattice = None
    elements = []
    frac_coords = []

    in_species = False
    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("<Definition.of.Atomic.Species"):
            in_species = True
            continue
        if stripped.startswith("Definition.of.Atomic.Species>"):
            in_species = False
            continue
        if in_species:
            fields = stripped.split()
            if len(fields) >= 2:
                basis = fields[1].rsplit("-", 1)[-1]
                orbital_tokens = re.findall(r"([spdf])(\d+)", basis)
                if orbital_tokens:
                    angular_momentum = {"s": 0, "p": 1, "d": 2, "f": 3}
                    orbital_by_element[fields[0]] = [
                        angular_momentum[name]
                        for name, count in orbital_tokens
                        for _ in range(int(count))
                    ]

        if "Atoms.UnitVectors.Unit" in line:
            fields = stripped.split()
            if len(fields) < 2 or fields[1].lower() != "ang":
                raise ValueError("OpenMX lattice vectors must use Angstrom units.")
            vectors = []
            cursor = index + 1
            while cursor < len(lines) and len(vectors) < 3:
                values = lines[cursor].split()
                if len(values) == 3:
                    try:
                        vectors.append([float(value) for value in values])
                    except ValueError:
                        pass
                cursor += 1
            if len(vectors) != 3:
                raise ValueError("Can not parse lattice vectors from OpenMX output.")
            lattice = np.asarray(vectors, dtype=np.float64)

        if "Fractional coordinates of the final structure" in line:
            cursor = index + 1
            started = False
            while cursor < len(lines):
                fields = lines[cursor].split()
                if len(fields) == 5 and fields[0].isdigit():
                    expected_index = len(elements) + 1
                    if int(fields[0]) != expected_index:
                        raise ValueError("Unexpected atom index in OpenMX output.")
                    elements.append(fields[1])
                    frac_coords.append([float(value) for value in fields[2:5]])
                    started = True
                elif started:
                    break
                cursor += 1

    if lattice is None:
        raise ValueError("Can not find lattice vectors in OpenMX output.")
    if not elements:
        raise ValueError("Can not find final fractional coordinates in OpenMX output.")
    missing_orbitals = sorted(set(elements) - set(orbital_by_element))
    if missing_orbitals:
        raise ValueError(
            "Can not find OpenMX orbital definitions for: " f"{missing_orbitals}"
        )
    return lattice, elements, np.asarray(frac_coords), orbital_by_element


def _read_openmx_overlaps(raw_dir, overlap_pattern):
    h5py = _import_h5py()
    overlap_paths = sorted(glob.glob(os.path.join(raw_dir, overlap_pattern)))
    if not overlap_paths:
        raise FileNotFoundError(
            f"No OpenMX overlap files match {overlap_pattern!r} in {raw_dir}."
        )

    overlaps = {}
    for overlap_path in overlap_paths:
        with h5py.File(overlap_path, "r") as overlap_file:
            for key, value in overlap_file.items():
                if key in overlaps:
                    raise ValueError(f"Duplicate OpenMX overlap block: {key}")
                overlaps[key] = value[...]
    return overlaps


def _build_local_coordinates(overlap_keys, cart_coords, lattice, tolerance=1e-8):
    neighbours = {index: [] for index in range(cart_coords.shape[0])}
    parsed_keys = []
    for key_str in overlap_keys:
        key = json.loads(key_str)
        if len(key) != 5:
            raise ValueError(f"Invalid DeepH overlap key: {key_str}")
        shift = np.asarray(key[:3], dtype=np.int64)
        atom_i = int(key[3]) - 1
        atom_j = int(key[4]) - 1
        vector = cart_coords[atom_j] + shift @ lattice - cart_coords[atom_i]
        distance = float(np.linalg.norm(vector))
        item = (distance, atom_j, shift, vector, key)
        neighbours.setdefault(atom_i, []).append(item)
        parsed_keys.append((key_str, item))

    for atom_i, atom_neighbours in neighbours.items():
        if not atom_neighbours:
            raise ValueError(f"Atom {atom_i} has no OpenMX overlap blocks.")
        atom_neighbours.sort(key=lambda item: item[0])
        if atom_neighbours[0][0] > tolerance:
            raise ValueError(f"Atom {atom_i} has no onsite overlap block.")

    rotations = {}
    for key_str, (_, _, _, vector_ij, key) in parsed_keys:
        atom_i = int(key[3]) - 1
        atom_neighbours = neighbours[atom_i]
        if np.linalg.norm(vector_ij) > tolerance:
            axis_source = vector_ij
        else:
            nonzero = [item[3] for item in atom_neighbours if item[0] > tolerance]
            if not nonzero:
                raise ValueError(f"Atom {atom_i} has no non-onsite overlap blocks.")
            axis_source = nonzero[0]

        second_axis_source = None
        for _, _, _, candidate, _ in atom_neighbours:
            cross = np.cross(axis_source, candidate)
            if np.linalg.norm(cross) > tolerance:
                second_axis_source = candidate
                break
        if second_axis_source is None:
            raise ValueError(
                "There is no linearly independent bond for DeepH local "
                f"coordinates around atom {atom_i}."
            )

        axis_1 = axis_source / np.linalg.norm(axis_source)
        cross = np.cross(axis_source, second_axis_source)
        axis_2 = cross / np.linalg.norm(cross)
        axis_3 = np.cross(axis_1, axis_2)
        rotations[key_str] = np.stack([axis_1, axis_2, axis_3], axis=-1)
    return rotations


def preprocess_openmx_overlap(
    raw_dir,
    output_dir,
    output_filename="openmx.out",
    overlap_pattern="output/overlaps_*.h5",
):
    """Convert overlap-only OpenMX output into DeepH inference inputs."""
    raw_dir = os.path.abspath(raw_dir)
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    overlaps = _read_openmx_overlaps(raw_dir, overlap_pattern)
    lattice, elements, frac_coords, orbital_by_element = _parse_openmx_metadata(
        os.path.join(raw_dir, output_filename)
    )
    cart_coords = frac_coords @ lattice

    np.savetxt(os.path.join(output_dir, "lat.dat"), lattice.T)
    np.savetxt(os.path.join(output_dir, "rlat.dat"), np.linalg.inv(lattice) * 2 * np.pi)
    np.savetxt(os.path.join(output_dir, "site_positions.dat"), cart_coords.T)
    np.savetxt(
        os.path.join(output_dir, "element.dat"),
        np.asarray([atomic_numbers[element] for element in elements]),
        fmt="%d",
    )
    with open(os.path.join(output_dir, "orbital_types.dat"), "w") as file:
        for element in elements:
            file.write(" ".join(str(value) for value in orbital_by_element[element]))
            file.write("\n")

    rotations = _build_local_coordinates(overlaps, cart_coords, lattice)
    np.savez(os.path.join(output_dir, "overlaps.npz"), **overlaps)
    np.savez(os.path.join(output_dir, "rc.npz"), **rotations)
    return output_dir


class DeepHDFTInferenceTask:
    def __init__(
        self,
        species,
        num_l,
        orbital=None,
        orbital_config_path=None,
        output_dir="deeph_dft_results",
        dtype="float32",
        run_dft=True,
        processed_dirs=None,
    ):
        if orbital is not None and OmegaConf.is_config(orbital):
            orbital = OmegaConf.to_container(orbital, resolve=True)
        self.orbital_config_path = orbital_config_path
        self.orbital = orbital
        self.species = list(species)
        self.num_l = num_l
        self.output_dir = output_dir
        self.dtype = dtype
        self.run_dft = run_dft
        self.processed_dirs = processed_dirs

    def __call__(self, interface_obj, structures):
        logger.info("Run OpenMX overlap calculation and Paddle DeepH inference.")
        interface_obj.run_deeph_inference(
            structures=structures,
            orbital_config_path=self.orbital_config_path,
            orbital=self.orbital,
            species=self.species,
            num_l=self.num_l,
            output_dir=self.output_dir,
            dtype=self.dtype,
            run_dft=self.run_dft,
            processed_dirs=self.processed_dirs,
        )
        logger.info("All DeepH DFT inference tasks finished successfully.")


class DeepHDFTCalculator:
    """Run overlap-only DFT through ASE and infer Hamiltonians with DeepH."""

    def __init__(
        self,
        predictor,
        calculator_cls="ase.calculators.openmx.OpenMX",
        calculator_kwargs=None,
        command=None,
        output_filename="openmx.log",
        overlap_pattern="output/overlaps_*.h5",
        **kwargs,
    ):
        del kwargs
        self.predictor = predictor
        self.calculator_cls = calculator_cls
        self.calculator_kwargs = dict(calculator_kwargs or {})
        self.command = command
        self.output_filename = output_filename
        self.overlap_pattern = overlap_pattern

    @staticmethod
    def _to_ase(structure):
        if isinstance(structure, Atoms):
            return structure.copy()
        return AseAtomsAdaptor.get_atoms(structure)

    def _build_calculator(self, dft_dir):
        module_name, class_name = self.calculator_cls.rsplit(".", 1)
        calculator_class = getattr(importlib.import_module(module_name), class_name)
        kwargs = dict(self.calculator_kwargs)
        kwargs.setdefault("label", os.path.join(dft_dir, "openmx"))
        if self.command is not None:
            kwargs.setdefault("command", self.command)
        return calculator_class(**kwargs)

    def _run_dft(self, structure, dft_dir):
        os.makedirs(dft_dir, exist_ok=True)
        atoms = self._to_ase(structure)
        write(os.path.join(dft_dir, "structure.xyz"), atoms)
        calculator = self._build_calculator(dft_dir)
        if hasattr(calculator, "write_input") and hasattr(calculator, "run"):
            calculator.atoms = atoms.copy()
            calculator.write_input(
                atoms=atoms,
                properties=[],
                system_changes=all_changes,
            )
            calculator.run()
        else:
            atoms.calc = calculator
            atoms.get_potential_energy()

    def run_deeph_inference(
        self,
        structures,
        species,
        num_l,
        output_dir,
        dtype,
        run_dft,
        processed_dirs,
        orbital=None,
        orbital_config_path=None,
    ):
        if not run_dft and processed_dirs is None:
            raise ValueError("processed_dirs is required when run_dft=False.")
        if processed_dirs is not None and len(processed_dirs) != len(structures):
            raise ValueError("processed_dirs and structures must have the same length.")

        if orbital is None:
            if orbital_config_path is not None:
                orbital = self.predictor.load_orbital_config(orbital_config_path)
            else:
                orbital = self.predictor.config["DeepH"]["basic"]["orbital"]
        os.makedirs(output_dir, exist_ok=True)
        for index, structure in enumerate(structures):
            sample_dir = os.path.join(output_dir, f"{index:06d}")
            processed_dir = os.path.join(sample_dir, "processed")
            if run_dft:
                dft_dir = os.path.join(sample_dir, "dft")
                self._run_dft(structure, dft_dir)
                preprocess_openmx_overlap(
                    raw_dir=dft_dir,
                    output_dir=processed_dir,
                    output_filename=self.output_filename,
                    overlap_pattern=self.overlap_pattern,
                )
            else:
                processed_dir = os.path.abspath(processed_dirs[index])

            required_files = {
                "lat.dat",
                "site_positions.dat",
                "element.dat",
                "orbital_types.dat",
                "rc.npz",
            }
            missing = sorted(
                filename
                for filename in required_files
                if not os.path.exists(os.path.join(processed_dir, filename))
            )
            if missing:
                raise FileNotFoundError(
                    f"Incomplete DeepH DFT data in {processed_dir}: {missing}"
                )

            os.makedirs(sample_dir, exist_ok=True)
            self.predictor.predict_hamiltonian(
                processed_dir=processed_dir,
                output_path=os.path.join(sample_dir, "rh_pred.npz"),
                orbital=orbital,
                species=species,
                num_l=num_l,
                dtype=dtype,
            )

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

"""Reference MatInvent population qualification with native Paddle relaxation."""

import gzip
import hashlib
import itertools
import multiprocessing
import os
import pickle
import re
import shutil
import tempfile
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache

import lmdb
import numpy as np
from ase import units
from ase.calculators.calculator import all_changes
from ase.filters import ExpCellFilter
from ase.optimize import FIRE
from monty.json import MontyDecoder
from pymatgen.analysis.phase_diagram import PhaseDiagram
from pymatgen.analysis.structure_analyzer import oxide_type
from pymatgen.analysis.structure_matcher import OrderDisorderElementComparator
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.entries.compatibility import MaterialsProject2020Compatibility
from pymatgen.entries.computed_entries import ComputedStructureEntry
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.vasp.sets import MPRelaxSet

from ppmat.calculator.ase import ASECalculator
from ppmat.models.matinvent.dataset import is_valid_structure
from ppmat.predictor.potential_predictor import PotentialPredictor
from ppmat.utils import download
from ppmat.utils import logger
from ppmat.utils.eager_comp_setting import setting_eager_mode


class _DisorderedMatcher(StructureMatcher):
    """Match the substitution rules in the bundled MatterGen reference matcher."""

    def __init__(self):
        super().__init__(
            ltol=0.2,
            stol=0.3,
            angle_tol=5,
            primitive_cell=True,
            scale=True,
            comparator=OrderDisorderElementComparator(),
            attempt_supercell=True,
            allow_subset=True,
        )
        self.ordered = StructureMatcher()

    def fit(self, struct1, struct2):
        first = struct1.copy().remove_oxidation_states()
        second = struct2.copy().remove_oxidation_states()
        if first == second:
            return True
        if first.is_ordered and second.is_ordered:
            if first.composition.reduced_formula != second.composition.reduced_formula:
                return False
            if self.ordered.fit(first, second):
                return True
            groups = []
            for left, right in itertools.combinations(first.composition.elements, 2):
                radius_difference = abs(left.atomic_radius - right.atomic_radius)
                mean_radius = (left.atomic_radius + right.atomic_radius) / 2
                if not (
                    radius_difference / mean_radius <= 0.3
                    and abs(left.X - right.X) <= 1
                ):
                    continue
                connected = {left, right}
                remaining = []
                for group in groups:
                    if connected & group:
                        connected.update(group)
                    else:
                        remaining.append(group)
                groups = remaining + [connected]
            if not groups:
                return False
            for group in groups:
                fractions = {
                    element: first.composition.get_atomic_fraction(element)
                    for element in group
                }
                total = sum(fractions.values())
                occupancy = {
                    element: value / total for element, value in fractions.items()
                }
                first.replace_species({element: occupancy for element in group})
        elif not first.composition.fractional_composition.almost_equals(
            second.composition.fractional_composition, atol=0.01, rtol=0.1
        ):
            return False
        return super().fit(first, second)


class _Reference:
    def __init__(self, path, cache_dir=None):
        digest = hashlib.sha256()
        with open(path, "rb") as source:
            for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        cache_dir = cache_dir or os.path.join(download.DATASETS_HOME, "matinvent")
        os.makedirs(cache_dir, exist_ok=True)
        database = os.path.join(cache_dir, f"{digest.hexdigest()}.lmdb")
        if not os.path.isfile(database):
            descriptor, temporary = tempfile.mkstemp(dir=cache_dir, suffix=".lmdb")
            try:
                with os.fdopen(descriptor, "wb") as target, gzip.open(
                    path, "rb"
                ) as source:
                    shutil.copyfileobj(source, target)
                os.replace(temporary, database)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
        self.environment = lmdb.open(
            database, subdir=False, readonly=True, lock=False, readahead=False
        )
        with self.environment.begin() as transaction:
            self.chemical_systems = set(self._get(transaction, "chemical_systems"))
        self.entries = lru_cache(maxsize=128)(self._read_entries)
        self.phase_diagram = lru_cache(maxsize=32)(self._build_phase_diagram)

    @staticmethod
    def _get(transaction, key):
        value = transaction.get(key.encode("ascii"))
        if value is None:
            raise ValueError(f"Missing reference database key: {key}")
        return pickle.loads(value)

    def _read_entries(self, chemical_system):
        if chemical_system not in self.chemical_systems:
            return ()
        with self.environment.begin() as transaction:
            formulas = self._get(transaction, f"{chemical_system}.reduced_formulas")
            entries = []
            for reduced_formula in formulas:
                prefix = f"{chemical_system}.{reduced_formula}"
                length = self._get(transaction, f"{prefix}.length")
                entries.extend(
                    MontyDecoder().process_decoded(
                        self._get(transaction, f"{prefix}.{i}")
                    )
                    for i in range(length)
                )
        return tuple(entries)

    def _build_phase_diagram(self, chemical_system):
        elements = chemical_system.split("-")
        entries = [
            entry
            for size in range(1, len(elements) + 1)
            for subsystem in itertools.combinations(elements, size)
            for entry in self.entries("-".join(subsystem))
            if not np.isnan(entry.energy)
        ]
        return PhaseDiagram(entries)


class _MatterSimCalculator(ASECalculator):
    """Adapt MatterSim units and refresh geometry for the shared ASE calculator."""

    def __init__(self, predictor):
        super().__init__(predictor)
        self.implemented_properties = ["energy", "free_energy", "forces", "stress"]

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        atoms.info["graph"] = self.predictor.graph_converter(
            AseAtomsAdaptor.get_structure(atoms)
        )
        try:
            super().calculate(atoms, properties, system_changes)
        finally:
            atoms.info.pop("graph", None)
            if self.atoms is not None:
                self.atoms.info.pop("graph", None)
        self.results["energy"] *= len(atoms)
        self.results["free_energy"] = self.results["energy"]
        self.results["stress"] *= units.GPa
        if not all(np.isfinite(value).all() for value in self.results.values()):
            self.results.clear()
            raise FloatingPointError("Nonfinite MatterSim relaxation prediction")


def _relax_population(structures, predictor_cfg, fmax, max_steps):
    # MatterSim primitive-gradient settings must not alter the RL process.
    setting_eager_mode(True, ["stack_grad", "assign_grad"])
    predictor = PotentialPredictor(
        **predictor_cfg, config_overrides=["Predict.eval_with_no_grad=False"]
    )
    if predictor.config["Model"]["__class_name__"] != "M3GNet":
        raise ValueError("MatInvent qualification requires the MatterSim M3GNet model")
    predictor.model.energy_key = "energy"
    predictor.model.force_key = "force"
    predictor.model.stress_key = "stress"
    retained_indices, relaxed, energies = [], [], []
    for index, structure in enumerate(structures):
        atoms = AseAtomsAdaptor.get_atoms(structure)
        atoms.calc = _MatterSimCalculator(predictor)
        optimizer = FIRE(ExpCellFilter(atoms), logfile=None)
        optimizer.fmax = fmax
        for step in range(max_steps):
            snapshot = AseAtomsAdaptor.get_structure(atoms)
            energy = atoms.get_potential_energy()
            forces = optimizer.optimizable.get_forces()
            optimizer.step()
            converged = optimizer.converged(forces)
            if step % 5 == 0 or converged:
                logger.info(
                    f"Relaxation {index + 1}/{len(structures)}, step {step + 1}, "
                    f"energy={energy:.6f}, "
                    f"max_force={np.linalg.norm(forces, axis=1).max():.6f}, "
                    f"converged={converged}"
                )
            if converged:
                retained_indices.append(index)
                relaxed.append(snapshot)
                energies.append(energy)
                break
        else:
            logger.warning(
                f"Rejected relaxation candidate {index + 1}/{len(structures)}: "
                f"did not converge in {max_steps} steps"
            )
    return retained_indices, relaxed, np.asarray(energies)


class PopulationQualifier:
    """Keep original training structures aligned with relaxed scoring structures."""

    def __init__(
        self,
        reference_path,
        predictor_cfg,
        cache_dir=None,
        stability_threshold=0.1,
        fmax=0.05,
        max_steps=1000,
    ):
        if fmax <= 0 or max_steps <= 0:
            raise ValueError("Relaxation fmax and max_steps must be positive")
        self.reference = _Reference(reference_path, cache_dir)
        self.predictor_cfg = predictor_cfg
        self.stability_threshold = stability_threshold
        self.fmax = fmax
        self.max_steps = max_steps
        self.matcher = _DisorderedMatcher()
        self.compatibility = MaterialsProject2020Compatibility()

    def _corrected_entry(self, structure, energy):
        settings = MPRelaxSet(
            structure,
            user_incar_settings={"KSPACING": 0.5},
            user_kpoints_settings=None,
        )
        incar = settings.incar
        symbols = settings.potcar_symbols
        hubbards = {}
        if incar.get("LDAU", False):
            us = incar.get("LDAUU", [])
            js = incar.get("LDAUJ", [])
            if len(js) != len(us):
                js = [0] * len(us)
            if len(us) == len(symbols):
                hubbards = {
                    re.split("_", symbol)[0]: u - j
                    for symbol, u, j in zip(symbols, us, js)
                }
            elif sum(us) != 0 or sum(js) != 0:
                raise ValueError("Hubbard parameters and atomic symbols must align")
        is_hubbard = bool(hubbards) and sum(hubbards.values()) > 1e-8
        entry = ComputedStructureEntry(
            structure,
            energy,
            parameters={
                "is_hubbard": is_hubbard,
                "hubbards": hubbards,
                "run_type": "GGA+U" if is_hubbard else "GGA",
                "potcar_symbols": [f"PBE {symbol}" for symbol in symbols],
            },
            data={
                "oxide_type": oxide_type(structure),
                "aspherical": incar.get("LASPH", False),
            },
        )
        self.compatibility.process_entry(entry)
        return entry

    def __call__(self, structures):
        originals = [
            structure
            for structure in structures
            if all(
                any(
                    np.isfinite(entry.energy)
                    for entry in self.reference.entries(el.symbol)
                )
                for el in structure.composition.elements
            )
        ]
        if not originals:
            return [], []
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=1, mp_context=context) as executor:
            retained_indices, relaxed, energies = executor.submit(
                _relax_population,
                originals,
                self.predictor_cfg,
                self.fmax,
                self.max_steps,
            ).result()
        retained = [originals[index] for index in retained_indices]
        logger.info(f"Relaxation retained {len(retained)}/{len(originals)} candidates")
        return self.select(retained, relaxed, energies)

    def select(self, originals, relaxed, energies):
        if len(originals) != len(relaxed) or len(relaxed) != len(energies):
            raise ValueError("Qualification populations and energies must be aligned")
        training, scoring, unique = [], [], []
        for original, structure, energy in zip(originals, relaxed, energies):
            if not np.isfinite(energy):
                raise FloatingPointError("Nonfinite qualification energy")
            distinct = not any(self.matcher.fit(structure, prior) for prior in unique)
            if distinct:
                unique.append(structure)
            formula = structure.composition.reduced_formula
            chemical_system = structure.composition.chemical_system
            novel = not any(
                self.matcher.fit(structure, entry.structure)
                for entry in self.reference.entries(chemical_system)
            )
            entry = self._corrected_entry(structure, energy)
            above_hull = self.reference.phase_diagram(chemical_system).get_e_above_hull(
                entry, allow_negative=True
            )
            valid = is_valid_structure(structure, max_lattice_param=float("inf"))
            stable = above_hull <= self.stability_threshold
            logger.info(
                f"Qualification {formula}: valid={valid}, unique={distinct}, "
                f"novel={novel}, energy_above_hull={above_hull:.6f}"
            )
            if valid and distinct and novel and stable:
                training.append(original)
                scoring.append(structure)
        return training, scoring

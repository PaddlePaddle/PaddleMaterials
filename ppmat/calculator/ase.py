# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any
from typing import List
import warnings
import numpy as np
from ase import Atoms
from ase import filters
from ase import units
from ase.calculators.calculator import Calculator
from ase.io import Trajectory
from ase.io import write
from ase.md import MDLogger
from ase.md.langevin import Langevin
from ase.optimize import BFGS
from ase.optimize import FIRE
from ase.optimize import LBFGS
from ase.optimize import BFGSLineSearch
from ase.optimize import GPMin
from ase.optimize import LBFGSLineSearch
from ase.optimize import MDMin
from ase.stress import full_3x3_to_voigt_6_stress
from pymatgen.io.ase import AseAtomsAdaptor
from tqdm import tqdm

from ppmat.utils import logger

CHARGE_RANGE = [-100, 100]
SPIN_RANGE = [0, 100]
DEFAULT_CHARGE = 0
DEFAULT_SPIN_OMOL = 1
DEFAULT_SPIN = 0


OPTIMIZERS = {
    "FIRE": FIRE,
    "BFGS": BFGS,
    "LBFGS": LBFGS,
    "MDMin": MDMin,
    "GPMin": GPMin,
    "LBFGSLineSearch": LBFGSLineSearch,
    "BFGSLineSearch": BFGSLineSearch,
}

warnings.filterwarnings('ignore', message='Skipping "graph" info', category=UserWarning)

class OptimizationTask:
    def __init__(
        self,
        optimizer="BFGS",
        filter="none",
        fmax=0.05,
        steps=100,
    ):
        self.optimizer = optimizer
        self.filter = filter
        self.fmax = fmax
        self.steps = steps

    def __call__(
        self,
        interface_obj,
        structures,
    ):
        logger.info("Relax structure.")
        interface_obj.run_opt(
            structures=structures,
            optimizer=self.optimizer,
            filter=self.filter,
            fmax=self.fmax,
            steps=self.steps,
        )
        logger.info("All relaxations finished successfully.")


class MDSimulationTask:
    def __init__(self, temperature=300, timestep=0.1, steps=100, interval=1, **kwargs):
        self.temperature = temperature
        self.timestep = timestep
        self.steps = steps
        self.interval = interval

    def __call__(
        self,
        interface_obj,
        structures,
    ):
        logger.info("Run MD simulation.")
        interface_obj.run_md(
            structures=structures,
            temperature=self.temperature,
            timestep=self.timestep,
            steps=self.steps,
            interval=self.interval,
        )
        logger.info("All MD simulations finished successfully.")


class ASECalculator(Calculator):
    def __init__(
        self,
        predictor,
        **kwargs: Any,
    ):
        """
        Initialize the ASECalculator from a model PPMatPredictor

        Args:
            predictor (PPMatPredictor): A pretrained PPMatPredictor.
        Notes:
            - For models that require total charge and spin multiplicity
                `charge` and `spin` (corresponding to `spin_multiplicity`) are
                    pulled from `atoms.info` during calculations.
                - `charge` must be an integer representing the total charge
                    on the system and can range from -100 to 100.
                - `spin` must be an integer representing the spin multiplicity
                    and can range from 0 to 100.
                - If `charge` or `spin` are not set in `atoms.info`,
                    they will default to charge=`0` and spin=`1`.
        """

        super().__init__(**kwargs)

        label_names = predictor.config["Global"]["label_names"]
        model_cls = predictor.config["Model"].get("__class_name__")
        if model_cls == "CHGNet":
            if "energy_per_atom" in label_names:
                logger.warning(
                    "'energy_per_atom' found in prediction. "
                    "Please ensure this corresponds to total energy as expected."
                )
            label_names = ["forces" if x == "force" else x for x in label_names]
            if "energy_per_atom" in label_names and "energy" not in label_names:
                label_names.append("energy")
        elif model_cls == "M3GNet":
            label_names = ["forces" if x == "force" else x for x in label_names]
        else:
            raise NotImplementedError(
                f"Model {model_cls} not supported. "
                f"Check that predicted properties (forces, energy, stress) "
                f"match ASE requirements. Add handling if needed."
            )

        self.implemented_properties = label_names

        self.predictor = predictor

    def structure_to_ase(self, structures):
        # Read raw file and convert to ASE Atom
        ase_structures = []

        for structure in tqdm(structures):
            graph = self.predictor.graph_converter(structure)
            # covert pymatgen Structure to ASE Atom
            atom = AseAtomsAdaptor().get_atoms(structure)
            # attach graph information to ASE Atom
            atom.info["graph"] = graph
            ase_structures.append(atom)

        logger.info("Successfully covert all structures to ASE Atoms.")
        return ase_structures

    def check_state(self, atoms: Atoms, tol: float = 1e-15) -> list:
        """
        Check for any system changes since the last calculation.

        Args:
            atoms (ase.Atoms): The atomic structure to check.
            tol (float): Tolerance for detecting changes.

        Returns:
            list: A list of changes detected in the system.
        """
        state = super().check_state(atoms, tol=tol)
        if (not state) and (self.atoms.info != atoms.info):
            state.append("info")
        return state

    def calculate(
        self,
        atoms: Atoms,
        properties: List[str],
        system_changes: List[str],
    ) -> None:
        """
        Perform the calculation for the given atomic structure.

        Args:
            atoms (Atoms): The atomic structure to calculate properties for.
            properties (list[str]): The list of properties to calculate.
            system_changes (list[str]): The list of changes in the system.

        Notes:
            - `charge` must be an integer representing the total charge
                on the system and can range from -100 to 100.
            - `spin` must be an integer representing the spin multiplicity
                and can range from 0 to 100.
            - If `charge` or `spin` are not set in `atoms.info`,
                they will default to `0`.
            - The `free_energy` is simply a copy of the `energy`
                and is not the actual electronic free energy.
              It is only set for ASE routines/optimizers that are hard-coded to use
                this rather than the `energy` key.
        """

        # Our calculators won't work if natoms=0
        if len(atoms) == 0:
            raise ValueError("Atoms object has no atoms inside.")

        # Check if the atoms object has periodic boundary conditions (PBC) set correctly
        self._check_atoms_pbc(atoms)

        # Validate that charge/spin are set correctly, or default to 0 otherwise
        self._validate_charge_and_spin(atoms)

        # Standard call to check system_changes etc
        Calculator.calculate(self, atoms, properties, system_changes)

        if len(atoms) == 1 and sum(atoms.pbc) == 0:
            self._get_single_atom_energies(atoms)
        else:
            # Predict
            graph = atoms.info["graph"]
            graph = graph.tensor()
            pred = self.predictor.model.predict(graph)
            pred = self.predictor.post_process(pred)

            # Collect the results into self.results
            self.results = {}

            # energy
            if "energy" in self.implemented_properties and "energy" in pred:
                energy = float(pred["energy"].squeeze())
            elif (
                "energy_per_atom" in self.implemented_properties
                and "energy_per_atom" in pred
            ):
                energy = float(pred["energy_per_atom"].squeeze())
            else:
                raise KeyError(
                    "Neither 'energy' nor 'energy_per_atom' found in prediction."
                )
            self.results["energy"] = self.results[
                "free_energy"
            ] = energy  # Free energy is a copy of energy

            # forces, stress
            name_map = {
                "forces": "force",
            }
            for calc_key in self.implemented_properties:
                pred_key = name_map.get(calc_key, calc_key)
                if calc_key in ("force", "forces"):
                    self.results["forces"] = pred[pred_key]
                elif calc_key == "stress":
                    stress = pred[pred_key]
                    stress_voigt = full_3x3_to_voigt_6_stress(stress)
                    self.results["stress"] = stress_voigt * units.GPa

    def _check_atoms_pbc(self, atoms) -> None:
        """
        Check for invalid PBC conditions

        Args:
            atoms (ase.Atoms): The atomic structure to check.
        """
        if np.all(atoms.pbc) and np.allclose(atoms.cell, 0):
            raise AllZeroUnitCellError
        if np.any(atoms.pbc) and not np.all(atoms.pbc):
            raise MixedPBCError

    def _validate_charge_and_spin(self, atoms: Atoms) -> None:
        """
        Validate and set default values for charge and spin.

        Args:
            atoms (Atoms): The atomic structure containing charge and spin information.
        """

        if "charge" not in atoms.info:
            atoms.info["charge"] = DEFAULT_CHARGE
            logger.warning(
                "Defaulting to charge=0. "
                "Ensure charge is an integer representing the total charge "
                "on the system and is within the range -100 to 100."
            )

        if "spin" not in atoms.info:
            atoms.info["spin"] = DEFAULT_SPIN
            logger.warning(
                "Defaulting to spin=1. "
                "Ensure spin is an integer representing "
                "the spin multiplicity from 0 to 100."
            )

        # Validate charge
        charge = atoms.info["charge"]
        if not isinstance(charge, int):
            raise TypeError(
                f"Invalid type for charge: {type(charge)}. "
                f"Charge must be an integer representing "
                f"the total charge on the system."
            )
        if not (CHARGE_RANGE[0] <= charge <= CHARGE_RANGE[1]):
            raise ValueError(
                f"Invalid value for charge: {charge}. "
                f"Charge must be within the range "
                f"{CHARGE_RANGE[0]} to {CHARGE_RANGE[1]}."
            )

        # Validate spin
        spin = atoms.info["spin"]
        if not isinstance(spin, int):
            raise TypeError(
                f"Invalid type for spin: {type(spin)}. "
                f"Spin must be an integer representing "
                f"the spin multiplicity."
            )
        if not (SPIN_RANGE[0] <= spin <= SPIN_RANGE[1]):
            raise ValueError(
                f"Invalid value for spin: {spin}. "
                f"Spin must be within the range "
                f"{SPIN_RANGE[0]} to {SPIN_RANGE[1]}."
            )

    def _get_single_atom_energies(self, atoms) -> dict:
        """
        Populate output with single atom energies
        """
        raise ValueError("Single atom systems are not handled by the model.")

    def run_opt(
        self,
        structures: List[Atoms],
        optimizer: str = "LBFGS",
        filter: str = "FrechetCellFilter",
        fmax: float = 0.05,
        steps: int = 100,
    ):
        # Convert structures to ASE format
        structures = self.structure_to_ase(structures)

        # Set filter and optimizer
        optimizer_cls = OPTIMIZERS[optimizer]
        filter_cls = getattr(filters, filter) if filter != "none" else None

        # Relax
        for idx, atoms in enumerate(structures):
            formula = atoms.get_chemical_formula()
            logger.info(f"[{idx+1}/{len(structures)}] Relaxing structure: {formula}")

            # Set calculator
            atoms.calc = self
            system = filter_cls(atoms) if filter_cls else atoms

            # Set up optimizer (with logfile and trajectory)
            opt = optimizer_cls(
                system,
                logfile=f"{idx}_{formula}.log",
                trajectory=f"{idx}_{formula}.traj",
            )

            # Run optimization
            try:
                opt.run(fmax=fmax, steps=steps)
                write(f"{idx}_{formula}.xyz", atoms)
            except Exception as e:
                logger.warning(f"Optimization failed for {formula}: {e}")
                continue

    def run_md(
        self,
        structures: List[Atoms],
        temperature: float = 300,
        timestep: float = 0.1,
        steps: int = 100,
        interval: int = 1,
    ):
        # Convert structures to ASE format
        structures = self.structure_to_ase(structures)

        # Run MD
        for idx, atoms in enumerate(structures):
            formula = atoms.get_chemical_formula()
            logger.info(f"[{idx+1}/{len(structures)}] Running MD: {formula}")

            # Set the calculator
            atoms.calc = self

            dyn = Langevin(
                atoms,
                timestep=timestep * units.fs,
                temperature_K=temperature,
                friction=0.01 / units.fs,
            )

            dyn.attach(
                MDLogger(
                    dyn,
                    atoms,
                    f"{idx}_{formula}.log",
                    header=True,
                    stress=False,
                    peratom=False,
                ),
                interval=interval,
            )

            # Trajectory
            trajectory = Trajectory(f"{idx}_{formula}.traj", "w", atoms)
            dyn.attach(trajectory.write, interval=interval)
            dyn.run(steps=steps)

            # Last structure
            write(f"{idx}_{formula}_final.xyz", atoms)


class MixedPBCError(ValueError):
    """Specific exception example."""

    def __init__(
        self,
        message="Attempted to guess PBC for an atoms object, "
        "but the atoms object has PBC set to True for somedimensions but not others. "
        "Please ensure that the atoms object has PBC set to True for all dimensions.",
    ):
        self.message = message
        super().__init__(self.message)


class AllZeroUnitCellError(ValueError):
    """Specific exception example."""

    def __init__(
        self,
        message="Atoms object claims to have PBC set, "
        "but the unit cell is identically 0. Please ensure that the atoms "
        "object has a non-zero unit cell.",
    ):
        self.message = message
        super().__init__(self.message)

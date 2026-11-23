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


import os

from ase.build import bulk
from hydra.utils import instantiate
from pymatgen.io.ase import AseAtomsAdaptor

from ppmat.utils import logger


def get_structure_from_file(work_dir, system, predictor):
    """
    Reads structure data from specified file path.
    Args:
        work_dir: working directory
        system: system configuration
        predictor: predictor object for processing structure data
    """
    
    system["file_path"] = os.path.join(work_dir, system["file_path"])
    logger.info(f"Loading structures from files: {system['file_path']}.")
    files, structures = predictor.collect_structures(file_path=system["file_path"])
    return files, structures


def get_structure_from_ase(system):
    """
    Generates pymatgen Structure objects from ASE Atoms objects.
    Args:
        system: system configuration containing structure information
    """

    files, structures = [], []
    for i, s in enumerate(system["structures"]):
        element = s["element"]
        atom = bulk(element)
        if "repeat" in s:
            atom = atom.repeat(s["repeat"])
        structure = AseAtomsAdaptor().get_structure(atom)
        formula = atom.get_chemical_formula()  # Get chemical formula
        structures.append(structure)
        files.append(f"structure_{i}_{formula}")
    logger.info(f"Using ASE provided structures (count: {len(structures)})")
    return files, structures

# TODO: need to refactor later
def build_init_structures(config, predictor):
    """
    Loads structure data from either file or ASE interface according to system config
    Args:
        config: system configuration
        predictor: Predictor object for processing structure data
    """

    system = instantiate(config["System"])
    if system["interface"] == "load_file":
        work_dir = config["Run"]["work_dir"]
        files, structures = get_structure_from_file(work_dir, system, predictor)
    elif system["interface"] == "ase":
        files, structures = get_structure_from_ase(system)
    else:
        raise ValueError(
            f"Unsupported System.interface: {system['interface']!r} now."
        )
    return files, structures

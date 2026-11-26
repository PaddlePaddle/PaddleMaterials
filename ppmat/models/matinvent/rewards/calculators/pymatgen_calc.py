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

"""Pymatgen-backed material properties for the MatInvent reward.

Every helper returns one value per structure and uses NaN to mark failures, so
``Reward`` can drop failed candidates without aborting a whole batch.
"""

import os

import numpy as np
from pymatgen.analysis.cost import CostAnalyzer
from pymatgen.analysis.cost import CostDBElements
from pymatgen.analysis.hhi import HHIModel
from pymatgen.analysis.interfaces.substrate_analyzer import SubstrateAnalyzer
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from ppmat.datasets.build_structure import BuildStructure
from ppmat.utils import logger

# Preferred miller indices for common substrates used by lattice matching.
SUB_MILLERS = {"Si": [(1, 0, 0)], "GaAs": [(1, 0, 0)], "InP": [(1, 0, 0)]}


def _per_structure(structures, fn):
    """Apply ``fn`` to every structure, mapping a failure to NaN and logging it."""
    results = []
    for structure in structures:
        try:
            results.append(fn(structure))
        except (KeyError, NameError, TypeError, ValueError) as error:
            logger.warning(f"Property computation failed: {error}")
            results.append(np.nan)
    return np.array(results, dtype=float)


def _abundance(structure):
    from smact import Element as SmactElement

    weighted = sum(
        weight * SmactElement(element).crustal_abundance
        for element, weight in structure.composition.to_weight_dict.items()
    )
    return np.nan if weighted <= 0.0 else weighted


def compute_density(structures):
    """Return the mass density of each structure in g/cm^3."""
    return _per_structure(structures, lambda s: s.density)


def compute_volume(structures):
    """Return the cell volume of each structure."""
    return _per_structure(structures, lambda s: s.volume)


def compute_num_atoms(structures):
    """Return the number of sites of each structure."""
    return _per_structure(structures, lambda s: float(len(s)))


def compute_num_elements(structures):
    """Return the number of distinct elements of each structure."""
    return _per_structure(structures, lambda s: float(len(s.composition.elements)))


def compute_abundance(structures):
    """Return the composition-weighted crustal abundance of each structure."""
    return _per_structure(structures, _abundance)


def compute_log_abundance(structures):
    """Return the base-10 logarithm of the crustal abundance."""
    return np.log10(_per_structure(structures, _abundance))


def compute_hhi(structures):
    """Return the Herfindahl-Hirschman index (HHI) of the reserve supply."""
    model = HHIModel()
    return _per_structure(
        structures, lambda s: model.get_hhi_reserve(s.composition) or np.nan
    )


def compute_price(structures):
    """Return the estimated material cost per kilogram."""
    analyzer = CostAnalyzer(CostDBElements())
    return _per_structure(structures, lambda s: analyzer.get_cost_per_kg(s.composition))


def compute_mcia(structures, substrate, millers=None):
    """Return the minimum lattice match area against a substrate.

    Args:
        structures: Film structures to match.
        substrate: pymatgen ``Structure`` of the substrate.
        millers: Optional substrate miller indices. When omitted, the values
            in :data:`SUB_MILLERS` are used for known substrates.
    """
    analyzer = SubstrateAnalyzer(film_max_miller=1, substrate_max_miller=1)
    substrate = SpacegroupAnalyzer(
        substrate, symprec=0.1
    ).get_conventional_standard_structure()
    formula = substrate.composition.reduced_formula
    if millers is None and formula in SUB_MILLERS:
        millers = SUB_MILLERS[formula]

    def min_match_area(structure):
        film = SpacegroupAnalyzer(
            structure, symprec=0.1
        ).get_conventional_standard_structure()
        matches = analyzer.calculate(
            film=film, substrate=substrate, substrate_millers=millers, lowest=True
        )
        return min(match.match_area for match in matches)

    return _per_structure(structures, min_match_area)


# Registry of single-structure property tasks keyed by their public task name.
PROPERTY_FUNCS = {
    "density": compute_density,
    "volume": compute_volume,
    "num_atoms": compute_num_atoms,
    "num_elements": compute_num_elements,
    "abundance": compute_abundance,
    "log_abundance": compute_log_abundance,
    "hhi": compute_hhi,
    "price": compute_price,
}


def compute_property(structures, task, substrate=None):
    """Compute ``task`` for every structure in ``structures``.

    Args:
        structures: Sequence of pymatgen ``Structure`` objects.
        task: One of :data:`PROPERTY_FUNCS` or ``"mcia"``.
        substrate: Substrate ``Structure``, required by ``"mcia"``.

    Returns:
        A float ``numpy.ndarray`` with one value per structure.
    """
    if task == "mcia":
        if substrate is None:
            raise ValueError(
                "The 'mcia' task requires a substrate Structure, but got None"
            )
        return compute_mcia(structures, substrate)
    if task not in PROPERTY_FUNCS:
        raise ValueError(
            f"Unknown task: {task}, available: {sorted([*PROPERTY_FUNCS, 'mcia'])}"
        )
    return PROPERTY_FUNCS[task](structures)


class PyMatGen:
    """Score a pymatgen property and keep the per-batch values on disk."""

    def __init__(self, root_dir, task="density", substrate_path=None):
        self.root_dir = root_dir
        self.task = task
        os.makedirs(root_dir, exist_ok=True)
        self.substrate_path = substrate_path
        self._substrate = None

    def _load_substrate(self):
        """Parse the substrate CIF required by the ``mcia`` task."""
        if self.substrate_path is None:
            raise ValueError(
                "The 'mcia' task requires substrate_path, a path to a substrate CIF"
            )
        if self._substrate is None:
            self._substrate = BuildStructure.build_one(
                self.substrate_path, "cif_file", niggli=False, canocial=False
            )
        return self._substrate

    def calc(self, structures, label="tmp"):
        substrate = self._load_substrate() if self.task == "mcia" else None
        results = compute_property(structures, self.task, substrate=substrate)
        out_path = os.path.abspath(os.path.join(self.root_dir, f"{label}.txt"))
        np.savetxt(out_path, results, fmt="%.8f")
        return results

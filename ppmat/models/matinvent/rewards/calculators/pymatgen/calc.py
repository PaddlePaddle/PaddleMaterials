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

import os

import numpy as np
from pymatgen.analysis.cost import CostAnalyzer
from pymatgen.analysis.cost import CostDBElements
from pymatgen.analysis.hhi import HHIModel
from pymatgen.analysis.interfaces.substrate_analyzer import SubstrateAnalyzer
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from ppmat.datasets.build_structure import BuildStructure
from ppmat.models.matinvent.rewards.calculators import Calculator
from ppmat.utils import logger

SUB_MILLERS = {"Si": [(1, 0, 0)], "GaAs": [(1, 0, 0)], "InP": [(1, 0, 0)]}


def _per_structure(strucs, fn):
    """Apply ``fn`` to every structure, mapping a failure to NaN and logging it."""
    results = []
    for struc in strucs:
        try:
            results.append(fn(struc))
        except (KeyError, NameError, TypeError, ValueError) as error:
            logger.warning(f"Property computation failed: {error}")
            results.append(np.nan)
    return np.array(results, dtype=float)


def _abundance(struc):
    from smact import Element as SmactElement

    weighted = sum(
        weight * SmactElement(element).crustal_abundance
        for element, weight in struc.composition.to_weight_dict.items()
    )
    return np.nan if weighted <= 0.0 else weighted


def _hhi(strucs):
    model = HHIModel()
    return _per_structure(
        strucs, lambda s: model.get_hhi_reserve(s.composition) or np.nan
    )


def _price(strucs):
    analyzer = CostAnalyzer(CostDBElements())
    return _per_structure(strucs, lambda s: analyzer.get_cost_per_kg(s.composition))


def _mcia(strucs, substrate, millers=None):
    analyzer = SubstrateAnalyzer(film_max_miller=1, substrate_max_miller=1)
    substrate = SpacegroupAnalyzer(
        substrate, symprec=0.1
    ).get_conventional_standard_structure()
    formula = substrate.composition.reduced_formula
    if millers is None and formula in SUB_MILLERS:
        millers = SUB_MILLERS[formula]

    def min_match_area(struc):
        film = SpacegroupAnalyzer(
            struc, symprec=0.1
        ).get_conventional_standard_structure()
        matches = analyzer.calculate(
            film=film, substrate=substrate, substrate_millers=millers, lowest=True
        )
        return min(match.match_area for match in matches)

    return _per_structure(strucs, min_match_area)


class PyMatGen(Calculator):
    """On-the-fly property calculator backed by pymatgen."""

    def __init__(self, root_dir, task="density", substrate_path=None):
        super().__init__(root_dir, task)
        self.substrate_path = substrate_path
        self._tasks = {
            "density": lambda x: np.array([s.density for s in x]),
            "hhi": _hhi,
            "price": _price,
            "abundance": lambda x: _per_structure(x, _abundance),
            "log_abundance": lambda x: np.log10(_per_structure(x, _abundance)),
            "mcia": lambda x: _mcia(x, self._load_substrate()),
            "num_atoms": lambda x: np.array([len(s) for s in x], dtype=float),
            "num_elements": lambda x: np.array(
                [len(s.composition.elements) for s in x], dtype=float
            ),
            "volume": lambda x: np.array([s.volume for s in x], dtype=float),
        }

    def _load_substrate(self):
        """Parse the substrate CIF required by the ``mcia`` task."""
        if self.substrate_path is None:
            raise ValueError(
                "The 'mcia' task requires substrate_path, a path to a substrate CIF"
            )
        return BuildStructure.build_one(
            self.substrate_path, "cif_file", niggli=False, canocial=False
        )

    def calc(self, samples, label="tmp"):
        struc_list = samples[0]
        fn = self._tasks.get(self.task)
        if fn is None:
            raise ValueError(
                f"Unknown task: {self.task}, available: {sorted(self._tasks)}"
            )
        results = fn(struc_list)
        out_path = os.path.abspath(os.path.join(self.root_dir, f"{label}.txt"))
        np.savetxt(out_path, results, fmt="%.8f")
        return results


__all__ = ["PyMatGen"]

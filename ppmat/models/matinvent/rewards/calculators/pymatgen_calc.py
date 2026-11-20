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

from ppmat.calculator import pymatgen as pymatgen_calc
from ppmat.datasets.build_structure import BuildStructure


class PyMatGen:
    """Reward adapter around :mod:`ppmat.calculator.pymatgen`.

    This glue belongs to the MatInvent reward layer: it resolves the per-property
    output layout and delegates the actual material-property computation to the
    shared, model-agnostic helpers in :mod:`ppmat.calculator.pymatgen`.
    """

    def __init__(self, root_dir, task="density", substrate_path=None):
        self.root_dir = root_dir
        self.task = task
        self.substrate_path = substrate_path
        os.makedirs(root_dir, exist_ok=True)
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

    def calc(self, samples, label="tmp"):
        structures = samples[0]
        substrate = self._load_substrate() if self.task == "mcia" else None
        results = pymatgen_calc.compute_property(
            structures, self.task, substrate=substrate
        )
        out_path = os.path.abspath(os.path.join(self.root_dir, f"{label}.txt"))
        np.savetxt(out_path, results, fmt="%.8f")
        return results


__all__ = ["PyMatGen"]

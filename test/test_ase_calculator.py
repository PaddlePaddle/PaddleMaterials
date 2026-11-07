# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

import numpy as np
import pytest
from ase import Atoms
from pymatgen.core import Lattice
from pymatgen.core import Structure

from ppmat.calculator.ase import ASECalculator


class StructureGraph:
    def __init__(self, structure):
        self.positions = np.array(structure.cart_coords)
        self.cell = np.array(structure.lattice.matrix)

    def tensor(self):
        return self


class GeometryPredictor:
    def __init__(self, model_name):
        self.config = {
            "Global": {"label_names": ["energy", "force"]},
            "Model": {"__class_name__": model_name},
        }
        self.model = self

    def graph_converter(self, structure):
        return StructureGraph(structure)

    def predict(self, graph):
        energy = np.sum(graph.positions**2) + np.sum(graph.cell**2)
        return {"energy": np.array([energy]), "force": -2 * graph.positions}

    def post_process(self, prediction):
        return prediction


@pytest.mark.parametrize("model_name", ["CHGNet", "M3GNet"])
def test_calculator_uses_current_positions_and_cell(model_name):
    structure = Structure(Lattice.cubic(5), ["Li", "Li"], [[0, 0, 0], [0.2, 0, 0]])
    calc = ASECalculator(GeometryPredictor(model_name))
    atoms = calc.structure_to_ase([structure])[0]
    atoms.info.update(charge=0, spin=0)
    atoms.calc = calc

    assert atoms.get_potential_energy() == pytest.approx(76.0)
    atoms.positions[1, 0] += 1.0
    assert atoms.get_potential_energy() == pytest.approx(79.0)
    np.testing.assert_allclose(atoms.get_forces()[1], [-4.0, 0.0, 0.0])

    atoms.set_cell(atoms.cell * 1.1, scale_atoms=True)
    assert atoms.get_potential_energy() == pytest.approx(95.59)
    np.testing.assert_allclose(atoms.get_forces()[1], [-4.4, 0.0, 0.0])


@pytest.mark.parametrize("model_name", ["CHGNet", "M3GNet"])
def test_calculator_accepts_ase_atoms_without_a_stored_graph(model_name):
    atoms = Atoms("Li2", positions=[[0, 0, 0], [1, 0, 0]], cell=[5, 5, 5], pbc=True)
    atoms.info.update(charge=0, spin=0)
    atoms.calc = ASECalculator(GeometryPredictor(model_name))

    assert atoms.get_potential_energy() == pytest.approx(76.0)
    np.testing.assert_allclose(atoms.get_forces()[1], [-2.0, 0.0, 0.0])

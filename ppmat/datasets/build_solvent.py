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
import pgl
from rdkit.Chem import rdMolDescriptors

from ppmat.datasets.build_molecule import BuildMolecule


class BuildSolvent:
    """Build molecular graphs and hydrogen-bond descriptors for solvents.

    Args:
        graph_converter: Converter built by ``build_graph_converter``.
        build_molecule_cfg: Options for ``BuildMolecule``. Defaults to SMILES.
    """

    def __init__(self, graph_converter, build_molecule_cfg=None):
        self.graph_converter = graph_converter
        self.build_molecule = BuildMolecule(
            **(build_molecule_cfg or {"format": "smiles"})
        )

    def __call__(self, molecule_data):
        molecule = self.build_molecule(molecule_data)
        if molecule is None:
            raise ValueError(f"Invalid molecule: {molecule_data}")
        graph = self.graph_converter(molecule)
        if graph is None:
            raise ValueError(f"Failed to build molecular graph: {molecule_data}")
        hba = rdMolDescriptors.CalcNumHBA(molecule)
        hbd = rdMolDescriptors.CalcNumHBD(molecule)
        return {"graph": graph, "hba": hba, "hbd": hbd, "intra_hb": min(hba, hbd)}


class BuildBinaryMixture:
    """Encode two built solvents, composition and optional log-activity labels."""

    def __call__(self, solvent1, solvent2, x1, gamma=None):
        x1 = float(x1)
        if not np.isfinite(x1) or not 0.0 <= x1 <= 1.0:
            raise ValueError("Mole fraction x1 must be finite and in [0, 1].")

        def column(value):
            return np.asarray([value], dtype="float32")

        sample = {
            "g1": solvent1["graph"],
            "g2": solvent2["graph"],
            "x1": column(x1),
            "x2": column(1.0 - x1),
            "intra_hb1": column(solvent1["intra_hb"]),
            "intra_hb2": column(solvent2["intra_hb"]),
            "inter_hb": column(
                min(solvent1["hba"], solvent2["hbd"])
                + min(solvent1["hbd"], solvent2["hba"])
            ),
            "empty_solvsys": self.generate_solvsys(),
        }
        if gamma is not None:
            sample["gamma1"] = column(float(gamma[0]))
            sample["gamma2"] = column(float(gamma[1]))
            sample["gamma"] = np.asarray(gamma, dtype="float32")
        return sample

    @staticmethod
    def generate_solvsys():
        return pgl.Graph(
            num_nodes=2,
            edges=np.asarray([[0, 1], [1, 0], [0, 0], [1, 1]], dtype=np.int64),
        )

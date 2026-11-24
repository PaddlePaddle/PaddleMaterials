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

"""Encode cached molecular arrays into PropMolFlow training fields."""

from __future__ import annotations

import numpy as np

from ppmat.datasets.custom_data_type import ConcatData
from ppmat.models.propmolflow.mol_converter import CHARGE_OFFSET
from ppmat.models.propmolflow.mol_converter import N_BOND_TYPES
from ppmat.models.propmolflow.mol_converter import N_CHARGES

__all__ = ["PropMolFlowStructure"]


class PropMolFlowStructure:
    """Encode the raw graph produced by PropMolFlowMolConverter.

    Centers coordinates and one-hot encodes atom, charge and upper-triangle
    bond classes. ``property_name`` identifies the normalized conditioning
    value, which is exposed as ``prop`` to the training model.
    """

    def __init__(self, atom_map: list, property_name: str = "prop"):
        self.n_atom_types = len(atom_map)
        self.property_name = property_name

    def __call__(self, data: dict) -> dict:
        molecule = data.pop("graph")
        if molecule is None:
            raise ValueError("PropMolFlowStructure requires a featurized molecule")
        positions = molecule["positions"].astype("float32")
        positions = positions - positions.mean(axis=0, keepdims=True)
        structure_array = {
            "num_atoms": np.array([molecule["num_atoms"]], dtype="int64"),
            "positions": positions,
            "atom_types": np.eye(self.n_atom_types, dtype="float32")[
                molecule["atom_types"]
            ],
            "atom_charges": np.eye(N_CHARGES, dtype="float32")[
                molecule["charges"] + CHARGE_OFFSET
            ],
            "bond_types": np.eye(N_BOND_TYPES, dtype="float32")[molecule["bond_types"]],
        }
        data["structure_array"] = {
            key: ConcatData(value) for key, value in structure_array.items()
        }
        data["prop"] = np.asarray(data.pop(self.property_name)).item()
        return data

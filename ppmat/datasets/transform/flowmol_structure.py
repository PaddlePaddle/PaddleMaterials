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

"""Encode raw RDKit-derived arrays into the FlowMol training fields."""

from __future__ import annotations

from typing import Optional
from typing import Sequence

import numpy as np
from rdkit import Chem

from ppmat.datasets.custom_data_type import ConcatData
from ppmat.models.flowmol.graph_utils import align_position_prior

__all__ = ["FlowMolStructure"]


class FlowMolStructure:
    """Assemble FlowMol ``structure_array`` and ``prior_positions`` from a molecule.

    Consumes ``data["graph"]`` produced by ``FlowMolMolConverter`` (raw positions,
    atomic numbers, formal charges and bonds) and mirrors the encoding of the
    model endpoint fields while resampling augmentation and priors per access.

    Args:
        atom_map (Sequence[str]): Atom symbols in one-hot order.
        fake_atom_p (float): Probability used to append up to
            ``ceil(num_atoms * fake_atom_p)`` fake atoms. Defaults to 0.0.
        fake_atom_std (float): Standard deviation of the fake-atom offsets.
            Defaults to 1.0.
        prior_std (float): Standard deviation of the Gaussian position prior.
            Defaults to 1.0.
        prior_align (bool): Whether to align the prior to the endpoint with
            Hungarian matching and Kabsch. Defaults to True.
        seed (Optional[int]): Random seed for fake atoms and the position prior.
            Defaults to None.
        n_charges (int): Number of formal charge classes. Defaults to 6.
        n_bond_types (int): Number of bond classes, including absent bonds.
            Defaults to 4.
        charge_offset (int): Offset for formal charge indices. Defaults to 2.
    """

    def __init__(
        self,
        atom_map: Sequence[str],
        fake_atom_p: float = 0.0,
        fake_atom_std: float = 1.0,
        prior_std: float = 1.0,
        prior_align: bool = True,
        seed: Optional[int] = None,
        n_charges: int = 6,
        n_bond_types: int = 4,
        charge_offset: int = 2,
    ) -> None:
        self.atom_map = list(atom_map)
        self.symbol_to_idx = {symbol: i for i, symbol in enumerate(self.atom_map)}
        self.n_atom_types = len(self.atom_map)
        self.fake_atom_p = fake_atom_p
        self.fake_atom_std = fake_atom_std
        self.use_fake_atoms = fake_atom_p > 0
        self.prior_std = prior_std
        self.prior_align = prior_align
        self.rng = np.random.RandomState(seed)
        self.n_charges = n_charges
        self.n_bond_types = n_bond_types
        self.charge_offset = charge_offset
        self._periodic_table = Chem.GetPeriodicTable()

    def _atom_type_idx(self, atomic_numbers: np.ndarray) -> np.ndarray:
        idx = []
        for z in atomic_numbers:
            symbol = self._periodic_table.GetElementSymbol(int(z))
            if symbol not in self.symbol_to_idx:
                raise ValueError(
                    f"Atom symbol {symbol!r} is not covered by atom_map "
                    f"{self.atom_map}."
                )
            idx.append(self.symbol_to_idx[symbol])
        return np.asarray(idx, dtype="int64")

    def __call__(self, data: dict) -> dict:
        molecule = data.pop("graph")
        if molecule is None:
            raise ValueError("FlowMolStructure expects a non-empty 'graph' field.")

        positions = molecule["positions"].astype("float32")
        atom_types_idx = self._atom_type_idx(molecule["atomic_numbers"])
        atom_charges_idx = molecule["formal_charges"].astype("int64")
        n_real_atoms = positions.shape[0]

        if self.use_fake_atoms:
            max_num_fake_atoms = int(np.ceil(n_real_atoms * self.fake_atom_p))
            num_fake_atoms = int(self.rng.randint(0, max_num_fake_atoms + 1))
            anchor_atom_idxs = self.rng.randint(0, n_real_atoms, num_fake_atoms)
            fake_positions = positions[anchor_atom_idxs]
            fake_positions = fake_positions + (
                self.rng.randn(*fake_positions.shape) * self.fake_atom_std
            ).astype("float32")

            positions = np.concatenate((positions, fake_positions), axis=0)
            atom_types_idx = np.concatenate(
                (
                    atom_types_idx,
                    np.full(num_fake_atoms, self.n_atom_types, dtype="int64"),
                ),
                axis=0,
            )
            atom_charges_idx = np.concatenate(
                (atom_charges_idx, np.zeros(num_fake_atoms, dtype="int64")),
                axis=0,
            )

        positions = positions - positions.mean(axis=0, keepdims=True)

        n_atoms = positions.shape[0]
        atom_types = np.eye(
            self.n_atom_types + int(self.use_fake_atoms), dtype="float32"
        )[atom_types_idx]
        charge_idx = atom_charges_idx + self.charge_offset
        if np.any((charge_idx < 0) | (charge_idx >= self.n_charges)):
            raise ValueError(
                "Formal charges are outside the configured charge classes."
            )
        atom_charges = np.eye(self.n_charges, dtype="float32")[charge_idx]

        adjacency = np.zeros((n_atoms, n_atoms), dtype="int64")
        bond_index = molecule["bond_index"]
        if bond_index.shape[0] > 0:
            adjacency[bond_index[:, 0], bond_index[:, 1]] = molecule["bond_class"]
            adjacency[bond_index[:, 1], bond_index[:, 0]] = molecule["bond_class"]
        upper_i, upper_j = np.triu_indices(n_atoms, k=1)
        bond_types = np.eye(self.n_bond_types, dtype="float32")[
            adjacency[upper_i, upper_j]
        ]

        prior_positions = self.rng.randn(n_atoms, 3).astype("float32") * self.prior_std
        prior_positions = prior_positions - prior_positions.mean(axis=0, keepdims=True)
        if self.prior_align:
            prior_positions = align_position_prior(prior_positions, positions)

        data["structure_array"] = {
            "num_atoms": ConcatData(np.array([n_atoms], dtype="int64")),
            "positions": ConcatData(positions),
            "atom_types": ConcatData(atom_types),
            "atom_charges": ConcatData(atom_charges),
            # upper and lower triangle edges of a pair share the bond one-hot
            "bond_types": ConcatData(np.concatenate((bond_types, bond_types), axis=0)),
        }
        data["prior_positions"] = ConcatData(prior_positions)
        return data

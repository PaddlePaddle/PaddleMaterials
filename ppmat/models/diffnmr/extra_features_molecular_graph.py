# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

import paddle

from ppmat.models.diffnmr.utils import diffgraphformer_utils


class ExtraMolecularFeatures:
    output_dims = {"X": 2, "E": 0, "y": 1}

    def __init__(self, dataset_infos):
        """
        dataset_infos:
          - remove_h: whether to remove hydrogens (bool)
          - valencies: dict or list, maximum/possible valences for each atom
          - max_weight: maximum value used for normalizing molecular weight
          - atom_weights: dict, recording the atomic weight of each type of atom
            (such as {'H':1, 'C':12, ...})
        """
        self.charge = ChargeFeature(
            remove_h=dataset_infos.remove_h, valencies=dataset_infos.valencies
        )
        self.valency = ValencyFeature()
        self.weight = WeightFeature(
            max_weight=dataset_infos.max_weight, atom_weights=dataset_infos.atom_weights
        )

    def __call__(self, noisy_data):
        """
        Calculate and concatenate molecule/atom additional features:
        Atomic charge (charge)
        Atomic actual valence (valency)
        Molecular weight (weight) Return diffgraphformer_utils.PlaceHolder in (X, E, y)
            format.
        """
        # charge / valency => (bs, n, 1)
        charge = paddle.unsqueeze(self.charge(noisy_data), axis=-1)
        valency = paddle.unsqueeze(self.valency(noisy_data), axis=-1)
        # weight => (bs, 1)
        weight = self.weight(noisy_data)

        # Edge-level additional features are defaulted to empty  (bs, n, n, 0)
        E_t = noisy_data["E_t"]
        extra_edge_attr = paddle.zeros(shape=E_t.shape[:-1] + [0], dtype=E_t.dtype)

        # Concatenate charge and valence to the last dimension X: (bs, n, 2)
        x_cat = paddle.concat([charge, valency], axis=-1)

        return diffgraphformer_utils.PlaceHolder(X=x_cat, E=extra_edge_attr, y=weight)


class ChargeFeature:
    def __init__(self, remove_h, valencies):
        self.remove_h = remove_h
        self.valencies = valencies

    def __call__(self, noisy_data):
        """
        Estimate the net charge of each atom = (ideal valence - current bonding number).
        bond_orders = [0, 1, 2, 3, 1.5]
        represent: no bond / single bond / double bond / triple bond / aromatic bond.
        """
        E_t = noisy_data["E_t"]
        dtype_ = E_t.dtype

        bond_orders = paddle.to_tensor([0, 1, 2, 3, 1.5], dtype=dtype_)
        bond_orders = paddle.reshape(bond_orders, [1, 1, 1, -1])  # (1,1,1,5)

        # E_t * bond_orders => (bs, n, n, de)，取 argmax => (bs, n, n)，再 sum => (bs,n)
        weighted_E = E_t * bond_orders
        current_valencies = paddle.argmax(weighted_E, axis=-1)  # (bs, n, n)
        current_valencies = paddle.sum(current_valencies, axis=-1)  # (bs, n)

        # Calculate ideal valence state
        X_t = noisy_data["X_t"]
        valency_tensor = paddle.to_tensor(
            self.valencies, dtype=X_t.dtype
        )  # shape (dx,)
        valency_tensor = paddle.reshape(valency_tensor, [1, 1, -1])  # (1,1,dx)
        X_val = X_t * valency_tensor  # (bs, n, dx)
        normal_valencies = paddle.argmax(X_val, axis=-1)  # (bs, n)

        # Charge = (Ideal Valence - Current Bonding Number)
        charge = normal_valencies - current_valencies
        return charge.astype(X_t.dtype)


class ValencyFeature:
    def __init__(self):
        pass

    def __call__(self, noisy_data):
        """
        Calculate the actual valence state of each atom
        (determined solely by the current bond type).
        """
        E_t = noisy_data["E_t"]
        dtype_ = E_t.dtype
        orders = paddle.to_tensor([0, 1, 2, 3, 1.5], dtype=dtype_)
        orders = paddle.reshape(orders, [1, 1, 1, -1])  # (1,1,1,5)

        E_weighted = E_t * orders  # (bs, n, n, de)
        valencies = paddle.argmax(E_weighted, axis=-1)  # (bs, n, n)
        valencies = paddle.sum(valencies, axis=-1)  # (bs, n)

        X_t = noisy_data["X_t"]
        return valencies.astype(X_t.dtype)


class WeightFeature:
    def __init__(self, max_weight, atom_weights):
        """Set weights for each type of atom based on their atomic weight.

        Args:
            max_weight (Int): Max weight of atom
                to normalize the total molecular mass.
            atom_weights (Dict): Atomic weight of each atom.
        """
        self.max_weight = max_weight
        self.atom_weight_list = paddle.to_tensor(
            list(atom_weights.values()), dtype="float32"
        )

    def __call__(self, noisy_data):
        X = paddle.argmax(noisy_data["X_t"], axis=-1)  # (bs, n)
        X_weights = self.atom_weight_list[X]  # (bs, n)
        return (
            X_weights.sum(axis=-1).unsqueeze(-1).astype(noisy_data["X_t"].dtype)
            / self.max_weight
        )  # (bs, 1)

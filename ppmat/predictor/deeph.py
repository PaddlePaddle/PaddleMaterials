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

from __future__ import annotations

import json
import os
import os.path as osp

import numpy as np

from ppmat.datasets.build_structure import BuildStructure
from ppmat.datasets.collate_fn import DeepHCollator
from ppmat.datasets.deeph_dataset import DeepHData
from ppmat.datasets.graph_utils.deeph_io import load_deeph_orbital_counts
from ppmat.models import build_graph_converter
from ppmat.predictor.base import BasePredictor


def build_deeph_hamiltonian_blocks(prediction, model_input, orbital, spinful=False):
    """Restore edge predictions to DeepH ``rh.npz`` Hamiltonian blocks."""
    import paddle

    if spinful:
        raise NotImplementedError("Spinful DeepH Hamiltonian export is not enabled.")

    def to_numpy(value):
        return value.numpy() if paddle.is_tensor(value) else np.asarray(value)

    folder = model_input["structure_folder"]
    if isinstance(folder, (list, tuple)):
        if len(folder) != 1:
            raise ValueError("DeepH Hamiltonian export expects batch_size=1.")
        folder = folder[0]
    if not isinstance(folder, str):
        raise TypeError("DeepH Hamiltonian export requires structure_folder.")

    orbital_counts = load_deeph_orbital_counts(osp.join(folder, "orbital_types.dat"))
    prediction = to_numpy(prediction)
    edge_index = to_numpy(model_input["edge_index"]).astype(np.int64)
    if edge_index.ndim == 2 and edge_index.shape[0] != 2:
        edge_index = edge_index.T
    edge_attr = to_numpy(model_input["edge_attr"])
    atomic_numbers = to_numpy(model_input["structure_atomic_numbers"]).astype(np.int64)
    lattice = to_numpy(model_input["structure_lattice"]).reshape(-1, 3, 3)[0]
    inverse_lattice = np.linalg.inv(lattice)
    lattice_shifts = np.rint(
        edge_attr[:, 4:7] @ inverse_lattice - edge_attr[:, 7:10] @ inverse_lattice
    ).astype(int)

    hamiltonian = {}
    for edge_id in range(edge_attr.shape[0]):
        atom_i, atom_j = edge_index[:, edge_id]
        block = np.zeros(
            (orbital_counts[atom_i], orbital_counts[atom_j]),
            dtype=prediction.dtype,
        )
        number_i = int(atomic_numbers[atom_i])
        number_j = int(atomic_numbers[atom_j])
        for output_index, orbital_dict in enumerate(orbital):
            for species_pair, orbital_pair in orbital_dict.items():
                expected_i, expected_j = map(int, species_pair.split())
                if number_i == expected_i and number_j == expected_j:
                    orbital_i, orbital_j = orbital_pair
                    block[orbital_i, orbital_j] = prediction[edge_id, output_index]
        key = json.dumps(
            [
                int(lattice_shifts[edge_id, 0]),
                int(lattice_shifts[edge_id, 1]),
                int(lattice_shifts[edge_id, 2]),
                int(atom_i) + 1,
                int(atom_j) + 1,
            ]
        )
        hamiltonian[key] = block
    return hamiltonian


def save_deeph_hamiltonian(path, prediction, model_input, orbital, spinful=False):
    """Write restored DeepH Hamiltonian blocks to a compressed NPZ file."""
    hamiltonian = build_deeph_hamiltonian_blocks(
        prediction, model_input, orbital, spinful=spinful
    )
    path = osp.abspath(path)
    os.makedirs(osp.dirname(path), exist_ok=True)
    np.savez(path, **hamiltonian)


class DeepHPredictor(BasePredictor):
    """Predict DeepH Hamiltonian blocks and export them in DeepH npz format."""

    def modify_model_config(self, interface_type, model_config):
        """Keep the DeepH model contract for the ppmatSim DFT interface."""
        if interface_type == "deeph_dft":
            return model_config
        return super().modify_model_config(interface_type, model_config)

    def build_inference_batch(
        self,
        processed_dir,
        species,
        num_l,
        dtype="float32",
    ):
        """Build a label-free DeepH batch from overlap-derived local coordinates."""
        structure = BuildStructure(
            format="deeph", primitive=False, niggli=False, canocial=False
        )(processed_dir)
        np_dtype = np.dtype(dtype).type
        graph_converter = build_graph_converter(
            {
                "__class_name__": "DeepHGraphConverter",
                "__init_params__": {
                    "radius": -1.0,
                    "max_num_nbr": 0,
                    "default_dtype": np_dtype,
                    "interface": "npz_rc_only",
                    "num_l": num_l,
                    "create_from_DFT": True,
                    "if_lcmp_graph": True,
                    "separate_onsite": False,
                    "target": "hamiltonian",
                },
            }
        )
        graph = graph_converter(structure, processed_dir)
        model_input = graph_converter.to_model_input(
            graph, structure, processed_dir, species=species
        )
        return DeepHCollator()([DeepHData(num_nodes=len(structure), **model_input)])

    def predict_hamiltonian(
        self,
        processed_dir,
        output_path,
        orbital,
        species,
        num_l,
        dtype="float32",
    ):
        """Run Paddle DeepH inference and save predicted Hamiltonian blocks."""
        batch = self.build_inference_batch(
            processed_dir=processed_dir,
            species=species,
            num_l=num_l,
            dtype=dtype,
        )
        prediction = self._run_model(batch)
        pred = prediction[self.model.target_name]
        save_deeph_hamiltonian(
            output_path,
            pred,
            batch,
            orbital,
            spinful=False,
        )
        return output_path

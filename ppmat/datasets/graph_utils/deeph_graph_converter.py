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

import os
from itertools import product

import numpy as np
from pymatgen.core.structure import Structure
from pymatgen.io.ase import AseAtomsAdaptor

from ppmat.datasets.graph_utils.deeph_graph_utils import build_deeph_graph
from ppmat.datasets.graph_utils.deeph_graph_utils import periodic_image_ranges
from ppmat.models.common.graph_converter import FindPointsInSpheres


class DeepHGraphConverter:
    """Build DeepH dataset features using the common periodic graph converter."""

    def __init__(
        self,
        radius,
        max_num_nbr,
        default_dtype,
        interface,
        num_l,
        create_from_DFT=False,
        if_lcmp_graph=True,
        separate_onsite=False,
        target="hamiltonian",
        huge_structure=False,
        if_new_sp=False,
        orbital=None,
        species=None,
        neighbor_cfg=None,
    ):
        self.radius = radius
        self.max_num_nbr = max_num_nbr
        self.default_dtype = np.dtype(default_dtype).type
        self.orbital = orbital
        self.species = species
        self.neighbor_converter = (
            None
            if create_from_DFT
            else FindPointsInSpheres(**(neighbor_cfg or {"cutoff": radius}))
        )
        self.interface = interface
        self.num_l = num_l
        self.create_from_DFT = create_from_DFT
        self.if_lcmp_graph = if_lcmp_graph
        self.separate_onsite = separate_onsite
        self.target = target
        self.huge_structure = huge_structure
        self.if_new_sp = if_new_sp
        if max_num_nbr > 0:
            raise NotImplementedError(
                "DeepH reuses the PaddleMaterials periodic graph converter and "
                "currently expects max_num_nbr=0."
            )

    @staticmethod
    def _to_structure(structure):
        if isinstance(structure, Structure):
            return structure
        return AseAtomsAdaptor.get_structure(structure)

    def __call__(
        self,
        structure,
        folder,
        converter_graph=None,
        only_periodic_images=False,
    ):
        structure = self._to_structure(structure)
        default_dtype = self.default_dtype
        if only_periodic_images:
            ranges = periodic_image_ranges(
                np.asarray(structure.frac_coords),
                np.asarray(structure.lattice.matrix),
                self.radius,
            )
            return np.asarray(list(product(*ranges)), dtype=default_dtype)
        if converter_graph is None and self.neighbor_converter is not None:
            converter_graph = self.neighbor_converter(structure)
        graph = build_deeph_graph(
            np.asarray(structure.cart_coords, dtype=default_dtype),
            np.asarray(structure.atomic_numbers, dtype=np.int64),
            os.path.basename(folder),
            lattice=np.asarray(structure.lattice.matrix, dtype=default_dtype),
            default_dtype=default_dtype,
            tb_folder=folder,
            interface=self.interface,
            num_l=self.num_l,
            create_from_DFT=self.create_from_DFT,
            if_lcmp_graph=self.if_lcmp_graph,
            separate_onsite=self.separate_onsite,
            target=self.target,
            huge_structure=self.huge_structure,
            if_new_sp=self.if_new_sp,
            converter_graph=converter_graph,
        )

        if self.orbital is not None and self.interface == "npz":
            self._make_hamiltonian_label(graph)
        return graph

    def _make_hamiltonian_label(self, graph):
        """Select supervised orbital entries from Hamiltonian matrix blocks."""
        if not np.all(graph.term_mask):
            raise ValueError("Graph contains edges without Hamiltonian labels.")
        orbital = self.orbital
        spinful = graph.spinful
        out_fea_len = len(orbital) * 8 if spinful else len(orbital)
        values = graph.term_real
        mask = np.zeros((graph.edge_attr.shape[0], out_fea_len), dtype=bool)
        label = np.zeros(mask.shape, dtype=values.dtype)
        species_i = graph.x[graph.edge_index[0]]
        species_j = graph.x[graph.edge_index[1]]
        for index, orbital_dict in enumerate(orbital):
            for species_pair, (orbital_i, orbital_j) in orbital_dict.items():
                number_i, number_j = map(int, species_pair.split())
                condition = (species_i == number_i) & (species_j == number_j)
                if spinful:
                    columns = slice(8 * index, 8 * (index + 1))
                    mask[:, columns] = condition[:, None]
                    label[:, columns] = np.where(
                        condition[:, None], values[:, orbital_i, orbital_j], 0
                    )
                else:
                    mask[:, index] |= condition
                    label[:, index] += np.where(
                        condition, values[:, orbital_i, orbital_j], 0
                    )
        graph.mask = mask
        graph.label = label
        del graph.term_mask
        del graph.term_real

    def to_model_input(self, graph, structure, folder, species=None):
        """Convert a DeepH graph and its unchanged structure to model arrays."""
        atomic_numbers = np.asarray(structure.atomic_numbers, dtype=np.int64)
        species = self.species if species is None else species
        if species is None:
            raise ValueError("Configure species in checkpoint embedding order.")
        species_to_index = {int(number): index for index, number in enumerate(species)}
        unknown_species = sorted(set(atomic_numbers) - set(species_to_index))
        if unknown_species:
            raise ValueError(
                "DeepH structure contains species missing from the model: "
                f"{unknown_species}"
            )
        atom_features = np.asarray(
            [species_to_index[int(number)] for number in atomic_numbers],
            dtype=np.int64,
        )
        subgraph = graph.subgraph_dict
        model_input = {
            "x": atom_features,
            "edge_index": np.asarray(graph.edge_index, dtype=np.int64),
            "edge_attr": np.asarray(graph.edge_attr, dtype=self.default_dtype),
            "pos": np.asarray(structure.cart_coords, dtype=self.default_dtype),
            "sub_atom_idx": np.asarray(subgraph["subgraph_atom_idx"], dtype=np.int64),
            "sub_edge_idx": np.asarray(subgraph["subgraph_edge_idx"], dtype=np.int64),
            "sub_edge_ang": np.asarray(
                subgraph["subgraph_edge_ang"], dtype=self.default_dtype
            ),
            "sub_index": np.asarray(subgraph["subgraph_index"], dtype=np.int64),
            "structure_lattice": np.asarray(
                structure.lattice.matrix, dtype=self.default_dtype
            ).reshape(1, 3, 3),
            "structure_frac_coords": np.asarray(
                structure.frac_coords, dtype=self.default_dtype
            ),
            "structure_atomic_numbers": atomic_numbers,
            "structure_folder": os.path.abspath(folder),
        }
        if hasattr(graph, "label"):
            model_input["label"] = np.asarray(graph.label, dtype=self.default_dtype)
        if hasattr(graph, "mask"):
            model_input["mask"] = np.asarray(graph.mask, dtype=bool)
        return model_input

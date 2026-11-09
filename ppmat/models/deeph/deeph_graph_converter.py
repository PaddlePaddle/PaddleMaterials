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

from ppmat.models.deeph.graph_utils import build_deeph_graph
from ppmat.models.deeph.graph_utils import periodic_image_ranges


class DeepHGraphConverter:
    """Convert a PaddleMaterials structure graph to a DeepH Hamiltonian graph."""

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
    ):
        self.radius = radius
        self.max_num_nbr = max_num_nbr
        self.default_dtype = default_dtype
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
        return build_deeph_graph(
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

    def to_model_input(self, graph, structure, folder, species=None):
        """Convert a DeepH graph and its unchanged structure to model arrays."""
        atomic_numbers = np.asarray(structure.atomic_numbers, dtype=np.int64)
        if species is None:
            atom_features = np.asarray(graph.x, dtype=np.int64)
        else:
            species_to_index = {
                int(number): index for index, number in enumerate(species)
            }
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

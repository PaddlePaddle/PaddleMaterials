# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from __future__ import annotations

import warnings
from typing import Optional
from typing import Tuple

import ase
import numpy as np
import pgl
from ase import Atoms
from p_tqdm import p_map
from pymatgen.core.structure import Structure
from pymatgen.optimization.neighbors import find_points_in_spheres

from .threebody_indices import compute_threebody as _compute_threebody

warnings.filterwarnings("once", category=UserWarning)
"""
Supported Properties:
    - "num_nodes"(set by default)  ## int
    - "num_edges"(set by default)  ## int
    - "num_atoms"                  ## int
    - "num_bonds"                  ## int
    - "atom_attr"                  ## tensor [num_atoms,atom_attr_dim=1]
    - "atom_pos"                   ## tensor [num_atoms,3]
    - "edge_length"                ## tensor [num_edges,1]
    - "edge_vector"                ## tensor [num_edges,3]
    - "edge_index"                 ## tensor [2,num_edges]
    - "three_body_indices"         ## tensor [num_three_body,2]
    - "num_three_body"              ## int
    - "num_triple_ij"              ## tensor [num_edges,1]
    - "num_triple_i"               ## tensor [num_atoms,1]
    - "num_triple_s"               ## tensor [1,1]
    - "theta_jik"                  ## tensor [num_three_body,1]
    - "triple_edge_length"         ## tensor [num_three_body,1]
    - "phi"                        ## tensor [num_three_body,1]
    - "energy"                     ## float
    - "forces"                     ## tensor [num_atoms,3]
    - "stress"                     ## tensor [3,3]
"""
"""
Computing various graph based operations (M3GNet)
"""


def compute_threebody_indices(
    bond_atom_indices: np.array,
    bond_length: np.array,
    n_atoms: int,
    atomic_number: np.array,
    threebody_cutoff: Optional[float] = None,
):
    """
    Given a graph without threebody indices, add the threebody indices
    according to a threebody cutoff radius
    Args:
        bond_atom_indices: np.array, [n_atoms, 2]
        bond_length: np.array, [n_atoms]
        n_atoms: int
        atomic_number: np.array, [n_atoms]
        threebody_cutoff: float, threebody cutoff radius

    Returns:
        triple_bond_indices, n_triple_ij, n_triple_i, n_triple_s

    """
    n_atoms = np.array(n_atoms).reshape(1)
    atomic_number = atomic_number.reshape(-1, 1)
    n_bond = tuple(bond_atom_indices.shape)[0]
    if n_bond > 0 and threebody_cutoff is not None:
        valid_three_body = bond_length <= threebody_cutoff
        ij_reverse_map = np.where(valid_three_body)[0]
        original_index = np.arange(n_bond)[valid_three_body]
        bond_atom_indices = bond_atom_indices[valid_three_body, :]
    else:
        ij_reverse_map = None
        original_index = np.arange(n_bond)
    if tuple(bond_atom_indices.shape)[0] > 0:
        bond_indices, n_triple_ij, n_triple_i, n_triple_s = _compute_threebody(
            np.ascontiguousarray(bond_atom_indices, dtype="int32"),
            np.array(n_atoms, dtype="int32"),
        )
        if ij_reverse_map is not None:
            n_triple_ij_ = np.zeros(shape=(n_bond,), dtype="int32")
            n_triple_ij_[ij_reverse_map] = n_triple_ij
            n_triple_ij = n_triple_ij_
        bond_indices = original_index[bond_indices]
        bond_indices = np.array(bond_indices, dtype="int32")
    else:
        bond_indices = np.reshape(np.array([], dtype="int32"), [-1, 2])
        if n_bond == 0:
            n_triple_ij = np.array([], dtype="int32")
        else:
            n_triple_ij = np.array([0] * n_bond, dtype="int32")
        n_triple_i = np.array([0] * len(atomic_number), dtype="int32")
        n_triple_s = np.array([0], dtype="int32")
    return bond_indices, n_triple_ij, n_triple_i, n_triple_s


def get_fixed_radius_bonding(
    structure: ase.Atoms,
    cutoff: float = 5.0,
    numerical_tol: float = 1e-08,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Get graph representations from structure within cutoff
    Args:
        structure (pymatgen Structure or molecule)
        cutoff (float): cutoff radius
        numerical_tol (float): numerical tolerance

    Returns:
        center_indices, neighbor_indices, images, distances
    """
    pbc_ = np.array(structure.pbc, dtype=int)
    lattice_matrix = np.ascontiguousarray(structure.cell[:], dtype=float)
    cart_coords = np.ascontiguousarray(np.array(structure.positions), dtype=float)
    r = float(cutoff)
    center_indices, neighbor_indices, images, distances = find_points_in_spheres(
        cart_coords,
        cart_coords,
        r=r,
        pbc=pbc_,
        lattice=lattice_matrix,
        tol=numerical_tol,
    )
    center_indices = center_indices.astype(np.int64)
    neighbor_indices = neighbor_indices.astype(np.int64)
    images = images.astype(np.int64)
    distances = distances.astype(float)
    exclude_self = (center_indices != neighbor_indices) | (distances > numerical_tol)
    return (
        center_indices[exclude_self],
        neighbor_indices[exclude_self],
        images[exclude_self],
        distances[exclude_self],
    )


class M3GNetGraphConvertor:
    """
    Convert ase.Atoms to Graph
    """

    def __init__(
        self,
        cutoff: float = 5.0,
        has_threebody: bool = True,
        threebody_cutoff: float = 4.0,
        num_cpus: Optional[int] = None,
    ):
        self.cutoff = cutoff
        self.threebody_cutoff = threebody_cutoff
        self.has_threebody = has_threebody
        self.num_cpus = num_cpus

    def __call__(self, structure: Structure):
        if isinstance(structure, Structure):
            graph = self.get_graph_by_m3gnet_graph(structure)
        elif isinstance(structure, list):
            graph = p_map(
                self.get_graph_by_m3gnet_graph,
                structure,
                num_cpus=self.num_cpus,
            )
            # the following code is equivalent to the above line, it is slower,
            # but easier to debug.
            # graph = [
            # self.get_graph_by_m3gnet_graph(struc)
            #     for struc in structure
            # ]
        else:
            raise TypeError("The input must be a pymatgen.Structure or a list of them.")
        return graph

    def get_graph_by_m3gnet_graph(self, structure: Structure):
        atoms = structure.to_ase_atoms()
        if isinstance(atoms, Atoms):
            pbc_ = np.array(atoms.pbc, dtype=int)
            if np.all(pbc_ < 0.01):
                min_x = np.min(atoms.positions[:, 0])
                min_y = np.min(atoms.positions[:, 1])
                min_z = np.min(atoms.positions[:, 2])
                max_x = np.max(atoms.positions[:, 0])
                max_y = np.max(atoms.positions[:, 1])
                max_z = np.max(atoms.positions[:, 2])
                x_len = max_x - min_x + max(self.cutoff, self.threebody_cutoff) * 5
                y_len = max_y - min_y + max(self.cutoff, self.threebody_cutoff) * 5
                z_len = max_z - min_z + max(self.cutoff, self.threebody_cutoff) * 5
                max_len = max(x_len, y_len, z_len)
                x_len = y_len = z_len = max_len
                lattice_matrix = np.eye(3) * max_len
                pbc_ = np.array([1, 1, 1], dtype=int)
                warnings.warn(
                    "No PBC detected, using a large supercell with size "
                    f"{x_len}x{y_len}x{z_len} Angstrom**3",
                    UserWarning,
                )
                atoms.set_cell(lattice_matrix)
                atoms.set_pbc(pbc_)
            elif np.all(abs(atoms.cell) < 1e-05):
                raise ValueError("Cell vectors are too small")
        else:
            raise ValueError("structure type not supported")
        scaled_pos = atoms.get_scaled_positions()
        scaled_pos = np.mod(scaled_pos, 1)
        atoms.set_scaled_positions(scaled_pos)

        (
            sent_index,
            receive_index,
            shift_vectors,
            distances,
        ) = get_fixed_radius_bonding(atoms, self.cutoff)
        edge_indices = [(u, v) for u, v in zip(sent_index, receive_index)]
        to_jimages = np.array(shift_vectors, dtype="float32")

        edge_features = {}
        if self.has_threebody:
            (
                triple_bond_index,
                n_triple_ij,
                n_triple_i,
                n_triple_s,
            ) = compute_threebody_indices(
                bond_atom_indices=np.asarray(edge_indices),
                bond_length=distances,
                n_atoms=atoms.positions.shape[0],
                atomic_number=atoms.get_atomic_numbers(),
                threebody_cutoff=self.threebody_cutoff,
            )

            edge_features["num_three_body"] = np.array([triple_bond_index.shape[0]])
            edge_features["three_body_indices"] = triple_bond_index.astype("int64")
            edge_features["num_triple_ij"] = n_triple_ij.astype("int64").reshape(-1, 1)

        graph = self.build_pgl_graph(
            Structure.from_ase_atoms(atoms),
            edge_indices,
            to_jimages,
            edge_features=edge_features,
        )

        return graph

    def build_pgl_graph(
        self,
        structure: Structure,
        edge_indices,
        to_jimages,
        node_features=None,
        edge_features=None,
    ):
        assert node_features is None or isinstance(node_features, dict)
        assert edge_features is None or isinstance(edge_features, dict)

        # get atom types
        atom_types = np.array([site.specie.Z for site in structure])

        # get lattice parameters and matrix
        lattice_parameters = structure.lattice.parameters
        lengths = np.array(lattice_parameters[:3], dtype="float32").reshape(1, 3)
        angles = np.array(lattice_parameters[3:], dtype="float32").reshape(1, 3)
        lattice = structure.lattice.matrix.astype("float32")

        # convert to numpy array
        edge_indices = np.array(edge_indices)
        if to_jimages is not None:
            to_jimages = np.array(to_jimages)
        num_atoms = tuple(atom_types.shape)[0]

        # After multiple graph batch operations by the dataloader,
        # graph.num_nodes remains an integer, which is the sum of the number of
        # nodes in all graphs
        graph = pgl.Graph(edge_indices, num_nodes=num_atoms)
        # node features: frac_coords, cart_coords, atom_types
        graph.node_feat["frac_coords"] = structure.frac_coords.astype("float32")
        graph.node_feat["cart_coords"] = structure.cart_coords.astype("float32")
        graph.node_feat["atom_types"] = atom_types

        # graph features: lengths, angles, lattice, num_atoms
        # Due to the inability of pgl.graph to store graph level features,
        # we will store these features under node_feat
        graph.node_feat["lengths"] = lengths
        graph.node_feat["angles"] = angles
        graph.node_feat["lattice"] = lattice.reshape(1, 3, 3)
        # graph.node_feat['num_atoms'] is different from graph.num_nodes
        # After multiple graph batch operations by the dataloader,
        # graph.node_feat['num_atoms'] is a tensor of shape (batch_size),
        # where each value is the number of atoms in the corresponding graph.
        graph.node_feat["num_atoms"] = np.array([num_atoms])
        # edge features: pbc_offset, bond_vec, bond_dist
        if to_jimages is not None:
            graph.edge_feat["pbc_offset"] = to_jimages
            offset = np.matmul(to_jimages, lattice)
            dst_pos = graph.node_feat["cart_coords"][graph.edges[:, 1]] + offset
            src_pos = graph.node_feat["cart_coords"][graph.edges[:, 0]]
            bond_vec = dst_pos - src_pos
            bond_dist = np.linalg.norm(bond_vec, axis=1)
            graph.edge_feat["bond_vec"] = bond_vec.astype("float32")
            graph.edge_feat["bond_dist"] = bond_dist.astype("float32")
        graph.edge_feat["num_edges"] = np.array([edge_indices.shape[0]])

        if node_features is not None:
            graph.node_feat.update(node_features)
        if edge_features is not None:
            graph.edge_feat.update(edge_features)
        return graph

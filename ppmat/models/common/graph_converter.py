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

from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union

import numpy as np
import pgl
from cvve import Structure as CVVEStructure
from p_tqdm import p_map
from pymatgen.analysis import local_env
from pymatgen.analysis.graphs import StructureGraph
from pymatgen.core.structure import Structure
from pymatgen.optimization.neighbors import find_points_in_spheres
from rdkit import Chem

from ppmat.utils import logger
from ppmat.utils.crystal import atomic_number_from_symbol
from ppmat.utils.crystal import lattice_params_to_matrix


def _build_crystal_pgl_graph(
    structure: Structure,
    edge_indices,
    to_jimages,
    node_features=None,
    edge_features=None,
):
    """Build one PGL crystal graph shared by the periodic converters.

    Stores lattice-level features under ``node_feat`` because pgl.Graph has
    no graph-level feature slot.
    """

    atom_types = np.array([site.specie.Z for site in structure])
    lattice_parameters = structure.lattice.parameters
    lengths = np.array(lattice_parameters[:3], dtype="float32").reshape(1, 3)
    angles = np.array(lattice_parameters[3:], dtype="float32").reshape(1, 3)
    lattice = structure.lattice.matrix.astype("float32")
    cart_coords = structure.cart_coords.astype("float32")
    edge_indices = np.asarray(edge_indices, dtype=np.int64).reshape(-1, 2)

    node_feat = {
        "frac_coords": structure.frac_coords.astype("float32"),
        "cart_coords": cart_coords,
        "atom_types": atom_types,
        "lengths": lengths,
        "angles": angles,
        "lattice": lattice.reshape(1, 3, 3),
        "num_atoms": np.array([len(atom_types)]),
    }
    edge_feat = {"num_edges": np.array([len(edge_indices)])}
    if to_jimages is not None:
        to_jimages = np.asarray(to_jimages, dtype=np.float32).reshape(-1, 3)
        offset = np.matmul(to_jimages, lattice)
        src_pos = cart_coords[edge_indices[:, 0]]
        dst_pos = cart_coords[edge_indices[:, 1]] + offset
        bond_vec = dst_pos - src_pos
        edge_feat.update(
            {
                "pbc_offset": to_jimages,
                "bond_vec": bond_vec.astype("float32"),
                "bond_dist": np.linalg.norm(bond_vec, axis=1).astype("float32"),
            }
        )

    if node_features is not None:
        node_feat.update(node_features)
    if edge_features is not None:
        edge_feat.update(edge_features)
    return pgl.Graph(
        edge_indices,
        num_nodes=len(atom_types),
        node_feat=node_feat,
        edge_feat=edge_feat,
    )


class FindPointsInSpheres:
    """Convert crystal structure to graph representation using spherical neighborhood
    search.

    This tool identifies neighboring atoms within a cutoff radius for each atom in a
    crystal structure, building a graph representation suitable for material analysis
    applications.

    Args:
        cutoff (float, optional): Cutoff radius (in Ångström) for neighborhood search.
            Defaults to 5.0.
        pbc (tuple[int, int, int], optional):Periodic boundary conditions along x/y/z
            axes. Each element 0 (disabled) or 1 (enabled). Defaults to (1, 1, 1).
        num_cpus (Optional[int], optional): Number of CPU cores for parallel processing:
            - None: Auto-detect all available cores (recommended)
            - Positive integer: Explicit core count.
            Defaults to None.
        eps (float, optional): Floating-point tolerance for numerical comparisons.
            Defaults to 1e-8.
        **kwargs: Reserved for future expansion (currently unused parameters)
    """

    def __init__(
        self,
        cutoff: float = 5.0,
        pbc: tuple[int, int, int] = (1, 1, 1),
        num_cpus: Optional[int] = None,
        eps: float = 1e-8,
        **kwargs,
    ) -> None:
        self.cutoff = cutoff
        self.pbc = np.array(pbc, dtype=int)
        self.num_cpus = num_cpus
        self.eps = eps

    def __call__(self, structure: Structure):
        if isinstance(structure, Structure):
            graph = self.get_graph_by_find_points_in_spheres(structure)
        elif isinstance(structure, list):
            graph = p_map(
                self.get_graph_by_find_points_in_spheres,
                structure,
                num_cpus=self.num_cpus,
            )
            # the following code is equivalent to the above line, it is slower,
            # but easier to debug.
            # graph = [
            # self.get_graph_by_find_points_in_spheres(struc)
            #     for struc in structure
            # ]
        else:
            raise TypeError("The input must be a pymatgen.Structure or a list of them.")
        return graph

    def get_graph_by_find_points_in_spheres(self, structure: Structure):
        lattice_matrix = structure.lattice.matrix
        cart_coords = structure.cart_coords

        cutoff = self.cutoff
        attempt = 3
        while attempt > 0:
            src_id, dst_id, images, bond_dist = find_points_in_spheres(
                cart_coords,
                cart_coords,
                r=cutoff,
                pbc=self.pbc,
                lattice=lattice_matrix,
                tol=self.eps,
            )
            exclude_self = (src_id != dst_id) | (bond_dist > self.eps)
            src_id, dst_id, images, bond_dist = (
                src_id[exclude_self],
                dst_id[exclude_self],
                images[exclude_self],
                bond_dist[exclude_self],
            )

            edge_indices = [(u, v) for u, v in zip(src_id, dst_id)]
            to_jimages = np.array(images, dtype="float32")
            if len(edge_indices) == 0:
                logger.warning(
                    f"No edges found within cutoff {cutoff:.5f}. Trying again with "
                    "larger cutoff."
                )
                cutoff *= 2
                attempt -= 1
            else:
                break
        if len(edge_indices) == 0:
            logger.warning(
                f"No edges found within cutoff {cutoff:.5f}. Set graph is None."
            )
            graph = None
        else:
            graph = self.build_pgl_graph(structure, edge_indices, to_jimages)
        return graph

    def build_pgl_graph(
        self,
        structure: Structure,
        edge_indices,
        to_jimages,
        node_features=None,
        edge_features=None,
    ):
        return _build_crystal_pgl_graph(
            structure,
            edge_indices,
            to_jimages,
            node_features=node_features,
            edge_features=edge_features,
        )


class CrystalNN:
    """Convert crystal structure to graph representation using
    CrystalNN-based graph generator method.

    This class uses pymatgen's CrystalNN local environment strategy to
    convert a pymatgen Structure into a PGL graph (pgl.Graph),
    capturing atomic connectivity under periodic boundary conditions.

    Core methods:
      - __init__: Configure neighbor search parameters and parallelism.
      - __call__:  Accepts a single Structure or a list and returns
                   a graph or list of graphs.
      - get_graph_by_crystalnn: Converts one Structure into a PGL graph.
      - build_pgl_graph: Assembles node/edge features into a PGL graph.

    Args:
        cutoff (float):
            Maximum neighbor search distance (in Å). Atom pairs farther
            apart than this are not considered bonded.
        pbc (tuple[int, int, int]):
            Periodic boundary flags along (a, b, c) axes.
            1 enables periodicity, 0 disables it.
        num_cpus (Optional[int]):
            Number of CPU cores to use when processing a list of structures
            in parallel. If None, processes sequentially.
        eps (float):
            Small constant for numerical stability (e.g., to avoid division
            by zero).
    """

    def __init__(
        self,
        cutoff: float = 5.0,
        pbc: tuple[int, int, int] = (1, 1, 1),
        num_cpus: Optional[int] = None,
        eps: float = 1e-8,
    ):
        self.cutoff = cutoff
        self.pbc = np.array(pbc, dtype=int)
        self.num_cpus = num_cpus
        self.eps = eps
        self.CrystalNN = local_env.CrystalNN(
            distance_cutoffs=None, x_diff_weight=-1, porous_adjustment=False
        )

    def __call__(self, structure: Structure):
        if isinstance(structure, Structure):
            graph = self.get_graph_by_crystalnn(structure)
        elif isinstance(structure, list):
            graph = p_map(
                self.get_graph_by_crystalnn, structure, num_cpus=self.num_cpus
            )
            # the following code is equivalent to the above line, it is slower,
            # but easier to debug.
            # graph = [self.get_graph_by_crystalnn(struc) for struc in structure]
        else:
            raise TypeError("The input must be a pymatgen.Structure or a list of them.")
        return graph

    def get_graph_by_crystalnn(self, structure: Structure):

        try:
            structure_graph = StructureGraph.with_local_env_strategy(
                structure, self.CrystalNN
            )
        except Exception:
            search_cutoff = 10
            while True:
                try:
                    crystalNN_tmp = local_env.CrystalNN(
                        distance_cutoffs=None,
                        x_diff_weight=-1,
                        porous_adjustment=False,
                        search_cutoff=search_cutoff,
                    )
                    structure_graph = StructureGraph.from_local_env_strategy(
                        structure, crystalNN_tmp
                    )
                    logger.info(
                        "Successfully generated graph by CrystalNN with "
                        f"search_cutoff={search_cutoff}."
                    )
                    break
                except Exception:
                    search_cutoff += 2
                    logger.info(f"Searching for new search_cutoff{search_cutoff}...")
                    if search_cutoff > 40:
                        logger.info(
                            "Failed to generate graph by CrystalNN with "
                            f"search_cutoff={search_cutoff}. "
                        )
                        break

        # atom_types = np.array(structure.atomic_numbers)
        lattice_parameters = structure.lattice.parameters
        lengths = lattice_parameters[:3]
        angles = lattice_parameters[3:]
        assert np.allclose(
            structure.lattice.matrix, lattice_params_to_matrix(*lengths, *angles)
        )

        edge_indices, to_jimages = [], []
        for i, j, to_jimage in structure_graph.graph.edges(data="to_jimage"):
            edge_indices.append([j, i])
            to_jimages.append(to_jimage)
            edge_indices.append([i, j])
            to_jimages.append(tuple(-tj for tj in to_jimage))

        graph = self.build_pgl_graph(structure, edge_indices, to_jimages)
        return graph

    def build_pgl_graph(
        self,
        structure: Structure,
        edge_indices,
        to_jimages,
        node_features=None,
        edge_features=None,
    ):
        return _build_crystal_pgl_graph(
            structure,
            edge_indices,
            to_jimages,
            node_features=node_features,
            edge_features=edge_features,
        )


class MolecularGraphConverter:
    """Convert RDKit Mol into PGL Graph.

    Args:
        vocab (Dict): Registered vocabularies containing ``atom`` and ``bond``
            roles.
        remove_h (bool): Controls whether hydrogen atoms are removed before building
            the molecular graph. Defaults to True.
        add_self_loops (bool): Adds self-loops to the graph (edges connecting each
            node to itself).
        num_cpus (Optional[int]): Number of CPUs for parallel graph construction.
            Defaults to 1。
    """

    def __init__(
        self,
        vocab: Dict,
        remove_h: bool = True,
        add_self_loops: bool = False,
        edge_mode: str = "bidirectional",
        num_cpus: Optional[int] = None,
    ) -> None:
        self.vocab = vocab
        self.remove_h = remove_h
        self.add_self_loops = add_self_loops
        self.edge_mode = edge_mode
        self.num_cpus = 1 if num_cpus is None else int(num_cpus)

    @staticmethod
    def build_one(
        mol: Chem.Mol,
        remove_h: bool,
        vocab: Dict,
        add_self_loops: bool,
        edge_mode: str = "bidirectional",
    ) -> Optional[pgl.Graph]:
        if mol is None:
            return None
        if remove_h:
            mol = Chem.RemoveHs(mol)

        N = mol.GetNumAtoms()
        if N == 0:
            return None

        atom_vocab = vocab["atom"]
        atom_token_to_id = atom_vocab["token_to_id"]
        num_atom_embeddings = int(atom_vocab["num_embeddings"])
        bond_vocab = vocab["bond"]
        bond_token_to_id = bond_vocab["token_to_id"]
        num_bond_embeddings = int(bond_vocab["num_embeddings"])
        no_bond_id = bond_token_to_id["NO_BOND"]

        # 1) Node Features: One-hot encoding of atomic symbols.
        idxs: List[int] = []
        for atom in mol.GetAtoms():
            sym = atom.GetSymbol()
            if sym not in atom_token_to_id:
                return None  # Unknown Elements: Can be replaced with an extended
                # vocabulary or placeholder <unk>
            idxs.append(atom_token_to_id[sym])
        idxs_np = np.asarray(idxs, dtype=np.int64)  # [N]
        x = np.eye(num_atom_embeddings, dtype=np.float32)[idxs_np]

        # 2) Build the edges first (construct edge_index/edge_attr)
        rows, cols, etypes = [], [], []

        def push(u, v, et):
            rows.append(u)
            cols.append(v)
            etypes.append(et)

        for b in mol.GetBonds():
            u, v = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
            bond_type = b.GetBondType()
            bond_token = getattr(
                bond_type,
                "name",
                str(bond_type).split(".")[-1],
            )
            et = bond_token_to_id.get(bond_token, no_bond_id)
            if edge_mode == "directed":
                push(u, v, et)
            elif edge_mode == "undirected":
                uu, vv = (u, v) if u < v else (v, u)
                push(uu, vv, et)
            elif edge_mode == "bidirectional":
                push(u, v, et)
                push(v, u, et)
            else:
                raise ValueError(f"Unknown edge_mode: {edge_mode}")

        if len(rows) == 0:
            edge_index = np.empty((2, 0), dtype=np.int64)
            edge_attr = np.empty((0, num_bond_embeddings), dtype=np.float32)
        else:
            row_np = np.asarray(rows, dtype=np.int64)
            col_np = np.asarray(cols, dtype=np.int64)
            et_np = np.asarray(etypes, dtype=np.int64)
            edge_attr = np.eye(num_bond_embeddings, dtype=np.float32)[et_np]
            # Deterministic ordering by (row, col)
            order = np.argsort(row_np * max(1, N) + col_np, kind="mergesort")
            row_np, col_np, edge_attr = row_np[order], col_np[order], edge_attr[order]
            edge_index = np.stack([row_np, col_np], axis=0)  # [2, E]

        # 3) (Optional) Add self-loops based on the updated node count
        N_new = int(x.shape[0])
        edges_e2 = edge_index.T.astype(np.int64)  # [E,2]
        if add_self_loops and N_new > 0:
            self_e2 = np.stack([np.arange(N_new), np.arange(N_new)], axis=1).astype(
                np.int64
            )
            self_ea = np.eye(num_bond_embeddings, dtype=np.float32)[
                np.full((N_new,), no_bond_id, dtype=np.int64)
            ]
            edges_e2 = np.concatenate([edges_e2, self_e2], axis=0)
            edge_attr = np.concatenate([edge_attr, self_ea], axis=0)

        # 4) Return a PGL graph. (If running in worker processes, consider returning
        #    NumPy arrays and wrapping into pgl.Graph)
        return pgl.Graph(
            num_nodes=int(x.shape[0]),
            edges=edges_e2,
            node_feat={"feat": x},
            edge_feat={"feat": edge_attr},
        )

    def __call__(
        self, mols: Union[Sequence[Chem.Mol], Chem.Mol]
    ) -> Union[List[pgl.Graph], pgl.Graph, None]:
        if isinstance(mols, (list, tuple)):
            return p_map(
                MolecularGraphConverter.build_one,
                mols,
                [self.remove_h] * len(mols),
                [self.vocab] * len(mols),
                [self.add_self_loops] * len(mols),
                [self.edge_mode] * len(mols),
                num_cpus=self.num_cpus,
                desc="Building graphs",
                dynamic_ncols=True,
                mininterval=0.2,
            )
        else:
            return MolecularGraphConverter.build_one(
                mols,
                self.remove_h,
                self.vocab,
                self.add_self_loops,
                self.edge_mode,
            )


class RadiusGraphConverter:
    """Convert atomic geometries into PGL radius graphs.

    The existing callable interface accepts an RDKit ``Mol`` or a list of them.
    ``from_arrays`` and ``from_structure`` provide equivalent paths for normalized
    atomic arrays and cvve/pymatgen structures.

    Args:
        cutoff: Neighbor cutoff distance in the processed coordinates.
        atom_vocab: Mapping from atomic number to feature index for optional PGL
            node features.
        add_self_loops: Whether to add self-loop edges.
        edge_mode: ``"directed"``, ``"undirected"``, or ``"bidirectional"``.
        include_distance: Whether to include distance in PGL edge features.
        include_bond_vec: Whether to include Cartesian bond vectors in PGL edge
            features.
        return_triplet_indices: Whether to cache SphereNet triplet indices.
        max_num_neighbors: Maximum incoming neighbors retained per atom. Neighbors
            are selected deterministically by distance and source index. ``None``
            keeps every neighbor inside ``cutoff``.
        num_cpus: Number of CPUs for parallel graph construction.
        inclusive_cutoff: Whether atom pairs exactly at ``cutoff`` are connected.
        vocab: Optional Dataset vocabulary. Its
            ``atom.atomic_number_to_id`` mapping overrides ``atom_vocab`` and
            is used to build ``node_feat["x"]``.
    """

    def __init__(
        self,
        cutoff: float = 5.0,
        atom_vocab: Optional[Dict[int, int]] = None,
        add_self_loops: bool = False,
        edge_mode: str = "bidirectional",
        include_distance: bool = True,
        include_bond_vec: bool = False,
        return_triplet_indices: bool = False,
        max_num_neighbors: Optional[int] = None,
        num_cpus: Optional[int] = None,
        inclusive_cutoff: bool = False,
        vocab: Optional[Dict] = None,
    ) -> None:
        self.require_atom_vocab = vocab is not None
        if vocab is not None:
            try:
                atom_vocab = vocab["atom"]["atomic_number_to_id"]
            except (KeyError, TypeError) as exc:
                raise KeyError("vocab must define atom.atomic_number_to_id.") from exc
        if atom_vocab is None:
            atom_vocab = {1: 0, 6: 1, 7: 2, 8: 3, 9: 4}
        if edge_mode not in {"directed", "undirected", "bidirectional"}:
            raise ValueError(f"Unknown edge_mode: {edge_mode}")
        if not isinstance(inclusive_cutoff, bool):
            raise TypeError("inclusive_cutoff must be a boolean.")
        self.cutoff = float(cutoff)
        self.atom_vocab = dict(atom_vocab)
        self.add_self_loops = add_self_loops
        self.edge_mode = edge_mode
        self.include_distance = include_distance
        self.include_bond_vec = include_bond_vec
        self.return_triplet_indices = return_triplet_indices
        self.max_num_neighbors = (
            None if max_num_neighbors is None else int(max_num_neighbors)
        )
        self.num_cpus = 1 if num_cpus is None else int(num_cpus)
        self.inclusive_cutoff = inclusive_cutoff

    def __call__(
        self, molecule: Union[Chem.Mol, List[Chem.Mol]]
    ) -> Union[Optional[pgl.Graph], List[Optional[pgl.Graph]]]:
        if isinstance(molecule, Chem.Mol):
            graph = self.get_graph_by_radius(molecule)
        elif isinstance(molecule, list):
            if self.num_cpus == 1:
                graph = [self.get_graph_by_radius(mol) for mol in molecule]
            else:
                graph = p_map(
                    self.get_graph_by_radius,
                    molecule,
                    num_cpus=self.num_cpus,
                    desc="Building graphs",
                    dynamic_ncols=True,
                    mininterval=0.2,
                )
        else:
            raise TypeError("The input must be an RDKit Mol or a list of them.")
        return graph

    def get_graph_by_radius(self, molecule: Chem.Mol) -> Optional[pgl.Graph]:
        if molecule is None:
            return None

        atomic_numbers, positions = self.get_molecule_array(molecule)
        return self.from_arrays(
            atomic_numbers,
            positions,
        )

    def from_arrays(
        self,
        atomic_numbers: np.ndarray,
        positions: np.ndarray,
        node_features: Optional[Dict[str, np.ndarray]] = None,
    ) -> Optional[pgl.Graph]:
        """Build a radius graph from atomic numbers and Cartesian positions.

        Args:
            atomic_numbers: Atomic numbers with shape ``[num_nodes]``.
            positions: Cartesian positions with shape ``[num_nodes, 3]``.
            node_features: Additional per-node arrays merged into
                ``graph.node_feat``.
        """

        atomic_numbers = np.asarray(atomic_numbers, dtype=np.int64)
        positions = np.asarray(positions, dtype=np.float32)

        edge_index, distances, directions = self.get_radius_edges(positions)
        triplet_indices = None
        if self.return_triplet_indices:
            triplet_indices = self.get_triplet_indices(edge_index)
        graph = self.build_pgl_graph(
            atomic_numbers,
            positions,
            edge_index,
            distances,
            directions,
            triplet_indices,
        )
        if node_features is not None:
            graph.node_feat.update(
                {name: np.asarray(feature) for name, feature in node_features.items()}
            )
        return graph

    def from_structure(
        self,
        structure: Union[CVVEStructure, Structure],
        node_features: Optional[Dict[str, np.ndarray]] = None,
    ) -> Optional[pgl.Graph]:
        """Build a radius graph from a cvve or pymatgen structure."""

        if isinstance(structure, CVVEStructure):
            atomic_numbers = np.asarray(
                [atomic_number_from_symbol(symbol) for symbol in structure.symbols],
                dtype=np.int64,
            )
            return self.from_arrays(
                atomic_numbers,
                structure.cartesian_positions(),
                node_features=node_features,
            )
        if isinstance(structure, Structure):
            return self.from_arrays(
                np.asarray(structure.atomic_numbers, dtype=np.int64),
                structure.cart_coords,
                node_features=node_features,
            )
        raise TypeError(
            "structure must be a cvve.Structure or pymatgen.Structure, but got "
            f"{type(structure)}."
        )

    def from_structures(
        self,
        structures: Sequence[Union[CVVEStructure, Structure]],
    ) -> List[Optional[pgl.Graph]]:
        """Build radius graphs from cvve or pymatgen structures."""

        if not isinstance(structures, (list, tuple)):
            raise TypeError("structures must be a list or tuple.")
        if not structures:
            return []
        if self.num_cpus == 1:
            return [self.from_structure(structure) for structure in structures]
        return p_map(
            self.from_structure,
            structures,
            num_cpus=self.num_cpus,
            desc="Building graphs",
            dynamic_ncols=True,
            mininterval=0.2,
        )

    def get_molecule_array(self, molecule: Chem.Mol) -> Tuple[np.ndarray, np.ndarray]:
        if molecule.GetNumConformers() == 0:
            raise ValueError("RDKit Mol has no conformer. Cannot build radius graph.")

        atomic_numbers = np.asarray(
            [atom.GetAtomicNum() for atom in molecule.GetAtoms()], dtype=np.int64
        )
        conf = molecule.GetConformer()
        positions = np.asarray(
            [
                [
                    conf.GetAtomPosition(i).x,
                    conf.GetAtomPosition(i).y,
                    conf.GetAtomPosition(i).z,
                ]
                for i in range(molecule.GetNumAtoms())
            ],
            dtype=np.float32,
        )
        return atomic_numbers, positions

    def get_radius_edges(
        self, positions: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        positions = np.asarray(positions, dtype=np.float32)
        num_nodes = positions.shape[0]
        if num_nodes == 0:
            return (
                np.empty((2, 0), dtype=np.int64),
                np.empty((0, 1), dtype=np.float32),
                np.empty((0, 3), dtype=np.float32),
            )

        displacement = positions[:, None, :] - positions[None, :, :]
        distance_matrix = np.linalg.norm(displacement, axis=-1)
        if self.inclusive_cutoff:
            mask = distance_matrix <= self.cutoff
        else:
            mask = distance_matrix < self.cutoff
        np.fill_diagonal(mask, False)
        targets, sources = np.where(mask)

        if self.edge_mode == "undirected":
            keep = sources < targets
            sources, targets = sources[keep], targets[keep]

        if self.add_self_loops:
            self_nodes = np.arange(num_nodes, dtype=np.int64)
            sources = np.concatenate([sources, self_nodes])
            targets = np.concatenate([targets, self_nodes])

        if sources.size == 0:
            return (
                np.empty((2, 0), dtype=np.int64),
                np.empty((0, 1), dtype=np.float32),
                np.empty((0, 3), dtype=np.float32),
            )

        edge_index = np.stack([sources, targets], axis=0).astype(np.int64)
        distances = distance_matrix[targets, sources].reshape(-1, 1)
        directions = displacement[targets, sources]
        directions = directions / np.maximum(distances, 1e-8)

        # Sort by (target, distance, source), then retain the nearest neighbors.
        # The deterministic tie-break keeps cached graphs reproducible.
        order = np.lexsort((edge_index[0], distances.reshape(-1), edge_index[1]))
        edge_index = edge_index[:, order]
        distances = distances[order].astype(np.float32)
        directions = directions[order].astype(np.float32)
        if self.max_num_neighbors is not None:
            target_counts = np.zeros(num_nodes, dtype=np.int64)
            keep = np.zeros(edge_index.shape[1], dtype=bool)
            for edge_id, target in enumerate(edge_index[1]):
                target = int(target)
                if target_counts[target] < self.max_num_neighbors:
                    keep[edge_id] = True
                    target_counts[target] += 1
            edge_index = edge_index[:, keep]
            distances = distances[keep]
            directions = directions[keep]
        return edge_index, distances, directions

    def get_node_feat(
        self, atomic_numbers: np.ndarray, positions: np.ndarray
    ) -> Dict[str, np.ndarray]:
        atomic_numbers = np.asarray(atomic_numbers, dtype=np.int64)
        node_feat = {
            "cart_coords": positions.astype(np.float32),
            "atom_types": atomic_numbers,
        }

        idxs = []
        for z in atomic_numbers:
            if int(z) not in self.atom_vocab:
                if self.require_atom_vocab:
                    raise KeyError(
                        f"Atomic number {int(z)} is missing from the atom vocabulary."
                    )
                return node_feat
            idxs.append(self.atom_vocab[int(z)])
        idxs = np.asarray(idxs, dtype=np.int64)
        node_feat["x"] = idxs
        node_feat["feat"] = np.eye(len(self.atom_vocab), dtype=np.float32)[idxs]
        return node_feat

    def build_pgl_graph(
        self,
        atomic_numbers: np.ndarray,
        positions: np.ndarray,
        edge_index: np.ndarray,
        distances: np.ndarray,
        directions: np.ndarray,
        triplet_indices: Optional[Dict[str, np.ndarray]] = None,
    ) -> pgl.Graph:
        edge_feat = {}
        edge_feats = []
        if self.include_distance:
            edge_feat["bond_dist"] = distances.reshape(-1)
            edge_feats.append(distances)
        if self.include_bond_vec:
            bond_vec = directions * distances
            edge_feat["bond_vec"] = bond_vec
            edge_feats.append(bond_vec)
        edge_feat["num_edges"] = np.asarray([edge_index.shape[1]], dtype=np.int64)
        if edge_feats:
            edge_feat["feat"] = np.concatenate(edge_feats, axis=-1).astype(np.float32)
        if triplet_indices is not None:
            edge_feat.update(triplet_indices)

        return pgl.Graph(
            num_nodes=positions.shape[0],
            edges=edge_index.T.astype(np.int64),
            node_feat=self.get_node_feat(atomic_numbers, positions),
            edge_feat=edge_feat,
        )

    def get_triplet_indices(self, edge_index: np.ndarray) -> Dict[str, np.ndarray]:
        edge_index = np.asarray(edge_index, dtype=np.int64)
        sources, targets = edge_index
        num_atoms = int(edge_index.max()) + 1 if edge_index.size else 0

        incoming_edges = [[] for _ in range(num_atoms)]
        for edge_id, target in enumerate(targets):
            incoming_edges[int(target)].append(edge_id)

        idx_kj = []
        idx_ji = []
        for edge_id, (source, target) in enumerate(edge_index.T):
            for incoming_edge in incoming_edges[int(source)]:
                if sources[incoming_edge] != target:
                    idx_kj.append(incoming_edge)
                    idx_ji.append(edge_id)

        return {
            "ti_idx_kj": np.asarray(idx_kj, dtype=np.int64),
            "ti_idx_ji": np.asarray(idx_ji, dtype=np.int64),
        }

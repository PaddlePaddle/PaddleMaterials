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

"""DeepH-specific Hamiltonian graph features.

Generic periodic neighbor discovery is delegated to PaddleMaterials graph
converters. This module only adapts those edges to DeepH's Hamiltonian layout
and builds the LCMP angular subgraph.
"""

from types import SimpleNamespace

import numpy as np

from ppmat.models.deeph.lcmp import build_lcmp_subgraph
from ppmat.utils.io import load_deeph_matrices


def periodic_image_ranges(frac_coords, lattice, radius):
    """Return periodic translations that can intersect a radius neighborhood."""
    reciprocal_lattice = np.linalg.inv(lattice).T * 2 * np.pi
    reciprocal_lengths = np.linalg.norm(reciprocal_lattice, axis=1)
    max_images = np.ceil((radius + 0.15) * reciprocal_lengths / (2 * np.pi))
    minimum = np.floor(np.min(frac_coords, axis=0)) - max_images
    maximum = np.ceil(np.max(frac_coords, axis=0)) + max_images
    return [
        np.arange(start, stop, dtype=np.int64) for start, stop in zip(minimum, maximum)
    ]


def _validate_supported_path(
    interface,
    target,
    create_from_DFT,
    if_lcmp_graph,
    separate_onsite,
    if_new_sp,
    huge_structure,
):
    if interface not in {"npz", "npz_rc_only"}:
        raise NotImplementedError(
            "DeepH PaddleMaterials integration supports interface='npz' and "
            "'npz_rc_only'."
        )
    if target != "hamiltonian":
        raise NotImplementedError(
            "DeepH PaddleMaterials integration currently supports target='hamiltonian'."
        )
    if interface == "npz_rc_only" and not create_from_DFT:
        raise ValueError("interface='npz_rc_only' requires create_from_DFT=True.")
    if not if_lcmp_graph:
        raise NotImplementedError(
            "DeepH PaddleMaterials integration expects if_lcmp_graph=True."
        )
    if separate_onsite:
        raise NotImplementedError(
            "DeepH separate_onsite=True is not enabled in this integration."
        )
    if if_new_sp:
        raise NotImplementedError(
            "DeepH new_sp=True is not enabled in this integration."
        )
    if huge_structure:
        raise NotImplementedError(
            "DeepH huge_structure=True is not enabled in this integration."
        )


def _build_edges_from_converter_graph(converter_graph, default_dtype):
    edge_idx = np.asarray(converter_graph.edges, dtype=np.int64).T
    cart_coords = np.asarray(
        converter_graph.node_feat["cart_coords"],
        dtype=default_dtype,
    )
    lattice = np.asarray(
        converter_graph.node_feat["lattice"],
        dtype=default_dtype,
    ).reshape(3, 3)
    pbc_offset = np.asarray(
        converter_graph.edge_feat["pbc_offset"],
        dtype=default_dtype,
    )
    edge_dist = np.asarray(
        converter_graph.edge_feat["bond_dist"],
        dtype=default_dtype,
    ).reshape(-1, 1)
    src, dst = edge_idx
    dst_cart_periodic = cart_coords[dst] + pbc_offset @ lattice
    edge_fea = np.concatenate(
        [
            edge_dist,
            cart_coords[src],
            dst_cart_periodic,
            cart_coords[dst],
        ],
        axis=-1,
    ).astype(default_dtype)
    onsite_src = np.arange(cart_coords.shape[0], dtype=np.int64)
    onsite_edge_idx = np.stack([onsite_src, onsite_src])
    onsite_edge_fea = np.concatenate(
        [
            np.zeros((cart_coords.shape[0], 1), dtype=default_dtype),
            cart_coords,
            cart_coords,
            cart_coords,
        ],
        axis=-1,
    )
    edge_idx = np.concatenate([edge_idx, onsite_edge_idx], axis=1)
    edge_fea = np.concatenate([edge_fea, onsite_edge_fea], axis=0)
    src, dst = edge_idx

    num_atom = cart_coords.shape[0]
    atom_idx_connect, edge_idx_connect = [], []
    for atom_idx in range(num_atom):
        outgoing_edge_idx = np.where(src == atom_idx)[0]
        edge_idx_connect.append(outgoing_edge_idx)
        atom_idx_connect.append(dst[outgoing_edge_idx])
    return edge_idx, edge_fea, atom_idx_connect, edge_idx_connect


def _build_edges_from_rotation_keys(
    cart_coords,
    lattice,
    local_rotation_dict,
    default_dtype,
):
    edge_keys = list(local_rotation_dict)
    if not edge_keys:
        raise ValueError("DeepH local-coordinate file does not contain any edges.")

    edge_idx = np.asarray([[key[3], key[4]] for key in edge_keys], dtype=np.int64).T
    lattice_shifts = np.asarray([key[:3] for key in edge_keys], dtype=default_dtype)
    src, dst = edge_idx
    dst_cart_periodic = cart_coords[dst] + lattice_shifts @ lattice
    edge_dist = np.linalg.norm(
        dst_cart_periodic - cart_coords[src], axis=1, keepdims=True
    )
    edge_fea = np.concatenate(
        [
            edge_dist,
            cart_coords[src],
            dst_cart_periodic,
            cart_coords[dst],
        ],
        axis=-1,
    ).astype(default_dtype)

    atom_idx_connect = []
    edge_idx_connect = []
    for atom_idx in range(cart_coords.shape[0]):
        outgoing_edge_idx = np.where(src == atom_idx)[0]
        if outgoing_edge_idx.size == 0:
            raise ValueError(f"Atom {atom_idx} has no overlap-derived DeepH edges.")
        edge_idx_connect.append(outgoing_edge_idx)
        atom_idx_connect.append(dst[outgoing_edge_idx])

    local_rotation = np.stack(
        [local_rotation_dict[key] for key in edge_keys], axis=0
    ).astype(default_dtype)
    return (
        edge_idx,
        edge_fea,
        atom_idx_connect,
        edge_idx_connect,
        local_rotation,
    )


def _attach_terms(
    edge_idx,
    edge_fea,
    lattice,
    atom_num_orbital,
    read_terms,
    local_rotation_dict,
    default_dtype,
):
    max_num_orbital = max(atom_num_orbital)
    term_mask = np.zeros(edge_fea.shape[0], dtype=bool)
    term_real = np.full(
        [edge_fea.shape[0], max_num_orbital, max_num_orbital],
        np.nan,
        dtype=default_dtype,
    )
    local_rotation = []
    inv_lattice = np.linalg.inv(lattice).astype(default_dtype)

    for index_edge in range(edge_fea.shape[0]):
        lattice_shift = (
            np.rint(
                edge_fea[index_edge, 4:7] @ inv_lattice
                - edge_fea[index_edge, 7:10] @ inv_lattice
            )
            .astype(int)
            .tolist()
        )
        i, j = edge_idx[:, index_edge]
        key_term = (*lattice_shift, int(i), int(j))
        if key_term not in read_terms:
            raise NotImplementedError(
                "Graph radius including hopping without calculation is not supported."
            )
        term_mask[index_edge] = True
        term_real[
            index_edge,
            : atom_num_orbital[i],
            : atom_num_orbital[j],
        ] = read_terms[key_term]
        local_rotation.append(local_rotation_dict[key_term])

    return term_mask, term_real, np.stack(local_rotation, axis=0)


def build_deeph_graph(
    cart_coords,
    numbers,
    stru_id,
    lattice,
    default_dtype,
    tb_folder,
    interface,
    num_l,
    create_from_DFT,
    if_lcmp_graph,
    separate_onsite,
    target="hamiltonian",
    huge_structure=False,
    if_new_sp=False,
    converter_graph=None,
):
    _validate_supported_path(
        interface,
        target,
        create_from_DFT,
        if_lcmp_graph,
        separate_onsite,
        if_new_sp,
        huge_structure,
    )
    if tb_folder is None:
        raise ValueError("DeepH graph construction requires tb_folder.")
    cart_coords = np.asarray(cart_coords, dtype=default_dtype)
    numbers = np.asarray(numbers, dtype=np.int64)
    lattice = np.asarray(lattice, dtype=default_dtype)

    term_mask = term_real = None
    if create_from_DFT:
        atom_num_orbital, local_rotation_dict, read_terms = load_deeph_matrices(
            tb_folder,
            default_dtype,
            include_hamiltonian=interface == "npz",
        )
        (
            edge_idx,
            edge_fea,
            atom_idx_connect,
            edge_idx_connect,
            local_rotation,
        ) = _build_edges_from_rotation_keys(
            cart_coords,
            lattice,
            local_rotation_dict,
            default_dtype,
        )
        if interface == "npz":
            term_mask, term_real, _ = _attach_terms(
                edge_idx,
                edge_fea,
                lattice,
                atom_num_orbital,
                read_terms,
                local_rotation_dict,
                default_dtype,
            )
    else:
        if converter_graph is None:
            raise ValueError(
                "Radius-based DeepH graph construction expects a "
                "PaddleMaterials graph_converter graph."
            )
        (
            edge_idx,
            edge_fea,
            atom_idx_connect,
            edge_idx_connect,
        ) = _build_edges_from_converter_graph(converter_graph, default_dtype)
        atom_num_orbital, local_rotation_dict, read_terms = load_deeph_matrices(
            tb_folder, default_dtype
        )
        term_mask, term_real, local_rotation = _attach_terms(
            edge_idx,
            edge_fea,
            lattice,
            atom_num_orbital,
            read_terms,
            local_rotation_dict,
            default_dtype,
        )
    subgraph = build_lcmp_subgraph(
        edge_idx,
        edge_fea,
        local_rotation,
        atom_idx_connect,
        edge_idx_connect,
        num_l,
    )

    data = SimpleNamespace(
        x=numbers,
        edge_index=edge_idx,
        edge_attr=edge_fea,
        stru_id=stru_id,
        onsite_term_real=None,
        atom_num_orbital=np.asarray(atom_num_orbital, dtype=np.int64),
        subgraph_dict=subgraph,
        spinful=False,
    )
    if term_mask is not None:
        data.term_mask = term_mask
        data.term_real = term_real
    return data

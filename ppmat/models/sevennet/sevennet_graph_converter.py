# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0.

"""Graph converter for SevenNet."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from ase.neighborlist import primitive_neighbor_list
from pymatgen.core import Structure

from ppmat.datasets.geometric_data_type.data import Data


class SevenNetGraphConverter:
    """Convert pymatgen structures to SevenNet atom graphs.

    Graph attributes are stored as NumPy arrays so that graphs are compatible
    with PaddleMaterials dataset caching via pickle. Conversion to Paddle
    tensors is performed at the SevenNet model boundary.

    Args:
        cutoff: Neighbor cutoff radius in Angstrom.
    """

    def __init__(
        self,
        cutoff: float = 5.0,
        **kwargs,
    ):
        self.cutoff = float(cutoff)

        if self.cutoff <= 0:
            raise ValueError(
                "cutoff must be positive."
            )

    def __call__(
        self,
        structure,
    ):
        if isinstance(
            structure,
            Structure,
        ):
            return self.get_graph(
                structure
            )

        if isinstance(
            structure,
            Sequence,
        ):
            return [
                self.get_graph(item)
                for item in structure
            ]

        raise TypeError(
            "SevenNetGraphConverter expects a pymatgen Structure "
            f"or a sequence of Structures, but got "
            f"{type(structure).__name__}."
        )

    def get_graph(
        self,
        structure: Structure,
    ) -> Data:
        """Build one cache-safe SevenNet graph."""

        if not isinstance(
            structure,
            Structure,
        ):
            raise TypeError(
                "Expected pymatgen Structure, "
                f"got {type(structure).__name__}."
            )

        pos64 = np.asarray(
            structure.cart_coords,
            dtype=np.float64,
        )

        cell = np.asarray(
            structure.lattice.matrix,
            dtype=np.float64,
        )

        pbc = np.asarray(
            [True, True, True],
            dtype=bool,
        )

        (
            edge_src,
            edge_dst,
            edge_vec,
            _,
        ) = primitive_neighbor_list(
            "ijDS",
            pbc,
            cell,
            pos64,
            self.cutoff,
            self_interaction=True,
        )

        # Match upstream SevenNet: remove only zero-vector self interactions.
        is_zero_vec = np.all(
            edge_vec == 0,
            axis=1,
        )

        is_self = (
            edge_src
            == edge_dst
        )

        keep = ~(
            is_zero_vec
            & is_self
        )

        edge_src = np.asarray(
            edge_src[keep],
            dtype=np.int64,
        )

        edge_dst = np.asarray(
            edge_dst[keep],
            dtype=np.int64,
        )

        edge_vec = np.asarray(
            edge_vec[keep],
            dtype=np.float32,
        )

        edge_index = np.stack(
            [
                edge_src,
                edge_dst,
            ],
            axis=0,
        ).astype(
            np.int64,
            copy=False,
        )

        atomic_numbers = np.asarray(
            [
                int(site.specie.Z)
                for site in structure
            ],
            dtype=np.int64,
        )

        pos = np.asarray(
            pos64,
            dtype=np.float32,
        )

        num_atoms = int(
            atomic_numbers.shape[0]
        )

        cell_volume = np.asarray(
            abs(
                np.linalg.det(
                    cell
                )
            ),
            dtype=np.float32,
        )

        # IMPORTANT:
        # Keep graph fields as NumPy arrays here. MPTrjDataset caches the Data
        # object using pickle, while Batch.from_data_list natively supports
        # NumPy graph attributes.
        graph = Data(
            x=atomic_numbers.copy(),

            edge_index=edge_index,

            pos=pos,

            atomic_numbers=atomic_numbers.copy(),

            edge_vec=edge_vec,

            cell_volume=cell_volume,

            num_atoms=np.asarray(
                num_atoms,
                dtype=np.int64,
            ),

            num_nodes=num_atoms,
        )

        return graph

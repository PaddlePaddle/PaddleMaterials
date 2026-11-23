# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0.

"""Standard PaddleMaterials periodic graphs for SevenNet."""

import numpy as np

from ppmat.models.common.graph_converter import FindPointsInSpheres


class SevenNetGraphConverter(FindPointsInSpheres):
    """Reuse the repository's periodic graph builder with a fixed cutoff."""

    def __init__(self, cutoff=5.0, num_cpus=1):
        super().__init__(cutoff=cutoff, num_cpus=num_cpus)

    def get_graph_by_find_points_in_spheres(self, structure):
        graph = super().get_graph_by_find_points_in_spheres(structure)
        if graph is None:
            return self.build_pgl_graph(structure, [], [])

        # The shared converter may retry an isolated structure at a larger
        # radius. SevenNet's neighbor topology must use the model cutoff.
        keep = graph.edge_feat["bond_dist"] <= self.cutoff
        if not np.all(keep):
            return self.build_pgl_graph(
                structure, graph.edges[keep], graph.edge_feat["pbc_offset"][keep]
            )
        return graph

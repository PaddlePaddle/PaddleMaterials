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

import numbers
from collections.abc import Mapping
from collections.abc import Sequence
from typing import Any
from typing import List

import numpy as np
import paddle
import pgl

from ppmat.datasets.custom_data_type import ConcatData
from ppmat.datasets.custom_data_type import ConcatNumpyWarper
from ppmat.datasets.geometric_data_type.batch import Batch
from ppmat.datasets.geometric_data_type.data import Data
from ppmat.utils import logger
from ppmat.utils.pgl_compat import patch_pgl_empty_edge_batch

patch_pgl_empty_edge_batch()


class DefaultCollator(object):
    def __call__(self, batch: List[Any]) -> Any:
        """Default_collate_fn for paddle dataloader.

        NOTE: This `default_collate_fn` is different from official `default_collate_fn`
        which specially adapt case where sample is `None` and `pgl.Graph`.

        ref: https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/io/dataloader/collate.py#L25

        Args:
            batch (List[Any]): Batch of samples to be collated.

        Returns:
            Any: Collated batch data.
        """
        sample = batch[0]
        if sample is None:
            return None
        elif isinstance(sample, ConcatNumpyWarper):
            batch = np.concatenate(batch, axis=0)
            return batch
        elif isinstance(sample, np.ndarray):
            batch = np.stack(batch, axis=0)
            return batch
        elif isinstance(sample, (paddle.Tensor, paddle.framework.core.eager.Tensor)):
            return paddle.stack(batch, axis=0)
        elif isinstance(sample, numbers.Number):
            batch = np.array(batch)
            return batch
        elif isinstance(sample, Data):
            # Geometric `Data` objects: batch them into a single `Batch`
            return Batch.from_data_list(batch)
        elif isinstance(sample, (str, bytes)):
            return batch
        elif isinstance(sample, Mapping):
            return {key: self([d[key] for d in batch]) for key in sample}
        elif isinstance(sample, Sequence):
            sample_fields_num = len(sample)
            if not all(len(sample) == sample_fields_num for sample in iter(batch)):
                raise RuntimeError("Fields number not same among samples in a batch")
            return [self(fields) for fields in zip(*batch)]
        elif str(type(sample)) == "<class 'pgl.graph.Graph'>":
            # use str(type()) instead of isinstance() in case of pgl is not installed.
            graphs = pgl.Graph.batch(batch)
            # NOTE: when num_works >1, graphs.tensor() will convert numpy.ndarray to
            # CPU Tensor, which will cause error in model training.
            # graphs.tensor()
            return graphs
        elif isinstance(sample, ConcatData):
            return ConcatData.batch(batch)
        raise TypeError(
            "batch data can only contains: paddle.Tensor, numpy.ndarray, "
            f"dict, list, number, None, pgl.Graph, but got {type(sample)}"
        )


class RadiusGraphCollator:
    """Collate PGL radius graphs and offset their edge-based triplet indices."""

    def __call__(self, batch):
        graphs = [sample["graph"] for sample in batch]
        num_edges = [np.asarray(graph.edges).shape[0] for graph in graphs]
        edge_offsets = np.cumsum([0] + num_edges[:-1])

        triplet_fields = {key: [] for key in ("idx_kj", "idx_ji")}
        for index, graph in enumerate(graphs):
            for key in triplet_fields:
                triplet_fields[key].append(
                    np.asarray(graph.edge_feat[f"ti_{key}"], dtype=np.int64)
                    + edge_offsets[index]
                )

        graph = pgl.Graph.batch(graphs)
        graph.edge_feat.update(
            {
                f"ti_{key}": np.concatenate(values)
                for key, values in triplet_fields.items()
            }
        )

        result = DefaultCollator()(
            [
                {key: value for key, value in sample.items() if key != "graph"}
                for sample in batch
            ]
        )
        result["graph"] = graph
        return result


class DensityCollator:
    def __init__(
        self,
        *,
        clip_max: float | None = None,
        pad_skew_warn_ratio: float = 4.0,
    ):
        self.clip_max = clip_max
        self.pad_skew_warn_ratio = float(pad_skew_warn_ratio)
        if self.pad_skew_warn_ratio <= 0.0:
            raise ValueError("pad_skew_warn_ratio must be positive.")

    def __call__(self, batch):
        densities = [sample["density"] for sample in batch]
        grid_coord = [sample["grid_coord"] for sample in batch]
        for density, coordinates in zip(densities, grid_coord):
            density_length = int(density.shape[0])
            coordinate_length = int(coordinates.shape[0])
            if density_length != coordinate_length:
                raise ValueError(
                    f"Density length ({density_length}) and grid length "
                    f"({coordinate_length}) must match."
                )
            if density_length == 0:
                raise ValueError("Empty density/grid pair encountered in batch.")
        channel_shapes = {tuple(density.shape[1:]) for density in densities}
        if len(channel_shapes) > 1:
            raise ValueError(
                "Density channel shapes must match across a batch, but got "
                f"{sorted(channel_shapes)}."
            )

        prepared_density, prepared_grid, masks = [], [], []
        lengths = [int(density.shape[0]) for density in densities]
        max_length = max(lengths)
        min_length = min(lengths)
        if min_length == max_length:
            # Every field already has the same length, so there is nothing to
            # mask. ``None`` lets the models skip the all-ones multiply.
            prepared_density = densities
            prepared_grid = grid_coord
            masks = [None] * len(densities)
        else:
            if min_length * self.pad_skew_warn_ratio < max_length:
                # Padding allocates every field at the longest length and the
                # padded grid points still cost graph edges and orbital math.
                logger.warning(
                    "Density padding is batching fields of very different "
                    f"sizes ({min_length} to {max_length} grid points): "
                    f"{sum(lengths) / (len(lengths) * max_length):.1%} of the "
                    "padded batch carries real data. Consider setting the "
                    "dataset's grid_sampler_cfg, or grouping fields of similar "
                    "size."
                )
            for density, coordinates in zip(densities, grid_coord):
                length = int(density.shape[0])
                pad_width = [(0, max_length - length)] + [
                    (0, 0) for _ in range(density.ndim - 1)
                ]
                prepared_density.append(np.pad(density, pad_width, constant_values=0.0))
                prepared_grid.append(
                    np.pad(
                        coordinates,
                        ((0, max_length - length), (0, 0)),
                        constant_values=0.0,
                    )
                )
                mask = np.zeros_like(prepared_density[-1], dtype=np.float32)
                mask[:length] = 1.0
                masks.append(mask)

        prepared_batch = []
        for sample, density, coordinates, mask in zip(
            batch,
            prepared_density,
            prepared_grid,
            masks,
        ):
            prepared_sample = dict(sample)
            prepared_sample["density"] = density
            prepared_sample["density_mask"] = mask
            prepared_sample["grid_coord"] = coordinates
            prepared_sample["info"] = {
                "cell": sample["info"]["cell"],
            }
            prepared_batch.append(prepared_sample)

        result = DefaultCollator()(prepared_batch)
        if self.clip_max is not None:
            result["density"] = np.minimum(result["density"], self.clip_max)
        return result

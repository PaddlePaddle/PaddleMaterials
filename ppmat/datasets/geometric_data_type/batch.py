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

# This code is adapted from https://github.com/jediofgever/pytorch_geometric

from collections.abc import Sequence
from typing import List

import numpy as np
import paddle

from ppmat.datasets.geometric_data_type.data import Data
from ppmat.datasets.geometric_data_type.dataset import IndexType

# from ppmat.utils.paddle_aux import *


class Batch(Data):
    """A plain old python object modeling a batch of graphs as one big
    (disconnected) graph. With :class:`torch_geometric.data.Data` being the
    base class, all its methods can also be used here.
    In addition, single graphs can be reconstructed via the assignment vector
    :obj:`batch`, which maps each node to its respective graph identifier.
    """

    def __init__(self, batch=None, ptr=None, **kwargs):
        super(Batch, self).__init__(**kwargs)
        for key, item in kwargs.items():
            if key == "num_nodes":
                self.__num_nodes__ = item
            else:
                self[key] = item
        self.batch = batch
        self.ptr = ptr
        self.__data_class__ = Data
        self.__slices__ = None
        self.__cumsum__ = None
        self.__cat_dims__ = None
        self.__num_nodes_list__ = None
        self.__num_graphs__ = None

    @classmethod
    def from_data_list(cls, data_list, follow_batch=[], exclude_keys=[]):
        """Constructs a batch object from a python list holding
        :class:`torch_geometric.data.Data` objects.
        The assignment vector :obj:`batch` is created on the fly.
        Additionally, creates assignment batch vectors for each key in
        :obj:`follow_batch`.
        Will exclude any keys given in :obj:`exclude_keys`."""
        if not data_list:
            raise ValueError("data_list must not be empty.")
        keys = list(set(data_list[0].keys) - set(exclude_keys))
        assert "batch" not in keys and "ptr" not in keys
        batch = cls()
        for key in data_list[0].__dict__.keys():
            if key[:2] != "__" and key[-2:] != "__":
                batch[key] = None
        batch.__num_graphs__ = len(data_list)
        batch.__data_class__ = data_list[0].__class__
        for key in keys + ["batch"]:
            batch[key] = []
        batch["ptr"] = [0]
        slices = {key: [0] for key in keys}
        cumsum = {key: [0] for key in keys}
        cat_dims = {}
        num_nodes_list = []
        for i, data in enumerate(data_list):
            for key in keys:
                original_item = data[key]
                item = original_item
                cum = cumsum[key][-1]
                if isinstance(item, paddle.Tensor) and item.dtype != paddle.bool:
                    if not isinstance(cum, int) or cum != 0:
                        item = item + cum
                elif isinstance(item, np.ndarray) and item.dtype != np.bool_:
                    if np.any(np.asarray(cum) != 0):
                        item = item + cum
                elif isinstance(item, (int, float)):
                    item = item + cum
                size = 1
                cat_dim = data.__cat_dim__(key, original_item)
                if isinstance(item, paddle.Tensor) and item.dim() == 0:
                    cat_dim = None
                elif isinstance(item, np.ndarray) and item.ndim == 0:
                    cat_dim = None
                if key in cat_dims and cat_dims[key] != cat_dim:
                    raise ValueError(
                        f"Inconsistent concatenation dimension for {key!r}."
                    )
                cat_dims[key] = cat_dim
                if isinstance(item, paddle.Tensor) and cat_dim is None:
                    cat_dim = 0
                    item = item.unsqueeze(axis=0)
                elif isinstance(item, paddle.Tensor):
                    size = item.shape[cat_dim]
                elif isinstance(item, np.ndarray) and cat_dim is None:
                    cat_dim = 0
                    item = np.expand_dims(item, axis=0)
                elif isinstance(item, np.ndarray):
                    size = item.shape[cat_dim]
                batch[key].append(item)
                slices[key].append(size + slices[key][-1])
                inc = data.__inc__(key, original_item)
                if isinstance(inc, (tuple, list)):
                    inc = (
                        np.asarray(inc)
                        if isinstance(original_item, np.ndarray)
                        else paddle.to_tensor(data=inc)
                    )
                cumsum[key].append(inc + cumsum[key][-1])
                if key in follow_batch:
                    if isinstance(size, paddle.Tensor):
                        for j, size in enumerate(size.tolist()):
                            tmp = f"{key}_{j}_batch"
                            batch[tmp] = [] if i == 0 else batch[tmp]
                            batch[tmp].append(
                                paddle.full(shape=(size,), fill_value=i, dtype="int64")
                            )
                    else:
                        tmp = f"{key}_batch"
                        batch[tmp] = [] if i == 0 else batch[tmp]
                        batch[tmp].append(
                            paddle.full(shape=(size,), fill_value=i, dtype="int64")
                        )
            if hasattr(data, "__num_nodes__"):
                num_nodes_list.append(data.__num_nodes__)
            else:
                num_nodes_list.append(None)
            num_nodes = data.num_nodes
            if num_nodes is not None:
                item = paddle.full(shape=(num_nodes,), fill_value=i, dtype="int64")
                batch.batch.append(item)
                batch.ptr.append(batch.ptr[-1] + num_nodes)
        batch.batch = None if len(batch.batch) == 0 else batch.batch
        batch.ptr = None if len(batch.ptr) == 1 else batch.ptr
        batch.__slices__ = slices
        batch.__cumsum__ = cumsum
        batch.__cat_dims__ = cat_dims
        batch.__num_nodes_list__ = num_nodes_list
        ref_data = data_list[0]
        for key in batch.keys:
            items = batch[key]
            item = items[0]
            cat_dim = cat_dims.get(key, ref_data.__cat_dim__(key, item))
            cat_dim = 0 if cat_dim is None else cat_dim
            if isinstance(item, paddle.Tensor):
                if not all(isinstance(value, paddle.Tensor) for value in items):
                    raise TypeError(f"All values for {key!r} must be Paddle tensors.")
                batch[key] = paddle.concat(x=items, axis=cat_dim)
            elif isinstance(item, np.ndarray):
                if not all(isinstance(value, np.ndarray) for value in items):
                    raise TypeError(f"All values for {key!r} must be NumPy arrays.")
                batch[key] = np.concatenate(items, axis=cat_dim)
            elif isinstance(item, (int, float)):
                batch[key] = paddle.to_tensor(data=items)
        return batch.contiguous()

    def get_example(self, idx: int) -> Data:
        """Reconstructs the :class:`torch_geometric.data.Data` object at index
        :obj:`idx` from the batch object.
        The batch object must have been created via :meth:`from_data_list` in
        order to be able to reconstruct the initial objects."""
        if self.__slices__ is None:
            raise RuntimeError(
                "Cannot reconstruct data list from batch because the batch object "
                "was not created using `Batch.from_data_list()`."
            )
        data = self.__data_class__()
        idx = self.num_graphs + idx if idx < 0 else idx
        for key in self.__slices__.keys():
            item = self[key]
            if self.__cat_dims__[key] is None:
                item = item[idx]
                if isinstance(self[key], np.ndarray):
                    item = np.asarray(item)
            elif isinstance(item, paddle.Tensor):
                dim = self.__cat_dims__[key]
                start = self.__slices__[key][idx]
                end = self.__slices__[key][idx + 1]
                start_0 = item.shape[dim] + start if start < 0 else start
                item = paddle.slice(item, [dim], [start_0], [start_0 + (end - start)])
            elif isinstance(item, np.ndarray):
                dim = self.__cat_dims__[key]
                start = self.__slices__[key][idx]
                end = self.__slices__[key][idx + 1]
                item_slice = [slice(None)] * item.ndim
                item_slice[dim] = slice(start, end)
                item = item[tuple(item_slice)]
            else:
                start = self.__slices__[key][idx]
                end = self.__slices__[key][idx + 1]
                item = item[start:end]
                item = item[0] if len(item) == 1 else item
            cum = self.__cumsum__[key][idx]
            if isinstance(item, paddle.Tensor) and item.dtype != paddle.bool:
                if not isinstance(cum, int) or cum != 0:
                    item = item - cum
            elif isinstance(item, np.ndarray) and item.dtype != np.bool_:
                if np.any(np.asarray(cum) != 0):
                    item = item - cum
            elif isinstance(item, (int, float)):
                item = item - cum
            data[key] = item
        if self.__num_nodes_list__[idx] is not None:
            data.num_nodes = self.__num_nodes_list__[idx]
        return data

    def index_select(self, idx: IndexType) -> List[Data]:
        if isinstance(idx, slice):
            idx = list(range(self.num_graphs)[idx])
        elif isinstance(idx, paddle.Tensor) and idx.dtype == paddle.int64:
            idx = idx.flatten().tolist()
        elif isinstance(idx, paddle.Tensor) and idx.dtype == paddle.bool:
            idx = idx.flatten().nonzero(as_tuple=False).flatten().tolist()
        elif isinstance(idx, np.ndarray) and idx.dtype == np.int64:
            idx = idx.flatten().tolist()
        elif isinstance(idx, np.ndarray) and idx.dtype == np.bool_:
            idx = idx.flatten().nonzero()[0].flatten().tolist()
        elif isinstance(idx, Sequence) and not isinstance(idx, str):
            pass
        else:
            raise IndexError(
                "Batch indices must be an integer, slice, non-string sequence, "
                "or a paddle.Tensor/numpy.ndarray with int64 or bool dtype; "
                f"got {type(idx).__name__}."
            )
        return [self.get_example(i) for i in idx]

    def __getitem__(self, idx):
        if isinstance(idx, str):
            return super(Batch, self).__getitem__(idx)
        elif isinstance(idx, (int, np.integer)):
            return self.get_example(idx)
        else:
            return self.index_select(idx)

    def to_data_list(self) -> List[Data]:
        """Reconstructs the list of :class:`torch_geometric.data.Data` objects
        from the batch object.
        The batch object must have been created via :meth:`from_data_list` in
        order to be able to reconstruct the initial objects."""
        return [self.get_example(i) for i in range(self.num_graphs)]

    @property
    def num_graphs(self) -> int:
        """Returns the number of graphs in the batch."""
        if self.__num_graphs__ is not None:
            return self.__num_graphs__
        elif self.ptr is not None:
            return self.ptr.size - 1
        elif self.batch is not None:
            return int(self.batch.max_func()) + 1
        else:
            raise ValueError

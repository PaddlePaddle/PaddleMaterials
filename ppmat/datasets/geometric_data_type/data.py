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

import collections
import copy
import re

import numpy as np
import paddle

# from .paddle_utils import *
# from ppmat.utils.paddle_aux import *

__num_nodes_warn_msg__ = (
    "The number of nodes in your data object can only be inferred by its {} "
    "indices, and hence may result in unexpected batch-wise behavior, e.g., "
    "in case there exists isolated nodes. Please set data.num_nodes explicitly."
)


def size_repr(key, item, indent=0):
    indent_str = " " * indent
    if paddle.is_tensor(x=item) and item.dim() == 0:
        out = item.item()
    elif paddle.is_tensor(x=item):
        out = str(list(tuple(item.shape)))
    elif isinstance(item, np.ndarray) and item.ndim == 0:
        out = item.item()
    elif isinstance(item, np.ndarray):
        out = str(list(item.shape))
    elif isinstance(item, list) or isinstance(item, tuple):
        out = str([len(item)])
    elif isinstance(item, dict):
        lines = [(indent_str + size_repr(k, v, 2)) for k, v in item.items()]
        out = "{\n" + ",\n".join(lines) + "\n" + indent_str + "}"
    elif isinstance(item, str):
        out = f'"{item}"'
    else:
        out = str(item)
    return f"{indent_str}{key}={out}"


class Data(object):
    """A plain old python object modeling a single graph with various
    (optional) attributes:

    Args:
        x (Tensor or numpy.ndarray, optional): Node feature matrix with shape
            :obj:`[num_nodes, num_node_features]`. (default: :obj:`None`)
        edge_index (Tensor or numpy.ndarray, optional): Graph connectivity in
            COO format with shape :obj:`[2, num_edges]`. (default: :obj:`None`)
        edge_attr (Tensor or numpy.ndarray, optional): Edge feature matrix with shape
            :obj:`[num_edges, num_edge_features]`. (default: :obj:`None`)
        y (Tensor or numpy.ndarray, optional): Graph or node targets with
            arbitrary shape. (default: :obj:`None`)
        pos (Tensor or numpy.ndarray, optional): Node position matrix with shape
            :obj:`[num_nodes, num_dimensions]`. (default: :obj:`None`)
        normal (Tensor or numpy.ndarray, optional): Normal vector matrix with shape
            :obj:`[num_nodes, num_dimensions]`. (default: :obj:`None`)
        face (Tensor or numpy.ndarray, optional): Face adjacency matrix with
            shape :obj:`[3, num_faces]`. (default: :obj:`None`)

    The data object is not restricted to these attributes and can be extended
    by any other additional data.

    Example::

        data = Data(x=x, edge_index=edge_index)
        data.train_idx = torch.tensor([...], dtype=torch.long)
        data.test_mask = torch.tensor([...], dtype=torch.bool)
    """

    def __init__(
        self,
        x=None,
        edge_index=None,
        edge_attr=None,
        y=None,
        pos=None,
        normal=None,
        face=None,
        **kwargs,
    ):
        self.x = x
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        self.y = y
        self.pos = pos
        self.normal = normal
        self.face = face
        for key, item in kwargs.items():
            if key == "num_nodes":
                self.__num_nodes__ = item
            else:
                self[key] = item

        if edge_index is not None and not isinstance(
            edge_index, (paddle.Tensor, np.ndarray)
        ):
            raise ValueError(
                "Argument `edge_index` needs to be a paddle.Tensor or "
                f"numpy.ndarray but found type `{type(edge_index)}`"
            )

        if face is not None and not isinstance(face, (paddle.Tensor, np.ndarray)):
            raise ValueError(
                "Argument `face` needs to be a paddle.Tensor or numpy.ndarray "
                f"but found type `{type(face)}`"
            )

    @classmethod
    def from_dict(cls, dictionary):
        """Creates a data object from a python dictionary."""
        data = cls()
        for key, item in dictionary.items():
            data[key] = item
        return data

    def to_dict(self):
        return {key: item for key, item in self}

    def to_namedtuple(self):
        keys = self.keys
        DataTuple = collections.namedtuple("DataTuple", keys)
        return DataTuple(*[self[key] for key in keys])

    def __getitem__(self, key):
        """Gets the data of the attribute :obj:`key`."""
        return getattr(self, key, None)

    def __setitem__(self, key, value):
        """Sets the attribute :obj:`key` to :obj:`value`."""
        setattr(self, key, value)

    def __delitem__(self, key):
        """Delete the data of the attribute :obj:`key`."""
        return delattr(self, key)

    @property
    def keys(self):
        """Returns all names of graph attributes."""
        keys = [key for key in self.__dict__.keys() if self[key] is not None]
        keys = [key for key in keys if key[:2] != "__" and key[-2:] != "__"]
        return keys

    def __len__(self):
        """Returns the number of all present attributes."""
        return len(self.keys)

    def __contains__(self, key):
        """Returns :obj:`True`, if the attribute :obj:`key` is present in the
        data."""
        return key in self.keys

    def __iter__(self):
        """Iterates over all present attributes in the data, yielding their
        attribute names and content."""
        for key in sorted(self.keys):
            yield key, self[key]

    def __call__(self, *keys):
        """Iterates over all attributes :obj:`*keys` in the data, yielding
        their attribute names and content.
        If :obj:`*keys` is not given this method will iterative over all
        present attributes."""
        for key in sorted(self.keys) if not keys else keys:
            if key in self:
                yield key, self[key]

    def __cat_dim__(self, key, value):
        """Returns the dimension for which :obj:`value` of attribute
        :obj:`key` will get concatenated when creating batches.

        .. note::

            This method is for internal use only, and should only be overridden
            if the batch concatenation process is corrupted for a specific data
            attribute.
        """
        if bool(re.search("(index|face)", key)):
            return -1
        return 0

    def __inc__(self, key, value):
        """Returns the incremental count to cumulatively increase the value
        of the next attribute of :obj:`key` when creating batches.

        .. note::

            This method is for internal use only, and should only be overridden
            if the batch concatenation process is corrupted for a specific data
            attribute.
        """
        return self.num_nodes if bool(re.search("(index|face)", key)) else 0

    @property
    def num_nodes(self):
        """Returns or sets the number of nodes in the graph.

        .. note::
            The number of nodes in your data object is typically automatically
            inferred, *e.g.*, when node features :obj:`x` are present.
            In some cases however, a graph may only be given by its edge
            indices :obj:`edge_index`.
            PyTorch Geometric then *guesses* the number of nodes
            according to :obj:`edge_index.max().item() + 1`, but in case there
            exists isolated nodes, this number has not to be correct and can
            therefore result in unexpected batch-wise behavior.
            Thus, we recommend to set the number of nodes in your data object
            explicitly via :obj:`data.num_nodes = ...`.
            You will be given a warning that requests you to do so.
        """
        if hasattr(self, "__num_nodes__"):
            return self.__num_nodes__
        for key, item in self("x", "pos", "normal", "batch"):
            return item.shape[self.__cat_dim__(key, item)]
        if hasattr(self, "adj"):
            return self.adj.shape[0]
        if hasattr(self, "adj_t"):
            return self.adj_t.shape[1]
        return None

    @num_nodes.setter
    def num_nodes(self, num_nodes):
        self.__num_nodes__ = num_nodes

    @property
    def num_edges(self):
        """
        Returns the number of edges in the graph.
        For undirected graphs, this will return the number of bi-directional
        edges, which is double the amount of unique edges.
        """
        for key, item in self("edge_index", "edge_attr"):
            return item.shape[self.__cat_dim__(key, item)]
        for key, item in self("adj", "adj_t"):
            return item.nnz()
        return None

    @property
    def num_faces(self):
        """Returns the number of faces in the mesh."""
        if self.face is not None:
            return self.face.shape[self.__cat_dim__("face", self.face)]
        return None

    @property
    def num_node_features(self):
        """Returns the number of features per node in the graph."""
        if self.x is None:
            return 0
        return 1 if len(self.x.shape) == 1 else self.x.shape[1]

    @property
    def num_features(self):
        """Alias for :py:attr:`~num_node_features`."""
        return self.num_node_features

    @property
    def num_edge_features(self):
        """Returns the number of features per edge in the graph."""
        if self.edge_attr is None:
            return 0
        return 1 if len(self.edge_attr.shape) == 1 else self.edge_attr.shape[1]

    def __apply__(self, item, func):
        if paddle.is_tensor(x=item):
            return func(item)
        elif isinstance(item, (tuple, list)):
            return [self.__apply__(v, func) for v in item]
        elif isinstance(item, dict):
            return {k: self.__apply__(v, func) for k, v in item.items()}
        else:
            return item

    def apply(self, func, *keys):
        """Applies the function :obj:`func` to all tensor attributes
        :obj:`*keys`. If :obj:`*keys` is not given, :obj:`func` is applied to
        all present attributes.
        """
        for key, item in self(*keys):
            self[key] = self.__apply__(item, func)
        return self

    def contiguous(self, *keys):
        """Ensures a contiguous memory layout for all attributes :obj:`*keys`.
        If :obj:`*keys` is not given, all present attributes are ensured to
        have a contiguous memory layout."""
        return self.apply(lambda x: x.contiguous(), *keys)

    def to(self, device, *keys, **kwargs):
        """Performs tensor dtype and/or device conversion to all attributes
        :obj:`*keys`.
        If :obj:`*keys` is not given, the conversion is applied to all present
        attributes."""
        return self.apply(lambda x: x.to(device, **kwargs), *keys)

    def cpu(self, *keys):
        """Copies all attributes :obj:`*keys` to CPU memory.
        If :obj:`*keys` is not given, the conversion is applied to all present
        attributes."""
        return self.apply(lambda x: x.cpu(), *keys)

    def cuda(self, device=None, non_blocking=False, *keys):
        """Copies all attributes :obj:`*keys` to CUDA memory.
        If :obj:`*keys` is not given, the conversion is applied to all present
        attributes."""
        return self.apply(
            lambda x: x.cuda(device_id=device, blocking=not non_blocking), *keys
        )

    def clone(self):
        """Performs a deep-copy of the data object."""
        return self.__class__.from_dict(
            {
                k: (v.clone() if paddle.is_tensor(x=v) else copy.deepcopy(v))
                for k, v in self.__dict__.items()
            }
        )

    def pin_memory(self, *keys):
        """Copies all attributes :obj:`*keys` to pinned memory.
        If :obj:`*keys` is not given, the conversion is applied to all present
        attributes."""
        return self.apply(lambda x: x.pin_memory(), *keys)

    def debug(self):
        if self.edge_index is not None:
            expected_dtype = (
                np.dtype("int64")
                if isinstance(self.edge_index, np.ndarray)
                else paddle.int64
            )
            if self.edge_index.dtype != expected_dtype:
                raise RuntimeError(
                    "Expected edge indices of dtype {}, but found dtype  {}".format(
                        "int64", self.edge_index.dtype
                    )
                )
        if self.face is not None:
            expected_dtype = (
                np.dtype("int64") if isinstance(self.face, np.ndarray) else paddle.int64
            )
            if self.face.dtype != expected_dtype:
                raise RuntimeError(
                    "Expected face indices of dtype {}, but found dtype  {}".format(
                        "int64", self.face.dtype
                    )
                )
        if self.edge_index is not None:
            if len(self.edge_index.shape) != 2 or self.edge_index.shape[0] != 2:
                raise RuntimeError(
                    "Edge indices should have shape [2, num_edges], but found "
                    f"shape {tuple(self.edge_index.shape)}"
                )
        if self.edge_index is not None and self.num_nodes is not None:
            if isinstance(self.edge_index, np.ndarray):
                if self.edge_index.size > 0:
                    min_index = self.edge_index.min()
                    max_index = self.edge_index.max()
                else:
                    min_index = max_index = 0
            elif int(self.edge_index.numel()) > 0:
                min_index = self.edge_index.min().item()
                max_index = self.edge_index.max().item()
            else:
                min_index = max_index = 0
            if min_index < 0 or max_index > self.num_nodes - 1:
                raise RuntimeError(
                    f"Edge indices must lie in [0, {self.num_nodes - 1}], but "
                    f"found [{min_index}, {max_index}]."
                )
        if self.face is not None:
            if len(self.face.shape) != 2 or self.face.shape[0] != 3:
                raise RuntimeError(
                    "Face indices should have shape [3, num_faces], but found "
                    f"shape {tuple(self.face.shape)}"
                )
        if self.face is not None and self.num_nodes is not None:
            if isinstance(self.face, np.ndarray):
                if self.face.size > 0:
                    min_index = self.face.min()
                    max_index = self.face.max()
                else:
                    min_index = max_index = 0
            elif int(self.face.numel()) > 0:
                min_index = self.face.min().item()
                max_index = self.face.max().item()
            else:
                min_index = max_index = 0
            if min_index < 0 or max_index > self.num_nodes - 1:
                raise RuntimeError(
                    f"Face indices must lie in [0, {self.num_nodes - 1}], but "
                    f"found [{min_index}, {max_index}]."
                )
        if self.edge_index is not None and self.edge_attr is not None:
            if self.edge_index.shape[1] != self.edge_attr.shape[0]:
                raise RuntimeError(
                    "Edge indices and edge attributes contain different numbers "
                    f"of edges: {tuple(self.edge_index.shape)} and "
                    f"{tuple(self.edge_attr.shape)}."
                )
        if self.x is not None and self.num_nodes is not None:
            if self.x.shape[0] != self.num_nodes:
                raise RuntimeError(
                    f"Node features should contain {self.num_nodes} rows, but "
                    f"found {self.x.shape[0]}."
                )
        if self.pos is not None and self.num_nodes is not None:
            if self.pos.shape[0] != self.num_nodes:
                raise RuntimeError(
                    f"Node positions should contain {self.num_nodes} rows, but "
                    f"found {self.pos.shape[0]}."
                )
        if self.normal is not None and self.num_nodes is not None:
            if self.normal.shape[0] != self.num_nodes:
                raise RuntimeError(
                    f"Node normals should contain {self.num_nodes} rows, but "
                    f"found {self.normal.shape[0]}."
                )

    def __repr__(self):
        cls = str(self.__class__.__name__)
        has_dict = any([isinstance(item, dict) for _, item in self])
        if not has_dict:
            info = [size_repr(key, item) for key, item in self]
            return "{}({})".format(cls, ", ".join(info))
        else:
            info = [size_repr(key, item, indent=2) for key, item in self]
            return "{}(\n{}\n)".format(cls, ",\n".join(info))

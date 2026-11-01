# 版权所有（C）2026 苏州国家实验室、百度飞桨团队
# 本代码由苏州国家实验室与百度飞桨团队合作开发完成。
# 其中，苏州国家实验室提供模型框架设计思路，百度飞桨团队负责代码实现。
# 双方对本代码及相关成果享有同等权利。

# Copyright (C) 2026 Suzhou National Laboratory and Baidu PaddlePaddle team
# This code was jointly developed by Suzhou National Laboratory and the Baidu
# PaddlePaddle team.
# Suzhou National Laboratory provided the design concept for the model framework,
# and Baidu PaddlePaddle team was responsible for the code implementation.
# Both parties shall have equal rights to use this code and related work products.

# Licensed under the Apache License, Version 2.0 (the License);
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an AS IS BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import math
import os
import sys
from collections.abc import Callable
from typing import Any
from typing import Dict
from typing import List
from typing import Literal
from typing import Optional
from typing import Tuple

import numpy as np
import paddle
from tqdm import tqdm

from ppmat.models.common.activation import ScaledSiLU
from ppmat.models.common.activation import SiQU
from ppmat.models.common.initializer import he_orthogonal_init
from ppmat.models.common.initializer import linear_init_
from ppmat.models.common.radial_basis import RadialBasis
from ppmat.models.common.runtime import RuntimeMixin
from ppmat.models.common.runtime import runtime_boundary
from ppmat.models.common.spherical_basis import CircularBasisLayer
from ppmat.models.common.time_embedding import NoiseLevelEncoding
from ppmat.models.common.time_embedding import UniformTimestepSampler
from ppmat.models.mattergen.globals import _USE_UNCONDITIONAL_EMBEDDING
from ppmat.models.mattergen.globals import MAX_ATOMIC_NUM
from ppmat.models.mattergen.property_embeddings import PropertyEmbedding
from ppmat.models.mattergen.property_embeddings import SetConditionalEmbeddingType
from ppmat.models.mattergen.property_embeddings import SetEmbeddingType
from ppmat.models.mattergen.property_embeddings import SetPropertyScalers
from ppmat.models.mattergen.property_embeddings import SetUnconditionalEmbeddingType
from ppmat.models.mattergen.property_embeddings import get_use_unconditional_embedding
from ppmat.schedulers import build_scheduler
from ppmat.utils import logger
from ppmat.utils import paddle_aux  # noqa
from ppmat.utils.crystal import frac_to_cart_coords_with_lattice
from ppmat.utils.crystal import lattice_params_to_matrix_paddle
from ppmat.utils.io import read_value_json
from ppmat.utils.io import update_json
from ppmat.utils.misc import aggregate_per_sample
from ppmat.utils.misc import make_noise_symmetric_preserve_variance
from ppmat.utils.misc import ragged_range
from ppmat.utils.misc import repeat_blocks
from ppmat.utils.paddle_aux import dim2perm
from ppmat.utils.scatter import scatter


def inner_product_normalized(x: paddle.Tensor, y: paddle.Tensor) -> paddle.Tensor:
    """
    Calculate the inner product between the given normalized vectors,
    giving a result between -1 and 1.
    """
    return paddle.sum(x=x * y, axis=-1).clip(min=-1, max=1)


def get_max_neighbors_mask(
    natoms: paddle.Tensor,
    index: paddle.Tensor,
    atom_distance_squared: paddle.Tensor,
    max_num_neighbors_threshold: int,
) -> tuple[paddle.Tensor, paddle.Tensor]:
    """
    Give a mask that filters out edges so that each atom has at most
    `max_num_neighbors_threshold` neighbors.
    Assumes that `index` is sorted.
    """
    num_atoms = natoms.sum()

    num_neighbors = paddle.zeros(shape=num_atoms)
    num_neighbors.index_add_(axis=0, index=index, value=paddle.ones(shape=len(index)))
    num_neighbors = num_neighbors.astype(dtype="int64")
    max_num_neighbors = paddle.max(x=num_neighbors).astype(dtype="int64")
    _max_neighbors = copy.deepcopy(num_neighbors)
    _max_neighbors[
        _max_neighbors > max_num_neighbors_threshold
    ] = max_num_neighbors_threshold
    _num_neighbors = paddle.zeros(shape=num_atoms + 1).astype(dtype="int64")
    _natoms = paddle.zeros(shape=tuple(natoms.shape)[0] + 1).astype(dtype="int64")
    _num_neighbors[1:] = paddle.cumsum(x=_max_neighbors, axis=0)
    _natoms[1:] = paddle.cumsum(x=natoms, axis=0)
    num_neighbors_image = _num_neighbors[_natoms[1:]] - _num_neighbors[_natoms[:-1]]

    if (
        max_num_neighbors <= max_num_neighbors_threshold
        or max_num_neighbors_threshold <= 0
    ):
        mask_num_neighbors = paddle.to_tensor(data=[True], dtype=bool).expand_as(
            y=index
        )
        return mask_num_neighbors, num_neighbors_image
    distance_sort = paddle.full(
        shape=[num_atoms * max_num_neighbors], fill_value=np.inf
    )
    index_neighbor_offset = paddle.cumsum(x=num_neighbors, axis=0) - num_neighbors
    index_neighbor_offset_expand = paddle.repeat_interleave(
        x=index_neighbor_offset, repeats=num_neighbors
    )
    index_sort_map = (
        index * max_num_neighbors
        + paddle.arange(end=len(index))
        - index_neighbor_offset_expand  # noqa
    )
    distance_sort.scatter_(index_sort_map, atom_distance_squared)
    distance_sort = distance_sort.view(num_atoms, max_num_neighbors)
    distance_sort, index_sort = paddle.sort(x=distance_sort, axis=1), paddle.argsort(
        x=distance_sort, axis=1
    )
    distance_sort = distance_sort[:, :max_num_neighbors_threshold]
    index_sort = index_sort[:, :max_num_neighbors_threshold]
    index_sort = index_sort + index_neighbor_offset.view(-1, 1).expand(
        shape=[-1, max_num_neighbors_threshold]
    )
    mask_finite = paddle.isfinite(x=distance_sort)
    index_sort = paddle.masked_select(x=index_sort, mask=mask_finite)
    mask_num_neighbors = paddle.zeros(shape=len(index), dtype=bool)
    mask_num_neighbors.index_fill_(axis=0, index=index_sort, value=True)
    return mask_num_neighbors, num_neighbors_image


def radius_graph_pbc_ocp(
    pos: paddle.Tensor,
    pbc: paddle.Tensor | None,
    natoms: paddle.Tensor,
    cell: paddle.Tensor,
    radius: float,
    max_num_neighbors_threshold: int,
    max_cell_images_per_dim: int = sys.maxsize,
) -> tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor]:
    """Function computing the graph in periodic boundary conditions on a (batched) set
    of positions and cells.

    This function is copied from
    https://github.com/Open-Catalyst-Project/ocp/blob/main/ocpmodels/common/utils.py,
    commit 480eb9279ec4a5885981f1ee588c99dcb38838b5

    Args:
        pos (LongTensor): Atomic positions in cartesian coordinates
            :obj:`[n, 3]`
        pbc (BoolTensor): indicates periodic boundary conditions per structure.
            :obj:`[n_structures, 3]`
        natoms (IntTensor): number of atoms per structure. Has shape
            :obj:`[n_structures]`
        cell (Tensor): atomic cell. Has shape
            :obj:`[n_structures, 3, 3]`
        radius (float): cutoff radius distance
        max_num_neighbors_threshold (int): Maximum number of neighbours to consider.

    Returns:
        edge_index (IntTensor): index of atoms in edges. Has shape
            :obj:`[n_edges, 2]`
        cell_offsets (IntTensor): cell displacement w.r.t. their original position of
            atoms in edges. Has shape
            :obj:`[n_edges, 3, 3]`
        num_neighbors_image (IntTensor): Number of neighbours per cell image.
            :obj:`[n_structures]`
        offsets (LongTensor): cartesian displacement w.r.t. their original position of
            atoms in edges. Has shape
            :obj:`[n_edges, 3, 3]`
        atom_distance (LongTensor): edge length. Has shape
            :obj:`[n_edges]`
    """
    batch_size = len(natoms)
    pbc_ = [False, False, False]
    if pbc is not None:
        pbc = paddle.atleast_2d(pbc)
        for i in range(3):
            if not paddle.any(x=pbc[:, i]).item():
                pbc_[i] = False
            elif paddle.all(x=pbc[:, i]).item():
                pbc_[i] = True
            else:
                raise RuntimeError(
                    "Different structures in the batch have different PBC "
                    "configurations. This is not currently supported."
                )
    natoms_squared = (natoms**2).astype(dtype="int64")
    index_offset = paddle.cumsum(x=natoms, axis=0) - natoms
    index_offset_expand = paddle.repeat_interleave(
        x=index_offset, repeats=natoms_squared
    )
    natoms_expand = paddle.repeat_interleave(x=natoms, repeats=natoms_squared)
    num_atom_pairs = paddle.sum(x=natoms_squared)
    index_squared_offset = paddle.cumsum(x=natoms_squared, axis=0) - natoms_squared
    index_squared_offset = paddle.repeat_interleave(
        x=index_squared_offset, repeats=natoms_squared
    )
    atom_count_squared = paddle.arange(end=num_atom_pairs) - index_squared_offset

    index1_tmp = paddle.divide(x=atom_count_squared, y=natoms_expand)
    index1 = paddle.floor(index1_tmp).astype("int64") + index_offset_expand
    index2 = atom_count_squared % natoms_expand + index_offset_expand
    pos1 = paddle.index_select(x=pos, axis=0, index=index1)
    pos2 = paddle.index_select(x=pos, axis=0, index=index2)
    cross_a2a3 = paddle.cross(x=cell[:, 1], y=cell[:, 2], axis=-1)
    cell_vol = paddle.sum(x=cell[:, 0] * cross_a2a3, axis=-1, keepdim=True)
    if pbc_[0]:
        inv_min_dist_a1 = paddle.linalg.norm(x=cross_a2a3 / cell_vol, p=2, axis=-1)
        rep_a1 = paddle.ceil(x=radius * inv_min_dist_a1)
    else:
        rep_a1 = paddle.zeros(shape=[1], dtype=cell.dtype)
    if pbc_[1]:
        cross_a3a1 = paddle.cross(x=cell[:, 2], y=cell[:, 0], axis=-1)
        inv_min_dist_a2 = paddle.linalg.norm(x=cross_a3a1 / cell_vol, p=2, axis=-1)
        rep_a2 = paddle.ceil(x=radius * inv_min_dist_a2)
    else:
        rep_a2 = paddle.zeros(shape=[1], dtype=cell.dtype)
    if pbc_[2]:
        cross_a1a2 = paddle.cross(x=cell[:, 0], y=cell[:, 1], axis=-1)
        inv_min_dist_a3 = paddle.linalg.norm(x=cross_a1a2 / cell_vol, p=2, axis=-1)
        rep_a3 = paddle.ceil(x=radius * inv_min_dist_a3)
    else:
        rep_a3 = paddle.zeros(shape=[1], dtype=cell.dtype)
    max_rep = [
        min(int(rep_a1.max()), max_cell_images_per_dim),
        min(int(rep_a2.max()), max_cell_images_per_dim),
        min(int(rep_a3.max()), max_cell_images_per_dim),
    ]
    cells_per_dim = [
        paddle.arange(start=-rep, end=rep + 1, dtype="float32") for rep in max_rep
    ]  # noqa
    cell_offsets = paddle.cartesian_prod(x=cells_per_dim)
    num_cells = len(cell_offsets)
    cell_offsets_per_atom = cell_offsets.view(1, num_cells, 3).tile(
        repeat_times=[len(index2), 1, 1]
    )
    cell_offsets = paddle.transpose(
        x=cell_offsets, perm=dim2perm(cell_offsets.ndim, 0, 1)
    )  # noqa
    cell_offsets_batch = cell_offsets.view(1, 3, num_cells).expand(
        shape=[batch_size, -1, -1]
    )  # noqa
    data_cell = paddle.transpose(x=cell, perm=dim2perm(cell.ndim, 1, 2))
    pbc_offsets = paddle.bmm(x=data_cell, y=cell_offsets_batch)
    pbc_offsets_per_atom = paddle.repeat_interleave(
        x=pbc_offsets, repeats=natoms_squared, axis=0
    )  # noqa
    pos1 = pos1.view(-1, 3, 1).expand(shape=[-1, -1, num_cells])
    pos2 = pos2.view(-1, 3, 1).expand(shape=[-1, -1, num_cells])
    index1 = index1.view(-1, 1).tile(repeat_times=[1, num_cells]).view(-1)
    index2 = index2.view(-1, 1).tile(repeat_times=[1, num_cells]).view(-1)
    pos2 = pos2 + pbc_offsets_per_atom
    atom_distance_squared = paddle.sum(x=(pos1 - pos2) ** 2, axis=1)
    atom_distance_squared = atom_distance_squared.view(-1)
    mask_within_radius = paddle.less_equal(
        x=atom_distance_squared, y=paddle.to_tensor(radius * radius)
    )
    mask_not_same = paddle.greater_than(
        x=atom_distance_squared, y=paddle.to_tensor(0.0001)
    )  # noqa
    mask = paddle.logical_and(x=mask_within_radius, y=mask_not_same)
    index1 = paddle.masked_select(x=index1, mask=mask)
    index2 = paddle.masked_select(x=index2, mask=mask)
    cell_offsets = paddle.masked_select(
        x=cell_offsets_per_atom.view(-1, 3),
        mask=mask.view(-1, 1).expand(shape=[-1, 3]),
    )
    cell_offsets = cell_offsets.view(-1, 3)
    atom_distance_squared = paddle.masked_select(x=atom_distance_squared, mask=mask)
    mask_num_neighbors, num_neighbors_image = get_max_neighbors_mask(
        natoms=natoms,
        index=index1,
        atom_distance_squared=atom_distance_squared,
        max_num_neighbors_threshold=max_num_neighbors_threshold,
    )
    if not paddle.all(x=mask_num_neighbors):
        index1 = paddle.masked_select(x=index1, mask=mask_num_neighbors)
        index2 = paddle.masked_select(x=index2, mask=mask_num_neighbors)
        atom_distance_squared = paddle.masked_select(
            x=atom_distance_squared, mask=mask_num_neighbors
        )
        cell_offsets = paddle.masked_select(
            x=cell_offsets.view(-1, 3),
            mask=mask_num_neighbors.view(-1, 1).expand(shape=[-1, 3]),
        )
        cell_offsets = cell_offsets.view(-1, 3)
    edge_index = paddle.stack(x=(index2, index1))
    cell_repeated = paddle.repeat_interleave(
        x=cell, repeats=num_neighbors_image, axis=0
    )  # noqa
    offsets = (
        -cell_offsets.astype(dtype="float32")
        .view(-1, 1, 3)
        .bmm(y=cell_repeated.astype(dtype="float32"))
        .view(-1, 3)
    )
    return (
        edge_index,
        cell_offsets,
        num_neighbors_image,
        offsets,
        paddle.sqrt(x=atom_distance_squared),
    )


def get_pbc_distances(
    coords: paddle.Tensor,
    edge_index: paddle.Tensor,
    lattice: paddle.Tensor,
    to_jimages: paddle.Tensor,
    num_atoms: paddle.Tensor,
    num_bonds: paddle.Tensor,
    coord_is_cart: bool = False,
    return_offsets: bool = False,
    return_distance_vec: bool = False,
) -> paddle.Tensor:
    if coord_is_cart:
        pos = coords
    else:
        lattice_nodes = paddle.repeat_interleave(x=lattice, repeats=num_atoms, axis=0)
        pos = paddle.bmm(coords.unsqueeze(1), lattice_nodes).squeeze(1)
    j_index, i_index = edge_index
    distance_vectors = pos[j_index] - pos[i_index]
    lattice_edges = paddle.repeat_interleave(x=lattice, repeats=num_bonds, axis=0)
    offsets = paddle.bmm(
        to_jimages.astype(dtype="float32").unsqueeze(1),
        lattice_edges,
    ).squeeze(1)
    distance_vectors += offsets
    distances = distance_vectors.norm(axis=-1)
    out = {"edge_index": edge_index, "distances": distances}
    if return_distance_vec:
        out["distance_vec"] = distance_vectors
    if return_offsets:
        out["offsets"] = offsets
    return out


def radius_graph_pbc(
    cart_coords: paddle.Tensor,
    lattice: paddle.Tensor,
    num_atoms: paddle.Tensor,
    radius: float,
    max_num_neighbors_threshold: int,
    max_cell_images_per_dim: int = 10,
    topk_per_pair: (paddle.Tensor | None) = None,
) -> tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]:
    """Computes pbc graph edges under pbc.

    topk_per_pair: (num_atom_pairs,), select topk edges per atom pair

    Note: topk should take into account self-self edge for (i, i)

        Keyword arguments
        -----------------
        cart_cords.shape=[Ntotal, 3] -- concatenate all atoms over all crystals
        lattice.shape=[Ncrystal, 3, 3]
        num_atoms.shape=[Ncrystal]
        max_cell_images_per_dim -- constrain the max. number of cell images per
                                dimension in event that infinitesimal angles between
                                lattice vectors are encountered.

    WARNING: It is possible (and has been observed) that for rare cases when periodic
    atom images are on or close to the cut off radius boundary, doing these operations
    in 32 bit floating point can lead to atoms being spuriously considered within or
    outside of the cut off radius. This can lead to invariance of the neighbour list
    under global translation of all atoms in the unit cell. For the rare cases where
    this was observed, switching to 64 bit precision solved the issue. Since all graph
    embeddings should taper messages from neighbours to zero at the cut off radius,
    the effect of these errors in 32-bit should be negligible in practice.
    """
    assert topk_per_pair is None, "non None values of topk_per_pair is not supported"
    edge_index, unit_cell, num_neighbors_image, _, _ = radius_graph_pbc_ocp(
        pos=cart_coords,
        cell=lattice,
        natoms=num_atoms,
        pbc=paddle.to_tensor(data=[True, True, True], dtype="float32")
        .to("bool")
        .to(cart_coords.place),
        radius=radius,
        max_num_neighbors_threshold=max_num_neighbors_threshold,
        max_cell_images_per_dim=max_cell_images_per_dim,
    )
    return edge_index, unit_cell, num_neighbors_image


def edge_score_to_lattice_score_frac_symmetric(
    score_d: paddle.Tensor,
    edge_index: paddle.Tensor,
    edge_vectors: paddle.Tensor,
    batch: paddle.Tensor,
) -> paddle.Tensor:
    """Converts a score per edge into a score for the atom coordinates and/or the
    lattice matrix via the chain rule. This method explicitly takes into account the
    fact that the cartesian coordinates depend on the lattice via the fractional
    coordinates. Moreover, we make sure to get a symmetric update:
    D_cart_norm @ Phi @ D_cart_norm^T, where Phi is a |E| x |E| diagonal matrix with
    the predicted edge scores

    Args:
        score_d (paddle.Tensor, [num_edges,]): A score per edge in the graph.
        edge_index (paddle.Tensor, [2, num_edges]): The edge indices in the graph.
        edge_vectors (paddle.Tensor, [num_edges, 3]): The vectors connecting the source
            of each edge to the target.
        lattice_matrix (paddle.Tensor, [num_nodes, 3, 3]): The lattice matrices for
            each crystal in num_nodes.
        batch (paddle.Tensor, [num_nodes,]): The pointer indicating for each atom which
            molecule in the batch it belongs to.

    Returns:
        paddle.Tensor: The predicted lattice score.
    """
    batch_edge = batch[edge_index[0]]
    unit_edge_vectors_cart = edge_vectors / edge_vectors.norm(axis=-1, keepdim=True)
    score_lattice = scatter(
        score_d[:, None, None]
        * (unit_edge_vectors_cart[:, :, None] @ unit_edge_vectors_cart[:, None, :]),
        batch_edge,
        dim=0,
        dim_size=batch.max() + 1,
        reduce="add",
    )
    score_lattice = score_lattice.transpose([0, -1, -2])
    return score_lattice


class AtomEmbedding(paddle.nn.Layer):
    """Atom Embedding Layer.
    This layer embeds the atomic number of each atom into a vector of size `emb_size`.
    The atomic number is assumed to be in the range [1, `MAX_ATOMIC_NUM`].

    Args:
        emb_size (int): Embedding dimension, i.e. the length of the embedding vector.
        with_mask_type (bool, optional): Whether to add an extra mask token. Defaults
            to False.
    """

    def __init__(self, emb_size: int, with_mask_type=False):
        super().__init__()
        self.emb_size = emb_size
        self.embeddings = paddle.nn.Embedding(
            num_embeddings=MAX_ATOMIC_NUM + int(with_mask_type), embedding_dim=emb_size
        )
        init_Uniform = paddle.nn.initializer.Uniform(low=-np.sqrt(3), high=np.sqrt(3))
        init_Uniform(self.embeddings.weight)

    def forward(self, Z):
        h = self.embeddings(Z - 1)
        return h


class AutomaticFit:
    """
    All added variables are processed in the order of creation.
    """

    activeVar = None
    queue = None
    fitting_mode = False

    def __init__(self, variable, scale_file, name):
        self.variable = variable
        self.scale_file = scale_file
        self._name = name
        self._fitted = False
        self.load_maybe()
        if AutomaticFit.fitting_mode and not self._fitted:
            if AutomaticFit.activeVar is None:
                AutomaticFit.activeVar = self
                AutomaticFit.queue = []
            else:
                self._add2queue()

    @classmethod
    def reset(self):
        AutomaticFit.activeVar = None
        AutomaticFit.all_processed = False

    @classmethod
    def fitting_completed(self):
        return AutomaticFit.queue is None

    @classmethod
    def set2fitmode(self):
        AutomaticFit.reset()
        AutomaticFit.fitting_mode = True

    def _add2queue(self):
        logger.debug(f"Add {self._name} to queue.")
        for var in AutomaticFit.queue:
            if self._name == var._name:
                raise ValueError(
                    f"Variable with the same name ({self._name}) was already added to "
                    "queue!"
                )
        AutomaticFit.queue += [self]

    def set_next_active(self):
        """
        Set the next variable in the queue that should be fitted.
        """
        queue = AutomaticFit.queue
        if len(queue) == 0:
            logger.debug("Processed all variables.")
            AutomaticFit.queue = None
            AutomaticFit.activeVar = None
            return
        AutomaticFit.activeVar = queue.pop(0)

    def load_maybe(self):
        """
        Load variable from file or set to initial value of the variable.
        """
        value = read_value_json(self.scale_file, self._name)
        if value is None:
            logger.debug(
                f"Initialize variable {self._name}' to {self.variable.numpy():.3f}"
            )
        else:
            self._fitted = True
            logger.debug(f"Set scale factor {self._name} : {value}")
            with paddle.no_grad():
                paddle.assign(paddle.to_tensor(data=value), output=self.variable)


class AutoScaleFit(AutomaticFit):
    """
    Class to automatically fit the scaling factors depending on the observed variances.

    Parameters
    ----------
        variable: paddle.Tensor
            Variable to fit.
        scale_file: str
            Path to the json file where to store/load from the scaling factors.
    """

    def __init__(self, variable, scale_file, name):
        super().__init__(variable, scale_file, name)
        if not self._fitted:
            self._init_stats()

    def _init_stats(self):
        self.variance_in = 0
        self.variance_out = 0
        self.nSamples = 0

    @paddle.no_grad()
    def observe(self, x, y):
        """
        Observe variances for input x and output y.
        The scaling factor alpha is calculated s.t. Var(alpha * y) ~ Var(x)
        """
        if self._fitted:
            return
        if AutomaticFit.activeVar == self:
            nSamples = tuple(y.shape)[0]
            self.variance_in += (
                paddle.mean(x=paddle.var(x=x, axis=0)).to(dtype="float32") * nSamples
            )
            self.variance_out += (
                paddle.mean(x=paddle.var(x=y, axis=0)).to(dtype="float32") * nSamples
            )
            self.nSamples += nSamples

    @paddle.no_grad()
    def fit(self):
        """
        Fit the scaling factor based on the observed variances.
        """
        if AutomaticFit.activeVar == self:
            if self.variance_in == 0:
                raise ValueError(
                    f"Did not track the variable {self._name}. Add observe calls to "
                    "track the variance before and after."
                )
            self.variance_in = self.variance_in / self.nSamples
            self.variance_out = self.variance_out / self.nSamples
            ratio = self.variance_out / self.variance_in
            value = paddle.sqrt(x=1 / ratio)
            logger.info(
                f"Variable: {self._name}, Var_in: {self.variance_in.item():.3f}, "
                f"Var_out: {self.variance_out.item():.3f}, Ratio: {ratio:.3f} => "
                f"Scaling factor: {value:.3f}"
            )
            paddle.assign(self.variable * value, output=self.variable)
            update_json(self.scale_file, {self._name: float(self.variable.item())})
            self.set_next_active()


class ScalingFactor(paddle.nn.Layer):
    """
    Scale the output y of the layer s.t. the (mean) variance wrt. to the reference
    input x_ref is preserved.

    Parameters
    ----------
        scale_file: str
            Path to the json file where to store/load from the scaling factors.
        name: str
            Name of the scaling factor
    """

    def __init__(self, scale_file, name, device=None):
        super().__init__()
        self.scale_factor = paddle.base.framework.EagerParamBase.from_tensor(
            tensor=paddle.to_tensor(data=1.0, place=device), trainable=False
        )
        self.autofit = AutoScaleFit(self.scale_factor, scale_file, name)

    def forward(self, x_ref, y):
        y = y * self.scale_factor
        self.autofit.observe(x_ref, y)
        return y


class Dense(paddle.nn.Layer):
    """Combines dense layer with scaling for swish activation.

    Args:
        in_features (int): Input dimension for the linear layer.
        out_features (int): Output dimension for the linear layer.
        bias (bool, optional): Whether to add a bias term. Defaults to False.
        activation (Optional[str], optional): Name of the activation function, support
            'swish', 'silu', 'siqu. If None, no activation will be applied. Defaults to
            None.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        activation: Optional[str] = None,
    ):
        super().__init__()
        self.linear = paddle.nn.Linear(
            in_features=in_features, out_features=out_features, bias_attr=bias
        )
        self.reset_parameters()
        if isinstance(activation, str):
            activation = activation.lower()
        if activation in ["swish", "silu"]:
            self._activation = ScaledSiLU()
        elif activation == "siqu":
            self._activation = SiQU()
        elif activation is None:
            self._activation = paddle.nn.Identity()
        else:
            raise NotImplementedError(
                "Activation function not implemented for GemNet (yet)."
            )

    def reset_parameters(self, initializer: Callable = he_orthogonal_init):
        initializer(self.linear.weight)
        if self.linear.bias is not None:
            self.linear.bias.data.fill_(value=0)

    def forward(self, x: paddle.Tensor):
        x = self.linear(x)
        x = self._activation(x)
        return x


class EdgeEmbedding(paddle.nn.Layer):
    """Edge embedding based on the concatenation of atom embeddings and subsequent dense
    layer.

    Args:
        atom_features (int): Atom embedding size.
        edge_features (int): Edge embedding size.
        out_features (int): Output embedding size.
        activation (str, optional): Name of the activation function. Defaults to None.
    """

    def __init__(
        self,
        atom_features: int,
        edge_features: int,
        out_features: int,
        activation: Optional[str] = None,
    ):
        super().__init__()
        in_features = 2 * atom_features + edge_features
        self.dense = Dense(in_features, out_features, activation=activation, bias=False)

    def forward(self, h, m_rbf, idx_s, idx_t):
        h_s = h[idx_s]
        h_t = h[idx_t]
        m_st = paddle.concat(x=[h_s, h_t, m_rbf], axis=-1)
        m_st = self.dense(m_st)
        return m_st


class EfficientInteractionDownProjection(paddle.nn.Layer):
    """Down projection in the efficient reformulation."""

    def __init__(self, num_spherical: int, num_radial: int, emb_size_interm: int):
        super().__init__()
        self.num_spherical = num_spherical
        self.num_radial = num_radial
        self.emb_size_interm = emb_size_interm
        self.reset_parameters()

    def reset_parameters(self):
        self.weight = paddle.base.framework.EagerParamBase.from_tensor(
            tensor=paddle.empty(
                shape=(self.num_spherical, self.num_radial, self.emb_size_interm)
            ),
            trainable=True,
        )
        he_orthogonal_init(self.weight)

    def forward(self, rbf, sph, id_ca, id_ragged_idx):
        num_edges = tuple(rbf.shape)[1]
        rbf_W1 = paddle.matmul(x=rbf, y=self.weight)
        rbf_W1 = rbf_W1.transpose(perm=[1, 2, 0])
        if tuple(sph.shape)[0] == 0:
            Kmax = 0
        else:
            Kmax = max(
                paddle.max(x=id_ragged_idx + 1),
                paddle.to_tensor(data=0).to(id_ragged_idx.place),
            )
        sph2 = paddle.zeros(
            shape=[num_edges, Kmax, self.num_spherical], dtype=sph.dtype
        )
        sph2[id_ca, id_ragged_idx] = sph
        sph2 = paddle.transpose(x=sph2, perm=dim2perm(sph2.ndim, 1, 2))
        return rbf_W1, sph2


class InteractionBlockTripletsOnly(paddle.nn.Layer):
    """Interaction block for GemNet-T/dT."""

    def __init__(
        self,
        emb_size_atom,
        emb_size_edge,
        emb_size_trip,
        emb_size_rbf,
        emb_size_cbf,
        emb_size_bil_trip,
        num_before_skip,
        num_after_skip,
        num_concat,
        num_atom,
        activation=None,
        scale_file=None,
        name="Interaction",
    ):
        super().__init__()
        self.name = name
        self.skip_connection_factor = 2.0**-0.5
        block_nr = name.split("_")[-1]
        self.dense_ca = Dense(
            emb_size_edge, emb_size_edge, activation=activation, bias=False
        )
        self.trip_interaction = TripletInteraction(
            emb_size_edge=emb_size_edge,
            emb_size_trip=emb_size_trip,
            emb_size_bilinear=emb_size_bil_trip,
            emb_size_rbf=emb_size_rbf,
            emb_size_cbf=emb_size_cbf,
            activation=activation,
            scale_file=scale_file,
            name=f"TripInteraction_{block_nr}",
        )
        self.layers_before_skip = paddle.nn.LayerList(
            sublayers=[
                ResidualLayer(emb_size_edge, activation=activation)
                for i in range(num_before_skip)
            ]
        )
        self.layers_after_skip = paddle.nn.LayerList(
            sublayers=[
                ResidualLayer(emb_size_edge, activation=activation)
                for i in range(num_after_skip)
            ]
        )
        self.atom_update = AtomUpdateBlock(
            emb_size_atom=emb_size_atom,
            emb_size_edge=emb_size_edge,
            emb_size_rbf=emb_size_rbf,
            nHidden=num_atom,
            activation=activation,
            scale_file=scale_file,
            name=f"AtomUpdate_{block_nr}",
        )
        self.concat_layer = EdgeEmbedding(
            emb_size_atom, emb_size_edge, emb_size_edge, activation=activation
        )
        self.residual_m = paddle.nn.LayerList(
            sublayers=[
                ResidualLayer(emb_size_edge, activation=activation)
                for _ in range(num_concat)
            ]
        )
        self.inv_sqrt_2 = 1 / math.sqrt(2.0)

    def forward(
        self,
        h,
        m,
        rbf3,
        cbf3,
        id3_ragged_idx,
        id_swap,
        id3_ba,
        id3_ca,
        rbf_h,
        idx_s,
        idx_t,
    ):
        x_ca_skip = self.dense_ca(m)
        x3 = self.trip_interaction(
            m, rbf3, cbf3, id3_ragged_idx, id_swap, id3_ba, id3_ca
        )
        x = x_ca_skip + x3
        x = x * self.inv_sqrt_2
        for i, layer in enumerate(self.layers_before_skip):
            x = layer(x)
        m = m + x
        m = m * self.inv_sqrt_2
        for i, layer in enumerate(self.layers_after_skip):
            m = layer(m)
        h2 = self.atom_update(h, m, rbf_h, idx_t)
        h = h + h2
        h = h * self.skip_connection_factor
        m2 = self.concat_layer(h, m, idx_s, idx_t)
        for i, layer in enumerate(self.residual_m):
            m2 = layer(m2)
        m = m + m2
        m = m * self.inv_sqrt_2
        return h, m


class EfficientInteractionBilinear(paddle.nn.Layer):
    """
    Efficient reformulation of the bilinear layer and subsequent summation.
    """

    def __init__(self, emb_size: int, emb_size_interm: int, units_out: int):
        super().__init__()
        self.emb_size = emb_size
        self.emb_size_interm = emb_size_interm
        self.units_out = units_out
        self.reset_parameters()

    def reset_parameters(self):
        out_0 = paddle.empty(
            shape=(self.emb_size, self.emb_size_interm, self.units_out)
        )
        out_0.stop_gradient = not True
        self.weight = paddle.base.framework.EagerParamBase.from_tensor(tensor=out_0)
        he_orthogonal_init(self.weight)

    def forward(self, basis, m, id_reduce, id_ragged_idx):
        rbf_W1, sph = basis
        nEdges = tuple(rbf_W1.shape)[0]
        if nEdges == 0:
            logger.warning(f"Zero graph edges found in {self.__class__}")
            return paddle.zeros(shape=(0, 0))
        Kmax = max(
            paddle.max(x=id_ragged_idx) + 1,
            paddle.to_tensor(data=0).to(id_ragged_idx.place),
        )
        m2 = paddle.zeros(shape=[nEdges, Kmax, self.emb_size], dtype=m.dtype)
        m2[id_reduce, id_ragged_idx] = m
        sum_k = paddle.matmul(x=sph, y=m2)
        rbf_W1_sum_k = paddle.matmul(x=rbf_W1, y=sum_k)
        m_ca = paddle.matmul(x=rbf_W1_sum_k.transpose(perm=[2, 0, 1]), y=self.weight)
        m_ca = paddle.sum(x=m_ca, axis=0)
        return m_ca


class TripletInteraction(paddle.nn.Layer):
    """
    Triplet-based message passing block.
    """

    def __init__(
        self,
        emb_size_edge,
        emb_size_trip,
        emb_size_bilinear,
        emb_size_rbf,
        emb_size_cbf,
        activation=None,
        scale_file=None,
        name="TripletInteraction",
        **kwargs,
    ):
        super().__init__()
        self.name = name
        self.dense_ba = Dense(
            emb_size_edge, emb_size_edge, activation=activation, bias=False
        )
        self.mlp_rbf = Dense(emb_size_rbf, emb_size_edge, activation=None, bias=False)
        self.scale_rbf = ScalingFactor(scale_file=scale_file, name=name + "_had_rbf")
        self.mlp_cbf = EfficientInteractionBilinear(
            emb_size_trip, emb_size_cbf, emb_size_bilinear
        )
        self.scale_cbf_sum = ScalingFactor(
            scale_file=scale_file, name=name + "_sum_cbf"
        )
        self.down_projection = Dense(
            emb_size_edge, emb_size_trip, activation=activation, bias=False
        )
        self.up_projection_ca = Dense(
            emb_size_bilinear, emb_size_edge, activation=activation, bias=False
        )
        self.up_projection_ac = Dense(
            emb_size_bilinear, emb_size_edge, activation=activation, bias=False
        )
        self.inv_sqrt_2 = 1 / math.sqrt(2.0)

    def forward(self, m, rbf3, cbf3, id3_ragged_idx, id_swap, id3_ba, id3_ca):
        """
        Returns
        -------
            m: paddle.Tensor, shape=(nEdges, emb_size_edge)
                Edge embeddings (c->a).
        """
        x_ba = self.dense_ba(m)
        rbf_emb = self.mlp_rbf(rbf3)
        x_ba2 = x_ba * rbf_emb
        x_ba = self.scale_rbf(x_ba, x_ba2)
        x_ba = self.down_projection(x_ba)
        x_ba = x_ba[id3_ba]
        x = self.mlp_cbf(cbf3, x_ba, id3_ca, id3_ragged_idx)
        x = self.scale_cbf_sum(x_ba, x)
        x_ca = self.up_projection_ca(x)
        x_ac = self.up_projection_ac(x)
        x_ac = x_ac[id_swap]
        x3 = x_ca + x_ac
        x3 = x3 * self.inv_sqrt_2
        return x3


class ResidualLayer(paddle.nn.Layer):
    """
    Residual block with output scaled by 1/sqrt(2).
    """

    def __init__(
        self, units: int, nLayers: int = 2, layer: Callable = Dense, **layer_kwargs
    ):
        super().__init__()
        self.dense_mlp = paddle.nn.Sequential(
            *[
                layer(in_features=units, out_features=units, bias=False, **layer_kwargs)
                for _ in range(nLayers)
            ]
        )
        self.inv_sqrt_2 = 1 / math.sqrt(2)

    def forward(self, input: paddle.Tensor):
        x = self.dense_mlp(input)
        x = input + x
        x = x * self.inv_sqrt_2
        return x


class AtomUpdateBlock(paddle.nn.Layer):
    """
    Aggregate the message embeddings of the atoms
    """

    def __init__(
        self,
        emb_size_atom: int,
        emb_size_edge: int,
        emb_size_rbf: int,
        nHidden: int,
        activation=None,
        scale_file=None,
        name: str = "atom_update",
    ):
        super().__init__()
        self.name = name
        self.dense_rbf = Dense(emb_size_rbf, emb_size_edge, activation=None, bias=False)
        self.scale_sum = ScalingFactor(scale_file=scale_file, name=name + "_sum")
        self.layers = self.get_mlp(emb_size_edge, emb_size_atom, nHidden, activation)

    def get_mlp(
        self, units_in: int, units: int, nHidden: int, activation: str
    ) -> paddle.nn.LayerList:
        dense1 = Dense(units_in, units, activation=activation, bias=False)
        mlp = [dense1]
        res = [
            ResidualLayer(units, nLayers=2, activation=activation)
            for i in range(nHidden)
        ]
        mlp += res
        return paddle.nn.LayerList(sublayers=mlp)

    def forward(
        self,
        h: paddle.Tensor,
        m: paddle.Tensor,
        rbf: paddle.Tensor,
        id_j: paddle.Tensor,
    ) -> paddle.Tensor:
        nAtoms = tuple(h.shape)[0]
        mlp_rbf = self.dense_rbf(rbf)
        x = m * mlp_rbf
        x2 = scatter(x, id_j, dim=0, dim_size=nAtoms, reduce="sum")
        x = self.scale_sum(m, x2)
        for layer in self.layers:
            x = layer(x)
        return x


class OutputBlock(AtomUpdateBlock):
    """
    Combines the atom update block and subsequent final dense layer.
    """

    def __init__(
        self,
        emb_size_atom: int,
        emb_size_edge: int,
        emb_size_rbf: int,
        nHidden: int,
        num_targets: int,
        activation=None,
        direct_forces=True,
        output_init="HeOrthogonal",
        scale_file=None,
        name: str = "output",
        **kwargs,
    ):
        super().__init__(
            name=name,
            emb_size_atom=emb_size_atom,
            emb_size_edge=emb_size_edge,
            emb_size_rbf=emb_size_rbf,
            nHidden=nHidden,
            activation=activation,
            scale_file=scale_file,
        )
        assert isinstance(output_init, str)
        self.output_init = output_init.lower()
        self.direct_forces = direct_forces
        self.seq_energy = self.layers
        self.out_energy = Dense(emb_size_atom, num_targets, bias=False, activation=None)
        if self.direct_forces:
            self.scale_rbf_F = ScalingFactor(scale_file=scale_file, name=name + "_had")
            self.seq_forces = self.get_mlp(
                emb_size_edge, emb_size_edge, nHidden, activation
            )
            self.out_forces = Dense(
                emb_size_edge, num_targets, bias=False, activation=None
            )
            self.dense_rbf_F = Dense(
                emb_size_rbf, emb_size_edge, activation=None, bias=False
            )
        self.reset_parameters()

    def reset_parameters(self):
        if self.output_init == "heorthogonal":
            self.out_energy.reset_parameters(he_orthogonal_init)
            if self.direct_forces:
                self.out_forces.reset_parameters(he_orthogonal_init)
        elif self.output_init == "zeros":
            self.out_energy.reset_parameters(paddle.nn.initializer.Constant)
            if self.direct_forces:
                self.out_forces.reset_parameters(paddle.nn.initializer.Constant)
        else:
            raise UserWarning(f"Unknown output_init: {self.output_init}")

    # def forward(
    #     self,
    #     h: paddle.Tensor,
    #     m: paddle.Tensor,
    #     rbf: paddle.Tensor,
    #     id_j: paddle.Tensor,
    # ) -> Tuple[paddle.Tensor, paddle.Tensor]:
    # nAtoms = tuple(h.shape)[0]
    # rbf_emb_E = self.dense_rbf(rbf)
    # x = m * rbf_emb_E
    # x_E = scatter(x, id_j, dim=0, dim_size=nAtoms, reduce="sum")
    # x_E = self.scale_sum(m, x_E)
    # for layer in self.seq_energy:
    #     x_E = layer(x_E)
    # x_E = self.out_energy(x_E)
    # if self.direct_forces:
    #     x_F = m
    #     for i, layer in enumerate(self.seq_forces):
    #         x_F = layer(x_F)
    #     rbf_emb_F = self.dense_rbf_F(rbf)
    #     x_F_rbf = x_F * rbf_emb_F
    #     x_F = self.scale_rbf_F(x_F, x_F_rbf)
    #     x_F = self.out_forces(x_F)
    # else:
    #     x_F = 0
    # return 0, x_F

    def forward(
        self,
        h: paddle.Tensor,
        m: paddle.Tensor,
        rbf: paddle.Tensor,
        id_j: paddle.Tensor,
    ) -> Tuple[paddle.Tensor, paddle.Tensor]:
        if self.direct_forces:
            x_F = m
            for i, layer in enumerate(self.seq_forces):
                x_F = layer(x_F)
            rbf_emb_F = self.dense_rbf_F(rbf)
            x_F_rbf = x_F * rbf_emb_F
            x_F = self.scale_rbf_F(x_F, x_F_rbf)
            x_F = self.out_forces(x_F)
        else:
            x_F = 0
        return 0, x_F


class RBFBasedLatticeUpdateBlock(paddle.nn.Layer):
    def __init__(
        self,
        emb_size: int,
        activation: str,
        emb_size_rbf: int,
        emb_size_edge: int,
        num_heads: int = 1,
    ):
        super().__init__()
        self.num_out = num_heads
        self.mlp = paddle.nn.Sequential(
            Dense(emb_size, emb_size, activation=activation), Dense(emb_size, emb_size)
        )
        self.dense_rbf_F = Dense(
            emb_size_rbf, emb_size_edge, activation=None, bias=False
        )
        self.out_forces = Dense(emb_size_edge, num_heads, bias=False, activation=None)

    def compute_score_per_edge(
        self, edge_emb: paddle.Tensor, rbf: paddle.Tensor
    ) -> paddle.Tensor:
        x_F = self.mlp(edge_emb)
        rbf_emb_F = self.dense_rbf_F(rbf)
        x_F_rbf = x_F * rbf_emb_F
        x_F = self.out_forces(x_F_rbf)
        return x_F


class RBFBasedLatticeUpdateBlockFrac(RBFBasedLatticeUpdateBlock):
    def __init__(
        self,
        emb_size: int,
        activation: str,
        emb_size_rbf: int,
        emb_size_edge: int,
        num_heads: int = 1,
    ):
        super().__init__(
            emb_size=emb_size,
            activation=activation,
            emb_size_rbf=emb_size_rbf,
            emb_size_edge=emb_size_edge,
            num_heads=num_heads,
        )

    def forward(
        self,
        edge_emb: paddle.Tensor,
        edge_index: paddle.Tensor,
        distance_vec: paddle.Tensor,
        lattice: paddle.Tensor,
        batch: paddle.Tensor,
        rbf: paddle.Tensor,
        normalize_score: bool = True,
    ) -> paddle.Tensor:
        edge_scores = self.compute_score_per_edge(edge_emb=edge_emb, rbf=rbf)
        if normalize_score:
            num_edges = scatter(
                paddle.ones_like(x=distance_vec[:, 0]), batch[edge_index[0]]
            )
            edge_scores /= num_edges[batch[edge_index[0]], None]
        outs = []
        for i in range(self.num_out):
            lattice_update = edge_score_to_lattice_score_frac_symmetric(
                score_d=edge_scores[:, i],
                edge_index=edge_index,
                edge_vectors=distance_vec,
                batch=batch,
            )
            outs.append(lattice_update)
        outs = paddle.stack(x=outs, axis=-1).sum(axis=-1)
        return outs


class GemNetT(paddle.nn.Layer):
    """GemNet-T, triplets-only variant of GemNet. This is a decoder model of MatterGen.

    Args:
        num_targets (int): Number of targets for output. In Gemnet, it means the number
            of the energy and force. Defaults to 1.
        latent_dim (int): The dimension of the latent space.
        atom_embedding_cfg (dict): The configuration of the atom embedding.
        num_spherical (int, optional): Controls maximum frequency of CircularBasisLayer.
            Defaults to 7.
        num_radial (int, optional): Controls maximum frequency of RadialBasis. Defaults
            to 128.
        num_blocks (int, optional): Number of interaction blocks. Defaults to 3.
        emb_size_atom (int, optional): Embedding size of the atoms. Defaults to 512.
        emb_size_edge (int, optional): Embedding size of the edges. Defaults to 512.
        emb_size_trip (int, optional): Embedding size in the triplet message passing
            block. Defaults to 64.
        emb_size_rbf (int, optional): Embedding size of the radial basis transformation.
            Defaults to 16.
        emb_size_cbf (int, optional): Embedding size of the circular basis
            transformation. Defaults to 16.
        emb_size_bil_trip (int, optional): Embedding size of the edge embeddings in the
            triplet-based message passing block after the bilinear layer. Defaults to
            64.
        num_before_skip (int, optional): Number of residual blocks before the first
            skip connection. Defaults to 1.
        num_after_skip (int, optional): Number of residual blocks after the first skip
            connection. Defaults to 2.
        num_concat (int, optional): Number of residual blocks after the concatenation.
            Defaults to 1.
        num_atom (int, optional): Number of residual blocks in the atom embedding
            blocks. Defaults to 3.
        cutoff (float, optional): Embedding cutoff for interactomic directions in
            Angstrom. Defaults to 6.0.
        max_neighbors (int, optional): Maximum neighbors per atom. Defaults to 50.
        rbf (dict, optional): Name and hyperparameters of the radial basis function.
            Defaults to {"name": "gaussian"}.
        envelope (dict, optional): Name and hyperparameters of the envelope function.
            Defaults to {"name": "polynomial", "exponent": 5}.
        cbf (dict, optional): Name and hyperparameters of the cosine basis function.
            Defaults to {"name": "spherical_harmonics"}.
        otf_graph (bool, optional): Whether to use On-The-Fly graph. Defaults to False.
        output_init (str, optional): Initialization method for the final dense layer.
            Defaults to "HeOrthogonal".
        activation (str, optional): Name of the activation function. Defaults to
            "swish".
        max_cell_images_per_dim (int, optional): Maximum cell images per dimension.
            Defaults to 5.
    """

    def __init__(
        self,
        num_targets: int,  # 1
        latent_dim: int,  #  512
        atom_embedding_cfg: dict,  # emb_size=512, with_mask_type=True
        num_spherical: int = 7,
        num_radial: int = 128,
        num_blocks: int = 3,
        emb_size_atom: int = 512,
        emb_size_edge: int = 512,
        emb_size_trip: int = 64,
        emb_size_rbf: int = 16,
        emb_size_cbf: int = 16,
        emb_size_bil_trip: int = 64,
        num_before_skip: int = 1,
        num_after_skip: int = 2,
        num_concat: int = 1,
        num_atom: int = 3,
        cutoff: float = 6.0,
        max_neighbors: int = 50,
        rbf: dict = {"name": "gaussian"},
        envelope: dict = {"name": "polynomial", "exponent": 5},
        cbf: dict = {"name": "spherical_harmonics"},
        otf_graph: bool = False,
        output_init: str = "HeOrthogonal",
        activation: str = "swish",
        max_cell_images_per_dim: int = 5,
        **kwargs,
    ):
        super().__init__()
        # scale_file = "ppmat/models/mattergen/gemnet-dT.json"
        scale_file = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "gemnet-dT.json"
        )
        self.num_targets = num_targets
        assert num_blocks > 0
        self.num_blocks = num_blocks
        atom_embedding = AtomEmbedding(**atom_embedding_cfg)
        emb_dim_atomic_number = getattr(atom_embedding, "emb_size")
        self.cutoff = cutoff
        self.max_neighbors = max_neighbors
        self.max_cell_images_per_dim = max_cell_images_per_dim
        self.otf_graph = otf_graph
        self.angle_edge_emb = paddle.nn.Sequential(
            paddle.nn.Linear(in_features=emb_size_edge + 3, out_features=emb_size_edge),
            paddle.nn.ReLU(),
            paddle.nn.Linear(in_features=emb_size_edge, out_features=emb_size_edge),
        )
        linear_init_(self.angle_edge_emb[0])
        linear_init_(self.angle_edge_emb[2])
        AutomaticFit.reset()
        self.radial_basis = RadialBasis(
            num_radial=num_radial, cutoff=cutoff, rbf=rbf, envelope=envelope
        )
        radial_basis_cbf3 = RadialBasis(
            num_radial=num_radial, cutoff=cutoff, rbf=rbf, envelope=envelope
        )
        self.cbf_basis3 = CircularBasisLayer(
            num_spherical, radial_basis=radial_basis_cbf3, cbf=cbf, efficient=True
        )
        self.lattice_out_blocks = paddle.nn.LayerList(
            sublayers=[
                RBFBasedLatticeUpdateBlockFrac(
                    emb_size_edge, activation, emb_size_rbf, emb_size_edge
                )
                for _ in range(num_blocks + 1)
            ]
        )
        self.mlp_rbf_lattice = Dense(
            num_radial, emb_size_rbf, activation=None, bias=False
        )
        self.mlp_rbf3 = Dense(num_radial, emb_size_rbf, activation=None, bias=False)
        self.mlp_cbf3 = EfficientInteractionDownProjection(
            num_spherical, num_radial, emb_size_cbf
        )
        self.mlp_rbf_h = Dense(num_radial, emb_size_rbf, activation=None, bias=False)
        self.mlp_rbf_out = Dense(num_radial, emb_size_rbf, activation=None, bias=False)
        self.atom_emb = atom_embedding
        self.atom_latent_emb = paddle.nn.Linear(
            in_features=emb_dim_atomic_number + latent_dim, out_features=emb_size_atom
        )
        linear_init_(self.atom_latent_emb)
        self.edge_emb = EdgeEmbedding(
            emb_size_atom, num_radial, emb_size_edge, activation=activation
        )
        out_blocks = []
        int_blocks = []
        interaction_block = InteractionBlockTripletsOnly
        for i in range(num_blocks):
            int_blocks.append(
                interaction_block(
                    emb_size_atom=emb_size_atom,
                    emb_size_edge=emb_size_edge,
                    emb_size_trip=emb_size_trip,
                    emb_size_rbf=emb_size_rbf,
                    emb_size_cbf=emb_size_cbf,
                    emb_size_bil_trip=emb_size_bil_trip,
                    num_before_skip=num_before_skip,
                    num_after_skip=num_after_skip,
                    num_concat=num_concat,
                    num_atom=num_atom,
                    activation=activation,
                    scale_file=scale_file,
                    name=f"IntBlock_{i + 1}",
                )
            )
        for i in range(num_blocks + 1):
            out_blocks.append(
                OutputBlock(
                    emb_size_atom=emb_size_atom,
                    emb_size_edge=emb_size_edge,
                    emb_size_rbf=emb_size_rbf,
                    nHidden=num_atom,
                    num_targets=num_targets,
                    activation=activation,
                    output_init=output_init,
                    direct_forces=True,
                    scale_file=scale_file,
                    name=f"OutBlock_{i}",
                )
            )
        self.out_blocks = paddle.nn.LayerList(sublayers=out_blocks)
        self.int_blocks = paddle.nn.LayerList(sublayers=int_blocks)
        self.shared_parameters = [
            (self.mlp_rbf3, self.num_blocks),
            (self.mlp_cbf3, self.num_blocks),
            (self.mlp_rbf_h, self.num_blocks),
            (self.mlp_rbf_out, self.num_blocks + 1),
        ]

    def get_triplets(
        self, edge_index: paddle.Tensor, num_atoms: int
    ) -> Tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]:
        """
        Get all b->a for each edge c->a.
        It is possible that b=c, as long as the edges are distinct.

        Returns
        -------
        id3_ba: paddle.Tensor, shape (num_triplets,)
            Indices of input edge b->a of each triplet b->a<-c
        id3_ca: paddle.Tensor, shape (num_triplets,)
            Indices of output edge c->a of each triplet b->a<-c
        id3_ragged_idx: paddle.Tensor, shape (num_triplets,)
            Indices enumerating the copies of id3_ca for creating a padded matrix
        """
        idx_s, idx_t = edge_index
        value = paddle.arange(start=1, end=idx_s.shape[0] + 1, dtype=idx_s.dtype)

        def custom_bincount(x, minlength=0):
            unique, counts = paddle.unique(x, return_counts=True)
            max_val = paddle.max(unique).numpy().item() if len(unique) > 0 else -1
            length = (max_val + 1) if (max_val + 1) > minlength else minlength
            result = paddle.zeros([length], dtype="int64")
            if len(unique) > 0:
                result = paddle.scatter_nd(unique.unsqueeze(1), counts, result.shape)
            return result

        n = idx_t.shape[0]
        rows = paddle.arange(n).unsqueeze(1)  # [0,1,2,...,n-1]^T
        cols = paddle.arange(n).unsqueeze(0)  # [0,1,2,...,n-1]
        mask = (idx_t.unsqueeze(1) == idx_t.unsqueeze(0)) & (cols <= rows)
        col = mask.sum(axis=1).astype("int64") - 1
        rows = idx_t
        indices = paddle.stack([rows, col], axis=1)

        shape = [num_atoms.item(), col.max().item() + 1]
        result = paddle.scatter_nd(indices, value, shape)
        mat = result

        id3_ba = mat[idx_t][mat[idx_t] > 0] - 1
        tmp_r = paddle.nonzero(mat[idx_t], as_tuple=False)
        id3_ca = tmp_r[:, 0]

        mask = id3_ba != id3_ca
        id3_ba = id3_ba[mask]
        id3_ca = id3_ca[mask]

        num_triplets = custom_bincount(id3_ca, minlength=idx_s.shape[0])

        id3_ragged_idx = ragged_range(num_triplets)
        return id3_ba, id3_ca, id3_ragged_idx

    def select_symmetric_edges(self, tensor, mask, reorder_idx, inverse_neg):
        tensor_directed = tensor[mask]
        sign = 1 - 2 * inverse_neg
        tensor_cat = paddle.concat(x=[tensor_directed, sign * tensor_directed])
        tensor_ordered = tensor_cat[reorder_idx]
        return tensor_ordered

    def reorder_symmetric_edges(
        self,
        edge_index: paddle.Tensor,
        cell_offsets: paddle.Tensor,
        neighbors: paddle.Tensor,
        edge_dist: paddle.Tensor,
        edge_vector: paddle.Tensor,
    ) -> Tuple[
        paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor
    ]:
        """
        Reorder edges to make finding counter-directional edges easier.

        Some edges are only present in one direction in the data,
        since every atom has a maximum number of neighbors. Since we only use i->j
        edges here, we lose some j->i edges and add others by
        making it symmetric.
        We could fix this by merging edge_index with its counter-edges,
        including the cell_offsets, and then running paddle.unique.
        But this does not seem worth it.
        """
        mask_sep_atoms = edge_index[0] < edge_index[1]
        cell_earlier = (
            (cell_offsets[:, 0] < 0)
            | (cell_offsets[:, 0] == 0) & (cell_offsets[:, 1] < 0)
            | (cell_offsets[:, 0] == 0)
            & (cell_offsets[:, 1] == 0)
            & (cell_offsets[:, 2] < 0)
        )
        mask_same_atoms = edge_index[0] == edge_index[1]
        mask_same_atoms &= cell_earlier
        mask = mask_sep_atoms | mask_same_atoms
        edge_index_new = edge_index[mask[None, :].expand(shape=[2, -1])].view(2, -1)
        edge_index_cat = paddle.concat(
            x=[
                edge_index_new,
                paddle.stack(x=[edge_index_new[1], edge_index_new[0]], axis=0),
            ],
            axis=1,
        )
        batch_edge = paddle.repeat_interleave(
            x=paddle.arange(end=neighbors.shape[0]), repeats=neighbors
        )
        batch_edge = batch_edge[mask]
        neighbors_new = 2 * paddle.bincount(x=batch_edge, minlength=neighbors.shape[0])
        edge_reorder_idx = repeat_blocks(
            neighbors_new // 2,
            repeats=2,
            continuous_indexing=True,
            repeat_inc=edge_index_new.shape[1],
        )
        edge_index_new = edge_index_cat[:, edge_reorder_idx]
        cell_offsets_new = self.select_symmetric_edges(
            cell_offsets, mask, edge_reorder_idx, True
        )
        edge_dist_new = self.select_symmetric_edges(
            edge_dist, mask, edge_reorder_idx, False
        )
        edge_vector_new = self.select_symmetric_edges(
            edge_vector, mask, edge_reorder_idx, True
        )
        return (
            edge_index_new,
            cell_offsets_new,
            neighbors_new,
            edge_dist_new,
            edge_vector_new,
        )

    def generate_interaction_graph(
        self,
        cart_coords: paddle.Tensor,
        lattice: paddle.Tensor,
        num_atoms: paddle.Tensor,
        edge_index: paddle.Tensor = None,
        to_jimages: paddle.Tensor = None,
        num_bonds: paddle.Tensor = None,
    ):
        if self.otf_graph:
            edge_index, to_jimages, num_bonds = radius_graph_pbc(
                cart_coords=cart_coords,
                lattice=lattice,
                num_atoms=num_atoms,
                radius=self.cutoff,
                max_num_neighbors_threshold=self.max_neighbors,
                max_cell_images_per_dim=self.max_cell_images_per_dim,
            )

        out = get_pbc_distances(
            cart_coords,
            edge_index,
            lattice,
            to_jimages,
            num_atoms,
            num_bonds,
            coord_is_cart=True,
            return_offsets=True,
            return_distance_vec=True,
        )
        edge_index = out["edge_index"]
        D_st = out["distances"]
        V_st = -out["distance_vec"] / D_st[:, None]
        edge_index, cell_offsets, neighbors, D_st, V_st = self.reorder_symmetric_edges(
            edge_index, to_jimages, num_bonds, D_st, V_st
        )
        block_sizes = neighbors // 2
        block_sizes = paddle.masked_select(x=block_sizes, mask=block_sizes > 0)

        id_swap = repeat_blocks(
            block_sizes,
            repeats=2,
            continuous_indexing=False,
            start_idx=block_sizes[0],
            block_inc=block_sizes[:-1] + block_sizes[1:],
            repeat_inc=-block_sizes,
        )

        id3_ba, id3_ca, id3_ragged_idx = self.get_triplets(
            edge_index, num_atoms=num_atoms.sum()
        )
        return (
            edge_index,
            neighbors,
            D_st,
            V_st,
            id_swap,
            id3_ba,
            id3_ca,
            id3_ragged_idx,
            cell_offsets,
        )

    def forward(
        self,
        z: paddle.Tensor,
        frac_coords: paddle.Tensor,
        atom_types: paddle.Tensor,
        num_atoms: paddle.Tensor,
        batch: paddle.Tensor,
        lattice: Optional[paddle.Tensor] = None,
        interaction_graph=None,
    ):
        """
        args:
            z: (N_cryst, num_latent)
            frac_coords: (N_atoms, 3)
            atom_types: (N_atoms, ) with D3PM need to use atomic number
            num_atoms: (N_cryst,)
            batch: (N_atoms, )
            lattice: (N_cryst, 3, 3) (optional, either lengths and angles or lattice
                must be passed)
        """
        assert lattice is not None
        distorted_lattice = lattice
        pos = frac_to_cart_coords_with_lattice(
            frac_coords, num_atoms, lattice=distorted_lattice
        )
        atomic_numbers = atom_types.cast(dtype="int64")
        if interaction_graph is None:
            interaction_graph = self.generate_interaction_graph(
                pos,
                distorted_lattice,
                num_atoms,
            )
        (
            edge_index,
            neighbors,
            D_st,
            V_st,
            id_swap,
            id3_ba,
            id3_ca,
            id3_ragged_idx,
            to_jimages,
        ) = interaction_graph

        idx_s, idx_t = edge_index
        cosφ_cab = inner_product_normalized(V_st[id3_ca], V_st[id3_ba])
        rad_cbf3, cbf3 = self.cbf_basis3(D_st, cosφ_cab, id3_ca)
        rbf = self.radial_basis(D_st)
        h = self.atom_emb(atomic_numbers)
        if z is not None:
            z_per_atom = z[batch]
            h = paddle.concat(x=[h, z_per_atom], axis=1)
            h = self.atom_latent_emb(h)
        m = self.edge_emb(h, rbf, idx_s, idx_t)
        batch_edge = batch[edge_index[0]]
        cosines = paddle.nn.functional.cosine_similarity(
            x1=V_st[:, None], x2=distorted_lattice[batch_edge], axis=-1
        )
        m = paddle.concat(x=[m, cosines], axis=-1)
        m = self.angle_edge_emb(m)
        rbf3 = self.mlp_rbf3(rbf)
        cbf3 = self.mlp_cbf3(rad_cbf3, cbf3, id3_ca, id3_ragged_idx)
        rbf_h = self.mlp_rbf_h(rbf)
        rbf_out = self.mlp_rbf_out(rbf)
        E_t, F_st = self.out_blocks[0](h, m, rbf_out, idx_t)
        distance_vec = V_st * D_st[:, None]

        rbf_lattice = self.mlp_rbf_lattice(rbf)
        lattice_update = self.lattice_out_blocks[0](
            edge_emb=m,
            edge_index=edge_index,
            distance_vec=distance_vec,
            lattice=distorted_lattice,
            batch=batch,
            rbf=rbf_lattice,
            normalize_score=True,
        )

        for i in range(self.num_blocks):
            h, m = self.int_blocks[i](
                h=h,
                m=m,
                rbf3=rbf3,
                cbf3=cbf3,
                id3_ragged_idx=id3_ragged_idx,
                id_swap=id_swap,
                id3_ba=id3_ba,
                id3_ca=id3_ca,
                rbf_h=rbf_h,
                idx_s=idx_s,
                idx_t=idx_t,
            )
            E, F = self.out_blocks[i + 1](h, m, rbf_out, idx_t)
            F_st += F
            E_t += E
            rbf_lattice = self.mlp_rbf_lattice(rbf)
            lattice_update += self.lattice_out_blocks[i + 1](
                edge_emb=m,
                edge_index=edge_index,
                distance_vec=distance_vec,
                lattice=distorted_lattice,
                batch=batch,
                rbf=rbf_lattice,
                normalize_score=True,
            )
        # nMolecules = paddle.max(x=batch) + 1
        # E_t = scatter(E_t, batch, dim=0, dim_size=nMolecules, reduce="sum")
        F_st_vec = F_st[:, :, None] * V_st[:, None, :]
        F_t = scatter(F_st_vec, idx_t, dim=0, dim_size=num_atoms.sum(), reduce="add")
        F_t = F_t.squeeze(axis=1)

        return h, F_t, lattice_update


class GemNetTCtrl(GemNetT):
    def __init__(self, condition_on_adapt: List[str], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.condition_on_adapt = condition_on_adapt
        self.cond_adapt_layers = paddle.nn.LayerDict()
        self.cond_mixin_layers = paddle.nn.LayerDict()
        self.emb_size_atom = (
            kwargs["emb_size_atom"] if "emb_size_atom" in kwargs else 512
        )
        for cond in condition_on_adapt:
            adapt_layers = []
            mixin_layers = []
            for _ in range(self.num_blocks):
                adapt_layers.append(
                    paddle.nn.Sequential(
                        paddle.nn.Linear(
                            in_features=self.emb_size_atom * 2,
                            out_features=self.emb_size_atom,
                        ),
                        paddle.nn.ReLU(),
                        paddle.nn.Linear(
                            in_features=self.emb_size_atom,
                            out_features=self.emb_size_atom,
                        ),
                    )
                )
                mixin_layers.append(
                    paddle.nn.Linear(
                        in_features=self.emb_size_atom,
                        out_features=self.emb_size_atom,
                        bias_attr=False,
                    )
                )
                init_Constant = paddle.nn.initializer.Constant(value=0.0)
                init_Constant(mixin_layers[-1].weight)
            self.cond_adapt_layers[cond] = paddle.nn.LayerList(sublayers=adapt_layers)
            self.cond_mixin_layers[cond] = paddle.nn.LayerList(sublayers=mixin_layers)

    def forward(
        self,
        z: paddle.Tensor,
        frac_coords: paddle.Tensor,
        atom_types: paddle.Tensor,
        num_atoms: paddle.Tensor,
        batch: paddle.Tensor,
        lattice: paddle.Tensor,
        cond_adapt: Optional[Dict[str, paddle.Tensor]] = None,
        cond_adapt_mask: Optional[Dict[str, paddle.Tensor]] = None,
        interaction_graph=None,
    ):
        assert lattice is not None
        distorted_lattice = lattice
        pos = frac_to_cart_coords_with_lattice(
            frac_coords, num_atoms, lattice=distorted_lattice
        )
        atomic_numbers = atom_types.cast(dtype="int64")
        if interaction_graph is None:
            interaction_graph = self.generate_interaction_graph(
                pos,
                distorted_lattice,
                num_atoms,
            )
        (
            edge_index,
            neighbors,
            D_st,
            V_st,
            id_swap,
            id3_ba,
            id3_ca,
            id3_ragged_idx,
            to_jimages,
        ) = interaction_graph
        idx_s, idx_t = edge_index
        cosφ_cab = inner_product_normalized(V_st[id3_ca], V_st[id3_ba])
        rad_cbf3, cbf3 = self.cbf_basis3(D_st, cosφ_cab, id3_ca)
        rbf = self.radial_basis(D_st)
        h = self.atom_emb(atomic_numbers)
        if z is not None:
            z_per_atom = z[batch]
            h = paddle.concat(x=[h, z_per_atom], axis=1)
            h = self.atom_latent_emb(h)
        m = self.edge_emb(h, rbf, idx_s, idx_t)
        batch_edge = batch[edge_index[0]]
        cosines = paddle.nn.functional.cosine_similarity(
            x1=V_st[:, None], x2=distorted_lattice[batch_edge], axis=-1
        )
        m = paddle.concat(x=[m, cosines], axis=-1)
        m = self.angle_edge_emb(m)
        rbf3 = self.mlp_rbf3(rbf)
        cbf3 = self.mlp_cbf3(rad_cbf3, cbf3, id3_ca, id3_ragged_idx)
        rbf_h = self.mlp_rbf_h(rbf)
        rbf_out = self.mlp_rbf_out(rbf)
        E_t, F_st = self.out_blocks[0](h, m, rbf_out, idx_t)
        distance_vec = V_st * D_st[:, None]
        lattice_update = None
        rbf_lattice = self.mlp_rbf_lattice(rbf)
        lattice_update = self.lattice_out_blocks[0](
            edge_emb=m,
            edge_index=edge_index,
            distance_vec=distance_vec,
            lattice=distorted_lattice,
            batch=batch,
            rbf=rbf_lattice,
            normalize_score=True,
        )
        if cond_adapt is not None and cond_adapt_mask is not None:
            cond_adapt_per_atom = {}
            cond_adapt_mask_per_atom = {}
            for cond in self.condition_on_adapt:
                cond_adapt_per_atom[cond] = cond_adapt[cond][batch]
                cond_adapt_mask_per_atom[cond] = 1.0 - cond_adapt_mask[cond][
                    batch
                ].astype(dtype="float32")
        for i in range(self.num_blocks):
            h_adapt = paddle.zeros_like(x=h)
            for cond in self.condition_on_adapt:
                h_adapt_cond = self.cond_adapt_layers[cond][i](
                    paddle.concat(x=[h, cond_adapt_per_atom[cond]], axis=-1)
                )
                h_adapt_cond = self.cond_mixin_layers[cond][i](h_adapt_cond)
                h_adapt += cond_adapt_mask_per_atom[cond] * h_adapt_cond
            h = h + h_adapt
            h, m = self.int_blocks[i](
                h=h,
                m=m,
                rbf3=rbf3,
                cbf3=cbf3,
                id3_ragged_idx=id3_ragged_idx,
                id_swap=id_swap,
                id3_ba=id3_ba,
                id3_ca=id3_ca,
                rbf_h=rbf_h,
                idx_s=idx_s,
                idx_t=idx_t,
            )
            E, F = self.out_blocks[i + 1](h, m, rbf_out, idx_t)
            F_st += F
            E_t += E
            rbf_lattice = self.mlp_rbf_lattice(rbf)
            lattice_update += self.lattice_out_blocks[i + 1](
                edge_emb=m,
                edge_index=edge_index,
                distance_vec=distance_vec,
                lattice=distorted_lattice,
                batch=batch,
                rbf=rbf_lattice,
                normalize_score=True,
            )
        F_st_vec = F_st[:, :, None] * V_st[:, None, :]
        F_t = scatter(F_st_vec, idx_t, dim=0, dim_size=num_atoms.sum(), reduce="add")
        F_t = F_t.squeeze(axis=1)
        return h, F_t, lattice_update


def get_property_embeddings(
    batch, property_embeddings: paddle.nn.LayerDict
) -> paddle.Tensor:
    """
    Keyword arguments
    -----------------
    property_embeddings: paddle.nn.ModuleDict[PropertyToConditonOn, PropertyEmbedding]
        -- a dictionary of property embeddings. The keys are the names of the
        conditional fields in the batch.
    """
    ordered_keys = sorted(property_embeddings.keys())
    if len(ordered_keys) > 0:
        return paddle.concat(
            x=[property_embeddings[k].forward(batch=batch) for k in ordered_keys],
            axis=-1,
        )
    else:
        return paddle.to_tensor(data=[], place=batch["num_atoms"].place)


def get_chemgraph_from_denoiser_output(
    pred_atom_types: paddle.Tensor,
    pred_lattice_eps: paddle.Tensor,
    pred_cart_pos_eps: paddle.Tensor,
    training: bool,
    element_mask_func: (Callable | None),
    x_input,
    batch_idx,
):
    """
    Convert raw denoiser output to Dict and optionally apply masking to element logits.

    Keyword arguments
    -----------------
    pred_atom_atoms: predicted logits for atom types
    pred_lattice_eps: predicted lattice noise
    pred_cart_pos_eps: predicted cartesian position noise
    training: whether or not the model is in training mode - logit masking is only
        applied when sampling
    element_mask_func: when not training, a function can be applied to mask logits for
        certain atom types
    x_input: the nosiy state input to the score model, contains the lattice to convert
        cartesisan to fractional noise.
    batch_idx: the index of the batch.
    """
    if not training and element_mask_func:
        pred_atom_types = element_mask_func(
            logits=pred_atom_types, x=x_input, batch_idx=x_input.get_batch_idx("pos")
        )
    replace_dict = dict(
        frac_coords=(
            x_input["lattice"]
            .inverse()
            .transpose(perm=dim2perm(x_input["lattice"].inverse().ndim, 1, 2))[
                batch_idx
            ]
            @ pred_cart_pos_eps.unsqueeze(axis=-1)
        ).squeeze(axis=-1),
        lattice=pred_lattice_eps,
        atom_types=pred_atom_types,
    )
    return replace_dict


class GemNetTDenoiser(paddle.nn.Layer):
    """A dinoiser that uses a GemNet to denoise the input.

    Args:
        gemnet_cfg (dict): configuration for the GemNet
        gemnet_type (str, optional): Type of GemNet to use, either 'GemNetT' or
            'GemNetTCtrl'. Defaults to 'GemNetT'.
        hidden_dim (int, optional): Number of hidden dimensions in the GemNet.
            Defaults to 512.
        denoise_atom_types (bool, optional): Whether to denoise the atom  types.
            Defaults to True.
        atom_type_diffusion (str, optional): Which type of atom type diffusion to use.
            Defaults to "mask".
        property_embeddings (paddle.nn.LayerDict  |  None, optional): A dictionary of
            property embeddings. Defaults to None.
    """

    def __init__(
        self,
        gemnet_cfg: dict,
        gemnet_type: str = "GemNetT",
        hidden_dim: int = 512,
        denoise_atom_types: bool = True,
        atom_type_diffusion: str = ["mask", "uniform"][0],
        property_embeddings: (paddle.nn.LayerDict | None) = None,
        property_embeddings_adapt_cfg: (Dict | None) = None,  # todo
        # element_mask_func: (Callable | None) = None, # todo
    ):

        super(GemNetTDenoiser, self).__init__()
        if gemnet_type == "GemNetT":
            self.gemnet = GemNetT(**gemnet_cfg)
        elif gemnet_type == "GemNetTCtrl":
            self.gemnet = GemNetTCtrl(**gemnet_cfg)
        else:
            raise NotImplementedError(f"{gemnet_type} not implemented.")
        self.gemnet_cfg = gemnet_cfg
        self.gemnet_type = gemnet_type

        self.noise_level_encoding = NoiseLevelEncoding(hidden_dim)
        self.hidden_dim = hidden_dim
        self.denoise_atom_types = denoise_atom_types
        self.atom_type_diffusion = atom_type_diffusion
        self.property_embeddings = paddle.nn.LayerDict(
            sublayers=property_embeddings or {}
        )
        with_mask_type = self.denoise_atom_types and "mask" in self.atom_type_diffusion
        self.fc_atom = paddle.nn.Linear(
            in_features=hidden_dim, out_features=MAX_ATOMIC_NUM + int(with_mask_type)
        )
        linear_init_(self.fc_atom)
        self.property_embeddings_adapt_cfg = property_embeddings_adapt_cfg
        if property_embeddings_adapt_cfg is not None:
            self.property_embeddings_adapt = paddle.nn.LayerDict()
            for key, config in property_embeddings_adapt_cfg.items():
                property_embedding_layer = PropertyEmbedding(**config)
                self.property_embeddings_adapt[key] = property_embedding_layer
        else:
            self.property_embeddings_adapt = None
        # self.element_mask_func = element_mask_func

    def build_interaction_graph(self, x):
        """Build the dynamic periodic topology outside the CINN boundary."""

        lattice = x["lattice"]
        cart_coords = frac_to_cart_coords_with_lattice(
            x["frac_coords"],
            x["num_atoms"],
            lattice=lattice,
        )
        return self.gemnet.generate_interaction_graph(
            cart_coords,
            lattice,
            x["num_atoms"],
        )

    def forward(self, x, t: paddle.Tensor, interaction_graph=None):
        """
        args:
            x: tuple containing:
                frac_coords: (N_atoms, 3)
                lattice: (N_cryst, 3, 3)
                atom_types: (N_atoms, ), need to use atomic number e.g. H = 1 or ion
                    state
                num_atoms: (N_cryst,)
                batch: (N_atoms,)
            t: (N_cryst,): timestep per crystal
        returns:
            tuple of:
                predicted epsilon: (N_atoms, 3)
                lattice update: (N_crystals, 3, 3)
                predicted atom types: (N_atoms, MAX_ATOMIC_NUM)
        """
        frac_coords, lattice, atom_types, num_atoms, batch = (
            x["frac_coords"],
            x["lattice"],
            x["atom_types"],
            x["num_atoms"],
            x["batch"],
        )
        t_enc = self.noise_level_encoding(t)
        z_per_crystal = t_enc
        if len(self.property_embeddings) > 0:
            property_embedding_values = get_property_embeddings(
                batch=x, property_embeddings=self.property_embeddings
            )
            z_per_crystal = paddle.concat(
                x=[z_per_crystal, property_embedding_values], axis=-1
            )
        if self.property_embeddings_adapt is not None:
            conditions_adapt_dict = {}
            conditions_adapt_mask_dict = {}
            for (
                cond_field,
                property_embedding,
            ) in self.property_embeddings_adapt.items():
                conditions_adapt_mask_dict[
                    cond_field
                ] = get_use_unconditional_embedding(batch=x, cond_field=cond_field)

                conditions_adapt_dict[cond_field] = property_embedding.forward(
                    data=x[cond_field],
                    use_unconditional_embedding=conditions_adapt_mask_dict[cond_field],
                )
            node_embeddings, pred_cart_pos_eps, pred_lattice_eps = self.gemnet(
                z=z_per_crystal,
                frac_coords=frac_coords,
                atom_types=atom_types,
                num_atoms=num_atoms,
                batch=batch,
                lattice=lattice,
                cond_adapt=conditions_adapt_dict,
                cond_adapt_mask=conditions_adapt_mask_dict,
                interaction_graph=interaction_graph,
            )

        else:
            node_embeddings, pred_cart_pos_eps, pred_lattice_eps = self.gemnet(
                z=z_per_crystal,
                frac_coords=frac_coords,
                atom_types=atom_types,
                num_atoms=num_atoms,
                batch=batch,
                lattice=lattice,
                interaction_graph=interaction_graph,
            )

        pred_atom_types = self.fc_atom(node_embeddings)

        return get_chemgraph_from_denoiser_output(
            pred_atom_types=pred_atom_types,
            pred_lattice_eps=pred_lattice_eps,
            pred_cart_pos_eps=pred_cart_pos_eps,
            training=self.training,
            element_mask_func=None,
            x_input=x,
            batch_idx=batch,
        )

    @property
    def cond_fields_model_was_trained_on(self):
        """
        We adopt the convention that all property embeddings are stored in
        paddle.nn.LayerDicts of name property_embeddings or property_embeddings_adapt
        in the case of a fine tuned model.

        This function returns the list of all field names that a given score model was
        trained to condition on.
        """
        return list(self.property_embeddings)


def get_pbc_offsets(pbc: paddle.Tensor, max_offset_integer: int = 3) -> paddle.Tensor:
    """Build the Cartesian product of integer offsets of the periodic boundary. That is,
    if dim=3 and max_offset_integer=1 we build the (2*1 + 1)^3 = 27 possible
    combinations of the Cartesian product of (i,j,k) for i,j,k in
    -max_offset_integer, ..., max_offset_integer. Then, we construct the tensor of
    integer offsets of the pbc vectors,
    i.e., L_{ijk} = row_stack([i * l_1, j * l_2, k * l_3]).

    Args:
        pbc (paddle.Tensor, [batch_size, dim, dim]): The input pbc matrix.
        max_offset_integer (int): The maximum integer offset per dimension to consider
            for the Cartesian product. Defaults to 3.

    Returns:
        paddle.Tensor, [batch_size, (2 * max_offset_integer + 1)^dim, dim]: The tensor
            containing the integer offsets of the pbc vectors.
    """
    offset_range = paddle.arange(start=-max_offset_integer, end=max_offset_integer + 1)
    meshgrid = paddle.stack(
        x=list(
            [i.T for i in paddle.meshgrid(offset_range, offset_range, offset_range)]
        ),
        axis=-1,
    )
    offset = (
        pbc[:, None, None, None] * meshgrid[None, :, :, :, :, None].astype("float32")
    ).sum(axis=-2)
    pbc_offset_per_molecule = offset.reshape([tuple(pbc.shape)[0], -1, 3])
    return pbc_offset_per_molecule


def wrapped_normal_score(
    x: paddle.Tensor,
    mean: paddle.Tensor,
    wrapping_boundary: paddle.Tensor,
    variance_diag: paddle.Tensor,
    batch: paddle.Tensor,
    max_offset_integer: int = 3,
) -> paddle.Tensor:
    """Approximate the the score of a 3D wrapped normal distribution with diagonal
    covariance matrix w.r.t. x via a truncated sum.
       See docstring of `wrapped_normal_score` for details about the arguments

    Args:
        x (paddle.Tensor, [num_atoms, dim])
        mean (paddle.Tensor, [num_atoms, dim])
        wrapping_boundary (paddle.Tensor, [num_molecules, dim, dim])
        variance_diag (paddle.Tensor, [num_atoms,])
        batch (paddle.Tensor, [num_atoms, ])
        max_offset_integer (int), Defaults to 3.

    Returns:
        paddle.Tensor, [num_atoms, dim]: The approximated score of the wrapped normal
            distribution.
    """
    offset_add = get_pbc_offsets(wrapping_boundary, max_offset_integer)
    diffs_k = (x - mean)[:, None] + offset_add[batch]
    dists_sqr_k = diffs_k.pow(y=2).sum(axis=-1)
    score_softmax = paddle.nn.functional.softmax(
        x=-dists_sqr_k / (2 * variance_diag[:, None]), axis=-1
    )
    score = -(score_softmax[:, :, None] * diffs_k).sum(axis=-2) / variance_diag[:, None]
    return score


def wrapped_normal_loss(
    *,
    corruption,
    score_model_output: paddle.Tensor,
    t: paddle.Tensor,
    batch_idx: Optional[paddle.Tensor],
    batch_size: int,
    x: paddle.Tensor,
    noisy_x: paddle.Tensor,
    reduce: Literal["sum", "mean"],
    batch,
    **_,
) -> paddle.Tensor:
    """Compute the loss for a wrapped normal distribution.
    Compares the score of the wrapped normal distribution to the score of the score
    model.
    """
    assert len(t) == batch_size
    _, std = corruption.marginal_prob(
        x=paddle.zeros(shape=(tuple(x.shape)[0], 1)),
        t=t,
        batch_idx=batch_idx,
        num_atoms=batch["num_atoms"],
    )
    pred: paddle.Tensor = score_model_output
    if pred.ndim != 2:
        raise NotImplementedError
    assert hasattr(
        corruption, "wrapping_boundary"
    ), "SDE must be a WrappedSDE, i.e., must have a wrapping boundary."
    wrapping_boundary = corruption.wrapping_boundary
    wrapping_boundary = wrapping_boundary * paddle.eye(num_rows=tuple(x.shape)[-1])[
        None
    ].expand(shape=[batch_size, -1, -1])
    target = (
        wrapped_normal_score(
            x=noisy_x,
            mean=x,
            wrapping_boundary=wrapping_boundary,
            variance_diag=std.squeeze() ** 2,
            batch=batch_idx,
        )
        * std
    )
    delta = target - pred
    losses = delta.square()
    return aggregate_per_sample(losses, batch_idx, reduce=reduce, batch_size=batch_size)


class MatterGen(RuntimeMixin, paddle.nn.Layer):
    """MatterGen: A generative model for inorganic materials design.
    https://www.nature.com/articles/s41586-025-08628-5

    Args:
        decoder_cfg (dict): Decoder configuration.
        lattice_noise_scheduler_cfg (dict): Lattice noise scheduler configuration.
        coord_noise_scheduler_cfg (dict): Coordinate noise scheduler configuration.
        atom_noise_scheduler_cfg (dict): Atom type noise scheduler configuration.
        num_train_timesteps (int, optional): Number of training steps for diffusion.
            Defaults to 1000.
        max_t (float, optional): Maximum diffusion time. Defaults to 1.0.
        time_dim (int, optional): Time dimension. Defaults to 256.
        lattice_loss_weight (float, optional): Lattice loss weight. Defaults to 1.0.
        coord_loss_weight (float, optional): Coordinate loss weight. Defaults to 0.1.
        atom_loss_weight (float, optional): Atom type loss weight. Defaults to 1.0.
        d3pm_hybrid_lambda (float, optional): D3PM hybrid lambda. Defaults to 0.01.
    """

    def __init__(
        self,
        decoder_cfg: dict,
        lattice_noise_scheduler_cfg: dict,
        coord_noise_scheduler_cfg: dict,
        atom_noise_scheduler_cfg: dict,
        num_train_timesteps: int = 1000,
        max_t: float = 1.0,
        time_dim: int = 256,
        lattice_loss_weight: float = 1.0,
        coord_loss_weight: float = 0.1,
        atom_loss_weight: float = 1.0,
        d3pm_hybrid_lambda: float = 0.01,
        execution_backend: str = "eager",
        runtime_options: dict | None = None,
    ) -> None:
        super().__init__()
        self._init_runtime(execution_backend, runtime_options)
        self.model = GemNetTDenoiser(**decoder_cfg)

        self.lattice_scheduler = build_scheduler(lattice_noise_scheduler_cfg)
        self.coord_scheduler = build_scheduler(coord_noise_scheduler_cfg)
        self.atom_scheduler = build_scheduler(atom_noise_scheduler_cfg)

        self.num_train_timesteps = num_train_timesteps
        self.max_t = max_t
        self.time_dim = time_dim
        self.lattice_loss_weight = lattice_loss_weight
        self.coord_loss_weight = coord_loss_weight
        self.atom_loss_weight = atom_loss_weight
        self.d3pm_hybrid_lambda = d3pm_hybrid_lambda

        self.timestep_sampler = UniformTimestepSampler(min_t=1e-05, max_t=max_t)

    def _denoise(self, structure_array, times):
        interaction_graph = self.model.build_interaction_graph(structure_array)
        return self._runtime_denoise(structure_array, times, interaction_graph)

    @runtime_boundary("denoise_step")
    def _runtime_denoise(self, structure_array, times, interaction_graph):
        return self.model(structure_array, times, interaction_graph)

    def forward(self, batch) -> Any:
        structure_array = batch["structure_array"]
        num_atoms = structure_array["num_atoms"]
        batch_size = structure_array["num_atoms"].shape[0]
        batch_idx = paddle.repeat_interleave(
            paddle.arange(batch_size), repeats=structure_array["num_atoms"]
        )
        times = self.timestep_sampler(batch_size)

        # coord noise
        frac_coords = structure_array["frac_coords"] % 1.0
        rand_x = paddle.randn(shape=frac_coords.shape, dtype=frac_coords.dtype)

        input_frac_coords = self.coord_scheduler.add_noise(
            frac_coords,
            rand_x,
            timesteps=times,
            batch_idx=batch_idx,
            num_atoms=num_atoms,
        )

        # lattice noise
        if "lattice" in structure_array.keys():
            lattices = structure_array["lattice"]
        else:
            lattices = lattice_params_to_matrix_paddle(
                structure_array["lengths"], structure_array["angles"]
            )

        rand_l = paddle.randn(shape=lattices.shape, dtype=lattices.dtype)
        rand_l = make_noise_symmetric_preserve_variance(rand_l)

        input_lattice = self.lattice_scheduler.add_noise(
            lattices,
            rand_l,
            timesteps=times,
            num_atoms=structure_array["num_atoms"],
        )

        # atom noise
        atom_type = structure_array["atom_types"]
        atom_type_zero_based = atom_type - 1

        input_atom_type_zero_based = self.atom_scheduler.add_noise(
            atom_type_zero_based,
            timesteps=times,
            batch_idx=batch_idx,
        )
        input_atom_type = input_atom_type_zero_based + 1

        noise_batch = {
            "frac_coords": input_frac_coords,
            "lattice": input_lattice,
            "atom_types": input_atom_type,
            "num_atoms": structure_array["num_atoms"],
            "batch": batch_idx,
        }

        score_model_output = self._denoise(noise_batch, times)

        # coord loss
        loss_coord = wrapped_normal_loss(
            corruption=self.coord_scheduler,
            score_model_output=score_model_output["frac_coords"],
            t=times,
            batch_idx=batch_idx,
            batch_size=batch_size,
            x=frac_coords,
            noisy_x=input_frac_coords,
            reduce="sum",
            batch=structure_array,
        )

        # lattice loss
        loss_lattice = (score_model_output["lattice"] + rand_l).square()
        loss_lattice = loss_lattice.mean(axis=[1, 2])

        # atom type loss
        (
            loss_atom_type,
            base_loss_atom_type,
            cross_entropy_atom_type,
        ) = self.atom_scheduler.compute_loss(
            score_model_output=score_model_output["atom_types"],
            t=times,
            batch_idx=batch_idx,
            batch_size=batch_size,
            x=atom_type_zero_based,
            noisy_x=input_atom_type_zero_based,
            reduce="sum",
            d3pm_hybrid_lambda=self.d3pm_hybrid_lambda,
        )

        loss_coord = loss_coord.mean()
        loss_lattice = loss_lattice.mean()
        loss_atom_type = loss_atom_type.mean()
        base_loss_atom_type = base_loss_atom_type.mean()
        cross_entropy_atom_type = cross_entropy_atom_type.mean()

        loss = (
            self.coord_loss_weight * loss_coord
            + self.lattice_loss_weight * loss_lattice
            + self.atom_loss_weight * loss_atom_type
        )
        return {
            "loss_dict": {
                "loss": loss,
                "loss_coord": loss_coord,
                "loss_lattice": loss_lattice,
                "loss_atom_type": loss_atom_type,
                "base_loss_atom_type": base_loss_atom_type,
                "cross_entropy_atom_type": cross_entropy_atom_type,
            }
        }

    @paddle.no_grad()
    def sample(
        self,
        batch_data,
        num_inference_steps=1000,
        _eps_t=0.001,
        n_step_corrector: int = 1,
        record: bool = False,
    ):
        structure_array = batch_data["structure_array"]
        num_atoms = structure_array["num_atoms"]
        batch_size = num_atoms.shape[0]
        batch_idx = paddle.repeat_interleave(
            paddle.arange(batch_size), repeats=num_atoms
        )

        # get the initial noise
        lattice = self.lattice_scheduler.prior_sampling(
            shape=(batch_size, 3, 3), num_atoms=num_atoms
        )
        frac_coords = self.coord_scheduler.prior_sampling(
            shape=(num_atoms.sum(), 3), num_atoms=num_atoms, batch_idx=batch_idx
        )
        atom_types_zero_based = self.atom_scheduler.prior_sampling(
            shape=(num_atoms.sum(),)
        )
        atom_types = atom_types_zero_based + 1

        # update the structure_array with the initial noise
        structure_array.update(
            {
                "frac_coords": frac_coords,
                "lattice": lattice,
                "atom_types": atom_types,
            }
        )
        structure_array["batch"] = batch_idx

        timesteps = paddle.linspace(self.max_t, stop=_eps_t, num=num_inference_steps)
        dt = -paddle.to_tensor(data=(self.max_t - _eps_t) / (num_inference_steps - 1))
        recorded_samples = []
        for i in tqdm(range(num_inference_steps), desc="Sampling..."):
            t = paddle.full(shape=(batch_size,), fill_value=timesteps[i])
            for _ in range(n_step_corrector):
                score_out = self._denoise(structure_array, t)

                if record:
                    recorded_samples.append(structure_array)

                score_out["frac_coords"] = (
                    score_out["frac_coords"]
                    / self.coord_scheduler.marginal_prob(
                        structure_array["frac_coords"],
                        t=t,
                        batch_idx=batch_idx,
                        num_atoms=num_atoms,
                    )[1]
                )
                score_out["lattice"] = (
                    score_out["lattice"]
                    / self.lattice_scheduler.marginal_prob(
                        structure_array["lattice"], t=t, num_atoms=num_atoms
                    )[1]
                )

                frac_coords, _ = self.coord_scheduler.step_correct(
                    model_output=score_out["frac_coords"],
                    timestep=t,
                    sample=structure_array["frac_coords"],
                    batch_idx=batch_idx,
                )
                cell, _ = self.lattice_scheduler.step_correct(
                    structure_array["lattice"],
                    batch_idx=None,
                    score=score_out["lattice"],
                    t=t,
                )
                structure_array.update(
                    {
                        "frac_coords": frac_coords,
                        "lattice": cell,
                    }
                )
            score_out = self._denoise(structure_array, t)

            if record:
                recorded_samples.append(structure_array)

            score_out["frac_coords"] = (
                score_out["frac_coords"]
                / self.coord_scheduler.marginal_prob(
                    structure_array["frac_coords"],
                    t=t,
                    batch_idx=batch_idx,
                    num_atoms=num_atoms,
                )[1]
            )
            score_out["lattice"] = (
                score_out["lattice"]
                / self.lattice_scheduler.marginal_prob(
                    structure_array["lattice"], t=t, num_atoms=num_atoms
                )[1]
            )

            frac_coords, frac_coords_mean = self.coord_scheduler.step_pred(
                frac_coords,
                t=t,
                dt=dt,
                batch_idx=batch_idx,
                score=score_out["frac_coords"],
                num_atoms=num_atoms,
            )

            cell, cell_mean = self.lattice_scheduler.step_pred(
                cell,
                t=t,
                dt=dt,
                batch_idx=None,
                score=score_out["lattice"],
                num_atoms=num_atoms,
            )
            atom_type, atom_type_mean = self.atom_scheduler.step(
                x=structure_array["atom_types"] - 1,
                t=t,
                batch_idx=batch_idx,
                score=score_out["atom_types"],
            )
            atom_type += 1
            atom_type_mean += 1

            structure_array.update(
                {"frac_coords": frac_coords, "lattice": cell, "atom_types": atom_type}
            )
            structure_array_mean = {
                "frac_coords": frac_coords_mean,
                "lattice": cell_mean,
                "atom_types": atom_type_mean,
                "num_atoms": num_atoms,
            }

        start_idx = 0
        result = []
        for i in range(batch_size):
            end_idx = start_idx + num_atoms[i]
            # for mattertgen, we need to use the mean value of the predicted structure
            result.append(
                {
                    "num_atoms": num_atoms[i].tolist(),
                    "atom_types": structure_array_mean["atom_types"][
                        start_idx:end_idx
                    ].tolist(),
                    "frac_coords": structure_array_mean["frac_coords"][
                        start_idx:end_idx
                    ].tolist(),
                    "lattice": structure_array_mean["lattice"][i].tolist(),
                }
            )
            # result.append(
            #     {
            #         "num_atoms": num_atoms[i].tolist(),
            #         "atom_types": structure_array["atom_types"][
            #             start_idx:end_idx
            #         ].tolist(),
            #         "frac_coords": structure_array["frac_coords"][
            #             start_idx:end_idx
            #         ].tolist(),
            #         "lattice": structure_array["lattice"][i].tolist(),
            #     }
            # )
            start_idx += num_atoms[i]

        return {"result": result}


class MatterGenWithCondition(RuntimeMixin, paddle.nn.Layer):
    """MatterGenWithCondition: A generative model for inorganic materials design.
    https://www.nature.com/articles/s41586-025-08628-5

    Args:
        set_embedding_type_cfg (dict): SetEmbeddingType configuration.
        condition_names (list): Attribute name as a conditional constraint.
        decoder_cfg (dict): Decoder configuration.
        lattice_noise_scheduler_cfg (dict): Lattice noise scheduler configuration.
        coord_noise_scheduler_cfg (dict): Coordinate noise scheduler configuration.
        atom_noise_scheduler_cfg (dict): Atom type noise scheduler configuration.
        num_train_timesteps (int, optional): Number of training steps for diffusion.
            Defaults to 1000.
        max_t (float, optional): Maximum diffusion time. Defaults to 1.0.
        time_dim (int, optional): Time dimension. Defaults to 256.
        lattice_loss_weight (float, optional): Lattice loss weight. Defaults to 1.0.
        coord_loss_weight (float, optional): Coordinate loss weight. Defaults to 0.1.
        atom_loss_weight (float, optional): Atom type loss weight. Defaults to 1.0.
        d3pm_hybrid_lambda (float, optional): D3PM hybrid lambda. Defaults to 0.01.
    """

    def __init__(
        self,
        set_embedding_type_cfg: dict,
        condition_names: list,
        decoder_cfg: dict,
        lattice_noise_scheduler_cfg: dict,
        coord_noise_scheduler_cfg: dict,
        atom_noise_scheduler_cfg: dict,
        num_train_timesteps: int = 1000,
        max_t: float = 1.0,
        time_dim: int = 256,
        lattice_loss_weight: float = 1.0,
        coord_loss_weight: float = 0.1,
        atom_loss_weight: float = 1.0,
        d3pm_hybrid_lambda: float = 0.01,
        execution_backend: str = "eager",
        runtime_options: dict | None = None,
    ) -> None:
        super().__init__()
        self._init_runtime(execution_backend, runtime_options)
        self.set_embedding_type_cfg = set_embedding_type_cfg
        self.condition_names = condition_names

        self.set_embedding_type = SetEmbeddingType(**set_embedding_type_cfg)

        self.model = GemNetTDenoiser(**decoder_cfg)

        self.lattice_scheduler = build_scheduler(lattice_noise_scheduler_cfg)
        self.coord_scheduler = build_scheduler(coord_noise_scheduler_cfg)
        self.atom_scheduler = build_scheduler(atom_noise_scheduler_cfg)

        self.num_train_timesteps = num_train_timesteps
        self.max_t = max_t
        self.time_dim = time_dim
        self.lattice_loss_weight = lattice_loss_weight
        self.coord_loss_weight = coord_loss_weight
        self.atom_loss_weight = atom_loss_weight
        self.d3pm_hybrid_lambda = d3pm_hybrid_lambda

        self.timestep_sampler = UniformTimestepSampler(min_t=1e-05, max_t=max_t)

    def _denoise(self, structure_array, times):
        interaction_graph = self.model.build_interaction_graph(structure_array)
        return self._runtime_denoise(structure_array, times, interaction_graph)

    @runtime_boundary("denoise_step")
    def _runtime_denoise(self, structure_array, times, interaction_graph):
        return self.model(structure_array, times, interaction_graph)

    def before_train(self, trainer):
        # This function serves as a pre-training hook, designed to execute
        # initialization/setup operations before the training pipeline begins. It
        # follows a dependency injection pattern - the Trainer instance must be fully
        # initialized before this hook is injected as a callback parameter into the
        # training workflow.
        set_property_scalers = SetPropertyScalers()
        set_property_scalers.on_fit_start(
            train_dataloader=trainer.train_dataloader, model=self
        )

    def forward(self, batch) -> Any:
        structure_array = batch["structure_array"]
        use_unconditional_embedding = self.set_embedding_type(
            batch, self.condition_names
        )

        num_atoms = structure_array["num_atoms"]
        batch_size = structure_array["num_atoms"].shape[0]
        batch_idx = paddle.repeat_interleave(
            paddle.arange(batch_size), repeats=structure_array["num_atoms"]
        )
        times = self.timestep_sampler(batch_size)

        # coord noise
        frac_coords = structure_array["frac_coords"] % 1.0
        rand_x = paddle.randn(shape=frac_coords.shape, dtype=frac_coords.dtype)

        input_frac_coords = self.coord_scheduler.add_noise(
            frac_coords,
            rand_x,
            timesteps=times,
            batch_idx=batch_idx,
            num_atoms=num_atoms,
        )

        # lattice noise
        if "lattice" in structure_array.keys():
            lattices = structure_array["lattice"]
        else:
            lattices = lattice_params_to_matrix_paddle(
                structure_array["lengths"], structure_array["angles"]
            )

        rand_l = paddle.randn(shape=lattices.shape, dtype=lattices.dtype)
        rand_l = make_noise_symmetric_preserve_variance(rand_l)

        input_lattice = self.lattice_scheduler.add_noise(
            lattices,
            rand_l,
            timesteps=times,
            num_atoms=structure_array["num_atoms"],
        )

        # atom noise
        atom_type = structure_array["atom_types"]
        atom_type_zero_based = atom_type - 1

        input_atom_type_zero_based = self.atom_scheduler.add_noise(
            atom_type_zero_based,
            timesteps=times,
            batch_idx=batch_idx,
        )
        input_atom_type = input_atom_type_zero_based + 1

        noise_batch = {
            "frac_coords": input_frac_coords,
            "lattice": input_lattice,
            "atom_types": input_atom_type,
            "num_atoms": structure_array["num_atoms"],
            "batch": batch_idx,
            _USE_UNCONDITIONAL_EMBEDDING: use_unconditional_embedding,
        }
        for condition_name in self.condition_names:
            noise_batch[condition_name] = batch[condition_name]

        score_model_output = self._denoise(noise_batch, times)

        # coord loss
        loss_coord = wrapped_normal_loss(
            corruption=self.coord_scheduler,
            score_model_output=score_model_output["frac_coords"],
            t=times,
            batch_idx=batch_idx,
            batch_size=batch_size,
            x=frac_coords,
            noisy_x=input_frac_coords,
            reduce="sum",
            batch=structure_array,
        )

        # lattice loss
        loss_lattice = (score_model_output["lattice"] + rand_l).square()
        loss_lattice = loss_lattice.mean(axis=[1, 2])

        # atom type loss
        (
            loss_atom_type,
            base_loss_atom_type,
            cross_entropy_atom_type,
        ) = self.atom_scheduler.compute_loss(
            score_model_output=score_model_output["atom_types"],
            t=times,
            batch_idx=batch_idx,
            batch_size=batch_size,
            x=atom_type_zero_based,
            noisy_x=input_atom_type_zero_based,
            reduce="sum",
            d3pm_hybrid_lambda=self.d3pm_hybrid_lambda,
        )

        loss_coord = loss_coord.mean()
        loss_lattice = loss_lattice.mean()
        loss_atom_type = loss_atom_type.mean()
        base_loss_atom_type = base_loss_atom_type.mean()
        cross_entropy_atom_type = cross_entropy_atom_type.mean()

        loss = (
            self.coord_loss_weight * loss_coord
            + self.lattice_loss_weight * loss_lattice
            + self.atom_loss_weight * loss_atom_type
        )
        return {
            "loss_dict": {
                "loss": loss,
                "loss_coord": loss_coord,
                "loss_lattice": loss_lattice,
                "loss_atom_type": loss_atom_type,
                "base_loss_atom_type": base_loss_atom_type,
                "cross_entropy_atom_type": cross_entropy_atom_type,
            }
        }

    @paddle.no_grad()
    def _score_fn(
        self,
        structure_array,
        t,
        batch_idx,
        num_atoms,
        guidance_scale: float = 2.0,
    ):
        if not hasattr(self, "set_conditional_embedding_type"):
            self.set_conditional_embedding_type = SetConditionalEmbeddingType()
        if not hasattr(self, "set_unconditional_embedding_type"):
            self.set_unconditional_embedding_type = SetUnconditionalEmbeddingType()

        conditional_embedding = self.set_conditional_embedding_type(
            structure_array, self.condition_names
        )
        uunconditional_embedding = self.set_unconditional_embedding_type(
            structure_array, self.condition_names
        )

        structure_array[_USE_UNCONDITIONAL_EMBEDDING] = conditional_embedding
        score_out_cond = self._denoise(structure_array, t)

        score_out_cond["frac_coords"] = (
            score_out_cond["frac_coords"]
            / self.coord_scheduler.marginal_prob(
                structure_array["frac_coords"],
                t=t,
                batch_idx=batch_idx,
                num_atoms=num_atoms,
            )[1]
        )
        score_out_cond["lattice"] = (
            score_out_cond["lattice"]
            / self.lattice_scheduler.marginal_prob(
                structure_array["lattice"], t=t, num_atoms=num_atoms
            )[1]
        )

        structure_array[_USE_UNCONDITIONAL_EMBEDDING] = uunconditional_embedding
        score_out_uncond = self._denoise(structure_array, t)

        score_out_uncond["frac_coords"] = (
            score_out_uncond["frac_coords"]
            / self.coord_scheduler.marginal_prob(
                structure_array["frac_coords"],
                t=t,
                batch_idx=batch_idx,
                num_atoms=num_atoms,
            )[1]
        )
        score_out_uncond["lattice"] = (
            score_out_uncond["lattice"]
            / self.lattice_scheduler.marginal_prob(
                structure_array["lattice"], t=t, num_atoms=num_atoms
            )[1]
        )

        score_out_uncond.update(
            {
                "frac_coords": paddle.lerp(
                    x=score_out_uncond["frac_coords"],
                    y=score_out_cond["frac_coords"],
                    weight=guidance_scale,
                ),
                "lattice": paddle.lerp(
                    x=score_out_uncond["lattice"],
                    y=score_out_cond["lattice"],
                    weight=guidance_scale,
                ),
                "atom_types": paddle.lerp(
                    x=score_out_uncond["atom_types"],
                    y=score_out_cond["atom_types"],
                    weight=guidance_scale,
                ),
            }
        )
        return score_out_uncond

    @paddle.no_grad()
    def sample(
        self,
        batch_data,
        num_inference_steps=1000,
        _eps_t=0.001,
        n_step_corrector: int = 1,
        record: bool = False,
        guidance_scale: float = 2.0,
    ):
        structure_array = batch_data["structure_array"]
        num_atoms = structure_array["num_atoms"]
        batch_size = num_atoms.shape[0]
        batch_idx = paddle.repeat_interleave(
            paddle.arange(batch_size), repeats=num_atoms
        )

        # get the initial noise
        lattice = self.lattice_scheduler.prior_sampling(
            shape=(batch_size, 3, 3), num_atoms=num_atoms
        )
        frac_coords = self.coord_scheduler.prior_sampling(
            shape=(num_atoms.sum(), 3), num_atoms=num_atoms, batch_idx=batch_idx
        )
        atom_types_zero_based = self.atom_scheduler.prior_sampling(
            shape=(num_atoms.sum(),)
        )
        atom_types = atom_types_zero_based + 1

        # update the structure_array with the initial noise
        structure_array.update(
            {
                "frac_coords": frac_coords,
                "lattice": lattice,
                "atom_types": atom_types,
            }
        )
        structure_array["batch"] = batch_idx
        for condition_name in self.condition_names:
            structure_array[condition_name] = batch_data[condition_name]

        timesteps = paddle.linspace(self.max_t, stop=_eps_t, num=num_inference_steps)
        dt = -paddle.to_tensor(data=(self.max_t - _eps_t) / (num_inference_steps - 1))
        recorded_samples = []
        for i in tqdm(range(num_inference_steps), desc="Sampling..."):
            t = paddle.full(shape=(batch_size,), fill_value=timesteps[i])
            for _ in range(n_step_corrector):
                score_out = self._score_fn(
                    structure_array, t, batch_idx, num_atoms, guidance_scale
                )

                if record:
                    recorded_samples.append(structure_array)
                frac_coords, _ = self.coord_scheduler.step_correct(
                    model_output=score_out["frac_coords"],
                    timestep=t,
                    sample=structure_array["frac_coords"],
                    batch_idx=batch_idx,
                )
                cell, _ = self.lattice_scheduler.step_correct(
                    structure_array["lattice"],
                    batch_idx=None,
                    score=score_out["lattice"],
                    t=t,
                )
                structure_array.update(
                    {
                        "frac_coords": frac_coords,
                        "lattice": cell,
                    }
                )
            score_out = self._score_fn(
                structure_array, t, batch_idx, num_atoms, guidance_scale
            )

            if record:
                recorded_samples.append(structure_array)

            frac_coords, frac_coords_mean = self.coord_scheduler.step_pred(
                frac_coords,
                t=t,
                dt=dt,
                batch_idx=batch_idx,
                score=score_out["frac_coords"],
                num_atoms=num_atoms,
            )

            cell, cell_mean = self.lattice_scheduler.step_pred(
                cell,
                t=t,
                dt=dt,
                batch_idx=None,
                score=score_out["lattice"],
                num_atoms=num_atoms,
            )
            atom_type, atom_type_mean = self.atom_scheduler.step(
                x=structure_array["atom_types"] - 1,
                t=t,
                batch_idx=batch_idx,
                score=score_out["atom_types"],
            )
            atom_type += 1
            atom_type_mean += 1

            structure_array.update(
                {"frac_coords": frac_coords, "lattice": cell, "atom_types": atom_type}
            )
            structure_array_mean = {
                "frac_coords": frac_coords_mean,
                "lattice": cell_mean,
                "atom_types": atom_type_mean,
                "num_atoms": num_atoms,
            }

        start_idx = 0
        result = []
        for i in range(batch_size):
            end_idx = start_idx + num_atoms[i]
            # for mattertgen, we need to use the mean value of the predicted structure
            result.append(
                {
                    "num_atoms": num_atoms[i].tolist(),
                    "atom_types": structure_array_mean["atom_types"][
                        start_idx:end_idx
                    ].tolist(),
                    "frac_coords": structure_array_mean["frac_coords"][
                        start_idx:end_idx
                    ].tolist(),
                    "lattice": structure_array_mean["lattice"][i].tolist(),
                }
            )
            # result.append(
            #     {
            #         "num_atoms": num_atoms[i].tolist(),
            #         "atom_types": structure_array["atom_types"][
            #             start_idx:end_idx
            #         ].tolist(),
            #         "frac_coords": structure_array["frac_coords"][
            #             start_idx:end_idx
            #         ].tolist(),
            #         "lattice": structure_array["lattice"][i].tolist(),
            #     }
            # )
            start_idx += num_atoms[i]

        return {"result": result}

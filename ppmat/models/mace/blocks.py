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

"""Paddle port of the ``mace.modules.blocks`` classes used by MACE-MP-0.

Only the two blocks that the released ``mace_mp_0`` (medium) checkpoint uses
are implemented here:

* :class:`EquivariantProductBasisBlock`
* :class:`RealAgnosticResidualInteractionBlock`

Parameter / buffer names match the official PyTorch modules one to one, so a
converted ``state_dict`` can be loaded without renaming.
"""

from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
import paddle
import paddle.nn.functional as F
from paddle import nn

from ppmat.models.common.e3nn import o3
from ppmat.models.common.e3nn.nn import Activation
from ppmat.models.common.e3nn.nn import FullyConnectedNet
from ppmat.models.mace.irreps_tools import reshape_irreps
from ppmat.models.mace.irreps_tools import tp_out_irreps_with_instructions
from ppmat.models.mace.symmetric_contraction import SymmetricContraction
from ppmat.utils.scatter import scatter_sum

__all__ = [
    "AtomicEnergiesBlock",
    "EquivariantProductBasisBlock",
    "LinearNodeEmbeddingBlock",
    "LinearReadoutBlock",
    "NonLinearReadoutBlock",
    "RealAgnosticResidualInteractionBlock",
    "ScaleShiftBlock",
]


def silu(x: paddle.Tensor) -> paddle.Tensor:
    """Numerically stable SiLU.

    ``paddle.nn.functional.silu`` produces ``nan`` gradients for the large
    pre-activations the MACE radial MLP reaches with real weights, so build it
    from the (stable) sigmoid instead.  Mathematically identical to SiLU.

    The ``e3nn_cst`` attribute pins ``normalize2mom`` to the exact constant
    e3nn uses, so the converted MACE-MP-0 weights reproduce the PyTorch model.
    Without the attribute the public e3nn package keeps its Monte-Carlo
    estimate, which is what the other models (InfGCN, SphereNet) were trained
    with.
    """
    return x * F.sigmoid(x)


silu.e3nn_cst = 1.6791767923989418


class EquivariantProductBasisBlock(nn.Layer):
    """Paddle port of ``mace.modules.blocks.EquivariantProductBasisBlock``."""

    def __init__(
        self,
        node_feats_irreps: Union[str, o3.Irreps],
        target_irreps: Union[str, o3.Irreps],
        correlation: int,
        use_sc: bool = True,
        num_elements: Optional[int] = None,
        use_agnostic_product: bool = False,
        dtype=None,
    ) -> None:
        super().__init__()

        self.use_sc = use_sc
        self.use_agnostic_product = use_agnostic_product
        if self.use_agnostic_product:
            num_elements = 1

        self.symmetric_contractions = SymmetricContraction(
            irreps_in=node_feats_irreps,
            irreps_out=target_irreps,
            correlation=correlation,
            num_elements=num_elements,
            dtype=dtype,
        )
        self.linear = o3.Linear(
            target_irreps,
            target_irreps,
            internal_weights=True,
            shared_weights=True,
        )

    def forward(
        self,
        node_feats: paddle.Tensor,
        sc: Optional[paddle.Tensor],
        node_attrs: paddle.Tensor,
    ) -> paddle.Tensor:
        if self.use_agnostic_product:
            node_attrs = paddle.ones(
                [node_feats.shape[0], 1], dtype=node_feats.dtype
            )
        node_feats = self.symmetric_contractions(node_feats, node_attrs)
        if self.use_sc and sc is not None:
            return self.linear(node_feats) + sc
        return self.linear(node_feats)


class RealAgnosticResidualInteractionBlock(nn.Layer):
    """Paddle port of ``mace.modules.blocks.RealAgnosticResidualInteractionBlock``.

    The LAMMPS multi-GPU ghost handling of the official implementation is not
    needed for inference / training on padded neighbour lists and is omitted.
    """

    def __init__(
        self,
        node_attrs_irreps: Union[str, o3.Irreps],
        node_feats_irreps: Union[str, o3.Irreps],
        edge_attrs_irreps: Union[str, o3.Irreps],
        edge_feats_irreps: Union[str, o3.Irreps],
        target_irreps: Union[str, o3.Irreps],
        hidden_irreps: Union[str, o3.Irreps],
        avg_num_neighbors: float,
        edge_irreps: Optional[Union[str, o3.Irreps]] = None,
        radial_MLP: Optional[List[int]] = None,
    ) -> None:
        super().__init__()

        self.node_attrs_irreps = o3.Irreps(node_attrs_irreps)
        self.node_feats_irreps = o3.Irreps(node_feats_irreps)
        self.edge_attrs_irreps = o3.Irreps(edge_attrs_irreps)
        self.edge_feats_irreps = o3.Irreps(edge_feats_irreps)
        self.target_irreps = o3.Irreps(target_irreps)
        self.hidden_irreps = o3.Irreps(hidden_irreps)
        self.avg_num_neighbors = avg_num_neighbors
        if radial_MLP is None:
            radial_MLP = [64, 64, 64]
        self.radial_MLP = radial_MLP
        if edge_irreps is None:
            edge_irreps = self.node_feats_irreps
        self.edge_irreps = o3.Irreps(edge_irreps)

        # First linear
        self.linear_up = o3.Linear(
            self.node_feats_irreps,
            self.edge_irreps,
            internal_weights=True,
            shared_weights=True,
        )
        # TensorProduct
        irreps_mid, instructions = tp_out_irreps_with_instructions(
            self.edge_irreps,
            self.edge_attrs_irreps,
            self.target_irreps,
        )
        self.conv_tp = o3.TensorProduct(
            self.edge_irreps,
            self.edge_attrs_irreps,
            irreps_mid,
            instructions=instructions,
            shared_weights=False,
            internal_weights=False,
        )
        self.irreps_mid = irreps_mid

        # Convolution weights
        input_dim = self.edge_feats_irreps.num_irreps
        self.conv_tp_weights = FullyConnectedNet(
            [input_dim] + self.radial_MLP + [self.conv_tp.weight_numel],
            silu,
        )

        # Linear
        self.irreps_out = self.target_irreps
        self.linear = o3.Linear(
            irreps_mid,
            self.irreps_out,
            internal_weights=True,
            shared_weights=True,
        )

        # Selector TensorProduct
        self.skip_tp = o3.FullyConnectedTensorProduct(
            self.node_feats_irreps,
            self.node_attrs_irreps,
            self.hidden_irreps,
            internal_weights=True,
            shared_weights=True,
        )
        self.reshape = reshape_irreps(self.irreps_out)

    def forward(
        self,
        node_attrs: paddle.Tensor,
        node_feats: paddle.Tensor,
        edge_attrs: paddle.Tensor,
        edge_feats: paddle.Tensor,
        edge_index: paddle.Tensor,
        cutoff: Optional[paddle.Tensor] = None,
    ) -> Tuple[paddle.Tensor, paddle.Tensor]:
        sc = self.skip_tp(node_feats, node_attrs)
        node_feats = self.linear_up(node_feats)
        tp_weights = self.conv_tp_weights(edge_feats)
        if cutoff is not None:
            tp_weights = tp_weights * cutoff
        mji = self.conv_tp(node_feats[edge_index[0]], edge_attrs, tp_weights)
        message = scatter_sum(
            src=mji, index=edge_index[1], dim=0, dim_size=node_feats.shape[0]
        )
        message = self.linear(message) / self.avg_num_neighbors
        return self.reshape(message), sc


class LinearNodeEmbeddingBlock(nn.Layer):
    """Paddle port of ``mace.modules.blocks.LinearNodeEmbeddingBlock``."""

    def __init__(
        self,
        irreps_in: Union[str, o3.Irreps],
        irreps_out: Union[str, o3.Irreps],
    ) -> None:
        super().__init__()
        self.linear = o3.Linear(
            irreps_in=irreps_in,
            irreps_out=irreps_out,
            internal_weights=True,
            shared_weights=True,
        )

    def forward(self, node_attrs: paddle.Tensor) -> paddle.Tensor:
        return self.linear(node_attrs)


class LinearReadoutBlock(nn.Layer):
    """Paddle port of ``mace.modules.blocks.LinearReadoutBlock``."""

    def __init__(
        self,
        irreps_in: Union[str, o3.Irreps],
        irrep_out: Union[str, o3.Irreps] = "1x0e",
    ) -> None:
        super().__init__()
        self.linear = o3.Linear(
            irreps_in=irreps_in,
            irreps_out=irrep_out,
            internal_weights=True,
            shared_weights=True,
        )

    def forward(
        self, x: paddle.Tensor, heads: Optional[paddle.Tensor] = None
    ) -> paddle.Tensor:
        del heads
        return self.linear(x)


class NonLinearReadoutBlock(nn.Layer):
    """Paddle port of ``mace.modules.blocks.NonLinearReadoutBlock``."""

    def __init__(
        self,
        irreps_in: Union[str, o3.Irreps],
        MLP_irreps: Union[str, o3.Irreps],
        gate,
        irrep_out: Union[str, o3.Irreps] = "1x0e",
        num_heads: int = 1,
    ) -> None:
        super().__init__()
        self.hidden_irreps = MLP_irreps
        self.num_heads = num_heads
        self.linear_1 = o3.Linear(
            irreps_in=irreps_in,
            irreps_out=self.hidden_irreps,
            internal_weights=True,
            shared_weights=True,
        )
        self.non_linearity = Activation(irreps_in=self.hidden_irreps, acts=[gate])
        self.linear_2 = o3.Linear(
            irreps_in=self.hidden_irreps,
            irreps_out=irrep_out,
            internal_weights=True,
            shared_weights=True,
        )

    def forward(
        self, x: paddle.Tensor, heads: Optional[paddle.Tensor] = None
    ) -> paddle.Tensor:
        del heads
        x = self.non_linearity(self.linear_1(x))
        return self.linear_2(x)


class AtomicEnergiesBlock(nn.Layer):
    """Paddle port of ``mace.modules.blocks.AtomicEnergiesBlock``."""

    def __init__(self, atomic_energies) -> None:
        super().__init__()
        energies = paddle.to_tensor(
            np.asarray(atomic_energies, dtype=np.float64),
            dtype=paddle.get_default_dtype(),
        )
        self.register_buffer("atomic_energies", energies)  # [n_elements, n_heads]

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        # The checkpoint stores a 1-D [n_elements] tensor, i.e. a single head.
        energies = self.atomic_energies.reshape([-1, 1]).astype(x.dtype)
        return paddle.matmul(x, energies)


class ScaleShiftBlock(nn.Layer):
    """Paddle port of ``mace.modules.blocks.ScaleShiftBlock``."""

    def __init__(self, scale: float, shift: float) -> None:
        super().__init__()
        dtype = paddle.get_default_dtype()
        self.register_buffer("scale", paddle.to_tensor(float(scale), dtype=dtype))
        self.register_buffer("shift", paddle.to_tensor(float(shift), dtype=dtype))

    def forward(self, x: paddle.Tensor, head: paddle.Tensor) -> paddle.Tensor:
        scale = self.scale
        shift = self.shift
        if scale.ndim == 0:
            scale = scale.reshape([1])
        if shift.ndim == 0:
            shift = shift.reshape([1])
        return scale[head] * x + shift[head]
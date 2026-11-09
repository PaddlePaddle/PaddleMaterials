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

"""Paddle port of ``mace.modules.symmetric_contraction``.

The official PyTorch implementation (``mace-torch==0.3.16``) is used as the
reference.  The forward pass below performs exactly the same tensor
contractions as the ``opt_einsum_fx`` optimized modules that are stored in the
released ``mace_mp_0`` checkpoint, but is written with ``tensordot`` /
``bmm`` so that Paddle's second order autograd is available (``paddle.einsum``
has no double backward in Paddle 3.x).

Tensor layout convention (identical to official MACE):
    x : node features, shape ``(num_nodes, num_features, num_ell)``
    y : one-hot node attributes, shape ``(num_nodes, num_elements)``
    output : shape ``(num_nodes, irreps_out.dim)``
"""

import collections
from typing import List
from typing import Optional
from typing import Union

import paddle
from paddle import nn

from ppmat.models.common.e3nn.o3._irreps import Irrep
from ppmat.models.common.e3nn.o3._irreps import Irreps
from ppmat.models.common.e3nn.o3._wigner import wigner_3j

_TP = collections.namedtuple("_TP", "op, args")
_INPUT = collections.namedtuple("_INPUT", "tensor, start, stop")


# Angular-momentum multisets for which ppmat's algorithmic ``wigner_3j``
# (the e3nn 0.3.x construction) and the precomputed table shipped with
# e3nn >= 0.4.4 pick opposite overall signs.  Both implementations are valid
# 3j symbols -- symmetry relations and normalisation agree to ~1e-16 -- the
# overall sign of a single triple is a free gauge choice: the table in
# ``e3nn/o3/constants.pt`` was produced by an eigen-decomposition
# (``_generate_wigner_3j``) fixed only by ``Q[l1, l2, l3] > 0``, so the sign
# has to be reproduced tabulated rather than derived.  The official MACE
# weights were trained with that table convention.  The entries below were
# obtained by an exhaustive comparison for every valid triple with
# ``l <= _GAUGE_LMAX`` (see ``_stage2/log_04_diag_sign_rule.txt``).
_GAUGE_SIGN = {
    (1, 2, 2): -1,
    (1, 3, 3): -1,
    (1, 4, 4): -1,
    (1, 5, 5): -1,
    (1, 6, 6): -1,
    (1, 7, 7): -1,
    (1, 8, 8): -1,
    (1, 9, 9): -1,
    (3, 4, 6): -1,
    (3, 5, 7): -1,
    (3, 6, 8): -1,
    (3, 7, 9): -1,
    (4, 4, 5): -1,
    (4, 5, 6): -1,
    (4, 6, 7): -1,
    (4, 7, 8): -1,
    (4, 8, 9): -1,
    (5, 5, 5): -1,
    (5, 6, 6): -1,
    (5, 7, 7): -1,
    (5, 8, 8): -1,
    (5, 9, 9): -1,
    (6, 6, 9): -1,
    (7, 7, 9): -1,
    (8, 8, 9): -1,
    (9, 9, 9): -1,
}
# ``SymmetricContraction`` with ``correlation=3`` and ``max_ell=3`` (the
# MACE-MP-0 setup) reaches intermediate angular momenta up to 9.
_GAUGE_LMAX = 9


def _e3nn_wigner_3j(l1, l2, l3, dtype=None):
    """Reproduce the ``e3nn >= 0.4`` (table driven) ``wigner_3j`` convention.

    Two differences with ppmat's :func:`wigner_3j` have to be compensated:

    1. e3nn >= 0.4 only stores the sorted triples ``l1 <= l2 <= l3`` and
       builds all other orders with a transpose plus the
       ``(-1) ** (l1 + l2 + l3)`` phase of an odd column permutation.
    2. for the multisets listed in :data:`_GAUGE_SIGN` the two
       implementations choose opposite overall signs.
    """
    order = (l1, l2, l3)
    srt = tuple(sorted(order))
    if srt[-1] > _GAUGE_LMAX:
        raise NotImplementedError(
            f"wigner_3j gauge correction is only verified for l <= {_GAUGE_LMAX}, "
            f"got {order}"
        )
    base = wigner_3j(*srt, dtype=dtype) * _GAUGE_SIGN.get(srt, 1)
    phase = (-1) ** (l1 + l2 + l3)

    def take(read_order):
        """``base`` permuted so that axis ``i`` corresponds to ``read_order[i]``."""
        todo = list(srt)
        perm = []
        for ell in read_order:
            j = todo.index(ell)
            perm.append(j)
            todo[j] = -1
        return base.transpose(perm)

    # ``if`` (not ``elif``): e3nn's branches are independent and the last
    # matching one wins (this happens when two angular momenta are equal).
    if l1 <= l2 <= l3:
        out = take(order)
    if l1 <= l3 <= l2:
        out = take((l1, l3, l2)).transpose([0, 2, 1]) * phase
    if l2 <= l1 <= l3:
        out = take((l2, l1, l3)).transpose([1, 0, 2]) * phase
    if l3 <= l2 <= l1:
        out = take((l3, l2, l1)).transpose([2, 1, 0]) * phase
    if l2 <= l3 <= l1:
        out = take((l2, l3, l1)).transpose([2, 1, 0]).transpose([0, 2, 1])
    if l3 <= l1 <= l2:
        out = take((l3, l1, l2)).transpose([2, 1, 0]).transpose([1, 0, 2])
    return out


def _as_dtype(dtype):
    """Normalize a dtype given as ``None`` / ``str`` / ``paddle.dtype``."""
    if dtype is None:
        dtype = paddle.get_default_dtype()
    if isinstance(dtype, str):
        dtype = getattr(paddle, dtype)
    return dtype


def _wigner_nj(
    irrepss: List[Irreps],
    normalization: str = "component",
    filter_ir_mid=None,
    dtype=None,
):
    """Paddle port of ``mace.tools.cg._wigner_nj``."""
    irrepss = [Irreps(irreps) for irreps in irrepss]
    if filter_ir_mid is not None:
        filter_ir_mid = [Irrep(ir) for ir in filter_ir_mid]
    dtype = _as_dtype(dtype)

    if len(irrepss) == 1:
        (irreps,) = irrepss
        ret = []
        e = paddle.eye(irreps.dim, dtype=dtype)
        i = 0
        for mul, ir in irreps:
            for _ in range(mul):
                sl = slice(i, i + ir.dim)
                ret += [(ir, _INPUT(0, sl.start, sl.stop), e[sl])]
                i += ir.dim
        return ret

    *irrepss_left, irreps_right = irrepss
    ret = []
    for ir_left, path_left, c_left in _wigner_nj(
        irrepss_left,
        normalization=normalization,
        filter_ir_mid=filter_ir_mid,
        dtype=dtype,
    ):
        i = 0
        for mul, ir in irreps_right:
            for ir_out in ir_left * ir:
                if filter_ir_mid is not None and ir_out not in filter_ir_mid:
                    continue

                c = _e3nn_wigner_3j(ir_out.l, ir_left.l, ir.l, dtype=dtype)
                if normalization == "component":
                    c = c * (ir_out.dim**0.5)
                if normalization == "norm":
                    c = c * (ir_left.dim**0.5) * (ir.dim**0.5)

                # equivalent of einsum("jk,ijl->ikl", c_left.flatten(1), c)
                dims_left = [irreps.dim for irreps in irrepss_left]
                prod_left = 1
                for d in dims_left:
                    prod_left *= d
                # ``c`` is (ir_out, ir_left, ir) and the contraction runs over
                # ir_left, so that axis must be brought to the front before
                # flattening (a plain ``reshape`` would interleave the axes).
                tmp = paddle.matmul(
                    c_left.reshape([c_left.shape[0], prod_left]).t(),
                    c.transpose([1, 0, 2]).reshape(
                        [c.shape[1], c.shape[0] * c.shape[2]]
                    ),
                )
                c = tmp.reshape([prod_left, ir_out.dim, ir.dim]).transpose([1, 0, 2])
                c = c.reshape([ir_out.dim] + dims_left + [ir.dim])

                for u in range(mul):
                    sl = slice(i + u * ir.dim, i + (u + 1) * ir.dim)
                    e = paddle.concat(
                        [
                            paddle.zeros(
                                [ir_out.dim] + dims_left + [sl.start], dtype=dtype
                            ),
                            c,
                            paddle.zeros(
                                [ir_out.dim]
                                + dims_left
                                + [irreps_right.dim - sl.stop],
                                dtype=dtype,
                            ),
                        ],
                        axis=-1,
                    )
                    ret += [
                        (
                            ir_out,
                            _TP(
                                op=(ir_left, ir, ir_out),
                                args=(
                                    path_left,
                                    _INPUT(len(irrepss_left), sl.start, sl.stop),
                                ),
                            ),
                            e,
                        )
                    ]
            i += mul * ir.dim
    return sorted(ret, key=lambda item: item[0])


def U_matrix_real(
    irreps_in: Union[str, Irreps],
    irreps_out: Union[str, Irreps],
    correlation: int,
    normalization: str = "component",
    dtype=None,
):
    """Paddle port of ``mace.tools.cg.U_matrix_real`` (non-cuequivariance path)."""
    irreps_out = Irreps(irreps_out)
    irrepss = [Irreps(irreps_in)] * correlation
    dtype = _as_dtype(dtype)

    wigners = _wigner_nj(irrepss, normalization, None, dtype)

    current_ir = wigners[0][0]
    out = []
    stack = []
    last_ir = None

    for ir, _, base_o3 in wigners:
        if ir in irreps_out and ir == current_ir:
            stack.append(base_o3.squeeze().unsqueeze(-1))
            last_ir = current_ir
        elif ir in irreps_out and ir != current_ir:
            if len(stack) != 0:
                out += [last_ir, paddle.concat(stack, axis=-1)]
            stack = [base_o3.squeeze().unsqueeze(-1)]
            current_ir, last_ir = ir, ir
        else:
            current_ir = ir
    if len(stack) != 0:
        out += [last_ir, paddle.concat(stack, axis=-1)]
    else:
        first_dim = irreps_out.dim
        if first_dim != 1:
            size = [first_dim] + [Irreps(irreps_in).dim] * correlation + [1]
        else:
            size = [Irreps(irreps_in).dim] * correlation + [1]
        out = [str(irreps_out)[:-2], paddle.zeros(size, dtype=dtype)]
    return out


def _node_major(tensor, num_nodes: int, num_features: int):
    """Move the trailing ``(num_nodes, num_features)`` axes to the front."""
    ndim = tensor.ndim
    perm = [ndim - 2, ndim - 1] + list(range(ndim - 2))
    return tensor.transpose(perm)


class Contraction(nn.Layer):
    """Paddle port of ``mace.modules.symmetric_contraction.Contraction``."""

    def __init__(
        self,
        irreps_in: Union[str, Irreps],
        irrep_out: Union[str, Irreps],
        correlation: int,
        num_elements: int,
        dtype=None,
    ) -> None:
        super().__init__()

        self.irreps_in = Irreps(irreps_in)
        self.irrep_out = Irreps(irrep_out)
        self.num_features = self.irreps_in.count((0, 1))
        self.coupling_irreps = Irreps([irrep.ir for irrep in self.irreps_in])
        self.correlation = correlation
        dtype = _as_dtype(dtype)

        for nu in range(1, correlation + 1):
            u_matrix = U_matrix_real(
                irreps_in=self.coupling_irreps,
                irreps_out=self.irrep_out,
                correlation=nu,
                dtype=dtype,
            )[-1]
            self.register_buffer(f"U_matrix_{nu}", u_matrix)

        self.weights = nn.ParameterList()
        for i in range(correlation, 0, -1):
            num_params = self.U_tensors(i).shape[-1]
            w = paddle.randn(
                [num_elements, num_params, self.num_features], dtype=dtype
            ) / num_params
            if i == correlation:
                self.weights_max = nn.Parameter(w)
            else:
                self.weights.append(nn.Parameter(w))

    def U_tensors(self, nu: int):
        return getattr(self, f"U_matrix_{nu}")

    def forward(self, x: paddle.Tensor, y: paddle.Tensor):
        out = self._main(self.U_tensors(self.correlation), self.weights_max, x, y)
        for i, weight in enumerate(self.weights):
            c_tensor = self._weighting(
                self.U_tensors(self.correlation - i - 1), weight, y
            )
            c_tensor = c_tensor + out
            out = self._features(c_tensor, x)
        return out.reshape([out.shape[0], -1])

    def _main(self, u_matrix, weight, feats, attrs):
        """Equivalent of the optimized ``graph_opt_main`` module."""
        tmp = paddle.tensordot(attrs, weight, axes=[[1], [0]])  # (B, P, F)
        num_nodes, num_params, num_features = tmp.shape
        num_ell = feats.shape[-1]
        prod = paddle.bmm(
            tmp.transpose([0, 2, 1]).reshape([num_nodes * num_features, num_params, 1]),
            feats.reshape([num_nodes * num_features, 1, num_ell]),
        ).reshape([num_nodes, num_features, num_params, num_ell])
        # u_matrix: [equiv?] + [num_ell] * correlation + [num_params]
        ell_axis = u_matrix.ndim - 2
        param_axis = u_matrix.ndim - 1
        out = paddle.tensordot(
            u_matrix, prod, axes=[[ell_axis, param_axis], [3, 2]]
        )
        return _node_major(out, num_nodes, num_features)

    def _weighting(self, u_matrix, weight, attrs):
        """Equivalent of a ``contractions_weighting`` module."""
        tmp = paddle.tensordot(attrs, weight, axes=[[1], [0]])  # (B, P, F)
        out = paddle.tensordot(u_matrix, tmp, axes=[[u_matrix.ndim - 1], [1]])
        return _node_major(out, tmp.shape[0], tmp.shape[2])

    def _features(self, c_tensor, feats):
        """Equivalent of a ``contractions_features`` module (contract ell axis)."""
        num_mid = c_tensor.ndim - 3
        shape = list(feats.shape[:2]) + [1] * num_mid + [feats.shape[-1]]
        return (c_tensor * feats.reshape(shape)).sum(axis=-1)


class SymmetricContraction(nn.Layer):
    """Paddle port of ``mace.modules.symmetric_contraction.SymmetricContraction``."""

    def __init__(
        self,
        irreps_in: Union[str, Irreps],
        irreps_out: Union[str, Irreps],
        correlation: Union[int, dict],
        num_elements: int,
        dtype=None,
    ) -> None:
        super().__init__()

        self.irreps_in = Irreps(irreps_in)
        self.irreps_out = Irreps(irreps_out)

        if not isinstance(correlation, dict):
            corr = correlation
            correlation = {}
            for irrep_out in self.irreps_out:
                correlation[irrep_out] = corr

        self.contractions = nn.LayerList()
        for irrep_out in self.irreps_out:
            self.contractions.append(
                Contraction(
                    irreps_in=self.irreps_in,
                    irrep_out=Irreps(str(irrep_out.ir)),
                    correlation=correlation[irrep_out],
                    num_elements=num_elements,
                    dtype=dtype,
                )
            )

    def forward(self, x: paddle.Tensor, y: paddle.Tensor):
        outs = [contraction(x, y) for contraction in self.contractions]
        return paddle.concat(outs, axis=-1)
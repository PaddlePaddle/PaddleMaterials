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

# Adapted from https://github.com/divelab/AIRS (OpenMat/HIENet)

"""Eager codegen for ``e3nn.o3.TensorProduct``.

At construction time each instruction is resolved through the
``specialized_code`` branch table to an einsum formula with its divisor
and coefficient, and stored as a picklable execution plan (a list of
NamedTuples). Wigner-3j and triu-index constants are registered as buffers
on first use (only the ones actually referenced). ``forward`` executes the
plan in a fixed order: broadcast -> expand -> flatten -> slice -> per-path
einsum -> path_weight scaling -> per-i_out grouped sum -> concat.
"""

from math import sqrt
from typing import List, NamedTuple, Optional, Tuple

import paddle

from .._irreps import Irreps
from .._wigner import wigner_3j
from ...util import prod

from ._instruction import Instruction


def _sum_tensors(
    xs: List[paddle.Tensor], shape: Tuple[int, ...], like: paddle.Tensor
) -> paddle.Tensor:
    if len(xs) > 0:
        out = xs[0]
        for x in xs[1:]:
            out = out + x
        return out
    return paddle.zeros(shape, dtype=like.dtype)


def _einsum(formula: str, *operands: paddle.Tensor) -> paddle.Tensor:
    """Explicit einsum decomposition over native ops.

    Replaces paddle.einsum on the traced forward path: no env flag, only
    transpose/reshape/matmul/multiply/sum (CINN-friendly primitives with
    double-grad support). Supports the codegen formula subset: explicit
    ``->``, no ellipsis, no repeated label within one operand. Summation
    order differs from paddle.einsum; fp64 agreement is <= 1e-12 on the
    model's formulas.
    """
    lhs, rhs = formula.split("->")
    in_labels = [list(s) for s in lhs.split(",")]
    out_labels = list(rhs)
    assert len(in_labels) == len(operands), formula
    ops = []
    for labels, t in zip(in_labels, operands):
        assert len(labels) == t.ndim, (formula, labels, tuple(t.shape))
        assert len(set(labels)) == len(labels), formula
        ops.append([labels, t])
    out_set = set(out_labels)

    # Pre-step: sum labels that live in exactly one operand and are not in
    # the output (commutes with the rest of the contraction).
    for i, (labels, t) in enumerate(ops):
        others = set()
        for j, (lj, _) in enumerate(ops):
            if j != i:
                others.update(lj)
        drop = [k for k, l in enumerate(labels) if l not in out_set and l not in others]
        if drop:
            t = paddle.sum(t, axis=drop)
            ops[i] = [[l for k, l in enumerate(labels) if k not in drop], t]

    def _align(labels, t, target):
        # permute into target-relative order, then add size-1 axes
        perm = sorted(range(len(labels)), key=lambda k: target.index(labels[k]))
        if perm != list(range(len(perm))):
            t = t.transpose(perm)
        dims = iter(t.shape)
        return t.reshape([next(dims) if l in labels else 1 for l in target])

    while len(ops) > 1:
        # operand count per label: a label shared by >=3 operands cannot be
        # contracted pairwise (its occurrences must meet elementwise and be
        # summed once at the end)
        counts = {}
        for labels, _ in ops:
            for l in labels:
                counts[l] = counts.get(l, 0) + 1
        pair = None
        for i in range(len(ops)):
            for j in range(i + 1, len(ops)):
                shared = set(ops[i][0]) & set(ops[j][0])
                if any(l not in out_set and counts[l] == 2 for l in shared):
                    pair = (i, j)
                    break
            if pair:
                break
        if pair is not None:
            # batched matmul over labels fully consumed by this pair
            i, j = pair
            li, ti = ops[i]
            lj, tj = ops[j]
            c = [l for l in li if l in lj and l not in out_set and counts[l] == 2]
            z = [l for l in li if l in lj and l not in c]
            fa = [l for l in li if l not in lj]
            fb = [l for l in lj if l not in li]
            n, m, k = len(z), len(fa), len(c)
            a = _align(li, ti, z + fa + c)
            sa = list(a.shape)
            a = a.reshape(sa[:n] + [prod(sa[n:n + m]), prod(sa[n + m:])])
            b = _align(lj, tj, z + c + fb)
            sb = list(b.shape)
            b = b.reshape(sb[:n] + [prod(sb[n:n + k]), prod(sb[n + k:])])
            r = paddle.matmul(a, b)
            r = r.reshape(sa[:n] + sa[n:n + m] + sb[n + k:])
            ops[i] = [z + fa + fb, r]
            del ops[j]
        else:
            # elementwise broadcast product (shared labels stay aligned)
            li, ti = ops[0]
            lj, tj = ops[1]
            target = li + [l for l in lj if l not in li]
            r = _align(li, ti, target) * _align(lj, tj, target)
            ops[0] = [target, r]
            del ops[1]

    labels, t = ops[0]
    for ax in range(len(labels) - 1, -1, -1):
        if labels[ax] not in out_set:  # defensive; removed above already
            t = paddle.sum(t, axis=ax)
            labels = labels[:ax] + labels[ax + 1:]
    perm = [labels.index(l) for l in out_labels]
    if perm != list(range(len(perm))):
        t = t.transpose(perm)
    return t


def _irreps_slices(irreps: Irreps) -> List[Tuple[int, int, int, int]]:
    """(start, stop, mul, ir.dim) per irrep."""
    return [
        (sl.start, sl.stop, mul_ir.mul, mul_ir.ir.dim)
        for sl, mul_ir in zip(irreps.slices(), irreps)
    ]


class _TPStep(NamedTuple):
    """Eager execution plan for a single path."""

    i_in1: int
    i_in2: int
    i_out: int
    path_weight: float
    out_dim: int
    m1_dim: int
    m2_dim: int
    has_weight: bool
    w_start: int
    w_len: int
    w_shape: tuple
    xx_mode: str
    formula: Optional[str]
    args: Tuple[str, ...]
    div: Optional[float]
    coef: Optional[float]
    w3j_name: Optional[str]
    triu_mul: Optional[int]


class EagerTensorProductLeftRight(paddle.nn.Layer):
    """Eager implementation of ``tp_forward`` (forward(x1, x2, w)).

    Built by :func:`codegen_tensor_product_left_right`; the plan and the
    constants are fixed at construction time.
    """

    def __init__(
        self,
        irreps_in1: Irreps,
        irreps_in2: Irreps,
        irreps_out: Irreps,
        instructions: List[Instruction],
        shared_weights: bool = False,
        specialized_code: bool = True,
    ) -> None:
        super().__init__()
        instructions = [ins for ins in instructions if 0 not in ins.path_shape]
        self._in1_dim = irreps_in1.dim
        self._in2_dim = irreps_in2.dim
        self._out_dim = irreps_out.dim
        self._shared_weights = shared_weights
        self._weight_numel = sum(
            prod(ins.path_shape) for ins in instructions if ins.has_weight
        )
        self._x1_slices = _irreps_slices(irreps_in1)
        self._x2_slices = _irreps_slices(irreps_in2)
        self._out_groups = [
            (i_out, mul_ir.dim)
            for i_out, mul_ir in enumerate(irreps_out)
            if mul_ir.mul > 0
        ]
        z = "" if shared_weights else "z"
        steps: List[_TPStep] = []
        w3j_names = {}  # name -> (l1, l2, l3), registered on first use
        triu_muls = []  # registered on first use
        flat_weight_index = 0
        for ins in instructions:
            mul_ir_in1 = irreps_in1[ins.i_in1]
            mul_ir_in2 = irreps_in2[ins.i_in2]
            mul_ir_out = irreps_out[ins.i_out]
            assert mul_ir_in1.ir.p * mul_ir_in2.ir.p == mul_ir_out.ir.p
            assert (
                abs(mul_ir_in1.ir.l - mul_ir_in2.ir.l)
                <= mul_ir_out.ir.l
                <= mul_ir_in1.ir.l + mul_ir_in2.ir.l
            )
            if mul_ir_in1.dim == 0 or mul_ir_in2.dim == 0 or mul_ir_out.dim == 0:
                continue
            assert ins.connection_mode in [
                "uvw",
                "uvu",
                "uvv",
                "uuw",
                "uuu",
                "uvuv",
                "uvu<v",
                "u<vw",
            ]
            w_start = w_len = 0
            if ins.has_weight:
                w_start, w_len = flat_weight_index, prod(ins.path_shape)
                flat_weight_index += prod(ins.path_shape)
            l1l2l3 = (mul_ir_in1.ir.l, mul_ir_in2.ir.l, mul_ir_out.ir.l)
            w3j_name = f"_w3j_{l1l2l3[0]}_{l1l2l3[1]}_{l1l2l3[2]}"
            triu_mul = None
            # ---- specialized_code branch table ----
            if ins.connection_mode == "uvw":
                assert ins.has_weight
                if specialized_code and l1l2l3 == (0, 0, 0):
                    formula, args, div, coef = (
                        f"{z}uvw,zu,zv->zw",
                        ("w", "x1r", "x2r"),
                        None,
                        None,
                    )
                elif specialized_code and mul_ir_in1.ir.l == 0:
                    formula, args, div, coef = (
                        f"{z}uvw,zu,zvj->zwj",
                        ("w", "x1r", "x2"),
                        sqrt(mul_ir_out.ir.dim),
                        None,
                    )
                elif specialized_code and mul_ir_in2.ir.l == 0:
                    formula, args, div, coef = (
                        f"{z}uvw,zui,zv->zwi",
                        ("w", "x1", "x2r"),
                        sqrt(mul_ir_out.ir.dim),
                        None,
                    )
                elif specialized_code and mul_ir_out.ir.l == 0:
                    formula, args, div, coef = (
                        f"{z}uvw,zui,zvi->zw",
                        ("w", "x1", "x2"),
                        sqrt(mul_ir_in1.ir.dim),
                        None,
                    )
                else:
                    formula, args, div, coef = (
                        f"{z}uvw,ijk,zuvij->zwk",
                        ("w", "w3j", "xx"),
                        None,
                        None,
                    )
                    w3j_names[w3j_name] = l1l2l3
            if ins.connection_mode == "uvu":
                assert mul_ir_in1.mul == mul_ir_out.mul
                if ins.has_weight:
                    if specialized_code and l1l2l3 == (0, 0, 0):
                        formula, args, div, coef = (
                            f"{z}uv,zu,zv->zu",
                            ("w", "x1r", "x2r"),
                            None,
                            None,
                        )
                    elif specialized_code and mul_ir_in1.ir.l == 0:
                        formula, args, div, coef = (
                            f"{z}uv,zu,zvj->zuj",
                            ("w", "x1r", "x2"),
                            sqrt(mul_ir_out.ir.dim),
                            None,
                        )
                    elif specialized_code and mul_ir_in2.ir.l == 0:
                        formula, args, div, coef = (
                            f"{z}uv,zui,zv->zui",
                            ("w", "x1", "x2r"),
                            sqrt(mul_ir_out.ir.dim),
                            None,
                        )
                    elif specialized_code and mul_ir_out.ir.l == 0:
                        formula, args, div, coef = (
                            f"{z}uv,zui,zvi->zu",
                            ("w", "x1", "x2"),
                            sqrt(mul_ir_in1.ir.dim),
                            None,
                        )
                    else:
                        formula, args, div, coef = (
                            f"{z}uv,ijk,zuvij->zuk",
                            ("w", "w3j", "xx"),
                            None,
                            None,
                        )
                        w3j_names[w3j_name] = l1l2l3
                else:
                    formula, args, div, coef = (
                        "ijk,zuvij->zuk",
                        ("w3j", "xx"),
                        None,
                        None,
                    )
                    w3j_names[w3j_name] = l1l2l3
            if ins.connection_mode == "uvv":
                assert mul_ir_in2.mul == mul_ir_out.mul
                if ins.has_weight:
                    if specialized_code and l1l2l3 == (0, 0, 0):
                        formula, args, div, coef = (
                            f"{z}uv,zu,zv->zv",
                            ("w", "x1r", "x2r"),
                            None,
                            None,
                        )
                    elif specialized_code and mul_ir_in1.ir.l == 0:
                        formula, args, div, coef = (
                            f"{z}uv,zu,zvj->zvj",
                            ("w", "x1r", "x2"),
                            sqrt(mul_ir_out.ir.dim),
                            None,
                        )
                    elif specialized_code and mul_ir_in2.ir.l == 0:
                        formula, args, div, coef = (
                            f"{z}uv,zui,zv->zvi",
                            ("w", "x1", "x2r"),
                            sqrt(mul_ir_out.ir.dim),
                            None,
                        )
                    elif specialized_code and mul_ir_out.ir.l == 0:
                        formula, args, div, coef = (
                            f"{z}uv,zui,zvi->zv",
                            ("w", "x1", "x2"),
                            sqrt(mul_ir_in1.ir.dim),
                            None,
                        )
                    else:
                        formula, args, div, coef = (
                            f"{z}uv,ijk,zuvij->zvk",
                            ("w", "w3j", "xx"),
                            None,
                            None,
                        )
                        w3j_names[w3j_name] = l1l2l3
                elif specialized_code and l1l2l3 == (0, 0, 0):
                    formula, args, div, coef = (
                        "zu,zv->zv",
                        ("x1r", "x2r"),
                        None,
                        None,
                    )
                elif specialized_code and mul_ir_in1.ir.l == 0:
                    formula, args, div, coef = (
                        "zu,zvj->zvj",
                        ("x1r", "x2"),
                        sqrt(mul_ir_out.ir.dim),
                        None,
                    )
                elif specialized_code and mul_ir_in2.ir.l == 0:
                    formula, args, div, coef = (
                        "zui,zv->zvi",
                        ("x1", "x2r"),
                        sqrt(mul_ir_out.ir.dim),
                        None,
                    )
                elif specialized_code and mul_ir_out.ir.l == 0:
                    formula, args, div, coef = (
                        "zui,zvi->zv",
                        ("x1", "x2"),
                        sqrt(mul_ir_in1.ir.dim),
                        None,
                    )
                else:
                    formula, args, div, coef = (
                        "ijk,zuvij->zvk",
                        ("w3j", "xx"),
                        None,
                        None,
                    )
                    w3j_names[w3j_name] = l1l2l3
            if ins.connection_mode == "uuw":
                assert mul_ir_in1.mul == mul_ir_in2.mul
                if ins.has_weight:
                    if specialized_code and l1l2l3 == (0, 0, 0):
                        formula, args, div, coef = (
                            f"{z}uw,zu,zu->zw",
                            ("w", "x1r", "x2r"),
                            None,
                            None,
                        )
                    elif specialized_code and mul_ir_in1.ir.l == 0:
                        formula, args, div, coef = (
                            f"{z}uw,zu,zuj->zwj",
                            ("w", "x1r", "x2"),
                            sqrt(mul_ir_out.ir.dim),
                            None,
                        )
                    elif specialized_code and mul_ir_in2.ir.l == 0:
                        formula, args, div, coef = (
                            f"{z}uw,zui,zu->zwi",
                            ("w", "x1", "x2r"),
                            sqrt(mul_ir_out.ir.dim),
                            None,
                        )
                    elif specialized_code and mul_ir_out.ir.l == 0:
                        formula, args, div, coef = (
                            f"{z}uw,zui,zui->zw",
                            ("w", "x1", "x2"),
                            sqrt(mul_ir_in1.ir.dim),
                            None,
                        )
                    else:
                        formula, args, div, coef = (
                            f"{z}uw,ijk,zuij->zwk",
                            ("w", "w3j", "xx"),
                            None,
                            None,
                        )
                        w3j_names[w3j_name] = l1l2l3
                else:
                    assert mul_ir_out.mul == 1
                    formula, args, div, coef = (
                        "ijk,zuij->zk",
                        ("w3j", "xx"),
                        None,
                        None,
                    )
                    w3j_names[w3j_name] = l1l2l3
            if ins.connection_mode == "uuu":
                assert mul_ir_in1.mul == mul_ir_in2.mul == mul_ir_out.mul
                if ins.has_weight:
                    if specialized_code and l1l2l3 == (0, 0, 0):
                        formula, args, div, coef = (
                            f"{z}u,zu,zu->zu",
                            ("w", "x1r", "x2r"),
                            None,
                            None,
                        )
                    elif specialized_code and l1l2l3 == (1, 1, 1):
                        formula, args, div, coef = (
                            f"{z}u,zui->zui",
                            ("w", "cross"),
                            sqrt(2 * 3),
                            None,
                        )
                    elif specialized_code and mul_ir_in1.ir.l == 0:
                        formula, args, div, coef = (
                            f"{z}u,zu,zuj->zuj",
                            ("w", "x1r", "x2"),
                            sqrt(mul_ir_out.ir.dim),
                            None,
                        )
                    elif specialized_code and mul_ir_in2.ir.l == 0:
                        formula, args, div, coef = (
                            f"{z}u,zui,zu->zui",
                            ("w", "x1", "x2r"),
                            sqrt(mul_ir_out.ir.dim),
                            None,
                        )
                    elif specialized_code and mul_ir_out.ir.l == 0:
                        formula, args, div, coef = (
                            f"{z}u,zui,zui->zu",
                            ("w", "x1", "x2"),
                            sqrt(mul_ir_in1.ir.dim),
                            None,
                        )
                    else:
                        formula, args, div, coef = (
                            f"{z}u,ijk,zuij->zuk",
                            ("w", "w3j", "xx"),
                            None,
                            None,
                        )
                        w3j_names[w3j_name] = l1l2l3
                elif specialized_code and l1l2l3 == (0, 0, 0):
                    formula, args, div, coef = (
                        "zu,zu->zu",
                        ("x1r", "x2r"),
                        None,
                        None,
                    )
                elif specialized_code and l1l2l3 == (1, 1, 1):
                    formula, args, div, coef = (
                        None,
                        ("cross",),
                        None,
                        1.0 / sqrt(2 * 3),
                    )
                elif specialized_code and mul_ir_in1.ir.l == 0:
                    formula, args, div, coef = (
                        "zu,zuj->zuj",
                        ("x1r", "x2"),
                        sqrt(mul_ir_out.ir.dim),
                        None,
                    )
                elif specialized_code and mul_ir_in2.ir.l == 0:
                    formula, args, div, coef = (
                        "zui,zu->zui",
                        ("x1", "x2r"),
                        sqrt(mul_ir_out.ir.dim),
                        None,
                    )
                elif specialized_code and mul_ir_out.ir.l == 0:
                    formula, args, div, coef = (
                        "zui,zui->zu",
                        ("x1", "x2"),
                        sqrt(mul_ir_in1.ir.dim),
                        None,
                    )
                else:
                    formula, args, div, coef = (
                        "ijk,zuij->zuk",
                        ("w3j", "xx"),
                        None,
                        None,
                    )
                    w3j_names[w3j_name] = l1l2l3
            if ins.connection_mode == "uvuv":
                assert mul_ir_in1.mul * mul_ir_in2.mul == mul_ir_out.mul
                if ins.has_weight:
                    formula, args, div, coef = (
                        f"{z}uv,ijk,zuvij->zuvk",
                        ("w", "w3j", "xx"),
                        None,
                        None,
                    )
                else:
                    formula, args, div, coef = (
                        "ijk,zuvij->zuvk",
                        ("w3j", "xx"),
                        None,
                        None,
                    )
                w3j_names[w3j_name] = l1l2l3
            if ins.connection_mode == "uvu<v":
                assert mul_ir_in1.mul == mul_ir_in2.mul
                assert mul_ir_in1.mul * (mul_ir_in1.mul - 1) // 2 == mul_ir_out.mul
                triu_mul = mul_ir_in1.mul
                if triu_mul not in triu_muls:
                    triu_muls.append(triu_mul)
                if ins.has_weight:
                    formula, args, div, coef = (
                        f"{z}w,ijk,zwij->zwk",
                        ("w", "w3j", "xx"),
                        None,
                        None,
                    )
                else:
                    formula, args, div, coef = (
                        "ijk,zwij->zwk",
                        ("w3j", "xx"),
                        None,
                        None,
                    )
                w3j_names[w3j_name] = l1l2l3
            if ins.connection_mode == "u<vw":
                assert mul_ir_in1.mul == mul_ir_in2.mul
                assert ins.has_weight
                triu_mul = mul_ir_in1.mul
                if triu_mul not in triu_muls:
                    triu_muls.append(triu_mul)
                formula, args, div, coef = (
                    f"{z}qw,ijk,zqij->zwk",
                    ("w", "w3j", "xx"),
                    None,
                    None,
                )
                w3j_names[w3j_name] = l1l2l3
            steps.append(
                _TPStep(
                    i_in1=ins.i_in1,
                    i_in2=ins.i_in2,
                    i_out=ins.i_out,
                    path_weight=ins.path_weight,
                    out_dim=mul_ir_out.dim,
                    m1_dim=mul_ir_in1.dim,
                    m2_dim=mul_ir_in2.dim,
                    has_weight=ins.has_weight,
                    w_start=w_start,
                    w_len=w_len,
                    w_shape=ins.path_shape,
                    xx_mode=ins.connection_mode[:2],
                    formula=formula,
                    args=args,
                    div=div,
                    coef=coef,
                    w3j_name=w3j_name if w3j_name in w3j_names else None,
                    triu_mul=triu_mul,
                )
            )
        self._steps = steps
        for name, (l1, l2, l3) in w3j_names.items():
            self.register_buffer(name, wigner_3j(l1, l2, l3))
        for mul in triu_muls:
            self.register_buffer(
                f"_triu_indices_{mul}", paddle.triu_indices(row=mul, col=mul, offset=1)
            )

    def forward(self, x1s, x2s, weights):
        if self._shared_weights:
            output_shape = list(
                paddle.broadcast_tensors([x1s[..., :1], x2s[..., :1]])[0].shape[:-1]
            )
        else:
            output_shape = list(
                paddle.broadcast_tensors(
                    [x1s[..., :1], x2s[..., :1], weights[..., :1]]
                )[0].shape[:-1]
            )
        if len(self._steps) == 0:
            return paddle.zeros(output_shape + [self._out_dim], dtype=x1s.dtype)
        bc_shape = output_shape + [-1]
        x1s = x1s.expand(bc_shape).reshape([-1, self._in1_dim])
        x2s = x2s.expand(bc_shape).reshape([-1, self._in2_dim])
        if not self._shared_weights:
            weights = weights.expand(bc_shape)
        if self._weight_numel > 0:
            weights = weights.reshape([-1, self._weight_numel])
        batch_numel = x1s.shape[0]
        if len(self._x1_slices) == 1:
            _, _, mul, dim = self._x1_slices[0]
            x1_list = [x1s.reshape([batch_numel, mul, dim])]
        else:
            x1_list = [
                x1s[:, a:b].reshape([batch_numel, mul, dim])
                for (a, b, mul, dim) in self._x1_slices
            ]
        if len(self._x2_slices) == 1:
            _, _, mul, dim = self._x2_slices[0]
            x2_list = [x2s.reshape([batch_numel, mul, dim])]
        else:
            x2_list = [
                x2s[:, a:b].reshape([batch_numel, mul, dim])
                for (a, b, mul, dim) in self._x2_slices
            ]
        xx_dict = dict()
        outputs = []
        for step in self._steps:
            x1 = x1_list[step.i_in1]
            x2 = x2_list[step.i_in2]
            env = {"x1": x1, "x2": x2}
            if step.has_weight:
                env["w"] = weights[:, step.w_start : step.w_start + step.w_len].reshape(
                    (() if self._shared_weights else (-1,)) + tuple(step.w_shape)
                )
            if "x1r" in step.args:
                env["x1r"] = x1.reshape([batch_numel, step.m1_dim])
            if "x2r" in step.args:
                env["x2r"] = x2.reshape([batch_numel, step.m2_dim])
            key = (step.i_in1, step.i_in2, step.xx_mode)
            if key not in xx_dict:
                if step.xx_mode == "uu":
                    xx_dict[key] = _einsum("zui,zuj->zuij", x1, x2)
                else:
                    xx_dict[key] = _einsum("zui,zvj->zuvij", x1, x2)
            xx = xx_dict[key]
            if step.triu_mul is not None:
                i = getattr(self, f"_triu_indices_{step.triu_mul}")
                xx = xx[:, i[0], i[1]]
            env["xx"] = xx
            if "w3j" in step.args:
                env["w3j"] = getattr(self, step.w3j_name)
            if "cross" in step.args:
                env["cross"] = paddle.cross(x1, x2, axis=2)
            if step.formula is None:
                result = env[step.args[0]]
            else:
                result = _einsum(step.formula, *[env[a] for a in step.args])
            if step.div is not None:
                result = result / step.div
            if step.coef is not None:
                result = result * step.coef
            result = step.path_weight * result
            outputs += [result.reshape([batch_numel, step.out_dim])]
        grouped = [
            _sum_tensors(
                [
                    out
                    for step, out in zip(self._steps, outputs)
                    if step.i_out == i_out
                ],
                shape=(batch_numel, dim),
                like=x1s,
            )
            for i_out, dim in self._out_groups
        ]
        if len(grouped) > 1:
            outputs = paddle.cat(grouped, dim=1)
        else:
            outputs = grouped[0]
        return outputs.reshape(output_shape + [self._out_dim])


class _TPRightStep(NamedTuple):
    """Execution plan for one ``tp_right`` path."""

    i_in1: int
    i_in2: int
    i_out: int
    path_weight: float
    m1_dim: int
    m2_dim: int
    out_dim: int
    has_weight: bool
    w_start: int
    w_len: int
    w_shape: tuple
    formula: str
    args: Tuple[str, ...]
    div: Optional[float]
    w3j_name: Optional[str]
    e1_size: int
    e2_size: int
    i1_size: int
    s2ones_size: int


class EagerTensorProductRight(paddle.nn.Layer):
    """Eager implementation of ``tp_right``: forward(x2, w), returns ``... x irreps_in1.dim x irreps_out.dim``."""

    def __init__(
        self,
        irreps_in1: Irreps,
        irreps_in2: Irreps,
        irreps_out: Irreps,
        instructions: List[Instruction],
        shared_weights: bool = False,
        specialized_code: bool = True,
    ) -> None:
        super().__init__()
        instructions = [ins for ins in instructions if 0 not in ins.path_shape]
        self._in1_dim = irreps_in1.dim
        self._in2_dim = irreps_in2.dim
        self._out_dim = irreps_out.dim
        self._shared_weights = shared_weights
        self._weight_numel = sum(
            prod(ins.path_shape) for ins in instructions if ins.has_weight
        )
        self._x2_slices = _irreps_slices(irreps_in2)
        self._out_groups = [
            (i_out, mul_ir.dim)
            for i_out, mul_ir in enumerate(irreps_out)
            if mul_ir.mul > 0
        ]
        self._in1_groups = [
            (i_in1, mul_ir.dim)
            for i_in1, mul_ir in enumerate(irreps_in1)
            if mul_ir.mul > 0
        ]
        z = "" if shared_weights else "z"
        steps: List[_TPRightStep] = []
        w3j_names = {}
        flat_weight_index = 0
        for ins in instructions:
            mul_ir_in1 = irreps_in1[ins.i_in1]
            mul_ir_in2 = irreps_in2[ins.i_in2]
            mul_ir_out = irreps_out[ins.i_out]
            assert mul_ir_in1.ir.p * mul_ir_in2.ir.p == mul_ir_out.ir.p
            assert (
                abs(mul_ir_in1.ir.l - mul_ir_in2.ir.l)
                <= mul_ir_out.ir.l
                <= mul_ir_in1.ir.l + mul_ir_in2.ir.l
            )
            if mul_ir_in1.dim == 0 or mul_ir_in2.dim == 0 or mul_ir_out.dim == 0:
                continue
            assert ins.connection_mode in [
                "uvw",
                "uvu",
                "uvv",
                "uuw",
                "uuu",
                "uvuv",
                "uvu<v",
                "u<vw",
            ]
            w_start = w_len = 0
            if ins.has_weight:
                w_start, w_len = flat_weight_index, prod(ins.path_shape)
                flat_weight_index += prod(ins.path_shape)
            l1l2l3 = (mul_ir_in1.ir.l, mul_ir_in2.ir.l, mul_ir_out.ir.l)
            w3j_name = f"_w3j_{l1l2l3[0]}_{l1l2l3[1]}_{l1l2l3[2]}"
            # ---- specialized_code branch table ----
            if ins.connection_mode == "uvw":
                assert ins.has_weight
                if specialized_code and l1l2l3 == (0, 0, 0):
                    formula, args, div = (f"{z}uvw,zv->zuw", ("w", "x2r"), None)
                elif specialized_code and mul_ir_in1.ir.l == 0:
                    formula, args, div = (
                        f"{z}uvw,zvi->zuwi",
                        ("w", "x2"),
                        sqrt(mul_ir_out.ir.dim),
                    )
                elif specialized_code and mul_ir_in2.ir.l == 0:
                    formula, args, div = (
                        f"{z}uvw,ij,zv->zuiwj",
                        ("w", "i1", "x2r"),
                        sqrt(mul_ir_out.ir.dim),
                    )
                elif specialized_code and mul_ir_out.ir.l == 0:
                    formula, args, div = (
                        f"{z}uvw,zvi->zuiw",
                        ("w", "x2"),
                        sqrt(mul_ir_in1.ir.dim),
                    )
                else:
                    formula, args, div = (
                        f"{z}uvw,ijk,zvj->zuiwk",
                        ("w", "w3j", "x2"),
                        None,
                    )
                    w3j_names[w3j_name] = l1l2l3
            if ins.connection_mode == "uvu":
                assert mul_ir_in1.mul == mul_ir_out.mul
                if ins.has_weight:
                    if specialized_code and l1l2l3 == (0, 0, 0):
                        formula, args, div = (
                            f"{z}uv,uw,zv->zuw",
                            ("w", "e1", "x2r"),
                            None,
                        )
                    elif specialized_code and mul_ir_in1.ir.l == 0:
                        formula, args, div = (
                            f"{z}uv,uw,zvi->zuwi",
                            ("w", "e1", "x2"),
                            sqrt(mul_ir_out.ir.dim),
                        )
                    elif specialized_code and mul_ir_in2.ir.l == 0:
                        formula, args, div = (
                            f"{z}uv,ij,uw,zv->zuiwj",
                            ("w", "i1", "e1", "x2r"),
                            sqrt(mul_ir_out.ir.dim),
                        )
                    elif specialized_code and mul_ir_out.ir.l == 0:
                        formula, args, div = (
                            f"{z}uv,uw,zvi->zuiw",
                            ("w", "e1", "x2"),
                            sqrt(mul_ir_in1.ir.dim),
                        )
                    else:
                        formula, args, div = (
                            f"{z}uv,ijk,uw,zvj->zuiwk",
                            ("w", "w3j", "e1", "x2"),
                            None,
                        )
                        w3j_names[w3j_name] = l1l2l3
                else:
                    formula, args, div = (
                        "ijk,uw,zvj->zuiwk",
                        ("w3j", "e1", "x2"),
                        None,
                    )
                    w3j_names[w3j_name] = l1l2l3
            if ins.connection_mode == "uvv":
                assert mul_ir_in2.mul == mul_ir_out.mul
                if ins.has_weight:
                    if specialized_code and l1l2l3 == (0, 0, 0):
                        formula, args, div = (
                            f"{z}uv,vw,zv->zuw",
                            ("w", "e2", "x2r"),
                            None,
                        )
                    elif specialized_code and mul_ir_in1.ir.l == 0:
                        formula, args, div = (
                            f"{z}uv,vw,zvi->zuwi",
                            ("w", "e2", "x2"),
                            sqrt(mul_ir_out.ir.dim),
                        )
                    elif specialized_code and mul_ir_in2.ir.l == 0:
                        formula, args, div = (
                            f"{z}uv,ij,vw,zv->zuiwj",
                            ("w", "i1", "e2", "x2r"),
                            sqrt(mul_ir_out.ir.dim),
                        )
                    elif specialized_code and mul_ir_out.ir.l == 0:
                        formula, args, div = (
                            f"{z}uv,vw,zvi->zuiw",
                            ("w", "e2", "x2"),
                            sqrt(mul_ir_in1.ir.dim),
                        )
                    else:
                        formula, args, div = (
                            f"{z}uv,ijk,zvj->zuivk",
                            ("w", "w3j", "x2"),
                            None,
                        )
                        w3j_names[w3j_name] = l1l2l3
                else:
                    formula, args, div = (
                        "u,ijk,zvj->zuivk",
                        ("s2ones", "w3j", "x2"),
                        None,
                    )
                    w3j_names[w3j_name] = l1l2l3
            if ins.connection_mode == "uuw":
                assert mul_ir_in1.mul == mul_ir_in2.mul
                if ins.has_weight:
                    formula, args, div = (
                        f"{z}uw,ijk,zuj->zuiwk",
                        ("w", "w3j", "x2"),
                        None,
                    )
                else:
                    assert mul_ir_out.mul == 1
                    formula, args, div = ("ijk,zuj->zuik", ("w3j", "x2"), None)
                w3j_names[w3j_name] = l1l2l3
            if ins.connection_mode == "uuu":
                assert mul_ir_in1.mul == mul_ir_in2.mul == mul_ir_out.mul
                if ins.has_weight:
                    if specialized_code and l1l2l3 == (0, 0, 0):
                        formula, args, div = (
                            f"{z}u,uw,zu->zuw",
                            ("w", "e2", "x2r"),
                            None,
                        )
                    elif specialized_code and l1l2l3 == (1, 1, 1):
                        formula, args, div = (
                            f"{z}u,ijk,uw,zuj->zuiwk",
                            ("w", "w3j", "e1", "x2"),
                            None,
                        )
                    elif specialized_code and mul_ir_in1.ir.l == 0:
                        formula, args, div = (
                            f"{z}u,uw,zui->zuwi",
                            ("w", "e2", "x2"),
                            sqrt(mul_ir_out.ir.dim),
                        )
                    elif specialized_code and mul_ir_in2.ir.l == 0:
                        formula, args, div = (
                            f"{z}u,ij,uw,zu->zuiwj",
                            ("w", "i1", "e2", "x2r"),
                            sqrt(mul_ir_out.ir.dim),
                        )
                    elif specialized_code and mul_ir_out.ir.l == 0:
                        formula, args, div = (
                            f"{z}u,uw,zui->zuiw",
                            ("w", "e2", "x2"),
                            sqrt(mul_ir_in1.ir.dim),
                        )
                    else:
                        formula, args, div = (
                            f"{z}u,ijk,uw,zuj->zuiwk",
                            ("w", "w3j", "e1", "x2"),
                            None,
                        )
                        w3j_names[w3j_name] = l1l2l3
                elif specialized_code and l1l2l3 == (0, 0, 0):
                    formula, args, div = ("uw,zu->zuw", ("e2", "x2r"), None)
                elif specialized_code and l1l2l3 == (1, 1, 1):
                    formula, args, div = (
                        "ijk,uw,zuj->zuiwk",
                        ("w3j", "e1", "x2"),
                        None,
                    )
                    w3j_names[w3j_name] = l1l2l3
                elif specialized_code and mul_ir_in1.ir.l == 0:
                    formula, args, div = (
                        "uw,zui->zuwi",
                        ("e2", "x2"),
                        sqrt(mul_ir_out.ir.dim),
                    )
                elif specialized_code and mul_ir_in2.ir.l == 0:
                    formula, args, div = (
                        "ij,uw,zu->zuiwj",
                        ("i1", "e2", "x2r"),
                        sqrt(mul_ir_out.ir.dim),
                    )
                elif specialized_code and mul_ir_out.ir.l == 0:
                    formula, args, div = (
                        "uw,zui->zuiw",
                        ("e2", "x2"),
                        sqrt(mul_ir_in1.ir.dim),
                    )
                else:
                    formula, args, div = (
                        "ijk,uw,zuj->zuiwk",
                        ("w3j", "e1", "x2"),
                        None,
                    )
                    w3j_names[w3j_name] = l1l2l3
            if ins.connection_mode == "uvuv":
                assert mul_ir_in1.mul * mul_ir_in2.mul == mul_ir_out.mul
                if ins.has_weight:
                    formula, args, div = (
                        f"{z}uv,ijk,uw,zvj->zuiwvk",
                        ("w", "w3j", "e1", "x2"),
                        None,
                    )
                else:
                    formula, args, div = (
                        "ijk,uw,zvj->zuiwvk",
                        ("w3j", "e1", "x2"),
                        None,
                    )
                w3j_names[w3j_name] = l1l2l3
            if ins.connection_mode == "uvu<v":
                raise NotImplementedError
            if ins.connection_mode == "u<vw":
                raise NotImplementedError
            steps.append(
                _TPRightStep(
                    i_in1=ins.i_in1,
                    i_in2=ins.i_in2,
                    i_out=ins.i_out,
                    path_weight=ins.path_weight,
                    m1_dim=mul_ir_in1.dim,
                    m2_dim=mul_ir_in2.dim,
                    out_dim=mul_ir_out.dim,
                    has_weight=ins.has_weight,
                    w_start=w_start,
                    w_len=w_len,
                    w_shape=ins.path_shape,
                    formula=formula,
                    args=args,
                    div=div,
                    w3j_name=w3j_name if w3j_name in w3j_names else None,
                    e1_size=mul_ir_in1.mul,
                    e2_size=mul_ir_in2.mul,
                    i1_size=mul_ir_in1.ir.dim,
                    s2ones_size=mul_ir_in1.mul,
                )
            )
        self._steps = steps
        for name, (l1, l2, l3) in w3j_names.items():
            self.register_buffer(name, wigner_3j(l1, l2, l3))

    def forward(self, x2s, weights):
        if self._shared_weights:
            output_shape = list(x2s.shape[:-1])
        else:
            output_shape = list(
                paddle.broadcast_tensors([x2s[..., 0], weights[..., 0]])[0].shape
            )
        if len(self._steps) == 0:
            return paddle.zeros(
                output_shape + [self._in1_dim, self._out_dim], dtype=x2s.dtype
            )
        if not self._shared_weights:
            x2s = x2s.broadcast_to(output_shape + [-1])
            weights = weights.broadcast_to(output_shape + [-1])
        output_shape = output_shape + [self._in1_dim, self._out_dim]
        x2s = x2s.reshape([-1, self._in2_dim])
        batch_numel = x2s.shape[0]
        if self._weight_numel > 0:
            weights = weights.reshape([-1, self._weight_numel])
        if len(self._x2_slices) == 1:
            _, _, mul, dim = self._x2_slices[0]
            x2_list = [x2s.reshape([batch_numel, mul, dim])]
        else:
            x2_list = [
                x2s[:, a:b].reshape([batch_numel, mul, dim])
                for (a, b, mul, dim) in self._x2_slices
            ]
        eyes = {}

        def _eye(n):
            if n not in eyes:
                eyes[n] = paddle.eye(n, dtype=x2s.dtype)
            return eyes[n]

        ones = {}

        def _ones(n):
            if n not in ones:
                ones[n] = paddle.ones([n], dtype=x2s.dtype)
            return ones[n]

        outputs = []
        for step in self._steps:
            x2 = x2_list[step.i_in2]
            env = {"x2": x2}
            if step.has_weight:
                env["w"] = weights[:, step.w_start : step.w_start + step.w_len].reshape(
                    (() if self._shared_weights else (-1,)) + tuple(step.w_shape)
                )
            if "x2r" in step.args:
                env["x2r"] = x2.reshape([batch_numel, step.m2_dim])
            if "w3j" in step.args:
                env["w3j"] = getattr(self, step.w3j_name)
            if "e1" in step.args:
                env["e1"] = _eye(step.e1_size)
            if "e2" in step.args:
                env["e2"] = _eye(step.e2_size)
            if "i1" in step.args:
                env["i1"] = _eye(step.i1_size)
            if "s2ones" in step.args:
                env["s2ones"] = _ones(step.s2ones_size)
            result = _einsum(step.formula, *[env[a] for a in step.args])
            if step.div is not None:
                result = result / step.div
            result = step.path_weight * result
            outputs += [result.reshape([batch_numel, step.m1_dim, step.out_dim])]
        per_in1 = []
        for i_in1, m1_dim in self._in1_groups:
            per_out = [
                _sum_tensors(
                    [
                        out
                        for step, out in zip(self._steps, outputs)
                        if (step.i_in1, step.i_out) == (i_in1, i_out)
                    ],
                    shape=(batch_numel, m1_dim, out_dim),
                    like=x2s,
                )
                for i_out, out_dim in self._out_groups
            ]
            per_in1.append(paddle.cat(per_out, dim=2))
        if len(per_in1) > 1:
            outputs = paddle.cat(per_in1, dim=1)
        else:
            outputs = per_in1[0]
        return outputs.reshape(output_shape)


class EagerRightPassthrough(paddle.nn.Layer):
    """Placeholder for ``compile_right=False``: forward(x2, w) -> x2."""

    def forward(self, x2, w):
        return x2


class EagerNotCompiledLeftRight(paddle.nn.Layer):
    """Runtime-error stub for ``compile_left_right=False``."""

    _MSG = (
        "`left_right` method is not compiled, set `compile_left_right` to True "
        "when creating the TensorProduct"
    )

    def forward(self, x1, x2, w):
        raise RuntimeError(self._MSG)


def codegen_tensor_product_left_right(
    irreps_in1: Irreps,
    irreps_in2: Irreps,
    irreps_out: Irreps,
    instructions: List[Instruction],
    shared_weights: bool = False,
    specialized_code: bool = True,
    optimize_einsums: bool = True,
) -> paddle.nn.Layer:
    """Build the eager ``tp_forward`` layer.

    ``optimize_einsums`` is kept for signature compatibility; the eager
    layer executes the source formulas directly.
    """
    return EagerTensorProductLeftRight(
        irreps_in1, irreps_in2, irreps_out, instructions, shared_weights, specialized_code
    )


def codegen_tensor_product_right(
    irreps_in1: Irreps,
    irreps_in2: Irreps,
    irreps_out: Irreps,
    instructions: List[Instruction],
    shared_weights: bool = False,
    specialized_code: bool = True,
    optimize_einsums: bool = True,
) -> paddle.nn.Layer:
    """Build the eager ``tp_right`` layer."""
    return EagerTensorProductRight(
        irreps_in1, irreps_in2, irreps_out, instructions, shared_weights, specialized_code
    )

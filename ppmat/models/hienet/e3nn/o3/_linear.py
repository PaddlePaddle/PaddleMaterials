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

from typing import List, NamedTuple, Optional, Tuple, Union

from .. import get_optimization_defaults
import paddle
from ._irreps import Irreps
from ..util import prod
from ..util.codegen import CodeGenMixin
from ..util.jit import compile_mode

from ._tensor_product._codegen import _einsum, _sum_tensors


class Instruction(NamedTuple):
    i_in: int
    i_out: int
    path_shape: tuple
    path_weight: float


class LinearSlices(NamedTuple):
    slice_1D: slice
    shape_2D: tuple


@compile_mode("script")
class Linear(CodeGenMixin, paddle.nn.Module):
    """Linear operation equivariant to :math:`O(3)`

    Notes
    -----
        `e3nn.o3.Linear` objects created with different partitionings of the same irreps, such as ``Linear("10x0e", "0e")``
        and ``Linear("3x0e + 7x0e", "0e")``, are *not* equivalent: the second module has more instructions, which affects
        normalization. In a rough sense:

            Linear("10x0e", "0e") = normalization_coeff_0 * W_0 @ input
            Linear("3x0e + 7x0e", "0e") = normalization_coeff_1 * W_1 @ input[:3] + normalization_coeff_2 * W_2 @ input[3:]

        To make them equivalent, simplify ``irreps_in`` before constructing network modules:

            o3.Irreps("3x0e + 7x0e").simplify()  # => 10x0e


    Parameters
    ----------
    irreps_in : `e3nn.o3.Irreps`
        representation of the input

    irreps_out : `e3nn.o3.Irreps`
        representation of the output

    internal_weights : bool
        whether the `e3nn.o3.Linear` should store its own weights. Defaults to ``True`` unless ``shared_weights`` is
        explicitly set to ``False``, for consistancy with `e3nn.o3.TensorProduct`.

    shared_weights : bool
        whether the `e3nn.o3.Linear` should be weighted individually for each input in a batch. Defaults to ``True``.
        Cannot be ``False`` if ``internal_weights`` is ``True``.

    instructions : list of 2-tuples, optional
        list of tuples ``(i_in, i_out)`` indicating which irreps in ``irreps_in`` should contribute to which irreps in
        ``irreps_out``. If ``None`` (the default), all allowable instructions will be created: every ``(i_in, i_out)`` such
        that ``irreps_in[i_in].ir == irreps_out[i_out].ir``.

    biases : list of bool, optional
        indicates for each element of ``irreps_out`` if it has a bias. By default there is no bias.
        If ``biases=True`` it gives bias to all scalars (l=0 and p=1).

    Attributes
    ----------
    weight_numel : int
        the size of the weights for this `e3nn.o3.Linear`

    Examples
    --------
    Linearly combines 4 scalars into 8 scalars and 16 vectors into 8 vectors.

    >>> lin = Linear("4x0e+16x1o", "8x0e+8x1o")
    >>> lin.weight_numel
    160

    Create a "block sparse" linear that does not combine two different groups of scalars;
    note that the number of weights is 4*4 + 3*3 = 25:

    >>> lin = Linear("4x0e + 3x0e", "4x0e + 3x0e", instructions=[(0, 0), (1, 1)])
    >>> lin.weight_numel
    25

    Be careful: because they have different instructions, the following two operations are not normalized in the same way,
    even though they contain all the same "connections":

    >>> lin1 = Linear("10x0e", "0e")
    >>> lin2 = Linear("3x0e + 7x0e", "0e")
    >>> lin1.weight_numel == lin2.weight_numel
    True
    >>> with torch.no_grad():
    ...     lin1.weight.fill_(1.0)
    ...     lin2.weight.fill_(1.0)
    Parameter containing:
    ...
    >>> x = torch.arange(10.0)
    >>> (lin1(x) - lin2(x)).abs().item() < 1e-5
    True

    """

    weight_numel: int
    internal_weights: bool
    shared_weights: bool

    def __init__(
        self,
        irreps_in: Irreps,
        irreps_out: Irreps,
        *,
        f_in: Optional[int] = None,
        f_out: Optional[int] = None,
        internal_weights: Optional[bool] = None,
        shared_weights: Optional[bool] = None,
        instructions: Optional[List[Tuple[int, int]]] = None,
        biases: Union[bool, List[bool]] = False,
        path_normalization: str = "element",
        _optimize_einsums: Optional[bool] = None,
    ) -> None:
        super().__init__()
        assert path_normalization in ["element", "path"]
        irreps_in = Irreps(irreps_in)
        irreps_out = Irreps(irreps_out)
        if instructions is None:
            instructions = [
                (i_in, i_out)
                for i_in, (_, ir_in) in enumerate(irreps_in)
                for i_out, (_, ir_out) in enumerate(irreps_out)
                if ir_in == ir_out
            ]
        instructions = [
            Instruction(
                i_in=i_in,
                i_out=i_out,
                path_shape=(irreps_in[i_in].mul, irreps_out[i_out].mul),
                path_weight=1,
            )
            for i_in, i_out in instructions
        ]

        def alpha(ins) -> float:
            x = sum(
                irreps_in[i.i_in if path_normalization == "element" else ins.i_in].mul
                for i in instructions
                if i.i_out == ins.i_out
            )
            if f_in is not None:
                x *= f_in
            return 1.0 if x == 0 else x

        instructions = [
            Instruction(
                i_in=ins.i_in,
                i_out=ins.i_out,
                path_shape=ins.path_shape,
                path_weight=alpha(ins) ** -0.5,
            )
            for ins in instructions
        ]
        for ins in instructions:
            if not ins.i_in < len(irreps_in):
                raise IndexError(f"{ins.i_in} is not a valid index for irreps_in")
            if not ins.i_out < len(irreps_out):
                raise IndexError(f"{ins.i_out} is not a valid index for irreps_out")
            if not (
                ins.i_in == -1 or irreps_in[ins.i_in].ir == irreps_out[ins.i_out].ir
            ):
                raise ValueError(
                    f"{ins.i_in} and {ins.i_out} do not have the same irrep"
                )
        if biases is None:
            biases = len(irreps_out) * (False,)
        if isinstance(biases, bool):
            biases = [(biases and ir.is_scalar()) for _, ir in irreps_out]
        assert len(biases) == len(irreps_out)
        assert all(ir.is_scalar() or not b for b, (_, ir) in zip(biases, irreps_out))
        instructions += [
            Instruction(i_in=-1, i_out=i_out, path_shape=(mul_ir.dim,), path_weight=1.0)
            for i_out, (bias, mul_ir) in enumerate(zip(biases, irreps_out))
            if bias
        ]
        if shared_weights is False and internal_weights is None:
            internal_weights = False
        if shared_weights is None:
            shared_weights = True
        if internal_weights is None:
            internal_weights = True
        assert shared_weights or not internal_weights
        self.internal_weights = internal_weights
        self.shared_weights = shared_weights
        self.irreps_in = irreps_in
        self.irreps_out = irreps_out
        self.instructions = instructions
        opt_defaults = get_optimization_defaults()
        self._optimize_einsums = (
            _optimize_einsums
            if _optimize_einsums is not None
            else opt_defaults["optimize_einsums"]
        )
        del opt_defaults
        graphmod, self.weight_numel, self.bias_numel = _codegen_linear(
            self.irreps_in,
            self.irreps_out,
            self.instructions,
            f_in,
            f_out,
            shared_weights=shared_weights,
            optimize_einsums=self._optimize_einsums,
        )
        self._codegen_register({"_compiled_main": graphmod})
        if internal_weights and self.weight_numel > 0:
            assert self.shared_weights, "Having internal weights impose shared weights"
            self.weight = paddle.nn.Parameter(
                # list-shape form: also accepted by the
                # randn_pt(shape, dtype, name) helper
                paddle.randn(
                    [*(((f_in, f_out) if f_in is not None else ())), self.weight_numel]
                )
            )
        else:
            self.register_buffer("weight", paddle.Tensor())
        if internal_weights and self.bias_numel > 0:
            assert self.shared_weights, "Having internal weights impose shared weights"
            self.bias = paddle.nn.Parameter(
                paddle.zeros(*((f_out,) if f_out is not None else ()), self.bias_numel)
            )
        else:
            self.register_buffer("bias", paddle.Tensor())
        if self.irreps_out.dim > 0:
            output_mask = paddle.cat(
                [
                    (
                        paddle.ones(mul_ir.dim)
                        if any(
                            ins.i_out == i_out and 0 not in ins.path_shape
                            for ins in self.instructions
                        )
                        else paddle.zeros(mul_ir.dim)
                    )
                    for i_out, mul_ir in enumerate(self.irreps_out)
                ]
            )
        else:
            output_mask = paddle.ones(0)
        self.register_buffer("output_mask", output_mask)
        self.weight_index_slices = []
        for i, ins in enumerate(self.instructions):
            offset = sum(prod(ins_pre.path_shape) for ins_pre in self.instructions[:i])
            self.weight_index_slices.append(
                LinearSlices(
                    slice(offset, offset + prod(ins.path_shape), None), ins.path_shape
                )
            )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.irreps_in} -> {self.irreps_out} | {self.weight_numel} weights)"

    def forward(
        self,
        features,
        weight: Optional[paddle.Tensor] = None,
        bias: Optional[paddle.Tensor] = None,
    ):
        """evaluate

        Parameters
        ----------
        features : `torch.Tensor`
            tensor of shape ``(..., irreps_in.dim)``

        weight : `torch.Tensor`, optional
            required if ``internal_weights`` is `False`

        Returns
        -------
        `torch.Tensor`
            tensor of shape ``(..., irreps_out.dim)``
        """
        if weight is None:
            if self.weight_numel > 0 and not self.internal_weights:
                raise RuntimeError(
                    "Weights must be provided when internal_weights = False"
                )
            weight = self.weight
        if bias is None:
            if self.bias_numel > 0 and not self.internal_weights:
                raise RuntimeError(
                    "Biases must be provided when internal_weights = False"
                )
            bias = self.bias
        return self._compiled_main(features, weight, bias)

    def weight_view_for_instruction(
        self, instruction: int, weight: Optional[paddle.Tensor] = None
    ) -> paddle.Tensor:
        """View of weights corresponding to ``instruction``.

        Parameters
        ----------
        instruction : int
            The index of the instruction to get a view on the weights for.

        weight : `torch.Tensor`, optional
            like ``weight`` argument to ``forward()``

        Returns
        -------
        `torch.Tensor`
            A view on ``weight`` or this object's internal weights for the weights corresponding to the ``instruction`` th
            instruction.
        """
        if weight is None:
            assert (
                self.internal_weights
            ), "Weights must be provided when internal_weights = False"
            weight = self.weight
        batchshape = weight.shape[:-1]
        offset = sum(prod(ins.path_shape) for ins in self.instructions[:instruction])
        ins = self.instructions[instruction]
        return paddle.slice(
            weight, [-1], [offset], [offset + prod(ins.path_shape)]
        ).reshape(batchshape + ins.path_shape)

    def weight_views(
        self, weight: Optional[paddle.Tensor] = None, yield_instruction: bool = False
    ):
        """Iterator over weight views for all instructions.

        Parameters
        ----------
        weight : `torch.Tensor`, optional
            like ``weight`` argument to ``forward()``

        yield_instruction : `bool`, default False
            Whether to also yield the corresponding instruction.

        Yields
        ------
        If ``yield_instruction`` is ``True``, yields ``(instruction_index, instruction, weight_view)``.
        Otherwise, yields ``weight_view``.
        """
        if weight is None:
            assert (
                self.internal_weights
            ), "Weights must be provided when internal_weights = False"
            weight = self.weight
        batchshape = weight.shape[:-1]
        offset = 0
        for ins_i, ins in enumerate(self.instructions):
            flatsize = prod(ins.path_shape)
            this_weight = paddle.slice(
                weight, [-1], [offset], [offset + flatsize]
            ).reshape(batchshape + ins.path_shape)
            offset += flatsize
            if yield_instruction:
                yield ins_i, ins, this_weight
            else:
                yield this_weight


class _LinearStep(NamedTuple):
    """Eager execution plan for a single instruction."""

    kind: str  # "w" (weight path) or "b" (bias path, i_in == -1)
    i_in: int
    i_out: int
    path_weight: float
    out_dim: int
    w_start: int
    w_len: int
    w_shape: tuple


class EagerLinear(paddle.nn.Layer):
    """Eager implementation of ``linear_forward`` (forward(x, w, b)).

    The instruction plan is fixed at construction time; ``forward`` executes
    the input slices, the per-path contractions, the path_weight scaling and
    the per-i_out grouped sums in order. No constant buffers are registered
    (Linear has no w3j-style constants).
    """

    def __init__(
        self,
        irreps_in: Irreps,
        irreps_out: Irreps,
        instructions: List[Instruction],
        f_in: Optional[int] = None,
        f_out: Optional[int] = None,
        shared_weights: bool = False,
    ) -> None:
        super().__init__()
        self._f_in = f_in
        self._f_out = f_out
        self._in_dim = irreps_in.dim
        self._out_dim = irreps_out.dim
        self._shared_weights = shared_weights
        self._z = "" if shared_weights else "z"
        self._bias_numel = sum(
            irreps_out[i.i_out].dim for i in instructions if i.i_in == -1
        )
        instructions = [ins for ins in instructions if 0 not in ins.path_shape]
        self._weight_numel = sum(
            prod(ins.path_shape) for ins in instructions if ins.i_in != -1
        )
        self._x_slices = [
            (sl.start, mul_ir.dim, mul_ir.mul, mul_ir.ir.dim)
            for sl, mul_ir in zip(irreps_in.slices(), irreps_in)
        ]
        self._out_groups = [
            (i_out, mul_ir.dim)
            for i_out, mul_ir in enumerate(irreps_out)
            if mul_ir.mul > 0
        ]
        steps: List[_LinearStep] = []
        flat_weight_index = 0
        flat_bias_index = 0
        for ins in instructions:
            mul_ir_out = irreps_out[ins.i_out]
            if ins.i_in == -1:
                w_start, w_len = flat_bias_index, prod(ins.path_shape)
                flat_bias_index += prod(ins.path_shape)
                steps.append(
                    _LinearStep(
                        kind="b",
                        i_in=ins.i_in,
                        i_out=ins.i_out,
                        path_weight=ins.path_weight,
                        out_dim=mul_ir_out.dim,
                        w_start=w_start,
                        w_len=w_len,
                        w_shape=ins.path_shape,
                    )
                )
            else:
                mul_ir_in = irreps_in[ins.i_in]
                if mul_ir_in.dim == 0 or mul_ir_out.dim == 0:
                    continue
                path_nweight = prod(ins.path_shape)
                steps.append(
                    _LinearStep(
                        kind="w",
                        i_in=ins.i_in,
                        i_out=ins.i_out,
                        path_weight=ins.path_weight,
                        out_dim=mul_ir_out.dim,
                        w_start=flat_weight_index,
                        w_len=path_nweight,
                        w_shape=ins.path_shape,
                    )
                )
                flat_weight_index += path_nweight
        self._steps = steps

    def forward(self, x, ws, bs):
        if self._f_in is None:
            outsize = list(x.shape[:-1]) + [self._out_dim]
        else:
            outsize = list(x.shape[:-2]) + [self._f_out, self._out_dim]
        if self._bias_numel > 0:
            bs = (
                bs.reshape([-1, self._bias_numel])
                if self._f_in is None
                else bs.reshape([-1, self._f_out, self._bias_numel])
            )
        if len(self._steps) == 0:
            return paddle.zeros(outsize, dtype=x.dtype)
        x = (
            x.reshape([-1, self._in_dim])
            if self._f_in is None
            else x.reshape([-1, self._f_in, self._in_dim])
        )
        batch_out = x.shape[0]
        if self._weight_numel > 0:
            ws = (
                ws.reshape([-1, self._weight_numel])
                if self._f_in is None
                else ws.reshape([-1, self._f_in, self._f_out, self._weight_numel])
            )
        f_mid = [] if self._f_in is None else [self._f_in]
        f_out_mid = [] if self._f_out is None else [self._f_out]
        if len(self._x_slices) == 1:
            _, _, mul, dim = self._x_slices[0]
            x_list = [x.reshape([batch_out] + f_mid + [mul, dim])]
        else:
            x_list = [
                paddle.slice(x, [-1], [a], [a + length]).reshape(
                    [batch_out] + f_mid + [mul, dim]
                )
                for (a, length, mul, dim) in self._x_slices
            ]
        out_list = []
        for step in self._steps:
            if step.kind == "b":
                b = paddle.slice(bs, [-1], [step.w_start], [step.w_start + step.w_len])
                out_list += [
                    (step.path_weight * b).reshape([1] + f_out_mid + [step.out_dim])
                ]
            else:
                if len(self._steps) == 1:
                    w = ws
                else:
                    w = paddle.slice(
                        ws, [-1], [step.w_start], [step.w_start + step.w_len]
                    )
                w = w.reshape(
                    (() if self._shared_weights else (-1,))
                    + (tuple(f_out_mid) if self._f_in is not None else ())
                    + tuple(step.w_shape)
                )
                if self._f_in is None:
                    # "zuw,zui->zwi" decomposed: contract u via one plain 2D
                    # matmul; the (z,i,u)@(z|u,w) operand order keeps the
                    # double-grad path of the training graph stable.
                    x_in = x_list[step.i_in]
                    ein_out = paddle.matmul(
                        x_in.transpose([0, 2, 1]), w
                    ).transpose([0, 2, 1])
                else:
                    ein_out = _einsum(
                        f"{self._z}xyuw,zxui->zywi", w, x_list[step.i_in]
                    )
                ein_out = step.path_weight * ein_out
                out_list += [
                    ein_out.reshape([batch_out] + f_out_mid + [step.out_dim])
                ]
        out = [
            _sum_tensors(
                [
                    o
                    for step, o in zip(self._steps, out_list)
                    if step.i_out == i_out
                ],
                shape=tuple([batch_out] + f_out_mid + [out_dim]),
                like=x,
            )
            for i_out, out_dim in self._out_groups
        ]
        if len(out) > 1:
            out = _place_lastdim(out)
        else:
            out = out[0]
        return out.reshape(outsize)


# Cache of 0/1 placement matrices for last-dim concat:
# (dims, dtype, place) -> [M_i]
_PLACE_MATS = {}


def _place_lastdim(tensors):
    # Last-dim concat as sum_i t_i @ M_i with 0/1 placement matrices
    # (bitwise-identical forward); matmul/add keep the double-grad path of
    # the create_graph training route.
    dims = [int(t.shape[-1]) for t in tensors]
    key = (tuple(dims), str(tensors[0].dtype), str(tensors[0].place))
    # Cache in dynamic mode only: a static trace builds one Program per
    # (train/eval) mode; cached python tensors must not leak across Programs.
    use_cache = paddle.in_dynamic_mode()
    mats = _PLACE_MATS.get(key) if use_cache else None
    if mats is None:
        total = sum(dims)
        mats = []
        ofs = 0
        for d in dims:
            m = paddle.zeros((d, total), dtype=tensors[0].dtype)
            m[0:d, ofs:ofs + d] = paddle.eye(d, dtype=tensors[0].dtype)
            m.stop_gradient = True
            mats.append(m)
            ofs += d
        if use_cache:
            _PLACE_MATS[key] = mats
    out = tensors[0] @ mats[0]
    for t, m in zip(tensors[1:], mats[1:]):
        out = out + t @ m
    return out


def _codegen_linear(
    irreps_in: Irreps,
    irreps_out: Irreps,
    instructions: List[Instruction],
    f_in: Optional[int] = None,
    f_out: Optional[int] = None,
    shared_weights: bool = False,
    optimize_einsums: bool = True,
) -> Tuple[paddle.nn.Layer, int, int]:
    """Build the eager ``linear_forward`` layer.

    Returns (layer, weight_numel, bias_numel). ``optimize_einsums`` is kept
    for signature compatibility; the eager layer executes the source
    formulas directly.
    """
    layer = EagerLinear(
        irreps_in, irreps_out, instructions, f_in, f_out, shared_weights
    )
    return layer, layer._weight_numel, layer._bias_numel

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

from typing import Callable
from typing import List
from typing import NamedTuple
from typing import Optional
from typing import Tuple
from typing import Union

import paddle

from ppmat.models.common.e3nn import o3
from ppmat.models.common.e3nn.util import prod
from ppmat.models.common.e3nn.util.codegen import CodeGenMixin

from ._tensor_product._codegen import _sum_tensors


class Instruction(NamedTuple):
    i_in: int
    i_out: int
    path_shape: tuple
    path_weight: float


class Linear(CodeGenMixin, paddle.nn.Layer):
    r"""Linear operation equivariant to O(3)"""

    weight_numel: int
    internal_weights: bool
    shared_weights: bool

    def __init__(
        self,
        irreps_in,
        irreps_out,
        *,
        f_in: Optional[int] = None,
        f_out: Optional[int] = None,
        internal_weights: Optional[bool] = None,
        shared_weights: Optional[bool] = None,
        instructions: Optional[List[Tuple[int, int]]] = None,
        biases: Union[bool, List[bool]] = False,
        path_normalization: str = "element",
    ):
        super().__init__()

        assert path_normalization in ["element", "path"]

        irreps_in = o3.Irreps(irreps_in)
        irreps_out = o3.Irreps(irreps_out)

        if instructions is None:
            # By default, make all possible connections
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

        def alpha(ins):
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
                path_weight=alpha(ins) ** (-0.5),
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
            biases = [biases and ir.is_scalar() for _, ir in irreps_out]

        assert len(biases) == len(irreps_out)
        assert all(ir.is_scalar() or (not b) for b, (_, ir) in zip(biases, irreps_out))

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

        # Generate forward function and weight sizes
        self.forward_fn, self.weight_numel, self.bias_numel = _codegen_linear(
            self.irreps_in,
            self.irreps_out,
            self.instructions,
            f_in,
            f_out,
            shared_weights=shared_weights,
        )

        # Generate weights
        if internal_weights and self.weight_numel > 0:
            assert self.shared_weights, "Having internal weights impose shared weights"
            weight_shape = ((f_in, f_out) if f_in is not None else ()) + (
                self.weight_numel,
            )
            self.weight = self.create_parameter(
                shape=weight_shape, default_initializer=paddle.nn.initializer.Normal()
            )
        else:
            # Avoid storing zero-sized tensors on the module: Paddle DataParallel
            # synchronizes buffers across ranks and does not support numel == 0.
            # When weight_numel == 0 (or weights are external), we create an empty
            # tensor on-the-fly in forward if needed.
            self.weight = None

        # Generate biases
        if internal_weights and self.bias_numel > 0:
            assert self.shared_weights, "Having internal weights impose shared weights"
            bias_shape = ((f_out,) if f_out is not None else ()) + (self.bias_numel,)
            self.bias = self.create_parameter(
                shape=bias_shape,
                default_initializer=paddle.nn.initializer.Constant(value=0.0),
            )
        else:
            self.bias = None

        # Compute output mask
        if self.irreps_out.dim > 0:
            output_mask = paddle.concat(
                [
                    (
                        paddle.ones([mul_ir.dim])
                        if any(
                            (ins.i_out == i_out) and (0 not in ins.path_shape)
                            for ins in self.instructions
                        )
                        else paddle.zeros([mul_ir.dim])
                    )
                    for i_out, mul_ir in enumerate(self.irreps_out)
                ]
            )
        else:
            output_mask = paddle.ones([0])
        self.register_buffer("output_mask", output_mask)

    def forward(
        self,
        features,
        weight: Optional[paddle.Tensor] = None,
        bias: Optional[paddle.Tensor] = None,
    ):
        if weight is None:
            if self.weight_numel > 0 and not self.internal_weights:
                raise RuntimeError(
                    "Weights must be provided when internal_weights = False"
                )
            if self.weight is None:
                weight = paddle.to_tensor(
                    [], dtype=features.dtype, place=features.place
                )
            else:
                weight = self.weight
        if bias is None:
            if self.bias_numel > 0 and not self.internal_weights:
                raise RuntimeError(
                    "Biases must be provided when internal_weights = False"
                )
            if self.bias is None:
                bias = paddle.to_tensor([], dtype=features.dtype, place=features.place)
            else:
                bias = self.bias
        return self.forward_fn(features, weight, bias)


def _codegen_linear(
    irreps_in: o3.Irreps,
    irreps_out: o3.Irreps,
    instructions: List[Instruction],
    f_in: Optional[int] = None,
    f_out: Optional[int] = None,
    shared_weights: bool = False,
) -> Tuple[Callable, int, int]:
    # Remove empty instructions
    instructions = [ins for ins in instructions if 0 not in ins.path_shape]

    def forward_fn(
        x: paddle.Tensor, ws: paddle.Tensor, bs: paddle.Tensor
    ) -> paddle.Tensor:
        if f_in is None:
            size = x.shape[:-1]
            outsize = size + [irreps_out.dim]
        else:
            size = x.shape[:-2]
            outsize = size + [f_out, irreps_out.dim]

        bias_numel = sum(irreps_out[i.i_out].dim for i in instructions if i.i_in == -1)

        if bias_numel > 0:
            if f_out is None:
                bs = bs.reshape([-1, bias_numel])
            else:
                bs = bs.reshape([-1, f_out, bias_numel])

        if len(instructions) == 0 and bias_numel == 0:
            return paddle.zeros(outsize, dtype=x.dtype)

        if f_in is None:
            x = x.reshape([-1, irreps_in.dim])
        else:
            x = x.reshape([-1, f_in, irreps_in.dim])
        batch_out = x.shape[0]

        weight_numel = sum(
            prod(ins.path_shape) for ins in instructions if ins.i_in != -1
        )
        if weight_numel > 0:
            ws = (
                ws.reshape([-1, weight_numel])
                if f_in is None
                else ws.reshape([-1, f_in, f_out, weight_numel])
            )

        # Extract individual input irreps
        if len(irreps_in) == 1:
            x_list = [
                x.reshape(
                    [batch_out]
                    + ([] if f_in is None else [f_in])
                    + [irreps_in[0].mul, irreps_in[0].ir.dim]
                )
            ]
        else:
            x_list = []
            start = 0
            for mul_ir in irreps_in:
                x_slice = paddle.slice(x, [-1], [start], [start + mul_ir.dim])
                x_list.append(
                    x_slice.reshape(
                        [batch_out]
                        + ([] if f_in is None else [f_in])
                        + [mul_ir.mul, mul_ir.ir.dim]
                    )
                )
                start += mul_ir.dim

        z = "" if shared_weights else "z"
        flat_weight_index = 0
        flat_bias_index = 0
        out_list = []

        # Process instructions
        for ins in instructions:
            mul_ir_out = irreps_out[ins.i_out]

            if ins.i_in == -1:
                # Handle bias
                b = paddle.slice(
                    bs,
                    [-1],
                    [flat_bias_index],
                    [flat_bias_index + prod(ins.path_shape)],
                )
                flat_bias_index += prod(ins.path_shape)
                out_list += [
                    (ins.path_weight * b).reshape(
                        [1] + ([] if f_out is None else [f_out]) + [mul_ir_out.dim]
                    )
                ]
            else:
                mul_ir_in = irreps_in[ins.i_in]
                if mul_ir_in.dim == 0 or mul_ir_out.dim == 0:
                    continue

                path_nweight = prod(ins.path_shape)
                w = (
                    ws
                    if len(instructions) == 1
                    else paddle.slice(
                        ws,
                        [-1],
                        [flat_weight_index],
                        [flat_weight_index + path_nweight],
                    )
                )
                w = w.reshape(
                    ([] if shared_weights else [-1])
                    + ([] if f_in is None else [f_in, f_out])
                    + list(ins.path_shape)
                )
                flat_weight_index += path_nweight

                if f_in is None:
                    ein_out = paddle.einsum(f"{z}uw,zui->zwi", w, x_list[ins.i_in])
                else:
                    ein_out = paddle.einsum(f"{z}xyuw,zxui->zywi", w, x_list[ins.i_in])

                ein_out = ins.path_weight * ein_out
                out_list += [
                    ein_out.reshape(
                        [batch_out]
                        + ([] if f_out is None else [f_out])
                        + [mul_ir_out.dim]
                    )
                ]

        # Combine outputs
        out = [
            _sum_tensors(
                [out for ins, out in zip(instructions, out_list) if ins.i_out == i_out],
                shape=[batch_out]
                + ([] if f_out is None else [f_out])
                + [mul_ir_out.dim],
                like=x,
            )
            for i_out, mul_ir_out in enumerate(irreps_out)
            if mul_ir_out.mul > 0
        ]

        if len(out) > 1:
            out = paddle.concat(out, axis=-1)
        else:
            out = out[0]

        return out.reshape(outsize)

    weight_numel = sum(prod(ins.path_shape) for ins in instructions if ins.i_in != -1)
    bias_numel = sum(irreps_out[i.i_out].dim for i in instructions if i.i_in == -1)

    return forward_fn, weight_numel, bias_numel

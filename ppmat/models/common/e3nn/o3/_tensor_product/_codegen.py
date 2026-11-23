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

from math import sqrt
from typing import Callable
from typing import List

import paddle

from ppmat.models.common.e3nn import o3
from ppmat.models.common.e3nn.util import prod

from ._instruction import Instruction


def reshape(self, *args, **kwargs):
    if args:
        if len(args) == 1 and isinstance(args[0], (tuple, list)):
            return paddle.reshape(self, args[0])
        else:
            return paddle.reshape(self, list(args))
    elif kwargs:
        assert "shape" in kwargs
        return paddle.reshape(self, shape=kwargs["shape"])


setattr(paddle.Tensor, "reshape", reshape)


def max_class_func(self, *args, **kwargs):
    if "other" in kwargs:
        kwargs["y"] = kwargs.pop("other")
        ret = paddle.maximum(self, *args, **kwargs)
    elif len(args) == 1 and isinstance(args[0], paddle.Tensor):
        ret = paddle.maximum(self, *args, **kwargs)
    else:
        if "dim" in kwargs:
            kwargs["axis"] = kwargs.pop("dim")

        if "axis" in kwargs or len(args) >= 1:
            ret = paddle.max(self, *args, **kwargs), paddle.argmax(
                self, *args, **kwargs
            )
        else:
            ret = paddle.max(self, *args, **kwargs)

    return ret


setattr(paddle.Tensor, "max_func", max_class_func)


def _sum_tensors(xs: List[paddle.Tensor], shape: list, like: paddle.Tensor):
    if len(xs) > 0:
        out = xs[0]
        for x in xs[1:]:
            out = out + x
        return out
    return paddle.zeros(shape=shape, dtype=like.dtype)


def codegen_tensor_product_right(
    irreps_in1: o3.Irreps,
    irreps_in2: o3.Irreps,
    irreps_out: o3.Irreps,
    instructions: List[Instruction],
    shared_weights: bool = False,
    specialized_code: bool = True,
    optimize_einsums: bool = True,
) -> Callable:
    """ """
    filtered_instructions = [ins for ins in instructions if 0 not in ins.path_shape]
    w3j_dict = {}
    for ins in filtered_instructions:
        i_in1, i_in2, i_out = ins.i_in1, ins.i_in2, ins.i_out
        mul_ir_in1 = irreps_in1[i_in1]
        mul_ir_in2 = irreps_in2[i_in2]
        mul_ir_out = irreps_out[i_out]
        if (mul_ir_in1.ir.l, mul_ir_in2.ir.l, mul_ir_out.ir.l) not in w3j_dict:
            w3j_dict[mul_ir_in1.ir.l, mul_ir_in2.ir.l, mul_ir_out.ir.l] = o3.wigner_3j(
                mul_ir_in1.ir.l, mul_ir_in2.ir.l, mul_ir_out.ir.l
            )

    def tp_right(x2: paddle.Tensor, weights: paddle.Tensor = None) -> paddle.Tensor:
        if not filtered_instructions:
            batch_shape = tuple(x2.shape)[:-1]
            return paddle.zeros(
                shape=batch_shape + (irreps_in1.dim, irreps_out.dim), dtype=x2.dtype
            )
        batch_shape = tuple(x2.shape)[:-1]
        batch_size = prod(batch_shape)
        x2_flat = x2.reshape(batch_size, irreps_in2.dim)
        x2_parts = []
        for i, mul_ir in enumerate(irreps_in2):
            slice_idx = irreps_in2.slices()[i]
            x2_parts.append(
                x2_flat[:, slice_idx].reshape(batch_size, mul_ir.mul, mul_ir.ir.dim)
            )
        result = paddle.zeros(
            shape=[batch_size, irreps_in1.dim, irreps_out.dim], dtype=x2.dtype
        )
        weight_idx = 0
        for ins in filtered_instructions:
            i_in1, i_in2, i_out = ins.i_in1, ins.i_in2, ins.i_out
            mul_ir_in1 = irreps_in1[i_in1]
            mul_ir_in2 = irreps_in2[i_in2]
            mul_ir_out = irreps_out[i_out]
            if mul_ir_in1.dim == 0 or mul_ir_in2.dim == 0 or mul_ir_out.dim == 0:
                continue
            x2_part = x2_parts[i_in2]
            in1_slice = irreps_in1.slices()[i_in1]
            out_slice = irreps_out.slices()[i_out]
            in1_dim = mul_ir_in1.ir.dim
            in2_dim = mul_ir_in2.ir.dim
            out_dim = mul_ir_out.ir.dim
            in1_mul = mul_ir_in1.mul
            in2_mul = mul_ir_in2.mul
            out_mul = mul_ir_out.mul
            in1_l = mul_ir_in1.ir.l
            in2_l = mul_ir_in2.ir.l
            out_l = mul_ir_out.ir.l
            e1 = paddle.eye(num_rows=in1_mul, dtype=x2.dtype)
            e2 = paddle.eye(num_rows=in2_mul, dtype=x2.dtype)
            i1 = paddle.eye(num_rows=in1_dim, dtype=x2.dtype)
            if ins.has_weight:
                weight_size = prod(ins.path_shape)
                if shared_weights:
                    w = weights[weight_idx : weight_idx + weight_size].reshape(
                        ins.path_shape
                    )
                else:
                    w = weights.reshape(batch_size, -1)[
                        :, weight_idx : weight_idx + weight_size
                    ]
                    w = w.reshape(batch_size, *ins.path_shape)
                weight_idx += weight_size
            if ins.connection_mode == "uvw":
                if ins.has_weight:
                    if specialized_code and (in1_l, in2_l, out_l) == (0, 0, 0):
                        x2_reshaped = x2_part.reshape(batch_size, in2_mul)
                        if shared_weights:
                            path_result = paddle.einsum("uvw,bv->buw", w, x2_reshaped)
                        else:
                            path_result = paddle.einsum("buvw,bv->buw", w, x2_reshaped)
                        path_result = path_result.reshape(batch_size, in1_mul, out_mul)
                    elif specialized_code and in1_l == 0:
                        if shared_weights:
                            path_result = paddle.einsum("uvw,bvi->buwi", w, x2_part)
                        else:
                            path_result = paddle.einsum("buvw,bvi->buwi", w, x2_part)
                        path_result = path_result.reshape(
                            batch_size, in1_mul, out_mul * out_dim
                        )
                        path_result = path_result / sqrt(out_dim)
                    elif specialized_code and in2_l == 0:
                        x2_reshaped = x2_part.reshape(batch_size, in2_mul)
                        if shared_weights:
                            path_result = paddle.einsum(
                                "uvw,ij,bv->buiwj", w, i1, x2_reshaped
                            )
                        else:
                            path_result = paddle.einsum(
                                "buvw,ij,bv->buiwj", w, i1, x2_reshaped
                            )
                        path_result = path_result.reshape(
                            batch_size, in1_mul * in1_dim, out_mul * out_dim
                        )
                        path_result = path_result / sqrt(out_dim)
                    elif specialized_code and out_l == 0:
                        if shared_weights:
                            path_result = paddle.einsum("uvw,bvi->buiw", w, x2_part)
                        else:
                            path_result = paddle.einsum("buvw,bvi->buiw", w, x2_part)
                        path_result = path_result.reshape(
                            batch_size, in1_mul * in1_dim, out_mul
                        )
                        path_result = path_result / sqrt(in1_dim)
                    else:
                        w3j = w3j_dict[in1_l, in2_l, out_l]
                        if shared_weights:
                            path_result = paddle.einsum(
                                "uvw,ijk,bvj->buiwk", w, w3j, x2_part
                            )
                        else:
                            path_result = paddle.einsum(
                                "buvw,ijk,bvj->buiwk", w, w3j, x2_part
                            )
                        path_result = path_result.reshape(
                            batch_size, in1_mul * in1_dim, out_mul * out_dim
                        )
                else:
                    w3j = w3j_dict[in1_l, in2_l, out_l]
                    path_result = paddle.einsum("ijk,bvj->bvik", w3j, x2_part)
                    path_result = path_result.reshape(
                        batch_size, in2_mul * in1_dim, out_dim
                    )
            elif ins.connection_mode == "uuw":
                if ins.has_weight:
                    if specialized_code and (in1_l, in2_l, out_l) == (0, 0, 0):
                        x2_reshaped = x2_part.reshape(batch_size, in2_mul)
                        if shared_weights:
                            path_result = paddle.einsum(
                                "u,uw,bu->buw", w, e2, x2_reshaped
                            )
                        else:
                            path_result = paddle.einsum(
                                "bu,uw,bu->buw", w, e2, x2_reshaped
                            )
                        path_result = path_result.reshape(batch_size, in1_mul, out_mul)
                    elif specialized_code and in1_l == 0:
                        if shared_weights:
                            path_result = paddle.einsum(
                                "u,uw,bui->buwi", w, e2, x2_part
                            )
                        else:
                            path_result = paddle.einsum(
                                "bu,uw,bui->buwi", w, e2, x2_part
                            )
                        path_result = path_result.reshape(
                            batch_size, in1_mul, out_mul * out_dim
                        )
                        path_result = path_result / sqrt(out_dim)
                    elif specialized_code and in2_l == 0:
                        x2_reshaped = x2_part.reshape(batch_size, in2_mul)
                        if shared_weights:
                            path_result = paddle.einsum(
                                "u,ij,uw,bu->buiwj", w, i1, e2, x2_reshaped
                            )
                        else:
                            path_result = paddle.einsum(
                                "bu,ij,uw,bu->buiwj", w, i1, e2, x2_reshaped
                            )
                        path_result = path_result.reshape(
                            batch_size, in1_mul * in1_dim, out_mul * out_dim
                        )
                        path_result = path_result / sqrt(out_dim)
                    elif specialized_code and out_l == 0:
                        if shared_weights:
                            path_result = paddle.einsum(
                                "u,uw,bui->buiw", w, e2, x2_part
                            )
                        else:
                            path_result = paddle.einsum(
                                "bu,uw,bui->buiw", w, e2, x2_part
                            )
                        path_result = path_result.reshape(
                            batch_size, in1_mul * in1_dim, out_mul
                        )
                        path_result = path_result / sqrt(in1_dim)
                    else:
                        w3j = w3j_dict[in1_l, in2_l, out_l]
                        if shared_weights:
                            path_result = paddle.einsum(
                                "u,ijk,uw,buj->buiwk", w, w3j, e1, x2_part
                            )
                        else:
                            path_result = paddle.einsum(
                                "bu,ijk,uw,buj->buiwk", w, w3j, e1, x2_part
                            )
                        path_result = path_result.reshape(
                            batch_size, in1_mul * in1_dim, out_mul * out_dim
                        )
                elif specialized_code and (in1_l, in2_l, out_l) == (0, 0, 0):
                    x2_reshaped = x2_part.reshape(batch_size, in2_mul)
                    path_result = paddle.einsum("uw,bu->buw", e2, x2_reshaped)
                    path_result = path_result.reshape(batch_size, in1_mul, out_mul)
                elif specialized_code and (in1_l, in2_l, out_l) == (1, 1, 1):
                    w3j = w3j_dict[in1_l, in2_l, out_l]
                    path_result = paddle.einsum("ijk,uw,buj->buiwk", w3j, e1, x2_part)
                    path_result = path_result.reshape(
                        batch_size, in1_mul * in1_dim, out_mul * out_dim
                    )
                elif specialized_code and in1_l == 0:
                    path_result = paddle.einsum("uw,bui->buwi", e2, x2_part)
                    path_result = path_result.reshape(
                        batch_size, in1_mul, out_mul * out_dim
                    )
                    path_result = path_result / sqrt(out_dim)
                elif specialized_code and in2_l == 0:
                    x2_reshaped = x2_part.reshape(batch_size, in2_mul)
                    path_result = paddle.einsum("ij,uw,bu->buiwj", i1, e2, x2_reshaped)
                    path_result = path_result.reshape(
                        batch_size, in1_mul * in1_dim, out_mul * out_dim
                    )
                    path_result = path_result / sqrt(out_dim)
                elif specialized_code and out_l == 0:
                    path_result = paddle.einsum("uw,bui->buiw", e2, x2_part)
                    path_result = path_result.reshape(
                        batch_size, in1_mul * in1_dim, out_mul
                    )
                    path_result = path_result / sqrt(in1_dim)
                else:
                    w3j = w3j_dict[in1_l, in2_l, out_l]
                    path_result = paddle.einsum("ijk,uw,buj->buiwk", w3j, e1, x2_part)
                    path_result = path_result.reshape(
                        batch_size, in1_mul * in1_dim, out_mul * out_dim
                    )
            elif ins.connection_mode == "uvuv":
                if ins.has_weight:
                    w3j = w3j_dict[in1_l, in2_l, out_l]
                    if shared_weights:
                        path_result = paddle.einsum(
                            "uv,ijk,uw,bvj->buiwvk", w, w3j, e1, x2_part
                        )
                    else:
                        path_result = paddle.einsum(
                            "buv,ijk,uw,bvj->buiwvk", w, w3j, e1, x2_part
                        )
                    path_result = path_result.reshape(
                        batch_size, in1_mul * in1_dim, out_mul * out_dim
                    )
                else:
                    w3j = w3j_dict[in1_l, in2_l, out_l]
                    path_result = paddle.einsum("ijk,uw,bvj->buiwvk", w3j, e1, x2_part)
                    path_result = path_result.reshape(
                        batch_size, in1_mul * in1_dim, out_mul * out_dim
                    )
            elif ins.connection_mode in ["uvu<v", "u<vw"]:
                raise NotImplementedError(
                    f"Not Yet Implemented Connection Mode: {ins.connection_mode}"
                )
            else:
                raise ValueError(f"Unknown Connection Mode: {ins.connection_mode}")
            path_result = path_result * ins.path_weight
            result[:, in1_slice, out_slice] += path_result
        return result.reshape(batch_shape + (irreps_in1.dim, irreps_out.dim))

    return tp_right


def codegen_tensor_product_left_right(
    irreps_in1: o3.Irreps,
    irreps_in2: o3.Irreps,
    irreps_out: o3.Irreps,
    instructions: List,
    shared_weights: bool = False,
    specialized_code: bool = True,
    optimize_einsums: bool = True,
) -> Callable:
    w3j_dict = {}
    for ins in instructions:
        if 0 in ins.path_shape:
            continue
        mul_ir_in1 = irreps_in1[ins.i_in1]
        mul_ir_in2 = irreps_in2[ins.i_in2]
        mul_ir_out = irreps_out[ins.i_out]
        key = mul_ir_in1.ir.l, mul_ir_in2.ir.l, mul_ir_out.ir.l
        if key not in w3j_dict:
            w3j_dict[key] = o3.wigner_3j(
                mul_ir_in1.ir.l, mul_ir_in2.ir.l, mul_ir_out.ir.l
            )
    filtered_instructions = [ins for ins in instructions if 0 not in ins.path_shape]

    def tp_forward(
        x1: paddle.Tensor, x2: paddle.Tensor, weights: paddle.Tensor = None
    ) -> paddle.Tensor:
        if len(filtered_instructions) == 0:
            if shared_weights:
                output_shape = tuple(
                    paddle.broadcast_tensors(
                        input=[
                            paddle.zeros(shape=tuple(x1.shape)[:-1]),
                            paddle.zeros(shape=tuple(x2.shape)[:-1]),
                        ]
                    )[0].shape
                )
            else:
                output_shape = tuple(
                    paddle.broadcast_tensors(
                        input=[
                            paddle.zeros(shape=tuple(x1.shape)[:-1]),
                            paddle.zeros(shape=tuple(x2.shape)[:-1]),
                            paddle.zeros(shape=tuple(weights.shape)[:-1]),
                        ]
                    )[0].shape
                )
            return paddle.zeros(shape=output_shape + (irreps_out.dim,), dtype=x1.dtype)
        if shared_weights:
            output_shape = tuple(
                paddle.broadcast_tensors(
                    input=[
                        paddle.zeros(shape=tuple(x1.shape)[:-1]),
                        paddle.zeros(shape=tuple(x2.shape)[:-1]),
                    ]
                )[0].shape
            )
            x1 = x1.broadcast_to(shape=output_shape + (-1,))
            x2 = x2.broadcast_to(shape=output_shape + (-1,))
        else:
            output_shape = tuple(
                paddle.broadcast_tensors(
                    input=[
                        paddle.zeros(shape=tuple(x1.shape)[:-1]),
                        paddle.zeros(shape=tuple(x2.shape)[:-1]),
                        paddle.zeros(shape=tuple(weights.shape)[:-1]),
                    ]
                )[0].shape
            )
            x1 = x1.broadcast_to(shape=output_shape + (-1,))
            x2 = x2.broadcast_to(shape=output_shape + (-1,))
            weights = weights.broadcast_to(shape=output_shape + (-1,))
        final_output_shape = output_shape + (irreps_out.dim,)
        x1 = x1.reshape(-1, irreps_in1.dim)
        x2 = x2.reshape(-1, irreps_in2.dim)
        batch_numel = tuple(x1.shape)[0]
        weight_numel = sum(
            prod(ins.path_shape) for ins in filtered_instructions if ins.has_weight
        )
        if weight_numel > 0 and weights is not None:
            weights = weights.reshape(-1, weight_numel)
        if len(irreps_in1) == 1:
            x1_list = [x1.reshape(batch_numel, irreps_in1[0].mul, irreps_in1[0].ir.dim)]
        else:
            x1_list = []
            for i, mul_ir in zip(irreps_in1.slices(), irreps_in1):
                x1_list.append(x1[:, i].reshape(batch_numel, mul_ir.mul, mul_ir.ir.dim))
        x2_list = []
        if len(irreps_in2) == 1:
            x2_list.append(
                x2.reshape(batch_numel, irreps_in2[0].mul, irreps_in2[0].ir.dim)
            )
        else:
            for i, mul_ir in zip(irreps_in2.slices(), irreps_in2):
                x2_list.append(x2[:, i].reshape(batch_numel, mul_ir.mul, mul_ir.ir.dim))
        z = "" if shared_weights else "z"
        xx_dict = {}
        flat_weight_index = 0
        outputs = []
        for ins in filtered_instructions:
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
            x1_tensor = x1_list[ins.i_in1]
            x2_tensor = x2_list[ins.i_in2]
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
            if ins.has_weight:
                w = weights[
                    :, flat_weight_index : flat_weight_index + prod(ins.path_shape)
                ].reshape((() if shared_weights else (-1,)) + tuple(ins.path_shape))
                flat_weight_index += prod(ins.path_shape)
            key = ins.i_in1, ins.i_in2, ins.connection_mode[:2]
            if key not in xx_dict:
                if ins.connection_mode[:2] == "uu":
                    xx_dict[key] = paddle.einsum("zui,zuj->zuij", x1_tensor, x2_tensor)
                else:
                    xx_dict[key] = paddle.einsum("zui,zvj->zuvij", x1_tensor, x2_tensor)
            xx = xx_dict[key]
            l1l2l3 = mul_ir_in1.ir.l, mul_ir_in2.ir.l, mul_ir_out.ir.l
            w3j = w3j_dict.get(l1l2l3, None)
            if ins.connection_mode == "uvw":
                assert ins.has_weight
                if specialized_code and l1l2l3 == (0, 0, 0):
                    result = paddle.einsum(
                        f"{z}uvw,zu,zv->zw",
                        w,
                        x1_tensor.reshape(batch_numel, mul_ir_in1.dim),
                        x2_tensor.reshape(batch_numel, mul_ir_in2.dim),
                    )
                elif specialized_code and mul_ir_in1.ir.l == 0:
                    result = paddle.einsum(
                        f"{z}uvw,zu,zvj->zwj",
                        w,
                        x1_tensor.reshape(batch_numel, mul_ir_in1.dim),
                        x2_tensor,
                    ) / sqrt(mul_ir_out.ir.dim)
                elif specialized_code and mul_ir_in2.ir.l == 0:
                    result = paddle.einsum(
                        f"{z}uvw,zui,zv->zwi",
                        w,
                        x1_tensor,
                        x2_tensor.reshape(batch_numel, mul_ir_in2.dim),
                    ) / sqrt(mul_ir_out.ir.dim)
                elif specialized_code and mul_ir_out.ir.l == 0:
                    result = paddle.einsum(
                        f"{z}uvw,zui,zvi->zw", w, x1_tensor, x2_tensor
                    ) / sqrt(mul_ir_in1.ir.dim)
                else:
                    result = paddle.einsum(f"{z}uvw,ijk,zuvij->zwk", w, w3j, xx)
            elif ins.connection_mode == "uvu":
                assert mul_ir_in1.mul == mul_ir_out.mul
                if ins.has_weight:
                    if specialized_code and l1l2l3 == (0, 0, 0):
                        result = paddle.einsum(
                            f"{z}uv,zu,zv->zu",
                            w,
                            x1_tensor.reshape(batch_numel, mul_ir_in1.dim),
                            x2_tensor.reshape(batch_numel, mul_ir_in2.dim),
                        )
                    elif specialized_code and mul_ir_in1.ir.l == 0:
                        result = paddle.einsum(
                            f"{z}uv,zu,zvj->zuj",
                            w,
                            x1_tensor.reshape(batch_numel, mul_ir_in1.dim),
                            x2_tensor,
                        ) / sqrt(mul_ir_out.ir.dim)
                    elif specialized_code and mul_ir_in2.ir.l == 0:
                        result = paddle.einsum(
                            f"{z}uv,zui,zv->zui",
                            w,
                            x1_tensor,
                            x2_tensor.reshape(batch_numel, mul_ir_in2.dim),
                        ) / sqrt(mul_ir_out.ir.dim)
                    elif specialized_code and mul_ir_out.ir.l == 0:
                        result = paddle.einsum(
                            f"{z}uv,zui,zvi->zu", w, x1_tensor, x2_tensor
                        ) / sqrt(mul_ir_in1.ir.dim)
                    else:
                        result = paddle.einsum(f"{z}uv,ijk,zuvij->zuk", w, w3j, xx)
                else:
                    result = paddle.einsum("ijk,zuvij->zuk", w3j, xx)
            elif ins.connection_mode == "uvv":
                assert mul_ir_in2.mul == mul_ir_out.mul
                if ins.has_weight:
                    if specialized_code and l1l2l3 == (0, 0, 0):
                        result = paddle.einsum(
                            f"{z}uv,zu,zv->zv",
                            w,
                            x1_tensor.reshape(batch_numel, mul_ir_in1.dim),
                            x2_tensor.reshape(batch_numel, mul_ir_in2.dim),
                        )
                    elif specialized_code and mul_ir_in1.ir.l == 0:
                        result = paddle.einsum(
                            f"{z}uv,zu,zvj->zvj",
                            w,
                            x1_tensor.reshape(batch_numel, mul_ir_in1.dim),
                            x2_tensor,
                        ) / sqrt(mul_ir_out.ir.dim)
                    elif specialized_code and mul_ir_in2.ir.l == 0:
                        result = paddle.einsum(
                            f"{z}uv,zui,zv->zvi",
                            w,
                            x1_tensor,
                            x2_tensor.reshape(batch_numel, mul_ir_in2.dim),
                        ) / sqrt(mul_ir_out.ir.dim)
                    elif specialized_code and mul_ir_out.ir.l == 0:
                        result = paddle.einsum(
                            f"{z}uv,zui,zvi->zv", w, x1_tensor, x2_tensor
                        ) / sqrt(mul_ir_in1.ir.dim)
                    else:
                        result = paddle.einsum(f"{z}uv,ijk,zuvij->zvk", w, w3j, xx)
                elif specialized_code and l1l2l3 == (0, 0, 0):
                    result = paddle.einsum(
                        "zu,zv->zv",
                        x1_tensor.reshape(batch_numel, mul_ir_in1.dim),
                        x2_tensor.reshape(batch_numel, mul_ir_in2.dim),
                    )
                elif specialized_code and mul_ir_in1.ir.l == 0:
                    result = paddle.einsum(
                        "zu,zvj->zvj",
                        x1_tensor.reshape(batch_numel, mul_ir_in1.dim),
                        x2_tensor,
                    ) / sqrt(mul_ir_out.ir.dim)
                elif specialized_code and mul_ir_in2.ir.l == 0:
                    result = paddle.einsum(
                        "zui,zv->zvi",
                        x1_tensor,
                        x2_tensor.reshape(batch_numel, mul_ir_in2.dim),
                    ) / sqrt(mul_ir_out.ir.dim)
                elif specialized_code and mul_ir_out.ir.l == 0:
                    result = paddle.einsum("zui,zvi->zv", x1_tensor, x2_tensor) / sqrt(
                        mul_ir_in1.ir.dim
                    )
                else:
                    result = paddle.einsum("ijk,zuvij->zvk", w3j, xx)
            elif ins.connection_mode == "uuw":
                assert mul_ir_in1.mul == mul_ir_in2.mul
                if ins.has_weight:
                    if specialized_code and l1l2l3 == (0, 0, 0):
                        result = paddle.einsum(
                            f"{z}uw,zu,zu->zw",
                            w,
                            x1_tensor.reshape(batch_numel, mul_ir_in1.dim),
                            x2_tensor.reshape(batch_numel, mul_ir_in2.dim),
                        )
                    elif specialized_code and mul_ir_in1.ir.l == 0:
                        result = paddle.einsum(
                            f"{z}uw,zu,zuj->zwj",
                            w,
                            x1_tensor.reshape(batch_numel, mul_ir_in1.dim),
                            x2_tensor,
                        ) / sqrt(mul_ir_out.ir.dim)
                    elif specialized_code and mul_ir_in2.ir.l == 0:
                        result = paddle.einsum(
                            f"{z}uw,zui,zu->zwi",
                            w,
                            x1_tensor,
                            x2_tensor.reshape(batch_numel, mul_ir_in2.dim),
                        ) / sqrt(mul_ir_out.ir.dim)
                    elif specialized_code and mul_ir_out.ir.l == 0:
                        result = paddle.einsum(
                            f"{z}uw,zui,zui->zw", w, x1_tensor, x2_tensor
                        ) / sqrt(mul_ir_in1.ir.dim)
                    else:
                        result = paddle.einsum(f"{z}uw,ijk,zuij->zwk", w, w3j, xx)
                else:
                    assert mul_ir_out.mul == 1
                    result = paddle.einsum("ijk,zuij->zk", w3j, xx)
            elif ins.connection_mode == "uuu":
                assert mul_ir_in1.mul == mul_ir_in2.mul == mul_ir_out.mul
                if ins.has_weight:
                    if specialized_code and l1l2l3 == (0, 0, 0):
                        result = paddle.einsum(
                            f"{z}u,zu,zu->zu",
                            w,
                            x1_tensor.reshape(batch_numel, mul_ir_in1.dim),
                            x2_tensor.reshape(batch_numel, mul_ir_in2.dim),
                        )
                    elif specialized_code and l1l2l3 == (1, 1, 1):
                        result = paddle.einsum(
                            f"{z}u,zui->zui",
                            w,
                            paddle.cross(x=x1_tensor, y=x2_tensor, axis=2),
                        ) / sqrt(2 * 3)
                    elif specialized_code and mul_ir_in1.ir.l == 0:
                        result = paddle.einsum(
                            f"{z}u,zu,zuj->zuj",
                            w,
                            x1_tensor.reshape(batch_numel, mul_ir_in1.dim),
                            x2_tensor,
                        ) / sqrt(mul_ir_out.ir.dim)
                    elif specialized_code and mul_ir_in2.ir.l == 0:
                        result = paddle.einsum(
                            f"{z}u,zui,zu->zui",
                            w,
                            x1_tensor,
                            x2_tensor.reshape(batch_numel, mul_ir_in2.dim),
                        ) / sqrt(mul_ir_out.ir.dim)
                    elif specialized_code and mul_ir_out.ir.l == 0:
                        result = paddle.einsum(
                            f"{z}u,zui,zui->zu", w, x1_tensor, x2_tensor
                        ) / sqrt(mul_ir_in1.ir.dim)
                    else:
                        result = paddle.einsum(f"{z}u,ijk,zuij->zuk", w, w3j, xx)
                elif specialized_code and l1l2l3 == (0, 0, 0):
                    result = paddle.einsum(
                        "zu,zu->zu",
                        x1_tensor.reshape(batch_numel, mul_ir_in1.dim),
                        x2_tensor.reshape(batch_numel, mul_ir_in2.dim),
                    )
                elif specialized_code and l1l2l3 == (1, 1, 1):
                    result = paddle.cross(x=x1_tensor, y=x2_tensor, axis=2) * (
                        1.0 / sqrt(2 * 3)
                    )
                elif specialized_code and mul_ir_in1.ir.l == 0:
                    result = paddle.einsum(
                        "zu,zuj->zuj",
                        x1_tensor.reshape(batch_numel, mul_ir_in1.dim),
                        x2_tensor,
                    ) / sqrt(mul_ir_out.ir.dim)
                elif specialized_code and mul_ir_in2.ir.l == 0:
                    result = paddle.einsum(
                        "zui,zu->zui",
                        x1_tensor,
                        x2_tensor.reshape(batch_numel, mul_ir_in2.dim),
                    ) / sqrt(mul_ir_out.ir.dim)
                elif specialized_code and mul_ir_out.ir.l == 0:
                    result = paddle.einsum("zui,zui->zu", x1_tensor, x2_tensor) / sqrt(
                        mul_ir_in1.ir.dim
                    )
                else:
                    result = paddle.einsum("ijk,zuij->zuk", w3j, xx)
            elif ins.connection_mode == "uvuv":
                assert mul_ir_in1.mul * mul_ir_in2.mul == mul_ir_out.mul
                if ins.has_weight:
                    result = paddle.einsum(f"{z}uv,ijk,zuvij->zuvk", w, w3j, xx)
                else:
                    result = paddle.einsum("ijk,zuvij->zuvk", w3j, xx)
            elif ins.connection_mode == "uvu<v":
                assert mul_ir_in1.mul == mul_ir_in2.mul
                assert mul_ir_in1.mul * (mul_ir_in1.mul - 1) // 2 == mul_ir_out.mul
                i = paddle.triu_indices(
                    row=mul_ir_in1.mul, col=mul_ir_in1.mul, offset=1
                )
                xx_subset = xx[:, i[0], i[1]]
                if ins.has_weight:
                    result = paddle.einsum(f"{z}w,ijk,zwij->zwk", w, w3j, xx_subset)
                else:
                    result = paddle.einsum("ijk,zwij->zwk", w3j, xx_subset)
            elif ins.connection_mode == "u<vw":
                assert mul_ir_in1.mul == mul_ir_in2.mul
                assert ins.has_weight
                i = paddle.triu_indices(
                    row=mul_ir_in1.mul, col=mul_ir_in1.mul, offset=1
                )
                xx_subset = xx[:, i[0], i[1]]
                result = paddle.einsum(f"{z}qw,ijk,zqij->zwk", w, w3j, xx_subset)
            result = ins.path_weight * result
            outputs.append(result.reshape(batch_numel, mul_ir_out.dim))
        final_outputs = []
        for i_out, mul_ir_out in enumerate(irreps_out):
            if mul_ir_out.mul > 0:
                relevant_outputs = [
                    out
                    for ins, out in zip(filtered_instructions, outputs)
                    if ins.i_out == i_out
                ]
                final_outputs.append(
                    _sum_tensors(
                        relevant_outputs, (batch_numel, mul_ir_out.dim), like=x1
                    )
                )
        if len(final_outputs) > 1:
            result = paddle.concat(x=final_outputs, axis=1)
        elif len(final_outputs) == 1:
            result = final_outputs[0]
        else:
            return paddle.zeros(shape=final_output_shape, dtype=x1.dtype)
        return result.reshape(final_output_shape)

    return tp_forward

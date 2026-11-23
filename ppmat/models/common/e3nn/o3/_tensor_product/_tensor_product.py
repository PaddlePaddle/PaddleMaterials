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

import warnings
from math import sqrt
from typing import Any
from typing import List
from typing import Optional
from typing import Union

import matplotlib
import matplotlib.pyplot as plt
import paddle
from matplotlib import patches
from matplotlib.path import Path

from ppmat.models.common.e3nn import o3
from ppmat.models.common.e3nn.util import prod

from ._codegen import codegen_tensor_product_left_right
from ._codegen import codegen_tensor_product_right
from ._instruction import Instruction


class TensorProduct(paddle.nn.Layer):
    instructions: List[Any]
    shared_weights: bool
    internal_weights: bool
    weight_numel: int

    def __init__(
        self,
        irreps_in1: o3.Irreps,
        irreps_in2: o3.Irreps,
        irreps_out: o3.Irreps,
        instructions: List[tuple],
        in1_var: Optional[Union[List[float], paddle.Tensor]] = None,
        in2_var: Optional[Union[List[float], paddle.Tensor]] = None,
        out_var: Optional[Union[List[float], paddle.Tensor]] = None,
        irrep_normalization: str = None,
        path_normalization: str = None,
        internal_weights: Optional[bool] = None,
        shared_weights: Optional[bool] = None,
        compile_left_right: bool = True,
        compile_right: bool = False,
        normalization=None,
        _specialized_code: Optional[bool] = None,
        _optimize_einsums: Optional[bool] = None,
    ):
        super().__init__()
        if normalization is not None:
            warnings.warn(
                "`normalization` have given up, please use `irrep_normalization`",
                DeprecationWarning,
            )
            irrep_normalization = normalization
        if irrep_normalization is None:
            irrep_normalization = "component"
        if path_normalization is None:
            path_normalization = "element"
        assert irrep_normalization in ["component", "norm", "none"]
        assert path_normalization in ["element", "path", "none"]
        self.irreps_in1 = o3.Irreps(irreps_in1)
        self.irreps_in2 = o3.Irreps(irreps_in2)
        self.irreps_out = o3.Irreps(irreps_out)
        instructions = [(x if len(x) == 6 else x + (1.0,)) for x in instructions]
        instructions = [
            Instruction(
                i_in1=i_in1,
                i_in2=i_in2,
                i_out=i_out,
                connection_mode=connection_mode,
                has_weight=has_weight,
                path_weight=path_weight,
                path_shape={
                    "uvw": (
                        self.irreps_in1[i_in1].mul,
                        self.irreps_in2[i_in2].mul,
                        self.irreps_out[i_out].mul,
                    ),
                    "uvu": (self.irreps_in1[i_in1].mul, self.irreps_in2[i_in2].mul),
                    "uvv": (self.irreps_in1[i_in1].mul, self.irreps_in2[i_in2].mul),
                    "uuw": (self.irreps_in1[i_in1].mul, self.irreps_out[i_out].mul),
                    "uuu": (self.irreps_in1[i_in1].mul,),
                    "uvuv": (self.irreps_in1[i_in1].mul, self.irreps_in2[i_in2].mul),
                    "uvu<v": (
                        self.irreps_in1[i_in1].mul
                        * (self.irreps_in2[i_in2].mul - 1)
                        // 2,
                    ),
                    "u<vw": (
                        self.irreps_in1[i_in1].mul
                        * (self.irreps_in2[i_in2].mul - 1)
                        // 2,
                        self.irreps_out[i_out].mul,
                    ),
                }[connection_mode],
            )
            for (
                i_in1,
                i_in2,
                i_out,
                connection_mode,
                has_weight,
                path_weight,
            ) in instructions
        ]
        if in1_var is None:
            in1_var = [(1.0) for _ in range(len(self.irreps_in1))]
        else:
            in1_var = [float(var) for var in in1_var]
            assert len(in1_var) == len(
                self.irreps_in1
            ), "The length of in1_var must be equal to the length of irreps_in1度"
        if in2_var is None:
            in2_var = [(1.0) for _ in range(len(self.irreps_in2))]
        else:
            in2_var = [float(var) for var in in2_var]
            assert len(in2_var) == len(
                self.irreps_in2
            ), "The length of in2_var must be equal to the length of irreps_in2"
        if out_var is None:
            out_var = [(1.0) for _ in range(len(self.irreps_out))]
        else:
            out_var = [float(var) for var in out_var]
            assert len(out_var) == len(
                self.irreps_out
            ), "The length of out_var must be equal to the length of irreps_out"

        def num_elements(ins):
            return {
                "uvw": self.irreps_in1[ins.i_in1].mul * self.irreps_in2[ins.i_in2].mul,
                "uvu": self.irreps_in2[ins.i_in2].mul,
                "uvv": self.irreps_in1[ins.i_in1].mul,
                "uuw": self.irreps_in1[ins.i_in1].mul,
                "uuu": 1,
                "uvuv": 1,
                "uvu<v": 1,
                "u<vw": self.irreps_in1[ins.i_in1].mul
                * (self.irreps_in2[ins.i_in2].mul - 1)
                // 2,
            }[ins.connection_mode]

        normalization_coefficients = []
        for ins in instructions:
            mul_ir_in1 = self.irreps_in1[ins.i_in1]
            mul_ir_in2 = self.irreps_in2[ins.i_in2]
            mul_ir_out = self.irreps_out[ins.i_out]
            assert mul_ir_in1.ir.p * mul_ir_in2.ir.p == mul_ir_out.ir.p
            assert (
                abs(mul_ir_in1.ir.l - mul_ir_in2.ir.l)
                <= mul_ir_out.ir.l
                <= mul_ir_in1.ir.l + mul_ir_in2.ir.l
            )
            if irrep_normalization == "component":
                alpha = mul_ir_out.ir.dim
            if irrep_normalization == "norm":
                alpha = mul_ir_in1.ir.dim * mul_ir_in2.ir.dim
            if irrep_normalization == "none":
                alpha = 1
            if path_normalization == "element":
                x = sum(
                    in1_var[i.i_in1] * in2_var[i.i_in2] * num_elements(i)
                    for i in instructions
                    if i.i_out == ins.i_out
                )
            if path_normalization == "path":
                x = in1_var[ins.i_in1] * in2_var[ins.i_in2] * num_elements(ins)
                x *= len([i for i in instructions if i.i_out == ins.i_out])
            if path_normalization == "none":
                x = 1
            if x > 0.0:
                alpha /= x
            alpha *= out_var[ins.i_out]
            alpha *= ins.path_weight
            normalization_coefficients += [sqrt(alpha)]
        self.instructions = [
            Instruction(
                ins.i_in1,
                ins.i_in2,
                ins.i_out,
                ins.connection_mode,
                ins.has_weight,
                alpha,
                ins.path_shape,
            )
            for ins, alpha in zip(instructions, normalization_coefficients)
        ]
        if internal_weights is None:
            internal_weights = False
        if shared_weights is None:
            shared_weights = True
        if not shared_weights and internal_weights:
            raise ValueError(
                "when internal_weights == True, the shared_weights must be True"
            )
        self.internal_weights = internal_weights
        self.shared_weights = shared_weights
        self.weight_numel = sum(
            prod(ins.path_shape) for ins in self.instructions if ins.has_weight
        )
        if internal_weights and self.weight_numel > 0:
            assert self.shared_weights, "Having internal weights impose shared weights"
            self.weight = paddle.base.framework.EagerParamBase.from_tensor(
                tensor=paddle.randn(shape=[self.weight_numel])
            )
        else:
            # Avoid registering zero-sized tensors as buffers: Paddle DataParallel
            # synchronizes buffers across ranks and does not support numel == 0.
            # For cases where weights are not used (weight_numel == 0) or are
            # provided externally (internal_weights == False), we create the
            # appropriate empty tensor on-the-fly in forward.
            self.weight = None
        if self.irreps_out.dim > 0:
            output_mask = paddle.concat(
                x=[
                    (
                        paddle.ones(shape=mul * ir.dim)
                        if any(
                            ins.i_out == i_out
                            and ins.path_weight != 0
                            and 0 not in ins.path_shape
                            for ins in self.instructions
                        )
                        else paddle.zeros(shape=mul * ir.dim)
                    )
                    for i_out, (mul, ir) in enumerate(self.irreps_out)
                ]
            )
        else:
            output_mask = paddle.ones(shape=[0])
        self.register_buffer(name="output_mask", tensor=output_mask)
        self._specialized_code = (
            True if _specialized_code is None else _specialized_code
        )
        self._optimize_einsums = (
            False if _optimize_einsums is None else _optimize_einsums
        )
        if compile_left_right:
            self._tp_forward = codegen_tensor_product_left_right(
                self.irreps_in1,
                self.irreps_in2,
                self.irreps_out,
                self.instructions,
                self.shared_weights,
                self._specialized_code,
                self._optimize_einsums,
            )
        else:
            self._tp_forward = None
        if compile_right:
            self._tp_right = codegen_tensor_product_right(
                self.irreps_in1,
                self.irreps_in2,
                self.irreps_out,
                self.instructions,
                self.shared_weights,
                self._specialized_code,
                self._optimize_einsums,
            )
        else:
            self._tp_right = None

    def __repr__(self):
        npath = sum(prod(i.path_shape) for i in self.instructions)
        return (
            f"{self.__class__.__name__}({self.irreps_in1.simplify()} x "
            f"{self.irreps_in2.simplify()} -> {self.irreps_out.simplify()} | "
            f"{npath} paths | {self.weight_numel} weights)"
        )

    def _prep_weights_python(
        self, weight: Optional[Union[paddle.Tensor, List[paddle.Tensor]]]
    ) -> Optional[paddle.Tensor]:
        if isinstance(weight, list):
            weight_shapes = [
                ins.path_shape for ins in self.instructions if ins.has_weight
            ]
            if not self.shared_weights:
                weight = [
                    w.reshape(-1, prod(shape))
                    for w, shape in zip(weight, weight_shapes)
                ]
            else:
                weight = [
                    w.reshape(prod(shape)) for w, shape in zip(weight, weight_shapes)
                ]
            return paddle.concat(x=weight, axis=-1)
        else:
            return weight

    def _get_weights(
        self,
        weight: Optional[Union[paddle.Tensor, List[paddle.Tensor]]],
        like: Optional[paddle.Tensor] = None,
    ) -> paddle.Tensor:
        weight = self._prep_weights_python(weight)
        if weight is None:
            if self.weight_numel > 0 and not self.internal_weights:
                raise RuntimeError(
                    "Weights must be provided when the TensorProduct does not "
                    "have `internal_weights`"
                )
            if self.weight is None:
                dtype = getattr(like, "dtype", None)
                place = getattr(like, "place", None)
                return paddle.to_tensor([], dtype=dtype, place=place)
            return self.weight

        if self.shared_weights:
            assert tuple(weight.shape) == (self.weight_numel,), "Invalid weight shape"
        else:
            assert tuple(weight.shape)[-1] == self.weight_numel, "Invalid weight shape"
            assert (
                weight.ndim > 1
            ), "When shared weights is false, weights must have batch dimension"
        return weight

    def right(self, y, weight=None):
        assert self._tp_right is not None, (
            "The right function is not compiled, please set compile_right=True "
            "when creating the TensorProduct."
        )
        assert (
            tuple(y.shape)[-1] == self.irreps_in2.dim
        ), f"The last dimension of y should be{self.irreps_in2.dim}"
        real_weight = self._get_weights(weight, like=y)
        return self._tp_right(y, real_weight)

    def forward(self, x, y, weight=None):
        assert self._tp_forward is not None, (
            "The forward function is not complied, please set "
            "compile_left_right=True when creating the TensorProduct"
        )
        assert (
            tuple(x.shape)[-1] == self.irreps_in1.dim
        ), f"The last dimension of x should be {self.irreps_in1.dim}"
        assert (
            tuple(y.shape)[-1] == self.irreps_in2.dim
        ), f"the last dimesion of y is {self.irreps_in2.dim}"
        real_weight = self._get_weights(weight, like=x)
        return self._tp_forward(x, y, real_weight)

    def weight_view_for_instruction(self, instruction_idx, weight=None):
        if not self.instructions[instruction_idx].has_weight:
            raise ValueError(f"{instruction_idx} have not weights")
        offset = sum(
            prod(ins.path_shape)
            for ins in self.instructions[:instruction_idx]
            if ins.has_weight
        )
        ins = self.instructions[instruction_idx]
        weight = self._get_weights(weight)
        batch_shape = tuple(weight.shape)[:-1]
        start_0 = weight.shape[-1] + offset if offset < 0 else offset
        return paddle.slice(
            weight, [-1], [start_0], [start_0 + prod(ins.path_shape)]
        ).view(batch_shape + ins.path_shape)

    def weight_views(self, weight=None, yield_instruction=False):
        weight = self._get_weights(weight)
        batch_shape = tuple(weight.shape)[:-1]
        offset = 0
        for ins_i, ins in enumerate(self.instructions):
            if ins.has_weight:
                flat_size = prod(ins.path_shape)
                start_1 = weight.shape[-1] + offset if offset < 0 else offset
                this_weight = paddle.slice(
                    weight, [-1], [start_1], [start_1 + flat_size]
                ).view(batch_shape + ins.path_shape)
                offset += flat_size
                if yield_instruction:
                    yield ins_i, ins, this_weight
                else:
                    yield this_weight

    def visualize(
        self,
        weight: Optional[paddle.Tensor] = None,
        plot_weight: bool = True,
        aspect_ratio=1,
        ax=None,
    ):
        """Visualize the connectivity of this `e3nn.o3.TensorProduct`

        Parameters
        ----------
        weight : `torch.Tensor`, optional
            like ``weight`` argument to ``forward()``

        plot_weight : `bool`, default True
            Whether to color paths by the sum of their weights.

        ax : ``matplotlib.Axes``, default None
            The axes to plot on. If ``None``, a new figure will be created.

        Returns
        -------
        (fig, ax)
            The figure and axes on which the plot was drawn.
        """
        import numpy as np

        def _intersection(x, u, y, v):
            u2 = np.sum(u**2)
            v2 = np.sum(v**2)
            uv = np.sum(u * v)
            det = u2 * v2 - uv**2
            mu = np.sum((u * uv - v * u2) * (y - x)) / det
            return y + mu * v

        if ax is None:
            ax = plt.gca()
        fig = ax.get_figure()
        verts = [
            np.array([np.cos(a * 2 * np.pi / 6), np.sin(a * 2 * np.pi / 6)])
            for a in range(6)
        ]
        verts = np.asarray(verts)
        if not (aspect_ratio in ["auto"] or isinstance(aspect_ratio, (float, int))):
            raise ValueError(
                f"aspect_ratio must be 'auto' or a float or int, got {aspect_ratio}"
            )
        if aspect_ratio == "auto":
            factor = 0.2 / 2
            min_aspect = 1 / 2
            h_factor = max(len(self.irreps_in2), len(self.irreps_in1))
            w_factor = len(self.irreps_out)
            if h_factor / w_factor < min_aspect:
                h_factor = min_aspect * w_factor
            verts[:, 1] *= h_factor * factor
            verts[:, 0] *= w_factor * factor
        if isinstance(aspect_ratio, (float, int)):
            factor = 0.1 * max(
                len(self.irreps_in2), len(self.irreps_in1), len(self.irreps_out)
            )
            verts[:, 1] *= factor
            verts[:, 0] *= aspect_ratio * factor
        codes = [
            Path.MOVETO,
            Path.LINETO,
            Path.MOVETO,
            Path.LINETO,
            Path.MOVETO,
            Path.LINETO,
        ]
        path = Path(verts, codes)
        patch = patches.PathPatch(path, facecolor="none", lw=1, zorder=2)
        ax.add_patch(patch)
        n = len(self.irreps_in1)
        b, a = verts[2:4]
        c_in1 = (a + b) / 2
        s_in1 = [(a + (i + 1) / (n + 1) * (b - a)) for i in range(n)]
        n = len(self.irreps_in2)
        b, a = verts[:2]
        c_in2 = (a + b) / 2
        s_in2 = [(a + (i + 1) / (n + 1) * (b - a)) for i in range(n)]
        n = len(self.irreps_out)
        a, b = verts[4:6]
        s_out = [(a + (i + 1) / (n + 1) * (b - a)) for i in range(n)]
        if weight is None and not self.internal_weights:
            plot_weight = False
        elif plot_weight:
            with paddle.no_grad():
                path_weight = []
                for ins_i, ins in enumerate(self.instructions):
                    if ins.has_weight:
                        this_weight = self.weight_view_for_instruction(
                            ins_i, weight=weight
                        ).cpu()
                        path_weight.append(this_weight.pow(y=2).mean())
                    else:
                        path_weight.append(0)
                path_weight = np.asarray(path_weight)
                path_weight /= np.abs(path_weight).max()
        cmap = matplotlib.cm.get_cmap("Blues")
        for ins_index, ins in enumerate(self.instructions):
            y = _intersection(s_in1[ins.i_in1], c_in1, s_in2[ins.i_in2], c_in2)
            verts = []
            codes = []
            verts += [s_out[ins.i_out], y]
            codes += [Path.MOVETO, Path.LINETO]
            verts += [s_in1[ins.i_in1], y]
            codes += [Path.MOVETO, Path.LINETO]
            verts += [s_in2[ins.i_in2], y]
            codes += [Path.MOVETO, Path.LINETO]
            if plot_weight:
                color = (
                    cmap(0.5 + 0.5 * path_weight[ins_index])
                    if ins.has_weight
                    else "black"
                )
            else:
                color = "green" if ins.has_weight else "black"
            ax.add_patch(
                patches.PathPatch(
                    Path(verts, codes),
                    facecolor="none",
                    edgecolor=color,
                    alpha=0.5,
                    ls="-",
                    lw=1.5,
                )
            )
        padding = 3
        fontsize = 10

        def format_ir(mul_ir):
            if mul_ir.mul == 1:
                return f"${mul_ir.ir}$"
            return f"${mul_ir.mul} \\times {mul_ir.ir}$"

        for i, mul_ir in enumerate(self.irreps_in1):
            ax.annotate(
                format_ir(mul_ir),
                s_in1[i],
                horizontalalignment="right",
                textcoords="offset points",
                xytext=(-padding, 0),
                fontsize=fontsize,
            )
        for i, mul_ir in enumerate(self.irreps_in2):
            ax.annotate(
                format_ir(mul_ir),
                s_in2[i],
                horizontalalignment="left",
                textcoords="offset points",
                xytext=(padding, 0),
                fontsize=fontsize,
            )
        for i, mul_ir in enumerate(self.irreps_out):
            ax.annotate(
                format_ir(mul_ir),
                s_out[i],
                horizontalalignment="center",
                verticalalignment="top",
                rotation=90,
                textcoords="offset points",
                xytext=(0, -padding),
                fontsize=fontsize,
            )
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.axis("equal")
        ax.axis("off")
        return fig, ax


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


def view(self, *args, **kwargs):
    if args:
        if len(args) == 1 and isinstance(args[0], str):
            return paddle.view(self, args[0])
        if len(args) == 1 and isinstance(args[0], (tuple, list)):
            return paddle.reshape(self, args[0])
        else:
            return paddle.reshape(self, list(args))
    elif kwargs:
        shape_or_dtype = list(kwargs.values())[0]
        if isinstance(shape_or_dtype, str):
            return paddle.view(self, shape_or_dtype)
        return paddle.reshape(self, shape=shape_or_dtype)


setattr(paddle.Tensor, "view", view)


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

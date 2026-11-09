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

from typing import Tuple

import paddle

from ..o3._irreps import Irrep, Irreps
from ..util.codegen import CodeGenMixin
from ..util.jit import compile_mode


@compile_mode("script")
class Extract(CodeGenMixin, paddle.nn.Module):
    def __init__(
        self, irreps_in, irreps_outs, instructions, squeeze_out: bool = False
    ) -> None:
        """Extract sub sets of irreps

        ``forward`` applies the slice instructions directly (paddle.slice +
        concat); no codegen submodule is registered.

        Parameters
        ----------
        irreps_in : `e3nn.o3.Irreps`
            representation of the input

        irreps_outs : list of `e3nn.o3.Irreps`
            list of representation of the outputs

        instructions : list of tuple of int
            list of tuples, one per output continaing each ``len(irreps_outs[i])`` int

        squeeze_out : bool, default False
            if ``squeeze_out`` and only one output exists, a ``torch.Tensor`` will be returned instead of a
            ``Tuple[torch.Tensor]``


        Examples
        --------

        >>> c = Extract('1e + 0e + 0e', ['0e', '0e'], [(1,), (2,)])
        >>> c(torch.tensor([0.0, 0.0, 0.0, 1.0, 2.0]))
        (tensor([1.]), tensor([2.]))
        """
        super().__init__()
        self.irreps_in = Irreps(irreps_in)
        self.irreps_outs = tuple(Irreps(irreps) for irreps in irreps_outs)
        self.instructions = instructions
        assert len(self.irreps_outs) == len(self.instructions)
        for irreps_out, ins in zip(self.irreps_outs, self.instructions):
            assert len(irreps_out) == len(ins)
        self._squeeze_out = squeeze_out
        self._full_perm = tuple(range(len(self.irreps_in)))

    def forward(self, x: paddle.Tensor):
        outs = []
        for irreps_out, ins in zip(self.irreps_outs, self.instructions):
            if tuple(ins) == self._full_perm:
                outs.append(x)
                continue
            pieces = []
            for s_out, i_in in zip(irreps_out.slices(), ins):
                i_start = self.irreps_in[:i_in].dim
                i_len = self.irreps_in[i_in].dim
                pieces.append(paddle.slice(x, [-1], [i_start], [i_start + i_len]))
            if len(pieces) > 1:
                outs.append(paddle.concat(pieces, dim=-1))
            elif len(pieces) == 1:
                outs.append(pieces[0])
            else:
                outs.append(
                    paddle.zeros(
                        list(x.shape[:-1]) + [irreps_out.dim], dtype=x.dtype
                    )
                )
        if self._squeeze_out and len(outs) == 1:
            return outs[0]
        return tuple(outs)


@compile_mode("script")
class ExtractIr(Extract):
    def __init__(self, irreps_in, ir) -> None:
        """Extract ``ir`` from irreps

        Parameters
        ----------
        irreps_in : `e3nn.o3.Irreps`
            representation of the input

        ir : `e3nn.o3.Irrep`
            representation to extract
        """
        ir = Irrep(ir)
        irreps_in = Irreps(irreps_in)
        self.irreps_out = Irreps([mul_ir for mul_ir in irreps_in if mul_ir.ir == ir])
        instructions = [
            tuple(i for i, mul_ir in enumerate(irreps_in) if mul_ir.ir == ir)
        ]
        super().__init__(irreps_in, [self.irreps_out], instructions, squeeze_out=True)

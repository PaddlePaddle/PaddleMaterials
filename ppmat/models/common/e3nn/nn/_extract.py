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

import paddle

from ppmat.models.common.e3nn import o3


class Extract(paddle.nn.Layer):
    def __init__(self, irreps_in, irreps_outs, instructions, squeeze_out: bool = False):
        r"""Extract sub sets of irreps

        Parameters
        ----------
        irreps_in : `e3nn.o3.Irreps`
            representation of the input

        irreps_outs : list of `e3nn.o3.Irreps`
            list of representation of the outputs

        instructions : list of tuple of int
            list of tuples, one per output continaing each ``len(irreps_outs[i])`` int

        squeeze_out : bool, default False
            if ``squeeze_out`` and only one output exists, a ``paddle.Tensor`` will be returned instead of a
            ``Tuple[paddle.Tensor]``
        """
        super().__init__()
        self.irreps_in = o3.Irreps(irreps_in)
        self.irreps_outs = tuple(o3.Irreps(irreps) for irreps in irreps_outs)
        self.instructions = instructions
        self.squeeze_out = squeeze_out

        assert len(self.irreps_outs) == len(self.instructions)
        for irreps_out, ins in zip(self.irreps_outs, self.instructions):
            assert len(irreps_out) == len(ins)

    def forward(self, x):
        assert x.shape[-1] == self.irreps_in.dim, "invalid input shape"

        # Assemble each output from slices joined along the last axis;
        # slice/concat keep the double-grad path of derivative-output
        # models alive where in-place scatter assignment does not.
        outs = []
        for irreps_out, ins in zip(self.irreps_outs, self.instructions):
            if tuple(ins) == tuple(range(len(self.irreps_in))):
                outs.append(x)
                continue
            pieces = []
            for s_out, i_in in zip(irreps_out.slices(), ins):
                i_start = self.irreps_in[:i_in].dim
                i_len = self.irreps_in[i_in].dim
                pieces.append(paddle.slice(x, [-1], [i_start], [i_start + i_len]))
            if len(pieces) > 1:
                outs.append(paddle.concat(pieces, axis=-1))
            elif len(pieces) == 1:
                outs.append(pieces[0])
            else:
                outs.append(
                    paddle.zeros(
                        list(x.shape[:-1]) + [irreps_out.dim], dtype=x.dtype
                    )
                )

        if self.squeeze_out and len(outs) == 1:
            return outs[0]
        return tuple(outs)


class ExtractIr(Extract):
    def __init__(self, irreps_in, ir):
        r"""Extract ``ir`` from irreps

        Parameters
        ----------
        irreps_in : `e3nn.o3.Irreps`
            representation of the input

        ir : `e3nn.o3.Irrep`
            representation to extract
        """
        ir = o3.Irrep(ir)
        irreps_in = o3.Irreps(irreps_in)
        self.irreps_out = o3.Irreps([mul_ir for mul_ir in irreps_in if mul_ir.ir == ir])
        instructions = [
            tuple(i for i, mul_ir in enumerate(irreps_in) if mul_ir.ir == ir)
        ]

        super().__init__(irreps_in, [self.irreps_out], instructions, squeeze_out=True)

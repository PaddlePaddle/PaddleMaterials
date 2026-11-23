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

        out = []
        input_slices = self.irreps_in.slices()
        for irreps_out, instructions in zip(self.irreps_outs, self.instructions):
            if instructions == tuple(range(len(input_slices))):
                out.append(paddle.assign(x))
                continue
            # Building the output from slices avoids slice assignment, whose
            # backward op cannot be differentiated for force losses.
            parts = [
                x.slice([-1], [input_slices[i].start], [input_slices[i].stop])
                for i in instructions
            ]
            if not parts:
                result = x.slice([-1], [0], [0])
            elif len(parts) == 1:
                result = parts[0]
            else:
                result = paddle.concat(parts, axis=-1)
            if result.shape[-1] != irreps_out.dim:
                raise ValueError("Extract instructions do not match output irreps")
            out.append(result)

        if self.squeeze_out and len(out) == 1:
            return out[0]
        return tuple(out)


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

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

        # Avoid Tensor slice assignment here.
        #
        # The previous implementation first created zero tensors and then
        # filled slices with:
        #
        #     out[i][...] = x_slice
        #
        # Paddle lowers that operation to set_value_with_tensor. Its gradient
        # currently has no second-order gradient implementation, which blocks
        # force/stress training where SevenNet requires higher-order autograd.
        #
        # Each output consists only of selected, ordered slices of the input,
        # so it is mathematically equivalent to concatenate those slices.
        out = []

        full_instruction = tuple(
            range(
                len(
                    self.irreps_in
                )
            )
        )

        for irreps_out, ins in zip(
            self.irreps_outs,
            self.instructions,
        ):

            if ins == full_instruction:

                # No copy or assignment is required for the identity case.
                out_tensor = x

            else:

                pieces = []

                for i_in in ins:

                    i_start = (
                        self.irreps_in[
                            :i_in
                        ].dim
                    )

                    i_len = (
                        self.irreps_in[
                            i_in
                        ].dim
                    )

                    x_slice = x.slice(
                        [-1],
                        [i_start],
                        [
                            i_start
                            + i_len
                        ],
                    )

                    pieces.append(
                        x_slice
                    )

                if len(pieces) == 0:

                    out_tensor = paddle.zeros(
                        list(
                            x.shape[:-1]
                        )
                        + [
                            irreps_out.dim
                        ],
                        dtype=x.dtype,
                    )

                elif len(pieces) == 1:

                    out_tensor = pieces[0]

                else:

                    out_tensor = paddle.concat(
                        pieces,
                        axis=-1,
                    )

            if (
                out_tensor.shape[-1]
                != irreps_out.dim
            ):

                raise RuntimeError(
                    "Extract output dimension mismatch: "
                    f"expected {irreps_out.dim}, "
                    f"got {out_tensor.shape[-1]}"
                )

            out.append(
                out_tensor
            )

        if (
            self.squeeze_out
            and len(out) == 1
        ):
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

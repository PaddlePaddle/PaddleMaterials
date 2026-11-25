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

"""Paddle port of ``mace.modules.irreps_tools`` (the subset used by
:class:`~ppmat.models.mace.blocks.RealAgnosticResidualInteractionBlock`)."""

import paddle
from paddle import nn

from ppmat.models.common.e3nn import o3

__all__ = ["tp_out_irreps_with_instructions", "reshape_irreps"]


def tp_out_irreps_with_instructions(irreps1, irreps2, target_irreps):
    """Enumerate the ``"uvu"`` tensor product paths that end up in
    ``target_irreps``.

    Returns the sorted output irreps and the corresponding instruction list,
    exactly as ``mace.modules.irreps_tools.tp_out_irreps_with_instructions``.
    """
    irreps1 = o3.Irreps(irreps1)
    irreps2 = o3.Irreps(irreps2)
    target_irreps = o3.Irreps(target_irreps)

    trainable = True
    irreps_out_list = []
    instructions = []
    for i, (mul, ir_in) in enumerate(irreps1):
        for j, (_, ir_edge) in enumerate(irreps2):
            for ir_out in ir_in * ir_edge:
                if ir_out in target_irreps:
                    k = len(irreps_out_list)
                    irreps_out_list.append((mul, ir_out))
                    instructions.append((i, j, k, "uvu", trainable))

    irreps_out = o3.Irreps(irreps_out_list)
    irreps_out, permut, _ = irreps_out.sort()

    instructions = [
        (i1, i2, permut[i_out], mode, train)
        for i1, i2, i_out, mode, train in instructions
    ]

    instructions = sorted(instructions, key=lambda x: x[2])
    return irreps_out, instructions


class reshape_irreps(nn.Layer):
    """Reshape ``[batch, irreps.dim]`` into ``[batch, mul * irreps.dim]`` blocks.

    Mirrors ``mace.modules.irreps_tools.reshape_irreps`` (the ``mul_ir``
    layout); the module has no parameters, so it does not appear in the
    checkpoint.
    """

    def __init__(self, irreps) -> None:
        super().__init__()
        self.irreps = o3.Irreps(irreps)
        self.dims = []
        self.muls = []
        for mul, ir in self.irreps:
            self.dims.append(ir.dim)
            self.muls.append(mul)

    def forward(self, tensor: paddle.Tensor) -> paddle.Tensor:
        ix = 0
        out = []
        batch = tensor.shape[0]
        for mul, d in zip(self.muls, self.dims):
            field = tensor[:, ix : ix + mul * d]
            ix += mul * d
            out.append(field.reshape([batch, mul, d]))
        return paddle.concat(out, axis=-1)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.irreps})"
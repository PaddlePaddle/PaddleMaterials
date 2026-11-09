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

import paddle
from ..math import normalize2mom
from ..o3._irreps import Irreps
from ..util.jit import compile_mode


@compile_mode("trace")
class Activation(paddle.nn.Module):
    """Scalar activation function.

    Odd scalar inputs require activation functions with a defined parity (odd or even).

    Parameters
    ----------
    irreps_in : `e3nn.o3.Irreps`
        representation of the input

    acts : list of function or None
        list of activation functions, `None` if non-scalar or identity

    Examples
    --------

    >>> a = Activation("256x0o", [torch.abs])
    >>> a.irreps_out
    256x0e

    >>> a = Activation("256x0o+16x1e", [None, None])
    >>> a.irreps_out
    256x0o+16x1e
    """

    def __init__(self, irreps_in, acts) -> None:
        super().__init__()
        irreps_in = Irreps(irreps_in)
        if len(irreps_in) != len(acts):
            raise ValueError(
                f"Irreps in and number of activation functions does not match: {len(acts), (irreps_in, acts)}"
            )
        acts = [(normalize2mom(act) if act is not None else None) for act in acts]
        from ..util._argtools import _get_device

        irreps_out = []
        for (mul, (l_in, p_in)), act in zip(irreps_in, acts):
            if act is not None:
                if l_in != 0:
                    raise ValueError(
                        "Activation: cannot apply an activation function to a non-scalar input."
                    )
                x = paddle.linspace(0, 10, 256, device=_get_device(act))
                a1, a2 = act(x), act(-x)
                if (a1 - a2).abs()._max() < 1e-05:
                    p_act = 1
                elif (a1 + a2).abs()._max() < 1e-05:
                    p_act = -1
                else:
                    p_act = 0
                p_out = p_act if p_in == -1 else p_in
                irreps_out.append((mul, (0, p_out)))
                if p_out == 0:
                    raise ValueError(
                        "Activation: the parity is violated! The input scalar is odd but the activation is neither even nor odd."
                    )
            else:
                irreps_out.append((mul, (l_in, p_in)))
        self.irreps_in = irreps_in
        self.irreps_out = Irreps(irreps_out)
        self.acts = paddle.nn.ModuleList(acts)
        self.paths = [
            (mul, (l, p), act) for (mul, (l, p)), act in zip(self.irreps_in, self.acts)
        ]
        assert len(self.irreps_in) == len(self.acts)

    def __repr__(self) -> str:
        acts = "".join([("x" if a is not None else " ") for a in self.acts])
        return f"{self.__class__.__name__} [{acts}] ({self.irreps_in} -> {self.irreps_out})"

    def forward(self, features, dim: int = -1):
        """evaluate

        Parameters
        ----------
        features : `torch.Tensor`
            tensor of shape ``(...)``

        Returns
        -------
        `torch.Tensor`
            tensor of shape the same shape as the input
        """
        output = []
        index = 0
        for mul, (l, _), act in self.paths:
            ir_dim = 2 * l + 1
            # narrow -> paddle.slice direct (negative dim supported)
            if act is not None:
                output.append(
                    act(paddle.slice(features, [dim], [index], [index + mul]))
                )
            else:
                output.append(
                    paddle.slice(features, [dim], [index], [index + mul * ir_dim])
                )
            index += mul * ir_dim
        if len(output) > 1:
            return paddle.cat(output, dim=dim)
        elif len(output) == 1:
            return output[0]
        else:
            return paddle.zeros_like(features)

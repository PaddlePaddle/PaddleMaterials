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
from ppmat.models.common.e3nn.math import normalize2mom
from ppmat.models.common.e3nn.paddle_utils import *


class Activation(paddle.nn.Layer):
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
    """

    def __init__(self, irreps_in, acts, sample_stream: str = "paddle"):
        super().__init__()
        irreps_in = o3.Irreps(irreps_in)  #
        if len(irreps_in) != len(acts):  #
            raise ValueError(
                f"Irreps in and number of activation functions does not match: {len(acts), (irreps_in, acts)}"
            )
        acts = [
            (normalize2mom(act, sample_stream=sample_stream)
             if act is not None else None)
            for act in acts
        ]

        irreps_out = []
        for (mul, (l_in, p_in)), act in zip(irreps_in, acts):
            if act is not None:
                if l_in != 0:
                    raise ValueError(
                        "Activation: cannot apply an activation function to a non-scalar input."
                    )
                x = paddle.linspace(start=0, stop=10, num=256)  #
                a1, a2 = act(x), act(-x)
                if (a1 - a2).abs().max_func() < 1e-05:
                    p_act = 1
                elif (a1 + a2).abs().max_func() < 1e-05:
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
        self.irreps_out = o3.Irreps(irreps_out)
        self.acts = paddle.nn.LayerList(sublayers=acts)
        assert len(self.irreps_in) == len(self.acts)

    def __repr__(self):
        acts = "".join([("x" if a is not None else " ") for a in self.acts])
        return f"{self.__class__.__name__} [{acts}] ({self.irreps_in} -> {self.irreps_out})"

    def forward(self, features, dim=-1):
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
        for (mul, ir), act in zip(self.irreps_in, self.acts):
            if act is not None:
                start_0 = features.shape[dim] + index if index < 0 else index
                output.append(
                    act(paddle.slice(features, [dim], [start_0], [start_0 + mul]))
                )
            else:
                start_1 = features.shape[dim] + index if index < 0 else index
                output.append(
                    paddle.slice(features, [dim], [start_1], [start_1 + mul * ir.dim])
                )
            index += mul * ir.dim
        if len(output) > 1:
            return paddle.concat(x=output, axis=dim)
        elif len(output) == 1:
            return output[0]
        else:
            return paddle.zeros_like(x=features)

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
from ppmat.models.common.e3nn.nn import Activation
from ppmat.models.common.e3nn.nn import Extract


class _Sortcut(paddle.nn.Layer):
    def __init__(self, *irreps_outs):
        super().__init__()
        self.irreps_outs = tuple(o3.Irreps(irreps).simplify() for irreps in irreps_outs)
        irreps_in = sum(self.irreps_outs, o3.Irreps([]))
        i = 0
        instructions = []
        for irreps_out in self.irreps_outs:
            instructions += [tuple(range(i, i + len(irreps_out)))]
            i += len(irreps_out)
        assert len(irreps_in) == i, (len(irreps_in), i)
        # Python-side (l, p) stable sort of the segments, replacing the
        # paddle.sort call that cannot consume Irreps objects.
        segments = sorted(
            ((ir, i, mul) for i, (mul, ir) in enumerate(irreps_in)),
            key=lambda t: (t[0].l, t[0].p),
        )
        irreps_in = o3.Irreps([(mul, ir) for ir, i, mul in segments])
        inv = tuple(i for _, i, _ in segments)
        perm = [0] * len(inv)
        for k, i in enumerate(inv):
            perm[i] = k
        instructions = [tuple(perm[i] for i in x) for x in instructions]
        self.cut = Extract(irreps_in, self.irreps_outs, instructions)
        self.irreps_in = irreps_in.simplify()

    def forward(self, x):
        return self.cut(x)


class Gate(paddle.nn.Layer):
    """Gate activation function.

    The gate activation is a direct sum of two sets of irreps. The first set
    of irreps is ``irreps_scalars`` passed through activation functions
    ``act_scalars``. The second set of irreps is ``irreps_gated`` multiplied
    by the scalars ``irreps_gates`` passed through activation functions
    ``act_gates``. Mathematically, this can be written as:

    .. math::
        \\left(\\bigoplus_i \\phi_i(x_i) \\right) \\oplus \\left(\\bigoplus_j \\phi_j(g_j) y_j \\right)

    where :math:`x_i` and :math:`\\phi_i` are from ``irreps_scalars`` and
    ``act_scalars``, and :math:`g_j`, :math:`\\phi_j`, and :math:`y_j` are
    from ``irreps_gates``, ``act_gates``, and ``irreps_gated``.

    The parameters passed in should adhere to the following conditions:

    1. ``len(irreps_scalars) == len(act_scalars)``.
    2. ``len(irreps_gates) == len(act_gates)``.
    3. ``irreps_gates.num_irreps == irreps_gated.num_irreps``.

    Parameters
    ----------
    irreps_scalars : `e3nn.o3.Irreps`
        Representation of the scalars that will be passed through the
        activation functions ``act_scalars``.

    act_scalars : list of function or None
        Activation functions acting on the scalars.

    irreps_gates : `e3nn.o3.Irreps`
        Representation of the scalars that will be passed through the
        activation functions ``act_gates`` and multiplied by the
        ``irreps_gated``.

    act_gates : list of function or None
        Activation functions acting on the gates. The number of functions in
        the list should match the number of irrep groups in ``irreps_gates``.

    irreps_gated : `e3nn.o3.Irreps`
        Representation of the gated tensors.
        ``irreps_gates.num_irreps == irreps_gated.num_irreps``

    Examples
    --------

    >>> g = Gate("16x0o", [torch.tanh], "32x0o", [torch.tanh], "16x1e+16x1o")
    >>> g.irreps_out
    16x0o+16x1o+16x1e
    """

    def __init__(
        self, irreps_scalars, act_scalars, irreps_gates, act_gates, irreps_gated,
        sample_stream: str = "paddle",
    ):
        super().__init__()
        irreps_scalars = o3.Irreps(irreps_scalars)
        irreps_gates = o3.Irreps(irreps_gates)
        irreps_gated = o3.Irreps(irreps_gated)
        if len(irreps_gates) > 0 and irreps_gates.lmax > 0:
            raise ValueError(
                f"Gate scalars must be scalars, instead got irreps_gates = {irreps_gates}"
            )
        if len(irreps_scalars) > 0 and irreps_scalars.lmax > 0:
            raise ValueError(
                f"Scalars must be scalars, instead got irreps_scalars = {irreps_scalars}"
            )
        if irreps_gates.num_irreps != irreps_gated.num_irreps:
            raise ValueError(
                f"There are {irreps_gated.num_irreps} irreps in irreps_gated, but a different number ({irreps_gates.num_irreps}) of gate scalars in irreps_gates"
            )
        self.sc = _Sortcut(irreps_scalars, irreps_gates, irreps_gated)
        self.irreps_scalars, self.irreps_gates, self.irreps_gated = self.sc.irreps_outs
        self._irreps_in = self.sc.irreps_in
        self.act_scalars = Activation(
            irreps_scalars, act_scalars, sample_stream=sample_stream
        )
        irreps_scalars = self.act_scalars.irreps_out
        self.act_gates = Activation(
            irreps_gates, act_gates, sample_stream=sample_stream
        )
        irreps_gates = self.act_gates.irreps_out
        self.mul = o3.ElementwiseTensorProduct(irreps_gated, irreps_gates)
        irreps_gated = self.mul.irreps_out
        self._irreps_out = irreps_scalars + irreps_gated

    def __repr__(self):
        return f"{self.__class__.__name__} ({self.irreps_in} -> {self.irreps_out})"

    def forward(self, features):
        """Evaluate the gated activation function.

        Parameters
        ----------
        features : `torch.Tensor`
            tensor of shape ``(..., irreps_in.dim)``

        Returns
        -------
        `torch.Tensor`
            tensor of shape ``(..., irreps_out.dim)``
        """
        scalars, gates, gated = self.sc(features)
        scalars = self.act_scalars(scalars)
        if tuple(gates.shape)[-1]:
            gates = self.act_gates(gates)
            gated = self.mul(gated, gates)
            features = paddle.concat(x=[scalars, gated], axis=-1)
        else:
            features = scalars
        return features

    @property
    def irreps_in(self):
        """Input representations."""
        return self._irreps_in

    @property
    def irreps_out(self):
        """Output representations."""
        return self._irreps_out

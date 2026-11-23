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


def swish(x: paddle.Tensor) -> paddle.Tensor:
    """Apply unscaled Swish using its explicit, unfused expression."""

    return x * paddle.nn.functional.sigmoid(x)


class ScaledSiLU(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.scale_factor = 1 / 0.6
        self._activation = paddle.nn.Silu()

    def forward(self, x: paddle.Tensor):
        return self._activation(x) * self.scale_factor


class SiQU(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self._activation = paddle.nn.Silu()

    def forward(self, x: paddle.Tensor):
        return x * self._activation(x)

class ScalarActivation(paddle.nn.Layer):
    """
    Use the invariant scalar features to gate higher order equivariant features.
    Adapted from `e3nn.nn.Gate`.
    """

    def __init__(self, irreps_in, act_scalars, act_gates):
        """
        :param irreps_in: input representations
        :param act_scalars: scalar activation function
        :param act_gates: gate activation function (for higher order features)
        """
        super(ScalarActivation, self).__init__()
        self.irreps_in = o3.Irreps(irreps_in)
        self.num_spherical = len(self.irreps_in)
        irreps_scalars = self.irreps_in[0:1]
        irreps_gates = irreps_scalars * (self.num_spherical - 1)
        irreps_gated = self.irreps_in[1:]
        self.act_scalars = Activation(irreps_scalars, [act_scalars])
        self.act_gates = Activation(
            irreps_gates, [act_gates] * (self.num_spherical - 1)
        )
        self.extract = Extract(
            self.irreps_in,
            [irreps_scalars, irreps_gated],
            instructions=[(0,), tuple(range(1, self.irreps_in.lmax + 1))],
        )
        self.mul = o3.ElementwiseTensorProduct(irreps_gates, irreps_gated)

    def forward(self, features):
        scalars, gated = self.extract(features)
        scalars_out = self.act_scalars(scalars)
        if tuple(gated.shape)[-1]:
            gates = self.act_gates(
                scalars.tile(repeat_times=[1, self.num_spherical - 1])
            )
            gated_out = self.mul(gates, gated)
            features = paddle.concat(x=[scalars_out, gated_out], axis=-1)
        else:
            features = scalars_out
        return features


class NormActivation(paddle.nn.Layer):
    """
    Use the norm of the higher order equivariant features to gate themselves.
    Idea from the TFN paper.
    """

    def __init__(
        self,
        irreps_in,
        act_scalars=paddle.nn.functional.silu,
        act_vectors=paddle.nn.functional.sigmoid,
    ):
        """
        :param irreps_in: input representations
        :param act_scalars: scalar activation function
        :param act_vectors: vector activation function (for the norm of higher order features)
        """
        super(NormActivation, self).__init__()
        self.irreps_in = o3.Irreps(irreps_in)
        self.scalar_irreps = self.irreps_in[0:1]
        self.vector_irreps = self.irreps_in[1:]
        self.act_scalars = act_scalars
        self.act_vectors = act_vectors
        self.scalar_idx = self.irreps_in[0].mul
        inner_out = o3.Irreps([(mul, (0, 1)) for mul, _ in self.vector_irreps])
        self.inner_prod = o3.TensorProduct(
            self.vector_irreps,
            self.vector_irreps,
            inner_out,
            [(i, i, i, "uuu", False) for i in range(len(self.vector_irreps))],
        )
        self.mul = o3.ElementwiseTensorProduct(inner_out, self.vector_irreps)

    def forward(self, features):
        scalars = self.act_scalars(features[..., : self.scalar_idx])
        vectors = features[..., self.scalar_idx :]
        norm = paddle.sqrt(x=self.inner_prod(vectors, vectors) + 1e-08)
        act = self.act_vectors(norm)
        vectors_out = self.mul(act, vectors)
        return paddle.concat(x=[scalars, vectors_out], axis=-1)

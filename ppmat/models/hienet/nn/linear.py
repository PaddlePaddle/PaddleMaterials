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

from .. import _keys as KEY
import paddle
from ppmat.models.common.e3nn.nn import FullyConnectedNet
from ppmat.models.common.e3nn.o3 import Irreps, Linear
from ppmat.models.common.e3nn.util.jit import compile_mode
from .._const import AtomGraphDataType
from .second_order import ManualDropout
from ppmat.utils.scatter import scatter


@compile_mode("script")
class IrrepsLinear(paddle.nn.Module):
    """
    wrapper class of e3nn Linear to operate on AtomGraphData
    """

    def __init__(
        self,
        irreps_in: Irreps,
        irreps_out: Irreps,
        data_key_in: str,
        data_key_out: str = None,
        **e3nn_linear_params
    ):
        super().__init__()
        self.key_input = data_key_in
        if data_key_out is None:
            self.key_output = data_key_in
        else:
            self.key_output = data_key_out
        self.linear = Linear(irreps_in, irreps_out, **e3nn_linear_params)

    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:
        data[self.key_output] = self.linear(data[self.key_input])
        return data


@compile_mode("script")
class AtomReduce(paddle.nn.Module):
    """
    atomic energy -> total energy
    constant is multiplied to data
    """

    def __init__(
        self, data_key_in: str, data_key_out: str, reduce="sum", constant: float = 1.0
    ):
        super().__init__()
        self.key_input = data_key_in
        self.key_output = data_key_out
        self.constant = constant
        self.reduce = reduce
        self._is_batch_data = True

    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:
        if self._is_batch_data:
            data[self.key_output] = (
                scatter(
                    data[self.key_input],
                    data[KEY.BATCH],
                    dim=0,
                    dim_size=data[KEY.NUM_ATOMS].shape[0],
                    reduce=self.reduce,
                )
                * self.constant
            )
            data[self.key_output] = data[self.key_output].squeeze(1)
        else:
            data[self.key_output] = paddle.sum(data[self.key_input]) * self.constant
        return data


@compile_mode("script")
class FCN_e3nn(paddle.nn.Module):
    """
    wrapper class of e3nn FullyConnectedNet
    """

    def __init__(
        self,
        irreps_in: Irreps,
        dim_out: int,
        hidden_neurons,
        activation,
        data_key_in: str,
        data_key_out: str = None,
        **e3nn_params
    ):
        super().__init__()
        self.key_input = data_key_in
        self.irreps_in = irreps_in
        if data_key_out is None:
            self.key_output = data_key_in
        else:
            self.key_output = data_key_out
        for _, irrep in irreps_in:
            assert irrep.is_scalar()
        inp_dim = irreps_in.dim
        self.fcn = FullyConnectedNet(
            [inp_dim] + hidden_neurons + [dim_out],
            activation,
            sample_stream="torch_seed0",
            **e3nn_params,
        )

    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:
        data[self.key_output] = self.fcn(data[self.key_input])
        return data


def get_linear(
    in_features, out_features, activation=None, dropout=0.0, **e3nn_linear_params
):
    """
    Build a linear layer with optional activation and dropout.
    """
    layers = [Linear(in_features, out_features, **e3nn_linear_params)]
    if activation:
        layers.append(build_activation(activation))
    if dropout > 0.0:
        # ManualDropout: paddle dropout op lacks double-grad
        layers.append(ManualDropout(dropout))
    return paddle.nn.Sequential(*layers)


class IrrepsDropoutLinear(paddle.nn.Module):
    """
    wrapper class of e3nn Linear to operate on AtomGraphData
    """

    def __init__(
        self,
        irreps_in: Irreps,
        irreps_out: Irreps,
        data_key_in: str,
        data_key_out: str = None,
        activation=None,
        dropout=0.0,
        **e3nn_linear_params
    ):
        super().__init__()
        self.key_input = data_key_in
        if data_key_out is None:
            self.key_output = data_key_in
        else:
            self.key_output = data_key_out
        self.linear = get_linear(
            irreps_in, irreps_out, activation, dropout, **e3nn_linear_params
        )

    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:
        data[self.key_output] = self.linear(data[self.key_input])
        return data

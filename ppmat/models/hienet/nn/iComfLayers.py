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

import math
import random
from collections import OrderedDict
from typing import Optional, Tuple, Union

from .. import _keys as KEY
from .. import util as util
import numpy as np
import paddle
from ..e3nn import o3
from ..e3nn.nn import BatchNorm, FullyConnectedNet
from ..e3nn.o3 import Irreps, Linear, TensorProduct
from ..e3nn.util.jit import compile_mode
from .._const import AtomGraphDataType
from .convolution import IrrepsConvolution
from .equivariant_gate import EquivariantGate, eComfEquivariantGate
from .linear import (AtomReduce, DynamicIrrepsLinear, FCN_e3nn,
                              IrrepsLinear, eComfIrrepsLinear)
from .second_order import ManualDropout, ManualLayerNorm
from .self_connection import (SelfConnectionIntro,
                                       SelfConnectionLinearIntro,
                                       SelfConnectionOutro)
from ppmat.models.common.message_passing.message_passing import MessagePassing
from ..paddle_compat import Adj, OptTensor, PairTensor
from ppmat.utils.scatter import scatter


@compile_mode("script")
class ComformerNodeConvLayer(MessagePassing):
    _alpha: OptTensor

    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        heads: int = 1,
        denominator: float = 1.0,
        concat: bool = True,
        beta: bool = False,
        dropout_mlp: float = 0.0,
        dropout_attn: float = 0.0,
        edge_dim: Optional[int] = None,
        bias: bool = True,
        root_weight: bool = True,
        **kwargs
    ):
        kwargs.setdefault("aggr", "add")
        super(ComformerNodeConvLayer, self).__init__(node_dim=0, **kwargs)
        self.denominator = paddle.nn.Parameter(
            paddle.FloatTensor([denominator]), requires_grad=False
        )
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.beta = beta and root_weight
        self.root_weight = root_weight
        self.concat = concat
        self.edge_dim = edge_dim
        self._alpha = None
        self.cnt = 0
        if isinstance(in_channels, int):
            in_channels = in_channels, in_channels
        self.linear_key = paddle.nn.Linear(in_channels[0], heads * out_channels)
        self.linear_query = paddle.nn.Linear(in_channels[1], heads * out_channels)
        self.linear_value = paddle.nn.Linear(in_channels[0], heads * out_channels)
        self.linear_edge = paddle.nn.Linear(edge_dim, heads * out_channels)
        self.linear_concate = paddle.nn.Linear(heads * out_channels, out_channels)
        self.msg_update = paddle.nn.Sequential(
            paddle.nn.Linear(out_channels * 3, out_channels),
            paddle.nn.SiLU(),
            paddle.nn.Linear(out_channels, out_channels),
        )
        self.softplus = paddle.nn.Softplus()
        self.silu = paddle.nn.SiLU()
        self.key_update = paddle.nn.Sequential(
            paddle.nn.Linear(out_channels * 3, out_channels),
            paddle.nn.SiLU(),
            paddle.nn.Linear(out_channels, out_channels),
        )
        # ManualLayerNorm/ManualDropout: paddle fused ops lack double-grad
        self.bn = ManualLayerNorm(out_channels)
        self.bn_att = ManualLayerNorm(out_channels)
        self.sigmoid = paddle.nn.Sigmoid()
        self.linear_reset = paddle.nn.Linear(in_channels[1], out_channels)
        self.linear_update = paddle.nn.Linear(in_channels[1], out_channels)
        self.dropout_mlp = ManualDropout(p=dropout_mlp)
        self.dropout_attn = ManualDropout(p=dropout_attn)

    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:
        x = data[KEY.NODE_FEATURE]
        edge_index = data[KEY.EDGE_IDX]
        edge_attr = data[KEY.EDGE_EMBEDDING]
        H, C = self.heads, self.out_channels
        if isinstance(x, paddle.Tensor):
            x: PairTensor = (x, x)
        query = self.linear_query(x[1]).reshape(-1, H, C)
        key = self.linear_key(x[0]).reshape(-1, H, C)
        value = self.linear_value(x[0]).reshape(-1, H, C)
        out = self.propagate(
            edge_index,
            query=query,
            key=key,
            value=value,
            edge_attr=edge_attr,
            size=None,
        )
        out = out.reshape(-1, self.heads * self.out_channels)
        out = self.linear_concate(out)
        reset_gate = self.sigmoid(self.linear_reset(x[1]))
        update_gate = self.sigmoid(self.linear_update(x[1]))
        x_reset = reset_gate * x[1]
        x = (1 - update_gate) * x[1] + update_gate * self.softplus(self.bn(out))
        x = (1 - update_gate) * x_reset + update_gate * x
        data[KEY.NODE_FEATURE] = x
        return data

    def message(
        self,
        query_i: paddle.Tensor,
        key_i: paddle.Tensor,
        key_j: paddle.Tensor,
        value_j: paddle.Tensor,
        value_i: paddle.Tensor,
        edge_attr: OptTensor,
        index: paddle.Tensor,
        ptr: OptTensor,
        size_i: Optional[int],
    ) -> paddle.Tensor:
        edge_attr = self.linear_edge(edge_attr).reshape(-1, self.heads, self.out_channels)
        edge_attr = self.dropout_mlp(edge_attr)
        key_j = self.key_update(paddle.cat((key_i, key_j, edge_attr), dim=-1))
        alpha = query_i * key_j / math.sqrt(self.out_channels)
        alpha = self.dropout_attn(alpha)
        out = self.msg_update(paddle.cat((value_i, value_j, edge_attr), dim=-1))
        out = out * self.sigmoid(
            self.bn_att(alpha.reshape(-1, self.out_channels)).reshape(
                -1, self.heads, self.out_channels
            )
        )
        return out


@compile_mode("script")
class ComformerConvEdgeLayer(paddle.nn.Module):
    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        heads: int = 1,
        concat: bool = True,
        beta: bool = False,
        dropout: float = 0.0,
        edge_dim: Optional[int] = None,
        bias: bool = True,
        root_weight: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.beta = beta and root_weight
        self.root_weight = root_weight
        self.concat = concat
        self.dropout = dropout
        self.edge_dim = edge_dim
        if isinstance(in_channels, int):
            in_channels = in_channels, in_channels
        self.embedding_dim = 32
        self.lin_key = paddle.nn.Linear(in_channels[0], heads * out_channels)
        self.lin_query = paddle.nn.Linear(in_channels[1], heads * out_channels)
        self.lin_value = paddle.nn.Linear(in_channels[0], heads * out_channels)
        self.lin_key_e1 = paddle.nn.Linear(in_channels[0], heads * out_channels)
        self.lin_value_e1 = paddle.nn.Linear(in_channels[0], heads * out_channels)
        self.lin_key_e2 = paddle.nn.Linear(in_channels[0], heads * out_channels)
        self.lin_value_e2 = paddle.nn.Linear(in_channels[0], heads * out_channels)
        self.lin_key_e3 = paddle.nn.Linear(in_channels[0], heads * out_channels)
        self.lin_value_e3 = paddle.nn.Linear(in_channels[0], heads * out_channels)
        self.lin_edge = paddle.nn.Linear(edge_dim, heads * out_channels, bias=False)
        self.lin_concate = paddle.nn.Linear(heads * out_channels, out_channels)
        self.lin_msg_update = paddle.nn.Sequential(
            paddle.nn.Linear(out_channels * 3, out_channels),
            paddle.nn.SiLU(),
            paddle.nn.Linear(out_channels, out_channels),
        )
        self.silu = paddle.nn.SiLU()
        self.softplus = paddle.nn.Softplus()
        self.key_update = paddle.nn.Sequential(
            paddle.nn.Linear(out_channels * 3, out_channels),
            paddle.nn.SiLU(),
            paddle.nn.Linear(out_channels, out_channels),
        )
        # latent site (use_edge_conv=False in the released config): kept on paddle ops
        self.bn_att = paddle.nn.LayerNorm(out_channels)
        self.bn = paddle.nn.LayerNorm(out_channels)
        self.sigmoid = paddle.nn.Sigmoid()

    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:
        edge = data[KEY.EDGE_EMBEDDING]
        edge_nei_len = data[KEY.LATTICE_EMBEDDING]
        edge_nei_angle = data[KEY.ANGLE_EMBEDDING]
        H, C = self.heads, self.out_channels
        if isinstance(edge, paddle.Tensor):
            edge: PairTensor = (edge, edge)
        query_x = self.lin_query(edge[1]).reshape(-1, H, C).unsqueeze(1).repeat(1, 3, 1, 1)
        key_x = self.lin_key(edge[0]).reshape(-1, H, C).unsqueeze(1).repeat(1, 3, 1, 1)
        value_x = self.lin_value(edge[0]).reshape(-1, H, C).unsqueeze(1).repeat(1, 3, 1, 1)
        key_y = paddle.cat(
            (
                self.lin_key_e1(edge_nei_len[:, 0, :]).reshape(-1, 1, H, C),
                self.lin_key_e2(edge_nei_len[:, 1, :]).reshape(-1, 1, H, C),
                self.lin_key_e3(edge_nei_len[:, 2, :]).reshape(-1, 1, H, C),
            ),
            dim=1,
        )
        value_y = paddle.cat(
            (
                self.lin_value_e1(edge_nei_len[:, 0, :]).reshape(-1, 1, H, C),
                self.lin_value_e2(edge_nei_len[:, 1, :]).reshape(-1, 1, H, C),
                self.lin_value_e3(edge_nei_len[:, 2, :]).reshape(-1, 1, H, C),
            ),
            dim=1,
        )
        edge_xy = self.lin_edge(edge_nei_angle).reshape(-1, 3, H, C)
        key = self.key_update(paddle.cat((key_x, key_y, edge_xy), dim=-1))
        alpha = query_x * key / math.sqrt(self.out_channels)
        out = self.lin_msg_update(paddle.cat((value_x, value_y, edge_xy), dim=-1))
        out = out * self.sigmoid(
            self.bn_att(alpha.reshape(-1, self.out_channels)).reshape(
                -1, 3, self.heads, self.out_channels
            )
        )
        out = out.reshape(-1, 3, self.heads * self.out_channels)
        out = self.lin_concate(out)
        out = out.sum(dim=1)
        data[KEY.EDGE_EMBEDDING] = self.softplus(edge[1] + self.bn(out))
        return data


@compile_mode("script")
class eComfEquivariantConvLayer(paddle.nn.Module):
    def __init__(
        self,
        node_features_in: Union[int, Tuple[int, int]],
        node_features_out: Union[int, Tuple[int, int]],
        edge_dim: Optional[int] = None,
        lmax: int = 2,
        parity_mode: str = "full",
        sh="1x0e + 1x1e + 1x2e",
        act_gate=None,
        act_scalar=None,
        dropout: float = 0.0,
        denominator: float = 1.0,
        weight_layer_input_to_hidden=[12],
        weight_layer_act="relu",
        use_bias_in_linear: bool = False,
    ):
        super().__init__()
        self.node_features_in = node_features_in
        self.node_features_out = node_features_out
        self.lmax = lmax
        self.parity_mode = parity_mode
        self.sh = sh
        node_features_in = Irreps(node_features_in)
        node_features_out = Irreps(node_features_out)
        tp_irreps_out = util.infer_irreps_out(
            node_features_in, self.sh, drop_l=self.lmax, parity_mode=self.parity_mode
        )
        self.gate = eComfEquivariantGate(node_features_out, act_scalar, act_gate)
        irreps_for_gate_in = self.gate.get_gate_irreps_in()
        self.skip_linear = eComfIrrepsLinear(node_features_in, irreps_for_gate_in)
        self.node_linear = eComfIrrepsLinear(
            node_features_in, node_features_in, biases=use_bias_in_linear
        )
        self.convolution = TensorProductConvLayer(
            in_irreps=node_features_in,
            sh_irreps=self.sh,
            out_irreps=tp_irreps_out,
            n_edge_features=edge_dim,
            residual=False,
            denominator=denominator,
            weight_layer_input_to_hidden=weight_layer_input_to_hidden,
            weight_layer_act=weight_layer_act,
        )
        # latent site (eComf path disabled in model_build): kept on paddle op
        self.dropout = paddle.nn.Dropout(p=dropout)
        self.l0_indices = [i for i, l in enumerate(tp_irreps_out.ls) if l == 0]
        self.node_linear_2 = eComfIrrepsLinear(
            tp_irreps_out, irreps_for_gate_in, biases=use_bias_in_linear
        )

    def forward(
        self, data: AtomGraphDataType, edge_nei_len: OptTensor = None
    ) -> AtomGraphDataType:
        edge_index = data[KEY.EDGE_IDX]
        edge_attr = data[KEY.EDGE_VEC]
        edge_feature = data[KEY.EDGE_EMBEDDING]
        node_feature = data[KEY.NODE_FEATURE]
        skip_connect = self.skip_linear(node_feature)
        node_feature = self.node_linear(node_feature)
        edge_irr = o3.spherical_harmonics(
            self.sh, edge_attr, normalize=True, normalization="component"
        )
        tp = self.convolution(node_feature, edge_index, edge_feature, edge_irr)
        tp[:, self.l0_indices] = self.dropout(tp[:, self.l0_indices])
        node_feature = self.node_linear_2(tp)
        node_feature = node_feature + skip_connect
        node_feature = self.gate(node_feature)
        data[KEY.NODE_FEATURE] = node_feature
        return data


class TensorProductConvLayer(paddle.nn.Module):
    def __init__(
        self,
        in_irreps,
        sh_irreps,
        out_irreps,
        n_edge_features,
        residual=True,
        denominator=1.0,
        weight_layer_input_to_hidden=[12],
        weight_layer_act="relu",
    ):
        super(TensorProductConvLayer, self).__init__()
        self.in_irreps = in_irreps
        self.out_irreps = out_irreps
        self.sh_irreps = sh_irreps
        self.residual = residual
        irreps_x = Irreps(in_irreps)
        irreps_filter = Irreps(sh_irreps)
        irreps_out = Irreps(out_irreps)
        instructions = []
        irreps_mid = []
        for i, (mul_x, ir_x) in enumerate(irreps_x):
            for j, (_, ir_filter) in enumerate(irreps_filter):
                for ir_out in ir_x * ir_filter:
                    if ir_out in irreps_out:
                        k = len(irreps_mid)
                        irreps_mid.append((mul_x, ir_out))
                        instructions.append((i, j, k, "uvu", True))
        irreps_mid = Irreps(irreps_mid)
        irreps_mid, p, _ = irreps_mid.sort()
        instructions = [
            (i_in1, i_in2, p[i_out], mode, train)
            for i_in1, i_in2, i_out, mode, train in instructions
        ]
        self.convolution = TensorProduct(
            irreps_x,
            irreps_filter,
            irreps_mid,
            instructions,
            shared_weights=False,
            internal_weights=False,
        )
        self.fc = FullyConnectedNet(
            weight_layer_input_to_hidden + [self.convolution.weight_numel],
            weight_layer_act,
        )
        self.denominator = denominator

    def forward(
        self, node_attr, edge_index, edge_attr, edge_sh, out_nodes=None, reduce="mean"
    ):
        edge_src, edge_dst = edge_index
        tp = self.convolution(node_attr[edge_dst], edge_sh, self.fc(edge_attr))
        out_nodes = out_nodes or node_attr.shape[0]
        out = scatter(tp, edge_src, dim=0, dim_size=out_nodes)
        out = out.div(self.denominator)
        if self.residual:
            padded = paddle.compat.nn.functional.pad(
                node_attr, (0, out.shape[-1] - node_attr.shape[-1])
            )
            out = out + padded
        return out

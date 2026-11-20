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
from typing import Optional, Tuple, Union

from .. import _keys as KEY
import paddle
from ppmat.models.common.e3nn.util.jit import compile_mode
from .._const import AtomGraphDataType
from .second_order import ManualDropout, ManualLayerNorm
from ppmat.models.common.message_passing.message_passing import MessagePassing
from ..paddle_compat import OptTensor, PairTensor


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

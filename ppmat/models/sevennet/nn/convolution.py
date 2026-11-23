from copy import deepcopy
from typing import List

import paddle

from ppmat.utils.scatter import scatter
from ppmat.models.sevennet import _keys as KEY
from ppmat.models.common.e3nn.nn import FullyConnectedNet
from ppmat.models.common.e3nn.o3 import Irreps, TensorProduct
from ppmat.models.common.e3nn.util.jit import compile_mode
from ppmat.models.sevennet._const import AtomGraphDataType

from .activation import ShiftedSoftPlus
from .util import broadcast


def message_gather(
    node_features: paddle.Tensor, edge_dst: paddle.Tensor, message: paddle.Tensor
) -> paddle.Tensor:
    index = broadcast(edge_dst, message, 0)
    out_shape = [len(node_features)] + list(message.shape[1:])
    out = paddle.zeros(
        out_shape, dtype=node_features.dtype
    )
    """Not Support auto convert *.scatter_reduce_, please judge whether it is Pytorch API and convert by yourself"""
    out = scatter(
        message,
        edge_dst,
        dim=0,
        dim_size=node_features.shape[0],
        reduce="sum",
    )
    return out


@compile_mode("script")
class IrrepsConvolution(paddle.nn.Layer):
    """
    convolution of (fig 2.b), comm. in LAMMPS
    """

    def __init__(
        self,
        irreps_x: Irreps,
        irreps_filter: Irreps,
        irreps_out: Irreps,
        weight_layer_input_to_hidden: List[int],
        weight_layer_act=ShiftedSoftPlus,
        denominator: float = 1.0,
        train_denominator: bool = False,
        data_key_x: str = KEY.NODE_FEATURE,
        data_key_filter: str = KEY.EDGE_ATTR,
        data_key_weight_input: str = KEY.EDGE_EMBEDDING,
        data_key_edge_idx: str = KEY.EDGE_IDX,
        lazy_layer_instantiate: bool = True,
        is_parallel: bool = False,
    ) -> None:
        super().__init__()
        denominator_value = float(denominator)
        self.denominator = self.create_parameter(
            shape=[1],
            dtype="float32",
            default_initializer=paddle.nn.initializer.Assign(
                paddle.to_tensor(
                    [denominator_value],
                    dtype="float32",
                )
            ),
        )
        self.denominator.stop_gradient = not train_denominator
        self.key_x = data_key_x
        self.key_filter = data_key_filter
        self.key_weight_input = data_key_weight_input
        self.key_edge_idx = data_key_edge_idx
        self.is_parallel = is_parallel
        instructions = []
        irreps_mid = []
        weight_numel = 0
        for i, (mul_x, ir_x) in enumerate(irreps_x):
            for j, (_, ir_filter) in enumerate(irreps_filter):
                for ir_out in ir_x * ir_filter:
                    if ir_out in irreps_out:
                        k = len(irreps_mid)
                        weight_numel += mul_x * 1
                        irreps_mid.append((mul_x, ir_out))
                        instructions.append((i, j, k, "uvu", True))
        irreps_mid = Irreps(irreps_mid)
        irreps_mid, p, _ = irreps_mid.sort()  # type: ignore
        instructions = [
            (i_in1, i_in2, p[i_out], mode, train)
            for i_in1, i_in2, i_out, mode, train in instructions
        ]
        self._instructions_before_sort = instructions
        instructions = sorted(instructions, key=lambda x: x[2])
        self.convolution_kwargs = dict(
            irreps_in1=irreps_x,
            irreps_in2=irreps_filter,
            irreps_out=irreps_mid,
            instructions=instructions,
            shared_weights=False,
            internal_weights=False,
        )
        self.weight_nn_kwargs = dict(
            hs=weight_layer_input_to_hidden + [weight_numel], act=weight_layer_act
        )
        self.convolution = None
        self.weight_nn = None
        self.layer_instantiated = False
        self.convolution_cls = TensorProduct
        self.weight_nn_cls = FullyConnectedNet
        if not lazy_layer_instantiate:
            self.instantiate()
        self._comm_size = irreps_x.dim

    def instantiate(self) -> None:
        if self.convolution is not None:
            raise ValueError("Convolution layer already exists")
        if self.weight_nn is not None:
            raise ValueError("Weight_nn layer already exists")
        self.convolution = self.convolution_cls(**self.convolution_kwargs)
        self.weight_nn = self.weight_nn_cls(**self.weight_nn_kwargs)
        self.layer_instantiated = True

    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:
        assert self.convolution is not None, "Convolution is not instantiated"
        assert self.weight_nn is not None, "Weight_nn is not instantiated"
        weight = self.weight_nn(data[self.key_weight_input])
        x = data[self.key_x]
        if self.is_parallel:
            x = paddle.concat([x, data[KEY.NODE_FEATURE_GHOST]])
        edge_src = data[self.key_edge_idx][1]
        edge_dst = data[self.key_edge_idx][0]
        message = self.convolution(x[edge_src], data[self.key_filter], weight)
        x = message_gather(x, edge_dst, message)
        x = x.div(self.denominator)
        if self.is_parallel:
            x = paddle.tensor_split(x, data[KEY.NLOCAL])[0]
        data[self.key_x] = x
        return data


@compile_mode("script")
class IrrepsScatterGatterFusedConvolution(paddle.nn.Layer):
    """
    Same as above but forward
    """

    def __init__(
        self,
        irreps_x: Irreps,
        irreps_filter: Irreps,
        irreps_out: Irreps,
        weight_layer_input_to_hidden: List[int],
        weight_layer_act=ShiftedSoftPlus,
        denominator: float = 1.0,
        train_denominator: bool = False,
        data_key_x: str = KEY.NODE_FEATURE,
        data_key_filter: str = KEY.EDGE_ATTR,
        data_key_weight_input: str = KEY.EDGE_EMBEDDING,
        data_key_edge_idx: str = KEY.EDGE_IDX,
        lazy_layer_instantiate: bool = True,
        is_parallel: bool = False,
    ) -> None:
        super().__init__()
        self.denominator = self.create_parameter(
            shape=[1],
            dtype='float32',
            default_initializer=paddle.nn.initializer.Assign(
                paddle.to_tensor(
                    [1.0 if isinstance(denominator, str) else float(denominator)],
                    dtype='float32'
                )
            )
        )
        self.key_x = data_key_x
        self.key_filter = data_key_filter
        self.key_weight_input = data_key_weight_input
        self.key_edge_idx = data_key_edge_idx
        self.is_parallel = is_parallel
        instructions = []
        irreps_mid = []
        weight_numel = 0
        for i, (mul_x, ir_x) in enumerate(irreps_x):
            for j, (_, ir_filter) in enumerate(irreps_filter):
                for ir_out in ir_x * ir_filter:
                    if ir_out in irreps_out:
                        k = len(irreps_mid)
                        weight_numel += mul_x * 1
                        irreps_mid.append((mul_x, ir_out))
                        instructions.append((i, j, k, "uvu", True))
        irreps_mid = Irreps(irreps_mid)
        irreps_mid, p, _ = irreps_mid.sort()  # type: ignore
        instructions = [
            (i_in1, i_in2, p[i_out], mode, train)
            for i_in1, i_in2, i_out, mode, train in instructions
        ]
        self._instructions_before_sort = instructions
        instructions = sorted(instructions, key=lambda x: x[2])
        self.convolution_kwargs = dict(
            irreps_in1=irreps_x,
            irreps_in2=irreps_filter,
            irreps_out=irreps_mid,
            instructions=instructions,
            shared_weights=False,
            internal_weights=False,
        )
        self.weight_nn_kwargs = dict(
            hs=weight_layer_input_to_hidden + [weight_numel], act=weight_layer_act
        )
        self.convolution = None
        self.weight_nn = None
        self.layer_instantiated = False
        self.convolution_cls = None
        self.weight_nn_cls = FullyConnectedNet
        if not lazy_layer_instantiate:
            self.instantiate()
        self._comm_size = irreps_x.dim
        self._out_dim: int = irreps_mid.dim

    @classmethod
    def from_irreps_convolution(cls, src: IrrepsConvolution):
        """
        I'm looking for better idea
        """
        irreps_x = src.convolution_kwargs["irreps_in1"]
        ret = cls(irreps_x, irreps_x, irreps_x, weight_layer_input_to_hidden=[1])
        ret.__dict__ = deepcopy(src.__dict__)
        ret._out_dim = src.convolution_kwargs["irreps_out"].dim
        return ret

    def instantiate(self) -> None:
        if self.convolution is not None:
            raise ValueError("Convolution layer already exists")
        if self.weight_nn is not None:
            raise ValueError("Weight_nn layer already exists")
        assert self.convolution_cls is not None
        self.convolution = self.convolution_cls(**self.convolution_kwargs)
        self.weight_nn = self.weight_nn_cls(**self.weight_nn_kwargs)
        self.layer_instantiated = True

    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:
        assert self.convolution is not None, "Convolution is not instantiated"
        assert self.weight_nn is not None, "Weight_nn is not instantiated"
        x = data[self.key_x]
        weight_input = data[self.key_weight_input]
        weight = self.weight_nn(weight_input)
        if self.is_parallel:
            x = paddle.concat([x, data[KEY.NODE_FEATURE_GHOST]])
        edge_src = data[self.key_edge_idx][1]
        edge_dst = data[self.key_edge_idx][0]
        edge_filter = data[self.key_filter]
        if edge_src.size == 0:
            x = x.new_zeros(x.shape[0], self._out_dim)
            x = x + (edge_filter.sum() + weight.sum()) * 0
        else:
            x = self.convolution(
                x,
                edge_filter,
                weight,
                edge_src.to(paddle.int32),
                edge_dst.to(paddle.int32),
            )
        x = x.div(self.denominator)
        if self.is_parallel:
            x = paddle.tensor_split(x, data[KEY.NLOCAL])[0]
        data[self.key_x] = x
        return data

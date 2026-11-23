from typing import Callable, List, Optional

import paddle

from ppmat.utils.scatter import scatter
from ppmat.models.sevennet import _keys as KEY
from ppmat.models.common.e3nn.nn import FullyConnectedNet
from ppmat.models.common.e3nn.o3 import Irreps, Linear
from ppmat.models.common.e3nn.util.jit import compile_mode
from ppmat.models.sevennet._const import AtomGraphDataType


@compile_mode("script")
class IrrepsLinear(paddle.nn.Layer):
    """
    wrapper class of e3nn Linear to operate on AtomGraphData
    """

    def __init__(
        self,
        irreps_in: Irreps,
        irreps_out: Irreps,
        data_key_in: str,
        data_key_out: Optional[str] = None,
        data_key_modal_attr: str = KEY.MODAL_ATTR,
        num_modalities: int = 0,
        lazy_layer_instantiate: bool = True,
        **linear_kwargs,
    ) -> None:
        super().__init__()
        self.key_input = data_key_in
        if data_key_out is None:
            self.key_output = data_key_in
        else:
            self.key_output = data_key_out
        self.key_modal_attr = data_key_modal_attr
        self._irreps_in_wo_modal = irreps_in
        self.irreps_in = irreps_in
        self.irreps_out = irreps_out
        self.linear_kwargs = linear_kwargs
        self.linear = None
        self.layer_instantiated = False
        self.num_modalities = num_modalities
        self._is_batch_data = True
        self.linear_cls = Linear
        if num_modalities > 1:
            self.set_num_modalities(num_modalities)
        if not lazy_layer_instantiate:
            self.instantiate()

    def instantiate(self) -> None:
        if self.linear is not None:
            raise ValueError("Linear layer already exists")
        self.linear = self.linear_cls(
            self.irreps_in, self.irreps_out, **self.linear_kwargs
        )
        self.layer_instantiated = True

    def set_num_modalities(self, num_modalities: int) -> None:
        if self.layer_instantiated:
            raise ValueError("Layer already instantiated, can not change modalities")
        irreps_in = self._irreps_in_wo_modal + Irreps(f"{num_modalities}x0e")
        self.num_modalities = num_modalities
        self.irreps_in = irreps_in

    def _patch_modal_to_data(self, data: AtomGraphDataType) -> AtomGraphDataType:
        if self._is_batch_data:
            batch = data[KEY.BATCH]
            batch_modality_onehot = data[self.key_modal_attr].reshape(
                -1, self.num_modalities
            )
            batch_modality_onehot = batch_modality_onehot.astype(
                data[self.key_input].dtype
            )
            data[self.key_input] = paddle.concat(
                [data[self.key_input], batch_modality_onehot[batch]], axis=1
            )
        else:
            modality_onehot = data[self.key_modal_attr].expand(
                len(data[self.key_input]), -1
            )
            modality_onehot = modality_onehot.astype(data[self.key_input].dtype)
            data[self.key_input] = paddle.concat(
                [data[self.key_input], modality_onehot], axis=1
            )
        return data

    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:
        assert self.linear is not None, "Layer is not instantiated"
        if self.num_modalities > 1:
            data = self._patch_modal_to_data(data)
        data[self.key_output] = self.linear(data[self.key_input])
        return data


@compile_mode("script")
class AtomReduce(paddle.nn.Layer):
    """
    atomic energy -> total energy
    constant is multiplied to data
    """

    def __init__(
        self,
        data_key_in: str,
        data_key_out: str,
        reduce: str = "sum",
        constant: float = 1.0,
    ) -> None:
        super().__init__()
        self.key_input = data_key_in
        self.key_output = data_key_out
        self.constant = constant
        self.reduce = reduce
        self._is_batch_data = True

    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:
        if self._is_batch_data:
            src = data[self.key_input].squeeze(1)
            size = int(paddle.max(data[KEY.BATCH]).item()) + 1
            output = paddle.zeros(shape=size, dtype=src.dtype)
            """Not Support auto convert *.scatter_reduce_, please judge whether it is Pytorch API and convert by yourself"""
            output = scatter(
                src,
                data[KEY.BATCH],
                dim=0,
                dim_size=size,
                reduce="sum",
            )
            data[self.key_output] = output * self.constant
        else:
            data[self.key_output] = paddle.sum(data[self.key_input]) * self.constant
        return data


@compile_mode("script")
class FCN_e3nn(paddle.nn.Layer):
    """
    wrapper class of e3nn FullyConnectedNet
    """

    def __init__(
        self,
        irreps_in: Irreps,
        dim_out: int,
        hidden_neurons: List[int],
        activation: Callable,
        data_key_in: str,
        data_key_out: Optional[str] = None,
        **e3nn_kwargs,
    ) -> None:
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
            [inp_dim] + hidden_neurons + [dim_out], activation, **e3nn_kwargs
        )

    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:
        data[self.key_output] = self.fcn(data[self.key_input])
        return data

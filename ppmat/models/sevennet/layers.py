"""SevenNet layers built on PaddleMaterials e3nn and scatter operators."""
from copy import deepcopy
from typing import Callable, Dict, List, Optional, Tuple, Union
import paddle
from ppmat.models.common.e3nn.nn import FullyConnectedNet, Gate
from ppmat.models.common.e3nn.o3 import FullyConnectedTensorProduct, Irreps, Linear, TensorProduct
from ppmat.models.common.e3nn.util.jit import compile_mode
from . import config as KEY
from ppmat.utils.scatter import scatter_sum

AtomGraphDataType = Dict[str, paddle.Tensor]
NUM_UNIV_ELEMENT = 119


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
        weight_layer_act=KEY.ShiftedSoftPlus,
        denominator: float = 1.0,
        train_denominator: bool = False,
        data_key_x: str = KEY.NODE_FEATURE,
        data_key_filter: str = KEY.EDGE_ATTR,
        data_key_weight_input: str = KEY.EDGE_EMBEDDING,
        data_key_edge_idx: str = KEY.EDGE_IDX,
        lazy_layer_instantiate: bool = True,
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
        edge_src = data[self.key_edge_idx][1]
        edge_dst = data[self.key_edge_idx][0]
        message = self.convolution(x[edge_src], data[self.key_filter], weight)
        x = scatter_sum(message, edge_dst, dim=0, dim_size=x.shape[0])
        x = x.div(self.denominator)
        data[self.key_x] = x
        return data


@compile_mode("script")
class EquivariantGate(paddle.nn.Layer):
    def __init__(
        self,
        irreps_x: Irreps,
        act_scalar_dict: Dict[str, Callable],
        act_gate_dict: Dict[str, Callable],
        data_key_x: str = KEY.NODE_FEATURE,
    ) -> None:
        super().__init__()
        self.key_x = data_key_x
        parity_map = {(1): "e", (-1): "o"}
        irreps_gated_elem = []
        irreps_scalars_elem = []
        for mul, irreps in irreps_x:
            if irreps.l > 0:
                irreps_gated_elem.append((mul, irreps))
            else:
                irreps_scalars_elem.append((mul, irreps))
        irreps_scalars = Irreps(irreps_scalars_elem)
        irreps_gated = Irreps(irreps_gated_elem)
        irreps_gates_parity = 1 if "0e" in irreps_scalars else -1
        irreps_gates = Irreps(
            [(mul, (0, irreps_gates_parity)) for mul, _ in irreps_gated]
        )
        act_scalars = [act_scalar_dict[parity_map[p]] for _, (_, p) in irreps_scalars]
        act_gates = [act_gate_dict[parity_map[p]] for _, (_, p) in irreps_gates]
        self.gate = Gate(
            irreps_scalars, act_scalars, irreps_gates, act_gates, irreps_gated
        )

    def get_gate_irreps_in(self):
        """
        user must call this function to get proper irreps in for forward
        """
        return self.gate.irreps_in

    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:
        data[self.key_x] = self.gate(data[self.key_x])
        return data


@compile_mode("script")
class IrrepsLinear(paddle.nn.Layer):
    """Apply the shared e3nn Linear layer to a named graph tensor."""

    def __init__(
        self,
        irreps_in: Irreps,
        irreps_out: Irreps,
        data_key_in: str,
        data_key_out: Optional[str] = None,
        **linear_kwargs,
    ) -> None:
        super().__init__()
        self.key_input = data_key_in
        self.key_output = data_key_out if data_key_out is not None else data_key_in
        self.irreps_in = irreps_in
        self.irreps_out = irreps_out
        self.linear_kwargs = linear_kwargs
        self.layer_instantiated = False

    def instantiate(self) -> None:
        self.linear = Linear(self.irreps_in, self.irreps_out, **self.linear_kwargs)
        self.layer_instantiated = True

    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:
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
            output = scatter_sum(src, data[KEY.BATCH], dim=0)
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


@compile_mode("script")
class OnehotEmbedding(paddle.nn.Layer):
    """
    x : tensor of shape (N, 1)
    x_after : tensor of shape (N, num_classes)
    It overwrite data_key_x
    and saves input to data_key_save and output to data_key_additional
    I know this is strange but it is for compatibility with previous version
    and to specie wise shift scale work
    ex) [0 1 1 0] -> [[1, 0] [0, 1] [0, 1] [1, 0]] (num_classes = 2)
    """

    def __init__(
        self,
        num_classes: int,
        data_key_x: str = KEY.NODE_FEATURE,
        data_key_out: Optional[str] = None,
        data_key_save: Optional[str] = None,
        data_key_additional: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.key_x = data_key_x
        if data_key_out is None:
            self.key_output = data_key_x
        else:
            self.key_output = data_key_out
        self.key_save = data_key_save
        self.key_additional_output = data_key_additional

    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:
        inp = data[self.key_x]
        embd = paddle.nn.functional.one_hot(inp, self.num_classes)
        embd = paddle.cast(embd, dtype="float32")
        data[self.key_output] = embd
        if self.key_additional_output is not None:
            data[self.key_additional_output] = embd
        if self.key_save is not None:
            data[self.key_save] = inp
        return data


def _as_univ(ss: List[float], type_map: Dict[int, int], default: float) -> List[float]:
    assert len(ss) <= NUM_UNIV_ELEMENT, "shift scale is too long"
    return [
        (ss[type_map[z]] if z in type_map else default) for z in range(NUM_UNIV_ELEMENT)
    ]


def _make_parameter(layer, value, trainable=False):
    """Create and register a Paddle parameter from a Tensor/value."""
    if not isinstance(value, paddle.Tensor):
        value = paddle.to_tensor(
            value,
            dtype=paddle.get_default_dtype(),
        )

    param = layer.create_parameter(
        shape=list(value.shape),
        dtype=value.dtype,
        default_initializer=paddle.nn.initializer.Assign(value),
    )

    # Paddle:
    # stop_gradient=False  <=> requires_grad=True
    param.stop_gradient = not bool(trainable)

    return param


@compile_mode("script")
class Rescale(paddle.nn.Layer):
    """
    Scaling and shifting energy (and automatically force and stress)
    """

    def __init__(
        self,
        shift: float,
        scale: float,
        data_key_in: str = KEY.SCALED_ATOMIC_ENERGY,
        data_key_out: str = KEY.ATOMIC_ENERGY,
        train_shift_scale: bool = False,
        **kwargs,
    ) -> None:
        assert isinstance(shift, float) and isinstance(scale, float)
        super().__init__()
        self.shift = _make_parameter(
            self,
            paddle.to_tensor([shift], dtype="float32"),
            train_shift_scale,
        )
        self.scale = _make_parameter(
            self,
            paddle.to_tensor([scale], dtype="float32"),
            train_shift_scale,
        )
        self.key_input = data_key_in
        self.key_output = data_key_out

    def get_shift(self) -> float:
        return self.shift.detach().cpu().tolist()[0]

    def get_scale(self) -> float:
        return self.scale.detach().cpu().tolist()[0]

    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:
        data[self.key_output] = data[self.key_input] * self.scale + self.shift
        return data


@compile_mode("script")
class SpeciesWiseRescale(paddle.nn.Layer):
    """
    Scaling and shifting energy (and automatically force and stress)
    Use as it is if given list, expand to list if one of them is float
    If two lists are given and length is not the same, raise error
    """

    def __init__(
        self,
        shift: Union[List[float], float],
        scale: Union[List[float], float],
        data_key_in: str = KEY.SCALED_ATOMIC_ENERGY,
        data_key_out: str = KEY.ATOMIC_ENERGY,
        data_key_indices: str = KEY.ATOM_TYPE,
        train_shift_scale: bool = False,
    ) -> None:
        super().__init__()
        assert isinstance(shift, float) or isinstance(shift, list)
        assert isinstance(scale, float) or isinstance(scale, list)
        if (
            isinstance(shift, list)
            and isinstance(scale, list)
            and len(shift) != len(scale)
        ):
            raise ValueError("List length should be same")
        if isinstance(shift, list):
            num_species = len(shift)
        elif isinstance(scale, list):
            num_species = len(scale)
        else:
            raise ValueError("Both shift and scale is not a list")
        shift = [shift] * num_species if isinstance(shift, float) else shift
        scale = [scale] * num_species if isinstance(scale, float) else scale
        self.shift = _make_parameter(
            self,
            paddle.to_tensor(shift, dtype="float32"),
            train_shift_scale,
        )
        self.scale = _make_parameter(
            self,
            paddle.to_tensor(scale, dtype="float32"),
            train_shift_scale,
        )
        self.key_input = data_key_in
        self.key_output = data_key_out
        self.key_indices = data_key_indices

    def get_shift(self, type_map: Optional[Dict[int, int]] = None) -> List[float]:
        """
        Return shift in list of float. If type_map is given, return type_map reversed
        shift, which index equals atomic_number. 0.0 is assigned for atomis not found
        """
        shift = self.shift.detach().cpu().tolist()
        if type_map:
            shift = _as_univ(shift, type_map, 0.0)
        return shift

    def get_scale(self, type_map: Optional[Dict[int, int]] = None) -> List[float]:
        """
        Return scale in list of float. If type_map is given, return type_map reversed
        scale, which index equals atomic_number. 1.0 is assigned for atomis not found
        """
        scale = self.scale.detach().cpu().tolist()
        if type_map:
            scale = _as_univ(scale, type_map, 1.0)
        return scale

    @staticmethod
    def from_mappers(
        shift: Union[float, List[float]],
        scale: Union[float, List[float]],
        type_map: Dict[int, int],
        **kwargs,
    ) -> "SpeciesWiseRescale":
        """
        Fit dimensions or mapping raw shift scale values to that is valid under
        the given type_map: (atomic_numbers -> type_indices)
        """
        shift_scale = []
        n_atom_types = len(type_map)
        for s in (shift, scale):
            if isinstance(s, list) and len(s) > n_atom_types:
                if len(s) != NUM_UNIV_ELEMENT:
                    raise ValueError("given shift or scale is strange")
                s = [s[z] for z in sorted(type_map, key=lambda x: type_map[x])]
            elif isinstance(s, float):
                s = [s] * n_atom_types
            elif isinstance(s, list) and len(s) == 1:
                s = s * n_atom_types
            shift_scale.append(s)
        assert all([(len(s) == n_atom_types) for s in shift_scale])
        shift, scale = shift_scale
        return SpeciesWiseRescale(shift, scale, **kwargs)

    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:
        indices = data[self.key_indices]
        data[self.key_output] = data[self.key_input] * self.scale[indices].view(
            -1, 1
        ) + self.shift[indices].view(-1, 1)
        return data


@compile_mode("script")
class SelfConnectionIntro(paddle.nn.Layer):
    """
    do TensorProduct of x and some data(here attribute of x)
    and save it (to concatenate updated x at SelfConnectionOutro)
    """

    def __init__(
        self,
        irreps_in: Irreps,
        irreps_operand: Irreps,
        irreps_out: Irreps,
        data_key_x: str = KEY.NODE_FEATURE,
        data_key_operand: str = KEY.NODE_ATTR,
        lazy_layer_instantiate: bool = True,
        **kwargs
    ) -> None:
        super().__init__()
        self.fc_tensor_product = FullyConnectedTensorProduct(
            irreps_in, irreps_operand, irreps_out
        )
        self.irreps_in1 = irreps_in
        self.irreps_in2 = irreps_operand
        self.irreps_out = irreps_out
        self.key_x = data_key_x
        self.key_operand = data_key_operand
        self.fc_tensor_product = None
        self.layer_instantiated = False
        self.fc_tensor_product_cls = FullyConnectedTensorProduct
        self.fc_tensor_product_kwargs = kwargs
        if not lazy_layer_instantiate:
            self.instantiate()

    def instantiate(self) -> None:
        if self.fc_tensor_product is not None:
            raise ValueError("fc_tensor_product layer already exists")
        self.fc_tensor_product = self.fc_tensor_product_cls(
            self.irreps_in1,
            self.irreps_in2,
            self.irreps_out,
            shared_weights=True,
            internal_weights=True,
            **self.fc_tensor_product_kwargs
        )
        self.layer_instantiated = True

    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:
        assert self.fc_tensor_product is not None, "Layer is not instantiated"
        x = data[self.key_x]
        operand = data[self.key_operand]
        data[KEY.SELF_CONNECTION_TEMP] = self.fc_tensor_product(x, operand)
        return data


@compile_mode("script")
class SelfConnectionLinearIntro(paddle.nn.Layer):
    """
    Linear style self connection update
    """

    def __init__(
        self,
        irreps_in: Irreps,
        irreps_out: Irreps,
        data_key_x: str = KEY.NODE_FEATURE,
        lazy_layer_instantiate: bool = True,
        **kwargs
    ) -> None:
        super().__init__()
        self.irreps_in = irreps_in
        self.irreps_out = irreps_out
        self.key_x = data_key_x
        self.linear = None
        self.layer_instantiated = False
        self.linear_cls = Linear
        kwargs.pop("irreps_operand")
        self.linear_kwargs = kwargs
        if not lazy_layer_instantiate:
            self.instantiate()

    def instantiate(self) -> None:
        if self.linear is not None:
            raise ValueError("Linear layer already exists")
        self.linear = self.linear_cls(
            self.irreps_in, self.irreps_out, **self.linear_kwargs
        )
        self.layer_instantiated = True

    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:
        assert self.linear is not None, "Layer is not instantiated"
        x = data[self.key_x]
        data[KEY.SELF_CONNECTION_TEMP] = self.linear(x)
        return data


@compile_mode("script")
class SelfConnectionOutro(paddle.nn.Layer):
    """
    do TensorProduct of x and some data(here attribute of x)
    and save it (to concatenate updated x at SelfConnectionOutro)
    """

    def __init__(self, data_key_x: str = KEY.NODE_FEATURE) -> None:
        super().__init__()
        self.key_x = data_key_x

    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:
        x = data[self.key_x]
        sc_temp = data[KEY.SELF_CONNECTION_TEMP]
        data[self.key_x] = x + sc_temp
        del data[KEY.SELF_CONNECTION_TEMP]
        return data


def NequIP_interaction_block(
    irreps_x: Irreps,
    irreps_filter: Irreps,
    irreps_out_tp: Irreps,
    irreps_out: Irreps,
    weight_nn_layers: List[int],
    conv_denominator: float,
    train_conv_denominator: bool,
    self_connection_pair: Tuple[Callable, Callable],
    act_scalar: Dict[str, Callable],
    act_gate: Dict[str, Callable],
    act_radial: Callable,
    bias_in_linear: bool,
    num_species: int,
    t: int,
    data_key_x: str = KEY.NODE_FEATURE,
    data_key_weight_input: str = KEY.EDGE_EMBEDDING,
    **conv_kwargs,
) -> Dict[str, paddle.nn.Layer]:
    block = {}
    irreps_node_attr = Irreps(f"{num_species}x0e")
    sc_intro, sc_outro = self_connection_pair
    gate_layer = EquivariantGate(irreps_out, act_scalar, act_gate)
    irreps_for_gate_in = gate_layer.get_gate_irreps_in()
    block[f"{t}_self_connection_intro"] = sc_intro(
        irreps_x, irreps_operand=irreps_node_attr, irreps_out=irreps_for_gate_in
    )
    block[f"{t}_self_interaction_1"] = IrrepsLinear(
        irreps_x, irreps_x, data_key_in=data_key_x, biases=bias_in_linear
    )
    block[f"{t}_convolution"] = IrrepsConvolution(
        irreps_x=irreps_x,
        irreps_filter=irreps_filter,
        irreps_out=irreps_out_tp,
        data_key_weight_input=data_key_weight_input,
        weight_layer_input_to_hidden=weight_nn_layers,
        weight_layer_act=act_radial,
        denominator=conv_denominator,
        train_denominator=train_conv_denominator,
        **conv_kwargs,
    )
    block[f"{t}_self_interaction_2"] = IrrepsLinear(
        irreps_out_tp, irreps_for_gate_in, data_key_in=data_key_x, biases=bias_in_linear
    )
    block[f"{t}_self_connection_outro"] = sc_outro()
    block[f"{t}_equivariant_gate"] = gate_layer
    return block

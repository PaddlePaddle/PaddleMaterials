import copy
from collections import OrderedDict
from typing import Any, Dict, List, Tuple, Type, Union

import paddle

from ppmat.models.common.e3nn.math import normalize2mom
from ppmat.models.chgnet.chgnet import CutoffPolynomial
from ppmat.models.common.e3nn.o3 import Irreps, SphericalHarmonics

from . import config as KEY
from .geometry import (
    BesselBasis, EdgeEmbedding, ForceStressOutputFromEdge, XPLORCutoff,
)
from .layers import (
    AtomReduce, FCN_e3nn, IrrepsLinear, NequIP_interaction_block, OnehotEmbedding,
    Rescale, SelfConnectionIntro, SelfConnectionLinearIntro,
    SelfConnectionOutro, SpeciesWiseRescale,
)


class AtomGraphSequential(paddle.nn.Sequential):
    def __init__(self, modules, type_map, data_key_grad):
        for layer in modules.values():
            if not getattr(layer, "layer_instantiated", True):
                layer.instantiate()
        # Preserve layer names for compatibility with converted checkpoints.
        super().__init__(*modules.items())
        self.key_grad = data_key_grad

        species_indices = [-1] * 120
        for atomic_number, index in type_map.items():
            atomic_number = int(atomic_number)
            if not 0 <= atomic_number < len(species_indices):
                raise ValueError(f"Unsupported atomic number in type map: {atomic_number}")
            species_indices[atomic_number] = int(index)
        self.register_buffer(
            "z_to_onehot_tensor", paddle.to_tensor(species_indices, dtype="int64")
        )

    def set_is_batch_data(self, flag):
        for layer in self.children():
            if hasattr(layer, "_is_batch_data"):
                layer._is_batch_data = bool(flag)

    def forward(self, graph):
        atomic_numbers = graph[KEY.ATOMIC_NUMBERS].astype("int64").reshape([-1])
        if paddle.any((atomic_numbers < 0) | (atomic_numbers >= 120)).item():
            raise ValueError("SevenNet input contains an invalid atomic number")
        species = paddle.gather(self.z_to_onehot_tensor, atomic_numbers)
        if paddle.any(species < 0).item():
            unknown = atomic_numbers[species < 0].numpy().tolist()
            raise ValueError(f"Unsupported atomic numbers: {unknown}")
        graph[KEY.NODE_FEATURE] = species
        graph[self.key_grad].stop_gradient = False
        return super().forward(graph)


def init_self_connection(config: Dict[str, Any]) -> List[Tuple[Type, Type]]:
    self_connection_type_list = config[KEY.SELF_CONNECTION_TYPE]
    num_conv = config[KEY.NUM_CONVOLUTION]
    if isinstance(self_connection_type_list, str):
        self_connection_type_list = [self_connection_type_list] * num_conv
    io_pair_list = []
    for sc_type in self_connection_type_list:
        if sc_type == "none":
            io_pair = None
        elif sc_type == "nequip":
            io_pair = SelfConnectionIntro, SelfConnectionOutro
        elif sc_type == "linear":
            io_pair = SelfConnectionLinearIntro, SelfConnectionOutro
        else:
            raise ValueError(f"Unknown self_connection_type found: {sc_type}")
        io_pair_list.append(io_pair)
    return io_pair_list


def init_edge_embedding(config: Dict[str, Any]) -> EdgeEmbedding:
    _cutoff_param = {"cutoff_length": config[KEY.CUTOFF]}
    rbf, env, sph = None, None, None
    rbf_dct = copy.deepcopy(config[KEY.RADIAL_BASIS])
    rbf_dct.update(_cutoff_param)
    rbf_name = rbf_dct.pop(KEY.RADIAL_BASIS_NAME)
    if rbf_name == "bessel":
        rbf = BesselBasis(**rbf_dct)
    envelop_dct = copy.deepcopy(config[KEY.CUTOFF_FUNCTION])
    envelop_dct.update(_cutoff_param)
    envelop_name = envelop_dct.pop(KEY.CUTOFF_FUNCTION_NAME)
    if envelop_name == "poly_cut":
        cutoff = envelop_dct.pop("cutoff_length")
        cutoff_coeff = envelop_dct.pop("poly_cut_p_value", 6)
        if envelop_dct:
            raise ValueError(f'Unknown polynomial cutoff options: {sorted(envelop_dct)}')
        env = CutoffPolynomial(cutoff=cutoff, cutoff_coeff=cutoff_coeff)
    elif envelop_name == "XPLOR":
        env = XPLORCutoff(**envelop_dct)
    lmax_edge = config[KEY.LMAX]
    if config[KEY.LMAX_EDGE] > 0:
        lmax_edge = config[KEY.LMAX_EDGE]
    parity = -1 if config[KEY.IS_PARITY] else 1
    _normalize_sph = config[KEY._NORMALIZE_SPH]
    sph = SphericalHarmonics(
        Irreps.spherical_harmonics(lmax_edge, parity),
        normalize=_normalize_sph,
        normalization="component",
        irreps_in=Irreps("1x1o" if parity == -1 else "1x1e"),
    )
    return EdgeEmbedding(basis_module=rbf, cutoff_module=env, spherical_module=sph)


def init_feature_reduce(config: Dict[str, Any], irreps_x: Irreps) -> OrderedDict:
    layers = OrderedDict()
    if config[KEY.READOUT_AS_FCN] is False:
        hidden_irreps = Irreps([(irreps_x.dim // 2, (0, 1))])
        layers.update(
            {
                "reduce_input_to_hidden": IrrepsLinear(
                    irreps_x,
                    hidden_irreps,
                    data_key_in=KEY.NODE_FEATURE,
                    biases=config[KEY.USE_BIAS_IN_LINEAR],
                ),
                "reduce_hidden_to_energy": IrrepsLinear(
                    hidden_irreps,
                    Irreps([(1, (0, 1))]),
                    data_key_in=KEY.NODE_FEATURE,
                    data_key_out=KEY.SCALED_ATOMIC_ENERGY,
                    biases=config[KEY.USE_BIAS_IN_LINEAR],
                ),
            }
        )
    else:
        act = KEY.ACTIVATION[config[KEY.READOUT_FCN_ACTIVATION]]
        hidden_neurons = config[KEY.READOUT_FCN_HIDDEN_NEURONS]
        layers.update(
            {
                "readout_FCN": FCN_e3nn(
                    dim_out=1,
                    hidden_neurons=hidden_neurons,
                    activation=act,
                    data_key_in=KEY.NODE_FEATURE,
                    data_key_out=KEY.SCALED_ATOMIC_ENERGY,
                    irreps_in=irreps_x,
                )
            }
        )
    return layers



def init_shift_scale(
    config: Dict[str, Any],
) -> Union[Rescale, SpeciesWiseRescale]:
    # for mm, ex, shift: modal_idx -> shifts
    shift_scale = []
    train_shift_scale = config[KEY.TRAIN_SHIFT_SCALE]
    type_map = config[KEY.TYPE_MAP]

    # in case of modal, shift or scale has more dims [][]
    # correct typing (I really want static python)
    for s in (config[KEY.SHIFT], config[KEY.SCALE]):
        if hasattr(s, 'tolist'):  # array-like object
            s = s.tolist()
        if isinstance(s, dict):
            s = {k: v.tolist() if hasattr(v, 'tolist') else v for k, v in s.items()}
        if isinstance(s, list) and len(s) == 1:
            s = s[0]
        shift_scale.append(s)
    shift, scale = shift_scale

    rescale_module = None
    if all([isinstance(s, float) for s in shift_scale]):
        rescale_module = Rescale(shift, scale, train_shift_scale=train_shift_scale)
    elif any([isinstance(s, list) for s in shift_scale]):
        rescale_module = SpeciesWiseRescale.from_mappers(  # type: ignore
            shift, scale, type_map=type_map, train_shift_scale=train_shift_scale
        )
    else:
        raise ValueError('shift, scale should be list of float or float')

    return rescale_module


def build_E3_equivariant_model(
    config: dict, parallel: bool = False
) -> AtomGraphSequential:
    """
    output shapes (w/o batch)

    PRED_TOTAL_ENERGY: (),
    ATOMIC_ENERGY: (natoms, 1),  # intended
    PRED_FORCE: (natoms, 3),
    PRED_STRESS: (6,),

    for data w/o cell volume, pred_stress has garbage values
    """
    if parallel or config.get(KEY.USE_MODALITY, False):
        raise NotImplementedError("SevenNet supports single-model, single-fidelity execution")
    layers = OrderedDict()
    cutoff = config[KEY.CUTOFF]
    num_species = config[KEY.NUM_SPECIES]
    feature_multiplicity = config[KEY.NODE_FEATURE_MULTIPLICITY]
    num_convolution_layer = config[KEY.NUM_CONVOLUTION]
    interaction_type = config[KEY.INTERACTION_TYPE]
    use_bias_in_linear = config[KEY.USE_BIAS_IN_LINEAR]
    lmax_node = config[KEY.LMAX]
    if config[KEY.LMAX_NODE] > 0:
        lmax_node = config[KEY.LMAX_NODE]
    act_radial = KEY.ACTIVATION[config[KEY.ACTIVATION_RADIAL]]
    self_connection_pair_list = init_self_connection(config)
    irreps_manual = None
    if config[KEY.IRREPS_MANUAL] is not False:
        irreps_manual = config[KEY.IRREPS_MANUAL]
        try:
            irreps_manual = [Irreps(irr) for irr in irreps_manual]
            assert len(irreps_manual) == num_convolution_layer + 1
        except Exception:
            raise RuntimeError("invalid irreps_manual input given")
    conv_denominator = config[KEY.CONV_DENOMINATOR]
    if not isinstance(conv_denominator, list):
        conv_denominator = [conv_denominator] * num_convolution_layer
    train_conv_denominator = config[KEY.TRAIN_DENOMINTAOR]
    edge_embedding = init_edge_embedding(config)
    irreps_filter = edge_embedding.spherical.irreps_out
    radial_basis_num = edge_embedding.basis_function.num_basis
    layers.update({"edge_embedding": edge_embedding})
    one_hot_irreps = Irreps(f"{num_species}x0e")
    irreps_x = (
        Irreps(f"{feature_multiplicity}x0e")
        if irreps_manual is None
        else irreps_manual[0]
    )
    layers.update(
        {
            "onehot_idx_to_onehot": OnehotEmbedding(
                num_classes=num_species,
                data_key_x=KEY.NODE_FEATURE,
                data_key_out=KEY.NODE_FEATURE,
                data_key_save=KEY.ATOM_TYPE,
                data_key_additional=KEY.NODE_ATTR,
            ),
            "onehot_to_feature_x": IrrepsLinear(
                irreps_in=one_hot_irreps,
                irreps_out=irreps_x,
                data_key_in=KEY.NODE_FEATURE,
                biases=use_bias_in_linear,
            ),
        }
    )
    weight_nn_hidden = config[KEY.CONVOLUTION_WEIGHT_NN_HIDDEN_NEURONS]
    weight_nn_layers = [radial_basis_num] + weight_nn_hidden
    param_interaction_block = {
        "irreps_filter": irreps_filter,
        "weight_nn_layers": weight_nn_layers,
        "train_conv_denominator": train_conv_denominator,
        "act_radial": act_radial,
        "bias_in_linear": use_bias_in_linear,
        "num_species": num_species,
    }
    interaction_builder = None
    if interaction_type in ["nequip"]:
        act_scalar = {}
        act_gate = {}
        for k, v in config[KEY.ACTIVATION_SCARLAR].items():
            act_scalar[k] = KEY.ACTIVATION_DICT[k][v]
        for k, v in config[KEY.ACTIVATION_GATE].items():
            act_gate[k] = KEY.ACTIVATION_DICT[k][v]
        param_interaction_block.update({"act_scalar": act_scalar, "act_gate": act_gate})
    if interaction_type == "nequip":
        interaction_builder = NequIP_interaction_block
    else:
        raise ValueError(f"Unknown interaction type: {interaction_type}")
    for t in range(num_convolution_layer):
        param_interaction_block.update(
            {
                "irreps_x": irreps_x,
                "t": t,
                "conv_denominator": conv_denominator[t],
                "self_connection_pair": self_connection_pair_list[t],
            }
        )
        if interaction_type == "nequip":
            parity_mode = "full"
            fix_multiplicity = False
            if t == num_convolution_layer - 1:
                lmax_node = 0
                parity_mode = "even"
            irreps_out = (
                KEY.infer_irreps_out(
                    irreps_x,
                    irreps_filter,
                    lmax_node,
                    parity_mode,
                    fix_multiplicity=feature_multiplicity,
                )
                if irreps_manual is None
                else irreps_manual[t + 1]
            )
            irreps_out_tp = KEY.infer_irreps_out(
                irreps_x, irreps_filter, irreps_out.lmax, parity_mode, fix_multiplicity
            )
        else:
            raise ValueError(f"Unknown interaction type: {interaction_type}")
        param_interaction_block.update(
            {"irreps_out_tp": irreps_out_tp, "irreps_out": irreps_out}
        )
        layers.update(interaction_builder(**param_interaction_block))
        irreps_x = irreps_out
    layers.update(init_feature_reduce(config, irreps_x))
    layers.update(
        {
            "rescale_atomic_energy": init_shift_scale(config),
            "reduce_total_enegy": AtomReduce(
                data_key_in=KEY.ATOMIC_ENERGY, data_key_out=KEY.PRED_TOTAL_ENERGY
            ),
        }
    )
    gradient_module = ForceStressOutputFromEdge()
    grad_key = gradient_module.get_grad_key()
    layers.update({"force_output": gradient_module})
    model = AtomGraphSequential(
        layers, type_map=config[KEY.TYPE_MAP], data_key_grad=grad_key
    )
    # Upstream e3nn uses a different random stream to estimate these
    # constants. Apply its values only to this model, including radial MLPs.
    factors = {
        "silu": 1.679176792398942,
        "tanh": 1.5937334472592695,
        "relu": 1.4163393446331367,
        "abs": 1.001110600838467,
    }
    for layer in model.sublayers():
        if isinstance(layer, normalize2mom):
            name = getattr(layer.f, "__name__", layer.f.__class__.__name__).lower()
            if name in factors:
                layer.cst = factors[name]
                layer._is_id = abs(layer.cst - 1.0) < 1e-4
    return model

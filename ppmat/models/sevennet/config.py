"""SevenNet configuration and irreducible representation selection."""

import math
from copy import deepcopy
from typing import Any, Dict, List, Union

import paddle
from ase.data import atomic_numbers, chemical_symbols

from ppmat.models.common.e3nn.o3 import FullTensorProduct, Irreps

ATOMIC_NUMBERS = "atomic_numbers"
CELL_VOLUME = "cell_volume"
EDGE_VEC = "edge_vec"
EDGE_LENGTH = "edge_length"
EDGE_IDX = "edge_index"
ATOM_TYPE = "atom_type"
NODE_FEATURE = "x"
NODE_ATTR = "node_attr"
EDGE_ATTR = "edge_attr"
EDGE_EMBEDDING = "edge_embedding"
SCALED_ATOMIC_ENERGY = "scaled_atomic_energy"
ATOMIC_ENERGY = "atomic_energy"
PRED_TOTAL_ENERGY = "inferred_total_energy"
PRED_FORCE = "inferred_force"
PRED_STRESS = "inferred_stress"
NUM_ATOMS = "num_atoms"
BATCH = "batch"
SELF_CONNECTION_TEMP = "self_cont_tmp"
USE_MODALITY = "use_modality"
CUTOFF = "cutoff"
CHEMICAL_SPECIES = "chemical_species"
CHEMICAL_SPECIES_BY_ATOMIC_NUMBER = "_chemical_species_by_atomic_number"
NUM_SPECIES = "_number_of_species"
TYPE_MAP = "_type_map"
IRREPS_MANUAL = "irreps_manual"
NODE_FEATURE_MULTIPLICITY = "channel"
RADIAL_BASIS = "radial_basis"
CUTOFF_FUNCTION = "cutoff_function"
LMAX = "lmax"
LMAX_EDGE = "lmax_edge"
LMAX_NODE = "lmax_node"
IS_PARITY = "is_parity"
CONVOLUTION_WEIGHT_NN_HIDDEN_NEURONS = "weight_nn_hidden_neurons"
NUM_CONVOLUTION = "num_convolution_layer"
ACTIVATION_SCARLAR = "act_scalar"
ACTIVATION_GATE = "act_gate"
ACTIVATION_RADIAL = "act_radial"
SELF_CONNECTION_TYPE = "self_connection_type"
RADIAL_BASIS_NAME = "radial_basis_name"
CUTOFF_FUNCTION_NAME = "cutoff_function_name"
USE_BIAS_IN_LINEAR = "use_bias_in_linear"
READOUT_AS_FCN = "readout_as_fcn"
READOUT_FCN_HIDDEN_NEURONS = "readout_fcn_hidden_neurons"
READOUT_FCN_ACTIVATION = "readout_fcn_activation"
CONV_DENOMINATOR = "conv_denominator"
SHIFT = "shift"
SCALE = "scale"
TRAIN_SHIFT_SCALE = "train_shift_scale"
TRAIN_DENOMINTAOR = "train_denominator"
INTERACTION_TYPE = "interaction_type"
USE_FLASH_TP = "use_flash_tp"
CUEQUIVARIANCE_CONFIG = "cuequivariance_config"
_NORMALIZE_SPH = "_normalize_sph"


@paddle.jit.to_static
def ShiftedSoftPlus(x: paddle.Tensor) -> paddle.Tensor:
    return paddle.nn.functional.softplus(x) - math.log(2.0)

ACTIVATION = {
    "relu": paddle.nn.functional.relu,
    "silu": paddle.nn.functional.silu,
    "tanh": paddle.tanh,
    "abs": paddle.abs,
    "ssp": ShiftedSoftPlus,
    "sigmoid": paddle.nn.functional.sigmoid,
    "elu": paddle.nn.functional.elu,
}

ACTIVATION_FOR_EVEN = {"ssp": ShiftedSoftPlus, "silu": paddle.nn.functional.silu}

ACTIVATION_FOR_ODD = {"tanh": paddle.tanh, "abs": paddle.abs}

ACTIVATION_DICT = {"e": ACTIVATION_FOR_EVEN, "o": ACTIVATION_FOR_ODD}

AtomGraphDataType = Dict[str, paddle.Tensor]

DEFAULT_E3_EQUIVARIANT_MODEL_CONFIG = {
    CUTOFF: 4.5,
    NODE_FEATURE_MULTIPLICITY: 32,
    IRREPS_MANUAL: False,
    LMAX: 1,
    LMAX_EDGE: -1,
    LMAX_NODE: -1,
    IS_PARITY: True,
    NUM_CONVOLUTION: 3,
    RADIAL_BASIS: {RADIAL_BASIS_NAME: "bessel"},
    CUTOFF_FUNCTION: {CUTOFF_FUNCTION_NAME: "poly_cut"},
    ACTIVATION_RADIAL: "silu",
    ACTIVATION_SCARLAR: {"e": "silu", "o": "tanh"},
    ACTIVATION_GATE: {"e": "silu", "o": "tanh"},
    CONVOLUTION_WEIGHT_NN_HIDDEN_NEURONS: [64, 64],
    CONV_DENOMINATOR: 20.0,
    SHIFT: 0.0,
    SCALE: 1.0,
    TRAIN_DENOMINTAOR: False,
    TRAIN_SHIFT_SCALE: False,
    USE_BIAS_IN_LINEAR: False,
    READOUT_AS_FCN: False,
    READOUT_FCN_HIDDEN_NEURONS: [30, 30],
    READOUT_FCN_ACTIVATION: "relu",
    SELF_CONNECTION_TYPE: "nequip",
    INTERACTION_TYPE: "nequip",
    _NORMALIZE_SPH: True,
    USE_FLASH_TP: False,
    CUEQUIVARIANCE_CONFIG: {},
}


def chemical_species_preprocess(
    input_chem: List[str], universal: bool = False
) -> Dict[str, Any]:
    """Create SevenNet's deterministic atomic-number-to-type mapping."""
    if universal:
        return {
            CHEMICAL_SPECIES: chemical_symbols,
            CHEMICAL_SPECIES_BY_ATOMIC_NUMBER: list(range(len(chemical_symbols))),
            NUM_SPECIES: len(chemical_symbols),
            TYPE_MAP: {z: z for z in range(len(chemical_symbols))},
        }
    species = sorted({element.strip() for element in input_chem if element.strip()})
    if not species:
        raise ValueError("chemical_species must contain at least one element")
    unknown = [element for element in species if element not in atomic_numbers]
    if unknown:
        raise ValueError(f"Unknown chemical species: {unknown}")
    return {
        CHEMICAL_SPECIES: species,
        CHEMICAL_SPECIES_BY_ATOMIC_NUMBER: [atomic_numbers[element] for element in species],
        NUM_SPECIES: len(species),
        TYPE_MAP: {atomic_numbers[element]: index for index, element in enumerate(species)},
    }


def infer_irreps_out(
    irreps_x: Irreps,
    irreps_operand: Irreps,
    drop_l: Union[bool, int] = False,
    parity_mode: str = "full",
    fix_multiplicity: Union[bool, int] = False,
) -> Irreps:
    """Infer output irreps used by SevenNet interaction blocks."""

    if parity_mode not in ["full", "even", "sph"]:
        raise ValueError(
            f"Unsupported parity_mode: {parity_mode}"
        )

    irreps_out = FullTensorProduct(
        irreps_x,
        irreps_operand,
    ).irreps_out.simplify()

    new_irreps_elem = []

    for mul, (l, p) in irreps_out:
        elem = (mul, (l, p))

        if drop_l is not False and l > drop_l:
            continue

        if parity_mode == "even" and p == -1:
            continue

        if parity_mode == "sph" and p != (-1) ** l:
            continue

        if fix_multiplicity:
            elem = (
                fix_multiplicity,
                (l, p),
            )

        new_irreps_elem.append(elem)

    return Irreps(new_irreps_elem)


def init_model_config(config):

    model_meta = {}

    # chemical species
    if CHEMICAL_SPECIES not in config:
        raise ValueError(
            "chemical_species missing"
        )

    input_chem = config[CHEMICAL_SPECIES]
    if isinstance(input_chem, str) and input_chem.lower() == "auto":
        raise ValueError("Set chemical_species to an element list or universal")
    elif isinstance(input_chem, str) and input_chem.lower() == "universal":
        model_meta.update(chemical_species_preprocess([], universal=True))
    else:
        if isinstance(input_chem, str):
            input_chem = [
                item for item in input_chem.replace("-", ",").replace(" ", ",").split(",")
                if item
            ]
        if not isinstance(input_chem, list) or not all(isinstance(item, str) for item in input_chem):
            raise ValueError("chemical_species must be 'universal', a list, or a comma-separated string")
        model_meta.update(chemical_species_preprocess(input_chem))


    model_meta.update(
        {
            key: config[key] if key in config else deepcopy(default)
            for key, default in DEFAULT_E3_EQUIVARIANT_MODEL_CONFIG.items()
        }
    )
    return model_meta

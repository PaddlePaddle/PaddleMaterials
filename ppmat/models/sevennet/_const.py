from typing import Dict

import paddle

from ppmat.models.sevennet import _keys as KEY
from ppmat.models.sevennet.nn.activation import ShiftedSoftPlus

NUM_UNIV_ELEMENT = 119

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
    KEY.CUTOFF: 4.5,
    KEY.NODE_FEATURE_MULTIPLICITY: 32,
    KEY.IRREPS_MANUAL: False,
    KEY.LMAX: 1,
    KEY.LMAX_EDGE: -1,
    KEY.LMAX_NODE: -1,
    KEY.IS_PARITY: True,
    KEY.NUM_CONVOLUTION: 3,
    KEY.RADIAL_BASIS: {KEY.RADIAL_BASIS_NAME: "bessel"},
    KEY.CUTOFF_FUNCTION: {KEY.CUTOFF_FUNCTION_NAME: "poly_cut"},
    KEY.ACTIVATION_RADIAL: "silu",
    KEY.ACTIVATION_SCARLAR: {"e": "silu", "o": "tanh"},
    KEY.ACTIVATION_GATE: {"e": "silu", "o": "tanh"},
    KEY.CONVOLUTION_WEIGHT_NN_HIDDEN_NEURONS: [64, 64],
    KEY.CONV_DENOMINATOR: 20.0,
    KEY.SHIFT: 0.0,
    KEY.SCALE: 1.0,
    KEY.TRAIN_DENOMINTAOR: False,
    KEY.TRAIN_SHIFT_SCALE: False,
    KEY.USE_BIAS_IN_LINEAR: False,
    KEY.USE_MODAL_NODE_EMBEDDING: False,
    KEY.USE_MODAL_SELF_INTER_INTRO: False,
    KEY.USE_MODAL_SELF_INTER_OUTRO: False,
    KEY.USE_MODAL_OUTPUT_BLOCK: False,
    KEY.READOUT_AS_FCN: False,
    KEY.READOUT_FCN_HIDDEN_NEURONS: [30, 30],
    KEY.READOUT_FCN_ACTIVATION: "relu",
    KEY.SELF_CONNECTION_TYPE: "nequip",
    KEY.INTERACTION_TYPE: "nequip",
    KEY._NORMALIZE_SPH: True,
    KEY.USE_FLASH_TP: False,
    KEY.CUEQUIVARIANCE_CONFIG: {},
}

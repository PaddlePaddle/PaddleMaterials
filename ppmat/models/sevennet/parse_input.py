from copy import deepcopy

import yaml

from ppmat.models.sevennet import _const as _const
from ppmat.models.sevennet import _keys as KEY
from . import util


def config_initialize(
    key,
    config,
    default,
):
    if key not in config:
        return default
    return config[key]


def init_model_config(config):

    model_meta = {}

    # chemical species
    if KEY.CHEMICAL_SPECIES not in config:
        raise ValueError(
            "chemical_species missing"
        )

    input_chem = config[KEY.CHEMICAL_SPECIES]
    if isinstance(input_chem, str) and input_chem.lower() == "auto":
        raise ValueError("Set chemical_species to an element list or universal")
    elif isinstance(input_chem, str) and input_chem.lower() == "universal":
        model_meta.update(util.chemical_species_preprocess([], universal=True))
    else:
        if isinstance(input_chem, str):
            input_chem = [
                item for item in input_chem.replace("-", ",").replace(" ", ",").split(",")
                if item
            ]
        if not isinstance(input_chem, list) or not all(isinstance(item, str) for item in input_chem):
            raise ValueError("chemical_species must be 'universal', a list, or a comma-separated string")
        model_meta.update(util.chemical_species_preprocess(input_chem))


    for key, default in _const.DEFAULT_E3_EQUIVARIANT_MODEL_CONFIG.items():

        model_meta[key] = config_initialize(
            key,
            config,
            deepcopy(default)
        )


    return model_meta



def read_config_yaml_global(filename):

    with open(filename,"r") as f:
        inputs=yaml.safe_load(f)


    model_config = init_model_config(
        inputs["model"]
    )


    # 保留train/data以后使用
    if "train" in inputs:
        model_config.update(
            inputs["train"]
        )

    if "data" in inputs:
        model_config.update(
            inputs["data"]
        )


    return model_config

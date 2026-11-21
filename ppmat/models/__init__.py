# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import inspect
from typing import Any
from typing import Callable
from typing import Dict
from typing import Optional

from omegaconf import OmegaConf

from ppmat.models.chgnet.chgnet import CHGNet
from ppmat.models.chgnet.chgnet_graph_converter import CHGNetGraphConverter
from ppmat.models.comformer.comformer import iComformer
from ppmat.models.comformer.comformer_graph_converter import ComformerGraphConverter
from ppmat.models.common.graph_converter import CrystalNN
from ppmat.models.common.graph_converter import FindPointsInSpheres
from ppmat.models.common.graph_converter import MolecularGraphConverter
from ppmat.models.common.graph_converter import RadiusGraphConverter
from ppmat.models.diffcsp.diffcsp import DiffCSP
from ppmat.models.diffnmr.diffnmr import DiffNMR
from ppmat.models.diffnmr.diffnmr import DiffPrior
from ppmat.models.diffnmr.diffnmr import MolecularGraphFormer
from ppmat.models.diffnmr.diffnmr import NMRNetCLIP
from ppmat.models.dimenetpp.dimenetpp import DimeNetPlusPlus
from ppmat.models.gegnn import GEGNNBinary
from ppmat.models.infgcn.infgcn import InfGCN
from ppmat.models.mattergen.mattergen import MatterGen
from ppmat.models.mattergen.mattergen import MatterGenWithCondition
from ppmat.models.mattersim.m3gnet import M3GNet
from ppmat.models.mattersim.m3gnet_graph_converter import M3GNetGraphConvertor
from ppmat.models.megnet.megnet import MEGNetPlus
from ppmat.models.sfin.sfin import SFIN
from ppmat.models.spherenet.spherenet import SphereNet
from ppmat.utils import download
from ppmat.utils import logger
from ppmat.utils import save_load
from ppmat.utils.model_package import get_model_config_path
from ppmat.utils.model_package import resolve_model_package_dir
from ppmat.vocab import build_vocab

__all__ = [
    "iComformer",
    "ComformerGraphConverter",
    "DiffCSP",
    "FindPointsInSpheres",
    "MEGNetPlus",
    "MatterGen",
    "MatterGenWithCondition",
    "DimeNetPlusPlus",
    "CrystalNN",
    "CHGNetGraphConverter",
    "CHGNet",
    "M3GNetGraphConvertor",
    "M3GNet",
    "MolecularGraphConverter",
    "RadiusGraphConverter",
    "MolecularGraphFormer",
    "NMRNetCLIP",
    "DiffPrior",
    "DiffNMR",
    "GEGNNBinary",
    "InfGCN",
    "SFIN",
    "SphereNet",
]

# Warning:
# Registered pretrained models must use a predictable archive layout. The archive
# name, top-level directory, and configuration stem must all match the
# `MODEL_REGISTRY` key:
#
# <model_name>.zip
# └── <model_name>/
#     ├── <model_name>.yaml
#     └── checkpoints/
#         ├── best.pdparams
#         └── latest.pdparams
MODEL_REGISTRY = {
    "comformer_mp2018_train_60k_e_form": "https://paddle-org.bj.bcebos.com/paddlematerial/checkpoints/property_prediction/comformer/comformer_mp2018_train_60k_e_form.zip",
    "comformer_mp2018_train_60k_band_gap": "https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/property_prediction/comformer/comformer_mp2018_train_60k_band_gap.zip",
    "comformer_mp2018_train_60k_G": "https://paddle-org.bj.bcebos.com/paddlematerial/checkpoints/property_prediction/comformer/comformer_mp2018_train_60k_G.zip",
    "comformer_mp2018_train_60k_K": "https://paddle-org.bj.bcebos.com/paddlematerial/checkpoints/property_prediction/comformer/comformer_mp2018_train_60k_K.zip",
    "comformer_mp2024_train_130k_e_form": "https://paddle-org.bj.bcebos.com/paddlematerial/checkpoints/property_prediction/comformer/comformer_mp2024_train_130k_e_form.zip",
    "comformer_jarvis_dft_2d_e_form": "https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/property_prediction/comformer/comformer_jarvis_dft_2d_e_form.zip",
    "comformer_jarvis_dft_3d_e_form": "https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/property_prediction/comformer/comformer_jarvis_dft_3d_e_form.zip",
    "comformer_jarvis_alex_pbe_2d_all_e_form": "https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/property_prediction/comformer/comformer_jarvis_alex_pbe_2d_all_e_form.zip",
    "megnet_mp2018_train_60k_e_form": "https://paddle-org.bj.bcebos.com/paddlematerial/checkpoints/property_prediction/megnet/megnet_mp2018_train_60k_e_form.zip",
    "megnet_mp2018_train_60k_band_gap": "https://paddle-org.bj.bcebos.com/paddlematerial/checkpoints/property_prediction/megnet/megnet_mp2018_train_60k_band_gap.zip",
    "megnet_mp2018_train_60k_G": "https://paddle-org.bj.bcebos.com/paddlematerial/checkpoints/property_prediction/megnet/megnet_mp2018_train_60k_G.zip",
    "megnet_mp2018_train_60k_K": "https://paddle-org.bj.bcebos.com/paddlematerial/checkpoints/property_prediction/megnet/megnet_mp2018_train_60k_K.zip",
    "megnet_mp2024_train_130k_e_form": "https://paddle-org.bj.bcebos.com/paddlematerial/checkpoints/property_prediction/megnet/megnet_mp2024_train_130k_e_form.zip",
    "megnet_jarvis_dft_2d_e_form": "https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/property_prediction/megnet/megnet_jarvis_dft_2d_e_form.zip",
    "megnet_jarvis_dft_3d_e_form": "https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/property_prediction/megnet/megnet_jarvis_dft_3d_e_form.zip",
    "megnet_jarvis_alex_pbe_2d_all_e_form": "https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/property_prediction/megnet/megnet_jarvis_alex_pbe_2d_all_e_form.zip",
    "diffcsp_mp20": "https://paddle-org.bj.bcebos.com/paddlematerial/checkpoints/structure_generation/diffcsp/diffcsp_mp20.zip",
    "mattergen_mp20": "https://paddle-org.bj.bcebos.com/paddlematerial/checkpoints/structure_generation/mattergen/mattergen_mp20.zip",
    "mattergen_mp20_chemical_system": "https://paddle-org.bj.bcebos.com/paddlematerial/checkpoints/structure_generation/mattergen/mattergen_mp20_chemical_system.zip",
    "mattergen_mp20_dft_band_gap": "https://paddle-org.bj.bcebos.com/paddlematerial/checkpoints/structure_generation/mattergen/mattergen_mp20_dft_band_gap.zip",
    "mattergen_mp20_dft_bulk_modulus": "https://paddle-org.bj.bcebos.com/paddlematerial/checkpoints/structure_generation/mattergen/mattergen_mp20_dft_bulk_modulus.zip",
    "mattergen_mp20_dft_mag_density": "https://paddle-org.bj.bcebos.com/paddlematerial/checkpoints/structure_generation/mattergen/mattergen_mp20_dft_mag_density.zip",
    "mattergen_alex_mp20": "https://paddle-org.bj.bcebos.com/paddlematerial/checkpoints/structure_generation/mattergen/mattergen_alex_mp20.zip",
    "mattergen_alex_mp20_dft_band_gap": "https://paddle-org.bj.bcebos.com/paddlematerial/checkpoints/structure_generation/mattergen/mattergen_alex_mp20_dft_band_gap.zip",
    "mattergen_alex_mp20_chemical_system": "https://paddle-org.bj.bcebos.com/paddlematerial/checkpoints/structure_generation/mattergen/mattergen_alex_mp20_chemical_system.zip",
    "mattergen_alex_mp20_dft_mag_density": "https://paddle-org.bj.bcebos.com/paddlematerial/checkpoints/structure_generation/mattergen/mattergen_alex_mp20_dft_mag_density.zip",
    "mattergen_alex_mp20_ml_bulk_modulus": "https://paddle-org.bj.bcebos.com/paddlematerial/checkpoints/structure_generation/mattergen/mattergen_alex_mp20_ml_bulk_modulus.zip",
    "mattergen_alex_mp20_space_group": "https://paddle-org.bj.bcebos.com/paddlematerial/checkpoints/structure_generation/mattergen/mattergen_alex_mp20_space_group.zip",
    "mattergen_alex_mp20_chemical_system_energy_above_hull": "https://paddle-org.bj.bcebos.com/paddlematerial/checkpoints/structure_generation/mattergen/mattergen_alex_mp20_chemical_system_energy_above_hull.zip",
    "mattergen_alex_mp20_dft_mag_density_hhi_score": "https://paddle-org.bj.bcebos.com/paddlematerial/checkpoints/structure_generation/mattergen/mattergen_alex_mp20_dft_mag_density_hhi_score.zip",
    "chgnet_mptrj": "https://paddle-org.bj.bcebos.com/paddlematerial/checkpoints/interatomic_potentials/chgnet/chgnet_mptrj.zip",
    "dimenetpp_mp2018_train_60k_e_form": "https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/property_prediction/dimenet++/dimenetpp_mp2018_train_60k_e_form.zip",
    "dimenetpp_mp2018_train_60k_band_gap": "https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/property_prediction/dimenet++/dimenetpp_mp2018_train_60k_band_gap.zip",
    "dimenetpp_mp2018_train_60k_G": "https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/property_prediction/dimenet++/dimenetpp_mp2018_train_60k_G.zip",
    "dimenetpp_mp2018_train_60k_K": "https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/property_prediction/dimenet++/dimenetpp_mp2018_train_60k_K.zip",
    "mattersim_1M": "https://paddle-org.bj.bcebos.com/paddlematerial/checkpoints/interatomic_potentials/mattersim/mattersim_1M.zip",
    "mattersim_5M": "https://paddle-org.bj.bcebos.com/paddlematerial/checkpoints/interatomic_potentials/mattersim/mattersim_5M.zip",
    "mattergen_ml2ddb": "https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/MatterGen/mattergen_ml2ddb.zip",
    "mattergen_ml2ddb_chemical_system": "https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/MatterGen/mattergen_ml2ddb_chemical_system.zip",
    "mattergen_ml2ddb_space_group": "https://paddle-org.bj.bcebos.com/paddlematerial/workflow/ml2ddb/mattergen_ml2ddb_space_group.zip",
    "sfin_haadf_enhance": "https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/spectrum_enhancement/sfin/sfin_haadf_enhance.zip",
    "sfin_haadf_detect": "https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/spectrum_enhancement/sfin/sfin_haadf_detect.zip",
    "sfin_bf_enhance": "https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/spectrum_enhancement/sfin/sfin_bf_enhance.zip",
    "sfin_bf_detect": "https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/spectrum_enhancement/sfin/sfin_bf_detect.zip",
    "spherenet_qm9_mu": "https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/property_prediction/spherenet/spherenet_qm9_mu.zip",
    "spherenet_qm9_alpha": "https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/property_prediction/spherenet/spherenet_qm9_alpha.zip",
    "spherenet_qm9_homo": "https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/property_prediction/spherenet/spherenet_qm9_homo.zip",
    "spherenet_qm9_lumo": "https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/property_prediction/spherenet/spherenet_qm9_lumo.zip",
    "spherenet_qm9_gap": "https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/property_prediction/spherenet/spherenet_qm9_gap.zip",
    "spherenet_qm9_r2": "https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/property_prediction/spherenet/spherenet_qm9_r2.zip",
    "spherenet_qm9_zpve": "https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/property_prediction/spherenet/spherenet_qm9_zpve.zip",
    "spherenet_qm9_U0": "https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/property_prediction/spherenet/spherenet_qm9_U0.zip",
    "spherenet_qm9_U": "https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/property_prediction/spherenet/spherenet_qm9_U.zip",
    "spherenet_qm9_H": "https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/property_prediction/spherenet/spherenet_qm9_H.zip",
    "spherenet_qm9_G": "https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/property_prediction/spherenet/spherenet_qm9_G.zip",
    "spherenet_qm9_Cv": "https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/property_prediction/spherenet/spherenet_qm9_Cv.zip",
    "spherenet_md17_aspirin": "https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/interatomic_potentials/spherenet/spherenet_md17_aspirin.zip",
    "spherenet_md17_benzene_old": "https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/interatomic_potentials/spherenet/spherenet_md17_benzene_old.zip",
    "spherenet_md17_ethanol": "https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/interatomic_potentials/spherenet/spherenet_md17_ethanol.zip",
    "spherenet_md17_malonaldehyde": "https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/interatomic_potentials/spherenet/spherenet_md17_malonaldehyde.zip",
    "spherenet_md17_naphthalene": "https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/interatomic_potentials/spherenet/spherenet_md17_naphthalene.zip",
    "spherenet_md17_salicylic": "https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/interatomic_potentials/spherenet/spherenet_md17_salicylic.zip",
    "spherenet_md17_toluene": "https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/interatomic_potentials/spherenet/spherenet_md17_toluene.zip",
    "spherenet_md17_uracil": "https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/interatomic_potentials/spherenet/spherenet_md17_uracil.zip",
    "infgcn_md17_benzene": "https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/electronic_structure/infgcn/infgcn_md17_benzene.zip",
    "infgcn_md17_ethane": "https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/electronic_structure/infgcn/infgcn_md17_ethane.zip",
    "infgcn_md17_ethanol": "https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/electronic_structure/infgcn/infgcn_md17_ethanol.zip",
    "infgcn_md17_malonaldehyde": "https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/electronic_structure/infgcn/infgcn_md17_malonaldehyde.zip",
    "infgcn_md17_phenol": "https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/electronic_structure/infgcn/infgcn_md17_phenol.zip",
    "infgcn_md17_resorcinol": "https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/electronic_structure/infgcn/infgcn_md17_resorcinol.zip",
    "infgcn_mp": "https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/electronic_structure/infgcn/infgcn_mp.zip",
    "infgcn_omol25_mc_5k_trimmed": "https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/electronic_structure/infgcn/infgcn_omol25_mc_5k_trimmed.zip",
    "infgcn_qm9": "https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/electronic_structure/infgcn/infgcn_qm9.zip",
    "diffnmr_msdnmr_nless15": "https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/spectrum_elucidation/diffnmr/diffnmr_msdnmr_nless15.zip",
}


def build_graph_converter(cfg: Dict, vocab=None):
    """Build graph converter.

    Args:
        cfg (Dict): Graph converter config.
    """
    if cfg is None:
        return None
    cfg = copy.deepcopy(cfg)
    class_name = cfg.pop("__class_name__")
    init_params = cfg.pop("__init_params__")
    if vocab is not None:
        init_params["vocab"] = vocab
    graph_converter = eval(class_name)(**init_params)
    logger.debug(str(graph_converter))

    return graph_converter


def build_model(
    cfg: Dict[str, Any],
    vocab=None,
    strict_unused: bool = False,  # True → raise if some runtime deps are not consumed
    override: bool = True,  # True → runtime_deps override same-named __init_params__
    **runtime_deps,
):
    """Build Model.

    Args:
        cfg (Dict): Model config.
            {
                "__class_name__": "pkg.module.MyModel",
                "__init_params__": { "encoder_cfg": {...}, "decoder_cfg": {...}, ... }
                    # Only serializable hyperparameters
            }
        strict_unused : bool, optional (default: False)
            If True, raise a TypeError when any key in `runtime_deps` is not consumed
            by the model constructor (i.e., the constructor does not accept that name).
        override : bool, optional (default: True)
            Conflict policy when a key exists in both `__init_params__` and
            `runtime_deps`. If True, the value from `runtime_deps` wins; otherwise the
            config value is kept and the runtime value is ignored.
        vocab: Optional vocabulary objects used by models with explicit vocabularies.
        runtime_deps: Runtime objects, such as dataset_infos=...

    Returns:
        nn.Layer: Model object.
    """
    if cfg is None:
        return None
    cfg = copy.deepcopy(cfg)
    class_name = cfg.pop("__class_name__")
    init_params = cfg.pop("__init_params__")

    cls = eval(class_name)

    sig = inspect.signature(cls.__init__)
    accepts_kwargs = any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values())

    params = dict(init_params)
    consumed = set()
    if vocab is not None:
        runtime_deps["vocab"] = vocab

    if accepts_kwargs:
        if override:
            params.update(runtime_deps)
    else:
        for k, v in runtime_deps.items():
            if k in sig.parameters:
                if override or (k not in params):
                    params[k] = v
                consumed.add(k)

    if strict_unused:
        unused = set(runtime_deps.keys()) - consumed
        if unused:
            raise TypeError(
                f"Unused runtime deps for {class_name}: {sorted(unused)} "
                f"(constructor params: {list(sig.parameters.keys())})"
            )

    model = cls(**params)
    logger.debug(str(model))

    return model


def build_model_from_name(
    model_name: str,
    weights_name: Optional[str] = None,
    model_config_modifier: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
):
    extracted_path = download.get_weights_path_from_url(MODEL_REGISTRY[model_name])
    path = resolve_model_package_dir(model_name, extracted_path)
    logger.info(f"Save model and configuration files in path: {path}")
    config_path = get_model_config_path(model_name, path)

    config = OmegaConf.load(config_path)
    config = OmegaConf.to_container(config, resolve=True)

    model_config = config.get("Model", None)
    assert model_config is not None, "Model config must be provided."
    if model_config_modifier is not None:
        model_config = model_config_modifier(model_config)
        config["Model"] = model_config
    vocab = build_vocab(config.get("Vocabulary"))
    model = build_model(model_config, vocab=vocab)

    save_load.load_pretrain(model, path, weights_name)

    return model, config

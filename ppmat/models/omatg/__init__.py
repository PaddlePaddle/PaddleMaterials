# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""OMatG: Open Materials Generation with Stochastic Interpolants."""

import os.path as osp

from ppmat.models.omatg.model import OMATGCSPNetFull
from ppmat.utils import download
from ppmat.utils import logger
from ppmat.utils import save_load

# Not in MODEL_REGISTRY: needs dataset x variant per-weight URLs (not one zip).
OMATG_WEIGHTS = {
    "perov_5_csp": {
        "encdec_ode_gamma": "https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/OMatG/omatg_perov_5_csp/EncDec-ODE-Gamma.pdparams",
        "encdec_sde_gamma": "https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/OMatG/omatg_perov_5_csp/EncDec-SDE-Gamma.pdparams",
        "linear_ode_gamma": "https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/OMatG/omatg_perov_5_csp/Linear-ODE-Gamma.pdparams",
        "linear_ode": "https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/OMatG/omatg_perov_5_csp/Linear-ODE.pdparams",
        "linear_sde_gamma": "https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/OMatG/omatg_perov_5_csp/Linear-SDE-Gamma.pdparams",
        "trig_ode_gamma": "https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/OMatG/omatg_perov_5_csp/Trig-ODE-Gamma.pdparams",
        "trig_ode": "https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/OMatG/omatg_perov_5_csp/Trig-ODE.pdparams",
        "trig_sde_gamma": "https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/OMatG/omatg_perov_5_csp/Trig-SDE-Gamma.pdparams",
        "vesbd_ode": "https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/OMatG/omatg_perov_5_csp/VESBD-ODE.pdparams",
        "vpsbd_ode": "https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/OMatG/omatg_perov_5_csp/VPSBD-ODE.pdparams",
        "vpsbd_sde": "https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/OMatG/omatg_perov_5_csp/VPSBD-SDE.pdparams",
    },
    "mpts_52_csp": {
        "encdec_ode_gamma": "https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/OMatG/omatg_mpts_52_csp/EncDec-ODE-Gamma.pdparams",
        "encdec_sde_gamma": "https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/OMatG/omatg_mpts_52_csp/EncDec-SDE-Gamma.pdparams",
        "linear_ode_gamma": "https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/OMatG/omatg_mpts_52_csp/Linear-ODE-Gamma.pdparams",
        "linear_ode": "https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/OMatG/omatg_mpts_52_csp/Linear-ODE.pdparams",
        "linear_sde_gamma": "https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/OMatG/omatg_mpts_52_csp/Linear-SDE-Gamma.pdparams",
        "trig_ode_gamma": "https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/OMatG/omatg_mpts_52_csp/Trig-ODE-Gamma.pdparams",
        "trig_ode": "https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/OMatG/omatg_mpts_52_csp/Trig-ODE.pdparams",
        "trig_sde_gamma": "https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/OMatG/omatg_mpts_52_csp/Trig-SDE-Gamma.pdparams",
    },
    "mp_20_dng": {
        "encdec_ode_gamma": "https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/OMatG/omatg_mp_20_dng/EncDec-ODE-Gamma.pdparams",
        "encdec_sde_gamma": "https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/OMatG/omatg_mp_20_dng/EncDec-SDE-Gamma.pdparams",
        "linear_ode_gamma": "https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/OMatG/omatg_mp_20_dng/Linear-ODE-Gamma.pdparams",
        "linear_ode": "https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/OMatG/omatg_mp_20_dng/Linear-ODE.pdparams",
        "linear_sde_gamma": "https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/OMatG/omatg_mp_20_dng/Linear-SDE-Gamma.pdparams",
        "trig_ode_gamma": "https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/OMatG/omatg_mp_20_dng/Trig-ODE-Gamma.pdparams",
        "trig_ode": "https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/OMatG/omatg_mp_20_dng/Trig-ODE.pdparams",
        "trig_sde_gamma": "https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/OMatG/omatg_mp_20_dng/Trig-SDE-Gamma.pdparams",
        "vesbd_ode": "https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/OMatG/omatg_mp_20_dng/VESBD-ODE.pdparams",
        "vpsbd_ode": "https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/OMatG/omatg_mp_20_dng/VPSBD-ODE.pdparams",
        "vpsbd_sde": "https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/OMatG/omatg_mp_20_dng/VPSBD-SDE.pdparams",
    },
    "mp_20_csp": {
        "encdec_ode_gamma": "https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/OMatG/omatg_mp_20_csp/EncDec-ODE-Gamma.pdparams",
        "encdec_sde_gamma": "https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/OMatG/omatg_mp_20_csp/EncDec-SDE-Gamma.pdparams",
        "linear_ode_gamma": "https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/OMatG/omatg_mp_20_csp/Linear-ODE-Gamma.pdparams",
        "linear_ode": "https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/OMatG/omatg_mp_20_csp/Linear-ODE.pdparams",
        "linear_sde_gamma": "https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/OMatG/omatg_mp_20_csp/Linear-SDE-Gamma.pdparams",
        "trig_ode_gamma": "https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/OMatG/omatg_mp_20_csp/Trig-ODE-Gamma.pdparams",
        "trig_ode": "https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/OMatG/omatg_mp_20_csp/Trig-ODE.pdparams",
        "trig_sde_gamma": "https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/OMatG/omatg_mp_20_csp/Trig-SDE-Gamma.pdparams",
        "vesbd_ode": "https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/OMatG/omatg_mp_20_csp/VESBD-ODE.pdparams",
        "vpsbd_ode": "https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/OMatG/omatg_mp_20_csp/VPSBD-ODE.pdparams",
        "vpsbd_sde": "https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/OMatG/omatg_mp_20_csp/VPSBD-SDE.pdparams",
    },
    "alex_mp_20_csp": {
        "encdec_ode_gamma": "https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/OMatG/omatg_alex_mp_20_csp/EncDec-ODE-Gamma.pdparams",
        "encdec_sde_gamma": "https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/OMatG/omatg_alex_mp_20_csp/EncDec-SDE-Gamma.pdparams",
        "linear_ode_gamma": "https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/OMatG/omatg_alex_mp_20_csp/Linear-ODE-Gamma.pdparams",
        "linear_ode": "https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/OMatG/omatg_alex_mp_20_csp/Linear-ODE.pdparams",
        "linear_sde_gamma": "https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/OMatG/omatg_alex_mp_20_csp/Linear-SDE-Gamma.pdparams",
        "trig_ode_gamma": "https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/OMatG/omatg_alex_mp_20_csp/Trig-ODE-Gamma.pdparams",
        "trig_ode": "https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/OMatG/omatg_alex_mp_20_csp/Trig-ODE.pdparams",
        "trig_sde_gamma": "https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/OMatG/omatg_alex_mp_20_csp/Trig-SDE-Gamma.pdparams",
        "vesbd_ode": "https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/OMatG/omatg_alex_mp_20_csp/VESBD-ODE.pdparams",
        "vpsbd_ode": "https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/OMatG/omatg_alex_mp_20_csp/VPSBD-ODE.pdparams",
        "vpsbd_sde": "https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/OMatG/omatg_alex_mp_20_csp/VPSBD-SDE.pdparams",
    },
}

__all__ = [
    "OMATG_WEIGHTS",
    "build_omatg_model",
    "get_omatg_model_url",
]


def build_omatg_model(
    dataset: str,
    variant: str,
    mode: str,
    si_scheduler_cfg: dict = None,
    sampler_cfg: dict = None,
    **model_kwargs,
):
    """Build an OMatG model, download the released weights and load them.

    Args:
        dataset: Dataset key of :data:`OMATG_WEIGHTS` (e.g. "mp_20_csp").
        variant: Interpolant variant key (e.g. "linear_ode").
        mode: "csp" or "dng". Selects ``pred_type`` and the masked-species
            token explicitly instead of guessing from the dataset name; it
            must match the mode encoded in the dataset key.
        si_scheduler_cfg / sampler_cfg: When ``si_scheduler_cfg`` is given,
            the SI sampling path is enabled and both configs are forwarded
            to :class:`OMATGCSPNetFull`.
        **model_kwargs: Forwarded to :class:`OMATGCSPNetFull`.

    Returns:
        The loaded :class:`OMATGCSPNetFull` model.
    """
    if mode not in ("csp", "dng"):
        raise ValueError(f"mode must be 'csp' or 'dng', got {mode!r}.")

    weight_url = get_omatg_model_url(dataset, variant)

    logger.info(f"Building OMatG model: {dataset} / {variant} / {mode}")
    logger.info(f"Weight URL: {weight_url}")

    cache_dir = osp.join(download.WEIGHTS_HOME, f"omatg_{dataset}")
    weight_path = download.get_path_from_url(
        weight_url, cache_dir, check_exist=True, decompress=False
    )
    logger.info(f"Weight saved to: {weight_path}")

    params = dict(model_kwargs)
    params.setdefault("pred_type", mode == "dng")
    if si_scheduler_cfg is not None:
        params["use_si"] = True
        params["si_scheduler_cfg"] = si_scheduler_cfg
        params["sampler_cfg"] = sampler_cfg or {}
    model = OMATGCSPNetFull(**params)
    if mode == "dng":
        model.enable_masked_species()

    save_load.load_pretrain(model, weight_path)

    logger.info(f"Successfully built and loaded OMatG model: {dataset}/{variant}")

    return model


def get_omatg_model_url(dataset: str, variant: str) -> str:
    """Return the weight URL for a dataset/variant.

    Resolved from :data:`OMATG_WEIGHTS`, which maps the released per-variant
    .pdparams files; a MODEL_REGISTRY lookup cannot address the dataset x variant
    matrix with a single model name.
    """
    if dataset not in OMATG_WEIGHTS:
        available_datasets = list(OMATG_WEIGHTS.keys())
        raise ValueError(
            f"Unknown dataset: {dataset}. Available datasets: {available_datasets}"
        )

    if variant not in OMATG_WEIGHTS[dataset]:
        available_variants = list(OMATG_WEIGHTS[dataset].keys())
        raise ValueError(
            f"Unknown variant: {variant} for dataset {dataset}. "
            f"Available variants: {available_variants}"
        )

    return OMATG_WEIGHTS[dataset][variant]

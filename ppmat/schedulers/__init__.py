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
from typing import Dict

import ppmat.schedulers.scheduling_wrapped_sde_ve as scheduling_wrapped_sde_ve  # noqa
from ppmat.schedulers.scheduling_asu_ve_sde import ASUVESDEScheduler
from ppmat.schedulers.scheduling_d3pm import D3PMScheduler
from ppmat.schedulers.scheduling_ddpm import DDPMScheduler
from ppmat.schedulers.scheduling_diffprior import NoiseScheduler
from ppmat.schedulers.scheduling_lattice_vp import LatticeVPSDEScheduler
from ppmat.schedulers.scheduling_sde_ve import ScoreSdeVeScheduler
from ppmat.schedulers.scheduling_sde_ve import ScoreSdeVeSchedulerWrapped

NumAtomsVarianceAdjustedWrappedVESDE = (
    scheduling_wrapped_sde_ve.NumAtomsVarianceAdjustedWrappedVESDE
)
__all__ = [
    "build_scheduler",
    "DDPMScheduler",
    "ScoreSdeVeScheduler",
    "ScoreSdeVeSchedulerWrapped",
    "LatticeVPSDEScheduler",
    "NumAtomsVarianceAdjustedWrappedVESDE",
    "D3PMScheduler",
    "NoiseScheduler",
    "ASUVESDEScheduler",
]


def build_scheduler(cfg: Dict):
    """Build scheduler.

    Args:
        cfg (Dict): Scheduler config.

    Returns:
        scheduler: Scheduler object.
    """
    if cfg is None:
        return None
    cfg = copy.deepcopy(cfg)

    if "__class_name__" not in cfg:
        assert isinstance(cfg, dict)
        scheduler_dict = {}
        for key, sub_cfg in cfg.items():
            scheduler_dict[key] = build_scheduler(sub_cfg)
        return scheduler_dict

    class_name = cfg.pop("__class_name__")
    init_params = cfg.pop("__init_params__")

    scheduler = eval(class_name)(**init_params)
    return scheduler

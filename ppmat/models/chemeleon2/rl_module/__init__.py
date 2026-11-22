# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from ppmat.models.chemeleon2.rl_module.components import CompositionDiversityReward
from ppmat.models.chemeleon2.rl_module.components import CreativityReward
from ppmat.models.chemeleon2.rl_module.components import CustomReward
from ppmat.models.chemeleon2.rl_module.components import EnergyReward
from ppmat.models.chemeleon2.rl_module.components import RewardComponent
from ppmat.models.chemeleon2.rl_module.components import StructureDiversityReward
from ppmat.models.chemeleon2.rl_module.reward import ReinforceReward
from ppmat.models.chemeleon2.rl_module.reward import build_reward

__all__ = [
    "Chemeleon2RLModule",
    "RewardComponent",
    "CreativityReward",
    "EnergyReward",
    "StructureDiversityReward",
    "CompositionDiversityReward",
    "CustomReward",
    "ReinforceReward",
    "build_reward",
]


def __getattr__(name):
    # Lazy import keeps the heavy GRPO module out of pure-reward consumers.
    if name == "Chemeleon2RLModule":
        from ppmat.models.chemeleon2.rl_module.rl_module import Chemeleon2RLModule

        return Chemeleon2RLModule
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

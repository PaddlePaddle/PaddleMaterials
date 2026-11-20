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

"""MatterChat: Crystal material generation based on Mistral-7B + Q-Former + CHGNet."""

from ppmat.models.matterchat.mistral import MistralConfig
from ppmat.models.matterchat.mistral import MistralForCausalLM
from ppmat.models.matterchat.mistral import MistralModel
from ppmat.models.matterchat.q_former import BertConfig
from ppmat.models.matterchat.q_former import Blip2Base
from ppmat.models.matterchat.q_former import Blip2MistralInstruct

__all__ = [
    "BertConfig",
    "Blip2Base",
    "Blip2MistralInstruct",
    "MistralConfig",
    "MistralForCausalLM",
    "MistralModel",
]

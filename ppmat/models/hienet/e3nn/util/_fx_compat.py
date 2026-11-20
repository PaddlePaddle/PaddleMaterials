# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

# Adapted from https://github.com/divelab/AIRS (OpenMat/HIENet)

"""Placeholder aliases for torch.fx.* constructors.

The e3nn codegen paths run as paddle eager code. These symbols only keep
annotation positions syntactically valid; every real instantiation raises
NotImplementedError instead of silently producing wrong results.
"""


class _FxStub:
    """Placeholder for torch.fx.* constructors; raises on instantiation."""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "torch.fx is unavailable in the paddle build; "
            "e3nn codegen paths run as paddle eager code"
        )


# Placeholder aliases for torch.fx.Graph / torch.fx.Proxy /
# torch.fx.proxy.GraphAppendingTracer / torch.fx.GraphModule
# (also valid in annotation positions).
FXGraph = _FxStub
FXProxy = _FxStub
FXAppendingTracer = _FxStub
FXGraphModule = _FxStub

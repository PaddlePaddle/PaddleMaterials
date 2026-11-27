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

import paddle
from paddle import nn

from ppmat.models.common.activation import swish
from ppmat.models.diffcsp.diffcsp import CSPNet


class _MiADSiLU(nn.Silu):
    """SiLU with a numerically stable backward pass on CPU."""

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        # Paddle 3.3 returns NaN gradients for SiLU at large negative inputs.
        if x.place.is_cpu_place() and paddle.is_grad_enabled():
            return swish(x)
        return super().forward(x)


class MiADCSPNet(CSPNet):
    """CSPNet denoiser for MiAD without property conditioning.

    MiAD requires the smooth atomic-number encoding, atom-type prediction and
    fully connected edges of CSPNet. Property conditioning is disabled so the
    parameter set matches the released checkpoint.
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        latent_dim: int = 256,
        num_layers: int = 4,
        act_fn: str = "silu",
        dis_emb: str = "sin",
        num_freqs: int = 10,
        edge_style: str = "fc",
        ln: bool = False,
        ip: bool = True,
        smooth: bool = False,
        pred_type: bool = False,
        num_classes: int = 100,
    ) -> None:
        if act_fn != "silu":
            raise ValueError("MiAD only supports act_fn='silu'")
        if not smooth or not pred_type:
            raise ValueError("MiAD requires smooth=True and pred_type=True")
        if edge_style != "fc":
            raise ValueError("MiAD only supports edge_style='fc'")
        if num_layers < 1:
            raise ValueError("MiAD requires at least one message-passing layer")

        super().__init__(
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            num_layers=num_layers,
            act_fn=act_fn,
            dis_emb=dis_emb,
            num_freqs=num_freqs,
            edge_style=edge_style,
            ln=ln,
            ip=ip,
            smooth=smooth,
            pred_type=pred_type,
            prop_dim=None,
            num_classes=num_classes,
            act_layer=_MiADSiLU,
        )

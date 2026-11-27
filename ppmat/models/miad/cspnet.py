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
from ppmat.models.diffcsp.diffcsp import CSPLayer
from ppmat.models.diffcsp.diffcsp import CSPNet
from ppmat.models.diffcsp.diffcsp import SinusoidsEmbedding

_SPACE_DIM = 3
_LATTICE_DIM = _SPACE_DIM**2


class _MiADSiLU(nn.Silu):
    """SiLU with a numerically stable backward pass on CPU."""

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        # Paddle 3.3 returns NaN gradients for SiLU at large negative inputs.
        if x.place.is_cpu_place() and paddle.is_grad_enabled():
            return swish(x)
        return super().forward(x)


class _MiADCSPLayer(CSPLayer):
    """Reuse CSP message passing without property-conditioning parameters."""

    def __init__(self, hidden_dim, act_fn, dis_emb, ln, ip):
        nn.Layer.__init__(self)
        self.dis_emb = dis_emb
        self.dis_dim = _SPACE_DIM if dis_emb is None else dis_emb.dim
        self.ip = ip
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + _LATTICE_DIM + self.dis_dim, hidden_dim),
            act_fn,
            nn.Linear(hidden_dim, hidden_dim),
            act_fn,
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            act_fn,
            nn.Linear(hidden_dim, hidden_dim),
            act_fn,
        )
        self.ln = ln
        if ln:
            self.layer_norm = nn.LayerNorm(hidden_dim)


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
        smooth: bool = True,
        pred_type: bool = True,
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

        nn.Layer.__init__(self)
        self.ip = ip
        self.smooth = smooth
        self.num_classes = num_classes
        self.node_embedding = nn.Linear(num_classes, hidden_dim)
        self.atom_latent_emb = nn.Linear(hidden_dim + latent_dim, hidden_dim)
        self.act_fn = _MiADSiLU()
        if dis_emb == "sin":
            self.dis_emb = SinusoidsEmbedding(n_frequencies=num_freqs)
        elif dis_emb == "none":
            self.dis_emb = None
        else:
            raise ValueError("MiAD requires dis_emb='sin' or dis_emb='none'")
        for i in range(num_layers):
            self.add_sublayer(
                name=f"csp_layer_{i}",
                sublayer=_MiADCSPLayer(hidden_dim, self.act_fn, self.dis_emb, ln, ip),
            )
        self.num_layers = num_layers
        self.coord_out = nn.Linear(hidden_dim, _SPACE_DIM, bias_attr=False)
        self.lattice_out = nn.Linear(hidden_dim, _LATTICE_DIM, bias_attr=False)
        self.pred_type = pred_type
        self.ln = ln
        self.edge_style = edge_style
        if ln:
            self.final_layer_norm = nn.LayerNorm(hidden_dim)
        self.type_out = nn.Linear(hidden_dim, num_classes)
        self.pred_scalar = False

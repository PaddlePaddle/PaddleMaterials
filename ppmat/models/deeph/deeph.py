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

from __future__ import annotations

import paddle
import paddle.nn.functional as F
from paddle import nn

from ppmat.losses import MaskMSELoss
from ppmat.utils.scatter import scatter_sum


class GraphLayerNorm(nn.Layer):
    def __init__(self, in_channels, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = self.create_parameter(
            [in_channels], default_initializer=nn.initializer.Constant(1.0)
        )
        self.bias = self.create_parameter(
            [in_channels], default_initializer=nn.initializer.Constant(0.0)
        )

    def forward(self, x, batch):
        batch = paddle.cast(batch, "int64")
        batch_size = int(paddle.max(batch).item()) + 1
        num_channels = x.shape[-1]

        ones = paddle.ones([x.shape[0], 1], dtype=x.dtype)
        count = scatter_sum(ones, batch, dim=0, dim_size=batch_size)
        norm = count * float(num_channels)

        summed = scatter_sum(x, batch, dim=0, dim_size=batch_size)
        mean = paddle.sum(summed, axis=-1, keepdim=True) / norm
        x_centered = x - paddle.gather(mean, batch, axis=0)

        var = scatter_sum(x_centered * x_centered, batch, dim=0, dim_size=batch_size)
        var = paddle.sum(var, axis=-1, keepdim=True) / norm
        out = x_centered / paddle.sqrt(paddle.gather(var, batch, axis=0) + self.eps)
        return out * self.weight + self.bias


def gaussian_smearing(distances, offset, widths, centered=False):
    if not centered:
        coeff = -0.5 / paddle.pow(widths, 2)
        diff = distances.unsqueeze(-1) - offset
    else:
        coeff = -0.5 / paddle.pow(offset, 2)
        diff = distances.unsqueeze(-1)
    return paddle.exp(coeff * paddle.pow(diff, 2))


class GaussianBasis(nn.Layer):
    def __init__(self, start=0.0, stop=5.0, n_gaussians=50):
        super().__init__()
        offset = paddle.linspace(start, stop, n_gaussians)
        widths = paddle.full(offset.shape, offset[1] - offset[0], dtype="float32")
        self.register_buffer("offsets", offset)
        self.register_buffer("width", widths)

    def forward(self, distances):
        return gaussian_smearing(distances, self.offsets, self.width, centered=False)


class MLPBlock(nn.Layer):
    def __init__(self, in_dim, hidden_dim, out_dim, final_activation=True):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.final_activation = final_activation

    def forward(self, x):
        x = F.silu(self.fc1(x))
        x = self.fc2(x)
        if self.final_activation:
            x = F.silu(x)
        return x


class CGConv(nn.Layer):
    def __init__(self, channels, dim, normalization="LayerNorm", if_exp=False):
        super().__init__()
        self.if_exp = if_exp
        self.lin_f = nn.Linear(channels * 2 + dim, channels)
        self.lin_s = nn.Linear(channels * 2 + dim, channels)
        self.ln = GraphLayerNorm(channels) if normalization == "LayerNorm" else None

    def forward(self, x, edge_index, edge_attr, batch, distance):
        row = paddle.cast(edge_index[0], "int64")
        col = paddle.cast(edge_index[1], "int64")
        x_j = paddle.gather(x, row, axis=0)
        x_i = paddle.gather(x, col, axis=0)
        z = paddle.concat([x_i, x_j, edge_attr], axis=-1)
        out = F.sigmoid(self.lin_f(z)) * F.softplus(self.lin_s(z))
        if self.if_exp:
            sigma = 3.0
            n = 2.0
            out = out * paddle.exp(-(distance**n) / (sigma**n) / 2.0).reshape(
                [-1, 1]
            )
        out = scatter_sum(out, col, dim=0, dim_size=x.shape[0])
        if self.ln is not None:
            out = self.ln(out, batch)
        return out + x


class MPLayer(nn.Layer):
    def __init__(
        self,
        atom_dim,
        edge_dim,
        out_edge_dim,
        if_exp,
        if_edge_update=True,
        normalization="LayerNorm",
    ):
        super().__init__()
        self.cgconv = CGConv(
            atom_dim, edge_dim, normalization=normalization, if_exp=if_exp
        )
        self.if_edge_update = if_edge_update
        if if_edge_update:
            final_activation = out_edge_dim != 169
            self.e_lin = MLPBlock(
                edge_dim + atom_dim * 2, 128, out_edge_dim, final_activation
            )

    def forward(self, atom_fea, edge_idx, edge_fea, batch, distance):
        atom_fea = self.cgconv(atom_fea, edge_idx, edge_fea, batch, distance)
        if self.if_edge_update:
            row = paddle.cast(edge_idx[0], "int64")
            col = paddle.cast(edge_idx[1], "int64")
            edge_fea = self.e_lin(
                paddle.concat(
                    [
                        paddle.gather(atom_fea, row, axis=0),
                        paddle.gather(atom_fea, col, axis=0),
                        edge_fea,
                    ],
                    axis=-1,
                )
            )
            return atom_fea, edge_fea
        return atom_fea


class LCMPLayer(nn.Layer):
    def __init__(self, atom_dim, edge_dim, out_dim, num_l, if_exp=False):
        super().__init__()
        self.if_exp = if_exp
        self.lin_f = nn.Linear(atom_dim * 2 + edge_dim, atom_dim)
        self.lin_s = nn.Linear(atom_dim * 2 + edge_dim, atom_dim)
        self.e_lin = MLPBlock(
            edge_dim + atom_dim * 2 - num_l**2, 128, out_dim, final_activation=False
        )

    def forward(
        self,
        atom_fea,
        edge_fea,
        sub_atom_idx,
        sub_edge_idx,
        sub_edge_ang,
        sub_index,
        distance,
    ):
        num_edge = edge_fea.shape[0]
        sub_atom_idx = paddle.cast(sub_atom_idx, "int64")
        sub_edge_idx = paddle.cast(sub_edge_idx, "int64")
        sub_index = paddle.cast(sub_index, "int64")
        gathered_atoms = paddle.gather(
            atom_fea, sub_atom_idx.reshape([-1]), axis=0
        ).reshape([sub_atom_idx.shape[0], 2, atom_fea.shape[1]])
        gathered_edge = paddle.gather(edge_fea, sub_edge_idx, axis=0)
        z = paddle.concat(
            [
                gathered_atoms[:, 0, :],
                gathered_atoms[:, 1, :],
                gathered_edge,
                sub_edge_ang,
            ],
            axis=-1,
        )
        out = F.sigmoid(self.lin_f(z)) * F.softplus(self.lin_s(z))
        if self.if_exp:
            sigma = 3.0
            n = 2.0
            edge_distance = paddle.gather(distance, sub_edge_idx, axis=0)
            out = out * paddle.exp(-(edge_distance**n) / (sigma**n) / 2.0).reshape(
                [-1, 1]
            )
        out = scatter_sum(out, sub_index, dim=0, dim_size=num_edge * 2)
        out = paddle.reshape(out, [num_edge, 2, -1])
        out = self.e_lin(paddle.concat([out[:, 0, :], out[:, 1, :], edge_fea], axis=-1))
        return out


class DeepHHamiltonian(nn.Layer):
    def __init__(
        self,
        num_species,
        in_atom_fea_len,
        in_edge_fea_len,
        num_orbital,
        num_l=5,
        gauss_stop=6.0,
        if_exp=True,
        normalization="LayerNorm",
        target_name="label",
        loss_eps=1e-8,
        **kwargs,
    ):
        super().__init__()
        self.num_species = num_species
        self.num_orbital = num_orbital
        self.num_l = num_l
        self.target_name = target_name
        self.loss_eps = loss_eps

        self.embed = nn.Embedding(num_species + 5, in_atom_fea_len)
        self.distance_expansion = GaussianBasis(0.0, gauss_stop, in_edge_fea_len)
        self.mp1 = MPLayer(
            in_atom_fea_len,
            in_edge_fea_len,
            in_edge_fea_len,
            if_exp=if_exp,
            normalization=normalization,
        )
        self.mp2 = MPLayer(
            in_atom_fea_len,
            in_edge_fea_len,
            in_edge_fea_len,
            if_exp=if_exp,
            normalization=normalization,
        )
        self.mp3 = MPLayer(
            in_atom_fea_len,
            in_edge_fea_len,
            in_edge_fea_len,
            if_exp=if_exp,
            normalization=normalization,
        )
        self.mp4 = MPLayer(
            in_atom_fea_len,
            in_edge_fea_len,
            in_edge_fea_len,
            if_exp=if_exp,
            normalization=normalization,
        )
        self.mp5 = MPLayer(
            in_atom_fea_len,
            in_edge_fea_len,
            in_edge_fea_len - num_l**2,
            if_exp=if_exp,
            normalization=normalization,
        )
        self.lcmp = LCMPLayer(
            in_atom_fea_len, in_edge_fea_len, num_orbital, num_l, if_exp=if_exp
        )
        self.loss_fn = MaskMSELoss()

    @staticmethod
    def _tensor(data, dtype):
        if paddle.is_tensor(data):
            return paddle.cast(data, dtype)
        return paddle.to_tensor(data, dtype=dtype)

    @classmethod
    def _prepare_batch(cls, batch):
        edge_index = cls._tensor(batch["edge_index"], "int64")
        if len(edge_index.shape) == 2 and edge_index.shape[0] != 2:
            edge_index = edge_index.transpose([1, 0])

        prepared = {
            "x": cls._tensor(batch["x"], "int64"),
            "edge_index": edge_index,
            "edge_attr": cls._tensor(batch["edge_attr"], "float32"),
            "batch": cls._tensor(batch["batch"], "int64"),
            "sub_atom_idx": cls._tensor(batch["sub_atom_idx"], "int64"),
            "sub_edge_idx": cls._tensor(batch["sub_edge_idx"], "int64"),
            "sub_edge_ang": cls._tensor(batch["sub_edge_ang"], "float32"),
            "sub_index": cls._tensor(batch["sub_index"], "int64"),
        }
        if "label" in batch:
            prepared["label"] = cls._tensor(batch["label"], "float32")
        if "mask" in batch:
            prepared["mask"] = cls._tensor(batch["mask"], "bool")
        return prepared

    def forward_backbone(
        self,
        atom_attr,
        edge_idx,
        edge_attr,
        batch,
        sub_atom_idx,
        sub_edge_idx,
        sub_edge_ang,
        sub_index,
    ):
        atom_attr = paddle.cast(atom_attr, "int64")
        edge_idx = paddle.cast(edge_idx, "int64")
        batch = paddle.cast(batch, "int64")
        distance = edge_attr[:, 0]

        atom_fea0 = self.embed(atom_attr)
        edge_fea0 = self.distance_expansion(distance)

        atom_fea, edge_fea = self.mp1(atom_fea0, edge_idx, edge_fea0, batch, distance)
        atom_fea, edge_fea = self.mp2(atom_fea, edge_idx, edge_fea, batch, distance)
        atom_fea0, edge_fea0 = atom_fea0 + atom_fea, edge_fea0 + edge_fea
        atom_fea, edge_fea = self.mp3(atom_fea0, edge_idx, edge_fea0, batch, distance)
        atom_fea, edge_fea = self.mp4(atom_fea, edge_idx, edge_fea, batch, distance)
        atom_fea0, edge_fea0 = atom_fea0 + atom_fea, edge_fea0 + edge_fea
        atom_fea, edge_fea = self.mp5(atom_fea0, edge_idx, edge_fea0, batch, distance)
        return self.lcmp(
            atom_fea,
            edge_fea,
            sub_atom_idx,
            sub_edge_idx,
            sub_edge_ang,
            sub_index,
            distance,
        )

    def forward(self, batch):
        batch = self._prepare_batch(batch)
        pred = self.forward_backbone(
            batch["x"],
            batch["edge_index"],
            batch["edge_attr"],
            batch["batch"],
            batch["sub_atom_idx"],
            batch["sub_edge_idx"],
            batch["sub_edge_ang"],
            batch["sub_index"],
        )
        label = batch[self.target_name]
        mask = batch["mask"]
        pred = paddle.reshape(pred, label.shape)

        loss = self.loss_fn(pred, label, mask)
        pred_dict = {
            self.target_name: pred,
        }
        loss_dict = {
            "loss": loss,
        }
        return {"loss_dict": loss_dict, "pred_dict": pred_dict}

    def predict(self, batch):
        batch = self._prepare_batch(batch)
        pred = self.forward_backbone(
            batch["x"],
            batch["edge_index"],
            batch["edge_attr"],
            batch["batch"],
            batch["sub_atom_idx"],
            batch["sub_edge_idx"],
            batch["sub_edge_ang"],
            batch["sub_index"],
        )
        if batch.get("label") is not None:
            pred = paddle.reshape(pred, batch["label"].shape)
        return {
            self.target_name: pred,
        }

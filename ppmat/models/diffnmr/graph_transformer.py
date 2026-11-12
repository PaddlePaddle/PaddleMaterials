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

import math

import paddle
import paddle.nn as nn
from paddle.nn import functional as F

from ppmat.models.diffnmr.utils import diffgraphformer_utils


class GraphTransformer(nn.Layer):
    """
    n_layers : int -- number of layers
    dims : dict -- contains dimensions for each feature type
    """

    def __init__(
        self,
        n_layers: int,
        input_dims: dict,
        hidden_mlp_dims: dict,
        hidden_dims: dict,
        output_dims: dict,
        act_fn_in=nn.ReLU(),
        act_fn_out=nn.ReLU(),
    ):
        super().__init__()
        self.n_layers = n_layers
        self.out_dim_X = output_dims["X"]
        self.out_dim_E = output_dims["E"]
        self.out_dim_y = output_dims["y"]

        self.mlp_in_X = nn.Sequential(
            nn.Linear(input_dims["X"], hidden_mlp_dims["X"]),
            act_fn_in,
            nn.Linear(hidden_mlp_dims["X"], hidden_dims["dx"]),
            act_fn_in,
        )

        self.mlp_in_E = nn.Sequential(
            nn.Linear(input_dims["E"], hidden_mlp_dims["E"]),
            act_fn_in,
            nn.Linear(hidden_mlp_dims["E"], hidden_dims["de"]),
            act_fn_in,
        )

        self.mlp_in_y = nn.Sequential(
            nn.Linear(input_dims["y"], hidden_mlp_dims["y"]),
            act_fn_in,
            nn.Linear(hidden_mlp_dims["y"], hidden_dims["dy"]),
            act_fn_in,
        )

        self.tf_layers = nn.LayerList(
            [
                XEyTransformerLayer(
                    dx=hidden_dims["dx"],
                    de=hidden_dims["de"],
                    dy=hidden_dims["dy"],
                    n_head=hidden_dims["n_head"],
                    dim_ffX=hidden_dims["dim_ffX"],
                    dim_ffE=hidden_dims["dim_ffE"],
                )
                for _ in range(n_layers)
            ]
        )

        self.mlp_out_X = (
            nn.Sequential(
                nn.Linear(hidden_dims["dx"], hidden_mlp_dims["X"]),
                act_fn_out,
                nn.Linear(hidden_mlp_dims["X"], output_dims["X"]),
            )
            if output_dims["X"] != 0
            else self.mlp_with_empty
        )

        self.mlp_out_E = (
            nn.Sequential(
                nn.Linear(hidden_dims["de"], hidden_mlp_dims["E"]),
                act_fn_out,
                nn.Linear(hidden_mlp_dims["E"], output_dims["E"]),
            )
            if output_dims["E"] != 0
            else self.mlp_with_empty
        )

        self.mlp_out_y = (
            nn.Sequential(
                nn.Linear(hidden_dims["dy"], hidden_mlp_dims["y"]),
                act_fn_out,
                nn.Linear(hidden_mlp_dims["y"], output_dims["y"]),
            )
            if output_dims["y"] != 0
            else self.mlp_with_empty
        )

    def mlp_with_empty(self, x):
        new_shape = x.shape[:-1] + [0]
        res = diffgraphformer_utils.return_empty(x, shape=new_shape)
        return res

    def forward(self, X, E, y, node_mask):
        bs, n = X.shape[0], X.shape[1]
        diag_mask = paddle.eye(n)
        diag_mask = ~diag_mask.astype(E.dtype).astype("bool")
        diag_mask = diag_mask.unsqueeze(0).unsqueeze(-1).expand([bs, -1, -1, -1])

        X_to_out = X[..., : self.out_dim_X]
        E_to_out = E[..., : self.out_dim_E]
        y_to_out = y[..., : self.out_dim_y]

        # Initial processing for X, E, y
        new_E = self.mlp_in_E(E)
        new_E = (new_E + new_E.transpose([0, 2, 1, 3])) / 2
        X = self.mlp_in_X(X)
        y = self.mlp_in_y(y)

        # Mask inline instead of PlaceHolder(...).mask(node_mask): building a
        # PlaceHolder from tensors and its symmetry check (a device-to-host read)
        # break the SOT graph. new_E is symmetric by construction.
        x_mask = paddle.unsqueeze(node_mask, axis=-1).astype(X.dtype)
        e_mask1 = paddle.unsqueeze(x_mask, axis=2)
        e_mask2 = paddle.unsqueeze(x_mask, axis=1)
        X = X * x_mask
        E = new_E * e_mask1 * e_mask2

        # Transformer layers
        for layer in self.tf_layers:
            X, E, y = layer(X, E, y, node_mask)

        # Output layers
        X = self.mlp_out_X(X)
        E = self.mlp_out_E(E)
        y = self.mlp_out_y(y)

        X = X + X_to_out
        E = (E + E_to_out) * diag_mask.astype(E.dtype)
        y = y + y_to_out

        # Symmetrize E
        E = 0.5 * (E + paddle.transpose(E, perm=[0, 2, 1, 3]))

        # Same inline masking as above; the PlaceHolder is built last so that the only
        # graph break comes after the network.
        X = X * x_mask
        E = E * e_mask1 * e_mask2
        return diffgraphformer_utils.PlaceHolder(X=X, E=E, y=y)


class MolecularEncoder(nn.Layer):
    """
    n_layers : int -- number of layers
    dims : dict -- contains dimensions for each feature type
    """

    def __init__(
        self,
        n_layers: int,
        input_dims: dict,
        hidden_mlp_dims: dict,
        hidden_dims: dict,
        output_dims: dict,
        act_fn_in=nn.ReLU(),
        act_fn_out=nn.ReLU(),
    ):
        super().__init__()
        self.n_layers = n_layers
        self.out_dim_X = output_dims["X"]
        self.out_dim_E = output_dims["E"]
        self.out_dim_y = output_dims["y"]

        self.mlp_in_X = nn.Sequential(
            nn.Linear(input_dims["X"], hidden_mlp_dims["X"]),
            act_fn_in,
            nn.Linear(hidden_mlp_dims["X"], hidden_dims["dx"]),
            act_fn_in,
        )

        self.mlp_in_E = nn.Sequential(
            nn.Linear(input_dims["E"], hidden_mlp_dims["E"]),
            act_fn_in,
            nn.Linear(hidden_mlp_dims["E"], hidden_dims["de"]),
            act_fn_in,
        )

        self.mlp_in_y = nn.Sequential(
            nn.Linear(input_dims["y"], hidden_mlp_dims["y"]),
            act_fn_in,
            nn.Linear(hidden_mlp_dims["y"], hidden_dims["dy"]),
            act_fn_in,
        )

        self.tf_layers = nn.LayerList(
            [
                XEyTransformerLayer(
                    dx=hidden_dims["dx"],
                    de=hidden_dims["de"],
                    dy=hidden_dims["dy"],
                    n_head=hidden_dims["n_head"],
                    dim_ffX=hidden_dims["dim_ffX"],
                    dim_ffE=hidden_dims["dim_ffE"],
                )
                for _ in range(n_layers)
            ]
        )

        self.mlp_out_X = nn.Sequential(
            nn.Linear(hidden_dims["dx"], hidden_mlp_dims["X"]),
            act_fn_out,
            nn.Linear(hidden_mlp_dims["X"], 512),
        )

    def forward(self, X, E, y, node_mask):
        """
        X: (bs, n, input_dims['X'])
        E: (bs, n, n, input_dims['E'])
        y: (bs, input_dims['y'])
        node_mask: (bs, n)
        """
        bs, n = X.shape[0], X.shape[1]

        diag_mask = paddle.eye(n, dtype="int64")  # (n, n)
        diag_mask = paddle.logical_not(diag_mask)
        diag_mask = (
            diag_mask.unsqueeze(0).unsqueeze(-1).expand([bs, -1, -1, -1])
        )  # (bs,n,n,1)

        # MLP in
        new_E = self.mlp_in_E(E)
        # symmetrize
        E_t = paddle.transpose(new_E, perm=[0, 2, 1, 3])
        new_E = (new_E + E_t) / 2.0

        X = self.mlp_in_X(X)
        Y = self.mlp_in_y(y)

        after_in = diffgraphformer_utils.PlaceHolder(X, E=new_E, y=Y).mask(node_mask)
        X, E, Y = after_in.X, after_in.E, after_in.y

        for layer in self.tf_layers:
            X, E, Y = layer(X, E, Y, node_mask)

        # Output
        X = self.mlp_out_X(X)  # (bs, n, 512)
        X_mean = paddle.mean(X, axis=1)  # (bs, 512)

        return X_mean


class XEyTransformerLayer(nn.Layer):
    """Transformer that updates node, edge and global features
    d_x: node features
    d_e: edge features
    dz : global features
    n_head: the number of heads in the multi_head_attention
    dim_feedforward: the dimension of the feedforward network model after self-attention
    dropout: dropout probablility. 0 to disable
    layer_norm_eps: eps value in layer normalizations.
    """

    def __init__(
        self,
        dx: int,
        de: int,
        dy: int,
        n_head: int,
        dim_ffX: int = 2048,
        dim_ffE: int = 128,
        dim_ffy: int = 2048,
        dropout: float = 0,  # TODO: 0.1,
        layer_norm_eps: float = 1e-5,
    ) -> None:
        super().__init__()

        self.self_attn = NodeEdgeBlock(dx, de, dy, n_head)

        self.linX1 = nn.Linear(dx, dim_ffX)
        self.linX2 = nn.Linear(dim_ffX, dx)
        self.normX1 = nn.LayerNorm(dx, epsilon=layer_norm_eps)
        self.normX2 = nn.LayerNorm(dx, epsilon=layer_norm_eps)
        self.dropoutX1 = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.dropoutX2 = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.dropoutX3 = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        self.linE1 = nn.Linear(de, dim_ffE)
        self.linE2 = nn.Linear(dim_ffE, de)
        self.normE1 = nn.LayerNorm(de, epsilon=layer_norm_eps)
        self.normE2 = nn.LayerNorm(de, epsilon=layer_norm_eps)
        self.dropoutE1 = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.dropoutE2 = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.dropoutE3 = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        self.lin_y1 = nn.Linear(dy, dim_ffy)
        self.lin_y2 = nn.Linear(dim_ffy, dy)
        self.norm_y1 = nn.LayerNorm(dy, epsilon=layer_norm_eps)
        self.norm_y2 = nn.LayerNorm(dy, epsilon=layer_norm_eps)
        self.dropout_y1 = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.dropout_y2 = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.dropout_y3 = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        self.activation = F.relu

    def forward(self, X, E, y, node_mask):
        """Pass the input through the encoder layer.
        X: (bs, n, d)
        E: (bs, n, n, d)
        y: (bs, dy)
        node_mask: (bs, n) Mask for the src keys per batch (optional)
        Output: newX, newE, new_y with the same shape.
        """
        newX, newE, new_y = self.self_attn(X, E, y, node_mask=node_mask)

        newX_d = self.dropoutX1(newX)
        X = self.normX1(X + newX_d)

        newE_d = self.dropoutE1(newE)
        E = self.normE1(E + newE_d)

        new_y_d = self.dropout_y1(new_y)
        y = self.norm_y1(y + new_y_d)

        ff_outputX = self.linX2(self.dropoutX2(self.activation(self.linX1(X))))
        ff_outputX = self.dropoutX3(ff_outputX)
        X = self.normX2(X + ff_outputX)

        ff_outputE = self.linE2(self.dropoutE2(self.activation(self.linE1(E))))
        ff_outputE = self.dropoutE3(ff_outputE)
        E = self.normE2(E + ff_outputE)

        ff_output_y = self.lin_y2(self.dropout_y2(self.activation(self.lin_y1(y))))
        ff_output_y = self.dropout_y3(ff_output_y)
        y = self.norm_y2(y + ff_output_y)

        return X, E, y


class NodeEdgeBlock(nn.Layer):
    """Self attention layer that also updates the representations on the edges."""

    def __init__(self, dx, de, dy, n_head):
        super().__init__()
        assert dx % n_head == 0, f"dx: {dx} -- n_head: {n_head}"
        self.dx = dx
        self.de = de
        self.dy = dy
        self.df = int(dx / n_head)
        self.n_head = n_head

        # Attention
        self.q = nn.Linear(dx, dx)
        self.k = nn.Linear(dx, dx)
        self.v = nn.Linear(dx, dx)

        # FiLM E to X
        self.e_add = nn.Linear(de, dx)
        self.e_mul = nn.Linear(de, dx)

        # FiLM y to E
        self.y_e_mul = nn.Linear(dy, dx)
        self.y_e_add = nn.Linear(dy, dx)

        # FiLM y to X
        self.y_x_mul = nn.Linear(dy, dx)
        self.y_x_add = nn.Linear(dy, dx)

        # Process y
        self.y_y = nn.Linear(dy, dy)
        self.x_y = Xtoy(dx, dy)
        self.e_y = Etoy(de, dy)

        # Output layers
        self.x_out = nn.Linear(dx, dx)
        self.e_out = nn.Linear(dx, de)
        self.y_out = nn.Sequential(nn.Linear(dy, dy), nn.ReLU(), nn.Linear(dy, dy))

    def forward(self, X, E, y, node_mask):
        """
        :param X: bs, n, d        node features
        :param E: bs, n, n, d     edge features
        :param y: bs, dz           global features
        :param node_mask: bs, n
        :return: newX, newE, new_y with the same shape.
        """
        bs, n, _ = X.shape
        x_mask = paddle.unsqueeze(node_mask, axis=-1).astype(X.dtype)  # bs, n, 1
        e_mask1 = paddle.unsqueeze(x_mask, axis=2)  # bs, n, 1, 1
        e_mask2 = paddle.unsqueeze(x_mask, axis=1)  # bs, 1, n, 1

        # 1. Map X to keys and queries
        Q = self.q(X) * x_mask
        K = self.k(X) * x_mask
        # No assert_correctly_masked checks in this block: they read a device value
        # (.item()), which breaks the SOT graph inside the layer loop.

        # 2. Reshape to (bs, n, n_head, df) with dx = n_head * df
        Q = paddle.reshape(Q, (Q.shape[0], Q.shape[1], self.n_head, self.df))
        K = paddle.reshape(K, (K.shape[0], K.shape[1], self.n_head, self.df))

        Q = paddle.unsqueeze(Q, axis=2)  # (bs, 1, n, n_head, df) (bs, n, 1, n_head, df)
        K = paddle.unsqueeze(K, axis=1)  # (bs, n, 1, n head, df) (bs, 1, n, n_head, df)

        # Compute unnormalized attentions. Y is (bs, n, n, n_head, df)
        Y = Q * K
        Y = Y / math.sqrt(Y.shape[-1])

        E1 = self.e_mul(E) * e_mask1 * e_mask2  # bs, n, n, dx
        E1 = paddle.reshape(
            E1, (E.shape[0], E.shape[1], E.shape[2], self.n_head, self.df)
        )

        E2 = self.e_add(E) * e_mask1 * e_mask2  # bs, n, n, dx
        E2 = paddle.reshape(
            E2, (E.shape[0], E.shape[1], E.shape[2], self.n_head, self.df)
        )

        # Incorporate edge features to the self attention scores.
        Y = Y * (E1 + 1) + E2  # (bs, n, n, n_head, df)

        # Incorporate y to E
        newE = paddle.flatten(Y, start_axis=3)  # bs, n, n, dx
        ye1 = paddle.unsqueeze(
            paddle.unsqueeze(self.y_e_add(y), axis=1), axis=1
        )  # bs, 1, 1, de
        ye2 = paddle.unsqueeze(paddle.unsqueeze(self.y_e_mul(y), axis=1), axis=1)
        newE = ye1 + (ye2 + 1) * newE

        # Output E
        newE = self.e_out(newE) * e_mask1 * e_mask2

        # Compute attentions. attn is still (bs, n, n, n_head, df)
        softmax_mask = paddle.expand(
            e_mask2, shape=(-1, n, -1, self.n_head)
        )  # bs, 1, n, 1   bs,n,n,n_head
        attn = masked_softmax(Y, softmax_mask, axis=2)  # bs, n, n, n_head, df

        V = self.v(X) * x_mask  # bs, n, dx
        V = paddle.reshape(V, (V.shape[0], V.shape[1], self.n_head, self.df))
        V = paddle.unsqueeze(V, axis=1)  # (bs, 1, n, n_head, df)

        # Compute weighted values
        weighted_V = attn * V
        weighted_V = paddle.sum(weighted_V, axis=2)

        # Send output to input dim
        weighted_V = paddle.flatten(weighted_V, start_axis=2)  # bs, n, dx

        # Incorporate y to X
        yx1 = paddle.unsqueeze(self.y_x_add(y), axis=1)
        yx2 = paddle.unsqueeze(self.y_x_mul(y), axis=1)
        newX = yx1 + (yx2 + 1) * weighted_V

        # Output X
        newX = self.x_out(newX) * x_mask

        # Process y based on X axnd E
        y = self.y_y(y)
        e_y = self.e_y(E, e_mask1, e_mask2)
        x_y = self.x_y(X, x_mask)
        new_y = y + x_y + e_y
        new_y = self.y_out(new_y)  # bs, dy

        return newX, newE, new_y


class Xtoy(nn.Layer):
    def __init__(self, dx, dy):
        """Map node features to global features"""
        super().__init__()
        self.lin = paddle.nn.Linear(in_features=4 * dx, out_features=dy)

    def forward(self, X, x_mask):
        """X: bs, n, dx."""
        x_mask = paddle.expand(x_mask, shape=[-1, -1, X.shape[-1]])
        float_imask = 1 - x_mask.astype("float32")
        m = paddle.sum(X, axis=1) / paddle.sum(x_mask, axis=1)
        mi = paddle.min(X + 1e6 * float_imask, axis=1)
        ma = paddle.max(X - 1e6 * float_imask, axis=1)
        std = paddle.sum((X - m.unsqueeze(1)) ** 2 * x_mask, axis=1) / paddle.sum(
            x_mask, axis=1
        )
        z = paddle.concat([m, mi, ma, std], axis=1)
        out = self.lin(z)
        return out


class Etoy(nn.Layer):
    def __init__(self, d, dy):
        """Map edge features to global features."""
        super().__init__()
        self.lin = paddle.nn.Linear(in_features=4 * d, out_features=dy)

    def forward(self, E, e_mask1, e_mask2):
        """
        E: bs, n, n, de
        Features relative to the diagonal of E could potentially be added.
        """
        mask = paddle.expand(e_mask1 * e_mask2, shape=[-1, -1, -1, E.shape[-1]])
        float_imask = 1 - mask.astype("float32")
        # Reduce over the flattened n * n axis: CINN in Paddle 3.3 generates an invalid
        # kernel for chained reductions such as min(min(E, axis=2), axis=1).
        E = paddle.flatten(E, start_axis=1, stop_axis=2)  # bs, n * n, de
        mask = paddle.flatten(mask, start_axis=1, stop_axis=2)
        float_imask = paddle.flatten(float_imask, start_axis=1, stop_axis=2)
        divide = paddle.sum(mask, axis=1)
        m = paddle.sum(E, axis=1) / divide
        mi = paddle.min(E + 1e6 * float_imask, axis=1)
        ma = paddle.max(E - 1e6 * float_imask, axis=1)
        std = paddle.sum((E - m.unsqueeze(1)) ** 2 * mask, axis=1) / divide
        z = paddle.concat([m, mi, ma, std], axis=1)
        out = self.lin(z)
        return out


def masked_softmax(x, mask, axis=-1):
    """
    Perform softmax over masked values in `x`.

    Args:
        x: Tensor, the input data.
        mask: Tensor, the binary mask of the same shape as `x`.
        axis: The axis to apply softmax.

    Returns:
        Tensor with masked softmax applied.
    """
    # A Python branch on a device value would break the SOT graph, so the all-masked
    # case is selected with paddle.where instead of an early return.
    all_masked = paddle.sum(mask) == 0

    # TODO: ndim check: only support adding dimensions backwards now
    x_dims = x.ndim
    mask_dims = mask.ndim
    if mask_dims < x_dims:
        diff = x_dims - mask_dims
        mask = paddle.unsqueeze(mask, axis=[-1] * diff)
        repeat_times = [1] * mask_dims + [x.shape[i] for i in range(mask_dims, x_dims)]
        mask = paddle.tile(mask, repeat_times=repeat_times)

    x_masked = x.clone()
    x_masked = paddle.where(
        mask == 0, paddle.to_tensor(-float("inf"), dtype=x.dtype), x_masked
    )
    # If everything is masked, x_masked is all -inf and softmax would give NaN, also in
    # the gradient; use zeros there, since x is returned in that case.
    x_masked = paddle.where(all_masked, paddle.zeros_like(x_masked), x_masked)

    return paddle.where(
        all_masked, x, paddle.nn.functional.softmax(x_masked, axis=axis)
    )

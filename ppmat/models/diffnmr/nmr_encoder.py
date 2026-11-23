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

import paddle
import paddle.nn as nn
import paddle.nn.functional as F


class H1nmr_encoder(nn.Layer):
    def __init__(
        self,
        d_model,
        dim_feedforward,
        n_head,
        num_layers,
        drop_prob,
        peakwidthemb_num,
        splitemb_num,
        integralemb_num,
    ):
        super(H1nmr_encoder, self).__init__()

        # for src padding mask
        self.num_heads = n_head

        self.embed = H1nmr_embedding(
            dim=d_model,
            drop_prob=drop_prob,
            peakwidthemb_num=peakwidthemb_num,
            splitemb_num=splitemb_num,
            integralemb_num=integralemb_num,
        )

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=dim_feedforward,
            dropout=drop_prob,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x, src_mask):
        # input format: [batch, len_peak, feat_dim]
        x_emb = self.embed(x, src_mask)

        # process for src_key_padding_mask
        pad_mask = src_mask == 1
        bsz, src_len, _ = x_emb.shape
        pad_mask = pad_mask.reshape([bsz, 1, 1, src_len]).expand(
            [-1, self.num_heads, src_len, -1]
        )

        out = self.encoder(src=x_emb, src_mask=pad_mask)
        return out


class C13nmr_encoder(nn.Layer):
    def __init__(self, d_model, dim_feedforward, n_head, num_layers, drop_prob):
        super(C13nmr_encoder, self).__init__()

        # for src padding mask
        self.num_heads = n_head

        self.embed = C13nmr_embedding(dim=d_model, drop_prob=drop_prob)
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=dim_feedforward,
            dropout=drop_prob,
            normalize_before=False,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x, src_mask):
        # input format: [batch, len_peak, feat_dim]
        x_emb = self.embed(x, src_mask)

        # process for src_key_padding_mask
        pad_mask = src_mask == 1
        bsz, src_len, _ = x_emb.shape
        pad_mask = pad_mask.reshape([bsz, 1, 1, src_len]).expand(
            [-1, self.num_heads, src_len, -1]
        )

        out = self.encoder(src=x_emb, src_mask=pad_mask)
        return out


class MaskedAttentionPool(nn.Layer):
    def __init__(self, dim):
        super(MaskedAttentionPool, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )  # 移除了Softmax，需手动处理

    def forward(self, x, mask=None):
        # x: [batch, seq_len, dim]
        # mask: [batch, seq_len] （1: valid，0: pad）
        attn_scores = self.attention(x)  # [batch, seq_len, 1]

        # add mask processing
        if mask is not None:
            # Set the attention scores at padding positions to -∞,
            # resulting in zero weight after Softmax
            attn_scores = attn_scores.masked_fill(
                mask.unsqueeze(-1) == 0, -float("inf")
            )

        attn_weights = F.softmax(attn_scores, axis=1)  # [batch, seq_len, 1]
        return (x * attn_weights).sum(axis=1)  # [batch, dim]


class NMR_fusion(nn.Layer):
    def __init__(
        self,
        dim_h=1024,
        dim_c=256,
        hidden_dim=512,
        n_head=8,
        out_dim=512,
        bi_crossattn_fusion_mode="",
        pool_mode="",
        crossmodal_fusion_mode="",
    ):
        super(NMR_fusion, self).__init__()

        # projection layer
        self.proj_h = nn.Linear(dim_h, hidden_dim)
        self.proj_c = nn.Linear(dim_c, hidden_dim)
        # Bidirectional cross-attention
        self.cross_attn_ab = nn.MultiHeadAttention(hidden_dim, num_heads=n_head)
        self.cross_attn_ba = nn.MultiHeadAttention(hidden_dim, num_heads=n_head)

        self.bi_crossattn_fusion_mode = bi_crossattn_fusion_mode
        self.pool_mode = pool_mode
        self.crossmodal_fusion = crossmodal_fusion_mode

        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

        self.gate_linear = nn.Linear(hidden_dim, 1)
        self.attn_pool = MaskedAttentionPool(dim=self.hidden_dim)
        self.weighted_sum = nn.Linear(1024, 1)
        self.concat_linear = nn.Linear(1024, 512)

        # for src padding mask
        self.num_heads = n_head

    def masked_mean_pool(self, tensor, mask):
        # tensor: [batch, seq_len, dim]
        # mask: [batch, seq_len] (1: valid，0: pad)
        lengths = mask.sum(axis=1, keepdim=True)  # [batch, 1]
        masked = tensor * mask.unsqueeze(-1)  # zero out padding positions
        return masked.sum(axis=1) / (lengths + 1e-6)  # [batch, dim]

    def forward(self, tensor_Hnmr, mask_H, tensor_Cnmr, mask_C):

        max_len_H = mask_H.sum(axis=-1).max().item()
        mask_H = mask_H[:, : int(max_len_H)]
        tensor_Hnmr = tensor_Hnmr[:, : int(max_len_H), :]
        max_len_C = mask_C.sum(axis=-1).max().item()
        mask_C = mask_C[:, : int(max_len_C)]
        tensor_Cnmr = tensor_Cnmr[:, : int(max_len_C), :]

        # project to uniform dimension
        H_aligned = self.proj_h(tensor_Hnmr)  # [B, Lh, D]
        C_aligned = self.proj_c(tensor_Cnmr)  # [B, Lc, D]

        # bidirectonal cross-attention
        pad_mask_H = mask_H == 1
        bsz_H, src_len_H, _ = H_aligned.shape
        pad_mask_C = mask_C == 1
        bsz_C, src_len_C, _ = C_aligned.shape
        pad_mask_H = pad_mask_H.reshape([bsz_H, 1, 1, src_len_H]).expand(
            [-1, self.num_heads, src_len_C, -1]
        )
        pad_mask_C = pad_mask_C.reshape([bsz_C, 1, 1, src_len_C]).expand(
            [-1, self.num_heads, src_len_H, -1]
        )

        attn_H2C = self.cross_attn_ab(
            query=H_aligned,
            key=C_aligned,
            value=C_aligned,
            attn_mask=pad_mask_C,
        )
        attn_C2H = self.cross_attn_ba(
            query=C_aligned,
            key=H_aligned,
            value=H_aligned,
            attn_mask=pad_mask_H,
        )

        # combine the cross-attention output of two modalities with the origin features
        if self.bi_crossattn_fusion_mode == "concat":
            # Method 1: Concatenate outputs from two directions
            fused_H = paddle.concat([H_aligned, attn_H2C], axis=-1)  # [B, Lh, 2*D]
            fused_C = paddle.concat([C_aligned, attn_C2H], axis=-1)  # [B, Lc, 2*D]

        elif self.bi_crossattn_fusion_mode == "add":
            # Method 2: Residual connection
            fused_H = H_aligned + attn_H2C  # [B, Lh, 2*D]
            fused_C = C_aligned + attn_C2H  # [B, Lc, 2*D]

        elif self.bi_crossattn_fusion_mode == "gated":
            # Method 3: Gated Fusion (Adaptive Weights)
            gate_H = F.sigmoid(self.gate_linear(attn_H2C))
            fused_H = (1 - gate_H) * H_aligned + gate_H * attn_H2C
            gate_C = F.sigmoid(self.gate_linear(attn_C2H))
            fused_C = (1 - gate_C) * C_aligned + gate_C * attn_C2H

        else:
            fused_H, fused_C = attn_H2C, attn_C2H

        # Intra-modal Aggregation (Temporal Pooling)
        if self.pool_mode == "mean_pool":
            # method 1：average pooling
            global_H = self.masked_mean_pool(fused_H, mask_H)  # [B, Dh]
            global_C = self.masked_mean_pool(fused_C, mask_C)  # [B, Dc]

        elif self.pool_mode == "attn_pool":
            # Apply attention pooling to each of the two modalities separately
            global_H = self.attn_pool(fused_H, mask_H)  # [B, D*]
            global_C = self.attn_pool(fused_C, mask_C)  # [B, D*]

        # cross-modal fusion (obtained spectrum_embedding)
        if self.crossmodal_fusion == "concat_linear":
            merged = paddle.concat([global_H, global_C], axis=-1)  # [B, 2D*]
            global_output = self.concat_linear(merged)  # [B, D]
        elif self.crossmodal_fusion == "weighted_sum":
            merged = paddle.concat([global_H, global_C], axis=-1)
            # option
            # merged = global_H + global_C
            gate = F.sigmoid(self.weighted_sum(merged))  # [B, 1]
            global_output = gate * global_H + (1 - gate) * global_C  # [B, D*]
        else:
            global_output = (global_H, global_C)

        spectrum_token_enc = paddle.concat(
            [fused_H, fused_C], axis=1
        )  # [B, Lh+Lc, D or 2*D]
        spectrum_token_mask = paddle.concat([mask_H, mask_C], axis=1)  # [B, Lh+Lc]

        return global_output, (spectrum_token_enc, spectrum_token_mask)


class NMR_fusion_H(nn.Layer):
    def __init__(
        self,
        dim_h=1024,
        dim_c=256,
        hidden_dim=512,
        n_head=8,
        out_dim=512,
        bi_crossattn_fusion_mode="",
        pool_mode="",
        crossmodal_fusion_mode="",
    ):
        super(NMR_fusion, self).__init__()

        # projection layer
        self.proj_h = nn.Linear(dim_h, hidden_dim)
        self.proj_c = nn.Linear(dim_c, hidden_dim)

        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

        self.attn_pool = MaskedAttentionPool(dim=self.hidden_dim)

        # for src padding mask
        self.num_heads = n_head

    def masked_mean_pool(self, tensor, mask):
        # tensor: [batch, seq_len, dim]
        # mask: [batch, seq_len] (1: valid，0: pad)
        lengths = mask.sum(axis=1, keepdim=True)  # [batch, 1]
        masked = tensor * mask.unsqueeze(-1)  # zero out padding positions
        return masked.sum(axis=1) / (lengths + 1e-6)  # [batch, dim]

    def forward(self, tensor_Hnmr, mask_H, tensor_Cnmr, mask_C):

        max_len_H = mask_H.sum(axis=-1).max().item()
        mask_H = mask_H[:, : int(max_len_H)]
        tensor_Hnmr = tensor_Hnmr[:, : int(max_len_H), :]
        max_len_C = mask_C.sum(axis=-1).max().item()
        mask_C = mask_C[:, : int(max_len_C)]
        tensor_Cnmr = tensor_Cnmr[:, : int(max_len_C), :]

        # project to uniform dimension
        H_aligned = self.proj_h(tensor_Hnmr)
        C_aligned = self.proj_c(tensor_Cnmr)

        fused_H = H_aligned
        fused_C = C_aligned

        # Apply attention pooling to each of the two modalities separately
        global_H = self.attn_pool(fused_H, mask_H)
        global_C = self.attn_pool(fused_C, mask_C)

        return global_H, global_C


class NMR_encoder(nn.Layer):
    def __init__(
        self,
        dim_H,
        dimff_H,
        dim_C,
        dimff_C,
        hidden_dim,
        n_head,
        num_layers,
        drop_prob,
        peakwidthemb_num,
        splitemb_num,
        integralemb_num,
    ):
        super(NMR_encoder, self).__init__()
        self.H1nmr_encoder = H1nmr_encoder(
            d_model=dim_H,
            dim_feedforward=dimff_H,
            n_head=n_head,
            num_layers=num_layers,
            drop_prob=drop_prob,
            peakwidthemb_num=peakwidthemb_num,
            splitemb_num=splitemb_num,
            integralemb_num=integralemb_num,
        )

        self.C13nmr_encoder = C13nmr_encoder(
            d_model=dim_C,
            dim_feedforward=dimff_C,
            n_head=n_head,
            num_layers=num_layers,
            drop_prob=drop_prob,
        )

        self.NMR_fusion = NMR_fusion(
            dim_H,
            dim_C,
            hidden_dim,
            n_head,
            bi_crossattn_fusion_mode="add",
            pool_mode="attn_pool",
            crossmodal_fusion_mode="concat_linear",
        )

    def create_mask(self, batch_size, max_seq_len, num_peak):

        mask = paddle.zeros([batch_size, max_seq_len], dtype="float32")
        for i, length in enumerate(num_peak):
            mask[i, :length] = 1
        return mask

    def forward(self, condition):
        H1nmr, num_H_peak, C13nmr, num_C_peak = condition

        batch_size, max_seq_len_H, _ = H1nmr.shape
        mask_H = self.create_mask(batch_size, max_seq_len_H, num_H_peak)
        _, max_seq_len_C = C13nmr.shape
        mask_C = self.create_mask(batch_size, max_seq_len_C, num_C_peak)

        h_feat = self.H1nmr_encoder(H1nmr, mask_H)  # [batch, h_seq, h_dim]
        c_feat = self.C13nmr_encoder(C13nmr, mask_C)  # [batch, c_seq, c_dim]

        fused_feat = self.NMR_fusion(
            h_feat, mask_H, c_feat, mask_C
        )  # [batch, fusion_dim]

        return fused_feat


class NMR_encoder_H(nn.Layer):
    def __init__(
        self,
        dim_H,
        dimff_H,
        dim_C,
        dimff_C,
        hidden_dim,
        n_head,
        num_layers,
        drop_prob,
        peakwidthemb_num,
        splitemb_num,
        integralemb_num,
    ):
        super(NMR_encoder_H, self).__init__()
        self.H1nmr_encoder = H1nmr_encoder(
            d_model=dim_H,
            dim_feedforward=dimff_H,
            n_head=n_head,
            num_layers=num_layers,
            drop_prob=drop_prob,
            peakwidthemb_num=peakwidthemb_num,
            splitemb_num=splitemb_num,
            integralemb_num=integralemb_num,
        )

        self.C13nmr_encoder = C13nmr_encoder(
            d_model=dim_C,
            dim_feedforward=dimff_C,
            n_head=n_head,
            num_layers=num_layers,
            drop_prob=drop_prob,
        )

        self.NMR_fusion = NMR_fusion_H(
            dim_H,
            dim_C,
            hidden_dim,
            n_head,
            bi_crossattn_fusion_mode="gated",
            pool_mode="attn_pool",
            crossmodal_fusion_mode="weighted_sum",
        )

    def create_mask(self, batch_size, max_seq_len, num_peak):

        mask = paddle.zeros([batch_size, max_seq_len], dtype="float32")
        for i, length in enumerate(num_peak):
            mask[i, :length] = 1
        return mask

    def forward(self, condition):
        H1nmr, num_H_peak, C13nmr, num_C_peak = condition

        batch_size, max_seq_len_H, _ = H1nmr.shape
        mask_H = self.create_mask(batch_size, max_seq_len_H, num_H_peak)
        _, max_seq_len_C = C13nmr.shape
        mask_C = self.create_mask(batch_size, max_seq_len_C, num_C_peak)

        h_feat = self.H1nmr_encoder(H1nmr, mask_H)  # [batch, h_seq, h_dim]
        c_feat = self.C13nmr_encoder(C13nmr, mask_C)  # [batch, c_seq, c_dim]

        global_H, global_C = self.NMR_fusion(
            h_feat, mask_H, c_feat, mask_C
        )  # [batch, fusion_dim]

        return global_H, global_C


class RBFEncoder(nn.Layer):
    def __init__(self, min, max, bins):
        super(RBFEncoder, self).__init__()
        self.centers = self.create_parameter(
            shape=[bins],
            default_initializer=nn.initializer.Assign(paddle.linspace(min, max, bins)),
        )
        self.centers.stop_gradient = True
        self.sigma = (max - min) / (bins - 1)  # adaptive bandwidth

    def forward(self, x):
        # x: (...,)
        diff = x.unsqueeze(-1) - self.centers  # (..., bins)
        return paddle.exp(-0.5 * (diff / self.sigma).pow(2))


class RBFEncoder_Jcouple(nn.Layer):
    def __init__(self, min1=0, max1=26, bins1=131, min2=27, max2=58, bins2=32):
        super(RBFEncoder_Jcouple, self).__init__()

        centers1 = paddle.linspace(min1, max1, bins1)
        sigma1 = (max1 - min1) / (bins1 - 1)  # 20/99 ≈ 0.202

        centers2 = paddle.linspace(min2, max2, bins2)
        sigma2 = (max2 - min2) / (bins2 - 1)  # 30/29 ≈ 1.034

        # 合并参数
        self.centers = self.create_parameter(
            shape=[bins1 + bins2],
            default_initializer=nn.initializer.Assign(
                paddle.concat([centers1, centers2])
            ),
        )
        self.centers.stop_gradient = True
        self.sigma = self.create_parameter(
            shape=[bins1 + bins2],
            default_initializer=nn.initializer.Assign(
                paddle.concat(
                    [paddle.full([bins1], sigma1), paddle.full([bins2], sigma2)]
                )
            ),
        )
        self.sigma.stop_gradient = True

    def forward(self, x):
        diff = x.unsqueeze(-1) - self.centers  # (..., 130)
        return paddle.exp(-0.5 * (diff / self.sigma).pow(2))


class H1nmr_embedding(nn.Layer):
    def __init__(
        self,
        split_dim=64,
        peakwidth_dim=40,
        integral_dim=32,
        H_shift_min=-1,
        H_shift_max=10,
        H_shift_bin=111,
        min_j=0,
        max_j=58,
        j_bins1=131,
        j_bins2=32,
        hidden=1024,
        dim=1024,
        drop_prob=0.1,
        *,
        peakwidthemb_num,
        splitemb_num,
        integralemb_num,
    ):
        super(H1nmr_embedding, self).__init__()

        self.shift_emb = RBFEncoder(
            min=H_shift_min, max=H_shift_max, bins=H_shift_bin
        )  # Covering common 1H ranges

        self.peakwidth_emb = nn.Embedding(
            peakwidthemb_num, peakwidth_dim, padding_idx=0
        )

        self.split_emb = nn.Embedding(splitemb_num, split_dim, padding_idx=0)

        self.integral_emb = nn.Embedding(integralemb_num, integral_dim, padding_idx=0)

        self.J_emb = RBFEncoder_Jcouple(
            min1=min_j, max1=26, bins1=j_bins1, min2=27, max2=max_j, bins2=j_bins2
        )

        self.d_model = (
            split_dim + peakwidth_dim + integral_dim + H_shift_bin + j_bins1 + j_bins2
        )

        self.peak_fuser = peak_fuser(self.d_model, dim, drop_prob)

    def forward(self, h1nmr, src_mask):

        hnmr = h1nmr

        h_shift, peakwidth, split, integral, j_couple = (
            hnmr[:, :, 0],
            hnmr[:, :, 1],
            hnmr[:, :, 2],
            hnmr[:, :, 3],
            hnmr[:, :, 4:],
        )

        h_shift_emb = self.shift_emb(h_shift) * src_mask.unsqueeze(-1)
        peakwidth_emb = self.peakwidth_emb(peakwidth.astype("int64"))
        split_emb = self.split_emb(split.astype("int64"))
        integral_emb = self.integral_emb(integral.astype("int64"))

        J_emb = self.J_emb(j_couple)
        J_emb = paddle.sum(J_emb, axis=-2) * src_mask.unsqueeze(-1)

        hnmr_emb = paddle.concat(
            [h_shift_emb, peakwidth_emb, split_emb, integral_emb, J_emb], axis=-1
        )
        hnmr_emb = self.peak_fuser(hnmr_emb)

        return hnmr_emb


class C13nmr_embedding(nn.Layer):
    def __init__(
        self,
        C_shift_min=-15,
        C_shift_max=229,
        C_bins=245,
        hidden=512,
        dim=256,
        drop_prob=0.1,
    ):
        super(C13nmr_embedding, self).__init__()

        self.shift_emb = RBFEncoder(min=C_shift_min, max=C_shift_max, bins=C_bins)

        self.peak_fuser = peak_fuser(C_bins, dim, drop_prob)

    def forward(self, c13nmr, src_mask):

        cnmr = c13nmr

        c_shift_emb = self.shift_emb(cnmr) * src_mask.unsqueeze(-1)

        cnmr_emb = self.peak_fuser(c_shift_emb)

        return cnmr_emb


class peak_fuser(nn.Layer):
    def __init__(self, d_model, hidden, drop_prob=0.1):
        super(peak_fuser, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden), nn.GELU(), nn.Dropout(drop_prob)
        )

    def forward(self, x):
        return self.net(x)

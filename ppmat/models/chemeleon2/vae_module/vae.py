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

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from ppmat.models.chemeleon2.common import DiagonalGaussianDistribution
from ppmat.models.chemeleon2.common import apply_augmentation
from ppmat.models.chemeleon2.common import apply_noise
from ppmat.models.chemeleon2.common.schema import CrystalBatch
from ppmat.models.chemeleon2.common.schema import build_structure_array
from ppmat.utils.crystal import lattice_params_to_matrix_paddle


class Chemeleon2VAEModule(nn.Layer):
    def __init__(
        self,
        encoder,
        decoder,
        latent_dim,
        loss_weights,
        augmentation=None,
        noise=None,
        atom_type_predict=True,
        fa_proj_dim=256,
    ):
        super().__init__()

        from ppmat.models import build_model

        if isinstance(encoder, dict):
            self.encoder = build_model(encoder)
        else:
            self.encoder = encoder

        if isinstance(decoder, dict):
            self.decoder = build_model(decoder)
        else:
            self.decoder = decoder
        self.latent_dim = latent_dim
        self.loss_weights = loss_weights
        self.augmentation = augmentation
        self.noise = noise
        self.atom_type_predict = atom_type_predict

        self.quant_conv = nn.Linear(
            self.encoder.hidden_dim, 2 * latent_dim, bias_attr=False
        )
        self.post_quant_conv = nn.Linear(
            latent_dim, self.decoder.hidden_dim, bias_attr=False
        )

        if self.loss_weights.get("fa", 0) > 0:
            self.proj = nn.Linear(self.latent_dim, fa_proj_dim)

    def encode(self, batch):
        encoded = self.encoder(batch)
        encoded["moments"] = self.quant_conv(encoded["x"])
        encoded["posterior"] = DiagonalGaussianDistribution(encoded["moments"])
        return encoded

    def decode(self, encoded):
        encoded["x"] = self.post_quant_conv(encoded["x"])
        decoder_out = self.decoder(encoded)
        return decoder_out

    def reconstruct(self, decoder_out, batch):
        batch_rec = CrystalBatch()

        if decoder_out["atom_types"].ndim == 2:
            batch_rec.atom_types = paddle.argmax(decoder_out["atom_types"], axis=1)
        else:
            batch_rec.atom_types = decoder_out["atom_types"]

        batch_rec.frac_coords = decoder_out["frac_coords"]

        # Scale back the decoder lengths by num_atoms^(1/3)
        lengths_scaled = decoder_out["lengths"]
        num_atoms = batch.num_atoms
        if num_atoms.ndim <= 1:
            num_atoms = num_atoms.unsqueeze(-1)

        lengths = lengths_scaled * num_atoms ** (1 / 3)
        batch_rec.lengths = lengths
        batch_rec.lengths_scaled = lengths_scaled

        angles_radians = decoder_out["angles"]
        angles_degrees = paddle.rad2deg(angles_radians)
        batch_rec.angles = angles_degrees
        batch_rec.angles_radians = angles_radians

        batch_rec.lattices = lattice_params_to_matrix_paddle(lengths, angles_degrees)

        batch_rec.num_atoms = batch.num_atoms
        batch_rec.batch = batch.batch
        batch_rec.token_idx = batch.token_idx
        batch_rec.num_nodes = batch.num_nodes
        batch_rec.num_graphs = batch.num_graphs
        return batch_rec

    def _convert_train_batch(self, batch):
        structure_array = batch["structure_array"]
        crystal_batch = build_structure_array(CrystalBatch(), structure_array)
        crystal_batch.lengths = structure_array["lengths"]
        crystal_batch.angles = structure_array["angles"]
        crystal_batch.angles_radians = paddle.deg2rad(structure_array["angles"])
        num_atoms_tensor = structure_array["num_atoms"]
        if num_atoms_tensor.ndim == 1:
            num_atoms_tensor = num_atoms_tensor.unsqueeze(-1)
        crystal_batch.lengths_scaled = structure_array["lengths"] / (
            num_atoms_tensor ** (1 / 3)
        )
        cart_coords = paddle.matmul(
            crystal_batch.frac_coords, crystal_batch.lattices[crystal_batch.batch]
        )
        crystal_batch.cart_coords = cart_coords
        return crystal_batch

    def forward(self, batch):
        crystal_batch = self._convert_train_batch(batch)
        loss_dict = self.calculate_loss(crystal_batch, training=True)
        loss_dict["loss"] = loss_dict.get("total_loss", paddle.to_tensor([0.0]))
        return {"loss_dict": loss_dict}

    def calculate_loss(self, batch, training=True):
        if training and self.augmentation is not None:
            translate = self.augmentation.get("translate", False)
            rotate = self.augmentation.get("rotate", False)
            batch = apply_augmentation(batch, translate=translate, rotate=rotate)

        if training and self.noise is not None:
            ratio = self.noise.get("ratio", 0.0)
            corruption_scale = self.noise.get("corruption_scale", 0.1)
            if ratio > 0:
                batch = apply_noise(
                    batch, ratio=ratio, corruption_scale=corruption_scale
                )

        # Direct encode/decode calls avoid recursion through forward
        encoded = self.encode(batch)
        z = encoded["posterior"].sample()
        encoded["x"] = z
        encoded["z"] = z
        decoder_out = self.decode(encoded)

        loss_atom_types = 0
        if self.atom_type_predict:
            loss_atom_types = F.cross_entropy(
                decoder_out["atom_types"], batch.atom_types
            )
        loss_lengths = F.mse_loss(decoder_out["lengths"], batch.lengths_scaled)
        loss_angles = F.mse_loss(decoder_out["angles"], batch.angles_radians)
        loss_frac_coords = F.mse_loss(decoder_out["frac_coords"], batch.frac_coords)

        loss_kl = encoded["posterior"].kl().mean()

        fa_loss = 0
        if self.loss_weights.get("fa", 0) > 0:
            if not hasattr(batch, "mace_features") or batch.mace_features is None:
                raise ValueError(
                    "loss_weights['fa'] > 0 requires 'mace_features' in the "
                    "structure_array. Enable the 'mace_features' property in "
                    "MP20Dataset.property_names or set loss_weights['fa'] to 0."
                )
            mace_features = getattr(batch, "mace_features")
            z = self.proj(encoded["z"])
            z_norm = F.normalize(z, axis=-1)
            mace_features_norm = F.normalize(mace_features, axis=-1)
            z_cos_sim = paddle.einsum("ij,kj->ik", z_norm, z_norm)
            mace_cos_sim = paddle.einsum(
                "ij,kj->ik", mace_features_norm, mace_features_norm
            )
            diff = paddle.abs(z_cos_sim - mace_cos_sim)
            fa_loss_1 = F.relu(diff - 0.25).mean()
            fa_loss_2 = F.relu(0.5 - F.cosine_similarity(mace_features, z)).mean()
            fa_loss = fa_loss_1 + fa_loss_2

        loss = (
            self.loss_weights.get("atom_types", 1.0) * loss_atom_types
            + self.loss_weights.get("lengths", 1.0) * loss_lengths
            + self.loss_weights.get("angles", 1.0) * loss_angles
            + self.loss_weights.get("frac_coords", 1.0) * loss_frac_coords
            + self.loss_weights.get("kl", 1.0) * loss_kl
            + self.loss_weights.get("fa", 0.0) * fa_loss
        )

        return {
            "total_loss": loss.mean(),
            "loss_atom_types": loss_atom_types,
            "loss_lengths": loss_lengths,
            "loss_angles": loss_angles,
            "loss_frac_coords": loss_frac_coords,
            "loss_kl": loss_kl,
            "fa_loss": fa_loss,
        }

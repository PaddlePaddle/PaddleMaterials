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

"""Single-task Paddle implementation of the UMA eSCN model.

The eSCN architecture is derived from FairChem, Copyright Meta Platforms, Inc.
and affiliates, and is used under the MIT license.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import paddle

from ppmat.models.common.radial_basis import PolynomialEnvelope

from .common.rotation import eulers_to_wigner
from .common.rotation import init_edge_rot_euler_angles
from .common.so3 import CoefficientMapping
from .escn_md_block import ESCNMDInteractionBlock
from .nn.layer_norm import EquivariantRMSNorm
from .nn.radial import RadialMLP
from .nn.so3_layers import SO3_Linear


class _UMAGaussianSmearing(paddle.nn.Layer):
    """Gaussian distance expansion with UMA's pretrained basis width."""

    def __init__(self, start: float, stop: float, num_gaussians: int) -> None:
        super().__init__()
        offset = paddle.linspace(start, stop, num_gaussians)
        spacing = 2.0 * (offset[1] - offset[0]).item()
        self.coeff = -0.5 / spacing**2
        self.register_buffer("offset", offset)

    def forward(self, distances: paddle.Tensor) -> paddle.Tensor:
        distances = distances.reshape([-1, 1]) - self.offset.reshape([1, -1])
        return paddle.exp(self.coeff * paddle.pow(distances, 2))


class EdgeDegreeEmbedding(paddle.nn.Layer):
    """Initialize equivariant node features from radial edge features."""

    def __init__(
        self,
        sphere_channels: int,
        edge_channels_list: list[int],
        rescale_factor: float,
        mapping,
    ) -> None:
        super().__init__()
        self.sphere_channels = sphere_channels
        self.m0_components = mapping.m_size[0]
        self.radial = RadialMLP(
            [*edge_channels_list, self.m0_components * sphere_channels]
        )
        self.rescale_factor = rescale_factor

    def forward(
        self,
        x: paddle.Tensor,
        edge_features: paddle.Tensor,
        edge_index: paddle.Tensor,
        wigner_inv: paddle.Tensor,
    ) -> paddle.Tensor:
        radial = self.radial(edge_features).reshape(
            [-1, self.m0_components, self.sphere_channels]
        )
        edge_embedding = paddle.bmm(
            wigner_inv[:, :, : self.m0_components], radial
        ).astype(x.dtype)
        return x.index_add(
            axis=0,
            index=edge_index[1],
            value=edge_embedding / self.rescale_factor,
        )


class UMA(paddle.nn.Layer):
    """Single-task UMA model for direct energy and force prediction."""

    def __init__(
        self,
        max_num_elements: int = 100,
        sphere_channels: int = 128,
        edge_channels: int = 128,
        hidden_channels: int = 128,
        num_distance_basis: int = 64,
        num_layers: int = 4,
        lmax: int = 2,
        mmax: int = 2,
        cutoff: float = 6.0,
        loss_weights_dict: dict[str, float] | None = None,
    ) -> None:
        super().__init__()
        if lmax < 1:
            raise ValueError("Direct force prediction requires lmax >= 1.")

        self.sphere_channels = sphere_channels
        self.lmax = lmax
        self.mmax = mmax
        self.cutoff = cutoff
        self.loss_weights_dict = (
            {"energy": 10.0, "forces": 30.0}
            if loss_weights_dict is None
            else loss_weights_dict
        )

        # FairChem's factorized Wigner-D path is substantially faster than the
        # matrix-exponential implementation in the shared e3nn module.
        jd_path = Path(__file__).with_name("Jd.pdparams")
        jd_tensors = paddle.load(str(jd_path))
        if len(jd_tensors) <= lmax:
            raise ValueError(f"{jd_path} does not contain coefficients for l={lmax}.")
        for degree in range(lmax + 1):
            jd = jd_tensors[degree]
            if not paddle.is_tensor(jd):
                jd = paddle.to_tensor(jd)
            self.register_buffer(
                f"Jd_{degree}",
                jd,
            )

        self.mapping = CoefficientMapping(lmax, mmax)
        coefficient_index = self.mapping.coefficient_idx(lmax, mmax)
        self.register_buffer("coefficient_index", coefficient_index, persistable=False)

        self.atom_embedding = paddle.nn.Embedding(max_num_elements, sphere_channels)
        self.distance_expansion = _UMAGaussianSmearing(
            0.0,
            cutoff,
            num_distance_basis,
        )
        self.source_embedding = paddle.nn.Embedding(max_num_elements, edge_channels)
        self.target_embedding = paddle.nn.Embedding(max_num_elements, edge_channels)
        paddle.nn.initializer.Uniform(-0.001, 0.001)(self.source_embedding.weight)
        paddle.nn.initializer.Uniform(-0.001, 0.001)(self.target_embedding.weight)
        edge_channels_list = [
            num_distance_basis + 2 * edge_channels,
            edge_channels,
            edge_channels,
        ]
        self.edge_degree_embedding = EdgeDegreeEmbedding(
            sphere_channels,
            edge_channels_list,
            rescale_factor=5.0,
            mapping=self.mapping,
        )
        self.envelope = PolynomialEnvelope(exponent=5)
        self.blocks = paddle.nn.LayerList(
            [
                ESCNMDInteractionBlock(
                    sphere_channels,
                    hidden_channels,
                    lmax,
                    mmax,
                    self.mapping,
                    edge_channels_list,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = EquivariantRMSNorm(lmax, sphere_channels)
        self.energy_head = paddle.nn.Sequential(
            paddle.nn.Linear(sphere_channels, hidden_channels, bias_attr=True),
            paddle.nn.SiLU(),
            paddle.nn.Linear(hidden_channels, hidden_channels, bias_attr=True),
            paddle.nn.SiLU(),
            paddle.nn.Linear(hidden_channels, 1, bias_attr=True),
        )
        self.force_head = SO3_Linear(sphere_channels, 1, lmax=1)

    def _wigner(
        self, edge_vectors: paddle.Tensor
    ) -> tuple[paddle.Tensor, paddle.Tensor]:
        jd = [
            getattr(self, f"Jd_{degree}").astype(edge_vectors.dtype)
            for degree in range(self.lmax + 1)
        ]
        eulers = init_edge_rot_euler_angles(edge_vectors)
        wigner = eulers_to_wigner(eulers, 0, self.lmax, jd)
        wigner_inv = wigner.transpose([0, 2, 1])
        if self.mmax != self.lmax:
            wigner = paddle.index_select(wigner, self.coefficient_index, axis=1)
            wigner_inv = paddle.index_select(wigner_inv, self.coefficient_index, axis=2)
        to_m = self.mapping.to_m.astype(wigner.dtype)
        wigner = paddle.einsum("mk,nkj->nmj", to_m, wigner)
        wigner_inv = paddle.einsum("njk,mk->njm", wigner_inv, to_m)
        return wigner, wigner_inv

    def _forward(self, data: dict) -> tuple[paddle.Tensor, paddle.Tensor]:
        graph = data["graph"].tensor()
        node_feat = graph.node_feat
        edge_feat = graph.edge_feat

        atomic_numbers = node_feat["atom_types"].astype("int64")
        natoms = node_feat["num_atoms"].reshape([-1]).astype("int64")
        batch = graph.graph_node_id.astype("int64")

        # FindPointsInSpheres stores center->neighbor edges. UMA aggregates
        # neighbor->center, so the PGL edge direction is reversed here.
        edge_index = graph.edges.transpose([1, 0])
        edge_index = paddle.stack([edge_index[1], edge_index[0]], axis=0).astype(
            "int64"
        )
        edge_vectors = edge_feat["bond_vec"]
        edge_distances = edge_feat["bond_dist"].reshape([-1])

        wigner, wigner_inv = self._wigner(edge_vectors)
        envelope = self.envelope(edge_distances / self.cutoff).reshape([-1, 1, 1])
        wigner_inv = wigner_inv * envelope

        num_atoms = atomic_numbers.shape[0]
        x = paddle.zeros(
            [
                num_atoms,
                (self.lmax + 1) ** 2,
                self.sphere_channels,
            ],
            dtype=edge_vectors.dtype,
        )
        x = paddle.concat(
            [self.atom_embedding(atomic_numbers).unsqueeze(1), x[:, 1:, :]],
            axis=1,
        )

        edge_features = paddle.concat(
            [
                self.distance_expansion(edge_distances),
                self.source_embedding(atomic_numbers[edge_index[0]]),
                self.target_embedding(atomic_numbers[edge_index[1]]),
            ],
            axis=1,
        )
        x = self.edge_degree_embedding(x, edge_features, edge_index, wigner_inv)
        for block in self.blocks:
            x = block(
                x,
                edge_features,
                edge_index,
                wigner,
                wigner_inv,
            )
        x = self.norm(x)

        node_energy = self.energy_head(x[:, 0, :]).reshape([-1])
        energy = paddle.zeros([natoms.shape[0]], dtype=node_energy.dtype).index_add(
            axis=0, index=batch, value=node_energy
        )
        energy = energy.reshape([-1, 1])

        force_embedding = self.force_head(x[:, :4, :])
        forces = force_embedding[:, 1:4, 0]
        return energy, forces

    def forward(
        self,
        data: dict,
        return_loss: bool = True,
        return_prediction: bool = True,
    ) -> dict[str, dict[str, paddle.Tensor]]:
        if not return_loss and not return_prediction:
            raise ValueError(
                "At least one of return_loss and return_prediction must be True."
            )
        energy, forces = self._forward(data)
        predictions = {"energy": energy, "forces": forces}

        loss_dict: dict[str, paddle.Tensor] = {}
        if return_loss:
            graph = data["graph"].tensor()
            natoms = graph.node_feat["num_atoms"].reshape([-1]).astype(energy.dtype)
            batch = graph.graph_node_id.astype("int64")
            energy_label = data["energy"]
            if not paddle.is_tensor(energy_label):
                energy_label = paddle.to_tensor(energy_label)
            energy_label = energy_label.astype(energy.dtype).reshape(energy.shape)
            forces_label = data["forces"]
            if not paddle.is_tensor(forces_label):
                forces_label = paddle.to_tensor(forces_label)
            forces_label = forces_label.astype(forces.dtype).reshape(forces.shape)

            scale = natoms.reshape([-1, 1])
            energy_loss = paddle.nn.functional.l1_loss(
                energy / scale,
                energy_label / scale,
            )
            atom_force_loss = paddle.linalg.norm(forces - forces_label, axis=1)
            structure_force_loss = paddle.zeros(
                [natoms.shape[0]], dtype=atom_force_loss.dtype
            ).index_add(axis=0, index=batch, value=atom_force_loss)
            forces_loss = (structure_force_loss / natoms).mean()

            loss_dict = {
                "energy": energy_loss,
                "forces": forces_loss,
                "loss": energy_loss * self.loss_weights_dict["energy"]
                + forces_loss * self.loss_weights_dict["forces"],
            }

        pred_dict = predictions if return_prediction else {}
        return {"loss_dict": loss_dict, "pred_dict": pred_dict}

    def predict(self, graph) -> dict[str, np.ndarray]:
        output = self.forward(
            {"graph": graph},
            return_loss=False,
            return_prediction=True,
        )
        return {name: value.numpy() for name, value in output["pred_dict"].items()}

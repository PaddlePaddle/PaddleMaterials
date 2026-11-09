# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

"""Paddle port of the MACE-MP-0 ``ScaleShiftMACE`` model.

The module tree, the parameter / buffer names and the forward pass follow
``mace.modules.models.ScaleShiftMACE`` one to one, so the official converted
``state_dict`` loads without renaming.  Only the features used by the released
``mace_mp_0`` (medium) checkpoint are implemented: no pair repulsion, no joint
embeddings and no LAMMPS ghost handling.
"""

from __future__ import annotations

from typing import Dict
from typing import Optional
from typing import Sequence
from typing import Union

import numpy as np
import paddle
import paddle.nn as nn

from ppmat.models.common.e3nn import o3
from ppmat.models.mace.blocks import AtomicEnergiesBlock
from ppmat.models.mace.blocks import EquivariantProductBasisBlock
from ppmat.models.mace.blocks import LinearNodeEmbeddingBlock
from ppmat.models.mace.blocks import LinearReadoutBlock
from ppmat.models.mace.blocks import NonLinearReadoutBlock
from ppmat.models.mace.blocks import RealAgnosticResidualInteractionBlock
from ppmat.models.mace.blocks import ScaleShiftBlock
from ppmat.models.mace.blocks import silu
from ppmat.models.mace.radial import RadialEmbeddingBlock
from ppmat.models.mace.utils import get_outputs
from ppmat.models.mace.utils import prepare_graph
from ppmat.utils.scatter import scatter_sum

# MACE-MP-0 covers Z=1..83 and Z=89..94, i.e. exactly num_elements=89 entries.
SUPPORTED_ATOMIC_NUMBERS = list(range(1, 84)) + list(range(89, 95))

# 1 eV/Angstrom^3 = 160.21766208 GPa, the unit used by the PaddleMaterials
# stress labels (same as the CHGNet implementation).
EV_A3_TO_GPA = 160.21766208


def _as_tensor(value, dtype=None):
    """Convert graph features (numpy or tensor) to Paddle tensors."""
    if value is None:
        return None
    if isinstance(value, paddle.Tensor):
        return value if dtype is None else value.astype(dtype)
    return paddle.to_tensor(np.asarray(value), dtype=dtype)


class MACE(nn.Layer):
    """Paddle port of ``mace.modules.models.ScaleShiftMACE``.

    Exposes the ``forward`` / ``predict`` interface used by the
    PaddleMaterials Trainer and Predictor and returns ``energy_per_atom``,
    ``force`` and ``stress``.

    Args:
        r_max: Cutoff radius in Angstrom.
        num_bessel: Number of Bessel radial basis functions.
        num_polynomial_cutoff: Polynomial cutoff exponent.
        max_ell: Maximum spherical harmonic degree of the edge attributes.
        num_interactions: Number of message passing layers.
        num_elements: Number of atomic-number one-hot entries.
        hidden_irreps: Hidden node feature irreps.
        MLP_irreps: Irreps of the non-linear readout MLP.
        correlation: Correlation order of the symmetric contractions.
        radial_MLP: Hidden widths of the radial (convolution weight) MLP.
        avg_num_neighbors: Average number of neighbors used to normalize the
            aggregated messages.
        atomic_numbers: Atomic numbers of the ``num_elements`` one-hot
            entries, in the order used by the checkpoint.
        atomic_energies: Reference (isolated atom) energies, one per element.
        atomic_inter_scale: Scale of the ``ScaleShiftBlock``.
        atomic_inter_shift: Shift of the ``ScaleShiftBlock``.
        apply_cutoff: Fold the polynomial cutoff into the radial features.
        keep_last_layer_irreps: Keep the full hidden irreps in the last layer.
        use_last_readout_only: Only use the final (non-linear) readout.
    """

    def __init__(
        self,
        r_max: float = 6.0,
        num_bessel: int = 8,
        num_polynomial_cutoff: int = 5,
        max_ell: int = 3,
        num_interactions: int = 2,
        num_elements: int = 89,
        hidden_irreps: Union[str, o3.Irreps] = "128x0e",
        MLP_irreps: Union[str, o3.Irreps] = "16x0e",
        correlation: Union[int, Sequence[int]] = 3,
        radial_MLP: Optional[Sequence[int]] = None,
        avg_num_neighbors: float = 1.0,
        atomic_numbers: Optional[Sequence[int]] = None,
        atomic_energies=None,
        atomic_inter_scale: float = 1.0,
        atomic_inter_shift: float = 0.0,
        apply_cutoff: bool = True,
        keep_last_layer_irreps: bool = False,
        use_last_readout_only: bool = False,
        property_names: Optional[Sequence[str]] = None,
        loss_type: str = "smooth_l1_loss",
        huber_loss_delta: float = 0.1,
        loss_weights_dict: Optional[Dict[str, float]] = None,
        **kwargs,
    ):
        super().__init__()
        if kwargs:
            raise TypeError(
                f"Unexpected MACE arguments: {sorted(kwargs)}. The model only "
                "implements the architecture of the released mace_mp_0 "
                "checkpoint."
            )

        self.num_elements = num_elements
        self.avg_num_neighbors = float(avg_num_neighbors)

        support_property_names = ["energy_per_atom", "force", "stress"]
        support_loss_weights = {
            "energy_per_atom": 1.0,
            "force": 1.0,
            "stress": 0.1,
        }
        if property_names is None:
            property_names = support_property_names
        for name in property_names:
            assert name in support_property_names, (
                f"{name} is not supported; choose from: {support_property_names}"
            )
        self.property_names = list(property_names)
        if loss_weights_dict is None:
            loss_weights_dict = {
                key: support_loss_weights[key] for key in self.property_names
            }
        self.loss_weights_dict = loss_weights_dict

        if loss_type == "mse_loss":
            self.loss_fn = nn.MSELoss()
        elif loss_type in ("smooth_l1_loss", "huber_loss"):
            self.loss_fn = nn.SmoothL1Loss(delta=huber_loss_delta)
        elif loss_type == "l1_loss":
            self.loss_fn = nn.L1Loss()
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}")

        # Buffers shared with the official checkpoint.
        if atomic_numbers is None:
            atomic_numbers = SUPPORTED_ATOMIC_NUMBERS
        atomic_numbers = np.asarray(atomic_numbers, dtype=np.int64).reshape([-1])
        if atomic_numbers.shape[0] != num_elements:
            raise ValueError(
                f"atomic_numbers has {atomic_numbers.shape[0]} entries but "
                f"num_elements={num_elements}."
            )
        self.register_buffer(
            "atomic_numbers", paddle.to_tensor(atomic_numbers, dtype="int64")
        )
        self.register_buffer(
            "r_max",
            paddle.to_tensor(float(r_max), dtype=paddle.get_default_dtype()),
        )
        self.register_buffer(
            "num_interactions", paddle.to_tensor(int(num_interactions), dtype="int64")
        )

        hidden_irreps = o3.Irreps(hidden_irreps)
        MLP_irreps = o3.Irreps(MLP_irreps)
        if isinstance(correlation, int):
            correlation = [correlation] * num_interactions
        if len(correlation) != num_interactions:
            raise ValueError(
                f"correlation has {len(correlation)} entries but "
                f"num_interactions={num_interactions}."
            )

        # Embedding
        node_attr_irreps = o3.Irreps([(num_elements, (0, 1))])
        node_feats_irreps = o3.Irreps(
            [(hidden_irreps.count(o3.Irrep(0, 1)), (0, 1))]
        )
        self.node_embedding = LinearNodeEmbeddingBlock(
            irreps_in=node_attr_irreps,
            irreps_out=node_feats_irreps,
        )
        self.radial_embedding = RadialEmbeddingBlock(
            r_max=r_max,
            num_bessel=num_bessel,
            num_polynomial_cutoff=num_polynomial_cutoff,
            apply_cutoff=apply_cutoff,
        )
        edge_feats_irreps = o3.Irreps(f"{self.radial_embedding.out_dim}x0e")

        sh_irreps = o3.Irreps.spherical_harmonics(max_ell)
        num_features = hidden_irreps.count(o3.Irrep(0, 1))
        interaction_irreps = (sh_irreps * num_features).sort()[0].simplify()
        interaction_irreps_first = (sh_irreps * num_features).sort()[0].simplify()

        self.spherical_harmonics = o3.SphericalHarmonics(
            sh_irreps, normalize=True, normalization="component"
        )
        if radial_MLP is None:
            radial_MLP = [64, 64, 64]
        if atomic_energies is None:
            atomic_energies = np.zeros([num_elements], dtype=np.float64)
        self.atomic_energies_fn = AtomicEnergiesBlock(atomic_energies)

        # Interactions and readout
        if num_interactions == 1:
            hidden_irreps_out = str(hidden_irreps[0])
        else:
            hidden_irreps_out = hidden_irreps
        inter = RealAgnosticResidualInteractionBlock(
            node_attrs_irreps=node_attr_irreps,
            node_feats_irreps=node_feats_irreps,
            edge_attrs_irreps=sh_irreps,
            edge_feats_irreps=edge_feats_irreps,
            target_irreps=interaction_irreps_first,
            hidden_irreps=hidden_irreps_out,
            avg_num_neighbors=avg_num_neighbors,
            radial_MLP=radial_MLP,
        )
        self.interactions = nn.LayerList([inter])

        # The residual interaction block always provides the self connection at
        # the first layer (its name contains "Residual").
        node_feats_irreps_out = inter.target_irreps
        prod = EquivariantProductBasisBlock(
            node_feats_irreps=node_feats_irreps_out,
            target_irreps=hidden_irreps_out,
            correlation=correlation[0],
            num_elements=num_elements,
            use_sc=True,
        )
        self.products = nn.LayerList([prod])

        self.readouts = nn.LayerList()
        if not use_last_readout_only:
            self.readouts.append(LinearReadoutBlock(hidden_irreps_out, "1x0e"))

        for i in range(num_interactions - 1):
            if i == num_interactions - 2 and not keep_last_layer_irreps:
                # Select only scalars for the last layer.
                hidden_irreps_out = str(hidden_irreps[0])
            else:
                hidden_irreps_out = hidden_irreps
            inter = RealAgnosticResidualInteractionBlock(
                node_attrs_irreps=node_attr_irreps,
                node_feats_irreps=hidden_irreps,
                edge_attrs_irreps=sh_irreps,
                edge_feats_irreps=edge_feats_irreps,
                target_irreps=interaction_irreps,
                hidden_irreps=hidden_irreps_out,
                avg_num_neighbors=avg_num_neighbors,
                radial_MLP=radial_MLP,
            )
            self.interactions.append(inter)
            prod = EquivariantProductBasisBlock(
                node_feats_irreps=interaction_irreps,
                target_irreps=hidden_irreps_out,
                correlation=correlation[i + 1],
                num_elements=num_elements,
                use_sc=True,
            )
            self.products.append(prod)
            if i == num_interactions - 2:
                self.readouts.append(
                    NonLinearReadoutBlock(
                        irreps_in=hidden_irreps_out,
                        MLP_irreps=(MLP_irreps * 1).simplify(),
                        gate=silu,
                        irrep_out="1x0e",
                        num_heads=1,
                    )
                )
            elif not use_last_readout_only:
                self.readouts.append(LinearReadoutBlock(hidden_irreps, "1x0e"))

        self.scale_shift = ScaleShiftBlock(
            scale=atomic_inter_scale, shift=atomic_inter_shift
        )

    def _graph_data(self, graph, dtype):
        """Map a PGL crystal graph to the tensors ``prepare_graph`` expects."""
        atom_types = (
            _as_tensor(graph.node_feat["atom_types"]).reshape([-1]).astype("int64")
        )
        positions = _as_tensor(graph.node_feat["cart_coords"], dtype=dtype)
        num_atoms = _as_tensor(graph.node_feat["num_atoms"]).reshape([-1])
        num_atoms = num_atoms.astype(dtype)
        batch = _as_tensor(graph.graph_node_id).reshape([-1]).astype("int64")

        lattice = graph.node_feat.get("lattice", None)
        lattice = _as_tensor(lattice, dtype=dtype)
        if lattice is not None:
            lattice = lattice.reshape([-1, 3, 3])

        # pgl stores edges as [n_edges, 2] with columns (sender, receiver).
        edges = _as_tensor(graph.edges).astype("int64")
        edge_index = paddle.transpose(edges, [1, 0])

        pbc_offset = graph.edge_feat.get("pbc_offset", None)
        if pbc_offset is None:
            unit_shifts = paddle.zeros([edge_index.shape[1], 3], dtype=dtype)
        else:
            unit_shifts = _as_tensor(pbc_offset, dtype=dtype).reshape([-1, 3])

        return {
            "atom_types": atom_types,
            "positions": positions,
            "num_atoms": num_atoms,
            "batch": batch,
            "cell": lattice,
            "edge_index": edge_index,
            "unit_shifts": unit_shifts,
            "num_graphs": int(num_atoms.shape[0]),
        }

    def _node_attrs(self, atom_types, dtype):
        """One-hot atomic-number descriptors ``[n_nodes, num_elements]``."""
        node_attrs = paddle.cast(
            atom_types.reshape([-1, 1]) == self.atomic_numbers.reshape([1, -1]),
            dtype,
        )
        # Every atom must match exactly one supported entry; otherwise the
        # one-hot row would be all zeros and silently change the prediction.
        if not bool(paddle.all(paddle.sum(node_attrs, axis=1) > 0).numpy()):
            unsupported = sorted(
                set(np.asarray(atom_types.numpy()).tolist())
                - set(np.asarray(self.atomic_numbers.numpy()).tolist())
            )
            raise ValueError(
                f"Atomic numbers {unsupported} are not supported by this MACE "
                "checkpoint (supported: Z=1-83 and Z=89-94)."
            )
        return node_attrs

    def _forward(self, data):
        """Compute standardized predictions from a batched PGL graph."""
        # Trainer passes {"graph": batched_graph, ...labels}; Predictor passes
        # the pgl.Graph directly.
        graph = data["graph"] if isinstance(data, dict) else data
        dtype = self.node_embedding.linear.weight.dtype

        need_force = "force" in self.property_names
        need_stress = "stress" in self.property_names

        graph_data = self._graph_data(graph, dtype)
        num_atoms = graph_data.pop("num_atoms")
        atom_types = graph_data.pop("atom_types")
        node_attrs = self._node_attrs(atom_types, dtype)

        num_atoms_arange = paddle.arange(atom_types.shape[0], dtype="int64")
        node_heads = paddle.zeros([atom_types.shape[0]], dtype="int64")

        ctx = prepare_graph(
            {
                **graph_data,
                "node_heads": node_heads,
            },
            compute_virials=need_stress,
            compute_stress=need_stress,
        )
        positions = ctx["positions"]
        vectors = ctx["vectors"]
        lengths = ctx["lengths"]
        cell = ctx["cell"]
        num_graphs = ctx["num_graphs"]

        # Atomic (reference) energies, one head per element.
        node_e0 = paddle.gather_nd(
            self.atomic_energies_fn(node_attrs),
            paddle.stack([num_atoms_arange, node_heads], axis=1),
        )
        e0 = scatter_sum(
            src=node_e0, index=graph_data["batch"], dim=0, dim_size=num_graphs
        ).astype(vectors.dtype)

        # Embeddings
        node_feats = self.node_embedding(node_attrs)
        edge_attrs = self.spherical_harmonics(vectors)
        edge_feats, cutoff = self.radial_embedding(
            lengths, node_attrs, graph_data["edge_index"], self.atomic_numbers
        )
        pair_node_energy = paddle.zeros_like(node_e0)

        # Interactions
        node_es_list = [pair_node_energy]
        node_feats_list = []
        for interaction, product in zip(self.interactions, self.products):
            node_feats, sc = interaction(
                node_attrs=node_attrs,
                node_feats=node_feats,
                edge_attrs=edge_attrs,
                edge_feats=edge_feats,
                edge_index=graph_data["edge_index"],
                cutoff=cutoff,
            )
            node_feats = product(node_feats=node_feats, sc=sc, node_attrs=node_attrs)
            node_feats_list.append(node_feats)

        for i, readout in enumerate(self.readouts):
            feat_idx = -1 if len(self.readouts) == 1 else i
            node_es_list.append(
                paddle.gather_nd(
                    readout(node_feats_list[feat_idx], node_heads),
                    paddle.stack([num_atoms_arange, node_heads], axis=1),
                )
            )

        node_inter_es = paddle.sum(paddle.stack(node_es_list, axis=0), axis=0)
        node_inter_es = self.scale_shift(node_inter_es, node_heads)
        inter_e = scatter_sum(
            src=node_inter_es,
            index=graph_data["batch"],
            dim=-1,
            dim_size=num_graphs,
        )

        total_energy = e0 + inter_e

        forces, _, stress = get_outputs(
            energy=inter_e,
            positions=positions,
            displacement=ctx["displacement"],
            vectors=vectors,
            cell=cell,
            training=self.training,
            compute_force=need_force,
            compute_virials=need_stress,
            compute_stress=need_stress,
        )
        if stress is not None:
            stress = stress * EV_A3_TO_GPA

        energy_per_atom = total_energy / paddle.clip(num_atoms, min=1.0)

        pred_dict = {}
        if "energy_per_atom" in self.property_names:
            pred_dict["energy_per_atom"] = energy_per_atom
        if "force" in self.property_names and forces is not None:
            pred_dict["force"] = forces
        if "stress" in self.property_names and stress is not None:
            pred_dict["stress"] = stress
        return pred_dict

    def forward(self, data, return_loss=True, return_prediction=True):
        """Return standardized predictions and optional training losses."""
        assert return_loss or return_prediction

        pred_dict = self._forward(data)

        loss_dict = {}
        if return_loss:
            if not isinstance(data, dict):
                raise TypeError(
                    "Training requires a dict with a 'graph' entry and property "
                    f"labels, but got {type(data)}."
                )
            loss = paddle.to_tensor(0.0, dtype="float32")
            for property_name, pred in pred_dict.items():
                if property_name not in data:
                    continue
                label = data[property_name]
                if label is None:
                    continue
                # Flatten and drop NaNs; labels from the collator may still be numpy.
                pred_flat = pred.reshape([-1])
                label_flat = _as_tensor(label, dtype=pred_flat.dtype).reshape([-1])
                valid = ~paddle.isnan(label_flat)
                if int(valid.astype("int64").sum()) == 0:
                    continue
                loss_property = self.loss_fn(pred_flat[valid], label_flat[valid])
                loss_dict[property_name] = loss_property
                loss = loss + loss_property * self.loss_weights_dict[property_name]
            loss_dict["loss"] = loss

        prediction = pred_dict if return_prediction else {}
        return {"loss_dict": loss_dict, "pred_dict": prediction}

    def predict(self, data):
        """Return standardized Paddle tensor predictions for Predictor."""
        return self.forward(
            data,
            return_loss=False,
            return_prediction=True,
        )["pred_dict"]
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

"""FlowMol: flow matching for 3D de novo small-molecule generation.

Reference: https://arxiv.org/abs/2508.12629

The model owns the numerical runtime (see ``CTMCVectorField``) and follows the
``forward``/``sample`` contracts of PaddleMaterials: the training forward
returns ``{"loss_dict": ...}`` and sampling returns ``{"pred": [...]}``
consumed by ``MolecularSampler``.
"""

from __future__ import annotations

import copy
import os
import os.path as osp

import numpy as np
import paddle
import paddle.nn as nn

from ppmat.models.common.runtime import RuntimeMixin
from ppmat.models.flowmol.graph_utils import build_batched_complete_graph
from ppmat.models.flowmol.vector_field import CTMCVectorField
from ppmat.models.flowmol.vector_field import centered_normal_prior_batched
from ppmat.models.flowmol.vector_field import ctmc_masked_prior
from ppmat.schedulers import build_scheduler
from ppmat.utils import logger

FEATURE_NAMES = ["positions", "atom_types", "atom_charges", "bond_types"]
# the mask token is appended after the real atom types and decodes to "Se"
MASK_ATOM_SYMBOL = "Se"


class FlowMol(RuntimeMixin, nn.Layer):
    """Unconditional 3D molecule generation with mixed CTMC flow matching."""

    def __init__(
        self,
        atom_type_map: list,
        total_loss_weights: dict,
        interpolant_scheduler_config: dict,
        vector_field_config: dict,
        prior_config: dict,
        time_scaled_loss: bool = False,
        fake_atom_p: float = 0.0,
        distort_p: float = 0.0,
        distort_t: float = 0.5,
        default_n_timesteps: int = 250,
        execution_backend: str = "eager",
        runtime_options: dict = None,
    ):
        super().__init__()

        self.atom_type_map = list(atom_type_map)
        self.fake_atom_p = fake_atom_p
        self.fake_atoms = fake_atom_p > 0
        self.n_atom_types = len(self.atom_type_map) + int(self.fake_atoms)
        self.time_scaled_loss = time_scaled_loss
        self.distort_p = distort_p
        self.distort_t = distort_t
        self.default_n_timesteps = default_n_timesteps
        self.prior_config = prior_config
        self.feature_names = FEATURE_NAMES

        for feat in self.feature_names:
            if feat not in total_loss_weights:
                raise ValueError(
                    f"total_loss_weights must provide a weight for feature {feat}"
                )
        self.total_loss_weights = total_loss_weights

        if prior_config["positions"]["type"] != "centered-normal":
            raise ValueError(
                "prior_config['positions']['type'] must be 'centered-normal', got "
                f"{prior_config['positions']['type']}"
            )
        for feat in ["atom_types", "atom_charges", "bond_types"]:
            if prior_config[feat]["type"] != "ctmc":
                raise ValueError(
                    f"prior_config['{feat}']['type'] must be 'ctmc', got "
                    f"{prior_config[feat]['type']}"
                )

        scheduler_cfg = copy.deepcopy(interpolant_scheduler_config)
        scheduler_cfg["__init_params__"]["feature_names"] = self.feature_names
        self.interpolant_scheduler = build_scheduler(scheduler_cfg)

        if time_scaled_loss:
            positions_reduction = "none"
        else:
            positions_reduction = "mean"
        self.loss_fn_positions = nn.MSELoss(reduction=positions_reduction)
        # normalize manually: paddle's 'mean' divides by total nodes while
        # torch divides by non-ignored nodes
        self.ignore_index = -100

        self.vector_field = CTMCVectorField(
            n_atom_types=self.n_atom_types,
            feature_names=self.feature_names,
            interpolant_scheduler=self.interpolant_scheduler,
            execution_backend=execution_backend,
            runtime_options=runtime_options,
            **vector_field_config,
        )

    @property
    def execution_backend(self):
        return self.vector_field.execution_backend

    @property
    def _runtime_cache(self):
        """Read-only view of the runtime cache owned by the vector field.

        The numerical boundary lives on ``CTMCVectorField``; exposing its
        cache keeps the top-level model contract-compatible with tooling
        that reads compiled runtime state from the workflow model.
        """
        return self.vector_field._runtime_cache

    def set_execution_backend(self, backend):
        self.vector_field.set_execution_backend(backend)

    def set_runtime_options(self, runtime_options):
        self.vector_field.set_runtime_options(runtime_options)

    def validate_execution_backend(self, *, use_amp=False, world_size=1):
        self.vector_field.validate_execution_backend(
            use_amp=use_amp, world_size=world_size
        )

    def _build_targets(self, batch: dict, state: dict, upper_edge_mask) -> dict:
        """Build the loss targets; already unmasked states are ignored.

        Nodes whose ``x_t`` token is no longer the mask token keep no
        supervision signal: their feature is already revealed along the CTMC
        path, so the cross-entropy is restricted to still-masked nodes via
        ``ignore_index=-100``.
        """
        targets = {"positions": batch["positions"]}
        for feat in ["atom_types", "atom_charges", "bond_types"]:
            target = batch[feat].argmax(-1)
            if feat == "bond_types":
                target = target[upper_edge_mask]
            xt_idxs = state[feat].argmax(-1)
            if feat == "bond_types":
                xt_idxs = xt_idxs[upper_edge_mask]
            n_cat = self.vector_field.n_cat_feats[feat]
            ignore = paddle.full_like(target, -100)
            target = paddle.where(xt_idxs != n_cat, ignore, target)
            targets[feat] = target
        return targets

    def _build_graph_fields(self, num_atoms: np.ndarray, place=None) -> dict:
        """Build the complete-graph bookkeeping shared by training and sampling."""
        graph = build_batched_complete_graph(num_atoms)
        return {
            "batch_size": len(num_atoms),
            "src": paddle.to_tensor(graph["src"], dtype="int64", place=place),
            "dst": paddle.to_tensor(graph["dst"], dtype="int64", place=place),
            "upper_edge_mask": paddle.to_tensor(graph["upper_edge_mask"], place=place),
            "node_batch_idx": paddle.to_tensor(
                graph["node_batch_idx"], dtype="int64", place=place
            ),
            "edge_batch_idx": paddle.to_tensor(
                graph["edge_batch_idx"], dtype="int64", place=place
            ),
        }

    def _prepare_batch(self, batch: dict) -> dict:
        """Convert the collated numpy batch into tensors and graph fields."""
        structure = batch["structure_array"]
        num_atoms = np.asarray(structure["num_atoms"], dtype="int64")
        prepared = self._build_graph_fields(num_atoms)
        prepared.update(
            {
                "positions": paddle.to_tensor(structure["positions"]),
                "atom_types": paddle.to_tensor(structure["atom_types"]),
                "atom_charges": paddle.to_tensor(structure["atom_charges"]),
                "bond_types": paddle.to_tensor(structure["bond_types"]),
                "prior_positions": paddle.to_tensor(batch["prior_positions"]),
            }
        )
        return prepared

    def forward(self, batch: dict) -> dict:
        """Compute the flow matching losses of a training batch.

        The ``pred_dict`` is not produced because the model predicts trajectory
        endpoints rather than per-attribute labels, so ``compute_metric_func_dict``
        does not apply to this model.
        """
        batch = self._prepare_batch(batch)
        batch_size = batch["batch_size"]
        node_batch_idx = batch["node_batch_idx"]
        edge_batch_idx = batch["edge_batch_idx"]
        upper_edge_mask = batch["upper_edge_mask"]

        t = paddle.rand([batch_size])
        prior_positions = batch["prior_positions"]

        state = self.vector_field.sample_conditional_path(
            batch,
            t,
            prior_positions,
            node_batch_idx,
            edge_batch_idx,
            upper_edge_mask,
        )

        if self.distort_p > 0.0:
            t_mask = (t > self.distort_t)[node_batch_idx]
            num_nodes = state["positions"].shape[0]
            distort_mask = paddle.rand([num_nodes, 1]) < self.distort_p
            distort_mask = distort_mask & t_mask.unsqueeze(-1)
            # 0.5 displacement std follows the upstream PyTorch implementation
            state["positions"] = (
                state["positions"]
                + paddle.randn(state["positions"].shape)
                * distort_mask.astype("float32")
                * 0.5
            )

        output = self.vector_field(
            state,
            t,
            node_batch_idx=node_batch_idx,
            upper_edge_mask=upper_edge_mask,
        )

        targets = self._build_targets(batch, state, upper_edge_mask)

        if self.time_scaled_loss:
            time_weights = self.interpolant_scheduler.loss_weights(t)

        losses = {}
        for feat_idx, feat in enumerate(self.feature_names):
            if feat == "positions":
                loss_per_feat = self.loss_fn_positions(output[feat], targets[feat])
                if self.time_scaled_loss:
                    weight = time_weights[:, feat_idx][node_batch_idx]
                    losses[feat] = (loss_per_feat * weight.unsqueeze(-1)).mean()
                else:
                    losses[feat] = loss_per_feat
                continue

            loss_per_feat = paddle.nn.functional.cross_entropy(
                output[feat],
                targets[feat],
                ignore_index=self.ignore_index,
                reduction="none",
            )
            if self.time_scaled_loss:
                weight = time_weights[:, feat_idx]
                if feat == "bond_types":
                    weight = weight[edge_batch_idx][upper_edge_mask]
                else:
                    weight = weight[node_batch_idx]
                losses[feat] = (loss_per_feat * weight).mean()
            else:
                valid = (targets[feat] != self.ignore_index).astype("float32")
                losses[feat] = loss_per_feat.sum() / valid.sum().clip(min=1.0)

        total_loss = None
        for feat in self.feature_names:
            term = self.total_loss_weights[feat] * losses[feat]
            total_loss = term if total_loss is None else total_loss + term

        loss_dict = {"loss": total_loss}
        for feat in self.feature_names:
            loss_dict[f"loss_{feat}"] = losses[feat]
        return {"loss_dict": loss_dict}

    def _build_sampling_state(self, num_atoms: np.ndarray, place) -> dict:
        num_atoms_np = np.asarray(num_atoms, dtype="int64")

        state = self._build_graph_fields(num_atoms_np, place=place)

        num_nodes = int(state["node_batch_idx"].shape[0])
        num_edges = int(state["upper_edge_mask"].shape[0])
        prior_kwargs = self.prior_config["positions"]["kwargs"]
        state["positions"] = centered_normal_prior_batched(
            num_nodes, state["node_batch_idx"], **prior_kwargs
        ).to(place)
        state["atom_types"] = ctmc_masked_prior(
            num_nodes, self.vector_field.n_cat_feats["atom_types"]
        ).to(place)
        state["atom_charges"] = ctmc_masked_prior(
            num_nodes, self.vector_field.n_cat_feats["atom_charges"]
        ).to(place)
        state["bond_types"] = ctmc_masked_prior(
            num_edges, self.vector_field.n_cat_feats["bond_types"]
        ).to(place)
        return state

    @paddle.no_grad()
    def sample(
        self,
        batch_data: dict,
        num_inference_steps: int = None,
        stochasticity: float = None,
        high_confidence_threshold: float = None,
    ) -> dict:
        """Sample molecules with the atom counts given in ``batch_data``.

        Args:
            batch_data (dict): A batch containing ``structure_array.num_atoms``.
            num_inference_steps (int): Number of integration steps; uses the
                configured ``default_n_timesteps`` when None.
            stochasticity (float): CTMC stochasticity; uses the configured value
                when None.
            high_confidence_threshold (float): Purity sampling threshold; uses
                the configured value when None.
        """
        if num_inference_steps is None:
            num_inference_steps = self.default_n_timesteps

        num_atoms = np.asarray(batch_data["structure_array"]["num_atoms"])
        place = next(iter(self.parameters())).place

        state = self._build_sampling_state(num_atoms, place)
        state = self.vector_field.integrate(
            state,
            state["node_batch_idx"],
            state["upper_edge_mask"],
            n_timesteps=num_inference_steps,
            stochasticity=stochasticity,
            high_confidence_threshold=high_confidence_threshold,
        )

        return {"pred": self._decode_samples(state, num_atoms)}

    def _decode_samples(self, state: dict, num_atoms: np.ndarray) -> list:
        positions = state["positions_1"].numpy()
        atom_tokens = state["atom_types_1"].argmax(-1).numpy()
        charges = (
            state["atom_charges_1"].argmax(-1).numpy() - self.vector_field.charge_offset
        )
        src = state["src"].numpy()[state["upper_edge_mask"].numpy()]
        dst = state["dst"].numpy()[state["upper_edge_mask"].numpy()]
        bond_types = (
            state["bond_types_1"].argmax(-1).numpy()[state["upper_edge_mask"].numpy()]
        )

        n_real_atom_types = len(self.atom_type_map)
        n_cat_atom = self.vector_field.n_cat_feats["atom_types"]
        fake_token_idx = n_cat_atom - 1 if self.fake_atoms else None

        results = []
        node_offset = 0
        edge_offset = 0
        for n in num_atoms.astype("int64"):
            n = int(n)
            n_edges = n * (n - 1) // 2
            node_slice = slice(node_offset, node_offset + n)
            edge_slice = slice(edge_offset, edge_offset + n_edges)

            molecule_tokens = atom_tokens[node_slice]
            if self.fake_atoms:
                keep_mask = molecule_tokens != fake_token_idx
            else:
                keep_mask = np.ones(n, dtype=bool)
            real_idxs = np.where(keep_mask)[0]
            old_to_new = -np.ones(n, dtype="int64")
            old_to_new[real_idxs] = np.arange(len(real_idxs))

            molecule_positions = positions[node_slice][real_idxs]
            molecule_charges = charges[node_slice][real_idxs]
            molecule_symbols = [
                self.atom_type_map[int(idx)]
                if idx < n_real_atom_types
                else MASK_ATOM_SYMBOL
                for idx in molecule_tokens[real_idxs]
            ]

            molecule_src = src[edge_slice]
            molecule_dst = dst[edge_slice]
            molecule_bonds = bond_types[edge_slice]
            n_bond_cats = self.vector_field.n_cat_feats["bond_types"]
            keep = (
                (molecule_bonds != 0)
                & (molecule_bonds < n_bond_cats)
                & keep_mask[molecule_src - node_offset]
                & keep_mask[molecule_dst - node_offset]
            )
            molecule_src = old_to_new[molecule_src[keep] - node_offset]
            molecule_dst = old_to_new[molecule_dst[keep] - node_offset]

            results.append(
                {
                    "num_atoms": len(real_idxs),
                    "atom_types": molecule_symbols,
                    "positions": molecule_positions.tolist(),
                    "charges": molecule_charges.tolist(),
                    "bonds": [
                        [int(s), int(d), int(b)]
                        for s, d, b in zip(
                            molecule_src, molecule_dst, molecule_bonds[keep]
                        )
                    ],
                }
            )
            node_offset += n
            edge_offset += n_edges

        return results

    def save_samples(self, molecules: list, output_dir: str) -> None:
        """Persist sampled molecules into an SDF file under ``output_dir``."""
        from rdkit import Chem

        os.makedirs(output_dir, exist_ok=True)
        sdf_path = osp.join(output_dir, "flowmol_samples.sdf")
        writer = Chem.SDWriter(sdf_path)
        writer.SetKekulize(False)
        n_written = 0
        for molecule in molecules:
            rdkit_mol = build_rdkit_mol(molecule)
            if rdkit_mol is not None:
                writer.write(rdkit_mol)
                n_written += 1
        writer.close()
        logger.info(f"Saved {n_written} molecules to {sdf_path}")


def build_rdkit_mol(molecule: dict):
    from rdkit import Chem
    from rdkit.Geometry import Point3D

    bond_type_map = [
        None,
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
    ]

    mol = Chem.RWMol()
    for symbol, charge in zip(molecule["atom_types"], molecule["charges"]):
        atom = Chem.Atom(symbol)
        if charge != 0:
            atom.SetFormalCharge(int(charge))
        mol.AddAtom(atom)

    for src_idx, dst_idx, bond_type in molecule["bonds"]:
        mol.AddBond(int(src_idx), int(dst_idx), bond_type_map[int(bond_type)])

    mol = mol.GetMol()

    conformer = Chem.Conformer(mol.GetNumAtoms())
    for i, (x, y, z) in enumerate(molecule["positions"]):
        conformer.SetAtomPosition(i, Point3D(float(x), float(y), float(z)))
    mol.AddConformer(conformer)

    return mol

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

"""OMATG-specific CSPNet with time embedding and dual output heads.

OMatG uses Stochastic Interpolants (continuous alpha/beta/gamma schedules),
not diffusion noise schedulers, so ppmat.schedulers (DDPMScheduler etc.) do
not apply.
"""

from typing import Dict
from typing import Optional

import paddle
import paddle.nn as nn

from ppmat.losses import MSELoss
from ppmat.models.common.runtime import RuntimeMixin
from ppmat.models.common.runtime import runtime_boundary
from ppmat.models.diffcsp.diffcsp import CSPNet
from ppmat.models.omatg.si.core import BIG_TIME
from ppmat.models.omatg.si.core import DEFAULT_MAX_ATOMS
from ppmat.models.omatg.si.core import SMALL_TIME
from ppmat.models.omatg.si.core import StochasticInterpolantSpecies
from ppmat.models.omatg.si.core import build_si_from_cfg
from ppmat.models.omatg.si.core import correct_for_minimum_permutation_distance
from ppmat.utils.crystal import frac_to_cart_coords_with_lattice
from ppmat.utils.crystal import lattices_to_params_shape_paddle
from ppmat.utils.crystal import radius_graph_pbc
from ppmat.utils.misc import repeat_blocks

__all__ = [
    "OMATGCSPNet",
    "OMATGCSPNetFull",
    "IndependentSampler",
    "build_sampler_from_cfg",
]

# Lattice log-length mean/std per dataset (upstream FlowMM lattice_params_stats).
LATTICE_PARAMS = {
    "carbon_24": {
        "means": [0.9852757453918457, 1.3865314722061157, 1.7068126201629639],
        "stds": [0.14957907795906067, 0.20431114733219147, 0.2403733879327774],
    },
    "mp_20": {
        "means": [1.575442910194397, 1.7017393112182617, 1.9781638383865356],
        "stds": [0.24437622725963593, 0.26526379585266113, 0.3535512685775757],
    },
    "mpts_52": {
        "means": [1.6565313339233398, 1.8407557010650635, 2.1225264072418213],
        "stds": [0.2952289581298828, 0.3340013027191162, 0.41885802149772644],
    },
    "perov_5": {
        "means": [1.419227957725525, 1.419227957725525, 1.419227957725525],
        "stds": [0.07268335670232773, 0.07268335670232773, 0.07268335670232773],
    },
    "alex_mp_20": {
        "means": [1.5808929163076058, 1.74672046352959, 2.065243388307474],
        "stds": [0.27284015410437057, 0.2944785731740152, 0.30899526911753017],
    },
}

_SAMPLER_CFG_KEYS = {
    "dataset_name",
    "lattice_means",
    "lattice_stds",
    "mirror_species",
    "mask_species",
    "max_atoms",
}


def sample_lattice_cell(lattice_means, lattice_stds):
    """Sample a cell from log-normal lengths and uniform angles."""
    from ase.geometry.cell import cellpar_to_cell

    lengths = paddle.exp(
        paddle.randn([3]) * paddle.to_tensor(lattice_stds)
        + paddle.to_tensor(lattice_means)
    )
    angles = paddle.rand([3]) * 60.0 + 60.0
    return cellpar_to_cell(paddle.concat((lengths, angles)).numpy())


def _normalize_sample_dict(data: dict) -> dict:
    """Normalize collated sample dict keys to n_atoms/species/cell/pos/batch/ptr."""
    out = {
        "n_atoms": data.get("n_atoms", data.get("num_atoms")),
        "species": data.get("species", data.get("atom_types")),
        "cell": data.get("cell", data.get("lattices")),
        "pos": data.get("pos", data.get("frac_coords")),
    }
    batch = data.get("batch", data.get("node2graph"))
    n_atoms = out["n_atoms"]
    if n_atoms.ndim > 1:
        n_atoms = n_atoms.squeeze(-1)
        out["n_atoms"] = n_atoms
    if out["cell"] is not None and out["cell"].ndim == 2:
        out["cell"] = out["cell"].unsqueeze(0)
    pos_is_fractional = data.get("pos_is_fractional")
    if pos_is_fractional is not None and pos_is_fractional.ndim > 1:
        pos_is_fractional = pos_is_fractional.squeeze(-1)
    out["pos_is_fractional"] = (
        pos_is_fractional
        if pos_is_fractional is not None
        else paddle.ones_like(n_atoms, dtype="bool")
    )
    if batch is None:
        batch = paddle.repeat_interleave(
            paddle.arange(n_atoms.shape[0], dtype="int64"), n_atoms
        )
    out["batch"] = batch
    out["ptr"] = paddle.concat(
        [
            paddle.to_tensor([0], dtype="int64"),
            paddle.cumsum(n_atoms, axis=0).cast("int64"),
        ]
    )
    return out


def _validate_relative_si_costs(relative_si_costs, loss_keys) -> dict:
    """Validate the SI loss weights against the keys the interpolants produce.

    ``relative_si_costs`` must weight every loss exactly once and sum to one, so a
    typo or a missing entry fails at build time instead of silently reweighting or
    dropping a loss term.
    """
    if relative_si_costs is None:
        raise ValueError(
            "si_scheduler_cfg requires 'relative_si_costs' mapping every loss key "
            f"to its weight; expected keys are {sorted(loss_keys)}."
        )
    if not all(cost >= 0.0 for cost in relative_si_costs.values()):
        raise ValueError("All relative_si_costs weights must be non-negative.")

    total = sum(relative_si_costs.values())
    if abs(total - 1.0) >= 1e-10:
        raise ValueError(f"relative_si_costs must sum to 1.0, got {total!r}.")
    if set(relative_si_costs) != set(loss_keys):
        raise ValueError(
            "relative_si_costs keys must match the SI loss keys exactly; got "
            f"{sorted(relative_si_costs)} while the interpolants produce "
            f"{sorted(loss_keys)}."
        )
    return dict(relative_si_costs)


class OMATGCSPNet(CSPNet):
    """OMATG-specific CSPNet extending diffcsp backbone."""

    def __init__(
        self,
        hidden_dim=128,
        num_layers=4,
        max_atoms=DEFAULT_MAX_ATOMS,
        act_fn="silu",
        dis_emb="sin",
        num_freqs=10,
        edge_style="fc",
        cutoff=6.0,
        max_neighbors=20,
        ln=False,
        ip=True,
        smooth=False,
        pred_type=False,
        pred_scalar=False,
        time_embed_dim=None,
    ):
        if pred_scalar:
            raise ValueError(
                "OMatG does not support pred_scalar: it always trains the "
                "dual b/eta heads required by stochastic interpolants."
            )
        super().__init__(
            hidden_dim=hidden_dim,
            latent_dim=time_embed_dim if time_embed_dim is not None else 1,
            num_layers=num_layers,
            act_fn=act_fn,
            dis_emb=dis_emb,
            num_freqs=num_freqs,
            edge_style=edge_style,
            ln=ln,
            ip=ip,
            smooth=smooth,
            pred_type=pred_type,
            prop_dim=hidden_dim,
            pred_scalar=pred_scalar,
            num_classes=max_atoms,
        )
        # OMatG has no property embedding; drop prop_mlp from CSPNet.
        for i in range(num_layers):
            getattr(self, "csp_layer_%d" % i).prop_mlp = None

        self.hidden_dim = hidden_dim
        self.cutoff = cutoff
        self.max_neighbors = max_neighbors
        self.species_shift = 1

        self.time_embed_dim = time_embed_dim
        if time_embed_dim is not None:
            from ppmat.models.common.time_embedding import SinusoidalTimeEmbeddings

            self.time_embedder = SinusoidalTimeEmbeddings(time_embed_dim)
            self.atom_latent_emb = nn.Linear(hidden_dim + time_embed_dim, hidden_dim)
        else:
            self.time_embedder = None
            self.atom_latent_emb = nn.Linear(hidden_dim + 1, hidden_dim)

        self.dis_dim = (num_freqs * 2 * 3) if dis_emb == "sin" else 0

        self.coord_out_2 = nn.Linear(hidden_dim, 3, bias_attr=False)
        self.lattice_out_2 = nn.Linear(hidden_dim, 9, bias_attr=False)
        if self.pred_type:
            self.type_out_2 = nn.Linear(hidden_dim, max_atoms)

    def enable_masked_species(self):
        self.node_embedding = nn.Embedding(self.num_classes + 1, self.hidden_dim)
        self.species_shift = 0

    def reorder_symmetric_edges(self, edge_index, cell_offsets, neighbors, edge_vector):
        mask_sep_atoms = edge_index[0] < edge_index[1]
        cell_earlier = (
            (cell_offsets[:, 0] < 0)
            | ((cell_offsets[:, 0] == 0) & (cell_offsets[:, 1] < 0))
            | (
                (cell_offsets[:, 0] == 0)
                & (cell_offsets[:, 1] == 0)
                & (cell_offsets[:, 2] < 0)
            )
        )
        mask_same_atoms = edge_index[0] == edge_index[1]
        mask_same_atoms = mask_same_atoms & cell_earlier
        mask = mask_sep_atoms | mask_same_atoms
        edge_index_new = edge_index[mask[None, :].expand([2, -1])].reshape([2, -1])
        edge_index_cat = paddle.concat(
            [
                edge_index_new,
                paddle.stack([edge_index_new[1], edge_index_new[0]], axis=0),
            ],
            axis=1,
        )
        batch_edge = paddle.repeat_interleave(
            paddle.arange(neighbors.shape[0]), neighbors
        )
        batch_edge = batch_edge[mask]
        neighbors_new = 2 * paddle.bincount(batch_edge, minlength=neighbors.shape[0])
        edge_reorder_idx = repeat_blocks(
            neighbors_new // 2,
            repeats=2,
            continuous_indexing=True,
            repeat_inc=edge_index_new.shape[1],
        )
        edge_index_new = edge_index_cat[:, edge_reorder_idx]
        cell_offsets_new = CSPNet.select_symmetric_edges(
            self, cell_offsets, mask, edge_reorder_idx, True
        )
        edge_vector_new = CSPNet.select_symmetric_edges(
            self, edge_vector, mask, edge_reorder_idx, True
        )
        return edge_index_new, cell_offsets_new, neighbors_new, edge_vector_new

    def gen_edges(self, num_atoms, frac_coords, lattices, node2graph):
        if self.edge_style == "fc":
            return CSPNet.gen_edges(self, num_atoms, frac_coords)

        cart_coords = frac_to_cart_coords_with_lattice(frac_coords, num_atoms, lattices)
        edge_index, to_jimages, num_bonds = radius_graph_pbc(
            cart_coords,
            lattices,
            num_atoms,
            self.cutoff,
            self.max_neighbors,
            num_atoms.place,
        )
        j_index, i_index = edge_index[0], edge_index[1]
        distance_vectors = frac_coords[j_index] - frac_coords[i_index]
        distance_vectors = distance_vectors + to_jimages
        edge_index_new, _, _, edge_vector_new = self.reorder_symmetric_edges(
            edge_index, to_jimages, num_bonds, distance_vectors
        )
        return edge_index_new, -edge_vector_new

    def forward_with_edges(
        self,
        t,
        atom_types,
        frac_coords,
        lattices,
        num_atoms,
        node2graph,
        edges,
        frac_diff,
    ):
        edge2graph = node2graph[edges[0]]

        if self.smooth:
            node_features = self.node_embedding(atom_types.cast("float32"))
        else:
            node_features = self.node_embedding(atom_types - self.species_shift)

        if t.ndim == 0:
            t = t.unsqueeze(0)

        if self.time_embed_dim is not None:
            t_embed = self.time_embedder(t)
        else:
            t_embed = t

        t_per_atom = paddle.repeat_interleave(t_embed, num_atoms, axis=0)
        node_features = paddle.concat([node_features, t_per_atom], axis=1)
        node_features = self.atom_latent_emb(node_features)

        for i in range(self.num_layers):
            layer = getattr(self, "csp_layer_%d" % i)
            node_features = layer(
                node_features,
                frac_coords,
                lattices,
                edges,
                edge2graph,
                frac_diff=frac_diff,
            )

        if self.ln:
            node_features = self.final_layer_norm(node_features)

        coord_out = self.coord_out(node_features)
        coord_out_2 = self.coord_out_2(node_features)

        graph_features = paddle.geometric.segment_mean(node_features, node2graph)

        lattice_out = self.lattice_out(graph_features)
        lattice_out = lattice_out.reshape([-1, 3, 3])
        if self.ip:
            lattice_out = paddle.matmul(lattice_out, lattices)

        lattice_out_2 = self.lattice_out_2(graph_features)
        lattice_out_2 = lattice_out_2.reshape([-1, 3, 3])
        if self.ip:
            lattice_out_2 = paddle.matmul(lattice_out_2, lattices)

        if self.pred_type:
            type_out = self.type_out(node_features)
            type_out_2 = self.type_out_2(node_features)
            return (
                lattice_out,
                coord_out,
                type_out,
                lattice_out_2,
                coord_out_2,
                type_out_2,
            )

        return lattice_out, coord_out, lattice_out_2, coord_out_2

    def forward_dict(self, t, atom_types, frac_coords, lattices, num_atoms, node2graph):
        edges, frac_diff = self.gen_edges(num_atoms, frac_coords, lattices, node2graph)
        return self._runtime_forward_dict(
            t,
            atom_types,
            frac_coords,
            lattices,
            num_atoms,
            node2graph,
            edges,
            frac_diff,
        )

    @runtime_boundary("denoise_step")
    def _runtime_forward_dict(
        self,
        t,
        atom_types,
        frac_coords,
        lattices,
        num_atoms,
        node2graph,
        edges,
        frac_diff,
    ):
        preds = OMATGCSPNet.forward_with_edges(
            self,
            t,
            atom_types,
            frac_coords,
            lattices,
            num_atoms,
            node2graph,
            edges,
            frac_diff,
        )
        if self.pred_type:
            return {
                "pos_b": preds[1],
                "pos_eta": preds[4],
                "cell_b": preds[0],
                "cell_eta": preds[3],
                "species_b": preds[2],
                "species_eta": preds[5],
            }
        return {
            "pos_b": preds[1],
            "pos_eta": preds[3],
            "cell_b": preds[0],
            "cell_eta": preds[2],
        }


class OMATGCSPNetFull(RuntimeMixin, OMATGCSPNet):
    """CSPNet with time embedding, checkpoint-compatible defaults."""

    # Defaults match the released checkpoints so a bare OMATGCSPNetFull()
    # can load pretrained weights; build_omatg_model() overrides from these.
    def __init__(
        self,
        hidden_dim: int = 512,
        num_layers: int = 6,
        max_atoms: int = DEFAULT_MAX_ATOMS,
        act_fn: str = "silu",
        dis_emb: str = "sin",
        num_freqs: int = 128,
        edge_style: str = "fc",
        cutoff: float = 7.0,
        max_neighbors: int = 20,
        ln: bool = True,
        ip: bool = True,
        smooth: bool = False,
        pred_type: bool = False,
        pred_scalar: bool = False,
        time_embed_dim: int = 256,
        use_si: bool = False,
        si_scheduler_cfg: dict = None,
        sampler_cfg: dict = None,
        use_min_perm_dist: bool = False,
        execution_backend: str = "eager",
        runtime_options: dict = None,
    ):
        super().__init__(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            max_atoms=max_atoms,
            act_fn=act_fn,
            dis_emb=dis_emb,
            num_freqs=num_freqs,
            edge_style=edge_style,
            cutoff=cutoff,
            max_neighbors=max_neighbors,
            ln=ln,
            ip=ip,
            smooth=smooth,
            pred_type=pred_type,
            pred_scalar=pred_scalar,
            time_embed_dim=time_embed_dim,
        )
        self._init_runtime(execution_backend, runtime_options)
        self.use_si = use_si
        self.use_min_perm_dist = use_min_perm_dist
        self._si = None
        self._sampler = None
        self._relative_si_costs = {}
        self._min_perm_dist_corrector = None
        self.mse_loss = MSELoss()
        if use_min_perm_dist and not use_si:
            raise ValueError(
                "use_min_perm_dist requires use_si=True: the correction reuses the "
                "position corrector of the stochastic interpolants."
            )
        if use_si:
            self._build_si(si_scheduler_cfg or {}, sampler_cfg or {})

    def forward(self, data, t=None):
        """Forward pass on a sample dict; returns loss_dict.

        OMatG training forward does not emit per-attribute predictions aligned
        with labels, so compute_metric_func_dict does not apply; evaluation is
        loss-based (eval_loss).
        """
        atom_types = data.get("atom_types", data.get("species"))
        frac_coords = data.get("frac_coords", data.get("pos"))
        lattices = data.get("lattices", data.get("cell"))
        num_atoms = data.get("num_atoms", data.get("n_atoms"))
        node2graph = data.get("node2graph", data.get("batch"))

        if self.use_si:
            if self._si is None or self._sampler is None:
                raise ValueError(
                    "use_si=True requires si_cfg and sampler_cfg to be provided "
                    "and valid (got si_cfg that built no stochastic interpolants)."
                )
            return self._si_forward(data)

        if t is None:
            batch_size = lattices.shape[0]
            t = paddle.rand([batch_size])

        predictions = self.forward_dict(
            t=t,
            atom_types=atom_types,
            frac_coords=frac_coords,
            lattices=lattices,
            num_atoms=num_atoms,
            node2graph=node2graph,
        )

        lattices_gt = lattices
        frac_coords_gt = frac_coords

        if self.pred_type:
            loss_lattice = self.mse_loss(predictions["cell_b"], lattices_gt)
            loss_coord = self.mse_loss(predictions["pos_b"], frac_coords_gt)
            loss_type = paddle.nn.functional.cross_entropy(
                input=predictions["species_b"], label=atom_types - 1
            )
            loss = loss_lattice + loss_coord + loss_type
            return {
                "loss_dict": {
                    "loss": loss,
                    "loss_lattice": loss_lattice,
                    "loss_coord": loss_coord,
                    "loss_type": loss_type,
                }
            }

        loss_lattice = self.mse_loss(predictions["cell_b"], lattices_gt)
        loss_coord = self.mse_loss(predictions["pos_b"], frac_coords_gt)
        loss = loss_lattice + loss_coord
        return {
            "loss_dict": {
                "loss": loss,
                "loss_lattice": loss_lattice,
                "loss_coord": loss_coord,
            }
        }

    def sample(self, batch_data, num_inference_steps=100, **kwargs):
        """Sample via SI integration; requires use_si=True (see configs)."""
        if not self.use_si:
            raise ValueError(
                "OMATGCSPNetFull.sample() requires the StochasticInterpolants "
                "sampling path (use_si=True). Build the model with an "
                "si_cfg/sampler_cfg, e.g. load one of the "
                "structure_generation/configs/omatg/*.yaml configs."
            )
        if self._si is None or self._sampler is None:
            raise ValueError(
                "use_si=True requires si_cfg and sampler_cfg to be provided "
                "and valid (got si_cfg that built no stochastic interpolants)."
            )
        return self._si_sample(batch_data, num_inference_steps, **kwargs)

    def _make_model_function(self):
        def model_function(x_t, time):
            return self.forward_dict(
                t=time,
                atom_types=x_t["species"],
                frac_coords=x_t["pos"],
                lattices=x_t["cell"],
                num_atoms=x_t["n_atoms"],
                node2graph=x_t["batch"],
            )

        return model_function

    @staticmethod
    def _build_sample_result(num_atoms_list, atom_types, frac_coords, lattices):
        lengths, angles = lattices_to_params_shape_paddle(lattices)
        start_idx = 0
        result = []
        for i, n in enumerate(num_atoms_list):
            end_idx = start_idx + n
            result.append(
                {
                    "num_atoms": n,
                    "atom_types": atom_types[start_idx:end_idx].tolist(),
                    "frac_coords": frac_coords[start_idx:end_idx].tolist(),
                    "lengths": lengths[i].tolist(),
                    "angles": angles[i].tolist(),
                }
            )
            start_idx += n
        return {"result": result}

    def _si_sample(self, batch_data, num_inference_steps, **kwargs):
        x_1 = self._data_to_omatg(batch_data)
        x_0 = self._sampler.sample_p_0(x_1)

        gen = self._si.integrate(
            x_0,
            self._make_model_function(),
            save_intermediate=False,
            integration_time_steps=num_inference_steps,
        )

        return self._build_sample_result(
            gen["n_atoms"].tolist(), gen["species"], gen["pos"], gen["cell"]
        )

    def _build_si(self, si_scheduler_cfg: dict, sampler_cfg: dict) -> None:
        # Normalize an OmegaConf DictConfig (direct build_omatg_model use) to a dict.
        if not isinstance(si_scheduler_cfg, dict):
            from omegaconf import OmegaConf

            si_scheduler_cfg = OmegaConf.to_container(si_scheduler_cfg, resolve=True)

        if not si_scheduler_cfg:
            raise ValueError(
                "use_si=True requires a non-empty si_scheduler_cfg mapping data "
                "fields to stochastic-interpolant configs (e.g. "
                "'si_scheduler_cfg: {species: {...}, pos: {...}, cell: {...}, "
                "integration_time_steps: 210}')."
            )

        self._si = build_si_from_cfg(si_scheduler_cfg)
        self._relative_si_costs = _validate_relative_si_costs(
            si_scheduler_cfg.get("relative_si_costs"), self._si.loss_keys()
        )

        # Masked-species schemes (DNG) need an extra embedding token (species 0).
        species_interpolant = self._si.get_stochastic_interpolant("species")
        if not isinstance(species_interpolant, StochasticInterpolantSpecies):
            raise ValueError(
                "The 'species' data field must be a StochasticInterpolantSpecies "
                f"interpolant, got {type(species_interpolant).__name__}."
            )
        if species_interpolant.uses_masked_species():
            self.enable_masked_species()

        if sampler_cfg:
            self._sampler = build_sampler_from_cfg(sampler_cfg)
            # Bind the parent model so a sampler without an explicit max_atoms
            # resolves the atomic-number range from the model itself.
            if isinstance(self._sampler, IndependentSampler):
                self._sampler.bind_model(self)

        if self.use_min_perm_dist:
            self._min_perm_dist_corrector = self._si.get_stochastic_interpolant(
                "pos"
            ).get_corrector()

    def _data_to_omatg(self, data):
        if isinstance(data, dict) and "structure_array" in data:
            # StructureSampler input. Without atom_types the species are drawn
            # uniformly from the model's atomic-number range (a CSP model then
            # mirrors them); use --mode by_chemical_formula to fix the species.
            sa = data["structure_array"]
            num_atoms = paddle.to_tensor(sa["num_atoms"], dtype="int64")
            if "atom_types" in sa:
                species = paddle.to_tensor(sa["atom_types"], dtype="int64")
            else:
                total = int(num_atoms.sum().item())
                species = paddle.randint(
                    1, self.num_classes + 1, [total], dtype="int64"
                )
            data = {
                "n_atoms": num_atoms,
                "species": species,
                "cell": paddle.eye(3).unsqueeze(0).tile([len(num_atoms), 1, 1]),
                "pos": paddle.zeros([int(num_atoms.sum().item()), 3]),
                "pos_is_fractional": paddle.ones([len(num_atoms)], dtype="bool"),
            }
        # DefaultCollator returns numpy arrays; convert to tensors.
        data = {
            k: (paddle.to_tensor(v) if not paddle.is_tensor(v) else v)
            for k, v in data.items()
            if v is not None
        }
        sample = _normalize_sample_dict(data)
        if not bool(paddle.all(sample["pos_is_fractional"])):
            raise ValueError(
                "OMatG evolves fractional coordinates, but the batch carries "
                "Cartesian positions; keep convert_to_fractional=True on the dataset."
            )
        return sample

    def _si_forward(self, data: dict) -> dict:
        """SI velocity-matching loss step."""
        x_1 = self._data_to_omatg(data)
        x_0 = self._sampler.sample_p_0(x_1)
        if self._min_perm_dist_corrector is not None:
            correct_for_minimum_permutation_distance(
                x_0, x_1, self._min_perm_dist_corrector
            )
        batch_size = len(x_1["n_atoms"])
        t = paddle.rand([batch_size]) * (BIG_TIME - SMALL_TIME) + SMALL_TIME

        losses = self._si.losses(self._make_model_function(), t, x_0, x_1)
        total_loss = paddle.to_tensor(0.0)
        loss_dict = {}
        for key, val in losses.items():
            weighted = self._relative_si_costs[key] * val
            loss_dict[key] = weighted
            total_loss = total_loss + weighted
        loss_dict["loss"] = total_loss
        return {"loss_dict": loss_dict}


class IndependentSampler:
    """Sample SI base distributions (lattice / species).

    Internal SI x_0 sampler; bridged to the public StructureSampler via
    model.sample() in sample.py.

    ``max_atoms`` defaults to ``None`` and is resolved at first use against
    the bound ``OMATGCSPNet`` model via :meth:`bind_model`; if used standalone
    without a model and no explicit value, falls back to ``DEFAULT_MAX_ATOMS``
    (the released-checkpoint default).
    """

    def __init__(
        self,
        dataset_name=None,
        lattice_means=None,
        lattice_stds=None,
        mirror_species=True,
        mask_species=False,
        max_atoms: Optional[int] = None,
    ):
        if (lattice_means is None) != (lattice_stds is None):
            raise ValueError(
                "lattice_means and lattice_stds must be provided together."
            )
        if lattice_means is None:
            if dataset_name is None:
                raise ValueError(
                    "dataset_name must be provided (or lattice_means/lattice_stds "
                    "injected explicitly) to sample the lattice base distribution."
                )
            params = LATTICE_PARAMS.get(dataset_name)
            if params is None:
                raise ValueError(
                    f"Unknown dataset_name '{dataset_name}' for lattice base "
                    f"distribution. Provide lattice_means/lattice_stds explicitly "
                    f"or use one of: {sorted(LATTICE_PARAMS)}."
                )
            lattice_means, lattice_stds = params["means"], params["stds"]
        self._lattice_means = list(lattice_means)
        self._lattice_stds = list(lattice_stds)
        self._mirror_species = mirror_species
        self._mask_species = mask_species
        self._max_atoms = max_atoms
        self._bound_model: Optional["OMATGCSPNetFull"] = None

    def bind_model(self, model: "OMATGCSPNetFull") -> None:
        """Bind the parent OMatG model so that the species range resolves to it.

        When ``max_atoms`` was not provided explicitly, the bound model's
        ``num_classes`` (the atomic-number cardinality of the released
        checkpoints) becomes the inclusive upper bound of the uniform species
        distribution. An explicit constructor ``max_atoms`` is not overridden.
        """
        self._bound_model = model
        if self._max_atoms is None:
            self._max_atoms = int(model.num_classes)

    def _resolve_max_atoms(self) -> int:
        if self._max_atoms is None and self._bound_model is not None:
            self._max_atoms = int(self._bound_model.num_classes)
        if self._max_atoms is None:
            return DEFAULT_MAX_ATOMS
        return int(self._max_atoms)

    def sample_p_0(self, x_1: Dict[str, paddle.Tensor]) -> Dict[str, paddle.Tensor]:
        batch_size = len(x_1["n_atoms"])
        ptr = (
            x_1["ptr"]
            if "ptr" in x_1
            else paddle.concat(
                [
                    paddle.to_tensor([0], dtype="int64"),
                    paddle.cumsum(x_1["n_atoms"], axis=0).cast("int64"),
                ]
            )
        )
        n_atoms_list = []
        species_list = []
        pos_list = []
        cell_list = []
        batch_indices = []
        for i in range(batch_size):
            sl = slice(int(ptr[i]), int(ptr[i + 1]))
            pos = paddle.rand(x_1["pos"][sl].shape, dtype=x_1["pos"].dtype)
            cell = paddle.to_tensor(
                sample_lattice_cell(self._lattice_means, self._lattice_stds),
                dtype=x_1["cell"].dtype,
            )
            if self._mask_species:
                species = paddle.zeros_like(x_1["species"][sl])
            elif self._mirror_species:
                species = x_1["species"][sl].clone()
            else:
                species = paddle.randint(
                    1,
                    self._resolve_max_atoms() + 1,
                    x_1["species"][sl].shape,
                    dtype=x_1["species"][sl].dtype,
                )
            n = len(species)
            n_atoms_list.append(paddle.to_tensor([n], dtype="int64"))
            species_list.append(species)
            pos_list.append(pos)
            cell_list.append(cell.unsqueeze(0))
            batch_indices.append(paddle.full([n], i, dtype="int64"))

        n_atoms = paddle.concat(n_atoms_list)
        return {
            "n_atoms": n_atoms,
            "species": paddle.concat(species_list),
            "cell": paddle.concat(cell_list),
            "pos": paddle.concat(pos_list),
            "pos_is_fractional": paddle.ones([batch_size], dtype="bool"),
            "batch": paddle.concat(batch_indices),
            "ptr": paddle.concat(
                [
                    paddle.to_tensor([0], dtype="int64"),
                    paddle.cumsum(n_atoms, axis=0).cast("int64"),
                ]
            ),
        }


def build_sampler_from_cfg(sampler_cfg: dict):
    """Build an IndependentSampler from a config dict.

    Unknown keys are rejected so that an unsupported or misspelled sampler option
    fails loudly instead of being silently ignored.
    """
    unknown_keys = sorted(set(sampler_cfg) - _SAMPLER_CFG_KEYS)
    if unknown_keys:
        raise ValueError(
            f"sampler_cfg has unknown keys {unknown_keys}; allowed keys are "
            f"{sorted(_SAMPLER_CFG_KEYS)}."
        )
    return IndependentSampler(
        dataset_name=sampler_cfg.get("dataset_name"),
        lattice_means=sampler_cfg.get("lattice_means"),
        lattice_stds=sampler_cfg.get("lattice_stds"),
        mirror_species=sampler_cfg.get("mirror_species", True),
        mask_species=sampler_cfg.get("mask_species", False),
        max_atoms=sampler_cfg.get("max_atoms"),
    )

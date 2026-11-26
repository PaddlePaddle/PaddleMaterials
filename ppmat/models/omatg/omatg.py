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

"""OMatG: Open Materials Generation with Stochastic Interpolants.

https://openreview.net/forum?id=gHGrzxFujU

The model evolves fractional coordinates, lattice matrices and atomic species
with stochastic interpolants (SIs) instead of diffusion noise schedulers. Each
data field owns its own interpolant and integrator, configured through
``si_scheduler_cfg``. The CSP variant keeps the input composition fixed, while
the DNG variant masks the species and draws them during integration.
"""

import math
from typing import Dict

import paddle
import paddle.nn as nn

from ppmat.models.common.activation import swish
from ppmat.models.common.runtime import RuntimeMixin
from ppmat.models.common.runtime import runtime_boundary
from ppmat.models.diffcsp.diffcsp import CSPNet
from ppmat.models.omatg.si.core import BIG_TIME
from ppmat.models.omatg.si.core import DEFAULT_MAX_ATOMS
from ppmat.models.omatg.si.core import SMALL_TIME
from ppmat.models.omatg.si.core import StochasticInterpolantSpecies
from ppmat.models.omatg.si.core import build_si_from_cfg
from ppmat.models.omatg.si.core import correct_for_minimum_permutation_distance
from ppmat.utils.crystal import lattice_params_to_matrix_paddle
from ppmat.utils.scatter import scatter_mean

__all__ = [
    "OMatG",
    "IndependentSampler",
    "build_sampler_from_cfg",
]

# Base-distribution lattice statistics used by the released checkpoints. They
# are only a fallback: sampler_cfg may pass lattice_means/lattice_stds instead.
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


def _as_tensor(value, dtype):
    if paddle.is_tensor(value):
        return value.cast(dtype)
    return paddle.to_tensor(value, dtype=dtype)


def _validate_relative_si_costs(relative_si_costs, loss_keys) -> dict:
    """Check that the SI loss weights cover every interpolant loss exactly once."""
    if relative_si_costs is None:
        raise ValueError(
            "si_scheduler_cfg requires 'relative_si_costs' mapping every loss key "
            f"to its weight; expected keys are {sorted(loss_keys)}."
        )
    if not all(
        math.isfinite(cost) and cost >= 0.0 for cost in relative_si_costs.values()
    ):
        raise ValueError(
            "All relative_si_costs weights must be finite and non-negative."
        )

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


class _UnfusedSiLU(nn.Silu):
    def forward(self, x):
        return swish(x)


class OMATGCSPNet(CSPNet):
    """CSPNet backbone with the OMatG time embedding and dual velocity heads."""

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
        if smooth:
            raise ValueError(
                "OMatG expects integer species; smooth=True is unsupported."
            )
        if act_fn != "silu":
            raise ValueError("OMatG supports act_fn='silu' only.")
        if edge_style != "fc":
            raise ValueError("OMatG supports edge_style='fc' only.")
        if time_embed_dim is not None and (time_embed_dim < 4 or time_embed_dim % 2):
            raise ValueError("time_embed_dim must be an even integer of at least four.")
        if num_layers < 1 or hidden_dim < 1 or max_atoms < 1:
            raise ValueError("Model dimensions and num_layers must be positive.")
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
        # OMatG has no property embedding; drop prop_mlp from CSPNet and use the
        # unfused Swish expression for the two-layer MLPs.
        self.act_fn = _UnfusedSiLU()
        for i in range(num_layers):
            layer = getattr(self, "csp_layer_%d" % i)
            layer.prop_mlp = None
            for mlp in (layer.edge_mlp, layer.node_mlp):
                mlp[1] = self.act_fn
                mlp[3] = self.act_fn

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

        self.coord_out_2 = nn.Linear(hidden_dim, 3, bias_attr=False)
        self.lattice_out_2 = nn.Linear(hidden_dim, 9, bias_attr=False)
        if not self.pred_type:
            self.type_out = nn.Linear(hidden_dim, max_atoms)
        self.type_out_2 = nn.Linear(hidden_dim, max_atoms)

    def enable_masked_species(self):
        if self.species_shift == 0:
            return
        self.node_embedding = nn.Embedding(self.num_classes + 1, self.hidden_dim)
        self.species_shift = 0

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
        property_emb=None,
        property_mask=None,
    ):
        edge2graph = node2graph[edges[0]]

        node_features = self.node_embedding(atom_types - self.species_shift)

        if t.ndim == 0:
            t = t.unsqueeze(0)

        if self.time_embed_dim is not None:
            t_embed = self.time_embedder(t)
        else:
            t_embed = t.reshape([-1, 1])

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
                num_atoms=num_atoms,
                property_emb=property_emb,
                property_mask=property_mask,
            )

        if self.ln:
            node_features = self.final_layer_norm(node_features)

        coord_out = self.coord_out(node_features)
        coord_out_2 = self.coord_out_2(node_features)

        graph_features = scatter_mean(
            node_features, node2graph, dim=0, dim_size=lattices.shape[0]
        )

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
        edges, frac_diff = self.gen_edges(num_atoms, frac_coords)
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


class OMatG(RuntimeMixin, OMATGCSPNet):
    """Open Materials Generation with Stochastic Interpolants.

    Args:
        decoder_cfg (dict): Backbone configuration, see :class:`OMATGCSPNet`.
        si_scheduler_cfg (dict): Per-field stochastic-interpolant configuration.
            It must provide ``species``, ``pos`` and ``cell`` entries plus
            ``integration_time_steps`` and ``relative_si_costs``.
        sampler_cfg (dict): Base-distribution sampler configuration.
        use_min_perm_dist (bool): Permute the base positions to minimise the
            periodic distance to the target before building the interpolants.
        execution_backend (str): Runtime backend used by the shared runtime mixin.
        runtime_options (dict, optional): Backend-specific runtime options.
    """

    supports_num_atoms_sampling = False

    def __init__(
        self,
        decoder_cfg: dict,
        si_scheduler_cfg: dict,
        sampler_cfg: dict,
        use_min_perm_dist: bool = False,
        execution_backend: str = "eager",
        runtime_options: dict = None,
    ) -> None:
        if not sampler_cfg:
            raise ValueError(
                "OMatG requires sampler_cfg describing the base distribution "
                "(dataset_name, or explicit lattice_means/lattice_stds)."
            )
        super().__init__(**decoder_cfg)
        self._init_runtime(execution_backend, runtime_options)

        self.use_min_perm_dist = use_min_perm_dist
        self._si = None
        self._sampler = None
        self._relative_si_costs = {}
        self._min_perm_dist_corrector = None

        self._build_si(si_scheduler_cfg, sampler_cfg)
        self._freeze_unused_heads()

    def _freeze_unused_heads(self):
        heads = [self.type_out_2]
        if not self.pred_type:
            heads.append(self.type_out)
        for field, head in (
            ("pos", self.coord_out_2),
            ("cell", self.lattice_out_2),
        ):
            if "loss_z" not in self._si.get_stochastic_interpolant(field).loss_keys():
                heads.append(head)
        for head in heads:
            for parameter in head.parameters():
                parameter.stop_gradient = True

    def forward(self, batch, **kwargs):
        x_1 = self._to_sample_dict(batch)
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

    @paddle.no_grad()
    def sample(self, batch_data, num_inference_steps=None, **kwargs):
        x_1 = self._to_sample_dict(batch_data)
        x_0 = self._sampler.sample_p_0(x_1)

        gen = self._si.integrate(
            x_0,
            self._make_model_function(),
            integration_time_steps=num_inference_steps,
        )

        return self._build_sample_result(
            gen["n_atoms"].tolist(), gen["species"], gen["pos"], gen["cell"]
        )

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
        start_idx = 0
        result = []
        for i, n in enumerate(num_atoms_list):
            end_idx = start_idx + n
            result.append(
                {
                    "num_atoms": n,
                    "atom_types": atom_types[start_idx:end_idx].tolist(),
                    "frac_coords": frac_coords[start_idx:end_idx].tolist(),
                    "lattice": lattices[i].tolist(),
                }
            )
            start_idx += n
        return {"result": result}

    def _build_si(self, si_scheduler_cfg: dict, sampler_cfg: dict) -> None:
        self._si = build_si_from_cfg(si_scheduler_cfg)
        for field in ("species", "pos", "cell"):
            self._si.get_stochastic_interpolant(field)
        self._relative_si_costs = _validate_relative_si_costs(
            si_scheduler_cfg.get("relative_si_costs"), self._si.loss_keys()
        )

        species_interpolant = self._si.get_stochastic_interpolant("species")
        if not isinstance(species_interpolant, StochasticInterpolantSpecies):
            raise ValueError(
                "The 'species' data field must be a StochasticInterpolantSpecies "
                f"interpolant, got {type(species_interpolant).__name__}."
            )
        masked_species = species_interpolant.uses_masked_species()
        if masked_species:
            if not self.pred_type or not sampler_cfg.get("mask_species", False):
                raise ValueError(
                    "Masked-species interpolation requires pred_type=True and "
                    "sampler_cfg.mask_species=True."
                )
            self.enable_masked_species()
            self.supports_num_atoms_sampling = True
        elif self.pred_type or sampler_cfg.get("mask_species", False):
            raise ValueError(
                "Fixed-species interpolation requires pred_type=False and "
                "sampler_cfg.mask_species=False."
            )

        active_costs = dict(self._relative_si_costs)
        if not self.pred_type:
            active_costs.pop("species_loss", None)
        if not any(cost > 0.0 for cost in active_costs.values()):
            raise ValueError("At least one model-dependent SI loss must have weight.")

        self._sampler = build_sampler_from_cfg(sampler_cfg)

        if self.use_min_perm_dist:
            self._min_perm_dist_corrector = self._si.get_stochastic_interpolant(
                "pos"
            ).get_corrector()

    def _to_sample_dict(self, batch):
        """Convert a PPMAT ``structure_array`` batch to the internal sample dict."""
        structure_array = batch["structure_array"]

        num_atoms = _as_tensor(structure_array["num_atoms"], "int64").reshape([-1])
        total = int(num_atoms.sum().item())

        if "atom_types" in structure_array:
            species = _as_tensor(structure_array["atom_types"], "int64").reshape([-1])
        else:
            # Sample requests carry no composition for DNG; the masked-species
            # interpolant replaces these placeholders with the mask token anyway.
            species = paddle.zeros([total], dtype="int64")

        if "lattice" in structure_array:
            cell = _as_tensor(structure_array["lattice"], "float32").reshape([-1, 3, 3])
        elif "lengths" in structure_array and "angles" in structure_array:
            cell = lattice_params_to_matrix_paddle(
                _as_tensor(structure_array["lengths"], "float32"),
                _as_tensor(structure_array["angles"], "float32"),
            )
        else:
            cell = paddle.eye(3).unsqueeze(0).tile([len(num_atoms), 1, 1])

        if "frac_coords" in structure_array:
            pos = _as_tensor(structure_array["frac_coords"], "float32").reshape([-1, 3])
        else:
            pos = paddle.zeros([total, 3])

        batch_idx = paddle.repeat_interleave(
            paddle.arange(len(num_atoms), dtype="int64"), num_atoms
        )
        return {
            "n_atoms": num_atoms,
            "species": species,
            "cell": cell,
            "pos": pos,
            "batch": batch_idx,
            "ptr": paddle.concat(
                [
                    paddle.to_tensor([0], dtype="int64"),
                    paddle.cumsum(num_atoms, axis=0).cast("int64"),
                ]
            ),
        }


class IndependentSampler:
    """Sample SI base distributions (lattice / species).

    Masked-species interpolation starts from the mask token, fixed-species
    interpolation mirrors the target species.
    """

    def __init__(
        self,
        dataset_name=None,
        lattice_means=None,
        lattice_stds=None,
        mask_species=False,
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
        self._mask_species = mask_species

    def sample_p_0(self, x_1: Dict[str, paddle.Tensor]) -> Dict[str, paddle.Tensor]:
        batch_size = len(x_1["n_atoms"])
        ptr = x_1["ptr"]
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
            else:
                species = x_1["species"][sl].clone()
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
            "batch": paddle.concat(batch_indices),
            "ptr": paddle.concat(
                [
                    paddle.to_tensor([0], dtype="int64"),
                    paddle.cumsum(n_atoms, axis=0).cast("int64"),
                ]
            ),
        }


def build_sampler_from_cfg(sampler_cfg: dict):
    """Build an IndependentSampler from a config dict."""
    unknown_keys = sorted(set(sampler_cfg) - _SAMPLER_CFG_KEYS)
    if unknown_keys:
        raise ValueError(
            f"sampler_cfg has unknown keys {unknown_keys}; allowed keys are "
            f"{sorted(_SAMPLER_CFG_KEYS)}."
        )
    mask_species = sampler_cfg.get("mask_species", False)
    if not mask_species and not sampler_cfg.get("mirror_species", True):
        raise ValueError(
            "Fixed-species sampling requires sampler_cfg.mirror_species=True."
        )
    return IndependentSampler(
        dataset_name=sampler_cfg.get("dataset_name"),
        lattice_means=sampler_cfg.get("lattice_means"),
        lattice_stds=sampler_cfg.get("lattice_stds"),
        mask_species=mask_species,
    )

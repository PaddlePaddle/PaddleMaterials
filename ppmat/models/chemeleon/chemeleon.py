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

# Adapted from https://github.com/hspark1212/chemeleon

"""Chemeleon: text-conditioned diffusion model for crystal structure generation.

Three diffusion processes share one CSPNet decoder pass: an absorbing-state
discrete diffusion (D3PM) over atom types, a variance-preserving diffusion
over the lattice matrix, and a variance-exploding diffusion over fractional
coordinates. Text conditions enter the decoder through a FiLM layer.

Interface notes:
  - plain paddle.nn.Layer; the training loop lives in the ppmat trainer.
  - forward takes a batch dict with structure_array (flat concatenated
    frac_coords [B_n, 3] / atom_types [B_n] / lattices [B, 3, 3] / natoms
    [B], e.g. from MP40Dataset + DefaultCollator) and a text list, and
    returns {"loss_dict": {scalar losses}, "monitor_dict": {tensors}}.
  - the text encoder is injected at construction time (any module exposing
    get_text_embeds(batch_text, cond_drop_prob), e.g. TextEncoder +
    MinimalBert); None builds an unguided model.
  - execution_backend="eager" (default) passes the denoise_step boundary
    straight through to the decoder; compiled backends are selected via
    execution_backend / runtime_options at construction time.
"""

from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

import paddle
import paddle.nn.functional as F
from tqdm import tqdm

from ppmat.models.chemeleon.cspnet import DECODER_OUTPUTS
from ppmat.models.chemeleon.cspnet import CSPNet
from ppmat.models.chemeleon.cspnet import SinusoidalTimeEmbeddings
from ppmat.models.chemeleon.diff_utils import D3PM
from ppmat.models.chemeleon.diff_utils import BetaScheduler
from ppmat.models.chemeleon.diff_utils import SigmaScheduler
from ppmat.models.chemeleon.diff_utils import d_log_p_wrapped_normal
from ppmat.models.chemeleon.minimal_bert import VOCAB_PATH
from ppmat.models.chemeleon.minimal_bert import BertWordPieceTokenizer
from ppmat.models.chemeleon.minimal_bert import MinimalBert
from ppmat.models.chemeleon.schema import TrajectoryContainer
from ppmat.models.chemeleon.schema import TrajectoryStep
from ppmat.models.chemeleon.text_encoder import TextEncoder
from ppmat.models.common.runtime import RuntimeMixin
from ppmat.models.common.runtime import runtime_boundary


def _build_text_encoder_from_cfg(cfg: Dict[str, Any]):
    """Assemble the bundled text condition encoder from a config dict.

    The default text front-end couples the bundled WordPiece tokenizer and
    MinimalBert transformer with the TextEncoder wrapper; a clip_dim entry
    adds the contrastive projection stage on top of the class-token
    embedding. trainable_text_encoder=False freezes the transformer
    backbone (the text_emb MLP and the null embedding stay trainable).
    """
    text_embed_dim = cfg.get("text_embed_dim", 768)
    clip_dim = cfg.get("clip_dim")
    text_proj = None
    if clip_dim is not None:
        text_proj = paddle.nn.Sequential(
            paddle.nn.Linear(text_embed_dim, text_embed_dim),
            paddle.nn.LayerNorm(text_embed_dim),
            paddle.nn.GELU(),
            paddle.nn.Linear(text_embed_dim, clip_dim),
        )
    text_transformer = MinimalBert()
    if not cfg.get("trainable_text_encoder", False):
        for param in text_transformer.parameters():
            param.stop_gradient = True
    return TextEncoder(
        text_embed_dim=text_embed_dim,
        max_text_len=cfg.get("max_text_len", 256),
        text_dim=cfg.get("text_dim", 512),
        tokenizer=BertWordPieceTokenizer(str(VOCAB_PATH), do_lower_case=False),
        text_transformer=text_transformer,
        text_proj=text_proj,
    )


class Chemeleon(RuntimeMixin, paddle.nn.Layer):
    def __init__(
        self,
        _config: Dict[str, Any],
        text_encoder=None,
        execution_backend: str = "eager",
        runtime_options: Optional[dict] = None,
        text_encoder_cfg: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        super().__init__()
        # execution-runtime dispatch; eager default = pass-through
        self._init_runtime(execution_backend, runtime_options)
        # time embedding
        self.time_embed = SinusoidalTimeEmbeddings(_config["time_dim"])
        # text embedding (injected module; None => unguided model)
        self.text_guide = bool(_config.get("text_guide", text_encoder is not None))
        if self.text_guide and text_encoder is None and text_encoder_cfg is not None:
            text_encoder = _build_text_encoder_from_cfg(text_encoder_cfg)
        if self.text_guide and text_encoder is None:
            raise ValueError(
                "text_guide=True requires a text_encoder module at construction."
            )
        self.text_encoder = text_encoder
        # ppmat sampler condition protocol (StructureSampler by_condition)
        self.condition_names = ["text"] if self.text_guide else []
        if self.text_guide:
            self.cond_drop_prob = _config["cond_drop_prob"]
        # set scheduler
        self.num_timesteps = _config["timesteps"]
        self.beta_scheduler = BetaScheduler(
            timesteps=self.num_timesteps, scheduler_mode=_config["beta_schedule"]
        )
        self.sigma_scheduler = SigmaScheduler(timesteps=self.num_timesteps)
        # d3pm for atom types
        self.max_atoms = _config["max_atoms"]
        self.d3pm = D3PM(
            beta_scheduler=self.beta_scheduler,
            num_timesteps=_config["timesteps"],
            max_atoms=_config["max_atoms"],
            d3pm_hybrid_coeff=_config["d3pm_hybrid_coeff"],
        )
        # lattice noise; plain attribute (bool tensor, kept out of state_dict)
        self.mask_lattice_matrix = paddle.to_tensor(
            [[1, 0, 1], [1, 1, 1], [0, 0, 1]]
        ).astype("bool")
        # all lattice matrix are re-defined from a function "from_paramters" in pymatgen
        # it will zero out three elements of the lattice matrix
        # decoder
        self.decoder = CSPNet(
            hidden_dim=_config["hidden_dim"],
            time_dim=_config["time_dim"],
            text_dim=_config["text_dim"] if self.text_guide else 0,
            num_layers=_config["num_layers"],
            max_atoms=_config["max_atoms"],
            act_fn=_config["act_fn"],
            dis_emb=_config["dis_emb"],
            num_freqs=_config["num_freqs"],
            edge_style=_config["edge_style"],
            cutoff=_config["cutoff"],
            max_neighbors=_config["max_neighbors"],
            ln=_config["ln"],
            ip=_config["ip"],
            smooth=_config["smooth"],
            pred_atom_types=_config["pred_atom_types"],
        )
        # set cost for loss
        self.cost_atom_types = _config["cost_atom_types"]
        self.cost_lattice = _config["cost_lattice"]
        self.cost_coords = _config["cost_coords"]

    # Two-stage decode: the eager stage builds the fully-connected edges
    # here, outside any compiled region (the per-graph python loop is cheap
    # eager but pathological under symbolic-shape compilation), then hands
    # the precomputed static-shape edges to the runtime boundary. The
    # boundary (denoise_step) is one complete decoder forward on those
    # edges and returns a plain tuple; _decode re-wraps it into the
    # DECODER_OUTPUTS namedtuple so internal call sites and the
    # monitor/loss contract see no change. CFG doubling and the
    # Predictor/Corrector composition stay outside. time_emb is always
    # computed by the caller outside the boundary.
    def _decode(
        self,
        time_emb,
        atom_types,
        frac_coords,
        lattices,
        num_atoms,
        node2graph,
        text_embeds=None,
    ):
        edges, frac_diff = self.decoder.gen_edges(num_atoms, frac_coords)
        return DECODER_OUTPUTS(
            *self._runtime_denoise(
                time_emb,
                atom_types,
                frac_coords,
                lattices,
                num_atoms,
                node2graph,
                edges,
                frac_diff,
                text_embeds,
            )
        )

    @runtime_boundary("denoise_step")
    def _runtime_denoise(
        self,
        time_emb,
        atom_types,
        frac_coords,
        lattices,
        num_atoms,
        node2graph,
        edges,
        frac_diff,
        text_embeds=None,
    ):
        return self.decoder.forward_with_edges(
            atom_types=atom_types,
            frac_coords=frac_coords,
            lattices=lattices,
            num_atoms=num_atoms,
            node2graph=node2graph,
            edges=edges,
            frac_diff=frac_diff,
            t=time_emb,
            text_embeds=text_embeds,
        )

    def _lattice_mask(self, like_tensor):
        """Lattice mask aligned to the dtype/place of the given tensor."""
        mask = self.mask_lattice_matrix
        if mask.place != like_tensor.place:
            mask = mask.to(like_tensor.place)
        return mask.astype(like_tensor.dtype)

    @property
    def _param_place(self):
        return self.beta_scheduler.betas.place

    def _to_tensor(self, value):
        if paddle.is_tensor(value):
            return value
        return paddle.to_tensor(value)

    # The trainer logs every loss_dict entry via float(), so loss_dict
    # carries scalar losses only. The noise monitoring tensors, the decoder
    # outputs and the x_t intermediates are exposed on the sibling
    # "monitor_dict" key instead.
    _MONITOR_KEYS = (
        "true_noise_lattice",
        "pred_noise_lattice",
        "true_noise_coords",
        "pred_noise_coords",
        "atom_types_out",
        "lattice_out",
        "x_t_atom_types",
        "x_t_lattice",
        "x_t_coords",
    )

    # Optional fixed draws for deterministic replay: fixed t / Gumbel u /
    # lattice eps / coords eps / drop mask / precomputed 768-d text
    # vectors. Default None keeps the random draw path; the numeric body is
    # shared, only the draw sites branch.
    def forward(
        self,
        batch: Dict[str, Any],
        batched_t=None,
        gumbel_u=None,
        eps_lattice=None,
        eps_coords=None,
        text_vec_768=None,
        text_keep_mask=None,
    ) -> Dict[str, Any]:
        """Calculate P_loss

        Args:
            batch (dict): batch dict with structure_array (frac_coords
                [B_n, 3], atom_types [B_n], lattices [B, 3, 3], natoms [B])
                and, for text-guided models, a text list of length B.
            batched_t (Optional): [B] int timesteps; None -> random draw.
            gumbel_u (Optional): [B_n, max_atoms] Gumbel noise (fp32 as
                drawn); None -> random draw.
            eps_lattice (Optional): [B, 3, 3] lattice noise, pre-mask
                multiply; None -> random draw.
            eps_coords (Optional): [B_n, 3] coords noise; None -> random
                draw.
            text_vec_768 (Optional): [B, 768] precomputed BERT-CLS-after-
                projection vectors, entering at the null-substitution point
                followed by the in-model text_emb MLP (requires
                text_keep_mask). None -> the regular text_encoder path.
            text_keep_mask (Optional): [B] bool condition keep mask.
        Returns:
            Dict[str, Any]: {"loss_dict": {five scalar losses},
            "monitor_dict": {noise monitoring tensors}}
        """
        out = self._forward(
            batch,
            batched_t=batched_t,
            gumbel_u=gumbel_u,
            eps_lattice=eps_lattice,
            eps_coords=eps_coords,
            text_vec_768=text_vec_768,
            text_keep_mask=text_keep_mask,
        )
        return {
            "loss_dict": {k: v for k, v in out.items() if k not in self._MONITOR_KEYS},
            "monitor_dict": {k: out[k] for k in self._MONITOR_KEYS},
        }

    def _forward(
        self,
        batch: Dict[str, Any],
        batched_t=None,
        gumbel_u=None,
        eps_lattice=None,
        eps_coords=None,
        text_vec_768=None,
        text_keep_mask=None,
    ) -> Dict[str, Any]:
        structure_array = batch["structure_array"]
        frac_coords = self._to_tensor(structure_array["frac_coords"])  # [B_n, 3]
        lattices = self._to_tensor(structure_array["lattices"])  # [B, 3, 3]
        a_0 = self._to_tensor(structure_array["atom_types"]).astype("int64")  # [B_n]
        natoms = self._to_tensor(structure_array["natoms"]).reshape([-1])  # [B]

        batch_size = natoms.shape[0]
        num_nodes = frac_coords.shape[0]
        # rebuild the node->graph index from natoms (flat concatenation)
        batch_idx = paddle.repeat_interleave(
            paddle.arange(batch_size), repeats=natoms
        )  # [B_n]

        # draw site: timesteps (injected or random)
        if batched_t is None:
            batched_t = self.beta_scheduler.uniform_sample_t(
                batch_size, device=self._param_place
            )  # [B]
        else:
            batched_t = self._to_tensor(batched_t)  # [B]

        time_emb = self.time_embed(batched_t)  # [B, time_dim]
        # time_emb content is fp32-computed (parameter-free arange * python
        # floats); upcast to the data dtype so non-fp32-default builds keep
        # one dtype through the FiLM concat.
        time_emb = time_emb.astype(lattices.dtype)

        alpha_cumprod = self.beta_scheduler.alphas_cumprod[batched_t]  # [B]

        c0 = paddle.sqrt(alpha_cumprod)  # [B]
        c1 = paddle.sqrt(1.0 - alpha_cumprod)  # [B]

        sigmas = self.sigma_scheduler.sigmas[batched_t]  # [B]
        sigmas_norm = self.sigma_scheduler.sigmas_norm[batched_t]  # [B]

        # q_sample (x_t | x_0)
        # 1) d3pm for atom types
        t_per_node = batched_t[batch_idx]  # [B_n]
        # draw site: Gumbel noise (injected stays fp32 as given; q_sample
        # promotes it against the buffer-dtype logits)
        if gumbel_u is None:
            noise_atom_types = paddle.rand(
                [num_nodes, self.max_atoms]
            )  # [B_n, max_atoms]
        else:
            noise_atom_types = self._to_tensor(gumbel_u)  # [B_n, max_atoms]
        x_t_atom_types = self.d3pm.q_sample(a_0, t_per_node, noise_atom_types)  # [B_n]

        # 2) variance-preserving for lattice
        l_0 = lattices
        mask_lattice_matrix = self._lattice_mask(l_0)
        # draw site: lattice noise (injected eps is pre-mask-multiply)
        if eps_lattice is None:
            noise_lattice = (
                paddle.randn(l_0.shape).astype(l_0.dtype) * mask_lattice_matrix
            )  # [B, 3, 3]
        else:
            noise_lattice = (
                self._to_tensor(eps_lattice).astype(l_0.dtype) * mask_lattice_matrix
            )
        x_t_lattice = c0[:, None, None] * l_0 + c1[:, None, None] * noise_lattice

        # 3) variance-exploding for fractional coordinates
        # draw site: coords noise
        if eps_coords is None:
            noise_coords = paddle.randn(frac_coords.shape).astype(frac_coords.dtype)
        else:
            noise_coords = self._to_tensor(eps_coords).astype(frac_coords.dtype)
        sigmas_per_atom = sigmas[batch_idx][:, None]  # [B_n, 1]
        sigmas_norm_per_atom = sigmas_norm[batch_idx][:, None]  # [B_n, 1]

        target_coords = d_log_p_wrapped_normal(
            sigmas_per_atom * noise_coords, sigmas_per_atom
        ) / paddle.sqrt(
            sigmas_norm_per_atom
        )  # [B_n, 1]
        x_t_coords = (frac_coords + sigmas_per_atom * noise_coords) % 1.0

        # text embedding
        # draw site: precomputed 768-d vectors + keep mask enter at the
        # null-substitution point, then the in-model text_emb MLP runs
        # (tail shared with TextEncoder.get_text_embeds)
        text_embeds = None
        if text_vec_768 is not None:
            assert text_keep_mask is not None, "text_vec_768 needs text_keep_mask"
            vec = self._to_tensor(text_vec_768).astype(
                self.text_encoder.null_text_embeds.dtype
            )  # [B, 768]
            keep = self._to_tensor(text_keep_mask).astype("bool")  # [B]
            text_embeds = paddle.where(
                keep[:, None],
                vec,
                self.text_encoder.null_text_embeds.expand(
                    [batch_size, self.text_encoder.text_embed_dim]
                ),
            )
            text_embeds = self.text_encoder.text_emb(text_embeds)  # [B, D]
        elif self.text_guide:
            text_embeds = self.text_encoder.get_text_embeds(
                batch["text"], self.cond_drop_prob
            )  # [B, D]

        # predict noise lattice and fractional coordinates (single decoder
        # forward through the denoise_step runtime boundary)
        decoder_output = self._decode(
            time_emb=time_emb,
            atom_types=x_t_atom_types,
            frac_coords=x_t_coords,
            lattices=x_t_lattice,
            num_atoms=natoms,
            node2graph=batch_idx,
            text_embeds=text_embeds,
        )
        pred_x_start_atom_types = decoder_output.atom_types_out  # [B_n, max_atoms]
        pred_noise_lattice = decoder_output.lattice_out  # [B, 3, 3]
        pred_noise_coords = decoder_output.coords_out  # [B_n, 3]

        # loss for atom_types (1) VB (2) CE loss
        true_q_posterior_logits = self.d3pm.q_posterior_logits(
            a_0, x_t_atom_types, t_per_node
        )
        pred_q_posterior_logits = self.d3pm.q_posterior_logits(
            pred_x_start_atom_types, x_t_atom_types, t_per_node, is_x_0_one_hot=True
        )
        vb_loss = self.d3pm.categorical_kl_logits(
            true_q_posterior_logits, pred_q_posterior_logits
        )
        ce_loss = F.cross_entropy(
            pred_x_start_atom_types.reshape([-1, self.max_atoms]), a_0
        )
        loss_atom_types = vb_loss + ce_loss * self.d3pm.hybrid_coeff
        # loss for lattice
        loss_lattice = F.mse_loss(
            pred_noise_lattice.masked_select(mask_lattice_matrix.astype("bool")),
            noise_lattice.masked_select(mask_lattice_matrix.astype("bool")),
        )
        # loss for coords
        loss_coords = F.mse_loss(pred_noise_coords, target_coords)

        loss = (
            self.cost_atom_types * loss_atom_types
            + self.cost_lattice * loss_lattice
            + self.cost_coords * loss_coords
        )
        return {
            "loss": loss,
            "vb": vb_loss,
            "ce": ce_loss,
            "lattice": loss_lattice,
            "coords": loss_coords,
            # monitoring tensors
            "true_noise_lattice": noise_lattice.masked_select(
                mask_lattice_matrix.astype("bool")
            ),
            "pred_noise_lattice": pred_noise_lattice.masked_select(
                mask_lattice_matrix.astype("bool")
            ),
            "true_noise_coords": target_coords,
            "pred_noise_coords": pred_noise_coords,
            # decoder outputs and noising intermediates (see the
            # loss/monitor split note)
            "atom_types_out": pred_x_start_atom_types,
            "lattice_out": pred_noise_lattice,
            "x_t_atom_types": x_t_atom_types,
            "x_t_lattice": x_t_lattice,
            "x_t_coords": x_t_coords,
        }

    def model_predictions(
        self,
        time_emb: paddle.Tensor,
        atom_types: paddle.Tensor,
        frac_coords: paddle.Tensor,
        lattices: paddle.Tensor,
        batch_natoms: paddle.Tensor,
        batch_idx: paddle.Tensor,
        cond_scale: float,
        text_embeds: Optional[paddle.Tensor] = None,
        null_text_embeds: Optional[paddle.Tensor] = None,
    ):
        if self.text_guide:
            # conditional sampling (CFG: two boundary calls, combination
            # outside)
            conditional_decoder_output = self._decode(
                time_emb=time_emb,
                atom_types=atom_types,
                frac_coords=frac_coords,
                lattices=lattices,
                num_atoms=batch_natoms,
                node2graph=batch_idx,
                text_embeds=text_embeds,
            )
            pred_a_cond = conditional_decoder_output.atom_types_out
            pred_l_cond = conditional_decoder_output.lattice_out
            pred_x_cond = conditional_decoder_output.coords_out

            # unconditional sampling
            unconditional_decoder_output = self._decode(
                time_emb=time_emb,
                atom_types=atom_types,
                frac_coords=frac_coords,
                lattices=lattices,
                num_atoms=batch_natoms,
                node2graph=batch_idx,
                text_embeds=null_text_embeds,
            )
            pred_a_null = unconditional_decoder_output.atom_types_out
            pred_l_null = unconditional_decoder_output.lattice_out
            pred_x_null = unconditional_decoder_output.coords_out

            # conditional free guidance
            pred_a = (1 - cond_scale) * pred_a_null + cond_scale * pred_a_cond
            pred_l = (1 - cond_scale) * pred_l_null + cond_scale * pred_l_cond
            pred_x = (1 - cond_scale) * pred_x_null + cond_scale * pred_x_cond
        else:
            decoder_output = self._decode(
                time_emb=time_emb,
                atom_types=atom_types,
                frac_coords=frac_coords,
                lattices=lattices,
                num_atoms=batch_natoms,
                node2graph=batch_idx,
            )
            pred_a = decoder_output.atom_types_out
            pred_l = decoder_output.lattice_out
            pred_x = decoder_output.coords_out
        return pred_a, pred_l, pred_x

    def _sample_generator(
        self,
        natoms: Union[int, List[int]],
        texts: Optional[Union[str, List[str]]] = None,
        cond_scale: float = 2.0,
        step_lr: float = 1e-5,
        time_start: Optional[int] = None,
        inject_l_T=None,
        inject_x_T=None,
        inject_noises=None,
        inject_text_embeds=None,
        inject_null_text_embeds=None,
        inject_time_emb=None,
        records=None,
    ):
        """Sample from the model

        Args:
            natoms (Union[int, List[int]]): number of atoms for each sample
                e.g., [10, 20, 30] or 10
            texts (Optional[Union[str, List[str]]], optional): text for
                conditional sampling. Defaults to None.
            cond_scale (float, optional): scale for conditional sampling.
                if cond_scale > 0.0, use conditional sampling
                Defaults to 2.0.
            step_lr (float, optional): step size for Langevin dynamics.
                Defaults to 1e-5.

        Deterministic-replay injection (all default None keep the random
        draw path; the numeric body is shared, only the draw sites branch):
            inject_l_T: [B, 3, 3] initial lattice noise, ALREADY multiplied by
                the lattice mask.
            inject_x_T: [B_n, 3] initial coords noise, pre-modulo (the loop
                still applies % 1.0). a_T stays the all-zero constant.
            inject_noises: dict {t: {"rand_a" | "rand_l" | "rand_x_pred" |
                "rand_x_corr": tensor}} per-step noises; rand_a stays fp32 as
                drawn, rand_l is pre-mask-multiply (the loop applies it).
            inject_text_embeds / inject_null_text_embeds: [B, D] tensors
                entering AFTER get_text_embeds (post text_emb MLP), replacing
                both text encoder calls (both must be given together).
            inject_time_emb: dict {t: [B, time_dim]} exact time embeddings
                (fp32-computed content, upcast in the loop like the regular
                path) for exact time-embedding replay.
            records: optional list; when given, one dict per step is appended
                with the pre/post states, the three predictions and the D3PM
                posterior logits.
        """
        # construct a batch
        if isinstance(natoms, int):
            natoms = [natoms]
        if texts is not None and isinstance(texts, str):
            texts = [texts]
            assert len(natoms) == len(
                texts
            ), "natoms and texts must have the same number of elements."

        batch_natoms = paddle.to_tensor(natoms, dtype="int64").to(self._param_place)
        batch_size = batch_natoms.shape[0]
        num_nodes = int(batch_natoms.sum())
        batch_idx = paddle.repeat_interleave(
            paddle.arange(batch_size), repeats=batch_natoms
        ).to(self._param_place)
        mask_lattice_matrix = self._lattice_mask(
            self.beta_scheduler.betas
        )  # placeholder for dtype/place alignment

        # sample from pure noise
        a_T = paddle.zeros([num_nodes], dtype="int64").to(self._param_place)
        # draw site: initial lattice noise (injected value is already masked)
        if inject_l_T is not None:
            l_T = self._to_tensor(inject_l_T).astype(mask_lattice_matrix.dtype)
        else:
            l_T = (
                paddle.randn([batch_size, 3, 3]).astype(mask_lattice_matrix.dtype)
                * mask_lattice_matrix
            )
        # draw site: initial coords noise (pre-modulo; the % 1.0 below stands)
        if inject_x_T is not None:
            x_T = self._to_tensor(inject_x_T).astype(mask_lattice_matrix.dtype)
        else:
            x_T = paddle.randn([num_nodes, 3]).to(self._param_place)

        # starting timestep of the reverse process; None = the full
        # training horizon (num_inference_steps of the ppmat sampler)
        if time_start is None:
            time_start = self.beta_scheduler.timesteps

        trajectory_container = TrajectoryContainer(total_steps=time_start)

        # Initialize the first step of the trajectory at t=T
        trajectory_container[time_start] = TrajectoryStep(
            num_atoms=batch_natoms,
            atom_types=a_T,
            frac_coords=x_T % 1.0,
            lattices=l_T,
            batch_idx=batch_idx,
        )

        with paddle.no_grad():
            # draw site: text embeddings entering after get_text_embeds (the
            # post-MLP [B, D] tensors replace both text encoder calls)
            if inject_text_embeds is not None or inject_null_text_embeds is not None:
                assert inject_text_embeds is not None and (
                    inject_null_text_embeds is not None
                ), "text embed injection needs both cond and null tensors"
                dtype = self.text_encoder.null_text_embeds.dtype
                text_embeds = self._to_tensor(inject_text_embeds).astype(dtype)
                null_text_embeds = self._to_tensor(inject_null_text_embeds).astype(
                    dtype
                )
            elif self.text_guide:
                text_embeds = self.text_encoder.get_text_embeds(
                    texts,
                    cond_drop_prob=0.0,
                )
                null_text_embeds = self.text_encoder.get_text_embeds(
                    texts,
                    cond_drop_prob=1.0,
                )
            else:
                text_embeds = None
                null_text_embeds = None

        for t in tqdm(range(time_start, 0, -1)):
            with paddle.no_grad():
                batched_t = paddle.full([batch_size], t, dtype="int64").to(
                    self._param_place
                )
                # draw site: time embedding (injected = exact external
                # computation; upcast like the regular path)
                if inject_time_emb is not None:
                    time_emb = self._to_tensor(inject_time_emb[t]).astype(l_T.dtype)
                else:
                    time_emb = self.time_embed(batched_t)
                # time_emb content is fp32-computed; upcast to the data
                # dtype (see _forward)
                time_emb = time_emb.astype(l_T.dtype)

                a_t = trajectory_container[t].atom_types
                x_t = trajectory_container[t].frac_coords
                l_t = trajectory_container[t].lattices
                if records is not None:
                    rec = {"t": t, "a_t_in": a_t, "x_t_in": x_t, "l_t_in": l_t}

                ### Predictor ###
                pred_a, pred_l, pred_x = self.model_predictions(
                    time_emb=time_emb,
                    atom_types=a_t,
                    frac_coords=x_t,
                    lattices=l_t,
                    batch_natoms=batch_natoms,
                    batch_idx=batch_idx,
                    cond_scale=cond_scale,
                    text_embeds=text_embeds,
                    null_text_embeds=null_text_embeds,
                )
                # update the state for atom types
                # draw site: Gumbel noise for the categorical posterior
                step_noise = inject_noises.get(t, {}) if inject_noises else {}
                if t > 1 and "rand_a" in step_noise:
                    rand_a = self._to_tensor(step_noise["rand_a"])  # fp32 as drawn
                else:
                    rand_a = (
                        paddle.rand([num_nodes, self.max_atoms]).to(self._param_place)
                        if t > 1
                        else paddle.zeros([num_nodes, self.max_atoms]).to(
                            self._param_place
                        )
                    )
                t_per_node = batched_t[batch_idx]
                pred_x_start_logits = pred_a  # [B_n, max_atoms]
                if records is not None:
                    # deterministic recompute of the posterior logits
                    # p_logits uses internally
                    rec["p_logits"] = self.d3pm.q_posterior_logits(
                        pred_x_start_logits,
                        a_t,
                        t_per_node,
                        is_x_0_one_hot=True,
                    )
                a_t_minus_1 = self.d3pm.p_logits(
                    pred_x_start_logits=pred_x_start_logits,
                    x_t_atom_types=a_t,
                    t_per_node=t_per_node,
                    noise=rand_a,
                )
                # update the state for lattice
                alphas = self.beta_scheduler.alphas[t]
                alphas_cumprod = self.beta_scheduler.alphas_cumprod[t]
                sigmas = self.beta_scheduler.sigmas[t]
                c0 = 1.0 / paddle.sqrt(alphas)
                c1 = (1.0 - alphas) / paddle.sqrt(1.0 - alphas_cumprod)
                # draw site: lattice noise (injected is pre-mask-multiply)
                if t > 1 and "rand_l" in step_noise:
                    rand_l = self._to_tensor(step_noise["rand_l"]).astype(l_T.dtype)
                else:
                    rand_l = (
                        paddle.randn(l_T.shape).astype(l_T.dtype)
                        if t > 1
                        else paddle.zeros(l_T.shape).astype(l_T.dtype)
                    )
                rand_l = rand_l * mask_lattice_matrix
                l_t_minus_1 = c0 * (l_t - c1 * pred_l) + sigmas * rand_l
                l_t_minus_1 = l_t_minus_1 * mask_lattice_matrix
                # clip the lattice matrix when t = T (for the first step of
                # prediction) otherwise, the lattice matrix could be diverged
                # when the first prediction is too large
                if t == time_start:
                    l_t_minus_1 = l_t_minus_1.clip(-6, 6)
                # update the state for coords (0 -> 0.5 step)
                sigma_x = self.sigma_scheduler.sigmas[t]
                sigma_norm = self.sigma_scheduler.sigmas_norm[t]
                adjacent_sigma_x = self.sigma_scheduler.sigmas[t - 1]
                step_size = sigma_x**2 - adjacent_sigma_x**2
                std_x = paddle.sqrt(
                    (adjacent_sigma_x**2 * (sigma_x**2 - adjacent_sigma_x**2))
                    / (sigma_x**2)
                )
                # draw site: predictor coords noise
                if t > 1 and "rand_x_pred" in step_noise:
                    rand_x = self._to_tensor(step_noise["rand_x_pred"]).astype(
                        x_T.dtype
                    )
                else:
                    rand_x = (
                        paddle.randn(x_T.shape).to(self._param_place)
                        if t > 1
                        else paddle.zeros(x_T.shape).to(self._param_place)
                    )
                pred_x = pred_x * paddle.sqrt(sigma_norm)
                x_t_minus_05 = x_t - step_size * pred_x + std_x * rand_x
                if records is not None:
                    rec["pred_a"] = pred_a
                    rec["pred_l"] = pred_l
                    rec["pred_x_pred"] = pred_x

                ### Corrector ###
                _, _, pred_x = self.model_predictions(
                    time_emb=time_emb,
                    atom_types=a_t_minus_1,
                    frac_coords=x_t_minus_05,
                    lattices=l_t_minus_1,
                    batch_natoms=batch_natoms,
                    batch_idx=batch_idx,
                    cond_scale=cond_scale,
                    text_embeds=text_embeds,
                    null_text_embeds=null_text_embeds,
                )

                # update the state for coords (0.5 -> 1 step)
                step_size = step_lr * (sigma_x / self.sigma_scheduler.sigma_begin) ** 2
                std_x = paddle.sqrt(2 * step_size)
                # draw site: corrector coords noise
                if t > 1 and "rand_x_corr" in step_noise:
                    rand_x = self._to_tensor(step_noise["rand_x_corr"]).astype(
                        x_T.dtype
                    )
                else:
                    rand_x = (
                        paddle.randn(x_T.shape).to(self._param_place)
                        if t > 1
                        else paddle.zeros(x_T.shape).to(self._param_place)
                    )
                pred_x = pred_x * paddle.sqrt(sigma_norm)
                x_t_minus_1 = x_t_minus_05 - step_size * pred_x + std_x * rand_x
                if records is not None:
                    rec.update(
                        {
                            "pred_x_corr": pred_x,
                            "a_t_out": a_t_minus_1,
                            "l_t_out": l_t_minus_1,
                            "x_t_minus_05": x_t_minus_05,
                            "x_t_out": x_t_minus_1,
                        }
                    )
                    records.append(rec)

                trajectory_step = TrajectoryStep(
                    num_atoms=batch_natoms,
                    atom_types=a_t_minus_1,
                    frac_coords=x_t_minus_1 % 1.0,
                    lattices=l_t_minus_1,
                    batch_idx=batch_idx,
                )
            trajectory_container[t - 1] = trajectory_step
            yield trajectory_container.get_atoms(t=t - 1)

    def sample(
        self,
        text_input,
        n_atoms=None,
        n_samples=None,
        cond_scale: float = 2.0,
        step_lr: float = 1e-5,
        return_trajectory: bool = False,
        stream: bool = False,
        num_inference_steps: Optional[int] = None,
        inject_l_T=None,
        inject_x_T=None,
        inject_noises=None,
        inject_text_embeds=None,
        inject_null_text_embeds=None,
        inject_time_emb=None,
        records=None,
    ):
        """Public sampling entry.

        A batch dict following the ppmat sampler protocol (structure_array
        holding a num_atoms entry plus a "text" condition field, as built by
        StructureSampler.sample_by_condition) may be passed as the first
        argument: it is unpacked into per-batch natoms/texts and the final
        states are returned as {"result": [ASE atoms]} for BuildStructure.

        num_inference_steps overrides the starting timestep of the reverse
        process (default: the full training horizon).

        The inject_* / records keyword arguments are the
        deterministic-replay hooks documented on _sample_generator (all
        default None keep the random draw path)."""
        if isinstance(text_input, dict):
            batch = text_input
            natoms = (
                paddle.to_tensor(batch["structure_array"]["num_atoms"])
                .reshape([-1])
                .tolist()
            )
            texts = batch.get("text")
            if texts is None or isinstance(texts, str):
                texts = [texts] * len(natoms)
            else:
                texts = list(texts)
            generator = self._sample_generator(
                natoms,
                texts,
                cond_scale,
                step_lr,
                time_start=num_inference_steps,
                inject_l_T=inject_l_T,
                inject_x_T=inject_x_T,
                inject_noises=inject_noises,
                inject_text_embeds=inject_text_embeds,
                inject_null_text_embeds=inject_null_text_embeds,
                inject_time_emb=inject_time_emb,
                records=records,
            )
            atoms_list = None
            for atoms_list in generator:
                pass
            return {"result": atoms_list}
        natoms = [n_atoms] * n_samples
        texts = [text_input] * n_samples
        generator = self._sample_generator(
            natoms,
            texts,
            cond_scale,
            step_lr,
            time_start=num_inference_steps,
            inject_l_T=inject_l_T,
            inject_x_T=inject_x_T,
            inject_noises=inject_noises,
            inject_text_embeds=inject_text_embeds,
            inject_null_text_embeds=inject_null_text_embeds,
            inject_time_emb=inject_time_emb,
            records=records,
        )
        if stream:
            return generator
        else:
            trajectory = list(generator)
            if return_trajectory:
                return trajectory
            else:
                return trajectory[-1]

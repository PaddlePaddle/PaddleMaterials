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
"""CrystalDiT joint lattice and atom diffusion transformer.

Reference: https://github.com/hanyi2021/CrystalDiT (AAAI 2026).
"""

import functools
import math
from typing import Sequence

import numpy as np
import paddle
import paddle.nn as nn
from scipy.special import erf

from ppmat.models.common import initializer
from ppmat.models.common.runtime import RuntimeMixin
from ppmat.models.common.runtime import runtime_boundary
from ppmat.models.common.time_embedding import uniform_sample_t
from ppmat.schedulers import build_scheduler
from ppmat.utils.misc import ragged_range

MAX_PERIOD = 7
MAX_GROUP = 18

ATOMIC_NUMBER_TO_POSITION = {
    0: (0, 0),
    1: (1, 1),
    2: (1, 18),
    3: (2, 1),
    4: (2, 2),
    5: (2, 13),
    6: (2, 14),
    7: (2, 15),
    8: (2, 16),
    9: (2, 17),
    10: (2, 18),
    11: (3, 1),
    12: (3, 2),
    13: (3, 13),
    14: (3, 14),
    15: (3, 15),
    16: (3, 16),
    17: (3, 17),
    18: (3, 18),
    19: (4, 1),
    20: (4, 2),
    21: (4, 3),
    22: (4, 4),
    23: (4, 5),
    24: (4, 6),
    25: (4, 7),
    26: (4, 8),
    27: (4, 9),
    28: (4, 10),
    29: (4, 11),
    30: (4, 12),
    31: (4, 13),
    32: (4, 14),
    33: (4, 15),
    34: (4, 16),
    35: (4, 17),
    36: (4, 18),
    37: (5, 1),
    38: (5, 2),
    39: (5, 3),
    40: (5, 4),
    41: (5, 5),
    42: (5, 6),
    43: (5, 7),
    44: (5, 8),
    45: (5, 9),
    46: (5, 10),
    47: (5, 11),
    48: (5, 12),
    49: (5, 13),
    50: (5, 14),
    51: (5, 15),
    52: (5, 16),
    53: (5, 17),
    54: (5, 18),
    55: (6, 1),
    56: (6, 2),
    57: (6, 3),
    58: (6, 3.01),
    59: (6, 3.02),
    60: (6, 3.03),
    61: (6, 3.04),
    62: (6, 3.05),
    63: (6, 3.06),
    64: (6, 3.07),
    65: (6, 3.08),
    66: (6, 3.09),
    67: (6, 3.10),
    68: (6, 3.11),
    69: (6, 3.12),
    70: (6, 3.13),
    71: (6, 3.14),
    72: (6, 4),
    73: (6, 5),
    74: (6, 6),
    75: (6, 7),
    76: (6, 8),
    77: (6, 9),
    78: (6, 10),
    79: (6, 11),
    80: (6, 12),
    81: (6, 13),
    82: (6, 14),
    83: (6, 15),
    84: (6, 16),
    85: (6, 17),
    86: (6, 18),
    87: (7, 1),
    88: (7, 2),
    89: (7, 3),
    90: (7, 3.01),
    91: (7, 3.02),
    92: (7, 3.03),
    93: (7, 3.04),
    94: (7, 3.05),
    95: (7, 3.06),
    96: (7, 3.07),
    97: (7, 3.08),
    98: (7, 3.09),
    99: (7, 3.10),
    100: (7, 3.11),
    101: (7, 3.12),
    102: (7, 3.13),
    103: (7, 3.14),
    104: (7, 4),
    105: (7, 5),
    106: (7, 6),
    107: (7, 7),
    108: (7, 8),
    109: (7, 9),
    110: (7, 10),
    111: (7, 11),
    112: (7, 12),
    113: (7, 13),
    114: (7, 14),
    115: (7, 15),
    116: (7, 16),
    117: (7, 17),
    118: (7, 18),
}


def modulate(x, shift, scale):
    """Apply adaLN modulation: ``x * (1 + scale) + shift``."""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TimestepEmbedder(nn.Layer):
    """Embed timesteps with cos-first GLIDE frequencies and an MLP."""

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias_attr=True),
            nn.Silu(),
            nn.Linear(hidden_size, hidden_size, bias_attr=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    def forward(self, t):
        half = self.frequency_embedding_size // 2
        freqs = paddle.exp(
            -math.log(10000) * paddle.arange(half, dtype="float32") / half
        )
        args = t[:, None].cast("float32") * freqs[None]
        embedding = paddle.concat([paddle.cos(args), paddle.sin(args)], axis=-1)
        if self.frequency_embedding_size % 2:
            embedding = paddle.concat(
                [embedding, paddle.zeros_like(embedding[:, :1])], axis=-1
            )
        return self.mlp(embedding)


class DiTBlock(nn.Layer):
    """DiT block with adaptive layer norm zero (adaLN-Zero) conditioning."""

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(
            hidden_size, weight_attr=False, bias_attr=False, epsilon=1e-6
        )
        self.attn = nn.MultiHeadAttention(hidden_size, num_heads)
        self.norm2 = nn.LayerNorm(
            hidden_size, weight_attr=False, bias_attr=False, epsilon=1e-6
        )
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size),
        )
        self.ada_ln_modulation = nn.Sequential(
            nn.Silu(), nn.Linear(hidden_size, 6 * hidden_size, bias_attr=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = paddle.split(
            self.ada_ln_modulation(c), 6, axis=1
        )
        x = x + gate_msa.unsqueeze(1) * self.attn(
            modulate(self.norm1(x), shift_msa, scale_msa)
        )
        x = x + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm2(x), shift_mlp, scale_mlp)
        )
        return x


class CrystalEmbedder(nn.Layer):
    """Embed crystal structure (lattice vectors and atom features)."""

    def __init__(self, hidden_size):
        super().__init__()

        self.lattice_embedder = nn.Linear(3, hidden_size)
        self.atom_embedder = nn.Linear(5, hidden_size)

        self.lattice_type_embedding = self.create_parameter(
            shape=[1, hidden_size], default_initializer=nn.initializer.Constant(0.0)
        )
        self.atom_type_embedding = self.create_parameter(
            shape=[1, hidden_size], default_initializer=nn.initializer.Constant(0.0)
        )

        self.lattice_pos_embed = self.create_parameter(
            shape=[1, 3, hidden_size],
            attr=paddle.ParamAttr(trainable=False),
            default_initializer=nn.initializer.Constant(0.0),
        )

        dim_t = np.arange(hidden_size // 2, dtype=np.float32)
        dim_t = 10000 ** (2 * (dim_t // 2) / hidden_size)
        pos_x = np.arange(3)[:, None] / dim_t
        pos_embed = np.zeros((3, hidden_size))
        pos_embed[:, 0::2] = np.sin(pos_x)
        pos_embed[:, 1::2] = np.cos(pos_x)
        self.lattice_pos_embed.set_value(
            paddle.to_tensor(pos_embed, dtype="float32").unsqueeze(0)
        )

        initializer.normal_(self.lattice_type_embedding, std=0.02)
        initializer.normal_(self.atom_type_embedding, std=0.02)

    def forward(self, lattice_vectors, atom_features):
        lattice_emb = self.lattice_embedder(lattice_vectors)
        lattice_emb = lattice_emb + self.lattice_pos_embed + self.lattice_type_embedding

        atom_emb = self.atom_embedder(atom_features)
        atom_emb = atom_emb + self.atom_type_embedding

        return lattice_emb, atom_emb


class FinalLayer(nn.Layer):
    """Final layer mapping hidden states back to lattice and atom outputs."""

    def __init__(self, hidden_size):
        super().__init__()
        self.norm_final_lattice = nn.LayerNorm(
            hidden_size, weight_attr=False, bias_attr=False, epsilon=1e-6
        )
        self.norm_final_atom = nn.LayerNorm(
            hidden_size, weight_attr=False, bias_attr=False, epsilon=1e-6
        )

        self.lattice_linear = nn.Linear(hidden_size, 3, bias_attr=True)
        self.atom_linear = nn.Linear(hidden_size, 5, bias_attr=True)

        self.ada_ln_modulation = nn.Sequential(
            nn.Silu(), nn.Linear(hidden_size, 4 * hidden_size, bias_attr=True)
        )

    def forward(self, atom_features, lattice_features, c):
        shift_atom, scale_atom, shift_lattice, scale_lattice = paddle.split(
            self.ada_ln_modulation(c), 4, axis=1
        )

        atom_features = modulate(
            self.norm_final_atom(atom_features), shift_atom, scale_atom
        )
        lattice_features = modulate(
            self.norm_final_lattice(lattice_features), shift_lattice, scale_lattice
        )

        lattice_out = self.lattice_linear(lattice_features)
        atom_out = self.atom_linear(atom_features)

        return lattice_out, atom_out


class CrystalDiTBackbone(nn.Layer):
    """DiT backbone producing lattice and atom noise predictions."""

    def __init__(
        self,
        max_atoms,
        hidden_size,
        depth,
        num_heads,
        mlp_ratio=4.0,
        frequency_embedding_size=256,
    ):
        super().__init__()
        self.max_atoms = max_atoms

        self.crystal_embedder = CrystalEmbedder(hidden_size)
        self.t_embedder = TimestepEmbedder(hidden_size, frequency_embedding_size)

        self.blocks = nn.LayerList(
            [
                DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio)
                for _ in range(depth)
            ]
        )

        self.final_layer = FinalLayer(hidden_size)
        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                initializer.xavier_uniform_(module.weight)
                if module.bias is not None:
                    initializer.constant_(module.bias, 0.0)

        self.apply(_basic_init)

        for block in self.blocks:
            initializer.constant_(block.ada_ln_modulation[-1].weight, 0.0)
            initializer.constant_(block.ada_ln_modulation[-1].bias, 0.0)

        initializer.constant_(self.final_layer.ada_ln_modulation[-1].weight, 0.0)
        initializer.constant_(self.final_layer.ada_ln_modulation[-1].bias, 0.0)
        initializer.constant_(self.final_layer.lattice_linear.weight, 0.0)
        initializer.constant_(self.final_layer.lattice_linear.bias, 0.0)
        initializer.constant_(self.final_layer.atom_linear.weight, 0.0)
        initializer.constant_(self.final_layer.atom_linear.bias, 0.0)

    def forward(self, lattice_vectors, atom_features, t):
        lattice_features, atom_features = self.crystal_embedder(
            lattice_vectors, atom_features
        )
        c = self.t_embedder(t)

        tokens = paddle.concat(x=[atom_features, lattice_features], axis=1)
        for block in self.blocks:
            tokens = block(tokens, c)

        atom_features = tokens[:, : self.max_atoms, :]
        lattice_features = tokens[:, self.max_atoms :, :]
        return self.final_layer(atom_features, lattice_features, c)


def gaussian_cdf(x, loc, scale):
    """Evaluate the normal CDF using the erf form for element discretization."""
    z = (x - loc) / (scale * math.sqrt(2.0))
    return 0.5 * (1.0 + erf(z))


# Gaussian smoothing width and lowest accepted probability from the original
# CrystalDiT implementation for mapping predicted positions back to elements.
ELEMENT_MAP_SIGMA = 0.1
ELEMENT_PROB_THRESHOLD = 1e-6


@functools.lru_cache(maxsize=4)
def _element_candidate_table(filter_rare_gases, max_atomic_number):
    """Cache float64 candidate boundaries to preserve CDF discretization precision."""
    numbers = np.array(list(ATOMIC_NUMBER_TO_POSITION), dtype=np.int64)
    positions = np.array(list(ATOMIC_NUMBER_TO_POSITION.values()), dtype=np.float64)
    keep = (numbers == 0) | (numbers <= max_atomic_number)
    if filter_rare_gases:
        keep &= ~np.isin(numbers, [2, 10, 18, 36, 54, 86])
    positions = positions[keep] / np.array([MAX_PERIOD, MAX_GROUP]) * 2 - 1
    step = 1.0 / np.array([MAX_PERIOD, MAX_GROUP])
    upper = np.where(positions < 1, positions + step, np.inf)
    lower = np.where(positions > -1, positions - step, -np.inf)
    return numbers[keep], upper[:, 0], lower[:, 0], upper[:, 1], lower[:, 1]


def map_to_element_batch(
    row_values,
    column_values,
    sigma=ELEMENT_MAP_SIGMA,
    filter_rare_gases=True,
    max_atomic_number=94,
):
    """Map batches of predicted (row, column) pairs to atomic numbers.

    The Gaussian CDF is evaluated with numpy over all candidate
    periodic-table positions at once. Returns 0 for atoms whose best
    candidate probability stays below ``ELEMENT_PROB_THRESHOLD``.

    Args:
        row_values (np.ndarray): ``[N]`` predicted normalized row values.
        column_values (np.ndarray): ``[N]`` predicted normalized column
            values.
        sigma (float, optional): Gaussian smoothing width. Defaults to
            ``ELEMENT_MAP_SIGMA``.
        filter_rare_gases (bool, optional): Exclude rare gas elements from
            the candidates. Defaults to True.
        max_atomic_number (int, optional): Highest atomic number accepted.
            Defaults to 94.

    Returns:
        np.ndarray: ``[N]`` int64 atomic numbers (0 marks invalid atoms).
    """
    table = _element_candidate_table(filter_rare_gases, max_atomic_number)
    numbers, row_upper, row_lower, column_upper, column_lower = table

    row_values = np.asarray(row_values, dtype=np.float64)[:, None]
    column_values = np.asarray(column_values, dtype=np.float64)[:, None]

    row_prob = gaussian_cdf(row_upper, row_values, sigma) - gaussian_cdf(
        row_lower, row_values, sigma
    )
    column_prob = gaussian_cdf(column_upper, column_values, sigma) - gaussian_cdf(
        column_lower, column_values, sigma
    )

    element_probs = row_prob * column_prob
    best_index = element_probs.argmax(axis=1)
    best_prob = element_probs[np.arange(row_values.shape[0]), best_index]
    return np.where(best_prob < ELEMENT_PROB_THRESHOLD, 0, numbers[best_index]).astype(
        np.int64
    )


class CrystalDiT(RuntimeMixin, nn.Layer):
    """Diffusion transformer generating lattice and atom features jointly.

    The model is an unconditional generator: sampling starts from pure noise
    for both the lattice matrix and the fixed-length atom features. Predicted
    lattice vectors are multiplied by ``max_length`` to restore Angstrom
    units, and predicted atoms are mapped back to atomic numbers through the
    2D periodic-table representation.

    Training consumes the standard concatenated ``structure_array`` batch.
    Fixed-length continuous atom tokens are derived inside the model; sampling
    is unconditional and consumes only the number of structures in the batch.

    Args:
        noise_scheduler_cfg (dict): DDPM noise scheduler configuration. See
            `DDPMScheduler` for more details.
        max_atoms (int, optional): Fixed atom sequence length. Defaults to 20.
        hidden_size (int, optional): Transformer hidden dimension. Defaults
            to 512.
        depth (int, optional): Number of DiT blocks. Defaults to 18.
        num_heads (int, optional): Attention head count. Defaults to 8.
        mlp_ratio (float, optional): MLP expansion ratio. Defaults to 4.0.
        frequency_embedding_size (int, optional): Sinusoidal timestep embedding
            dimension feeding the timestep MLP. Defaults to 256.
        max_length (float, optional): Lattice normalization constant used in
            both training data and structure reconstruction. Defaults to
            46.7425.
        atom_loss_weight (float, optional): Atom loss weight. Defaults to
            100.0.
        feature_weights (Sequence[float], optional): Per-feature MSE weights
            over ``(period, group, x, y, z)``. Defaults to
            ``(1.5, 2.0, 1.0, 1.0, 1.0)``.
        execution_backend (str, optional): Execution backend name. Defaults
            to ``"eager"``.
        runtime_options (dict, optional): Runtime options for the execution
            backend. Defaults to None.
    """

    supports_num_atoms_sampling = False

    def __init__(
        self,
        noise_scheduler_cfg: dict,
        max_atoms: int = 20,
        hidden_size: int = 512,
        depth: int = 18,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        frequency_embedding_size: int = 256,
        max_length: float = 46.7425,
        atom_loss_weight: float = 100.0,
        feature_weights: Sequence[float] = (1.5, 2.0, 1.0, 1.0, 1.0),
        execution_backend: str = "eager",
        runtime_options: dict | None = None,
    ) -> None:
        super().__init__()
        self._init_runtime(execution_backend, runtime_options)

        self.max_atoms = max_atoms
        self.max_length = max_length
        self.atom_loss_weight = atom_loss_weight
        self.register_buffer(
            "feature_weights",
            paddle.to_tensor(feature_weights, dtype="float32"),
            persistable=False,
        )
        positions = np.array(list(ATOMIC_NUMBER_TO_POSITION.values()), dtype="float64")
        positions = positions / np.array([MAX_PERIOD, MAX_GROUP]) * 2 - 1
        self.register_buffer(
            "element_positions",
            paddle.to_tensor(positions.astype("float32")),
            persistable=False,
        )

        self.backbone = CrystalDiTBackbone(
            max_atoms=max_atoms,
            hidden_size=hidden_size,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            frequency_embedding_size=frequency_embedding_size,
        )

        self.noise_scheduler = build_scheduler(noise_scheduler_cfg)
        self.num_train_timesteps = self.noise_scheduler.num_train_timesteps

    @runtime_boundary("denoise_step")
    def _runtime_denoise(self, lattice_vectors, atom_features, t):
        return self.backbone(lattice_vectors, atom_features, t)

    def forward(self, batch):
        """Compute the training loss for a batch.

        ``batch["structure_array"]`` contains concatenated atom types and
        fractional coordinates, per-structure atom counts and lattice matrices.
        Per-attribute ``compute_metric_func_dict`` is not applicable to this
        unconditional objective. Supervision is exposed through ``label_dict``.
        """
        structure = batch["structure_array"]
        num_atoms = structure["num_atoms"]
        if bool(paddle.any(num_atoms > self.max_atoms)):
            raise ValueError("The input structure exceeds max_atoms.")
        batch_size = num_atoms.shape[0]
        batch_index = paddle.repeat_interleave(paddle.arange(batch_size), num_atoms)
        atom_index = ragged_range(num_atoms)
        features = paddle.concat(
            [self.element_positions[structure["atom_types"]], structure["frac_coords"]],
            axis=-1,
        )
        atom_features = paddle.scatter(
            paddle.full([batch_size * self.max_atoms, 5], -1.0),
            batch_index * self.max_atoms + atom_index,
            features,
        ).reshape([batch_size, self.max_atoms, 5])
        lattice_vectors = structure["lattice"] / self.max_length

        times = uniform_sample_t(batch_size, self.num_train_timesteps)
        rand_lattice = paddle.randn(
            shape=lattice_vectors.shape, dtype=lattice_vectors.dtype
        )
        rand_atom = paddle.randn(shape=atom_features.shape, dtype=atom_features.dtype)

        noisy_lattice = self.noise_scheduler.add_noise(
            lattice_vectors, rand_lattice, timesteps=times
        )
        noisy_atom = self.noise_scheduler.add_noise(
            atom_features, rand_atom, timesteps=times
        )

        pred_lattice, pred_atom = self._runtime_denoise(
            noisy_lattice, noisy_atom, times
        )

        lattice_loss = paddle.mean((rand_lattice - pred_lattice) ** 2, axis=[1, 2])
        weighted_sq_diff = (rand_atom - pred_atom) ** 2 * self.feature_weights.reshape(
            [1, 1, -1]
        )
        atom_loss = paddle.mean(weighted_sq_diff, axis=[1, 2])

        loss = paddle.mean(lattice_loss + self.atom_loss_weight * atom_loss)
        return {
            "loss_dict": {
                "loss": loss,
                "lattice_loss": lattice_loss.mean(),
                "atom_loss": atom_loss.mean(),
            },
            "label_dict": {"structure_array": structure},
        }

    @paddle.no_grad()
    def sample(
        self,
        batch_data,
        num_inference_steps=None,
        filter_rare_gases=True,
        max_atomic_number=94,
    ):
        """Generate crystal structures from pure noise.

        Args:
            batch_data (dict): Batch dict; only the batch size is consumed
                because the model is unconditional.
            num_inference_steps (int, optional): DDPM sampling steps. Defaults
                to the scheduler training step count.
            filter_rare_gases (bool, optional): Exclude rare gas elements from
                the atom mapping candidates. Defaults to True.
            max_atomic_number (int, optional): Highest atomic number accepted
                in the atom mapping. Defaults to 94.

        Returns:
            dict: ``{"result": [...]}`` where every element holds
            ``num_atoms`` / ``atom_types`` / ``frac_coords`` / ``lattice``.
        """
        structure = batch_data["structure_array"]
        if "atom_types" in structure:
            raise ValueError(
                "CrystalDiT is unconditional. Use a NumAtomsCrystalDataset batch "
                "with by_dataloader sampling, without chemical-formula constraints."
            )
        batch_size = structure["num_atoms"].shape[0]
        if num_inference_steps is None:
            num_inference_steps = self.num_train_timesteps

        lattice_t = paddle.randn([batch_size, 3, 3], dtype="float32")
        atom_t = paddle.randn([batch_size, self.max_atoms, 5], dtype="float32")

        self.noise_scheduler.set_timesteps(num_inference_steps)

        for timestep in self.noise_scheduler.timesteps:
            t_step = int(timestep)
            times = paddle.full([batch_size], t_step, dtype="int64")
            pred_lattice, pred_atom = self._runtime_denoise(lattice_t, atom_t, times)
            lattice_t = self.noise_scheduler.step(
                pred_lattice, t_step, lattice_t
            ).prev_sample
            atom_t = self.noise_scheduler.step(pred_atom, t_step, atom_t).prev_sample

        result = self._postprocess(
            lattice_t,
            atom_t,
            filter_rare_gases=filter_rare_gases,
            max_atomic_number=max_atomic_number,
        )
        return {"result": result}

    def _postprocess(self, lattice_t, atom_t, filter_rare_gases, max_atomic_number):
        """Restore physical units and map atoms back to atomic numbers."""
        lattice_vectors = lattice_t.numpy() * self.max_length
        atom_features = atom_t.numpy()

        atomic_numbers = map_to_element_batch(
            atom_features[:, :, 0].reshape(-1),
            atom_features[:, :, 1].reshape(-1),
            filter_rare_gases=filter_rare_gases,
            max_atomic_number=max_atomic_number,
        ).reshape(atom_features.shape[:2])

        results = []
        for sample_idx in range(atom_features.shape[0]):
            valid_mask = atomic_numbers[sample_idx] > 0
            frac_coords = (atom_features[sample_idx, valid_mask, 2:5] % 1.0).tolist()
            sample_numbers = atomic_numbers[sample_idx][valid_mask].tolist()

            results.append(
                {
                    "num_atoms": len(sample_numbers),
                    "atom_types": sample_numbers,
                    "frac_coords": frac_coords,
                    "lattice": lattice_vectors[sample_idx].astype("float32").tolist(),
                }
            )
        return results

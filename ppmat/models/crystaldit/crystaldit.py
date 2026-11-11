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
"""CrystalDiT: a diffusion transformer for crystal structure generation.

Migrated from https://github.com/hanyi2021/CrystalDiT (AAAI 2026). The model
denoises lattice vectors ``[B, 3, 3]`` and atom features
``[B, max_atoms, 5]`` (period, group, frac x/y/z) jointly with a unified
DiT backbone (adaLN-Zero conditioning). The periodic-table 2D atomic
representation and the standard DDPM schedule follow the original
implementation.
"""

import functools
import math

import numpy as np
import paddle
import paddle.nn as nn
from scipy.special import erf

from ppmat.models.common.runtime import RuntimeMixin
from ppmat.models.common.runtime import runtime_boundary
from ppmat.models.common.time_embedding import uniform_sample_t
from ppmat.schedulers import build_scheduler
from ppmat.utils.periodic_table import ATOMIC_NUMBER_TO_POSITION
from ppmat.utils.periodic_table import MAX_GROUP
from ppmat.utils.periodic_table import MAX_PERIOD
from ppmat.utils.periodic_table import normalize_column
from ppmat.utils.periodic_table import normalize_row


def modulate(x, shift, scale):
    """Apply adaLN modulation: ``x * (1 + scale) + shift``."""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def timestep_embedding(t, dim, max_period=10000):
    """Sinusoidal timestep embedding (cos-first layout, GLIDE style)."""
    half = dim // 2
    freqs = paddle.exp(
        -math.log(max_period) * paddle.arange(start=0, end=half, dtype="float32") / half
    )
    args = t[:, None].cast("float32") * freqs[None]
    embedding = paddle.concat(x=[paddle.cos(args), paddle.sin(args)], axis=-1)
    if dim % 2:
        embedding = paddle.concat(
            x=[embedding, paddle.zeros_like(embedding[:, :1])], axis=-1
        )
    return embedding


class TimestepEmbedder(nn.Layer):
    """Embed scalar timesteps into vector representations."""

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias_attr=True),
            nn.Silu(),
            nn.Linear(hidden_size, hidden_size, bias_attr=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    def forward(self, t):
        t_freq = timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class Attention(nn.Layer):
    """Multi-head self attention equivalent to timm ``vision_transformer``.

    Parameter names (``qkv`` / ``proj``) follow timm so that the released
    checkpoint keys map directly onto this layer.
    """

    def __init__(self, dim, num_heads=8, qkv_bias=False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias_attr=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        batch_size, seq_len, dim = x.shape
        qkv = self.qkv(x)
        qkv = qkv.reshape(
            [batch_size, seq_len, 3, self.num_heads, dim // self.num_heads]
        ).transpose([2, 0, 3, 1, 4])
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = paddle.matmul(q * self.scale, k, transpose_y=True)
        attn = paddle.nn.functional.softmax(attn, axis=-1)
        x = paddle.matmul(attn, v)
        x = x.transpose([0, 2, 1, 3]).reshape([batch_size, seq_len, dim])
        return self.proj(x)


class Mlp(nn.Layer):
    """MLP equivalent to timm ``vision_transformer`` with tanh-approx GELU."""

    def __init__(self, in_features, hidden_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, in_features)

    def forward(self, x):
        return self.fc2(paddle.nn.functional.gelu(self.fc1(x), approximate="tanh"))


class DiTBlock(nn.Layer):
    """DiT block with adaptive layer norm zero (adaLN-Zero) conditioning."""

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(
            hidden_size, weight_attr=False, bias_attr=False, epsilon=1e-6
        )
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True)
        self.norm2 = nn.LayerNorm(
            hidden_size, weight_attr=False, bias_attr=False, epsilon=1e-6
        )
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim)
        self.adaLN_modulation = nn.Sequential(
            nn.Silu(), nn.Linear(hidden_size, 6 * hidden_size, bias_attr=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = paddle.split(
            self.adaLN_modulation(c), 6, axis=1
        )
        x = x + gate_msa.unsqueeze(1) * self.attn(
            modulate(self.norm1(x), shift_msa, scale_msa)
        )
        x = x + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm2(x), shift_mlp, scale_mlp)
        )
        return x


def get_1d_sincos_pos_embed(embed_dim, length):
    """Generate 1D sinusoidal positional embeddings."""
    positions = np.arange(length)
    dim_t = np.arange(embed_dim // 2, dtype=np.float32)
    dim_t = 10000 ** (2 * (dim_t // 2) / embed_dim)

    pos_x = positions[:, np.newaxis] / dim_t
    pos_embed = np.zeros((length, embed_dim))

    pos_embed[:, 0::2] = np.sin(pos_x)
    pos_embed[:, 1::2] = np.cos(pos_x)

    return pos_embed


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

        self.initialize_weights()

    def initialize_weights(self):
        nn.init.xavier_uniform_(self.lattice_embedder.weight)
        nn.init.constant_(self.lattice_embedder.bias, 0.0)
        nn.init.xavier_uniform_(self.atom_embedder.weight)
        nn.init.constant_(self.atom_embedder.bias, 0.0)

        pos_embed = get_1d_sincos_pos_embed(self.lattice_pos_embed.shape[-1], 3).astype(
            "float32"
        )
        self.lattice_pos_embed.set_value(paddle.to_tensor(pos_embed).unsqueeze(0))

        nn.init.normal_(self.lattice_type_embedding, std=0.02)
        nn.init.normal_(self.atom_type_embedding, std=0.02)

    def forward(self, lattice_vectors, atom_features):
        lattice_emb = self.lattice_embedder(lattice_vectors)
        lattice_emb = lattice_emb + self.lattice_pos_embed + self.lattice_type_embedding

        atom_emb = self.atom_embedder(atom_features)
        atom_emb = atom_emb + self.atom_type_embedding

        return lattice_emb, atom_emb


class SimpleDiTBlock(nn.Layer):
    """DiT block concatenating atom and lattice features for self attention."""

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, max_atoms=20):
        super().__init__()
        self.max_atoms = max_atoms
        self.dit_block = DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio)

    def forward(self, atom_features, lattice_features, c):
        combined_features = paddle.concat(x=[atom_features, lattice_features], axis=1)
        combined_features = self.dit_block(combined_features, c)

        atom_features_out = combined_features[:, : self.max_atoms, :]
        lattice_features_out = combined_features[:, self.max_atoms :, :]

        return atom_features_out, lattice_features_out


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

        self.adaLN_modulation = nn.Sequential(
            nn.Silu(), nn.Linear(hidden_size, 4 * hidden_size, bias_attr=True)
        )

    def forward(self, atom_features, lattice_features, c):
        shift_atom, scale_atom, shift_lattice, scale_lattice = paddle.split(
            self.adaLN_modulation(c), 4, axis=1
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
                SimpleDiTBlock(
                    hidden_size, num_heads, mlp_ratio=mlp_ratio, max_atoms=max_atoms
                )
                for _ in range(depth)
            ]
        )

        self.final_layer = FinalLayer(hidden_size)
        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

        self.apply(_basic_init)

        for block in self.blocks:
            nn.init.constant_(block.dit_block.adaLN_modulation[-1].weight, 0.0)
            nn.init.constant_(block.dit_block.adaLN_modulation[-1].bias, 0.0)

        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0.0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0.0)
        nn.init.constant_(self.final_layer.lattice_linear.weight, 0.0)
        nn.init.constant_(self.final_layer.lattice_linear.bias, 0.0)
        nn.init.constant_(self.final_layer.atom_linear.weight, 0.0)
        nn.init.constant_(self.final_layer.atom_linear.bias, 0.0)

    def forward(self, lattice_vectors, atom_features, t):
        lattice_emb, atom_emb = self.crystal_embedder(lattice_vectors, atom_features)
        c = self.t_embedder(t)

        atom_features = atom_emb
        lattice_features = lattice_emb

        for block in self.blocks:
            atom_features, lattice_features = block(atom_features, lattice_features, c)

        lattice_out, atom_out = self.final_layer(atom_features, lattice_features, c)
        lattice_out = lattice_out.reshape([-1, 3, 3])

        return lattice_out, atom_out


def gaussian_cdf(x, loc, scale):
    """Standard normal CDF evaluated with the vectorized ``scipy.special.erf``.

    ``scipy.special.ndtr`` / ``scipy.stats.norm.cdf`` are bitwise different
    (ULP-level) from this erf path; the erf form is kept because the element
    mapping is verified bitwise against the original implementation.
    """
    z = (x - loc) / (scale * math.sqrt(2.0))
    return 0.5 * (1.0 + erf(z))


# Gaussian smoothing width and lowest accepted probability from the original
# CrystalDiT implementation for mapping predicted positions back to elements.
ELEMENT_MAP_SIGMA = 0.1
ELEMENT_PROB_THRESHOLD = 1e-6


def _element_candidate_table(filter_rare_gases, max_atomic_number):
    """Build the float64 boundary table for candidate atomic numbers.

    All CDF arithmetic stays in float64: the per-sample loop this table
    replaces promoted the float32 predictions to float64 before evaluation,
    so keeping float64 here preserves the results bitwise.
    """
    rare_gas_elements = [2, 10, 18, 36, 54, 86]

    candidates = [(0, 0, 0)]
    for atomic_number, (elem_row, elem_column) in ATOMIC_NUMBER_TO_POSITION.items():
        if atomic_number == 0:
            continue
        if filter_rare_gases and atomic_number in rare_gas_elements:
            continue
        if atomic_number > max_atomic_number:
            continue
        candidates.append((atomic_number, elem_row, elem_column))

    row_step = 1.0 / MAX_PERIOD
    column_step = 1.0 / MAX_GROUP

    numbers, row_upper, row_lower, column_upper, column_lower = [], [], [], [], []
    for atomic_number, elem_row, elem_column in candidates:
        norm_row = normalize_row(elem_row)
        norm_column = normalize_column(elem_column)

        numbers.append(atomic_number)
        row_upper.append(norm_row + row_step if norm_row < 1 else np.inf)
        row_lower.append(norm_row - row_step if norm_row > -1 else -np.inf)
        column_upper.append(norm_column + column_step if norm_column < 1 else np.inf)
        column_lower.append(norm_column - column_step if norm_column > -1 else -np.inf)

    return (
        np.array(numbers, dtype=np.int64),
        np.array(row_upper, dtype=np.float64),
        np.array(row_lower, dtype=np.float64),
        np.array(column_upper, dtype=np.float64),
        np.array(column_lower, dtype=np.float64),
    )


@functools.lru_cache(maxsize=4)
def _cached_candidate_table(filter_rare_gases, max_atomic_number):
    """Cache the candidate boundary table per filter configuration."""
    return _element_candidate_table(filter_rare_gases, max_atomic_number)


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
    numbers, row_upper, row_lower, column_upper, column_lower = _cached_candidate_table(
        filter_rare_gases, max_atomic_number
    )

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

    Unlike the variable-length crystal generators (DiffCSP/MatterGen) that
    read ``batch_data["structure_array"]``, this model consumes fixed-length
    top-level fields (``lattice_vectors``/``atom_features``) because the
    padding/packing happens once in the dataset; ``sample`` still accepts a
    ``structure_array`` batch for compatibility with the shared sampler
    entry points.

    Args:
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
        num_train_timesteps (int, optional): Diffusion step count. Defaults
            to 1000.
        beta_start (float, optional): Linear schedule start. Defaults to
            1e-4.
        beta_end (float, optional): Linear schedule end. Defaults to 0.02.
        variance_type (str, optional): DDPM posterior variance type. Defaults
            to ``"fixed_small"`` (the original ``FIXED_SMALL`` choice).
        clip_sample (bool, optional): Clip the predicted ``x_0`` during
            sampling. Defaults to True (the original ``clip_denoised``).
        clip_sample_range (float, optional): Sampling clip range. Defaults
            to 1.0 (the original ``clamp(-1, 1)``).
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
        max_atoms=20,
        hidden_size=512,
        depth=18,
        num_heads=8,
        mlp_ratio=4.0,
        frequency_embedding_size=256,
        max_length=46.7425,
        num_train_timesteps=1000,
        beta_start=1e-4,
        beta_end=0.02,
        variance_type="fixed_small",
        clip_sample=True,
        clip_sample_range=1.0,
        atom_loss_weight=100.0,
        feature_weights=(1.5, 2.0, 1.0, 1.0, 1.0),
        execution_backend="eager",
        runtime_options=None,
    ):
        super().__init__()
        self._init_runtime(execution_backend, runtime_options)

        self.max_atoms = max_atoms
        self.hidden_size = hidden_size
        self.max_length = max_length
        self.num_train_timesteps = num_train_timesteps
        self.atom_loss_weight = atom_loss_weight
        self.feature_weights = paddle.to_tensor(
            np.array(feature_weights, dtype="float32")
        )

        self.backbone = CrystalDiTBackbone(
            max_atoms=max_atoms,
            hidden_size=hidden_size,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            frequency_embedding_size=frequency_embedding_size,
        )

        # The original code uses the OpenAI linear beta schedule computed in
        # float64; passing it as trained_betas keeps the schedule bit-exact.
        betas = np.linspace(beta_start, beta_end, num_train_timesteps, dtype=np.float64)
        scheduler_cfg = {
            "__class_name__": "DDPMScheduler",
            "__init_params__": {
                "num_train_timesteps": num_train_timesteps,
                "trained_betas": betas.astype(np.float32),
                "variance_type": variance_type,
                "clip_sample": clip_sample,
                "clip_sample_range": clip_sample_range,
            },
        }
        self.noise_scheduler = build_scheduler(scheduler_cfg)

    def _denoise(self, lattice_vectors, atom_features, t):
        return self._runtime_denoise(lattice_vectors, atom_features, t)

    @runtime_boundary("denoise_step")
    def _runtime_denoise(self, lattice_vectors, atom_features, t):
        return self.backbone(lattice_vectors, atom_features, t)

    def forward(self, batch, **kwargs):
        """Compute the training loss for a batch.

        Args:
            batch (dict): Batch dict with ``lattice_vectors`` ``[B, 3, 3]``
                and ``atom_features`` ``[B, max_atoms, 5]``.

        Returns:
            dict: ``{"loss_dict": {...}, "label_dict": {...}}``. This is an
            unconditional generation model: no per-attribute prediction keys
            exist, so ``compute_metric_func_dict`` does not apply and the
            supervised fields are passed through in ``label_dict``.
        """
        lattice_vectors = batch["lattice_vectors"]
        atom_features = batch["atom_features"]
        batch_size = lattice_vectors.shape[0]

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

        pred_lattice, pred_atom = self._denoise(noisy_lattice, noisy_atom, times)

        lattice_loss = paddle.mean((rand_lattice - pred_lattice) ** 2, axis=[1, 2])
        weighted_sq_diff = (rand_atom - pred_atom) ** 2 * self.feature_weights.reshape(
            [1, 1, -1]
        )
        atom_loss = paddle.mean(weighted_sq_diff, axis=[1, 2])

        loss = paddle.mean(lattice_loss + self.atom_loss_weight * atom_loss)
        loss_dict = {
            "loss": loss,
            "lattice_loss": lattice_loss.mean(),
            "atom_loss": atom_loss.mean(),
        }

        return {
            "loss_dict": loss_dict,
            "label_dict": {
                "lattice_vectors": lattice_vectors,
                "atom_features": atom_features,
            },
        }

    @paddle.no_grad()
    def sample(
        self,
        batch_data,
        num_inference_steps=1000,
        filter_rare_gases=True,
        max_atomic_number=94,
        **kwargs,
    ):
        """Generate crystal structures from pure noise.

        Args:
            batch_data (dict): Batch dict; only the batch size is consumed
                because the model is unconditional.
            num_inference_steps (int, optional): DDPM sampling steps. Defaults
                to 1000.
            filter_rare_gases (bool, optional): Exclude rare gas elements from
                the atom mapping candidates. Defaults to True.
            max_atomic_number (int, optional): Highest atomic number accepted
                in the atom mapping. Defaults to 94.

        Returns:
            dict: ``{"result": [...]}`` where every element holds
            ``num_atoms`` / ``atom_types`` / ``frac_coords`` / ``lattice``.
        """
        if (
            "structure_array" in batch_data
            and batch_data["structure_array"] is not None
        ):
            batch_size = batch_data["structure_array"]["num_atoms"].shape[0]
        elif "lattice_vectors" in batch_data:
            batch_size = batch_data["lattice_vectors"].shape[0]
        else:
            raise ValueError(
                "batch_data must contain either 'structure_array' with "
                "'num_atoms' or 'lattice_vectors' to determine the batch size."
            )

        lattice_t = paddle.randn([batch_size, 3, 3], dtype="float32")
        atom_t = paddle.randn([batch_size, self.max_atoms, 5], dtype="float32")

        self.noise_scheduler.set_timesteps(num_inference_steps)

        for timestep in self.noise_scheduler.timesteps:
            t_step = int(timestep)
            times = paddle.full([batch_size], t_step, dtype="int64")
            pred_lattice, pred_atom = self._denoise(lattice_t, atom_t, times)
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

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

"""Reward components for Chemeleon2 reinforcement learning.

Each component reads the quantities it needs from a shared ``RewardContext``
built once per rollout by :class:`ReinforceReward`, so expensive metrics
(StructureMatcher grouping, reference features, energy above hull) are only
computed when a component actually requires them.
"""

from collections import defaultdict

import numpy as np
import paddle

_AMD_AVAILABLE = True
try:  # Optional dependency, see the README for the port deviation note.
    import amd
except ImportError:  # pragma: no cover - exercised only when amd is absent
    amd = None
    _AMD_AVAILABLE = False


class RewardComponent(paddle.nn.Layer):
    """Base class for all reward components."""

    required_metrics = []

    def __init__(self, weight=1.0, normalize_fn=None, eps=1e-4):
        super().__init__()
        self.weight = weight
        self.normalize_fn = normalize_fn
        self.eps = eps

    def compute(self, ctx):
        """Compute the raw reward tensor for every generated structure."""
        raise NotImplementedError

    def forward(self, ctx, device=None):
        rewards = self.compute(ctx)
        if self.normalize_fn:
            rewards = self._normalize(rewards)
        return paddle.cast(rewards, "float32") * self.weight

    def _normalize(self, rewards):
        if self.normalize_fn == "norm":
            return normalize(rewards, eps=self.eps)
        if self.normalize_fn == "std":
            return standardize(rewards, eps=self.eps)
        if self.normalize_fn == "subtract_mean":
            return rewards - rewards.mean()
        if self.normalize_fn == "clip":
            return paddle.clip(rewards, min=-1.0, max=1.0)
        raise ValueError(
            f"Unknown normalization type: {self.normalize_fn}. "
            "Use 'norm', 'std', 'subtract_mean', 'clip', or None."
        )


class CustomReward(RewardComponent):
    """Placeholder for user-defined reward functions."""

    def compute(self, ctx):
        return paddle.zeros([len(ctx.gen_structures)], dtype="float32")


class CreativityReward(RewardComponent):
    """Combined unique and novel reward with a structural-distance fallback."""

    required_metrics = ["unique", "novel"]

    def compute(self, ctx):
        unique = ctx.unique_flags
        novel = ctx.novel_flags
        rewards = []
        for index, structure in enumerate(ctx.gen_structures):
            u, v = bool(unique[index]), bool(novel[index])
            if u and v:
                rewards.append(1.0)
            elif not u and not v:
                rewards.append(0.0)
            else:
                rewards.append(self._partial_reward(ctx, structure))
        return paddle.to_tensor(np.asarray(rewards), dtype="float32")

    def _partial_reward(self, ctx, structure):
        formula = structure.reduced_formula
        candidates = ctx.reference_by_formula.get(formula, [])
        if not candidates:
            # Unique but identical to no reference structure of this formula.
            return 1.0
        if _AMD_AVAILABLE:
            try:
                amds = _structures_to_amd([structure] + candidates, 100)
                dists = amd.AMD_cdist(amds, amds)[0]
                positive = dists[dists > 0]
                if len(positive):
                    return float(np.clip(positive.min(), 0.0, 1.0))
            except Exception:  # pragma: no cover - fall back on any AMD failure
                pass
        return _lattice_distance_reward(structure, candidates)


class EnergyReward(RewardComponent):
    """Rewards structures with low energy above the convex hull."""

    required_metrics = ["e_above_hull"]

    def compute(self, ctx):
        if ctx.e_above_hull is None:
            raise ValueError(
                "EnergyReward requires a phase diagram and an energy predictor. "
                "Configure `phase_diagram_path` and `energy_model_name` on the "
                "reward function."
            )
        energy = np.nan_to_num(np.asarray(ctx.e_above_hull), nan=1.0)
        energy = np.clip(energy, 0.0, 1.0) * -1.0
        return paddle.to_tensor(energy, dtype="float32")


class StructureDiversityReward(RewardComponent):
    """Rewards diverse crystal structures using an MMD objective."""

    required_metrics = ["structure_diversity"]

    def compute(self, ctx):
        if ctx.reference_structure_features is None:
            raise ValueError(
                "StructureDiversityReward requires reference structures to "
                "compute reference features."
            )
        gen_features = ctx.gen_structure_features
        ref_features = ctx.reference_structure_features
        if ref_features.shape[0] > 50000:
            keep = paddle.randperm(ref_features.shape[0])[:50000]
            ref_features = paddle.gather(ref_features, keep, axis=0)
        return mmd_reward(gen_features, ref_features)["r_indiv"]


class CompositionDiversityReward(RewardComponent):
    """Rewards diverse chemical compositions using an MMD objective."""

    required_metrics = ["composition_diversity"]

    def compute(self, ctx):
        if ctx.reference_composition_features is None:
            raise ValueError(
                "CompositionDiversityReward requires reference structures to "
                "compute reference features."
            )
        gen_features = ctx.gen_composition_features
        ref_features = ctx.reference_composition_features
        if ref_features.shape[0] > 50000:
            keep = paddle.randperm(ref_features.shape[0])[:50000]
            ref_features = paddle.gather(ref_features, keep, axis=0)
        return mmd_reward(gen_features, ref_features)["r_indiv"]


def standardize(x, eps=1e-4):
    std = x.std(unbiased=False)
    if float(std) < eps:
        return paddle.zeros_like(x)
    return (x - x.mean()) / (std + eps)


def normalize(x, eps=1e-4):
    span = float(x.max() - x.min())
    if span < eps:
        return paddle.zeros_like(x)
    return paddle.clip((x - x.min()) / (x.max() - x.min() + eps), 0.0, 1.0)


def mmd_reward(z_gen, z_ref):
    """Per-sample MMD reward (``Training Diffusion Models Towards Diverse Image
    Generation with Reinforcement Learning``)."""

    def poly_k(z, y, deg=3):
        d = z.shape[-1]
        return (paddle.matmul(z, y, transpose_y=True) / d + 1) ** deg

    m, n = z_gen.shape[0], z_ref.shape[0]
    if m < 3 or n < 2:
        raise ValueError(
            "mmd_reward requires at least 3 generated and 2 reference samples."
        )
    k_gg = poly_k(z_gen, z_gen)
    k_rr = poly_k(z_ref, z_ref)
    k_gr = poly_k(z_gen, z_ref)

    r_term = (k_rr.sum() - paddle.trace(k_rr)) / (n * (n - 1))
    g = k_gg.sum() - paddle.trace(k_gg)
    c = k_gr.sum()
    mmd_full = g / (m * (m - 1)) + r_term - 2 * c / (m * n)

    s = k_gg.sum(axis=1) - paddle.diagonal(k_gg)
    t = k_gr.sum(axis=1)
    mp = m - 1
    ap = mp * (mp - 1)
    mmd_drop = (g - 2 * s) / ap + r_term - 2 * (c - t) / (mp * n)
    return {"r": -mmd_full, "r_indiv": mmd_drop - mmd_full}


def _structures_to_amd(structures, k=100):
    import os
    import shutil
    import tempfile

    temp_dir = tempfile.mkdtemp()
    try:
        paths = []
        for index, structure in enumerate(structures):
            path = os.path.join(temp_dir, f"{index:06d}.cif")
            structure.to(filename=path, fmt="cif")
            paths.append(path)
        sets = [amd.CifReader(path).read() for path in paths]
        return [amd.AMD(periodic_set, k) for periodic_set in sets]
    finally:
        shutil.rmtree(temp_dir)


def _lattice_distance_reward(structure, candidates):
    """AMD-free fallback: inverse lattice-parameter distance in ``[0, 1]``."""

    target = np.asarray(list(structure.lattice.abc) + list(structure.lattice.angles))
    scale = np.asarray([10.0, 10.0, 10.0, 90.0, 90.0, 90.0])
    best = np.inf
    for ref in candidates:
        ref_vec = np.asarray(list(ref.lattice.abc) + list(ref.lattice.angles))
        distance = float(np.abs(target - ref_vec).dot(1.0 / scale))
        best = min(best, distance)
    return float(best / (1.0 + best))


def group_reference_by_formula(structures):
    grouped = defaultdict(list)
    for structure in structures:
        grouped[structure.reduced_formula].append(structure)
    return grouped

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

"""Reward orchestration for Chemeleon2 reinforcement learning.

:class:`ReinforceReward` composes one or more
:class:`~ppmat.models.chemeleon2.rl_module.components.RewardComponent`
instances. It builds a :class:`RewardContext` per rollout that lazily computes
the quantities each component declares in ``required_metrics`` (uniqueness,
novelty, energy above hull, VAE structure/composition features).
"""

import gzip
import pickle
from collections import defaultdict

import numpy as np
import paddle

from ppmat.models.chemeleon2.rl_module.components import CompositionDiversityReward
from ppmat.models.chemeleon2.rl_module.components import CreativityReward
from ppmat.models.chemeleon2.rl_module.components import CustomReward
from ppmat.models.chemeleon2.rl_module.components import EnergyReward
from ppmat.models.chemeleon2.rl_module.components import StructureDiversityReward
from ppmat.models.chemeleon2.rl_module.components import normalize
from ppmat.models.chemeleon2.rl_module.components import standardize
from ppmat.models.chemeleon2.rl_module.featurize import featurize

_COMPONENT_REGISTRY = {
    "CustomReward": CustomReward,
    "CreativityReward": CreativityReward,
    "EnergyReward": EnergyReward,
    "StructureDiversityReward": StructureDiversityReward,
    "CompositionDiversityReward": CompositionDiversityReward,
}


class RewardContext:
    """Lazily computed shared quantities for the reward components."""

    def __init__(
        self,
        gen_structures,
        reference_structures,
        vae,
        matcher,
        energy_provider=None,
        reference_feature_cache=None,
        reference_by_formula=None,
        diversity_reference_size=2000,
        seed=42,
    ):
        self.gen_structures = gen_structures
        self.reference_structures = reference_structures
        self.vae = vae
        self.matcher = matcher
        self.energy_provider = energy_provider
        self.reference_feature_cache = reference_feature_cache
        self.diversity_reference_size = diversity_reference_size
        self.seed = seed

        if reference_by_formula is None:
            reference_by_formula = defaultdict(list)
            for structure in reference_structures:
                reference_by_formula[structure.reduced_formula].append(structure)
        self.reference_by_formula = reference_by_formula

        self._unique = None
        self._novel = None
        self._e_above_hull = None
        self._gen_structure_features = None
        self._gen_composition_features = None
        self._reference_structure_features = None
        self._reference_composition_features = None

    @property
    def unique_flags(self):
        if self._unique is None:
            try:
                groups = self.matcher.group_structures(self.gen_structures)
                first_ids = {id(group[0]) for group in groups}
                self._unique = np.array(
                    [id(structure) in first_ids for structure in self.gen_structures]
                )
            except Exception:
                # Never let a malformed lattice abort RL training; treat every
                # structure as its own group so the reward stays well defined.
                self._unique = np.ones(len(self.gen_structures), dtype=bool)
        return self._unique

    @property
    def novel_flags(self):
        if self._novel is None:
            flags = []
            for structure in self.gen_structures:
                candidates = self.reference_by_formula.get(
                    structure.reduced_formula, []
                )
                try:
                    novel = all(
                        not self.matcher.fit(structure, reference)
                        for reference in candidates
                    )
                except Exception:
                    novel = False
                flags.append(novel)
            self._novel = np.array(flags)
        return self._novel

    @property
    def e_above_hull(self):
        if self._e_above_hull is None:
            if self.energy_provider is None:
                raise ValueError(
                    "e_above_hull requested but no energy provider is configured."
                )
            self._e_above_hull = np.asarray(
                self.energy_provider(self.gen_structures), dtype="float64"
            )
        return self._e_above_hull

    def _ensure_reference_features(self):
        if self.reference_feature_cache is not None:
            self._reference_structure_features = self.reference_feature_cache[
                "structure"
            ]
            self._reference_composition_features = self.reference_feature_cache[
                "composition"
            ]
            return
        if not self.reference_structures:
            raise ValueError("Diversity rewards require reference structures.")
        rng = np.random.default_rng(self.seed)
        count = min(self.diversity_reference_size, len(self.reference_structures))
        indices = rng.choice(len(self.reference_structures), size=count, replace=False)
        subset = [self.reference_structures[index] for index in indices]
        features = featurize(subset, self.vae)
        cache = {
            "structure": features["structure_features"],
            "composition": features["composition_features"],
        }
        self.reference_feature_cache = cache
        self._reference_structure_features = cache["structure"]
        self._reference_composition_features = cache["composition"]

    @property
    def gen_structure_features(self):
        if self._gen_structure_features is None:
            features = featurize(self.gen_structures, self.vae)
            self._gen_structure_features = features["structure_features"]
            self._gen_composition_features = features["composition_features"]
        return self._gen_structure_features

    @property
    def gen_composition_features(self):
        if self._gen_composition_features is None:
            _ = self.gen_structure_features
        return self._gen_composition_features

    @property
    def reference_structure_features(self):
        self._ensure_reference_features()
        return self._reference_structure_features

    @property
    def reference_composition_features(self):
        self._ensure_reference_features()
        return self._reference_composition_features


class ReinforceReward(paddle.nn.Layer):
    """Compose reward components into a single per-structure reward."""

    def __init__(
        self,
        components=None,
        normalize_fn=None,
        eps=1e-4,
        reference_file_path=None,
        reference_structures=None,
        diversity_reference_size=2000,
        seed=42,
        stol=0.5,
        angle_tol=10,
        ltol=0.3,
        energy_calculator=None,
        phase_diagram_path=None,
    ):
        super().__init__()
        from pymatgen.analysis.structure_matcher import StructureMatcher

        if reference_file_path is not None and reference_structures is not None:
            raise ValueError(
                "Pass either reference_file_path or reference_structures, not both."
            )
        if reference_file_path is not None:
            reference_structures = _load_reference_structures(reference_file_path)
        self.reference_structures = reference_structures
        self.reference_by_formula = defaultdict(list)
        for structure in reference_structures or []:
            self.reference_by_formula[structure.reduced_formula].append(structure)
        self.normalize_fn = normalize_fn
        self.eps = eps
        self.diversity_reference_size = diversity_reference_size
        self.seed = seed
        self.matcher = StructureMatcher(stol=stol, angle_tol=angle_tol, ltol=ltol)

        self.components = paddle.nn.LayerList(
            [self._build_component(cfg) for cfg in (components or [])]
        )
        self.required_metrics = set()
        for component in self.components:
            self.required_metrics.update(getattr(component, "required_metrics", []))

        self.energy_provider = None
        if "e_above_hull" in self.required_metrics:
            self.energy_provider = EnergyAboveHullProvider(
                phase_diagram_path=phase_diagram_path,
                calculator=energy_calculator,
            )

        object.__setattr__(self, "_reference_feature_cache", None)

    @staticmethod
    def _build_component(cfg):
        cfg = dict(cfg)
        class_name = cfg.pop("__class_name__")
        init_params = cfg.pop("__init_params__", {})
        if class_name not in _COMPONENT_REGISTRY:
            raise KeyError(
                f"Unknown reward component {class_name!r}. "
                f"Available: {sorted(_COMPONENT_REGISTRY)}"
            )
        return _COMPONENT_REGISTRY[class_name](**init_params)

    def forward(self, gen_structures, vae=None, device=None):
        ctx = RewardContext(
            gen_structures=gen_structures,
            reference_structures=self.reference_structures or [],
            vae=vae,
            matcher=self.matcher,
            energy_provider=self.energy_provider,
            reference_feature_cache=self._reference_feature_cache,
            reference_by_formula=self.reference_by_formula,
            diversity_reference_size=self.diversity_reference_size,
            seed=self.seed,
        )
        total = paddle.zeros([len(gen_structures)], dtype="float32")
        for component in self.components:
            total = total + component(ctx, device=device)
        object.__setattr__(
            self, "_reference_feature_cache", ctx.reference_feature_cache
        )
        if paddle.isnan(total).any():
            raise ValueError("NaN values found in rewards.")
        return total

    def normalize(self, rewards):
        if self.normalize_fn == "norm":
            return normalize(rewards, eps=self.eps)
        if self.normalize_fn == "std":
            return standardize(rewards, eps=self.eps)
        if self.normalize_fn == "subtract_mean":
            return rewards - rewards.mean()
        if self.normalize_fn == "clip":
            return paddle.clip(rewards, min=-1.0, max=1.0)
        if self.normalize_fn is None:
            return rewards
        raise ValueError(
            f"Unknown normalization type: {self.normalize_fn}. "
            "Use 'norm', 'std', 'subtract_mean', 'clip', or None."
        )


class EnergyAboveHullProvider:
    """Compute energy above the convex hull with a pymatgen phase diagram."""

    def __init__(self, phase_diagram_path, calculator):
        if phase_diagram_path is None or calculator is None:
            raise ValueError(
                "EnergyReward requires both `phase_diagram_path` (a pickled "
                "pymatgen phase diagram) and an ASE `calculator`."
            )
        self.calculator = calculator
        self.phase_diagram = _load_phase_diagram(phase_diagram_path)

    def __call__(self, structures):
        from pymatgen.analysis.phase_diagram import PDEntry

        energies = []
        for structure in structures:
            try:
                atoms = structure.to_ase_atoms()
                atoms.calc = self.calculator
                energy = self.calculator.get_potential_energy(atoms)
                entry = PDEntry(composition=structure.composition, energy=energy)
                energies.append(
                    self.phase_diagram.get_e_above_hull(
                        entry, allow_negative=True, check_stable=False
                    )
                )
            except Exception:
                energies.append(np.nan)
        return energies


def build_reward(cfg):
    """Build a reward function from a ``__class_name__``/``__init_params__`` cfg."""

    if cfg is None:
        return None
    cfg = dict(cfg)
    class_name = cfg.pop("__class_name__")
    init_params = cfg.pop("__init_params__", {})
    if class_name in ("ReinforceReward", "Chemeleon2ReinforceReward"):
        return ReinforceReward(**init_params)
    raise KeyError(f"Unknown reward function {class_name!r}.")


def _load_reference_structures(path):
    if str(path).endswith(".pkl"):
        with open(path, "rb") as handle:
            return pickle.load(handle)
    if str(path).endswith(".gz"):
        with gzip.open(path, "rb") as handle:
            return pickle.load(handle)
    from monty.serialization import loadfn

    return loadfn(path)


def _load_phase_diagram(path):
    with gzip.open(path, "rb") as handle:
        return pickle.load(handle)


__all__ = [
    "ReinforceReward",
    "Chemeleon2ReinforceReward",
    "RewardContext",
    "EnergyAboveHullProvider",
    "build_reward",
]
Chemeleon2ReinforceReward = ReinforceReward

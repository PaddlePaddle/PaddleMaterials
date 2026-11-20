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

from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
import pandas as pd
from pymatgen.core.structure import Structure

# Reporting thresholds: burden needs enough candidates; diversity only in budget.
_MIN_HIGH_REWARD_CANDIDATES = 100
_DIVERSITY_BUDGET = 3000


def _reduced_formulas(strucs: List[Structure]) -> List[str]:
    return [struc.composition.reduced_formula for struc in strucs]


def _element_combinations(strucs: List[Structure]) -> List[Tuple[str, ...]]:
    return [tuple(sorted({str(e) for e in struc.species})) for struc in strucs]


class LongTimeMem:
    """Long-term memory of every scored structure, used for diversity filtering."""

    def __init__(self):
        self.memory = pd.DataFrame(
            columns=["struc", "comp", "ele_comb", "reward", "RL_step"]
        )
        self.unique_comps = []

    def extend(self, strucs: List[Structure], rewards: np.ndarray, step: int):
        df = pd.DataFrame.from_dict(
            {
                "struc": strucs,
                "comp": _reduced_formulas(strucs),
                "ele_comb": _element_combinations(strucs),
                "reward": rewards,
                "RL_step": [step] * len(strucs),
            }
        )
        self.memory = pd.concat([self.memory, df]) if len(self.memory) > 0 else df
        self.unique_comps = list(self.memory["comp"].unique())

    def div_filter(
        self,
        strucs: List[Structure],
        rewards: np.ndarray,
        tol: int,
        buff: int,
        method: str = "composition",
    ) -> Tuple[np.ndarray, List[int], int, int]:
        if tol >= buff:
            raise ValueError(f"tol must be smaller than buff, but got {tol} and {buff}")
        if method == "composition":
            key, values = "comp", _reduced_formulas(strucs)
        elif method == "element_comb":
            key, values = "ele_comb", _element_combinations(strucs)
        else:
            raise ValueError(
                f"method must be 'composition' or 'element_comb', but got {method}"
            )
        new_rewards, penalty_idx, tol_n, buff_n = [], [], 0, 0
        for i, value in enumerate(values):
            occ = self.memory[key].value_counts().get(value, 0)
            if occ <= tol:
                new_rewards.append(rewards[i])
            elif occ < buff:
                new_rewards.append(rewards[i] * (buff - occ) / (buff - tol))
                tol_n += 1
            else:
                new_rewards.append(0.0)
                penalty_idx.append(i)
                buff_n += 1
        return np.array(new_rewards), penalty_idx, tol_n, buff_n

    def calc_metrics(self, threshold: float) -> Tuple[Optional[float], Optional[float]]:
        if len(self.memory) == 0:
            return None, None
        candidates = (
            self.memory.sort_values("reward", ascending=False).drop_duplicates(
                subset=["comp"]
            )["reward"]
            > threshold
        ).sum()
        burden = (
            len(self.memory) / candidates
            if candidates >= _MIN_HIGH_REWARD_CANDIDATES
            else None
        )
        div_ratio = (
            len(self.unique_comps) / len(self.memory)
            if len(self.memory) <= _DIVERSITY_BUDGET
            else None
        )
        return burden, div_ratio

    def save(self, save_path: str):
        df = self.memory.copy()
        df["cif"] = [struc.to(fmt="cif") for struc in df["struc"].values]
        df.to_csv(save_path, index=False, quoting=1)

    def __len__(self):
        return len(self.memory)


class ReplayBuffer:
    """Replay buffer keeping the best structure per composition."""

    def __init__(
        self, buffer_size: int = 100, sample_size: int = 8, reward_cutoff: float = 0.0
    ):
        self.buffer_size = buffer_size
        self.sample_size = sample_size
        self.reward_cutoff = reward_cutoff
        self.buffer = pd.DataFrame(columns=["struc", "comp", "reward"])

    def extend(
        self,
        strucs: List[Structure],
        rewards: np.ndarray,
        scoring_structures: Optional[List[Structure]] = None,
    ):
        scoring = strucs if scoring_structures is None else scoring_structures
        if len(strucs) != len(scoring) or len(strucs) != len(rewards):
            raise ValueError("Replay training, scoring and rewards must align")
        df = pd.DataFrame.from_dict(
            {"struc": strucs, "comp": _reduced_formulas(scoring), "reward": rewards}
        )
        merged = pd.concat([self.buffer, df]) if len(self.buffer) > 0 else df
        self.buffer = (
            merged.sort_values("reward", ascending=False)
            .drop_duplicates(subset=["comp"])
            .head(self.buffer_size)
        )
        self.buffer = self.buffer[self.buffer["reward"] > self.reward_cutoff]

    def sample(self) -> Tuple[list, np.ndarray]:
        n = min(len(self.buffer), self.sample_size)
        if n > 0:
            sampled = self.buffer.sample(n)
            return sampled["struc"].values.tolist(), sampled["reward"].values
        return [], []

    def memory_purge(self, strucs: List[Structure]) -> None:
        comps = _reduced_formulas(strucs)
        self.buffer = self.buffer[~self.buffer["comp"].isin(comps)]

    def __len__(self) -> int:
        return len(self.buffer)

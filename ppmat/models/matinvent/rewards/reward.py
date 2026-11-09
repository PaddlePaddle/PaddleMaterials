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

import os
from typing import List
from typing import Tuple

import numpy as np
from pymatgen.core.structure import Structure

from ppmat.models.matinvent.rewards.calculators.dft import DFTCalc
from ppmat.models.matinvent.rewards.calculators.fairchem import FairChem
from ppmat.models.matinvent.rewards.calculators.pymatgen import PyMatGen

CALCULATOR_REGISTRY = {
    "PyMatGen": PyMatGen,
    "DFTCalc": DFTCalc,
    "FairChem": FairChem,
}


def _linear_scale(values, minv, maxv):
    ss = (values - minv) / (maxv - minv)
    ss[ss > 1.0] = 1.0
    ss[ss < 0.0] = 0.0
    return ss


class Reward:
    def __init__(
        self,
        root_dir: str,
        prop_cfg: List,
        reward_threshold: float,
        reduce: str = "mean",
    ):
        if reduce not in ("mean", "min", "weight"):
            raise ValueError(
                f"reduce must be one of 'mean', 'min', 'weight', but got {reduce}"
            )
        self.root_dir = root_dir
        self.prop_cfg = prop_cfg
        self.threshold = reward_threshold
        self.reduce = reduce
        self._cache = {}
        names = [cfg["name"] for cfg in prop_cfg]
        if not names or len(set(names)) != len(names):
            raise ValueError("Reward property names must be nonempty and unique")
        if any(
            not name or name in (".", "..") or os.path.basename(name) != name
            for name in names
        ):
            raise ValueError("Reward property names must be directory basenames")
        os.makedirs(root_dir, exist_ok=True)

    def calc_props(self, samples: Tuple[List[Structure], str], label: str = "tmp"):
        values = {}
        failed = None
        for idx, cfg in enumerate(self.prop_cfg):
            calc = self._cache.get(idx)
            if calc is None:
                calculator_cfg = dict(cfg["calculator"])
                cls_name = calculator_cfg.pop("__class_name__")
                if cls_name not in CALCULATOR_REGISTRY:
                    raise ValueError(
                        f"Unknown calculator: {cls_name}, available: "
                        f"{sorted(CALCULATOR_REGISTRY)}"
                    )
                if "root_dir" in calculator_cfg:
                    raise ValueError(
                        "calculator root_dir is owned by Reward and must not be set "
                        "in the config"
                    )
                calc = CALCULATOR_REGISTRY[cls_name](
                    root_dir=os.path.join(self.root_dir, cfg["name"]), **calculator_cfg
                )
                self._cache[idx] = calc
            raw = np.asarray(calc.calc(samples, label), dtype=float)
            values[cfg["name"]] = np.nan_to_num(raw, nan=0.0).astype(float)
            failed_current = ~np.isfinite(raw)
            failed = failed_current if failed is None else (failed | failed_current)
        return values, failed

    def scoring(self, samples: Tuple[List[Structure], str], label: str = "tmp"):
        prop_dict, failed_mask = self.calc_props(samples, label)
        scaled = {}
        for cfg in self.prop_cfg:
            v = prop_dict[cfg["name"]]
            target = cfg["target"]
            if target == "ascending":
                sv = _linear_scale(v, cfg["minv"], cfg["maxv"])
            elif target == "descending":
                sv = _linear_scale(-v, -cfg["maxv"], -cfg["minv"])
            elif isinstance(target, float):
                sv = _linear_scale(-np.abs(v - target), -cfg["maxv"], -cfg["minv"])
            else:
                raise TypeError(
                    "prop cfg.target must be a float, 'ascending', or 'descending'"
                )
            scaled[cfg["name"]] = sv
        if self.reduce == "mean":
            rewards = np.mean(list(scaled.values()), axis=0)
        elif self.reduce == "min":
            rewards = np.min(list(scaled.values()), axis=0)
        elif self.reduce == "weight":
            for cfg in self.prop_cfg:
                scaled[cfg["name"]] *= cfg["weight"]
            rewards = np.sum(list(scaled.values()), axis=0)
        rewards[failed_mask] = 0.0
        return rewards, prop_dict, failed_mask

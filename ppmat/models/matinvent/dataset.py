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

"""RL-side data pipeline for MatInvent.

Samples carry the RL ``reward`` and feed the backbones' ``num_atoms`` /
``atom_types`` conventions, so they stay under the model directory.
"""

import os
from functools import lru_cache
from typing import List

import numpy as np
import paddle.io as io
import smact
from pymatgen.core.structure import Structure
from smact.screening import pauling_test

from ppmat.datasets.collate_fn import DefaultCollator
from ppmat.datasets.custom_data_type import ConcatData
from ppmat.utils import logger as ppmat_logger

_MP20_ATOM_COUNT_PROBABILITIES = (
    0.0,
    0.0021742334905660377,
    0.021079009433962265,
    0.019826061320754717,
    0.15271226415094338,
    0.047132959905660375,
    0.08464770047169812,
    0.021079009433962265,
    0.07808814858490566,
    0.03434551886792453,
    0.0972877358490566,
    0.013303360849056603,
    0.09669811320754718,
    0.02155807783018868,
    0.06522700471698113,
    0.014372051886792452,
    0.06703272405660378,
    0.00972877358490566,
    0.053176591981132074,
    0.010576356132075472,
    0.08995430424528301,
)


def _sample_mp20_num_atoms(total_num):
    return np.random.choice(
        len(_MP20_ATOM_COUNT_PROBABILITIES),
        size=total_num,
        p=_MP20_ATOM_COUNT_PROBABILITIES,
    )


@lru_cache(maxsize=None)
def _is_valid_composition(composition):
    symbols, counts = zip(*composition)
    if len(symbols) == 1 or all(symbol in smact.metals for symbol in symbols):
        return True
    elements = smact.element_dictionary(symbols)
    electronegativities = [elements[symbol].pauling_eneg for symbol in symbols]
    oxidation_states = [elements[symbol].oxidation_states for symbol in symbols]
    known_electronegativities = all(value is not None for value in electronegativities)
    reachable = [set() for _ in range(len(symbols) + 1)]
    reachable[-1].add(0)
    for index in range(len(symbols) - 1, -1, -1):
        reachable[index] = {
            state * counts[index] + charge
            for state in oxidation_states[index]
            for charge in reachable[index + 1]
        }

    def neutral_states(index, charge, states):
        if index == len(symbols):
            yield states
            return
        for state in oxidation_states[index]:
            next_charge = charge + state * counts[index]
            if -next_charge not in reachable[index + 1]:
                continue
            next_states = states + (state,)
            if known_electronegativities and not pauling_test(
                next_states, electronegativities[: index + 1]
            ):
                continue
            yield from neutral_states(index + 1, next_charge, next_states)

    for states in neutral_states(0, 0, ()):
        try:
            if pauling_test(states, electronegativities):
                return True
        except TypeError:
            return True
    return False


def is_valid_structure(
    struc: Structure,
    min_volume: float = 0.1,
    max_lattice_param: float = 25.0,
    min_interatomic_dist: float = 0.5,
) -> bool:
    """Apply the reference geometric, cell-size and SMACT validity checks."""
    if struc is None or struc.num_sites == 0 or struc.volume < min_volume:
        return False
    if max(struc.lattice.abc) >= max_lattice_param:
        return False
    dmat = struc.distance_matrix.copy()
    np.fill_diagonal(dmat, np.inf)
    if dmat.min() < min_interatomic_dist:
        return False
    composition = tuple(sorted(struc.composition.get_el_amt_dict().items()))
    return _is_valid_composition(composition)


def save_structures(structures: List[Structure], save_dir: str, filename: str) -> str:
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, filename)
    from ase.io import write
    from pymatgen.io.ase import AseAtomsAdaptor

    adaptor = AseAtomsAdaptor()
    atoms = [adaptor.get_atoms(struc) for struc in structures]
    write(out_path, atoms, format="extxyz")
    ppmat_logger.info(f"Saved {len(structures)} structures to {out_path}")
    return out_path


class RLDataset(io.Dataset):
    def __init__(self, structures: List[Structure], rewards: np.ndarray):
        if len(structures) != len(rewards):
            raise ValueError(
                f"structures and rewards must have the same length, but got "
                f"{len(structures)} and {len(rewards)}"
            )
        self.structures = structures
        self.rewards = rewards
        ppmat_logger.info(f"Created RLDataset with {len(structures)} samples")

    def __len__(self):
        return len(self.structures)

    def __getitem__(self, idx):
        s = self.structures[idx]
        frac = np.array([site.frac_coords for site in s.sites], dtype=np.float32)
        return {
            "structure_array": {
                "frac_coords": ConcatData(frac),
                "atom_types": ConcatData(
                    np.array([site.specie.Z for site in s.sites], dtype=np.int64)
                ),
                "lattice": s.lattice.matrix.astype(np.float32),
                "num_atoms": len(s),
                "reward": np.float32(self.rewards[idx]),
            }
        }


def create_rl_dataloader(
    structures: List[Structure],
    rewards: np.ndarray,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 0,
) -> io.DataLoader:
    dl = io.DataLoader(
        dataset=RLDataset(structures, rewards),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=DefaultCollator(),
    )
    ppmat_logger.info(f"DataLoader: {len(dl.dataset)} samples, batch_size={batch_size}")
    return dl

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

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from ase.io import write
from ase.visualize.plot import plot_atoms

from ppmat.visualization.adapters import to_ase_atoms


class AtomicStructureVisualizer:
    """Render molecules and crystals through ASE's structure representation."""

    def render(
        self,
        structure,
        *,
        rotation: str = "",
        show_cell: bool = True,
        radii: float = 0.8,
        colors=None,
    ):
        atoms = to_ase_atoms(structure)
        figure, axis = plt.subplots()
        plot_atoms(
            atoms,
            axis,
            rotation=rotation,
            show_unit_cell=2 if show_cell else 0,
            radii=radii,
            colors=colors,
        )
        axis.set_axis_off()
        return figure

    @staticmethod
    def save(figure, path: str | Path, *, dpi: int = 200) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, dpi=dpi, bbox_inches="tight")
        return path

    @staticmethod
    def write(structure, path: str | Path, **kwargs) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        write(path, to_ase_atoms(structure), **kwargs)
        return path

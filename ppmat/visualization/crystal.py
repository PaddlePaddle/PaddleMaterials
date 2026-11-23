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

from pymatgen.core import Structure


class CrystalVisualizer:
    """Render periodic crystals with Pymatgen's VTK visualizer."""

    def render(
        self,
        structure: Structure,
        *,
        show_unit_cell: bool = True,
        show_bonds: bool = False,
        show_polyhedra: bool = False,
        repeat: tuple[int, int, int] = (1, 1, 1),
    ):
        from pymatgen.vis.structure_vtk import StructureVis

        if not isinstance(structure, Structure):
            raise TypeError(
                "structure must be pymatgen.Structure, "
                f"got {type(structure).__name__}."
            )
        repeated = structure * repeat
        visualizer = StructureVis(
            show_unit_cell=show_unit_cell,
            show_bonds=show_bonds,
            show_polyhedron=show_polyhedra,
        )
        visualizer.set_structure(repeated)
        return visualizer

    @staticmethod
    def save(
        visualizer,
        path: str | Path,
        *,
        magnification: int = 1,
    ) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        visualizer.write_image(
            str(path),
            magnification=magnification,
            image_format=path.suffix.lstrip(".") or "png",
        )
        return path

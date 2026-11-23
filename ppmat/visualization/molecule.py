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

from collections.abc import Sequence
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
from rdkit.Geometry import Point3D


class MoleculeVisualizer:
    """Render RDKit molecules as two-dimensional chemical diagrams."""

    def render(
        self,
        molecule: Chem.Mol,
        *,
        size: tuple[int, int] = (300, 300),
        legend: str = "",
        highlight_atoms: Sequence[int] | None = None,
        highlight_bonds: Sequence[int] | None = None,
    ):
        if not isinstance(molecule, Chem.Mol):
            raise TypeError(
                f"molecule must be rdkit.Chem.Mol, got {type(molecule).__name__}."
            )
        return Draw.MolToImage(
            molecule,
            size=size,
            legend=legend,
            highlightAtoms=list(highlight_atoms or []),
            highlightBonds=list(highlight_bonds or []),
        )

    def render_grid(
        self,
        molecules: Sequence[Chem.Mol],
        *,
        legends: Sequence[str] | None = None,
        molecules_per_row: int = 4,
        sub_image_size: tuple[int, int] = (300, 300),
    ):
        molecules = list(molecules)
        if not all(isinstance(molecule, Chem.Mol) for molecule in molecules):
            raise TypeError("molecules must contain only rdkit.Chem.Mol objects.")
        return Draw.MolsToGridImage(
            molecules,
            legends=None if legends is None else list(legends),
            molsPerRow=molecules_per_row,
            subImgSize=sub_image_size,
        )

    def render_trajectory(
        self,
        molecules: Sequence[Chem.Mol],
        *,
        size: tuple[int, int] = (300, 300),
    ):
        """Render a molecular trajectory aligned to its final frame."""

        molecules = list(molecules)
        if not molecules:
            raise ValueError("molecules must not be empty.")
        final_molecule = molecules[-1]
        AllChem.Compute2DCoords(final_molecule)
        conformer = final_molecule.GetConformer()
        coordinates = [
            conformer.GetAtomPosition(index)
            for index in range(final_molecule.GetNumAtoms())
        ]
        images = []
        for index, molecule in enumerate(molecules):
            if molecule.GetNumAtoms() != final_molecule.GetNumAtoms():
                raise ValueError(
                    "All trajectory molecules must contain the same number of atoms."
                )
            AllChem.Compute2DCoords(molecule)
            frame_conformer = molecule.GetConformer()
            for atom_index, position in enumerate(coordinates):
                frame_conformer.SetAtomPosition(
                    atom_index, Point3D(position.x, position.y, position.z)
                )
            images.append(self.render(molecule, size=size, legend=f"Frame {index}"))
        return images

    @staticmethod
    def save(image, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        image.save(path)
        return path

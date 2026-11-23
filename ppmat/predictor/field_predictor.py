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

import os
import os.path as osp
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from cvve import GridSpec
from tqdm import tqdm

from ppmat.datasets.build_field import BuildField
from ppmat.datasets.build_grid import BuildGrid
from ppmat.datasets.build_molecule import BuildMolecule
from ppmat.datasets.build_structure import BuildStructure
from ppmat.predictor.base import BasePredictor
from ppmat.utils import logger
from ppmat.utils.crystal import atomic_number_from_symbol
from ppmat.utils.io import write_cube


class FieldPredictor(BasePredictor):
    """Field predictor.

    This class provides an interface for predicting scalar fields on molecular or
    crystal grids using pre-trained deep learning models. Supports two initialization
    modes:

    1. **Automatic Model Loading**
       Specify `model_name` and `weights_name` to automatically download
       and load pre-trained weights from the `MODEL_REGISTRY`.

    2. **Custom Model Loading**
       Provide explicit `config_path` and `checkpoint_path` to load
       custom-trained models from local files.

    Args:
        model_name (Optional[str], optional): Name of the pre-defined model architecture
            from the `MODEL_REGISTRY` registry. When specified, associated weights
            will be automatically downloaded. Defaults to None.

        weights_name (Optional[str], optional): Specific pre-trained weight identifier.
            Used only when `model_name` is provided. Valid options include:
            - 'best.pdparams' (highest validation performance)
            - 'latest.pdparams' (most recent training checkpoint)
            - Custom weight files ending with '.pdparams'
            Defaults to None.

        config_path (Optional[str], optional): Path to model configuration file (YAML)
            for custom models. Required when not using predefined `model_name`.
            Defaults to None.
        checkpoint_path (Optional[str], optional): Path to a model checkpoint file
            (.pdparams) for custom models. If omitted, `Predict.checkpoint_path` from
            the config is used. Defaults to None.
    """

    def __init__(
        self,
        model_name: str | None = None,
        weights_name: str | None = None,
        config_path: str | None = None,
        checkpoint_path: str | None = None,
        device: str | None = None,
        config_overrides: Sequence[str] | None = None,
    ):
        super().__init__(
            model_name=model_name,
            weights_name=weights_name,
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            work_dir="",
            device=device,
            config_overrides=config_overrides,
        )
        self.load_inference_model()

    def from_structures(
        self,
        structures,
        grid: GridSpec,
        grid_batch_size: int | None = None,
    ):
        """Predict a field from a structure on an explicit sampling grid."""
        graph = self.graph_converter_fn.from_structure(structures)
        data = {
            "graph": graph,
            "grid": grid,
            "grid_batch_size": int(
                self.predict_config.get("grid_batch_size", 4096)
                if grid_batch_size is None
                else grid_batch_size
            ),
        }
        return self._run_model(data)

    def from_molecule(
        self,
        molecule,
        grid: GridSpec,
        grid_batch_size: int | None = None,
    ):
        """Predict a field from a molecule on an explicit sampling grid."""
        graph = self.graph_converter(molecule)
        data = {
            "graph": graph,
            "grid": grid,
            "grid_batch_size": int(
                self.predict_config.get("grid_batch_size", 4096)
                if grid_batch_size is None
                else grid_batch_size
            ),
        }
        return self._run_model(data)

    def from_cif_file(
        self,
        cif_file_path: str,
        save_path: str | None = None,
        grid_shape: Sequence[int] = (80, 80, 80),
        grid_batch_size: int | None = None,
    ):
        """Predict from one CIF file or every CIF file in a directory.

        Args:
            cif_file_path: Path to a single ``.cif`` file or a directory of
                ``.cif`` files.
            save_path: Optional output directory for predicted CUBE files.
            grid_shape: Number of sampling points along each unit-cell axis.
            grid_batch_size: Maximum grid points processed per forward pass.

        Returns:
            List of prediction dictionaries.
        """
        if osp.isdir(cif_file_path):
            cif_files = sorted(
                osp.join(cif_file_path, file_name)
                for file_name in os.listdir(cif_file_path)
                if file_name.endswith(".cif")
            )
        else:
            cif_files = [cif_file_path]

        results = []
        samples = []
        for cif_file in tqdm(cif_files, desc="Predict"):
            structure = BuildStructure(
                format="cif_file",
                primitive=False,
                niggli=False,
                canocial=False,
            )(cif_file)
            lattice = structure.lattice.matrix.astype("float32")
            grid = BuildGrid(format="array", coordinate_unit="angstrom")(
                {
                    "shape": grid_shape,
                    "voxel_vectors": lattice
                    / np.asarray(grid_shape, dtype=np.float32)[:, None],
                    "periodic": (True, True, True),
                    "cell": lattice,
                }
            )
            results.append(
                self.from_structures(
                    structure,
                    grid,
                    grid_batch_size=grid_batch_size,
                )
            )
            samples.append((cif_file, structure, grid))

        if save_path is not None and results:
            for (cif_file, structure, grid), result in zip(samples, results):
                file_name = osp.splitext(osp.basename(cif_file))[0]
                prediction_path = osp.join(save_path, f"{file_name}_pred.cube")
                write_cube(
                    prediction_path,
                    structure.atomic_numbers,
                    structure.cart_coords,
                    result[self.model.target_name].numpy(),
                    grid,
                )
            logger.info(f"Saved prediction results to: {save_path}")

        return results

    def from_mol_file(
        self,
        mol_file_path: str,
        save_path: str | None = None,
        grid_shape: Sequence[int] = (80, 80, 80),
        grid_padding: float = 6.0,
        grid_batch_size: int | None = None,
    ):
        """Predict from one MOL file or every MOL file in a directory.

        ``save_path`` is an output directory. When provided, predicted CUBE files
        are written there.
        """
        input_path = Path(mol_file_path).expanduser()
        if input_path.is_dir():
            mol_files = sorted(
                path
                for path in input_path.iterdir()
                if path.is_file() and path.suffix.lower() == ".mol"
            )
        else:
            mol_files = [input_path]
        results = []
        samples = []
        for mol_path in tqdm(mol_files, desc="Predict"):
            molecule = BuildMolecule(format="mol_file", sanitize=False)(mol_path)
            grid = BuildGrid(
                format="bounding_box",
                shape=grid_shape,
                padding=grid_padding,
                coordinate_unit="angstrom",
            )(molecule.GetConformer().GetPositions())
            results.append(
                self.from_molecule(
                    molecule,
                    grid,
                    grid_batch_size=grid_batch_size,
                )
            )
            samples.append((mol_path, molecule, grid))

        if save_path is not None and results:
            for (mol_path, molecule, grid), result in zip(samples, results):
                prediction_path = osp.join(save_path, f"{mol_path.stem}_pred.cube")
                write_cube(
                    prediction_path,
                    [atom.GetAtomicNum() for atom in molecule.GetAtoms()],
                    molecule.GetConformer().GetPositions(),
                    result[self.model.target_name].numpy(),
                    grid,
                )
            logger.info(f"Saved prediction results to: {save_path}")

        return results

    def from_xyz_file(
        self,
        xyz_file_path: str,
        save_path: str | None = None,
        grid_shape: Sequence[int] = (80, 80, 80),
        grid_padding: float = 6.0,
        grid_batch_size: int | None = None,
    ):
        """Predict from one XYZ file or every XYZ file in a directory.

        ``save_path`` is an output directory for predicted CUBE files.
        """
        if osp.isdir(xyz_file_path):
            xyz_files = sorted(
                [
                    osp.join(xyz_file_path, file_name)
                    for file_name in os.listdir(xyz_file_path)
                    if file_name.endswith(".xyz")
                ]
            )
        else:
            xyz_files = [xyz_file_path]

        results = []
        samples = []
        for xyz_file in tqdm(xyz_files, desc="Predict"):
            molecule = BuildMolecule(format="xyz_file", sanitize=False)(xyz_file)
            grid = BuildGrid(
                format="bounding_box",
                shape=grid_shape,
                padding=grid_padding,
                coordinate_unit="angstrom",
            )(molecule.GetConformer().GetPositions())
            results.append(
                self.from_molecule(
                    molecule,
                    grid,
                    grid_batch_size=grid_batch_size,
                )
            )
            samples.append((xyz_file, molecule, grid))

        if save_path is not None and results:
            for (xyz_file, molecule, grid), result in zip(samples, results):
                file_name = osp.splitext(osp.basename(xyz_file))[0]
                prediction_path = osp.join(save_path, f"{file_name}_pred.cube")
                write_cube(
                    prediction_path,
                    [atom.GetAtomicNum() for atom in molecule.GetAtoms()],
                    molecule.GetConformer().GetPositions(),
                    result[self.model.target_name].numpy(),
                    grid,
                )
            logger.info(f"Saved prediction results to: {save_path}")

        return results

    def from_field_file(
        self,
        field_file_path: str,
        field_format: str,
        save_path: str | None = None,
        grid_batch_size: int | None = None,
    ):
        """Predict on the structure and grid stored in volumetric field files.

        Args:
            field_file_path: Path to one field file or a directory of files.
            field_format: Field format accepted by :class:`BuildField`.
            save_path: Optional directory for predicted CUBE files.
            grid_batch_size: Maximum grid points processed per forward pass.

        Returns:
            List of prediction dictionaries.
        """
        if osp.isdir(field_file_path):
            field_files = sorted(
                osp.join(field_file_path, file_name)
                for file_name in os.listdir(field_file_path)
                if osp.isfile(osp.join(field_file_path, file_name))
            )
        else:
            field_files = [field_file_path]

        results = []
        samples = []
        for field_file in tqdm(field_files, desc="Predict"):
            field = BuildField(format=field_format, name=self.model.target_name)(
                field_file
            )
            results.append(
                self.from_structures(
                    field.structure,
                    field.grid,
                    grid_batch_size=grid_batch_size,
                )
            )
            samples.append((field_file, field))

        if save_path is not None and results:
            for (field_file, field), result in zip(samples, results):
                file_name = osp.splitext(osp.splitext(osp.basename(field_file))[0])[0]
                prediction_path = osp.join(save_path, f"{file_name}_pred.cube")
                write_cube(
                    prediction_path,
                    [
                        atomic_number_from_symbol(symbol)
                        for symbol in field.structure.symbols
                    ],
                    field.structure.cartesian_positions(),
                    result[self.model.target_name].numpy(),
                    field.grid,
                )
            logger.info(f"Saved prediction results to: {save_path}")

        return results

    def from_cube_file(
        self,
        cube_file_path: str,
        save_path: str | None = None,
        grid_batch_size: int | None = None,
    ):
        """Predict from one CUBE file or a directory of CUBE files."""
        return self.from_field_file(
            cube_file_path,
            "cube",
            save_path,
            grid_batch_size,
        )

    def from_chgcar_file(
        self,
        chgcar_file_path: str,
        save_path: str | None = None,
        grid_batch_size: int | None = None,
    ):
        """Predict from one CHGCAR file or a directory of CHGCAR files."""
        return self.from_field_file(
            chgcar_file_path,
            "chgcar",
            save_path,
            grid_batch_size,
        )

    def from_json_file(
        self,
        json_file_path: str,
        save_path: str | None = None,
        grid_batch_size: int | None = None,
    ):
        """Predict from one density JSON file or a directory of JSON files."""
        return self.from_field_file(
            json_file_path,
            "json",
            save_path,
            grid_batch_size,
        )

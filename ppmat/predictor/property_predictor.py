# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.

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
import os.path as osp
from collections import defaultdict
from collections.abc import Sequence

import pandas as pd
from tqdm import tqdm

from ppmat.datasets.build_molecule import BuildMolecule
from ppmat.datasets.build_structure import BuildStructure
from ppmat.predictor.base import BasePredictor
from ppmat.utils import logger


class PropertyPredictor(BasePredictor):
    """Property predictor.

    This class provides an interface for predicting properties of crystal structures and
    molecules using pre-trained deep learning models. Supports two initialization
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

    def from_structures(self, structures):
        data = self.graph_converter(structures)
        return self._run_model(data)

    def from_molecule(self, molecule):
        data = self.graph_converter(molecule)
        return self._run_model(data)

    def from_cif_file(self, cif_file_path, save_path=None):
        """Predict crystal properties from CIF file(s).

        Args:
            cif_file_path: Path to a single ``.cif`` file or a directory
                of ``.cif`` files.
            save_path: Optional CSV path.

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
        for cif_file in tqdm(cif_files, desc="Predict"):
            structure = BuildStructure(
                format="cif_file",
                primitive=False,
                niggli=False,
                canocial=False,
            )(cif_file)
            results.append(self.from_structures(structure))

        if save_path is not None and results:
            keys = list(results[0].keys())
            result_properties = defaultdict(list)
            for key in keys:
                for result in results:
                    result_properties[key].append(result[key])

            df = pd.DataFrame({"cif_file": cif_files, **result_properties})
            df.to_csv(save_path, index=False)
            logger.info(f"Saved the prediction result to {save_path}")

        return results

    def from_xyz_file(self, xyz_file_path, save_path=None):
        """Predict molecular properties from XYZ file(s).

        Builds each ``.xyz`` file with :class:`BuildMolecule`, then delegates
        to :meth:`from_molecule`.

        Args:
            xyz_file_path: Path to a single ``.xyz`` file or a directory
                of ``.xyz`` files.
            save_path: Optional CSV path.

        Returns:
            List of prediction dictionaries.
        """
        if save_path is not None:
            assert save_path.endswith(".csv"), "save_path must end with .csv"

        if osp.isdir(xyz_file_path):
            xyz_files = sorted(
                [
                    osp.join(xyz_file_path, f)
                    for f in os.listdir(xyz_file_path)
                    if f.endswith(".xyz")
                ]
            )
        else:
            xyz_files = [xyz_file_path]

        results = []
        for xyz_path in tqdm(xyz_files, desc="Predict"):
            molecule = BuildMolecule(format="xyz_file", sanitize=False)(xyz_path)
            result = self.from_molecule(molecule)
            results.append(result)

        if save_path is not None and results:
            keys = list(results[0].keys())
            props = defaultdict(list)
            for key in keys:
                for result in results:
                    props[key].append(result[key])
            df = pd.DataFrame(
                {"xyz_file": [osp.basename(path) for path in xyz_files], **props}
            )
            df.to_csv(save_path, index=False)
            logger.info(f"Saved prediction results to {save_path}")

        return results

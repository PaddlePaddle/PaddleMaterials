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
from typing import Optional
from typing import Sequence
from typing import Union

import paddle
import pandas as pd
from omegaconf import OmegaConf
from pymatgen.core import Structure
from tqdm import tqdm

from ppmat.datasets.transform import build_post_transforms
from ppmat.models import build_graph_converter
from ppmat.models import build_model
from ppmat.models import build_model_from_name
from ppmat.utils import logger
from ppmat.utils import save_load
from ppmat.utils.download import is_url
from ppmat.utils.execution import configure_execution_backend
from ppmat.vocab import build_vocab

PathLike = Union[str, os.PathLike]


class BasePredictor:
    """Common model loading and runtime behavior for predictors.

    Supports two initialization modes:

    1. **Automatic Model Loading**
       Specify `model_name` and `weights_name` to automatically download
       and load pre-trained weights from the `MODEL_REGISTRY`.

    2. **Custom Model Loading**
       Provide explicit `config_path` and `checkpoint_path` to load
       custom-trained models from local files.

    Args:
        model_name (Optional[str], optional):
            Name of the pre-defined model architecture
            from the `MODEL_REGISTRY` registry.
            When specified, associated weights
            will be automatically downloaded. Defaults to None.

        weights_name (Optional[str], optional):
            Specific pre-trained weight identifier.
            Used only when `model_name` is provided. Valid options include:
            - 'best.pdparams' (highest validation performance)
            - 'latest.pdparams' (most recent training checkpoint)
            - Custom weight files ending with '.pdparams'
            Defaults to None.

        config_path (Optional[str], optional):
            Path to model configuration file (YAML)
            for custom models. Required when not using predefined `model_name`.
            Defaults to None.

        checkpoint_path (Optional[str], optional):
            Path to a model checkpoint file (.pdparams) for custom models. If omitted,
            `Predict.checkpoint_path` from the config is used. Defaults to None.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        weights_name: Optional[str] = None,
        config_path: Optional[PathLike] = None,
        checkpoint_path: Optional[PathLike] = None,
        work_dir: Optional[PathLike] = None,
        device: Optional[str] = None,
        config_overrides: Optional[Sequence[str]] = None,
    ):
        work_dir = work_dir or ""
        self.work_dir = work_dir
        self.model_name = model_name
        self.device = device
        self.config_path = self._resolve_work_path(config_path)
        self.checkpoint_path = self._resolve_work_path(checkpoint_path)
        self.weights_name = weights_name
        self.config_overrides = config_overrides

    def _resolve_work_path(self, path: Optional[PathLike]) -> Optional[str]:
        if path is None:
            return path
        path = os.fspath(path)
        if osp.isabs(path) or is_url(path):
            return path
        return osp.join(self.work_dir, path)

    def load_inference_model(self, interface_type: Optional[str] = None):
        if self.device is not None:
            paddle.set_device(self.device)
        self.device = paddle.get_device()
        if self.model_name is None:
            if self.config_path is None:
                raise ValueError("config_path is required when model_name is not set.")
            if self.weights_name is not None:
                raise ValueError("weights_name can only be used with model_name.")

            config = OmegaConf.load(self.config_path)
            if self.config_overrides:
                config = OmegaConf.merge(
                    config, OmegaConf.from_dotlist(self.config_overrides)
                )
            config = OmegaConf.to_container(config, resolve=True)

            checkpoint_path = self.checkpoint_path
            if checkpoint_path is None:
                checkpoint_path = (config.get("Predict") or {}).get("checkpoint_path")
                checkpoint_path = self._resolve_work_path(checkpoint_path)
            if checkpoint_path is None:
                raise ValueError(
                    "checkpoint_path is required when the config does not define "
                    "Predict.checkpoint_path."
                )
            logger.info(f"Loading model weights from {checkpoint_path}.")
            model_config = config.get("Model", None)
            assert model_config is not None, "Model config must be provided."
            if interface_type:
                model_config = self.modify_model_config(interface_type, model_config)
            else:
                logger.info("No interface, use the model directly")
            vocab = build_vocab(config.get("Vocabulary"))
            model = build_model(model_config, vocab=vocab)
            save_load.load_pretrain(model, checkpoint_path)
        else:
            if self.config_path is not None or self.checkpoint_path is not None:
                raise ValueError(
                    "config_path and checkpoint_path cannot be combined with "
                    "model_name."
                )
            unsupported_overrides = []
            for item in self.config_overrides or ():
                key = item.split("=", 1)[0]
                root = key.split(".", 1)[0]
                if root not in {"Predict", "Dataset", "Execution"}:
                    unsupported_overrides.append(item)
            if unsupported_overrides:
                raise ValueError(
                    "Registered-model overrides only support Predict.*, Dataset.*, "
                    "and Execution.*; use config_path and checkpoint_path for other "
                    f"changes. Unsupported overrides: {unsupported_overrides}."
                )
            logger.info("Since model_name is given, downloading it...")
            if interface_type:
                model, config = build_model_from_name(
                    self.model_name,
                    self.weights_name,
                    model_config_modifier=lambda model_config: (
                        self.modify_model_config(interface_type, model_config)
                    ),
                )
            else:
                model, config = build_model_from_name(
                    self.model_name, self.weights_name
                )
            if self.config_overrides:
                config = OmegaConf.merge(
                    OmegaConf.create(config),
                    OmegaConf.from_dotlist(self.config_overrides),
                )
                config = OmegaConf.to_container(config, resolve=True)
            if not interface_type:
                logger.info("No interface, use the model directly")

        self.model = model
        self.config = config
        self.vocab = build_vocab(config.get("Vocabulary"))

        self.predict_config = config.get("Predict") or {}
        execution_config = config.get("Execution") or {}
        self.execution_backend = configure_execution_backend(
            self.model,
            execution_config.get("backend"),
            init_params=execution_config.get("__init_params__"),
            world_size=paddle.distributed.get_world_size(),
            owner="Predictor",
        )
        # Backend selection may replace the callable used for the forward pass.
        # Apply eval mode after that hook so the runtime sees the same contract.
        self.model.eval()

        self.eval_with_no_grad = self.predict_config.get("eval_with_no_grad", True)

        self.graph_converter_fn = None
        graph_converter_config = self.predict_config.get("graph_converter")
        if graph_converter_config is not None:
            self.graph_converter_fn = build_graph_converter(
                graph_converter_config, vocab=self.vocab
            )

        self.post_transforms_cfg = self.predict_config.get("post_transforms")
        if self.post_transforms_cfg is not None:
            self.post_transforms = build_post_transforms(self.post_transforms_cfg)
        else:
            self.post_transforms = None

    def modify_model_config(
        self,
        interface_type,
        model_config,
    ):
        # TODO: support more models
        if interface_type == "ase":
            logger.info("Integrate ASE calculator")
            if model_config["__class_name__"] == "CHGNet":
                # CHGNet by default predicts energy per atom;
                # convert it to total energy
                model_config["__init_params__"]["is_intensive"] = False
                logger.warning(
                    "CHGNet by default predicts energy per atom; "
                    "change 'is_intensive' to False to "
                    "predict total energy for ASE integration."
                )
            elif model_config["__class_name__"] == "M3GNet":
                pass
            else:
                raise NotImplementedError(
                    f"The model '{model_config.get('__class_name__')}' "
                    f"is not yet supported with ASE integration.\n"
                    f"Please ensure that the model predicts total energy, "
                    f"or manually adjust parameter according to the model.\n"
                    f"If this model should be supported, "
                    f"please add a special handling case here."
                )
        elif interface_type == "lammps":
            pass
        return model_config

    def collect_structures(
        self,
        file_path: str,
    ):
        """
        pymatgen.core.Structure supported formats include:
            CIF, POSCAR/CONTCAR, CHGCAR, LOCPOT, vasprun.xml, CSSR,
            Netcdf and pymatgen's JSON-serialized structures.

        Args:
            file_path (str):
                The path of the input file or directory.
        """
        if osp.isdir(file_path):
            all_files = [
                osp.join(file_path, f)
                for f in os.listdir(file_path)
                if f.endswith(".cif")
            ]
        else:
            all_files = [file_path]
        logger.info(f"Load {len(all_files)} structures from {file_path}")

        # Read raw file
        structures = []
        for file in tqdm(all_files):
            try:
                # read file by pymatgen package
                structure = Structure.from_file(file)
                structures.append(structure)
            except Exception as e:
                logger.warning("Error reading file: {}, skip it.\n{}".format(file, e))
        logger.info("Successfully read raw files and convert to pymatgen format")
        return all_files, structures

    def graph_converter(self, structure):
        if self.graph_converter_fn is None:
            return structure
        return self.graph_converter_fn(structure)

    def post_process(self, data):
        if self.post_transforms is None:
            return data
        return self.post_transforms(data)

    def _run_model(self, data):
        if self.eval_with_no_grad:
            with paddle.no_grad():
                output = self.model.predict(data)
        else:
            output = self.model.predict(data)
        return self.post_process(output)

    def get_predict(
        self,
        files: list,
        structures: list,
    ):
        results = []
        for structure in tqdm(structures):
            data = self.graph_converter(structure)
            data = data.tensor()
            results.append(self._run_model(data))

        # save file names and output to csv file
        if not results:
            raise ValueError("No results to save csv file.")
        df = pd.DataFrame(results)
        df.insert(0, "file_name", files)
        df.to_csv("results_pred_property.csv", index=False)
        logger.info("Saved the prediction results.")

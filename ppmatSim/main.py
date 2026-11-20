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

from __future__ import annotations

import os

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
from omegaconf import OmegaConf

from ppmat.predictor.structures import build_init_structures
from ppmat.utils import logger


@hydra.main(config_path="configs", version_base=None)
def main(cfg: DictConfig):
    # Save the loaded config
    OmegaConf.save(cfg, "config_saved.yaml")

    # Initialize logger
    log_file = cfg.Logger.get("log_file", "out.log")
    logger.init_logger(log_file=log_file, log_level=cfg.get("log_level", "INFO"))
    logger.info("[PPMaterial] Logger initialized")
    logger.info(f"Working directory: {os.getcwd()}")
    logger.info(f"Log file path    : {os.path.abspath(log_file)}")

    # Initialize the model
    load_model = instantiate(cfg.Model)
    if cfg.get("Predictor") is None:
        from ppmat.predictor import BasePredictor

        predictor = BasePredictor(
            work_dir=cfg.Run.work_dir, device=cfg.device, **load_model
        )
    else:
        predictor = instantiate(
            cfg.Predictor,
            work_dir=cfg.Run.work_dir,
            device=cfg.device,
            **load_model,
        )

    # Detect interface type and interface object
    if cfg.get("Calculator") is not None:
        # Read interface type
        interface_type = cfg.Calculator.get("type", None)
        logger.info(f"Interface type is {interface_type}")
        # Load inference model
        predictor.load_inference_model(interface_type=interface_type)
        # Initialize the interface object
        interface_obj = instantiate(cfg.Calculator, predictor=predictor)
        logger.info(f"Interface object is {interface_obj}")
    else:
        # Load inference model
        predictor.load_inference_model(interface_type=None)

    # Load structures
    files, structures = build_init_structures(cfg, predictor)

    if cfg.get("Task") is not None:
        # Initialize the task
        task = instantiate(cfg.Task)
        # Run the task
        task(interface_obj, structures)
    else:
        predictor.get_predict(files, structures)

    logger.info("All tasks finished successfully.")


if __name__ == "__main__":
    main()

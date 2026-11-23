# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

import argparse
import os
import os.path as osp
from typing import Any
from typing import Dict

import paddle.distributed as dist
import paddle.distributed.fleet as fleet
from omegaconf import OmegaConf

from ppmat.datasets import build_dataloader
from ppmat.datasets import set_signal_handlers
from ppmat.models import build_model
from ppmat.optimizer import build_optimizer
from ppmat.trainer.base_trainer import BaseTrainer
from ppmat.utils import logger
from ppmat.utils import misc
from ppmat.utils.eager_comp_setting import setting_eager_mode
from ppmat.utils.io import append_timestamp_to_output_dir


def read_independent_dataloader_config(config: Dict[str, Any]):
    if config["Global"].get("do_train", True):
        train_data_cfg = config["Dataset"].get("train")
        assert (
            train_data_cfg is not None
        ), "train_data_cfg must be defined when Global.do_train is True"
        train_loader = build_dataloader(train_data_cfg)
    else:
        train_loader = None

    if config["Global"].get("do_eval", False) or config["Global"].get("do_train", True):
        val_data_cfg = config["Dataset"].get("val")
        if val_data_cfg is not None:
            val_loader = build_dataloader(val_data_cfg)
        else:
            logger.info("No validation dataset defined.")
            val_loader = None
    else:
        val_loader = None

    if config["Global"].get("do_test", False):
        test_data_cfg = config["Dataset"].get("test")
        assert (
            test_data_cfg is not None
        ), "test_data_cfg must be defined when Global.do_test is True"
        test_loader = build_dataloader(test_data_cfg)
    else:
        test_loader = None
    return train_loader, val_loader, test_loader


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case",
        type=str,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="./spectrum_enhancement/configs/sfin/sfin_haadf_enhance.yaml",
        help="Path to config file.",
    )
    return parser.parse_known_args()


def main():
    if dist.get_world_size() > 1:
        fleet.init(is_collective=True)

    args, dynamic_args = parse_args()

    cfg = OmegaConf.load(args.config)
    cli_cfg = OmegaConf.from_dotlist(dynamic_args)
    cfg = OmegaConf.merge(cfg, cli_cfg)

    append_timestamp_to_output_dir(cfg)

    if dist.get_rank() == 0:
        os.makedirs(cfg["Trainer"]["output_dir"], exist_ok=True)
        config_name = os.path.basename(args.config)
        OmegaConf.save(cfg, osp.join(cfg["Trainer"]["output_dir"], config_name))

    config = OmegaConf.to_container(cfg, resolve=True)

    logger_path = osp.join(config["Trainer"]["output_dir"], "run.log")
    logger.init_logger(log_file=logger_path)
    logger.info(f"Logger saved to {logger_path}")

    seed = config["Trainer"].get("seed", 42)
    misc.set_random_seed(seed)
    logger.info(f"Set random seed to {seed}")

    enabled = config["Global"].get("prim_eager_enabled", False)
    white_list = config["Global"].get("prim_backward_white_list", None)
    setting_eager_mode(enabled, white_list)

    model_cfg = config["Model"]

    set_signal_handlers()
    if config["Dataset"].get("split_dataset_ratio") is not None:
        loader = build_dataloader(config["Dataset"])
        train_loader = loader.get("train", None)
        val_loader = loader.get("val", None)
        test_loader = loader.get("test", None)
    else:
        train_loader, val_loader, test_loader = read_independent_dataloader_config(
            config
        )

    model = build_model(model_cfg)

    if config.get("Optimizer") is not None and config["Global"].get("do_train", True):
        assert (
            train_loader is not None
        ), "train_loader must be defined when Optimizer is provided."
        assert (
            config["Trainer"].get("max_epochs") is not None
        ), "Trainer.max_epochs must be defined when Optimizer is provided."
        optimizer, lr_scheduler = build_optimizer(
            config["Optimizer"],
            model,
            config["Trainer"]["max_epochs"],
            len(train_loader),
        )
    else:
        optimizer, lr_scheduler = None, None

    metric_cfg = config.get("Metric")

    trainer = BaseTrainer(
        config["Trainer"],
        model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        compute_metric_func_dict=None,
        execution_config=config.get("Execution"),
    )
    trainer.attach_metrics(metric_cfg, model=model)

    if config["Global"].get("do_train", True):
        trainer.train()
    if config["Global"].get("do_eval", False):
        logger.info("Evaluating on validation set")
        trainer.eval(val_loader)
    if config["Global"].get("do_test", False):
        logger.info("Evaluating on test set")
        trainer.eval(test_loader)


if __name__ == "__main__":
    main()

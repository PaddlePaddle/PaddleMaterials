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

import argparse
import os
import os.path as osp

import paddle.distributed as dist
import paddle.distributed.fleet as fleet
from omegaconf import OmegaConf

from ppmat.datasets import build_dataloader
from ppmat.datasets import build_dataset_infos
from ppmat.datasets import set_signal_handlers
from ppmat.models import build_model
from ppmat.optimizer import build_optimizer
from ppmat.trainer.base_trainer import BaseTrainer
from ppmat.utils import logger
from ppmat.utils import misc
from ppmat.utils.io import append_timestamp_to_output_dir
from ppmat.vocab import build_vocab


def read_independent_dataloader_config(config, vocab=None):
    do_train = config["Global"].get("do_train", True)
    do_eval = config["Global"].get("do_eval", False)
    do_test = config["Global"].get("do_test", False)
    if not any((do_train, do_eval, do_test)):
        raise ValueError(
            "At least one of Global.do_train, Global.do_eval, or Global.do_test "
            "must be True."
        )

    # DiffNMR dataset statistics and novelty metrics use the training SMILES even
    # during evaluation-only and test-only runs.
    train_data_cfg = config["Dataset"].get("train")
    assert (
        train_data_cfg is not None
    ), "train_data_cfg must be defined for DiffNMR train/eval/test."
    train_loader = build_dataloader(train_data_cfg, vocab=vocab)

    if do_eval or do_train:
        val_data_cfg = config["Dataset"].get("val")
        if val_data_cfg is None and do_eval:
            raise ValueError("val_data_cfg must be defined when do_eval is true.")
        if val_data_cfg is None:
            logger.info("No validation dataset defined.")
            val_loader = None
        else:
            val_loader = build_dataloader(val_data_cfg, vocab=vocab)
    else:
        val_loader = None

    if do_test:
        test_data_cfg = config["Dataset"].get("test")
        assert (
            test_data_cfg is not None
        ), "test_data_cfg must be defined, when do_test is true"
        test_loader = build_dataloader(test_data_cfg, vocab=vocab)
    else:
        test_loader = None
    return train_loader, val_loader, test_loader


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="./spectrum_elucidation/configs/diffnmr/DiffNMR.yaml",
        help="Path to config file",
    )
    return parser.parse_known_args()


if __name__ == "__main__":
    if dist.get_world_size() > 1:
        fleet.init(is_collective=True)

    args, dynamic_args = parse_args()

    # load config and merge with cli args
    config = OmegaConf.load(args.config)
    cli_config = OmegaConf.from_dotlist(dynamic_args)
    config = OmegaConf.merge(config, cli_config)

    seed = config["Trainer"].get("seed", 42)
    append_timestamp_to_output_dir(config)

    # save config to output_dir, only rank 0 process will do this
    if dist.get_rank() == 0:
        os.makedirs(config["Trainer"]["output_dir"], exist_ok=True)
        config_name = os.path.basename(args.config)
        OmegaConf.save(config, osp.join(config["Trainer"]["output_dir"], config_name))
    # convert config to dict
    config = OmegaConf.to_container(config, resolve=True)

    # init logger
    logger_path = osp.join(config["Trainer"]["output_dir"], "run.log")
    logger.init_logger(log_file=logger_path)
    logger.info(f"Logger saved to {logger_path}")

    # set random seed
    misc.set_random_seed(seed)
    logger.info(f"Set random seed to {seed}")

    vocab = build_vocab(config.get("Vocabulary"))

    # load dataloader from config
    set_signal_handlers()
    train_loader, val_loader, test_loader = read_independent_dataloader_config(
        config, vocab=vocab
    )

    # build datasetinfo
    dataloaders = {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
    }
    dataset_infos = build_dataset_infos(
        dataloaders=dataloaders,
        cfg=config,
        vocab=vocab,
        recompute_statistics=False,
    )
    # Optional models are built independently and injected through model config.
    if config.get("CLIP", None) is not None:
        clip_module = build_model(
            config["CLIP"],
            vocab=vocab,
            dataset_infos=dataset_infos,
        )
    else:
        clip_module = None

    # build model from config
    model_cfg = config["Model"]
    model = build_model(
        model_cfg,
        vocab=vocab,
        dataset_infos=dataset_infos,
        clip=clip_module,
    )

    # build optimizer and learning rate scheduler from config
    if config.get("Optimizer") is not None and config["Global"].get("do_train", True):
        assert (
            train_loader is not None
        ), "train_loader must be defined when optimizer is defined."
        assert (
            config["Trainer"].get("max_epochs") is not None
        ), "max_epochs must be defined when optimizer is defined."
        optimizer, lr_scheduler = build_optimizer(
            config["Optimizer"],
            model,
            config["Trainer"]["max_epochs"],
            len(train_loader),
        )
    else:
        optimizer, lr_scheduler = None, None

    metric_cfg = config.get("Metric")

    # initialize trainer
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

    trainer.attach_metrics(
        metric_cfg,
        dataset_infos=dataset_infos,
        clip=clip_module,
        model=model,
    )

    if config["Global"].get("do_train", True):
        trainer.train()
    if config["Global"].get("do_eval", False):
        logger.info("Evaluating on validation set")
        time_info, loss_info, metric_info = trainer.eval(val_loader)
    if config["Global"].get("do_test", False):
        logger.info("Evaluating on test set")
        time_info, loss_info, metric_info = trainer.eval(test_loader)

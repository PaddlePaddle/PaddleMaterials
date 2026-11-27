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

import copy
from typing import Dict
from typing import Tuple

import paddle
import paddle.nn as nn

from ppmat.optimizer import lr_scheduler
from ppmat.optimizer.optimizer import LBFGS
from ppmat.optimizer.optimizer import SGD
from ppmat.optimizer.optimizer import Adam
from ppmat.optimizer.optimizer import AdamW
from ppmat.optimizer.optimizer import Momentum
from ppmat.optimizer.optimizer import OptimizerList
from ppmat.optimizer.optimizer import RMSProp

__all__ = [
    "LBFGS",
    "SGD",
    "Adam",
    "AdamW",
    "Momentum",
    "RMSProp",
    "OptimizerList",
    "lr_scheduler",
]


def build_lr_scheduler(cfg: Dict, epochs: int, iters_per_epoch: int):
    """Build learning rate scheduler.

    Args:
        cfg (Dict): Learning rate scheduler config.
        epochs (int): Total epochs.
        iters_per_epoch (int): Number of iterations of one epoch.

    Returns:
        LRScheduler: Learning rate scheduler.
    """
    cfg = copy.deepcopy(cfg)
    cfg.update({"epochs": epochs, "iters_per_epoch": iters_per_epoch})
    lr_scheduler_cls = cfg.pop("__class_name__")
    init_params = cfg.pop("__init_params__")
    lr_scheduler_ = getattr(lr_scheduler, lr_scheduler_cls)(
        epochs=epochs, iters_per_epoch=iters_per_epoch, **init_params
    )
    return lr_scheduler_()


def build_optimizer(
    cfg: Dict, model_list: Tuple[nn.Layer, ...], epochs: int, iters_per_epoch: int
):
    """Build optimizer and learning rate scheduler

    Args:
        cfg (Dict): Learning rate scheduler config.
        model_list (Tuple[nn.Layer, ...]): Tuple of model(s).
        epochs (int): Total epochs.
        iters_per_epoch (int): Number of iterations of one epoch.

    Returns:
        Optimizer, LRScheduler: Optimizer and learning rate scheduler.

    ``groups`` opts into named Adam optimizers on sublayers of one model.
    Each group adds ``model_attr`` to a regular Adam config with a validation
    ReduceOnPlateau scheduler. Set ``clip_norm_global`` on the outer config to
    clip all groups together before any optimizer updates its parameters.
    Groups are saved in name order and must partition the trainable parameters.
    """
    if "groups" in cfg:
        return _build_grouped_optimizer(cfg, model_list, epochs, iters_per_epoch)

    # build lr_scheduler
    cfg = copy.deepcopy(cfg)
    init_params = cfg.pop("__init_params__")
    lr_cfg = init_params.pop("lr")
    if isinstance(lr_cfg, float):
        lr_scheduler = lr_cfg
    else:
        lr_scheduler = build_lr_scheduler(lr_cfg, epochs, iters_per_epoch)

    # build optimizer
    opt_cls = cfg.pop("__class_name__")
    if "clip_norm" in cfg:
        clip_norm = cfg.pop("clip_norm")
        grad_clip = paddle.nn.ClipGradByNorm(clip_norm=clip_norm)
    elif "clip_norm_global" in cfg:
        clip_norm = cfg.pop("clip_norm_global")
        grad_clip = paddle.nn.ClipGradByGlobalNorm(clip_norm=clip_norm)
    elif "clip_value" in cfg:
        clip_value = cfg.pop("clip_value")
        grad_clip = paddle.nn.ClipGradByValue(clip_value)
    else:
        grad_clip = None

    optimizer = eval(opt_cls)(
        learning_rate=lr_scheduler, grad_clip=grad_clip, **init_params
    )(model_list)

    if isinstance(lr_scheduler, float):
        return optimizer, None
    return optimizer, lr_scheduler


def _build_grouped_optimizer(cfg, model, epochs, iters_per_epoch):
    if set(cfg) - {"groups", "clip_norm_global"}:
        raise ValueError("Grouped optimizers only support outer clip_norm_global.")
    if isinstance(model, (list, tuple)) and len(model) == 1:
        model = model[0]
    if not isinstance(model, nn.Layer):
        raise ValueError("Grouped optimizers require a single model.")
    if not cfg["groups"]:
        raise ValueError("Optimizer.groups must contain at least one group.")

    optimizers, schedulers = [], []
    parameters = {}
    group_names = tuple(sorted(cfg["groups"]))
    for name in group_names:
        group = cfg["groups"][name]
        layer = getattr(model, group["model_attr"], None)
        if not isinstance(layer, nn.Layer):
            raise ValueError(f"Optimizer group {name!r} must select a model sublayer.")
        if group.get("__class_name__") != "Adam":
            raise ValueError(f"Optimizer group {name!r} must use Adam.")
        if {"clip_norm", "clip_norm_global", "clip_value"} & group.keys():
            raise ValueError("Only outer clip_norm_global is supported for groups.")
        group_parameters = {id(param): param for param in layer.parameters()}
        if parameters.keys() & group_parameters.keys():
            raise ValueError("A parameter cannot belong to multiple optimizer groups.")
        parameters.update(group_parameters)
        child_cfg = {key: value for key, value in group.items() if key != "model_attr"}
        optimizer, scheduler = build_optimizer(
            child_cfg, layer, epochs, iters_per_epoch
        )
        if not isinstance(
            scheduler, paddle.optimizer.lr.ReduceOnPlateau
        ) or scheduler.indicator not in ("eval_loss", "eval_metric"):
            raise ValueError(
                f"Optimizer group {name!r} requires a validation ReduceOnPlateau."
            )
        optimizers.append(optimizer)
        schedulers.append(scheduler)

    if any(
        not param.stop_gradient and id(param) not in parameters
        for param in model.parameters()
    ):
        raise ValueError("Optimizer groups must cover all trainable parameters.")

    clip_norm = cfg.get("clip_norm_global")
    clip_parameters = tuple(parameters.values()) if clip_norm is not None else ()
    return (
        OptimizerList(
            tuple(optimizers),
            group_names=group_names,
            clip_norm_global=clip_norm,
            clip_parameters=clip_parameters,
        ),
        lr_scheduler.SchedulerList(tuple(schedulers), by_epoch=True),
    )

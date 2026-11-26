# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""MatInvent reinforcement-learning trainer.

One epoch is one RL round: sample structures, score them, select the top-k and
fine-tune the diffusion backbone. The checkpoint lifecycle, the optimizer, the
learning-rate schedule, logging and validation all stay in ``BaseTrainer``.
"""

import copy
import os
import time

import paddle.distributed as dist

from ppmat.datasets.collate_fn import DefaultCollator
from ppmat.datasets.transform import build_transforms
from ppmat.models import MODEL_REGISTRY
from ppmat.models.matinvent.dataset import to_tensor_batch
from ppmat.models.matinvent.qualification import PopulationQualifier
from ppmat.models.matinvent.rewards.reward import Reward
from ppmat.models.matinvent.rl_loop import MatInventRL
from ppmat.models.matinvent.sampler import MatInventSampler
from ppmat.trainer.base_trainer import BaseTrainer
from ppmat.utils import AverageMeter
from ppmat.utils import download
from ppmat.utils import logger
from ppmat.utils import save_load
from ppmat.utils.model_package import resolve_model_package_dir


class MatInventTrainer(BaseTrainer):
    """Round-based trainer for MatInvent RL fine-tuning.

    Args:
        config (dict): The ``Trainer`` block of the task config.
        model (nn.Layer): The diffusion backbone to fine-tune.
        train_dataloader (paddle.io.DataLoader): Anchor structures; the dataset must
            produce the framework ``structure_array`` contract.
        val_dataloader (paddle.io.DataLoader, optional): Validation structures, also
            used as the fixed early-stop probe.
        optimizer (paddle.optimizer.Optimizer): Fine-tuning optimizer.
        lr_scheduler: Learning rate scheduler built from the same ``Optimizer`` block.
        pretrained_model_name (str, optional): Registered package that initializes both
            the agent and the frozen prior.
        topk_ratio (float): Fraction of ``finetune_cfg.batch_size`` kept per round.
        sample_cfg (dict): Structure sampling configuration.
        finetune_cfg (dict): Reward-weighted fine-tuning configuration.
        reward_cfg (dict): Property reward configuration.
        anchor_cfg (dict): Denoising-anchor configuration.
        replay_cfg (dict, optional): Replay buffer configuration.
        div_filter_cfg (dict, optional): Diversity filter configuration.
        early_stop_cfg (dict): Probe-based early stopping configuration.
        qualification_cfg (dict, optional): Stability qualification configuration.
        build_structure_cfg (dict): Structure parser used to score samples.
        transforms (list, optional): Lattice parameterization shared with the dataset.
    """

    def __init__(
        self,
        config,
        model,
        train_dataloader=None,
        val_dataloader=None,
        optimizer=None,
        lr_scheduler=None,
        compute_metric_func_dict=None,
        execution_config=None,
        pretrained_model_name=None,
        topk_ratio=0.5,
        sample_cfg=None,
        finetune_cfg=None,
        reward_cfg=None,
        anchor_cfg=None,
        replay_cfg=None,
        div_filter_cfg=None,
        early_stop_cfg=None,
        qualification_cfg=None,
        build_structure_cfg=None,
        transforms=None,
    ):
        if dist.get_world_size() > 1:
            raise ValueError(
                "MatInvent RL samples and scores structures per process, so it "
                "supports a single process only."
            )
        if config.get("use_amp", False):
            raise ValueError(
                "MatInvent RL keeps the diffusion sampling path in float32 and does "
                "not support use_amp."
            )
        if train_dataloader is None:
            raise ValueError("MatInvent RL needs a training dataloader for anchors")
        if optimizer is None:
            raise ValueError(
                "MatInvent RL needs the Optimizer block of the task config"
            )
        if (
            config.get("pretrained_model_path") is not None
            and pretrained_model_name is not None
        ):
            raise ValueError(
                "Set either Trainer.pretrained_model_path or "
                "Trainer.__init_params__.pretrained_model_name, not both."
            )

        super().__init__(
            config,
            model,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            compute_metric_func_dict=compute_metric_func_dict,
            execution_config=execution_config,
        )

        if self.lr_scheduler is not None and not self.lr_scheduler.by_epoch:
            raise ValueError(
                "MatInvent RL steps the learning-rate schedule once per round, so "
                "the Optimizer lr schedule must set by_epoch=True."
            )
        if pretrained_model_name is not None:
            save_load.load_pretrain(
                self.model,
                self._resolve_package(pretrained_model_name),
                "best.pdparams",
            )

        self.prior = copy.deepcopy(self.model)
        for parameter in self.prior.parameters():
            parameter.stop_gradient = True
        self.prior.eval()

        self.anchor_dataset = self.train_dataloader.dataset
        anchor_batch = self._build_batch(
            self.anchor_dataset, int(anchor_cfg["num_samples"]), "Anchor batch"
        )
        probe_dataset = (
            self.val_dataloader.dataset
            if self.val_dataloader is not None
            else self.anchor_dataset
        )
        probe_batch = self._build_batch(
            probe_dataset, int(early_stop_cfg["probe_num_samples"]), "Probe batch"
        )

        reward = Reward(
            root_dir=os.path.join(self.output_dir, "rewards"),
            prop_cfg=reward_cfg["prop_cfg"],
            reward_threshold=reward_cfg["reward_threshold"],
            reduce=reward_cfg["reduce"],
        )
        sampler = MatInventSampler(
            batch_size=int(sample_cfg["batch_size"]),
            num_batches=int(sample_cfg["num_batches"]),
            num_inference_steps=int(sample_cfg["num_inference_steps"]),
            build_structure_cfg=build_structure_cfg,
            atom_count_distribution=sample_cfg["atom_count_distribution"],
        )
        qualifier = (
            PopulationQualifier(**qualification_cfg)
            if qualification_cfg is not None
            else None
        )

        self.rl_loop = MatInventRL(
            agent=self.model,
            prior=self.prior,
            sampler=sampler,
            reward=reward,
            optimizer=self.optimizer,
            sample_cfg=sample_cfg,
            finetune_cfg=finetune_cfg,
            topk_ratio=topk_ratio,
            anchor_batch=anchor_batch,
            anchor_dataset=self.anchor_dataset,
            anchor_cfg=anchor_cfg,
            probe_batch=probe_batch,
            replay_cfg=replay_cfg,
            div_filter_cfg=div_filter_cfg,
            early_stop_cfg=early_stop_cfg,
            qualifier=qualifier,
            transforms=build_transforms(transforms),
            save_dir=self.output_dir,
        )
        self.rl_state_path = os.path.join(self.output_dir, "rl_state.pkl")
        self.rl_state_loaded = False
        self.stop_training = False

    @staticmethod
    def _resolve_package(model_name):
        if model_name not in MODEL_REGISTRY:
            raise ValueError(
                f"{model_name} is not registered in MODEL_REGISTRY, "
                f"available: {sorted(MODEL_REGISTRY)}"
            )
        extracted_path = download.get_weights_path_from_url(MODEL_REGISTRY[model_name])
        return resolve_model_package_dir(model_name, extracted_path)

    @staticmethod
    def _build_batch(dataset, num_samples, label):
        num_samples = min(num_samples, len(dataset))
        if num_samples <= 0:
            raise ValueError(f"{label}: num_samples must select at least one sample")
        batch = DefaultCollator()([dataset[index] for index in range(num_samples)])
        logger.info(f"{label}: {num_samples} samples")
        return to_tensor_batch(batch)

    def train_epoch(self, dataloader, val_dataloader=None):
        if not self.rl_state_loaded:
            if os.path.exists(self.rl_state_path):
                logger.info(f"Resuming MatInvent RL state from {self.rl_state_path}")
                self.rl_loop.load_state(self.rl_state_path)
            self.rl_state_loaded = True
        self.rl_loop.step = self.state.epoch - 1
        round_tic = time.perf_counter()
        metrics = self.rl_loop.run_round()
        time_info = {"round_cost": AverageMeter(name="round_cost", postfix="s")}
        time_info["round_cost"].update(time.perf_counter() - round_tic)

        updates = int(metrics["updates"])
        self.state.step_in_train_epoch = updates
        self.state.max_steps_in_train_epoch = updates
        self.state.global_step += updates

        loss_info = {}
        for name in ("loss", "loss_diff", "loss_kl", "anchor_loss"):
            loss_info[name] = AverageMeter(name=name)
            loss_info[name].update(float(metrics[name]))
        metric_info = {}
        for name, value in metrics.items():
            if name in loss_info or name in ("updates", "stop"):
                continue
            metric_info[name] = AverageMeter(name=name)
            metric_info[name].update(float(value))

        self.rl_loop.save_state(self.rl_state_path)
        self.stop_training = bool(metrics["stop"])
        return time_info, loss_info, metric_info

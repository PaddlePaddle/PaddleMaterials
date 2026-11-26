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

"""MatInvent training state bound through the existing model lifecycle hook."""

import copy
import os

import paddle
import paddle.distributed as dist

from ppmat.datasets import build_dataloader
from ppmat.datasets.collate_fn import DefaultCollator
from ppmat.datasets.transform import build_transforms
from ppmat.metrics.streaming_base import StreamingMetricBase
from ppmat.models.matinvent.dataset import RLDataset
from ppmat.models.matinvent.dataset import to_tensor_batch
from ppmat.models.matinvent.qualification import PopulationQualifier
from ppmat.models.matinvent.rewards.reward import Reward
from ppmat.models.matinvent.rl_loop import MatInventRL
from ppmat.models.matinvent.sampler import MatInventSampler


class MatInventTraining(StreamingMetricBase):
    """Prepare round data and monitor rewards and probes after optimizer updates.

    The model returns one timestep objective per forward. BaseTrainer performs
    backward, accumulation and optimization before notifying this monitor.
    """

    def __init__(self, trainer, cfg):
        if dist.get_world_size() > 1:
            raise ValueError("MatInvent RL supports a single process only")
        if trainer.use_amp:
            raise ValueError("MatInvent RL does not support use_amp")
        for name in ("eval_freq_steps", "save_freq_steps", "max_iter"):
            if trainer.config.get(name) is not None:
                raise ValueError(f"MatInvent RL does not support Trainer.{name}")
        if trainer.optimizer is None:
            raise ValueError("MatInvent RL training requires an Optimizer")
        finetune = cfg["finetune_cfg"]
        if trainer.gradient_accumulation_steps != finetune["accum_steps"]:
            raise ValueError("Trainer gradient accumulation must match finetune_cfg")
        loader = trainer.train_dataloader
        self.padded_timesteps = loader.dataset.padded_timesteps
        self.num_steps = len(loader.dataset)
        if (
            loader.dataset.timesteps != finetune["timesteps"]
            or loader.dataset.epochs != finetune["epochs"]
            or loader.dataset.accum_steps != finetune["accum_steps"]
            or len(loader) != self.num_steps
            or loader.batch_sampler.batch_size != 1
            or loader.batch_sampler.shuffle
            or loader.num_workers != 0
        ):
            raise ValueError(
                "MatInvent RL needs ordered, single-sample timestep batches with "
                "num_workers=0 and the configured round length"
            )
        self.trainer = trainer
        agent = trainer.model
        prior = copy.deepcopy(agent)
        prior.training_cfg = None
        prior._pending_training_state = None
        for parameter in prior.parameters():
            parameter.stop_gradient = True
        prior.eval()
        anchor_cfg = cfg["anchor_cfg"]
        use_anchor = anchor_cfg["loss_weight"] > 0
        use_probe = cfg["early_stop_cfg"]["enabled"]
        anchor_dataset = None
        anchor_batch = None
        probe_batch = None
        if use_anchor or (use_probe and trainer.val_dataloader is None):
            anchor_dataset = build_dataloader(anchor_cfg["data_cfg"]).dataset
        if use_anchor:
            anchor_batch = self.build_batch(anchor_dataset, anchor_cfg["num_samples"])
        if use_probe:
            probe_dataset = (
                trainer.val_dataloader.dataset
                if trainer.val_dataloader is not None
                else anchor_dataset
            )
            probe_batch = self.build_batch(
                probe_dataset, cfg["early_stop_cfg"]["probe_num_samples"]
            )
        self.loop = MatInventRL(
            agent=agent,
            prior=prior,
            sampler=MatInventSampler(
                data_cfg=cfg["sample_cfg"]["data_cfg"],
                num_inference_steps=cfg["sample_cfg"]["num_inference_steps"],
                build_structure_cfg=cfg["build_structure_cfg"],
            ),
            reward=Reward(
                root_dir=os.path.join(trainer.output_dir, "rewards"),
                **cfg["reward_cfg"],
            ),
            sample_cfg=cfg["sample_cfg"],
            finetune_cfg=finetune,
            topk_ratio=cfg["topk_ratio"],
            anchor_batch=anchor_batch,
            anchor_dataset=anchor_dataset,
            anchor_cfg=anchor_cfg,
            probe_batch=probe_batch,
            replay_cfg=cfg.get("replay_cfg"),
            div_filter_cfg=cfg.get("div_filter_cfg"),
            early_stop_cfg=cfg["early_stop_cfg"],
            qualifier=(
                PopulationQualifier(**cfg["qualification_cfg"])
                if cfg.get("qualification_cfg") is not None
                else None
            ),
            transforms=build_transforms(cfg["transforms"]),
            save_dir=trainer.output_dir,
        )
        self.batch = None
        self.metrics = {}
        self.updates = 0
        self.stopped = False

    @staticmethod
    def build_batch(dataset, count):
        count = min(count, len(dataset))
        if count <= 0:
            raise ValueError("Anchor and probe batches must contain structures")
        return to_tensor_batch(
            DefaultCollator()([dataset[index] for index in range(count)])
        )

    def forward(self, batch):
        step = int(batch["rl_index"].item())
        if step == 0:
            self.loop.step = self.trainer.state.epoch - 1
            structures, rewards, self.metrics = self.loop.prepare_round()
            dataset = RLDataset(structures, rewards, self.loop.transforms)
            self.batch = self.build_batch(dataset, len(dataset))
            self.updates = 0
        self.loop.agent.train()
        index = step % self.padded_timesteps
        cfg = self.loop.finetune_cfg
        if step % cfg["accum_steps"] == 0:
            self.group_metrics = {
                name: 0.0
                for name in ("objective", "loss_diff", "loss_kl", "anchor_loss")
            }
            self.group_count = 0
        if index >= cfg["timesteps"]:
            zero = (
                next(
                    p for p in self.loop.agent.parameters() if not p.stop_gradient
                ).sum()
                * 0.0
            )
            losses = {"loss": zero}
        else:
            losses = self.loop.compute_loss(self.batch, index)
            for name in self.group_metrics:
                self.group_metrics[name] += float(losses[name])
            self.group_count += 1
        if (step + 1) % cfg["accum_steps"] == 0:
            averages = {
                name: value / self.group_count
                for name, value in self.group_metrics.items()
            }
            loss = losses["loss"]
            losses = {
                name: paddle.to_tensor(value, dtype=loss.dtype)
                for name, value in averages.items()
            }
            losses["loss"] = loss + averages["objective"] - loss.detach()
        return {"loss_dict": losses}

    def update_step(self, *, result, batch, stage):
        if stage != "train":
            return
        self.updates += 1
        cfg = self.loop.early_stop_cfg
        if (
            self.loop.early_stop_enabled
            and self.updates % int(cfg["probe_every_updates"]) == 0
            and self.loop._check_early_stop(probe_only=True)
        ):
            self.stopped = True
        if int(batch["rl_index"].item()) + 1 == self.num_steps:
            self.stopped = self.stopped or self.loop._check_early_stop()
        if self.stopped:
            self.trainer.max_train_steps = self.trainer.state.global_step

    def compute_epoch(self, *, stage):
        if stage != "train":
            return {}
        return dict(self.metrics)

    def reset(self):
        return

    def state_dict(self):
        return {
            "loop": self.loop.state_dict(),
            "coord_sigmas_norm": (
                None
                if self.loop.continuous_timesteps
                else self.loop.agent.coord_scheduler.discrete_sigmas_norm.numpy()
            ),
            "prior": {
                name: value.numpy()
                for name, value in self.loop.prior.state_dict().items()
            },
            "anchor_batch": paddle.utils.map_structure(
                lambda value: value.numpy()
                if isinstance(value, paddle.Tensor)
                else value,
                self.loop.anchor_batch,
            ),
            "probe_batch": paddle.utils.map_structure(
                lambda value: value.numpy()
                if isinstance(value, paddle.Tensor)
                else value,
                self.loop.probe_batch,
            ),
            "stopped": self.stopped,
        }

    def set_state_dict(self, state):
        self.loop.set_state_dict(state["loop"])
        self.loop.prior.set_state_dict(state["prior"])
        if not self.loop.continuous_timesteps:
            self.loop.prior.coord_scheduler.discrete_sigmas_norm = paddle.to_tensor(
                state["coord_sigmas_norm"]
            )
        self.loop.anchor_batch = to_tensor_batch(state["anchor_batch"])
        self.loop.probe_batch = to_tensor_batch(state["probe_batch"])
        self.stopped = state["stopped"]
        if self.stopped:
            self.trainer.max_train_steps = 0

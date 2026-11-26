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

"""One MatInvent round: sample, score, select and fine-tune.

The model owns sampling, selection and the objective. BaseTrainer owns backward,
gradient accumulation, optimizer updates and checkpoints.
"""

import os

import numpy as np
import paddle
from ase.io import write
from pymatgen.io.ase import AseAtomsAdaptor

from ppmat.datasets.collate_fn import DefaultCollator
from ppmat.models.matinvent.dataset import to_tensor_batch
from ppmat.models.matinvent.memory import LongTimeMem
from ppmat.models.matinvent.memory import ReplayBuffer
from ppmat.models.matinvent.qualification import is_valid_structure
from ppmat.utils import logger
from ppmat.utils.scatter import scatter

# Fixed offset of the reference advantage term: ``1.1 - reward``.
_ADVANTAGE_REWARD_OFFSET = 1.1


def save_structures(structures, save_dir, filename) -> None:
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, filename)
    adaptor = AseAtomsAdaptor()
    atoms = [adaptor.get_atoms(struc) for struc in structures]
    write(out_path, atoms, format="extxyz")
    logger.info(f"Saved {len(structures)} structures to {out_path}")


class MatInventRL:
    """Reward-weighted fine-tuning of a diffusion backbone in one round."""

    def __init__(
        self,
        agent,
        prior,
        sampler,
        reward,
        sample_cfg,
        finetune_cfg,
        topk_ratio,
        anchor_batch,
        anchor_dataset,
        anchor_cfg,
        probe_batch,
        replay_cfg,
        div_filter_cfg,
        early_stop_cfg,
        qualifier,
        transforms,
        save_dir,
    ):
        if not 0.0 < topk_ratio <= 1.0:
            raise ValueError(f"topk_ratio must be in (0.0, 1.0], but got {topk_ratio}")
        batch_size = finetune_cfg["batch_size"]
        topk_num = int(batch_size * topk_ratio)
        if topk_num < 1:
            raise ValueError(
                f"finetune_cfg.batch_size * topk_ratio must select at least one "
                f"structure, but got batch_size={batch_size} and "
                f"topk_ratio={topk_ratio}"
            )
        if not 0 < finetune_cfg["timesteps"] <= agent.num_train_timesteps:
            raise ValueError("Fine-tuning timesteps must fit the noise schedule")
        if finetune_cfg["accum_steps"] <= 0:
            raise ValueError("Fine-tuning accum_steps must be positive")
        self.early_stop_enabled = bool(early_stop_cfg["enabled"])
        if self.early_stop_enabled:
            for key in ("probe_every", "probe_every_updates", "probe_draws"):
                if int(early_stop_cfg[key]) <= 0:
                    raise ValueError(f"early_stop_cfg.{key} must be positive")
        if int(anchor_cfg["every_n_steps"]) <= 0:
            raise ValueError("anchor_cfg.every_n_steps must be positive")

        self.agent = agent
        self.prior = prior
        self.sampler = sampler
        self.reward = reward
        self.sample_cfg = sample_cfg
        self.finetune_cfg = finetune_cfg
        self.topk_num = topk_num
        self.anchor_batch = anchor_batch
        self.anchor_dataset = anchor_dataset
        self.anchor_cfg = anchor_cfg
        self.probe_batch = probe_batch
        self.early_stop_cfg = early_stop_cfg
        self.qualifier = qualifier
        self.transforms = transforms
        self.sample_dir = os.path.join(save_dir, "samples")
        os.makedirs(self.sample_dir, exist_ok=True)
        self.replay = ReplayBuffer(**replay_cfg) if replay_cfg is not None else None
        self.div_filter_cfg = div_filter_cfg
        self.continuous_timesteps = bool(agent.continuous_timesteps)
        self.step = 0
        self.cost = 0
        self.ltm = LongTimeMem()
        self.probe_initial = self._probe_anchor() if self.early_stop_enabled else None
        self.last_hhi = None
        self.last_valid_ratio = None
        self.last_div_ratio = None
        self.hhi_best = float("inf")
        self.hhi_bad_rounds = 0

    def _probe_anchor(self):
        draws = int(self.early_stop_cfg["probe_draws"])
        was_training = self.agent.training
        numpy_state = np.random.get_state()
        paddle_state = paddle.get_rng_state()
        cpu_state = paddle.get_rng_state("cpu")
        self.agent.eval()
        values = {
            "loss": [],
            "loss_lattice": [],
            "loss_coord": [],
            "loss_atom_type": [],
        }
        try:
            for index in range(draws):
                np.random.seed(20000 + index)
                paddle.seed(20000 + index)
                with paddle.no_grad():
                    loss_dict = self.agent.diffusion_forward(self.probe_batch)[
                        "loss_dict"
                    ]
                for key in values:
                    if key in loss_dict:
                        values[key].append(float(loss_dict[key]))
        finally:
            np.random.set_state(numpy_state)
            paddle.set_rng_state(paddle_state)
            paddle.set_rng_state(cpu_state, "cpu")
            self.agent.train(was_training)
        return {
            key: float(np.mean(items)) if items else 0.0
            for key, items in values.items()
        }

    def _check_early_stop(self, *, probe_only=False):
        if not self.early_stop_enabled:
            return False
        cfg = self.early_stop_cfg
        reason = None
        probe = None
        if probe_only or (self.step + 1) % int(cfg["probe_every"]) == 0:
            probe = self._probe_anchor()
            logger.info(f"Anchor probe round {self.step}: {probe}")
            for key, ratio_key in (
                ("loss", "max_loss_ratio"),
                ("loss_lattice", "max_lattice_ratio"),
                ("loss_atom_type", "max_type_ratio"),
            ):
                initial = self.probe_initial[key]
                limit = cfg[ratio_key]
                if not np.isfinite(probe[key]):
                    reason = f"Nonfinite probe {key}: {probe[key]}"
                    break
                if initial > 0 and probe[key] > initial * float(limit):
                    reason = f"{key} {probe[key]:.6f} > initial {initial:.6f} x {limit}"
                    break
        if (
            not probe_only
            and reason is None
            and self.last_hhi is not None
            and cfg.get("hhi_patience") is not None
        ):
            patience = int(cfg["hhi_patience"])
            rebound = float(cfg["hhi_rebound"])
            if self.last_hhi < self.hhi_best:
                self.hhi_best = self.last_hhi
                self.hhi_bad_rounds = 0
            else:
                self.hhi_bad_rounds += 1
            if self.hhi_bad_rounds >= patience and self.last_hhi > self.hhi_best * (
                1.0 + rebound
            ):
                reason = (
                    f"hhi {self.last_hhi:.1f} rebounded > {rebound:.0%} "
                    f"from best {self.hhi_best:.1f} for {patience} rounds"
                )
        if not probe_only and reason is None and self.last_valid_ratio is not None:
            minimum = cfg.get("min_valid_ratio")
            if minimum is not None and self.last_valid_ratio < float(minimum):
                reason = (
                    f"valid ratio {self.last_valid_ratio:.3f} < {float(minimum):.3f}"
                )
        if not probe_only and reason is None and self.last_div_ratio is not None:
            minimum = cfg.get("min_div_ratio")
            if minimum is not None and self.last_div_ratio < float(minimum):
                reason = f"div ratio {self.last_div_ratio:.3f} < {float(minimum):.3f}"
        if reason is not None:
            logger.warning(f"EARLY STOP at round {self.step}: {reason}")
        return reason is not None

    def _sample_step(self):
        self.agent.eval()
        structures = self.sampler.generate(self.agent)
        valid_structures = [
            structure for structure in structures if is_valid_structure(structure)
        ]
        self.last_valid_ratio = len(valid_structures) / max(1, len(structures))
        logger.info(
            f"Valid ratio before max_num: {self.last_valid_ratio:.3f} "
            f"({len(valid_structures)}/{len(structures)})"
        )
        if len(valid_structures) == 0:
            logger.warning("No valid structures generated!")
            return [], []
        save_structures(
            structures=valid_structures,
            save_dir=self.sample_dir,
            filename=f"step_{self.step:0>4d}_valid.extxyz",
        )
        training_structures, scoring_structures = (
            self.qualifier(valid_structures)
            if self.qualifier is not None
            else (valid_structures, valid_structures)
        )
        if not scoring_structures:
            logger.warning("No structures passed population qualification")
            return [], []
        max_num = self.sample_cfg.get("max_num")
        if max_num is not None and len(scoring_structures) > int(max_num):
            training_structures = training_structures[: int(max_num)]
            scoring_structures = scoring_structures[: int(max_num)]
        save_structures(
            structures=scoring_structures,
            save_dir=self.sample_dir,
            filename=f"step_{self.step:0>4d}_eval.extxyz",
        )
        return training_structures, scoring_structures

    def _score_step(self, training_structures, scoring_structures):
        if not scoring_structures:
            return [], [], np.empty(0, dtype=float), {}
        label = f"step_{self.step:0>4d}"
        rewards, prop_dict, failed_mask = self.reward.scoring(scoring_structures, label)
        self.cost += len(scoring_structures)
        success = ~failed_mask
        scaled_rewards = rewards[success].astype(float)
        scaled_props = {key: value[success] for key, value in prop_dict.items()}
        structures = [
            structure for structure, keep in zip(scoring_structures, success) if keep
        ]
        training = [
            structure for structure, keep in zip(training_structures, success) if keep
        ]
        logger.info(f"Evaluation costs to date: {self.cost}")
        logger.info(
            f"Number of samples that successfully obtained rewards: {len(structures)}"
        )
        if len(scaled_rewards) > 0:
            logger.info(
                f"reward mean={scaled_rewards.mean():.4f} "
                f"std={scaled_rewards.std():.4f}"
            )
            logger.info(
                " | ".join(
                    f"{key} mean={value.mean():.4f} std={value.std():.4f}"
                    for key, value in scaled_props.items()
                )
            )
        else:
            logger.info("reward mean=nan std=nan")
        return training, structures, scaled_rewards, scaled_props

    def _times_tensor(self, num_atoms, index):
        batch_size = num_atoms.shape[0]
        if self.continuous_timesteps:
            schedule = paddle.linspace(
                self.agent.max_t,
                1.0 / self.agent.num_train_timesteps,
                self.agent.num_train_timesteps,
            )
            return paddle.full([batch_size], schedule[index])
        return paddle.full(
            [batch_size],
            self.agent.num_train_timesteps - 1 - index,
            dtype="int64",
        )

    def _weighted_loss(self, loss_dict):
        loss = (
            self.agent.lattice_loss_weight * loss_dict["loss_lattice"]
            + self.agent.coord_loss_weight * loss_dict["loss_coord"]
        )
        if "loss_atom_type" in loss_dict:
            loss = loss + self.agent.atom_loss_weight * loss_dict["loss_atom_type"]
        return loss

    def _kl_divergence(self, agent_pred, prior_pred, num_atoms):
        batch_idx = paddle.repeat_interleave(
            paddle.arange(num_atoms.shape[0]), repeats=num_atoms
        )
        kl = paddle.pow(agent_pred["lattice"] - prior_pred["lattice"], 2).mean(
            axis=(1, 2)
        )
        kl += scatter(
            paddle.pow(agent_pred["frac_coords"] - prior_pred["frac_coords"], 2).mean(
                axis=1
            ),
            batch_idx,
            dim=0,
            reduce="mean",
        )
        if "atom_types" in agent_pred:
            kl += scatter(
                paddle.pow(agent_pred["atom_types"] - prior_pred["atom_types"], 2).mean(
                    axis=1
                ),
                batch_idx,
                dim=0,
                reduce="mean",
            )
        return kl

    def _anchor_training_batch(self):
        if not self.anchor_cfg["resample"]:
            return self.anchor_batch
        count = int(self.anchor_batch["structure_array"]["num_atoms"].shape[0])
        indices = np.random.choice(len(self.anchor_dataset), size=count, replace=False)
        batch = DefaultCollator()([self.anchor_dataset[int(i)] for i in indices])
        return to_tensor_batch(batch)

    def compute_loss(self, batch, index):
        cfg = self.finetune_cfg
        times = self._times_tensor(batch["structure_array"]["num_atoms"], index)
        denoised = self.agent.denoise_with_time(batch["structure_array"], times)
        sample_loss = self._weighted_loss(denoised["loss_dict"])
        with paddle.no_grad():
            prior_denoised = self.prior.denoise_with_time(
                batch["structure_array"], times, noise_state=denoised["noise_state"]
            )
        kl_term = self._kl_divergence(
            denoised["pred_dict"],
            prior_denoised["pred_dict"],
            batch["structure_array"]["num_atoms"],
        )
        loss_diff = (batch["reward"] * sample_loss).mean()
        loss_kl = (kl_term * (_ADVANTAGE_REWARD_OFFSET - batch["reward"])).mean()
        anchor_loss = paddle.zeros_like(loss_diff)
        if (
            float(self.anchor_cfg["loss_weight"]) > 0
            and (index + 1) % int(self.anchor_cfg["every_n_steps"]) == 0
        ):
            anchor_loss = self.agent.diffusion_forward(self._anchor_training_batch())[
                "loss_dict"
            ]["loss"]
        objective = (
            loss_diff
            + cfg["sigma"] * loss_kl
            + float(self.anchor_cfg["loss_weight"]) * anchor_loss
        )
        group_start = index // cfg["accum_steps"] * cfg["accum_steps"]
        group_size = min(cfg["accum_steps"], cfg["timesteps"] - group_start)
        loss = objective / group_size
        if not bool(paddle.isfinite(loss).all()):
            raise FloatingPointError(f"Nonfinite RL loss at timestep {index}")
        return {
            "loss": loss,
            "objective": objective,
            "loss_diff": loss_diff,
            "loss_kl": loss_kl,
            "anchor_loss": anchor_loss,
        }

    def prepare_round(self):
        logger.info(f"*****   ROUND {self.step} START   *****")
        logger.info("SAMPLE:")
        training_structures, scoring_structures = self._sample_step()
        logger.info("SCORE:")
        training_structures, scoring_structures, rewards, prop_dict = self._score_step(
            training_structures, scoring_structures
        )
        metrics = {
            "num_scored": float(len(scoring_structures)),
            "reward_mean": float("nan"),
            "reward_std": float("nan"),
            "burden": float("nan"),
            "div_ratio": float("nan"),
            "cost": float(self.cost),
        }
        for name, values in prop_dict.items():
            metrics[f"{name}_mean"] = (
                float(values.mean()) if len(values) else float("nan")
            )
            metrics[f"{name}_std"] = (
                float(values.std()) if len(values) else float("nan")
            )
        if "hhi" in prop_dict and len(prop_dict["hhi"]) > 0:
            self.last_hhi = float(np.mean(prop_dict["hhi"]))
        else:
            self.last_hhi = None

        if len(rewards) > 0:
            metrics["reward_mean"] = float(rewards.mean())
            metrics["reward_std"] = float(rewards.std())
            self.ltm.extend(scoring_structures, rewards, self.step)
        else:
            logger.warning("No successful rewards; checking the replay buffer.")
        burden, div_ratio = self.ltm.calc_metrics(self.reward.threshold)
        self.last_div_ratio = div_ratio
        self.ltm.save(os.path.join(self.sample_dir, "long_term_memory.csv"))
        if burden is not None:
            metrics["burden"] = float(burden)
        if div_ratio is not None:
            metrics["div_ratio"] = float(div_ratio)
        logger.info(f"round={self.step} metrics={metrics}")
        if len(rewards) == 0:
            if self.replay is not None and len(self.replay) > 0:
                replay_data, replay_rewards = self.replay.sample()
                return replay_data, np.asarray(replay_rewards, dtype=float), metrics
            raise RuntimeError("No scored structures or replay samples for RL training")

        penalty_structures = []
        if self.div_filter_cfg is not None:
            rewards, penalty_idx, tol_n, buff_n = self.ltm.div_filter(
                scoring_structures,
                rewards,
                tol=self.div_filter_cfg["tol"],
                buff=self.div_filter_cfg["buff"],
                method=self.div_filter_cfg["method"],
            )
            penalty_structures = [scoring_structures[i] for i in penalty_idx]
            logger.info(f"Diversity filter: tol_n={tol_n}, buff_n={buff_n}")
        sort_idx = np.argsort(rewards)[::-1]
        topk_idx = sort_idx[: self.topk_num]
        structures_topk = [scoring_structures[i] for i in topk_idx]
        training_topk = [training_structures[i] for i in topk_idx]
        reward_topk = rewards[topk_idx]
        if self.replay is not None:
            if len(penalty_structures) > 0:
                self.replay.memory_purge(penalty_structures)
            replay_data, replay_rewards = self.replay.sample()
            finetune_data = training_topk + replay_data
            finetune_rewards = np.concatenate((reward_topk, replay_rewards))
            self.replay.extend(
                training_topk, reward_topk, scoring_structures=structures_topk
            )
            logger.info(f"replay buffer size={len(self.replay)}")
        else:
            finetune_data, finetune_rewards = training_topk, reward_topk
        return finetune_data, finetune_rewards, metrics

    def state_dict(self):
        return {
            "step": self.step,
            "cost": self.cost,
            "ltm": self.ltm,
            "replay": self.replay,
            "probe_initial": self.probe_initial,
            "last_hhi": self.last_hhi,
            "hhi_best": self.hhi_best,
            "hhi_bad_rounds": self.hhi_bad_rounds,
            "last_valid_ratio": self.last_valid_ratio,
            "last_div_ratio": self.last_div_ratio,
        }

    def set_state_dict(self, state):
        for key, value in state.items():
            setattr(self, key, value)

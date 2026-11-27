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

import paddle
import pytest

from ppmat.optimizer import build_optimizer
from ppmat.trainer.base_trainer import BaseTrainer


class _Model(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.model_global = paddle.nn.Linear(1, 1, bias_attr=False)
        self.model_local = paddle.nn.Linear(1, 1, bias_attr=False)
        for layer in (self.model_global, self.model_local):
            layer.weight.set_value(paddle.zeros_like(layer.weight))

    def forward(self, batch):
        features = batch["local"]
        global_value = self.model_global(features).mean()
        local_value = self.model_local(features).mean()
        return {
            "loss_dict": {
                "loss": 3 * global_value + 4 * local_value,
                "global": 1 + global_value,
                "local": 1 - local_value,
            },
            "pred_dict": {"local": 1 - local_value},
        }


class _OneBatch(paddle.io.Dataset):
    def __len__(self):
        return 1

    def __getitem__(self, index):
        return {"local": paddle.ones([1])}


def _adam(indicator, indicator_name, learning_rate):
    return {
        "__class_name__": "Adam",
        "__init_params__": {
            "beta1": 0.5,
            "lr": {
                "__class_name__": "ReduceOnPlateau",
                "__init_params__": {
                    "indicator": indicator,
                    "indicator_name": indicator_name,
                    "learning_rate": learning_rate,
                    "factor": 0.5,
                    "patience": 0,
                },
            },
        },
    }


def _trainer(
    output_dir, max_epochs, *, grouped=True, eval_freq_steps=1, reverse_groups=False
):
    model = _Model()
    loader = paddle.io.DataLoader(_OneBatch(), batch_size=1)
    if grouped:
        optimizer_cfg = {
            "groups": {
                "global": {
                    "model_attr": "model_global",
                    **_adam("eval_loss", "global", 0.1),
                },
                "local": {
                    "model_attr": "model_local",
                    **_adam("eval_metric", "local", 0.2),
                },
            },
            "clip_norm_global": 4.0,
        }
        if reverse_groups:
            optimizer_cfg["groups"] = dict(reversed(optimizer_cfg["groups"].items()))
    else:
        optimizer_cfg = {
            **_adam("eval_metric", "local", 0.2),
            "clip_norm_global": 4.0,
        }

    optimizer, scheduler = build_optimizer(
        optimizer_cfg, model, max_epochs, len(loader)
    )
    return BaseTrainer(
        {
            "max_epochs": max_epochs,
            "output_dir": str(output_dir),
            "save_freq": 1,
            "log_freq": 1,
            "start_eval_epoch": 1,
            "eval_freq": 1,
            "eval_freq_steps": eval_freq_steps,
            "save_freq_steps": 1,
            "seed": 42,
            "compute_metric_during_train": False,
        },
        model,
        train_dataloader=loader,
        val_dataloader=loader,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        compute_metric_func_dict={"local": paddle.nn.functional.l1_loss},
    )


def _first_moments(state):
    return [value.item() for key, value in state.items() if "_moment1_" in key]


def test_grouped_adam_clips_jointly_before_either_update(tmp_path):
    trainer = _trainer(tmp_path, 1)
    trainer.train()

    # Grads 3 and 4 individually fit; together they need the common scale 4/5.
    # Adam's first moment with beta1=0.5 preserves the scale of its input.
    global_state, local_state = trainer.optimizer.state_dict()
    assert _first_moments(global_state) == pytest.approx([1.2])
    assert _first_moments(local_state) == pytest.approx([1.6])


@pytest.mark.parametrize("eval_freq_steps", [1, None], ids=["step", "epoch"])
def test_each_plateau_reads_its_own_validation_signal(tmp_path, eval_freq_steps):
    trainer = _trainer(tmp_path, 2, eval_freq_steps=eval_freq_steps)
    trainer.train()

    # Both parameters fall, so global eval loss falls and local eval MAE rises.
    assert [opt.get_lr() for opt in trainer.optimizer] == pytest.approx([0.1, 0.1])


@pytest.mark.parametrize(
    ("eval_freq_steps", "checkpoint_name"),
    [(1, "step_2"), (None, "epoch_2")],
    ids=["step", "epoch"],
)
def test_grouped_checkpoint_restores_next_adam_update(
    tmp_path, eval_freq_steps, checkpoint_name
):
    original_dir = tmp_path / "original"
    with paddle.utils.unique_name.guard():
        original = _trainer(original_dir, 3, eval_freq_steps=eval_freq_steps)
        original.train()

    # Resuming after the second validation should reproduce the third update and
    # the second local LR decay, including Adam's accumulated moments.
    with paddle.utils.unique_name.guard():
        resumed = _trainer(
            tmp_path / "resumed",
            3,
            eval_freq_steps=eval_freq_steps,
            reverse_groups=True,
        )
        checkpoint = original_dir / "checkpoints" / checkpoint_name
        resumed.train(resume_from_checkpoint=str(checkpoint))

    assert resumed.state.global_step == original.state.global_step == 3
    assert [opt.get_lr() for opt in resumed.optimizer] == pytest.approx([0.1, 0.05])
    for expected, actual in zip(
        original.model.parameters(), resumed.model.parameters()
    ):
        assert actual.item() == pytest.approx(expected.item(), abs=1e-7)
    for expected, actual in zip(
        original.optimizer.state_dict(), resumed.optimizer.state_dict()
    ):
        assert _first_moments(actual) == pytest.approx(_first_moments(expected))


def test_single_adam_keeps_existing_clip_plateau_and_checkpoint_shape(tmp_path):
    trainer = _trainer(tmp_path, 2, grouped=False)
    trainer.train()

    assert trainer.optimizer.get_lr() == pytest.approx(0.1)
    state = paddle.load(str(tmp_path / "checkpoints" / "step_1.pdopt"))
    assert isinstance(state, dict)
    assert sorted(_first_moments(state)) == pytest.approx([1.2, 1.6])


def test_grouped_optimizer_rejects_shared_parameter_ownership():
    model = _Model()
    model.model_local = model.model_global
    groups = {
        "global": {"model_attr": "model_global", **_adam("eval_loss", "global", 0.1)},
        "local": {"model_attr": "model_local", **_adam("eval_loss", "local", 0.2)},
    }
    with pytest.raises(ValueError, match="multiple optimizer groups"):
        build_optimizer({"groups": groups}, model, 1, 1)


def test_grouped_optimizer_rejects_uncovered_trainable_parameters():
    model = _Model()
    model.extra = model.create_parameter([1])
    groups = {
        "global": {"model_attr": "model_global", **_adam("eval_loss", "global", 0.1)},
        "local": {"model_attr": "model_local", **_adam("eval_loss", "local", 0.2)},
    }
    with pytest.raises(ValueError, match="trainable parameters"):
        build_optimizer({"groups": groups, "clip_norm_global": 4.0}, model, 1, 1)

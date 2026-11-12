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


"""Zero frequencies in the trainer config disable the matching periodic action."""

from __future__ import annotations

import paddle

from ppmat.trainer.base_trainer import BaseTrainer


class _LinearModel(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.linear = paddle.nn.Linear(1, 1)

    def forward(self, batch, return_loss=True, return_prediction=True):
        prediction = self.linear(batch["x"])
        output = {"loss_dict": {}, "pred_dict": {"value": prediction}}
        if return_loss:
            output["loss_dict"]["loss"] = paddle.nn.functional.mse_loss(
                prediction, batch["y"]
            )
        return output


def _loader():
    samples = [
        {"x": paddle.to_tensor([[1.0]]), "y": paddle.to_tensor([[2.0]])},
        {"x": paddle.to_tensor([[2.0]]), "y": paddle.to_tensor([[4.0]])},
    ]
    return paddle.io.DataLoader(samples, batch_size=1, shuffle=False, return_list=True)


def test_zero_save_eval_and_log_freq_disable_periodic_actions(tmp_path):
    output_dir = tmp_path / "out"
    config = {
        "max_epochs": 2,
        "output_dir": str(output_dir),
        "save_freq": 0,
        "log_freq": 0,
        "start_eval_epoch": 1,
        "eval_freq": 0,
        "seed": 2026,
        "compute_metric_during_train": False,
        "use_amp": False,
        "eval_with_no_grad": True,
        "gradient_accumulation_steps": 1,
    }
    model = _LinearModel()
    optimizer = paddle.optimizer.Adam(learning_rate=1e-3, parameters=model.parameters())
    trainer = BaseTrainer(
        config,
        model,
        train_dataloader=_loader(),
        val_dataloader=_loader(),
        optimizer=optimizer,
        execution_config={"backend": "eager"},
    )
    eval_calls = []
    train_eval_epoch = trainer.eval_epoch

    def counting_eval_epoch(dataloader):
        eval_calls.append(trainer.state.epoch)
        return train_eval_epoch(dataloader)

    trainer.eval_epoch = counting_eval_epoch

    trainer.train()

    checkpoints = sorted(p.name for p in (output_dir / "checkpoints").iterdir())
    assert not any(name.startswith("epoch_") for name in checkpoints), checkpoints
    assert any(name.startswith("latest") for name in checkpoints), checkpoints
    assert eval_calls == []

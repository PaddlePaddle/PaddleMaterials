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

import paddle  # noqa

from ppmat.metrics.csp_metric import CSPMetric
from ppmat.metrics.diffnmr_streaming_adapter import DiffNMRStreamingAdapter
from ppmat.metrics.sfin_metric import SFINStreamingAdapter

__all__ = [
    "build_metric",
    "CSPMetric",
    "DiffNMRStreamingAdapter",
    "SFINStreamingAdapter",
    # "DiffNMRMetric",
    # "NLL", "CrossEntropyMetric", "SumExceptBatchMetric", "SumExceptBatchKL",
]


class IgnoreNanMetricWrapper:
    """Compute a metric while ignoring ``nan`` labels.

    ``flatten_if_same_numel`` exists for layouts that keep the same element
    count with different trailing dimensions, e.g. stress stored as ``[1, 3, 3]``
    per sample while the model emits ``[3, 3]``.  It is off by default so every
    existing model keeps the previous strict shape behaviour.
    """

    def __init__(self, flatten_if_same_numel=False, **metric_cfg):
        self._flatten_if_same_numel = flatten_if_same_numel
        self._metric_cfg = metric_cfg
        self._metric = build_metric(self._metric_cfg)

    def __call__(self, pred, label):
        if pred.shape != label.shape:
            if not self._flatten_if_same_numel:
                raise ValueError(
                    f"pred shape {list(pred.shape)} != label shape "
                    f"{list(label.shape)}. Set flatten_if_same_numel=True only "
                    "when both tensors hold the same element count with "
                    "different trailing dimensions."
                )
            if pred.numel() != label.numel():
                raise ValueError(
                    f"cannot flatten pred shape {list(pred.shape)} and label "
                    f"shape {list(label.shape)}: element counts differ "
                    f"({pred.numel()} vs {label.numel()})."
                )
            pred = pred.reshape([-1])
            label = label.reshape([-1])

        valid_value_indices = ~paddle.isnan(label)
        valid_label = label[valid_value_indices]
        valid_pred = pred[valid_value_indices]
        if valid_label.numel() > 0:
            metric_value = self._metric(valid_pred, valid_label)
        else:
            metric_value = paddle.nan
        return metric_value


def build_metric(cfg):
    """Build metric.

    Args:
        cfg (DictConfig): Metric config.

    Returns:
        Metric: Callable Metric object.
    """
    if cfg is None:
        return None
    cfg = copy.deepcopy(cfg)

    if "__class_name__" not in cfg:
        assert isinstance(cfg, dict)
        metric_dict = {}
        for key, sub_cfg in cfg.items():
            metric_dict[key] = build_metric(sub_cfg)
        return metric_dict

    class_name = cfg.pop("__class_name__")
    init_params = cfg.pop("__init_params__")

    metric = eval(class_name)(**init_params)
    return metric

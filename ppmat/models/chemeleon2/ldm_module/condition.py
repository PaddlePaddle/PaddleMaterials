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

import numpy as np
import paddle
import paddle.nn as nn


def _double_condition(value):
    """Repeat each value for the unconditional and conditional CFG halves."""
    if isinstance(value, paddle.Tensor):
        return paddle.concat([value, value], axis=0)
    if isinstance(value, np.ndarray):
        return np.concatenate([value, value], axis=0)
    if isinstance(value, (list, tuple)):
        return list(value) * 2
    return [value, value]


class Chemeleon2ConditionModule(nn.Layer):
    def __init__(self, property_embeddings_adapt_cfg, hidden_dim, p_unconditional):
        super().__init__()
        if not 0 <= p_unconditional <= 1:
            raise ValueError("p_unconditional must lie in [0, 1].")
        self.target_condition = list(property_embeddings_adapt_cfg)
        if not self.target_condition:
            raise ValueError("At least one condition embedding is required.")
        self.hidden_dim = hidden_dim
        self.p_unconditional = p_unconditional
        self.encoders = nn.LayerDict()
        for name, cfg in property_embeddings_adapt_cfg.items():
            self.encoders[name] = eval(cfg["__class_name__"])(
                hidden_dim=hidden_dim, **cfg["__init_params__"]
            )

        self.proj = nn.Sequential(
            nn.Linear(len(self.encoders) * hidden_dim, hidden_dim, bias_attr=True),
            nn.Silu(),
            nn.Linear(hidden_dim, hidden_dim, bias_attr=True),
        )

    def forward(self, batch_y, training=True):
        target_conditions = list(batch_y.keys())
        if set(target_conditions) != set(self.target_condition):
            raise ValueError(
                f"Expected conditions {self.target_condition}, "
                f"but got {target_conditions}"
            )

        first = list(batch_y.values())[0]
        if isinstance(first, (int, float)):
            batch_y = {
                k: ([v] if isinstance(v, (int, float)) else v)
                for k, v in batch_y.items()
            }
        batch_size = len(list(batch_y.values())[0])
        if training:
            drop_mask = paddle.rand([batch_size]) < self.p_unconditional
        else:
            drop_mask = paddle.concat(
                [paddle.zeros([batch_size]), paddle.ones([batch_size])]
            )
            drop_mask = drop_mask.astype("bool")
            batch_y = {k: _double_condition(v) for k, v in batch_y.items()}

        cond_embeds = []
        for cond_name, encoder in self.encoders.items():
            y = batch_y[cond_name]
            embed = encoder(y, drop_mask)
            cond_embeds.append(embed)

        cond_embeds = paddle.concat(cond_embeds, axis=-1)
        cond_embeds = self.proj(cond_embeds)
        return cond_embeds


class BaseEncoder(nn.Layer):
    def __init__(
        self,
        *,
        in_dim,
        hidden_dim,
        preprocess,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim

        self.preprocess = preprocess

        self.embedding = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias_attr=True),
            nn.Silu(),
            nn.Linear(hidden_dim, hidden_dim, bias_attr=True),
        )
        self.null_embed = paddle.create_parameter(
            shape=[1, in_dim],
            dtype="float32",
            default_initializer=nn.initializer.Normal(),
        )

    def forward(self, y, drop_mask=None):
        y = self.preprocess(y)
        if drop_mask is not None:
            y = paddle.where(
                drop_mask.unsqueeze(-1).expand_as(y), self.null_embed.expand_as(y), y
            )
        return self.embedding(y)


class ValueEncoder(BaseEncoder):
    def __init__(self, hidden_dim, mean=None, std=None):
        def preprocess(batch):
            if not isinstance(batch, paddle.Tensor):
                batch = paddle.to_tensor(batch, dtype="float32")
            batch = batch.reshape([-1]).astype("float32").unsqueeze(-1)
            if mean is not None and std is not None:
                batch = (batch - mean) / std
            return batch

        super().__init__(
            in_dim=1,
            hidden_dim=hidden_dim,
            preprocess=preprocess,
        )
        self.mean = mean
        self.std = std


class CategoricalEncoder(BaseEncoder):
    def __init__(self, in_dim, hidden_dim):
        def preprocess(batch):
            if isinstance(batch, list):
                idx = paddle.to_tensor(batch, dtype="int64")
            else:
                idx = batch.astype("int64")

            return paddle.nn.functional.one_hot(idx, num_classes=in_dim).astype(
                "float32"
            )

        super().__init__(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            preprocess=preprocess,
        )
        self.num_classes = in_dim


class _ElementEncoder(BaseEncoder):
    def __init__(self, in_dim, hidden_dim, fill_fn):
        def preprocess(batch):
            vals = [self._to_embeds(s) for s in batch]
            return paddle.concat(vals, axis=0)

        super().__init__(in_dim=in_dim, hidden_dim=hidden_dim, preprocess=preprocess)
        self._in_dim = in_dim
        self._fill = fill_fn

    def _to_embeds(self, s):
        from pymatgen.core import Element

        v = paddle.zeros([self._in_dim])
        return self._fill(v, s, Element).unsqueeze(0)


class CompositionEncoder(_ElementEncoder):
    def __init__(self, in_dim, hidden_dim):
        def fill(v, comp_str, Element):
            from pymatgen.core import Composition

            comp = Composition(comp_str).reduced_composition
            for el, amt in comp.get_el_amt_dict().items():
                atomic_number = Element(el).Z
                if atomic_number >= in_dim:
                    raise ValueError(
                        f"Atomic number {atomic_number} requires a larger in_dim."
                    )
                v[atomic_number] = float(amt)
            return v / v.sum()

        super().__init__(in_dim, hidden_dim, fill)


class ChemicalSystemEncoder(_ElementEncoder):
    def __init__(self, in_dim, hidden_dim):
        def fill(v, cs, Element):
            for el in cs.split("-"):
                atomic_number = Element(el).Z
                if atomic_number >= in_dim:
                    raise ValueError(
                        f"Atomic number {atomic_number} requires a larger in_dim."
                    )
                v[atomic_number] = 1.0
            return v

        super().__init__(in_dim, hidden_dim, fill)

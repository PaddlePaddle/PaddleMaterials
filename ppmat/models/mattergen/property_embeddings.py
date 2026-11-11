# 版权所有（C）2026 苏州国家实验室、百度飞桨团队
# 本代码由苏州国家实验室与百度飞桨团队合作开发完成。
# 其中，苏州国家实验室提供模型框架设计思路，百度飞桨团队负责代码实现。
# 双方对本代码及相关成果享有同等权利。

# Copyright (C) 2026 Suzhou National Laboratory and Baidu PaddlePaddle team
# This code was jointly developed by Suzhou National Laboratory and Baidu
# PaddlePaddle team.
# Suzhou National Laboratory provided the design concept for the model framework,
# and Baidu PaddlePaddle team was responsible for the code implementation.
# Both parties shall have equal rights to use this code and related work products.

# Licensed under the Apache License, Version 2.0 (the License);
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an AS IS BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import math
from collections import defaultdict
from functools import lru_cache
from typing import Dict
from typing import Sequence
from typing import TypeVar
from typing import Union

import paddle
import paddle.distributed as dist
from pymatgen.core import Element
from tqdm.auto import tqdm

from ppmat.models.mattergen.globals import _USE_UNCONDITIONAL_EMBEDDING
from ppmat.models.mattergen.globals import MAX_ATOMIC_NUM

TensorOrStringType = TypeVar("TensorOrStringType", paddle.Tensor, list[str])


@lru_cache
def get_atomic_number(symbol: str) -> int:
    return Element(symbol).Z


def get_use_unconditional_embedding(batch, cond_field: str) -> paddle.bool:
    """
    Returns
    -------
    paddle.BoolTensor, shape=(n_structures_in_batch, 1) -- whether to use the
        unconditional embedding for cond_field.  When True, we use unconditional
        embedding.

    NOTE: When _USE_UNCONDITIONAL_EMBEDDING is not in batch or cond_field is not
        in batch[_USE_UNCONDITIONAL_EMBEDDING] we return a paddle.BoolTensor with
        True values.
    """
    try:
        return batch[_USE_UNCONDITIONAL_EMBEDDING][cond_field]
    except KeyError:
        return paddle.ones_like(x=batch["num_atoms"], dtype="bool").reshape(-1, 1)


def tensor_is_not_nan(x: paddle.Tensor) -> paddle.bool:
    """
    Keyword arguments
    -----------------
    x: paddle.Tensor, shape = (n_structures_in_batch, Ndim) -- labels for a single
        conditional field. We assume that when a label is not present, the
        corresponding value is specified as paddle.nan.

    Returns
    -------
    paddle.BoolTensor, shape = (n_structures_in_batch,) -- index i is True if x[i]
        contains no NaNs
    """
    return paddle.all(
        x=paddle.reshape(
            x=paddle.logical_not(x=paddle.isnan(x=x)), shape=(tuple(x.shape)[0], -1)
        ),
        axis=1,
    )


def data_is_not_nan(
    x: Union[paddle.Tensor, list[str | None], list[list[str] | None]]
) -> paddle.bool:
    """
    Returns (n_structures_in_batch,) paddle.BoolTensor of whether the conditional values
    for a given property are not nan.

    """
    if isinstance(x, paddle.Tensor):
        return tensor_is_not_nan(x=x)
    else:
        return paddle.to_tensor(data=[(_x is not None) for _x in x])


class SetEmbeddingType:
    def __init__(self, p_unconditional: float, dropout_fields_iid: bool = False):
        """
        In PropertyEmbedding.forward we choose to concatenate either an unconditional
        embedding (ignores the value of a property) or a conditional embedding
        (depends on the value of a property) to the tensor that is input to the first
        node layer of each atom. This utility sets the internal state of batch_data to
        randomly select either the conditional or unconditional embedding for each
        structure in the batch.

        This utility operates in 2 modes:
        1) dropout_fields_iid = True -- We randomly assign which conditional fields are
            unconditional and which are conditional for fields that are not nan
            independently of whether all conditional fields are not nan for that
            structure. This means that for a structure conditioned on (y1,y2) we can
            generate embeddings corresponding to p(x), p(x|y1), p(x|y2), p(x|y1,y2).
        2) dropout_fields_iid = False - We assign conditional or unconditional
            embeddings to all conditional fields of a single structure simultaneously.
            This means that for a structure conditioned on (y1,y2) we can only generate
            embeddings corresponding to p(x) and p(|y1,y2).

        Keyword args:
        -------------
        p_unconditional: float -- the probability of using the unconditional embedding
            in the score model.
        dropout_fields_iid: bool -- whether to mask the conditional embedding of fields
            independently and identically distributed according to p_unconditional. If
            False, the score model is only exposed to two scenarios: 1) all conditional
            fields have their unconditional embedding. 2) all conditional fields have
            their conditional embedding. If True, the score model is exposed to all
            possible combinations of conditional fields having their unconditional or
            conditional embeddings, ie the score model will learn p(x), p(x|y1),
            p(x_y2), p(x|y1,y2),...

            Note: when dropout_fields_iid=False, the conditional embedding will only be
            used when all conditional fields have data present. If no single data point
            has data present for all conditional fields, then the score model will only
            be exposed to the unconditional embedding state p(x) and the joint
            p(x|y1,y2,...) will not be learned.
        """
        self.p_unconditional = p_unconditional
        self.dropout_fields_iid = dropout_fields_iid

    def __call__(self, x, cond_fields):
        if len(cond_fields) == 0:
            return x
        else:
            batch_size = len(x[cond_fields[0]])
            data_is_not_nan_dict: Dict[str, paddle.Tensor] = {
                cond_field: data_is_not_nan(x=x[cond_field])
                for cond_field in cond_fields
            }
            alldata_is_not_nan: paddle.bool = paddle.all(
                x=paddle.concat(
                    x=[
                        cond_data_not_nan.reshape(-1, 1)
                        for cond_data_not_nan in data_is_not_nan_dict.values()
                    ],
                    axis=1,
                ),
                axis=1,
            )
            use_unconditional_embedding: Dict[str, paddle.Tensor] = {}
            for cond_field in cond_fields:
                embedding_type = paddle.ones(shape=(batch_size, 1), dtype="bool")
                if self.dropout_fields_iid:
                    cond_data_is_not_nan = data_is_not_nan_dict[cond_field]
                else:
                    cond_data_is_not_nan = alldata_is_not_nan
                embedding_type[cond_data_is_not_nan] = (
                    paddle.rand(shape=(cond_data_is_not_nan.sum(), 1))
                    <= self.p_unconditional
                )
                use_unconditional_embedding[cond_field] = embedding_type
            return use_unconditional_embedding


class SetUnconditionalEmbeddingType:
    def __call__(self, x, cond_fields):
        use_unconditional_embedding = {
            cond_field: paddle.ones(shape=(len(x[cond_field]), 1), dtype="bool")
            for cond_field in cond_fields
        }
        return use_unconditional_embedding


class SetConditionalEmbeddingType:
    def __call__(self, x, cond_fields):
        use_unconditional_embedding = {}
        for cond_field in cond_fields:
            use_unconditional_embedding[cond_field] = paddle.zeros(
                shape=(len(x[cond_field]), 1), dtype="bool"
            )
        return use_unconditional_embedding


class BaseUnconditionalEmbeddingModule(paddle.nn.Layer):
    only_depends_on_shape_of_input: bool
    hidden_dim: int


class EmbeddingVector(BaseUnconditionalEmbeddingModule):
    only_depends_on_shape_of_input: bool = True

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.embedding = paddle.nn.Embedding(num_embeddings=1, embedding_dim=hidden_dim)
        self.hidden_dim = hidden_dim

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        """
        This forward depends only on the shape of x and returns a tensor of zeros.
        """
        return self.embedding(paddle.zeros(shape=len(x), dtype="int64"))


class SpaceGroupEmbeddingVector(BaseUnconditionalEmbeddingModule):
    only_depends_on_shape_of_input: bool = True

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.embedding = paddle.nn.Embedding(
            num_embeddings=230, embedding_dim=hidden_dim
        )
        self.hidden_dim = hidden_dim

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        """
        Return embedding of the space group, 1 is subtracted from the space group
        number to make it zero-indexed.
        """
        return self.embedding(x.astype(dtype="int64") - 1)


class ZerosEmbedding(BaseUnconditionalEmbeddingModule):
    """
    Return a [n_crystals_in_batch, self.hidden_dim] tensor of zeros. This is helpfuln
    as the unconditional embedding for a property included in the adapter module if we
    do not want to change the unconditional score of the base model when properties are
    added in the adapter module.
    """

    only_depends_on_shape_of_input: bool = True

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim

    def forward(self, x: (paddle.Tensor | list[str])) -> paddle.Tensor:
        """
        This forward depends only on the shape of x.
        """
        return paddle.zeros(shape=[len(x), self.hidden_dim])


class ChemicalSystemMultiHotEmbedding(paddle.nn.Layer):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embedding = paddle.nn.Linear(
            in_features=MAX_ATOMIC_NUM + 1, out_features=hidden_dim
        )

    @property
    def device(self):
        return self.parameters()[0].place
        # return next(self.parameters()).place

    @staticmethod
    def _sequence_to_multi_hot(
        x: Sequence[str], device: (paddle.CPUPlace, paddle.CUDAPlace, str)
    ) -> paddle.Tensor:
        """
        Converts a sequence of unique elements present in a single structure to a
        multi-hot vectors of 1s (present) and 0s (not present) for each unique element.

        Returns
        -------
        paddle.Tensor, shape = (1, MAX_ATOMIC_NUM + 1)
        """
        chemical_system_numbers: paddle.int64 = paddle.to_tensor(
            data=[get_atomic_number(symbol=_element) for _element in x],
            dtype="int64",
            place=device,
        )
        chemical_system_condition = paddle.zeros(shape=MAX_ATOMIC_NUM + 1)
        chemical_system_condition[chemical_system_numbers] = 1.0
        return chemical_system_condition.reshape(1, -1)

    @staticmethod
    def sequences_to_multi_hot(
        x: list[list[str]], device: (paddle.CPUPlace, paddle.CUDAPlace, str)
    ) -> paddle.Tensor:
        """
        Convert a list of sequences of unique elements present in a list of structures
        to a multi-hot tensor of 1s (present) and 0s (not present) for each unique
        element.

        Returns
        -------
        paddle.Tensor, shape = (n_structures_in_batch, MAX_ATOMIC_NUM + 1)
        """
        return paddle.concat(
            x=[
                ChemicalSystemMultiHotEmbedding._sequence_to_multi_hot(
                    _x, device=device
                )
                for _x in x
            ],
            axis=0,
        )

    @staticmethod
    def convert_to_list_of_str(x: (list[str] | list[list[str]])) -> list[list[str]]:
        """
        Returns
        -------
        list[list[str]] -- a list of length n_structures_in_batch of chemical systems
        for each structure where the chemical system is specified as a list of unique
        elements in the structure.
        """
        if isinstance(x[0], str):
            x = [_x.split("-") for _x in x if isinstance(_x, str)]
        return x

    def forward(self, x: (list[str] | list[list[str]])) -> paddle.Tensor:
        """
        Keyword arguments
        -----------------
        x: Union[list[str], list[Sequence[str]]] -- if elements are a string, they are
            assumed to be a '-' delimited list of unique elements. If a sequence of
            strings, it is assumed to be a list of unique elements in the structure.
        """
        x = self.convert_to_list_of_str(x=x)
        multi_hot_representation: paddle.Tensor = self.sequences_to_multi_hot(
            x=x, device=self.device
        )
        return self.embedding(multi_hot_representation)


def paddle_nanstd(x: paddle.Tensor, dim: int, unbiased: bool) -> paddle.Tensor:
    data_is_present = paddle.all(
        x=paddle.reshape(
            x=paddle.logical_not(x=paddle.isnan(x=x)), shape=(tuple(x.shape)[0], -1)
        ),  # noqa
        axis=1,
    )
    return paddle.std(x=x[data_is_present], axis=dim, unbiased=unbiased)


class StandardScalerPaddle(paddle.nn.Layer):
    """Normalizes the targets of a dataset."""

    def __init__(
        self,
        means: (paddle.Tensor | None) = None,
        stds: (paddle.Tensor | None) = None,
        stats_dim: tuple[int] = (1,),
    ):
        super().__init__()
        self.register_buffer(
            name="means",
            tensor=paddle.atleast_1d(means)
            if means is not None
            else paddle.zeros(shape=stats_dim),  # noqa
        )
        self.register_buffer(
            name="stds",
            tensor=paddle.atleast_1d(stds)
            if stds is not None
            else paddle.ones(shape=stats_dim),  # noqa
        )

    def fit(self, X: paddle.Tensor):
        means: paddle.Tensor = paddle.atleast_1d(paddle.nanmean(x=X, axis=0))  # noqa
        stds: paddle.Tensor = paddle.atleast_1d(
            paddle_nanstd(X, dim=0, unbiased=False) + 1e-5
        )
        assert tuple(means.shape) == tuple(
            self.means.shape
        ), f"Mean shape mismatch: {tuple(means.shape)} != {tuple(self.means.shape)}"
        assert tuple(stds.shape) == tuple(
            self.stds.shape
        ), f"Std shape mismatch: {tuple(stds.shape)} != {tuple(self.stds.shape)}"
        self.means = means
        self.stds = stds

    def transform(self, X: paddle.Tensor) -> paddle.Tensor:
        assert self.means is not None and self.stds is not None
        return (X - self.means) / self.stds

    def inverse_transform(self, X: paddle.Tensor) -> paddle.Tensor:
        assert self.means is not None and self.stds is not None
        return X * self.stds + self.means

    def copy(self) -> "StandardScalerPaddle":
        return StandardScalerPaddle(
            means=self.means.clone().detach(), stds=self.stds.clone().detach()
        )

    def forward(self, X: paddle.Tensor) -> paddle.Tensor:
        return self.transform(X)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(means: {self.means.tolist() if self.means is not None else None}, stds: {self.stds.tolist() if self.stds is not None else None})"  # noqa


class PropertyEmbedding(paddle.nn.Layer):
    def __init__(
        self,
        # name: str,
        conditional_embedding_module_name: str,
        conditional_embedding_module_cfg: dict,
        unconditional_embedding_module_name: str,
        unconditional_embedding_module_cfg: dict,
        scaler_name: str,
        scaler_cfg: dict,
    ):
        super().__init__()
        # self.name = name
        self.conditional_embedding_module_name = conditional_embedding_module_name
        self.conditional_embedding_module_cfg = conditional_embedding_module_cfg
        self.unconditional_embedding_module_name = unconditional_embedding_module_name
        self.unconditional_embedding_module_cfg = unconditional_embedding_module_cfg
        self.scaler_name = scaler_name
        self.scaler_cfg = scaler_cfg

        if conditional_embedding_module_name == "ChemicalSystemMultiHotEmbedding":
            self.conditional_embedding_module = ChemicalSystemMultiHotEmbedding(
                **conditional_embedding_module_cfg
            )
        elif conditional_embedding_module_name == "NoiseLevelEncoding":
            self.conditional_embedding_module = NoiseLevelEncoding(
                **conditional_embedding_module_cfg
            )
        elif conditional_embedding_module_name == "SpaceGroupEmbeddingVector":
            self.conditional_embedding_module = SpaceGroupEmbeddingVector(
                **conditional_embedding_module_cfg
            )
        else:
            raise ValueError(
                "Invalid conditional_embedding_module_name: "
                f"{conditional_embedding_module_name}"
            )

        if unconditional_embedding_module_name == "ZerosEmbedding":
            self.unconditional_embedding_module = ZerosEmbedding(
                **unconditional_embedding_module_cfg
            )
        else:
            raise ValueError(
                "Invalid unconditional_embedding_module_name: "
                f"{unconditional_embedding_module_name}"
            )

        if scaler_name == "Identity":
            self.scaler = paddle.nn.Identity(**scaler_cfg)
        elif scaler_name == "StandardScalerPaddle":
            self.scaler = StandardScalerPaddle(**scaler_cfg)
        else:
            raise ValueError(f"Invalid scaler_name: {scaler_name}")

    def forward(self, data, use_unconditional_embedding) -> paddle.Tensor:
        if (
            paddle.all(x=use_unconditional_embedding)
            and self.unconditional_embedding_module.only_depends_on_shape_of_input
        ):
            return self.unconditional_embedding_module(x=data)
        else:
            # data = batch[self.name]
            if isinstance(data, paddle.Tensor) and data.dim() == 2:
                data = data.squeeze(axis=-1)
            data = self.scaler(data)
            conditional_embedding: paddle.Tensor = self.conditional_embedding_module(
                data
            )
            unconditional_embedding: paddle.Tensor = (
                self.unconditional_embedding_module(x=data)
            )
            return paddle.where(
                condition=use_unconditional_embedding,
                x=unconditional_embedding,
                y=conditional_embedding,
            )

    def fit_scaler(self, all_data):
        if isinstance(self.scaler, paddle.nn.Identity):
            return
        self.scaler.fit(all_data)


def get_property_embeddings(
    batch, property_embeddings: paddle.nn.LayerDict
) -> paddle.Tensor:
    """
    Keyword arguments
    -----------------
    property_embeddings: paddle.nn.ModuleDict[PropertyToConditonOn, PropertyEmbedding]
        -- a dictionary of property embeddings. The keys are the names of the
        conditional fields in the batch.
    """
    ordered_keys = sorted(property_embeddings.keys())
    if len(ordered_keys) > 0:
        return paddle.concat(
            x=[property_embeddings[k].forward(batch=batch) for k in ordered_keys],
            axis=-1,
        )
    else:
        return paddle.to_tensor(data=[], place=batch["num_atoms"].place)


def set_conditional_property_values(batch, properties: str):
    not_numeric = [k for k, v in properties.items() if not isinstance(v, (int, float))]
    cond_values = {
        k: (
            [properties[k]] * len(batch["num_atoms"])
            if k in not_numeric
            else paddle.full_like(x=batch["num_atoms"], fill_value=v).reshape(-1, 1)
        )
        for k, v in properties.items()
    }
    return batch.replace(**cond_values)


def maybe_to_tensor(values: list[TensorOrStringType]) -> TensorOrStringType:
    if isinstance(values[0], paddle.Tensor):
        return paddle.concat(x=values)
    return [el for x in values for el in x]


class SetPropertyScalers:
    """
    Utility callback; at the start of training, this computes the mean and std of the
    property data and adds the property scalers to the model.
    """

    @staticmethod
    def _compute_property_scalers(
        train_dataloader,
        property_embeddings: paddle.nn.LayerDict,
    ):
        property_values = defaultdict(list)
        property_names = [
            name
            for name, p in property_embeddings.items()
            if not isinstance(p.scaler, paddle.nn.Identity)
        ]
        if len(property_names) == 0:
            return
        for batch in tqdm(train_dataloader, desc="Fitting property scalers"):
            for property_name in property_names:
                property_values[property_name].append(batch[property_name])
        for property_name in property_names:
            values = maybe_to_tensor(values=property_values[property_name])
            if dist.is_initialized():
                if isinstance(values, paddle.Tensor):
                    values_list = []
                    dist.all_gather(values_list, values)
                    values = paddle.concat(x=values_list)
                else:
                    print(f"Property {property_name} cannot be gathered")
            property_embeddings[property_name].fit_scaler(all_data=values)

    def on_fit_start(self, train_dataloader, model):
        model = model.model
        self._compute_property_scalers(
            train_dataloader=train_dataloader,
            property_embeddings=model.property_embeddings,
        )
        if hasattr(model, "property_embeddings_adapt"):
            self._compute_property_scalers(
                train_dataloader=train_dataloader,
                property_embeddings=model.property_embeddings_adapt,
            )


class NoiseLevelEncoding(paddle.nn.Layer):
    def __init__(self, d_model: int, dropout: float = 0.0):
        super().__init__()
        self.dropout = paddle.nn.Dropout(p=dropout)
        self.d_model = d_model
        div_term = paddle.exp(
            x=paddle.arange(start=0, end=d_model, step=2)
            * (-math.log(10000.0) / d_model)
        )
        self.register_buffer(name="div_term", tensor=div_term)

    def forward(self, t: paddle.Tensor) -> paddle.Tensor:
        """
        Args:
            t: Tensor, shape [batch_size]
        """
        x = paddle.zeros(shape=(tuple(t.shape)[0], self.d_model))
        x[:, 0::2] = paddle.sin(x=t[:, None] * self.div_term[None])
        x[:, 1::2] = paddle.cos(x=t[:, None] * self.div_term[None])
        return self.dropout(x)

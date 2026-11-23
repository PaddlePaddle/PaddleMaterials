# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

"""PaddleMaterials interface for SevenNet."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from collections.abc import Sequence

import paddle

from ppmat.models.sevennet import _keys as KEY
from ppmat.models.sevennet.model_build import build_E3_equivariant_model
from ppmat.models.sevennet.parse_input import init_model_config


EV_A3_TO_GPA = 160.21766208


class SevenNet(paddle.nn.Layer):
    """SevenNet interatomic potential for PaddleMaterials.

    The internal SevenNet core keeps its native physical outputs:

    - total energy: eV
    - force: eV / Angstrom
    - stress: eV / Angstrom^3 in 6-component Voigt form

    The PaddleMaterials training interface exposes:

    - energy_per_atom: eV / atom
    - force: eV / Angstrom
    - stress: GPa in [B, 3, 3] form
    """

    def __init__(
        self,
        model_config: dict,
        parallel: bool = False,
        auto_batch: bool = True,
        property_names: Sequence[str] = (
            "energy_per_atom",
            "force",
            "stress",
        ),
        loss_type: str = "huber_loss",
        huber_loss_delta: float = 0.01,
        loss_weights: dict | None = None,
    ):
        super().__init__()
        if parallel:
            raise NotImplementedError("SevenNet currently supports only the serial model")

        if not isinstance(
            model_config,
            Mapping,
        ):
            raise TypeError(
                "model_config must be a mapping, "
                f"but got {type(model_config).__name__}."
            )

        config = copy.deepcopy(
            dict(model_config)
        )

        if KEY.TYPE_MAP not in config or KEY.NUM_SPECIES not in config:
            config = init_model_config(config)

        type_map = config.get(
            "_type_map"
        )

        if isinstance(
            type_map,
            dict,
        ):
            try:
                config["_type_map"] = {
                    int(key): int(value)
                    for key, value
                    in type_map.items()
                }
            except (
                TypeError,
                ValueError,
            ):
                pass

        config[
            "use_flash_tp"
        ] = False

        self.model_config = config

        self.parallel = bool(
            parallel
        )

        self.auto_batch = bool(
            auto_batch
        )

        self._manual_batch_mode = False

        self.property_names = list(
            property_names
        )

        supported = {
            "energy_per_atom",
            "force",
            "stress",
        }

        unsupported = (
            set(
                self.property_names
            )
            - supported
        )

        if unsupported:
            raise ValueError(
                "Unsupported SevenNet properties: "
                f"{sorted(unsupported)}"
            )

        default_weights = {
            "energy_per_atom": 1.0,
            "force": 1.0,
            "stress": 0.01,
        }

        if loss_weights is not None:
            default_weights.update(
                loss_weights
            )

        self.loss_weights_dict = {
            key: float(
                default_weights[key]
            )
            for key
            in self.property_names
        }

        self.loss_type = loss_type

        if loss_type in {
            "huber_loss",
            "smooth_l1_loss",
            "Huber",
        }:
            self.loss_fn = (
                paddle.nn.SmoothL1Loss(
                    delta=huber_loss_delta
                )
            )

        elif loss_type in {
            "mse_loss",
            "mse",
        }:
            self.loss_fn = (
                paddle.nn.MSELoss()
            )

        elif loss_type in {
            "l1_loss",
            "l1",
        }:
            self.loss_fn = (
                paddle.nn.L1Loss()
            )

        else:
            raise ValueError(
                f"Unknown loss type: {loss_type}"
            )

        self.huber_loss_delta = float(
            huber_loss_delta
        )

        self.core = (
            build_E3_equivariant_model(
                config,
                parallel=self.parallel,
            )
        )

    def set_is_batch_data(
        self,
        is_batch_data: bool,
    ):
        self._manual_batch_mode = bool(
            is_batch_data
        )

        self.core.set_is_batch_data(
            self._manual_batch_mode
        )

    def _resolve_batch_mode(
        self,
        data,
    ) -> bool:

        if self.auto_batch:

            if KEY.BATCH not in data:
                return False

            batch = data[
                KEY.BATCH
            ]

            if batch is None:
                return False

            if hasattr(
                batch,
                "shape",
            ):
                shape = tuple(
                    batch.shape
                )

                if len(shape) == 0:
                    return False

                if shape[0] == 0:
                    return False

            return True

        return self._manual_batch_mode

    def _prepare_data(
        self,
        data,
    ):
        """Extract and tensorize SevenNet graph fields.

        Dataset/cache graphs may contain NumPy arrays. The internal SevenNet
        core receives Paddle tensors only.
        """

        if isinstance(
            data,
            Mapping,
        ):

            if "graph" in data:

                graph = data[
                    "graph"
                ]

                if hasattr(
                    graph,
                    "to_dict",
                ):
                    prepared = dict(
                        graph.to_dict()
                    )

                elif isinstance(
                    graph,
                    Mapping,
                ):
                    prepared = dict(
                        graph
                    )

                else:
                    raise TypeError(
                        "data['graph'] must be a PaddleMaterials "
                        "Data/Batch object or mapping, "
                        f"but got {type(graph).__name__}."
                    )

            else:
                prepared = dict(
                    data
                )

        elif hasattr(
            data,
            "to_dict",
        ):

            prepared = dict(
                data.to_dict()
            )

        else:

            raise TypeError(
                "SevenNet expects a mapping or PaddleMaterials "
                "Data/Batch, "
                f"but got {type(data).__name__}."
            )


        # These names are the actual SevenNet graph fields.
        integer_keys = (
            "x",
            "edge_index",
            "atomic_numbers",
            "batch",
            "ptr",
            "num_atoms",
        )

        float_keys = (
            "pos",
            "edge_vec",
            "cell_volume",
        )


        for key in integer_keys:

            if (
                key not in prepared
                or prepared[key] is None
            ):
                continue

            value = prepared[
                key
            ]

            if not isinstance(
                value,
                paddle.Tensor,
            ):

                value = paddle.to_tensor(
                    value,
                    dtype="int64",
                )

            elif value.dtype != paddle.int64:

                value = value.astype(
                    "int64"
                )

            prepared[
                key
            ] = value


        for key in float_keys:

            if (
                key not in prepared
                or prepared[key] is None
            ):
                continue

            value = prepared[
                key
            ]

            if not isinstance(
                value,
                paddle.Tensor,
            ):

                value = paddle.to_tensor(
                    value,
                    dtype="float32",
                )

            elif value.dtype != paddle.float32:

                value = value.astype(
                    "float32"
                )

            prepared[
                key
            ] = value


        if "edge_vec" not in prepared:

            raise KeyError(
                "SevenNet input requires 'edge_vec'."
            )


        edge_vec = prepared[
            "edge_vec"
        ]


        # SevenNet derives forces and stresses through edge_vec.
        if edge_vec.stop_gradient:

            edge_vec = edge_vec.detach()

            edge_vec.stop_gradient = False


        prepared[
            "edge_vec"
        ] = edge_vec


        return prepared

    def _forward(
        self,
        data,
    ):
        """Run native SevenNet core.

        Returns native SevenNet units so that numerical parity with the
        upstream model remains directly testable.
        """

        prepared = self._prepare_data(
            data
        )

        is_batch_data = (
            self._resolve_batch_mode(
                prepared
            )
        )

        self.core.set_is_batch_data(
            is_batch_data
        )

        output = self.core(
            prepared
        )

        return {
            "energy": output[
                KEY.PRED_TOTAL_ENERGY
            ],
            "force": output[
                KEY.PRED_FORCE
            ],
            "stress": output[
                KEY.PRED_STRESS
            ],
        }

    @staticmethod
    def _stress_voigt_to_matrix(
        stress,
    ):
        """Convert SevenNet [xx, yy, zz, xy, yz, xz] to [B, 3, 3]."""

        squeeze = False

        if stress.ndim == 1:

            stress = stress.reshape(
                [1, 6]
            )

            squeeze = True

        if (
            stress.ndim != 2
            or stress.shape[-1] != 6
        ):
            raise ValueError(
                "SevenNet stress must have shape "
                "[6] or [B, 6], "
                f"but got {tuple(stress.shape)}."
            )

        xx = stress[:, 0]
        yy = stress[:, 1]
        zz = stress[:, 2]
        xy = stress[:, 3]
        yz = stress[:, 4]
        xz = stress[:, 5]

        row0 = paddle.stack(
            [
                xx,
                xy,
                xz,
            ],
            axis=-1,
        )

        row1 = paddle.stack(
            [
                xy,
                yy,
                yz,
            ],
            axis=-1,
        )

        row2 = paddle.stack(
            [
                xz,
                yz,
                zz,
            ],
            axis=-1,
        )

        matrix = paddle.stack(
            [
                row0,
                row1,
                row2,
            ],
            axis=1,
        )

        if squeeze:
            return matrix[0]

        return matrix

    def _to_training_predictions(
        self,
        data,
        raw_prediction,
    ):
        """Convert native SevenNet outputs to PaddleMaterials MPtrj convention."""

        total_energy = raw_prediction[
            "energy"
        ].reshape(
            [-1]
        )

        # Resolve graph fields from either the legacy flat graph mapping or
        # the standard PaddleMaterials {"graph": Batch, labels...} format.
        if isinstance(
            data,
            Mapping,
        ) and KEY.NUM_ATOMS in data:

            num_atoms = data[
                KEY.NUM_ATOMS
            ]

        elif (
            isinstance(data, Mapping)
            and "graph" in data
        ):

            graph = data[
                "graph"
            ]

            if hasattr(
                graph,
                "to_dict",
            ):
                graph_data = graph.to_dict()

            elif isinstance(
                graph,
                Mapping,
            ):
                graph_data = graph

            else:
                raise TypeError(
                    "data['graph'] must be a PaddleMaterials Data/Batch "
                    "object or a mapping."
                )

            if KEY.NUM_ATOMS not in graph_data:
                raise KeyError(
                    f"SevenNet training requires '{KEY.NUM_ATOMS}' "
                    "in data['graph']."
                )

            num_atoms = graph_data[
                KEY.NUM_ATOMS
            ]

        elif hasattr(
            data,
            "to_dict",
        ):

            graph_data = data.to_dict()

            if KEY.NUM_ATOMS not in graph_data:
                raise KeyError(
                    f"SevenNet training requires '{KEY.NUM_ATOMS}'."
                )

            num_atoms = graph_data[
                KEY.NUM_ATOMS
            ]

        else:
            raise KeyError(
                f"SevenNet training requires '{KEY.NUM_ATOMS}'."
            )

        if not isinstance(
            num_atoms,
            paddle.Tensor,
        ):
            num_atoms = paddle.to_tensor(
                num_atoms
            )

        num_atoms = (
            num_atoms.reshape([-1])
            .astype(
                total_energy.dtype
            )
        )

        if (
            total_energy.shape[0]
            != num_atoms.shape[0]
        ):
            raise ValueError(
                "Energy / num_atoms batch mismatch: "
                f"energy={tuple(total_energy.shape)}, "
                f"num_atoms={tuple(num_atoms.shape)}"
            )

        energy_per_atom = (
            total_energy
            / num_atoms
        )

        stress_gpa = (
            self._stress_voigt_to_matrix(
                raw_prediction[
                    "stress"
                ]
            )
            * EV_A3_TO_GPA
        )

        return {
            "energy_per_atom":
                energy_per_atom,

            "force":
                raw_prediction[
                    "force"
                ],

            "stress":
                stress_gpa,
        }

    @staticmethod
    def _align_label_shape(
        label,
        pred,
        property_name,
    ):

        if not isinstance(
            label,
            paddle.Tensor,
        ):
            label = paddle.to_tensor(
                label
            )

        label = label.astype(
            pred.dtype
        )

        if tuple(
            label.shape
        ) == tuple(
            pred.shape
        ):
            return label

        if label.numel() == pred.numel():

            return label.reshape(
                pred.shape
            )

        raise ValueError(
            f"Shape mismatch for {property_name}: "
            f"label={tuple(label.shape)}, "
            f"prediction={tuple(pred.shape)}"
        )

    def forward(
        self,
        data,
        return_loss: bool = True,
        return_prediction: bool = True,
    ):
        """PaddleMaterials training / evaluation interface."""

        if not (
            return_loss
            or return_prediction
        ):
            raise ValueError(
                "At least one of return_loss or "
                "return_prediction must be True."
            )

        raw_prediction = (
            self._forward(
                data
            )
        )

        property_prediction = (
            self._to_training_predictions(
                data,
                raw_prediction,
            )
        )

        loss_dict = {}

        if return_loss:

            total_loss = None

            for property_name in (
                self.property_names
            ):

                if property_name not in data:
                    raise KeyError(
                        f"Training label "
                        f"'{property_name}' "
                        "is missing."
                    )

                pred = property_prediction[
                    property_name
                ]

                label = (
                    self._align_label_shape(
                        data[
                            property_name
                        ],
                        pred,
                        property_name,
                    )
                )

                valid_mask = (
                    ~paddle.isnan(
                        label
                    )
                )

                valid_label = (
                    paddle.masked_select(
                        label,
                        valid_mask,
                    )
                )

                valid_pred = (
                    paddle.masked_select(
                        pred,
                        valid_mask,
                    )
                )

                if (
                    valid_label.numel()
                    == 0
                ):
                    continue

                loss_property = (
                    self.loss_fn(
                        input=valid_pred,
                        label=valid_label,
                    )
                )

                loss_dict[
                    property_name
                ] = loss_property

                weighted = (
                    loss_property
                    * self.loss_weights_dict[
                        property_name
                    ]
                )

                total_loss = (
                    weighted
                    if total_loss is None
                    else total_loss
                    + weighted
                )

            if total_loss is None:
                raise ValueError(
                    "No valid training labels "
                    "were found in this batch."
                )

            loss_dict[
                "loss"
            ] = total_loss

        prediction = {}

        if return_prediction:

            for property_name in (
                self.property_names
            ):
                prediction[
                    property_name
                ] = property_prediction[
                    property_name
                ]

        return {
            "loss_dict":
                loss_dict,

            "pred_dict":
                prediction,
        }

    def predict(
        self,
        samples,
    ):
        """Predict native SevenNet energy / force / stress.

        Force and stress require automatic differentiation, therefore this
        method must not be globally wrapped by paddle.no_grad().
        """

        is_sequence = isinstance(
            samples,
            (
                list,
                tuple,
            ),
        )

        sample_list = (
            list(samples)
            if is_sequence
            else [samples]
        )

        results = []

        for sample in sample_list:

            prediction = (
                self._forward(
                    sample
                )
            )

            result = {
                key:
                    value.numpy()

                for key, value
                in prediction.items()
            }

            results.append(
                result
            )

        return (
            results
            if is_sequence
            else results[0]
        )

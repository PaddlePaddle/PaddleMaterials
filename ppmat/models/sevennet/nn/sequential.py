# -*- coding: utf-8 -*-

import warnings
from collections import OrderedDict
from typing import Dict, Optional

import paddle
import paddle.nn as nn

from ppmat.models.sevennet import _keys as KEY
from ppmat.models.sevennet._const import AtomGraphDataType


def _instantiate_modules(modules):
    """
    SevenNet lazy-layer initialization.

    IrrepsLinear and SelfConnectionLinearIntro are created lazily.
    They must be instantiated before being registered into the
    top-level AtomGraphSequential container.
    """
    for module in modules.values():
        if not getattr(module, "layer_instantiated", True):
            module.instantiate()


class _ModalInputPrepare(nn.Layer):

    def __init__(
        self,
        modal_idx: int,
    ):
        super().__init__()
        self.modal_idx = int(modal_idx)

    def forward(
        self,
        data: AtomGraphDataType,
    ) -> AtomGraphDataType:

        data[KEY.MODAL_TYPE] = paddle.to_tensor(
            self.modal_idx,
            dtype="int64",
        )

        return data


class AtomGraphSequential(nn.Sequential):
    """
    Pure Paddle wrapper of the SevenNet model.

    Important:
      1. instantiate lazy SevenNet modules first;
      2. register only top-level graph modules;
      3. do not iterate recursively over Paddle sublayers;
      4. preserve SevenNet atomic-number mapping;
      5. preserve force/stress gradient input handling.
    """

    def __init__(
        self,
        modules: Dict[str, nn.Layer],
        cutoff: float = 0.0,
        type_map: Optional[Dict[int, int]] = None,
        modal_map: Optional[Dict[str, int]] = None,
        eval_type_map: bool = True,
        eval_modal_map: bool = False,
        data_key_atomic_numbers: str = KEY.ATOMIC_NUMBERS,
        data_key_node_feature: str = KEY.NODE_FEATURE,
        data_key_grad: Optional[str] = None,
        **kwargs,
    ):

        if not isinstance(modules, OrderedDict):
            modules = OrderedDict(modules)

        # ----------------------------------------------------------
        # Critical SevenNet step:
        # instantiate IrrepsLinear / SelfConnectionLinearIntro etc.
        # ----------------------------------------------------------
        _instantiate_modules(modules)

        # Paddle parent must be initialized before assigning
        # parameters / buffers / sublayers to self.
        super().__init__()

        self.cutoff = float(cutoff)

        # Normalize YAML keys such as "72" -> 72 if necessary.
        if type_map is not None:
            self.type_map = {
                int(z): int(index)
                for z, index in type_map.items()
            }
        else:
            self.type_map = None

        self.eval_type_map = bool(eval_type_map)
        self.modal_map = modal_map
        self.eval_modal_map = bool(eval_modal_map)

        self.key_atomic_numbers = data_key_atomic_numbers
        self.key_node_feature = data_key_node_feature
        self.key_grad = data_key_grad

        self.is_batch_data = True

        if self.cutoff == 0.0:
            warnings.warn(
                "cutoff is 0.0 or not given",
                UserWarning,
            )

        # ----------------------------------------------------------
        # Atomic number -> SevenNet species index lookup
        # ----------------------------------------------------------

        if self.type_map is None:

            warnings.warn(
                "type_map is not given",
                UserWarning,
            )

            self.eval_type_map = False

        else:

            z_to_onehot = [-1] * 120

            for z, onehot in self.type_map.items():

                if z < 0 or z >= len(z_to_onehot):
                    raise ValueError(
                        f"Atomic number {z} is outside supported range"
                    )

                z_to_onehot[z] = onehot

            z_to_onehot_tensor = paddle.to_tensor(
                z_to_onehot,
                dtype="int64",
            )

            self.register_buffer(
                "z_to_onehot_tensor",
                z_to_onehot_tensor,
            )

        if self.eval_modal_map and self.modal_map is None:
            raise ValueError(
                "eval_modal_map is True but modal_map is None"
            )

        # ----------------------------------------------------------
        # Register ONLY top-level SevenNet modules.
        #
        # Do not use self.sublayers() during forward because Paddle
        # recursively exposes BesselBasis / XPLORCutoff / e3nn child
        # layers as well.
        # ----------------------------------------------------------

        for name, module in modules.items():

            if not isinstance(module, paddle.nn.Layer):
                raise TypeError(
                    f"{name} is not a paddle.nn.Layer: {type(module)}"
                )

            self.add_sublayer(
                str(name),
                module,
            )


    def set_is_batch_data(
        self,
        flag: bool,
    ):
        """
        Set whether graph input is batched.

        Several SevenNet modules alter their behaviour depending
        on whether the input is a batch or a single structure.
        """

        flag = bool(flag)

        for module in self._sub_layers.values():

            if hasattr(module, "_is_batch_data"):
                module._is_batch_data = flag

        self.is_batch_data = flag


    def get_irreps_in(
        self,
        module_name: str,
        attr_key: str = "irreps_in",
    ):

        module = self._sub_layers[module_name]

        if hasattr(module, attr_key):
            return repr(
                getattr(module, attr_key)
            )

        for child in module.sublayers():

            if hasattr(child, attr_key):
                return repr(
                    getattr(child, attr_key)
                )

        return None


    def prepand_module(
        self,
        key: str,
        module: nn.Layer,
    ):
        """
        Compatibility helper for original SevenNet API.
        """

        new_layers = OrderedDict()

        new_layers[key] = module

        for name, old_module in self._sub_layers.items():
            new_layers[name] = old_module

        self._sub_layers = new_layers


    def replace_module(
        self,
        key: str,
        module: nn.Layer,
    ):

        if key not in self._sub_layers:
            raise KeyError(
                f"Unknown module: {key}"
            )

        self._sub_layers[key] = module


    def delete_module_by_key(
        self,
        key: str,
    ):

        if key in self._sub_layers:
            del self._sub_layers[key]


    def _atomic_numbers_to_onehot(
        self,
        atomic_numbers: paddle.Tensor,
    ):

        if self.type_map is None:
            raise RuntimeError(
                "type_map is not available"
            )

        atomic_numbers = paddle.cast(
            atomic_numbers,
            dtype="int64",
        )

        # Paddle index lookup.
        onehot = paddle.gather(
            self.z_to_onehot_tensor,
            atomic_numbers.reshape([-1]),
            axis=0,
        )

        if bool(
            paddle.any(onehot < 0).item()
        ):
            bad_z = atomic_numbers[
                onehot < 0
            ].numpy().tolist()

            raise ValueError(
                f"Unsupported atomic numbers: {bad_z}"
            )

        return onehot


    def _eval_modal_map(
        self,
        data: AtomGraphDataType,
    ):

        if self.modal_map is None:
            raise RuntimeError(
                "modal_map is not available"
            )

        if not self.is_batch_data:

            modal = data[KEY.DATA_MODALITY]

            modal_idx = self.modal_map[
                modal
            ]

            data[KEY.MODAL_TYPE] = paddle.to_tensor(
                modal_idx,
                dtype="int64",
            )

        else:

            modal_values = data[
                KEY.DATA_MODALITY
            ]

            modal_idx = [
                self.modal_map[item]
                for item in modal_values
            ]

            data[KEY.MODAL_TYPE] = paddle.to_tensor(
                modal_idx,
                dtype="int64",
            )


    def _preprocess(
        self,
        data: AtomGraphDataType,
    ) -> AtomGraphDataType:

        if self.eval_type_map:

            atomic_numbers = data[
                self.key_atomic_numbers
            ]

            onehot = self._atomic_numbers_to_onehot(
                atomic_numbers
            )

            data[
                self.key_node_feature
            ] = onehot

        if self.eval_modal_map:
            self._eval_modal_map(data)

        # Enable gradient tracking in Paddle:
        # stop_gradient=False
        if self.key_grad is not None:

            if self.key_grad not in data:
                raise KeyError(
                    f"Gradient key '{self.key_grad}' "
                    "is not present in graph data"
                )

            data[
                self.key_grad
            ].stop_gradient = False

        return data


    def prepare_modal_deploy(
        self,
        modal: str,
    ):

        if self.modal_map is None:
            return

        self.eval_modal_map = False

        self.set_is_batch_data(
            False
        )

        modal_idx = self.modal_map[
            modal
        ]

        self.prepand_module(
            "modal_input_prepare",
            _ModalInputPrepare(
                modal_idx
            ),
        )


    def forward(
        self,
        input: AtomGraphDataType,
    ) -> AtomGraphDataType:

        data = self._preprocess(
            input
        )

        # Only execute TOP-LEVEL SevenNet layers.
        # Using self.sublayers() here would recursively run internal
        # BesselBasis / XPLORCutoff / e3nn layers on the whole dict.
        for module in self._sub_layers.values():

            data = module(
                data
            )

        return data

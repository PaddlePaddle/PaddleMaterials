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

"""CINN execution-backend implementation."""

from __future__ import annotations

from collections.abc import Callable
from collections.abc import Mapping
from collections.abc import Sequence
from typing import Any

import paddle


def compile_cinn(
    function: Callable[..., Any] | paddle.nn.Layer,
    *,
    full_graph: bool = False,
    input_spec: Sequence[Any] | None = None,
) -> Callable[..., Any] | paddle.nn.Layer:
    """Compile one numerical callable with the CINN backend.

    ``input_spec`` lists one entry per argument: a ``paddle.static.InputSpec``
    for a tensor (``None`` marks a dynamic axis) or the value of a non-tensor
    argument. Without it, every new input shape traces a new program.
    """

    return paddle.jit.to_static(
        function,
        input_spec=input_spec,
        backend="CINN",
        full_graph=full_graph,
    )


class CinnBackend:
    """Adapt CINN to the common model-owned runtime contract."""

    name = "cinn"

    def normalize_options(
        self,
        options: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        normalized = dict(options or {})
        unknown = set(normalized) - {"full_graph", "dynamic_shape"}
        if unknown:
            raise ValueError(f"Unsupported CINN runtime options: {sorted(unknown)}.")

        full_graph = normalized.get("full_graph", False)
        if not isinstance(full_graph, bool):
            raise TypeError("CINN runtime option 'full_graph' must be a bool.")
        dynamic_shape = normalized.get("dynamic_shape", True)
        if not isinstance(dynamic_shape, bool):
            raise TypeError("CINN runtime option 'dynamic_shape' must be a bool.")
        return {"full_graph": full_graph, "dynamic_shape": dynamic_shape}

    def accepts_input_spec(self, options: Mapping[str, Any]) -> bool:
        """Whether boundaries with dynamic_dims get an InputSpec.

        Only AST capture (``full_graph=True``) traces the boundary from an
        InputSpec; SOT capture tracks shapes itself. ``dynamic_shape=False``
        restores one program per input shape.
        """

        return bool(options["full_graph"] and options["dynamic_shape"])

    def validate(
        self,
        model: paddle.nn.Layer,
        *,
        use_amp: bool = False,
        world_size: int = 1,
    ) -> None:
        model_name = type(model).__name__
        if use_amp:
            raise ValueError(
                "execution_backend='cinn' does not support AMP; set use_amp=False "
                "until the CINN AMP path is validated."
            )
        if world_size > 1:
            raise ValueError(
                "execution_backend='cinn' supports world_size=1 only; distributed "
                "CINN execution is not enabled yet."
            )
        if not paddle.is_compiled_with_cuda():
            raise RuntimeError(
                f"{model_name} execution_backend='cinn' requires a CUDA-enabled "
                "Paddle installation; use execution_backend='eager' on CPU."
            )
        if not paddle.base.is_compiled_with_cinn():
            raise RuntimeError(
                f"{model_name} execution_backend='cinn' requires a Paddle build "
                "with CINN support."
            )
        device = paddle.get_device()
        if not device.startswith("gpu"):
            raise RuntimeError(
                f"{model_name} execution_backend='cinn' requires a GPU device, "
                f"but the current device is {device!r}."
            )

    def compile(
        self,
        function: Callable[..., Any] | paddle.nn.Layer,
        *,
        options: Mapping[str, Any],
        input_spec: Sequence[Any] | None = None,
    ) -> Callable[..., Any] | paddle.nn.Layer:
        if input_spec is None:
            return compile_cinn(function, full_graph=options["full_graph"])
        return compile_cinn(
            function,
            full_graph=options["full_graph"],
            input_spec=input_spec,
        )


CINN_BACKEND = CinnBackend()

__all__ = ["CINN_BACKEND", "CinnBackend", "compile_cinn"]

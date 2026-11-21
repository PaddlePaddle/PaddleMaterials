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

"""Shared execution-backend hooks used by training and prediction.

The trainer and predictor own orchestration, while a model owns the numerical
runtime and its parameters. A model that implements a compiled backend provides::

    set_execution_backend(name)
    validate_execution_backend()

Keeping the protocol here prevents public workflows from depending on a
model-specific adapter and leaves eager models completely unchanged.
"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
from typing import Any
from typing import Optional

import paddle

DEFAULT_EXECUTION_BACKEND = "eager"

_COMPILED_EXECUTION = False


@contextmanager
def compiled_execution():
    """Mark the calls inside the block as running through a compiled backend."""
    global _COMPILED_EXECUTION
    previous = _COMPILED_EXECUTION
    _COMPILED_EXECUTION = True
    try:
        yield
    finally:
        _COMPILED_EXECUTION = previous


def in_compiled_execution() -> bool:
    """Return True while a compiled backend runs or converts the current call.

    SOT capture runs with Paddle in dynamic mode, so compiled runtime calls are
    marked with ``compiled_execution()``; AST conversion is detected directly.
    """
    return _COMPILED_EXECUTION or paddle.base.dygraph.base.in_to_static_mode()


def configure_execution_backend(
    model: Any,
    backend: Optional[str],
    *,
    init_params: Mapping[str, Any] | None = None,
    use_amp: bool = False,
    world_size: int = 1,
    owner: str,
) -> str:
    """Configure and validate a workflow-level execution backend.

    ``eager`` is a valid no-op for legacy models.  Any compiled (or otherwise
    non-eager) backend must be implemented by the model so that the model,
    trainer, predictor, and checkpoint loader share one numerical contract.
    """

    if backend is None:
        backend = getattr(model, "execution_backend", DEFAULT_EXECUTION_BACKEND)
    if not isinstance(backend, str) or not backend:
        raise ValueError(
            f"{owner} execution backend must be a non-empty string, got {backend!r}."
        )

    setter = getattr(model, "set_execution_backend", None)
    if setter is None:
        # Without a setter the model cannot change runtime, so the only
        # consistent outcome is eager on both sides of the request.
        current = getattr(model, "execution_backend", DEFAULT_EXECUTION_BACKEND)
        if backend != DEFAULT_EXECUTION_BACKEND or current != DEFAULT_EXECUTION_BACKEND:
            raise ValueError(
                f"{owner} cannot select execution backend {backend!r}: "
                f"{type(model).__name__} does not implement "
                f"set_execution_backend() and reports "
                f"execution_backend={current!r}."
            )
        return DEFAULT_EXECUTION_BACKEND

    setter(backend)
    # Execution.__init_params__ configures a compiled runtime. Keep an eager
    # override usable when a shared workflow config still carries backend
    # options such as ``full_graph``; eager has no runtime to configure.
    if init_params is not None and backend != DEFAULT_EXECUTION_BACKEND:
        options_setter = getattr(model, "set_runtime_options", None)
        if options_setter is None:
            raise ValueError(
                f"{owner} cannot configure runtime options for {backend!r}: "
                f"{type(model).__name__} does not implement set_runtime_options()."
            )
        options_setter({backend: init_params})

    active = getattr(model, "execution_backend", backend)
    if active != backend:
        raise ValueError(
            f"{owner} requested execution backend {backend!r}, but the model "
            f"selected {active!r}."
        )

    if active == DEFAULT_EXECUTION_BACKEND:
        return active

    validator = getattr(model, "validate_execution_backend", None)
    if validator is None:
        raise ValueError(
            f"{owner} requested execution backend {active!r}, which requires a "
            "model with a validated execution runtime (missing "
            "validate_execution_backend())."
        )
    validator(use_amp=use_amp, world_size=world_size)
    return active


__all__ = [
    "DEFAULT_EXECUTION_BACKEND",
    "compiled_execution",
    "configure_execution_backend",
    "in_compiled_execution",
]

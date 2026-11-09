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

"""Model-owned execution-runtime dispatch and caching."""

from __future__ import annotations

from collections.abc import Callable
from collections.abc import Mapping
from functools import wraps
from typing import Any
from typing import Protocol

import paddle

from ppmat.models.common.cinn import CINN_BACKEND

DEFAULT_BACKEND = "eager"
RuntimeCallable = Callable[..., Any] | paddle.nn.Layer


class RuntimeBackend(Protocol):
    """Contract implemented by compiled execution backends."""

    name: str

    def normalize_options(
        self,
        options: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        ...

    def validate(
        self,
        model: paddle.nn.Layer,
        *,
        use_amp: bool = False,
        world_size: int = 1,
    ) -> None:
        ...

    def compile(
        self,
        function: RuntimeCallable,
        *,
        options: Mapping[str, Any],
    ) -> RuntimeCallable:
        ...


_RUNTIME_BACKENDS: dict[str, RuntimeBackend] = {}


def register_runtime_backend(backend: RuntimeBackend) -> None:
    """Register one compiled runtime backend."""

    name = backend.name
    if not isinstance(name, str) or not name:
        raise ValueError("Runtime backend name must be a non-empty string.")
    if name == DEFAULT_BACKEND:
        raise ValueError("'eager' is reserved for uncompiled execution.")
    if name in _RUNTIME_BACKENDS:
        raise ValueError(f"Runtime backend {name!r} is already registered.")
    _RUNTIME_BACKENDS[name] = backend


def get_runtime_backend(name: str) -> RuntimeBackend:
    """Return a registered compiled backend."""

    try:
        return _RUNTIME_BACKENDS[name]
    except KeyError as exc:
        available = (DEFAULT_BACKEND, *_RUNTIME_BACKENDS)
        raise ValueError(
            f"execution_backend must be one of {available}, got {name!r}."
        ) from exc


def check_backend(backend: str | None) -> str:
    """Normalize and validate an execution-backend name."""

    if backend is None:
        return DEFAULT_BACKEND
    if backend != DEFAULT_BACKEND:
        get_runtime_backend(backend)
    return backend


def runtime_boundary(name: str):
    """Mark a complete numerical method as a runtime execution boundary."""

    def decorate(function):
        # AST conversion evaluates decorators found in transformed source. Keep
        # the converted function free of a recursively installed dispatcher.
        if paddle.base.dygraph.base.in_to_static_mode():
            return function

        @wraps(function)
        def wrapped(self, *args, **kwargs):
            bound = function.__get__(self, type(self))
            return self._run_runtime(name, bound, *args, **kwargs)

        return wrapped

    return decorate


class RuntimeMixin:
    """Own backend selection, compilation and runtime caching."""

    def _init_runtime(
        self,
        execution_backend: str = DEFAULT_BACKEND,
        runtime_options: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> None:
        self.execution_backend = check_backend(execution_backend)
        object.__setattr__(self, "_runtime_cache", {})
        object.__setattr__(self, "_runtime_options", {})
        self.set_runtime_options(runtime_options or {})

    def validate_execution_backend(
        self,
        *,
        use_amp: bool = False,
        world_size: int = 1,
    ) -> None:
        if self.execution_backend == DEFAULT_BACKEND:
            return
        get_runtime_backend(self.execution_backend).validate(
            self,
            use_amp=use_amp,
            world_size=world_size,
        )

    def set_execution_backend(self, backend: str) -> None:
        backend = check_backend(backend)
        if backend != self.execution_backend:
            self.execution_backend = backend
            self.invalidate_runtime()

    def set_runtime_options(
        self,
        runtime_options: Mapping[str, Mapping[str, Any]],
    ) -> None:
        normalized = {}
        for backend_name, options in runtime_options.items():
            if backend_name == DEFAULT_BACKEND:
                if options:
                    raise ValueError(
                        "The eager backend does not accept runtime options."
                    )
                continue
            backend = get_runtime_backend(backend_name)
            normalized[backend_name] = backend.normalize_options(options)

        if normalized != self._runtime_options:
            object.__setattr__(self, "_runtime_options", normalized)
            self.invalidate_runtime()

    def get_runtime_options(self, backend: str) -> dict[str, Any]:
        backend = check_backend(backend)
        if backend == DEFAULT_BACKEND:
            return {}
        options = self._runtime_options.get(backend)
        if options is None:
            options = get_runtime_backend(backend).normalize_options(None)
        return dict(options)

    def invalidate_runtime(self) -> None:
        self._runtime_cache.clear()

    def _run_runtime(self, name: str, function, *args, **kwargs):
        """Run one numerical callable eagerly or through a cached backend."""

        backend_name = self.execution_backend
        if backend_name == DEFAULT_BACKEND:
            return function(*args, **kwargs)

        mode = "train" if self.training else "eval"
        key = (backend_name, mode, name)
        runtime = self._runtime_cache.get(key)
        if runtime is None:
            self.validate_execution_backend()
            runtime = get_runtime_backend(backend_name).compile(
                function,
                options=self.get_runtime_options(backend_name),
            )
            if hasattr(runtime, "training"):
                runtime.training = self.training
            self._runtime_cache[key] = runtime
        return runtime(*args, **kwargs)


register_runtime_backend(CINN_BACKEND)

__all__ = [
    "RuntimeBackend",
    "RuntimeMixin",
    "check_backend",
    "get_runtime_backend",
    "register_runtime_backend",
    "runtime_boundary",
]

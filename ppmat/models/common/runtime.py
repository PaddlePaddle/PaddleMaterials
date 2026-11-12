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

import inspect
from collections.abc import Callable
from collections.abc import Mapping
from collections.abc import Sequence
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


# A backend may also implement ``accepts_input_spec(options) -> bool``. When it
# returns True, boundaries that declare dynamic dimensions are compiled through
# ``compile(function, options=..., input_spec=...)`` once per static signature.

DynamicDims = Mapping[str, int | Sequence[int]]
_DYNAMIC_DIMS_ATTR = "_runtime_dynamic_dims"


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


def _normalize_dynamic_dims(
    dynamic_dims: DynamicDims | None,
) -> dict[str, tuple[int, ...]]:
    normalized = {}
    for name, axes in (dynamic_dims or {}).items():
        if not isinstance(name, str) or not name:
            raise TypeError("dynamic_dims keys must be argument names.")
        axes = (axes,) if isinstance(axes, int) else tuple(axes)
        if not axes or not all(
            isinstance(axis, int) and not isinstance(axis, bool) and axis >= 0
            for axis in axes
        ):
            raise ValueError(
                f"dynamic_dims[{name!r}] must be one or more non-negative axes, "
                f"got {axes!r}."
            )
        normalized[name] = tuple(sorted(set(axes)))
    return normalized


def runtime_boundary(name: str, *, dynamic_dims: DynamicDims | None = None):
    """Mark a complete numerical method as a runtime execution boundary.

    ``dynamic_dims`` maps argument names to the axes whose sizes change from
    call to call, for example ``{"edge_index": 1}`` for the edge axis of a
    ``[2, num_edges]`` index. A backend that captures the whole boundary as one
    static program (CINN with ``full_graph=True``) compiles those axes as
    dynamic in eval mode, so a new edge or triplet count reuses the compiled
    program. Every other axis and every non-tensor argument stays part of the
    program cache key.
    """

    dims = _normalize_dynamic_dims(dynamic_dims)

    def decorate(function):
        # AST conversion evaluates decorators found in transformed source. Keep
        # the converted function free of a recursively installed dispatcher.
        if paddle.base.dygraph.base.in_to_static_mode():
            return function

        if dims:
            parameters = inspect.signature(function).parameters
            unknown = sorted(set(dims) - set(parameters))
            if unknown:
                raise ValueError(
                    f"dynamic_dims of {function.__qualname__} names unknown "
                    f"arguments {unknown}."
                )
            setattr(function, _DYNAMIC_DIMS_ATTR, dims)

        @wraps(function)
        def wrapped(self, *args, **kwargs):
            bound = function.__get__(self, type(self))
            return self._run_runtime(name, bound, *args, **kwargs)

        return wrapped

    return decorate


def boundary_dynamic_dims(function: Callable[..., Any]) -> dict[str, tuple[int, ...]]:
    """Dynamic dimensions declared by ``runtime_boundary`` for ``function``."""

    function = getattr(function, "__func__", function)
    return dict(getattr(function, _DYNAMIC_DIMS_ATTR, {}))


def _freeze(value: Any) -> Any:
    """Hashable stand-in for a non-tensor argument in a program cache key."""

    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (tuple, list)):
        return (type(value).__name__, tuple(_freeze(item) for item in value))
    if isinstance(value, Mapping):
        return (
            "mapping",
            tuple(sorted((str(key), _freeze(item)) for key, item in value.items())),
        )
    # Unknown objects are compared by identity, like a new constant in a trace.
    return ("object", type(value).__qualname__, id(value))


class DynamicShapeRuntime:
    """Compile one static program per signature, with declared axes dynamic.

    The signature holds the static axes, dtype and ``stop_gradient`` of every
    tensor argument and the value of every other argument. A call that only
    changes a dynamic axis reuses the program; any other change compiles a new
    one, so a different atom count can never run a program traced for another.
    """

    def __init__(
        self,
        backend: RuntimeBackend,
        function: Callable[..., Any],
        options: Mapping[str, Any],
        dynamic_dims: Mapping[str, tuple[int, ...]],
    ) -> None:
        self._backend = backend
        self._function = function
        self._options = dict(options)
        self._dynamic_dims = dict(dynamic_dims)
        self._signature = inspect.signature(function)
        variadic = [
            parameter.name
            for parameter in self._signature.parameters.values()
            if parameter.kind
            in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
        ]
        if variadic:
            raise TypeError(
                "A boundary with dynamic_dims must have named arguments only, "
                f"found {variadic}."
            )
        self.programs: dict[tuple[Any, ...], RuntimeCallable] = {}

    def _arguments(self, args, kwargs) -> list[tuple[str, Any]]:
        bound = self._signature.bind(*args, **kwargs)
        bound.apply_defaults()
        return list(bound.arguments.items())

    def _key_and_spec(self, arguments):
        key, input_spec = [], []
        for name, value in arguments:
            if not isinstance(value, paddle.Tensor):
                key.append((name, _freeze(value)))
                input_spec.append(value)
                continue
            shape = list(value.shape)
            for axis in self._dynamic_dims.get(name, ()):
                if axis >= len(shape):
                    raise ValueError(
                        f"dynamic_dims[{name!r}] names axis {axis}, but the "
                        f"argument has shape {shape}."
                    )
                shape[axis] = None
            key.append((name, tuple(shape), str(value.dtype), value.stop_gradient))
            input_spec.append(
                paddle.static.InputSpec(
                    shape=shape,
                    dtype=value.dtype,
                    name=name,
                    stop_gradient=value.stop_gradient,
                )
            )
        return tuple(key), input_spec

    def __call__(self, *args, **kwargs):
        arguments = self._arguments(args, kwargs)
        key, input_spec = self._key_and_spec(arguments)
        program = self.programs.get(key)
        if program is None:
            program = self._backend.compile(
                self._function,
                options=self._options,
                input_spec=input_spec,
            )
            self.programs[key] = program
        return program(*(value for _, value in arguments))


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
            backend = get_runtime_backend(backend_name)
            options = self.get_runtime_options(backend_name)
            # Dynamic axes are an inference feature: a training batch also
            # changes the atom count, and force training relies on the traced
            # empty-input branches staying static (see docs/cinn.md).
            dynamic_dims = boundary_dynamic_dims(function) if mode == "eval" else {}
            accepts_input_spec = getattr(backend, "accepts_input_spec", None)
            if dynamic_dims and accepts_input_spec and accepts_input_spec(options):
                runtime = DynamicShapeRuntime(backend, function, options, dynamic_dims)
            else:
                runtime = backend.compile(function, options=options)
            if hasattr(runtime, "training"):
                runtime.training = self.training
            self._runtime_cache[key] = runtime
        return runtime(*args, **kwargs)


register_runtime_backend(CINN_BACKEND)

__all__ = [
    "DynamicShapeRuntime",
    "RuntimeBackend",
    "RuntimeMixin",
    "boundary_dynamic_dims",
    "check_backend",
    "get_runtime_backend",
    "register_runtime_backend",
    "runtime_boundary",
]

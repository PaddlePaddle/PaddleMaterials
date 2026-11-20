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

# Adapted from https://github.com/divelab/AIRS (OpenMat/HIENet)

"""Paddle adaptation of ``e3nn.util.jit``.

The ``compile_mode`` decorator and the ``compile`` entry point keep their
API surface; every non-eager compile mode (script/trace/unsupported) raises
NotImplementedError inside ``compile()``. The default jit_mode is 'eager'
(see e3nn/__init__.py), so the main path never enters the compile branches.
"""

import copy
import inspect
import warnings
from contextlib import contextmanager
from typing import Callable, Optional, Tuple

import paddle
from .. import get_optimization_defaults, set_optimization_defaults

ModuleFactory = Callable[..., paddle.nn.Module]
TypeTuple = Tuple[type, ...]
_E3NN_COMPILE_MODE = "__e3nn_compile_mode__"
_VALID_MODES = "trace", "script", "unsupported", None
_MAKE_TRACING_INPUTS = "_make_tracing_inputs"

_JIT_UNAVAILABLE = (
    "torch.jit is unavailable in the paddle build; e3nn codegen runs eagerly "
    "(jit_mode='eager')"
)


def compile_mode(mode: str):
    """Decorator to set the compile mode of a module.

    Parameters
    ----------
        mode : str
            'script', 'trace', or None
    """
    if mode not in _VALID_MODES:
        raise ValueError("Invalid compile mode")

    def decorator(obj):
        if not (inspect.isclass(obj) and issubclass(obj, paddle.nn.Module)):
            raise TypeError(
                "@e3nn.util.jit.compile_mode can only decorate classes derived from torch.nn.Module"
            )
        setattr(obj, _E3NN_COMPILE_MODE, mode)
        return obj

    return decorator


def get_compile_mode(mod: paddle.nn.Module) -> str:
    """Get the compilation mode of a module.

    Parameters
    ----------
        mod : torch.nn.Module

    Returns
    -------
    'script', 'trace', or None if the module was not decorated with @compile_mode
    """
    if hasattr(mod, _E3NN_COMPILE_MODE):
        mode = getattr(mod, _E3NN_COMPILE_MODE)
    else:
        mode = getattr(type(mod), _E3NN_COMPILE_MODE, None)
    assert mode in _VALID_MODES, "Invalid compile mode `%r`" % mode
    return mode


def compile(
    mod: paddle.nn.Module,
    n_trace_checks: int = 1,
    script_options: dict = None,
    trace_options: dict = None,
    in_place: bool = True,
    recurse: bool = True,
):
    """Recursively compile a module and all submodules according to their decorators.

    (Sub)modules without decorators will be unaffected.

    Parameters
    ----------
        mod : torch.nn.Module
            The module to compile. The module will have its submodules compiled replaced in-place.
        n_trace_checks : int, default = 1
            How many random example inputs to generate when tracing a module. Must be at least one in order to have tracing
            input. Extra example inputs will be pased to ``torch.jit.trace`` to confirm that the traced copmute graph doesn't
            change.
        script_options : dict, default = {}
            Extra kwargs for ``torch.jit.script``.
        trace_options : dict, default = {}
            Extra kwargs for ``torch.jit.trace``.
        in_place : bool, default True
            Whether to insert the recursively compiled submodules in-place, or do a deepcopy first.
        recurse : bool, default True
            Whether to recurse through the module's children before passing the parent to TorchScript

    Returns
    -------
    Returns the compiled module.
    """
    script_options = script_options or {}
    trace_options = trace_options or {}
    mode = get_compile_mode(mod)
    if mode == "unsupported":
        raise NotImplementedError(
            f"{type(mod).__name__} does not support TorchScript compilation"
        )
    if not in_place:
        mod = copy.deepcopy(mod)
    assert n_trace_checks >= 1
    if recurse:
        for submod_name, submod in mod.named_children():
            setattr(
                mod,
                submod_name,
                compile(
                    submod,
                    n_trace_checks=n_trace_checks,
                    script_options=script_options,
                    trace_options=trace_options,
                    in_place=True,
                    recurse=recurse,
                ),
            )
    if mode == "script":
        raise NotImplementedError(_JIT_UNAVAILABLE)
    elif mode == "trace":
        check_inputs = get_tracing_inputs(mod, n_trace_checks)
        assert len(check_inputs) >= 1, "Must have at least one tracing input."
        raise NotImplementedError(_JIT_UNAVAILABLE)
    return mod


def get_tracing_inputs(
    mod: paddle.nn.Module,
    n: int = 1,
    device: Optional[paddle.device] = None,
    dtype: Optional[paddle.dtype] = None,
):
    """Get random tracing inputs for ``mod``.

    First checks if ``mod`` has a ``_make_tracing_inputs`` method. If so, calls it with ``n`` as the single argument and
    returns its results.

    Otherwise, attempts to infer the input signature of the module using ``e3nn.util._argtools._get_io_irreps``.

    Parameters
    ----------
        mod : torch.nn.Module
        n : int, default = 1
            A hint for how many inputs are wanted. Usually n will be returned, but modules don't necessarily have to.
        device : torch.device
            The device to do tracing on. If `None` (default), will be guessed.
        dtype : torch.dtype
            The dtype to trace with. If `None` (default), will be guessed.

    Returns
    -------
    list of dict
        Tracing inputs in the format of ``torch.jit.trace_module``: dicts mapping method names like ``'forward'`` to tuples of
        arguments.
    """
    from ._argtools import (_get_device, _get_floating_dtype, _get_io_irreps,
                            _rand_args, _to_device_dtype)

    if hasattr(mod, _MAKE_TRACING_INPUTS):
        trace_inputs = mod._make_tracing_inputs(n)
        assert isinstance(trace_inputs, list)
        for d in trace_inputs:
            assert isinstance(
                d, dict
            ), "_make_tracing_inputs must return a list of dict[str, tuple]"
            assert all(
                isinstance(k, str) and isinstance(v, tuple) for k, v in d.items()
            ), "_make_tracing_inputs must return a list of dict[str, tuple]"
    else:
        irreps_in, _ = _get_io_irreps(mod, irreps_out=[None])
        trace_inputs = [{"forward": _rand_args(irreps_in)} for _ in range(n)]
    if device is None:
        device = _get_device(mod)
    if dtype is None:
        dtype = _get_floating_dtype(mod)
    trace_inputs = _to_device_dtype(trace_inputs, device, dtype)
    return trace_inputs


def trace_module(
    mod: paddle.nn.Module,
    inputs: dict = None,
    check_inputs: list = None,
    in_place: bool = True,
):
    """Trace a module.

    Identical signature to ``torch.jit.trace_module``, but first recursively compiles ``mod`` using ``compile``.

    Parameters
    ----------
        mod : torch.nn.Module
        inputs : dict
        check_inputs : list of dict
    Returns
    -------
    Traced module.
    """
    check_inputs = check_inputs or []
    old_mode = getattr(mod, _E3NN_COMPILE_MODE, None)
    if old_mode is not None and old_mode != "trace":
        warnings.warn(
            f"Trying to trace a module of type {type(mod).__name__} marked with @compile_mode != 'trace', expect errors!"
        )
    setattr(mod, _E3NN_COMPILE_MODE, "trace")
    old_make_tracing_input = None
    if inputs is not None:
        old_make_tracing_input = getattr(mod, _MAKE_TRACING_INPUTS, None)
        setattr(mod, _MAKE_TRACING_INPUTS, lambda num: [inputs] + check_inputs)
    out = compile(mod, in_place=in_place)
    if old_mode is not None:
        setattr(mod, _E3NN_COMPILE_MODE, old_mode)
    if old_make_tracing_input is not None:
        setattr(mod, _MAKE_TRACING_INPUTS, old_make_tracing_input)
    return out


def trace(
    mod: paddle.nn.Module,
    example_inputs: tuple = None,
    check_inputs: list = None,
    in_place: bool = True,
):
    """Trace a module.

    Identical signature to ``torch.jit.trace``, but first recursively compiles ``mod`` using :func:``compile``.

    Parameters
    ----------
        mod : torch.nn.Module
        example_inputs : tuple
        check_inputs : list of tuple
    Returns
    -------
    Traced module.
    """
    check_inputs = check_inputs or []
    return trace_module(
        mod=mod,
        inputs={"forward": example_inputs} if example_inputs is not None else None,
        check_inputs=[{"forward": c} for c in check_inputs],
        in_place=in_place,
    )


def script(mod: paddle.nn.Module, in_place: bool = True):
    """Script a module.

    Like ``torch.jit.script``, but first recursively compiles ``mod`` using :func:``compile``.

    Parameters
    ----------
        mod : torch.nn.Module
    Returns
    -------
    Scripted module.
    """
    old_mode = getattr(mod, _E3NN_COMPILE_MODE, None)
    if old_mode is not None and old_mode != "script":
        warnings.warn(
            f"Trying to script a module of type {type(mod).__name__} marked with @compile_mode != 'script', expect errors!"
        )
    setattr(mod, _E3NN_COMPILE_MODE, "script")
    out = compile(mod, in_place=in_place)
    if old_mode is not None:
        setattr(mod, _E3NN_COMPILE_MODE, old_mode)
    return out


@contextmanager
def disable_e3nn_codegen():
    """Context manager that disables the legacy PyTorch code generation used in e3nn."""
    init_val = get_optimization_defaults()["jit_script_fx"]
    set_optimization_defaults(jit_script_fx=False)
    yield
    set_optimization_defaults(jit_script_fx=init_val)

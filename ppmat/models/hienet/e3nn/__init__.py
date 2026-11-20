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

import paddle

# ---------------------------------------------------------------------------
# Tensor shims (import side effects), inlined here so the vendored
# subpackage is self-contained.
# ---------------------------------------------------------------------------


def _Tensor_max(self, *args, **kwargs):
    if "other" in kwargs:
        kwargs["y"] = kwargs.pop("other")
        ret = paddle.maximum(self, *args, **kwargs)
    elif len(args) == 1 and isinstance(args[0], paddle.Tensor):
        ret = paddle.maximum(self, *args, **kwargs)
    else:
        if "dim" in kwargs:
            kwargs["axis"] = kwargs.pop("dim")

        if "axis" in kwargs or len(args) >= 1:
            ret = paddle.max(self, *args, **kwargs), paddle.argmax(self, *args, **kwargs)
        else:
            ret = paddle.max(self, *args, **kwargs)

    return ret

if not hasattr(paddle.Tensor, "_max"):
    setattr(paddle.Tensor, "_max", _Tensor_max)

__version__ = "0.6.0"
from typing import Dict

import packaging.version

_TORCH_VERSION = packaging.version.parse(paddle.__version__.split("+")[0])
_DEFAULT_JIT_MODE = (
    "eager" if _TORCH_VERSION >= packaging.version.parse("2.10") else "script"
)
_OPT_DEFAULTS: Dict[str, bool] = dict(
    specialized_code=True,
    optimize_einsums=True,
    jit_script_fx=True,
    jit_mode=_DEFAULT_JIT_MODE,
)


def _handle_jit_script_fx_legacy(jit_script_fx: bool, current_jit_mode: str) -> str:
    """Handle the legacy jit_script_fx flag mapping to jit_mode.

    Parameters
    ----------
    jit_script_fx : bool
        The legacy jit_script_fx flag value
    current_jit_mode : str
        The current jit_mode value

    Returns
    -------
    str
        The new jit_mode value based on the legacy mapping rules
    """
    if not jit_script_fx and current_jit_mode == "eager":
        return "eager"
    elif not jit_script_fx:
        return "eager"
    elif jit_script_fx and current_jit_mode not in ["script", "inductor"]:
        return "script"
    return current_jit_mode


def _validate_and_set_jit_mode(jit_mode: str) -> None:
    """Validate and set the jit_mode in _OPT_DEFAULTS."""
    assert jit_mode in [
        "script",
        "inductor",
        "eager",
    ], f"Invalid jit_mode: {jit_mode}. Expected 'script', 'inductor', or 'eager'."
    _OPT_DEFAULTS["jit_mode"] = jit_mode


def set_optimization_defaults(**kwargs) -> None:
    """Globally set the default optimization settings.

    Parameters
    ----------
    **kwargs
        Keyword arguments to set the default optimization settings.
    """
    for k, v in kwargs.items():
        if k not in _OPT_DEFAULTS:
            raise ValueError(f"Unknown optimization option: {k}")
        if k == "jit_script_fx":
            new_jit_mode = _handle_jit_script_fx_legacy(v, _OPT_DEFAULTS["jit_mode"])
            _validate_and_set_jit_mode(new_jit_mode)
            _OPT_DEFAULTS[k] = v
        elif k == "jit_mode":
            _validate_and_set_jit_mode(v)
        else:
            _OPT_DEFAULTS[k] = v


def get_optimization_defaults() -> Dict[str, bool]:
    """Get the global default optimization settings."""
    return dict(_OPT_DEFAULTS)

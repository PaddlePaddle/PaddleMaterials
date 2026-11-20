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

"""Paddle eager version of ``CodeGenMixin``.

Codegen products are registered as ``paddle.nn.Layer`` submodules (eager
forward, constants precomputed at construction as buffers). For
deepcopy/pickle compatibility, ``__getstate__`` moves the codegen
submodules out of ``_sub_layers`` and stores them as pickle bytes
(buffer_type "paddle_eager"); ``__setstate__`` restores them.
"""

import pickle
from typing import Any, Dict

from ... import get_optimization_defaults
import paddle


class CodeGenMixin:
    """Mixin for classes that dynamically generate code.

    This class manages evaluating and compiling generated code for subclasses
    while remaining pickle/deepcopy compatible. If subclasses need to override
    ``__getstate__``/``__setstate__``, they should be sure to call CodeGenMixin's
    implimentation first and use its output.
    """

    def _codegen_register(self, funcs: Dict[str, Any]) -> None:
        """Register eager codegen modules as submodules.

        Parameters
        ----------
            funcs : Dict[str, paddle.nn.Layer]
                Dictionary mapping submodule names to eager codegen layers.
        """
        if not hasattr(self, "__codegen__"):
            self.__codegen__ = []
        self.__codegen__.extend(funcs.keys())
        opt_defaults = get_optimization_defaults()
        for fname, graphmod in funcs.items():
            assert isinstance(
                graphmod, paddle.nn.Layer
            ), "paddle build: codegen must register paddle.nn.Layer (eager) modules"
            if opt_defaults["jit_mode"] == "eager":
                scriptmod = graphmod
            else:
                # non-eager jit modes are rejected here
                raise NotImplementedError(
                    f"jit_mode={opt_defaults['jit_mode']!r} requires torch.jit, "
                    "unavailable in the paddle build; only jit_mode='eager' is supported"
                )
            # Add the eager module as a submodule so it can be called
            self.add_module(fname, scriptmod)

    # In order to support copy.deepcopy and pickling, we need to not save the compiled functions:
    # See pickle docs: https://docs.python.org/3/library/pickle.html#pickling-class-instances
    def __getstate__(self):
        # - Get a state to work with -
        # We need to check if other parent classes of self define __getstate__
        # paddle.nn.Layer implements __getstate__ as returning self.__dict__,
        # which is why we have these hasattr checks for other superclasses.
        if hasattr(super(CodeGenMixin, self), "__getstate__"):
            out = super(CodeGenMixin, self).__getstate__()
        else:
            out = self.__dict__

        out = out.copy()
        # We need a copy of the _sub_layers dict (paddle's submodule container)
        # Otherwise, modifying the returned state would modify the current module itself
        out["_sub_layers"] = out["_sub_layers"].copy()

        # - Add saved versions of the eager codegen modules to the state -
        codegen_state = {}
        if hasattr(self, "__codegen__"):
            for fname in self.__codegen__:
                # Get the module
                smod = getattr(self, fname)
                buffer_type: str
                buffer: bytes
                buffer_type = "paddle_eager"
                # pickle the eager layer normally
                buffer = pickle.dumps(smod)
                # Save the buffer and a note on what it is so we know how to load it
                codegen_state[fname] = (buffer_type, buffer)
                # Remove the compiled submodule from being a submodule
                # of the saved module
                del out["_sub_layers"][fname]
            out["__codegen__"] = codegen_state
        return out

    def __setstate__(self, d) -> None:
        d = d.copy()
        # We don't want to add this to the object when we call super's __setstate__
        codegen_state = d.pop("__codegen__", None)

        # We need to initialize self first so that we can add submodules
        # We need to check if other parent classes of self define __getstate__
        if hasattr(super(CodeGenMixin, self), "__setstate__"):
            super(CodeGenMixin, self).__setstate__(d)
        else:
            self.__dict__.update(d)

        if codegen_state is not None:
            for fname, (buffer_type, buffer) in codegen_state.items():
                assert isinstance(fname, str)
                assert isinstance(buffer_type, str)
                # Make sure bytes, not modules, got made
                assert isinstance(buffer, bytes)
                if buffer_type == "paddle_eager":
                    smod = pickle.loads(buffer)
                    assert isinstance(smod, paddle.nn.Layer)
                else:
                    raise NotImplementedError
                # Add the eager codegen module back as a submodule
                setattr(self, fname, smod)
            self.__codegen__ = list(codegen_state.keys())

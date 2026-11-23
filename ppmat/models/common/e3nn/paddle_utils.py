# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle


def max_class_func(self, *args, **kwargs):
    if "other" in kwargs:
        kwargs["y"] = kwargs.pop("other")
        ret = paddle.maximum(self, *args, **kwargs)
    elif len(args) == 1 and isinstance(args[0], paddle.Tensor):
        ret = paddle.maximum(self, *args, **kwargs)
    else:
        if "dim" in kwargs:
            kwargs["axis"] = kwargs.pop("dim")

        if "axis" in kwargs or len(args) >= 1:
            ret = paddle.max(self, *args, **kwargs), paddle.argmax(
                self, *args, **kwargs
            )
        else:
            ret = paddle.max(self, *args, **kwargs)

    return ret


setattr(paddle.Tensor, "max_func", max_class_func)


def reshape(self, *args, **kwargs):
    if args:
        if len(args) == 1 and isinstance(args[0], (tuple, list)):
            return paddle.reshape(self, args[0])
        else:
            return paddle.reshape(self, list(args))
    elif kwargs:
        assert "shape" in kwargs
        return paddle.reshape(self, shape=kwargs["shape"])


setattr(paddle.Tensor, "reshape", reshape)


def mul(self, *args, **kwargs):
    if "other" in kwargs:
        y = kwargs["other"]
    elif "y" in kwargs:
        y = kwargs["y"]
    else:
        y = args[0]

    if not isinstance(y, paddle.Tensor):
        y = paddle.to_tensor(y)

    return paddle.multiply(self, y)


setattr(paddle.Tensor, "mul", mul)
setattr(paddle.Tensor, "multiply", mul)


def add(self, *args, **kwargs):
    if "other" in kwargs:
        y = kwargs["other"]
    elif "y" in kwargs:
        y = kwargs["y"]
    else:
        y = args[0]
    if "alpha" in kwargs:
        alpha = kwargs["alpha"]
        if alpha != 1:
            if not isinstance(y, paddle.Tensor):
                y = paddle.to_tensor(alpha * y)
            else:
                y = alpha * y
    else:
        if not isinstance(y, paddle.Tensor):
            y = paddle.to_tensor(y)
    return paddle.add(self, y)


setattr(paddle.Tensor, "add", add)


def sub(self, *args, **kwargs):
    if "other" in kwargs:
        y = kwargs["other"]
    elif "y" in kwargs:
        y = kwargs["y"]
    else:
        y = args[0]
    if "alpha" in kwargs:
        alpha = kwargs["alpha"]
        if alpha != 1:
            if not isinstance(y, paddle.Tensor):
                y = paddle.to_tensor(alpha * y)
            else:
                y = alpha * y
    else:
        if not isinstance(y, paddle.Tensor):
            y = paddle.to_tensor(y)
    return paddle.subtract(self, y)


setattr(paddle.Tensor, "sub", sub)
setattr(paddle.Tensor, "subtract", sub)


def dim2perm(ndim, dim0, dim1):
    perm = list(range(ndim))
    perm[dim0], perm[dim1] = perm[dim1], perm[dim0]
    return perm


def min_class_func(self, *args, **kwargs):
    if "other" in kwargs:
        kwargs["y"] = kwargs.pop("other")
        ret = paddle.minimum(self, *args, **kwargs)
    elif len(args) == 1 and isinstance(args[0], paddle.Tensor):
        ret = paddle.minimum(self, *args, **kwargs)
    else:
        if "dim" in kwargs:
            kwargs["axis"] = kwargs.pop("dim")

        if "axis" in kwargs or len(args) >= 1:
            ret = paddle.min(self, *args, **kwargs), paddle.argmin(
                self, *args, **kwargs
            )
        else:
            ret = paddle.min(self, *args, **kwargs)

    return ret


setattr(paddle.Tensor, "min_func", min_class_func)


def div(self, *args, **kwargs):
    if "other" in kwargs:
        y = kwargs["other"]
    elif "y" in kwargs:
        y = kwargs["y"]
    else:
        y = args[0]

    if not isinstance(y, paddle.Tensor):
        y = paddle.to_tensor(y)

    res = paddle.divide(self, y)

    if "rounding_mode" in kwargs:
        rounding_mode = kwargs["rounding_mode"]
        if rounding_mode == "trunc":
            res = paddle.trunc(res)
        elif rounding_mode == "floor":
            res = paddle.floor(res)

    return res


setattr(paddle.Tensor, "div", div)
setattr(paddle.Tensor, "divide", div)


def view(self, *args, **kwargs):
    if args:
        if len(args) == 1 and isinstance(args[0], str):
            return paddle.view(self, args[0])
        if len(args) == 1 and isinstance(args[0], (tuple, list)):
            return paddle.reshape(self, args[0])
        else:
            return paddle.reshape(self, list(args))
    elif kwargs:
        shape_or_dtype = list(kwargs.values())[0]
        if isinstance(shape_or_dtype, str):
            return paddle.view(self, shape_or_dtype)
        return paddle.reshape(self, shape=shape_or_dtype)


setattr(paddle.Tensor, "view", view)

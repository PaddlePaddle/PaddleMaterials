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

from __future__ import annotations

import collections
import functools
import random
import time
from contextlib import ContextDecorator
from typing import Callable
from typing import Dict
from typing import List
from typing import Literal
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union

import numpy as np
import paddle
from paddle import distributed as dist

from ppmat.utils import logger
from ppmat.utils.paddle_aux import dim2perm
from ppmat.utils.scatter import scatter

__all__ = [
    "AverageMeter",
    "PrettyOrderedDict",
    "Prettydefaultdict",
    "RankZeroOnly",
    "Timer",
    "format_time_manual",
    "all_gather",
    "concat_dict_list",
    "convert_to_array",
    "convert_to_dict",
    "stack_dict_list",
    "cartesian_product",
    "combine_array_with_time",
    "set_random_seed",
    "run_on_eval_mode",
    "run_at_rank0",
]


class AverageMeter:
    """
    Computes and stores the average and current value
    Code was based on https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """

    def __init__(self, name="", fmt="f", postfix="", need_avg=True, smooth_window=1):
        self.name = name
        self.fmt = fmt
        self.postfix = postfix
        self.need_avg = need_avg
        self.smooth_window = smooth_window
        self.reset()

    def reset(self):
        """Reset."""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.history = []

    def update(self, val, n=1):
        """Update."""
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.history.append(val)

    @property
    def smooth_avg(self):
        if len(self.history) >= self.smooth_window:
            avg = sum(self.history[-self.smooth_window :]) / self.smooth_window
        else:
            avg = sum(self.history) / len(self.history)
        return avg

    @property
    def avg_info(self):
        if isinstance(self.avg, paddle.Tensor):
            self.avg = float(self.avg)
        return f"{self.name}: {self.avg:.5f}"

    @property
    def total(self):
        return f"{self.name}_sum: {self.sum:{self.fmt}}{self.postfix}"

    @property
    def total_minute(self):
        return f"{self.name} {self.sum / 60:{self.fmt}}{self.postfix} min"

    @property
    def mean(self):
        return (
            f"{self.name}: {self.avg:{self.fmt}}{self.postfix}" if self.need_avg else ""
        )

    @property
    def value(self):
        return f"{self.name}: {self.val:{self.fmt}}{self.postfix}"


class PrettyOrderedDict(collections.OrderedDict):
    """
    The ordered dict which can be prettily printed.

    Examples:
        >>> import ppsci
        >>> dic = ppsci.utils.misc.PrettyOrderedDict()
        >>> dic.update({'a':1, 'b':2, 'c':3})
        >>> print(dic)
        ('a', 1)('b', 2)('c', 3)
    """

    def __str__(self):
        return "".join([str((k, v)) for k, v in self.items()])


class Prettydefaultdict(collections.defaultdict):
    """
    The default dict which can be prettily printed.

    Examples:
        >>> import ppsci
        >>> dic = ppsci.utils.misc.Prettydefaultdict()
        >>> dic.update({'a':1, 'b':2, 'c':3})
        >>> print(dic)
        ('a', 1)('b', 2)('c', 3)
    """

    def __str__(self):
        return "".join([str((k, v)) for k, v in self.items()])


class RankZeroOnly:
    """
    A context manager that ensures the code inside it is only executed by the process
    with rank zero. All rank will be synchronized by `dist.barrier` in
    distributed environment.

    NOTE: Always used for time consuming code blocks, such as initialization of log
    writer, saving result to disk, etc.

    Args:
        rank (Optional[int]): The rank of the current process. If not provided,
            it will be obtained from `dist.get_rank()`.

    Examples:
        >>> import paddle.distributed as dist
        >>> with RankZeroOnly(dist.get_rank()) as is_master:
        ...     if is_master:
        ...         # code here which should only be executed in the master process
        ...         pass
    """

    def __init__(self, rank: Optional[int] = None):
        """
        Enter the context and check if the current process is the master.

        Args:
            rank (Optional[int]): The rank of the current process. If not provided,
                it will be obtained from `dist.get_rank()`.
        """
        super().__init__()
        self.rank = rank if (rank is not None) else dist.get_rank()
        self.is_master = self.rank == 0

    def __enter__(self) -> bool:
        """
        Enter the context and check if the current process is the master.

        Returns:
            bool: True if the current process is the master (rank zero),
                False otherwise.
        """
        return self.is_master

    def __exit__(self, exc_type, exc_value, traceback):
        if dist.get_world_size() > 1:
            dist.barrier()


class Timer(ContextDecorator):
    """Count time cost for code block within context.

    Args:
        name (str, optional): Name of timer discriminate different code block.
            Defaults to "Timer".
        auto_print (bool, optional): Whether print time cost when exit context.
            Defaults to True.
    """

    interval: float  # Time cost for code within Timer context

    def __init__(self, name: str = "Timer", auto_print: bool = True):
        super().__init__()
        self.name = name
        self.auto_print = auto_print

    def __enter__(self):
        paddle.device.synchronize()
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, type, value, traceback):
        paddle.device.synchronize()
        self.end_time = time.perf_counter()
        self.interval = self.end_time - self.start_time
        if self.auto_print:
            logger.message(f"{self.name}.time_cost = {self.interval:.2f} s")

    def start(self, name: str = "Timer"):
        """Push a new timer context.

        Args:
            name (str, optional): Name of code block to be clocked. Defaults to "Timer".
        """
        paddle.device.synchronize()
        self.start_time = time.perf_counter()

    def end(self):
        """End current timer context and print time cost."""
        paddle.device.synchronize()
        self.end_time = time.perf_counter()
        self.interval = self.end_time - self.start_time
        if self.auto_print:
            logger.message(f"{self.name}.time_cost = {self.interval:.2f} s")


def format_time_manual(seconds):
    seconds = int(seconds)
    hours = seconds // 3600
    remainder = seconds % 3600
    minutes = remainder // 60
    seconds = remainder % 60
    return f"{hours:02d}h-{minutes:02d}m-{seconds:02d}s"


def convert_to_dict(array: np.ndarray, keys: Tuple[str, ...]) -> Dict[str, np.ndarray]:
    """Split given array into single channel array at axis -1 in order of given keys.

    Args:
        array (np.ndarray): Array to be split.
        keys (Tuple[str, ...]): Keys used in split.

    Returns:
        Dict[str, np.ndarray]: Split dict.

    Examples:
        >>> import numpy as np
        >>> import ppsci
        >>> arr = np.array([[1., 2., 3.], [4., 5., 6.]])
        >>> result = ppsci.utils.misc.convert_to_dict(arr, ("x", "y", "z"))
        >>> print(arr.shape)
        (2, 3)
        >>> for k, v in result.items():
        ...    print(k, v.shape)
        x (2, 1)
        y (2, 1)
        z (2, 1)
    """
    if array.shape[-1] != len(keys):
        raise ValueError(
            f"dim of array({array.shape[-1]}) must equal to " f"len(keys)({len(keys)})"
        )

    split_array = np.split(array, len(keys), axis=-1)
    return {key: split_array[i] for i, key in enumerate(keys)}


def all_gather(
    tensor: paddle.Tensor, concat: bool = True, axis: int = 0
) -> Union[paddle.Tensor, List[paddle.Tensor]]:
    """Gather tensor from all devices, concatenate them along given axis if specified.

    Args:
        tensor (paddle.Tensor): Tensor to be gathered from all GPUs.
        concat (bool, optional): Whether to concatenate gathered Tensors.
            Defaults to True.
        axis (int, optional): Axis which concatenated along. Defaults to 0.

    Returns:
        Union[paddle.Tensor, List[paddle.Tensor]]: Gathered Tensors.

    Examples:
        >>> import paddle
        >>> import ppsci
        >>> import paddle.distributed as dist
        >>> dist.init_parallel_env()      # doctest: +SKIP
        >>> if dist.get_rank() == 0:      # doctest: +SKIP
        ...     data = paddle.to_tensor([[1, 2, 3], [4, 5, 6]])
        ... else:
        ...     data = paddle.to_tensor([[7, 8, 9], [10, 11, 12]])
        >>> result = ppsci.utils.misc.all_gather(data)    # doctest: +SKIP
        >>> print(result.numpy())     # doctest: +SKIP
        [[ 1  2  3]
         [ 4  5  6]
         [ 7  8  9]
         [10 11 12]]
    """
    result: List[paddle.Tensor] = []

    # NOTE: Put tensor to CUDAPlace from CUDAPinnedPlace to use communication.
    if tensor.place.is_cuda_pinned_place():
        tensor = tensor.cuda()

    # TODO(HydrogenSulfate): As non-contiguous(strided) tensor is not supported in
    # dist.all_gather, manually convert given Tensor to contiguous below. Strided tensor
    # will be supported in future.
    dist.all_gather(result, tensor.contiguous())

    if concat:
        return paddle.concat(result, axis)
    return result


def convert_to_array(dict_: Dict[str, np.ndarray], keys: Tuple[str, ...]) -> np.ndarray:
    """Concatenate arrays in axis -1 in order of given keys.

    Args:
        dict_ (Dict[str, np.ndarray]): Dict contains arrays.
        keys (Tuple[str, ...]): Concatenate keys used in concatenation.

    Returns:
        np.ndarray: Concatenated array.

    Examples:
        >>> import numpy as np
        >>> import ppsci
        >>> dic = {"x": np.array([[1., 2.], [3., 4.]]),
        ...        "y": np.array([[5., 6.], [7., 8.]]),
        ...        "z": np.array([[9., 10.], [11., 12.]])}
        >>> result = ppsci.utils.misc.convert_to_array(dic, ("x", "z"))
        >>> print(result)
        [[ 1.  2.  9. 10.]
         [ 3.  4. 11. 12.]]
    """
    return np.concatenate([dict_[key] for key in keys], axis=-1)


def concat_dict_list(
    dict_list: Sequence[Dict[str, np.ndarray]]
) -> Dict[str, np.ndarray]:
    """Concatenate arrays in tuple of dicts at axis 0.

    Args:
        dict_list (Sequence[Dict[str, np.ndarray]]): Sequence of dicts.

    Returns:
        Dict[str, np.ndarray]: A dict with concatenated arrays for each key.

    """
    ret = {}
    for key in dict_list[0].keys():
        ret[key] = np.concatenate([_dict[key] for _dict in dict_list], axis=0)
    return ret


def stack_dict_list(
    dict_list: Sequence[Dict[str, np.ndarray]]
) -> Dict[str, np.ndarray]:
    """Stack arrays in tuple of dicts at axis 0.

    Args:
        dict_list (Sequence[Dict[str, np.ndarray]]): Sequence of dicts.

    Returns:
        Dict[str, np.ndarray]: A dict with stacked arrays for each key.
    """
    ret = {}
    for key in dict_list[0].keys():
        ret[key] = np.stack([_dict[key] for _dict in dict_list], axis=0)
    return ret


def typename(obj: object) -> str:
    """Return type name of given object.

    Args:
        obj (object): Python object which is instantiated from a class.

    Returns:
        str: Class name of given object.
    """
    return obj.__class__.__name__


def combine_array_with_time(x: np.ndarray, t: Tuple[int, ...]) -> np.ndarray:
    """Combine given data x with time sequence t.
    Given x with shape (N, D) and t with shape (T, ),
    this function will repeat t_i for N times and will concat it with data x for each
    t_i in t, finally return the stacked result, which is of shape (N×T, D+1).

    Args:
        x (np.ndarray): Points data with shape (N, D).
        t (Tuple[int, ...]): Time sequence with shape (T, ).

    Returns:
        np.ndarray: Combined data with shape of (N×T, D+1).

    Examples:
        >>> import numpy as np
        >>> import ppsci
        >>> data_point = np.arange(10).reshape((2, 5))
        >>> time = (1, 2, 3)
        >>> result = ppsci.utils.misc.combine_array_with_time(data_point, time)
        >>> print(result)
        [[1. 0. 1. 2. 3. 4.]
         [1. 5. 6. 7. 8. 9.]
         [2. 0. 1. 2. 3. 4.]
         [2. 5. 6. 7. 8. 9.]
         [3. 0. 1. 2. 3. 4.]
         [3. 5. 6. 7. 8. 9.]]
    """
    nx = len(x)
    tx = []
    for ti in t:
        tx.append(
            np.hstack(
                (np.full([nx, 1], float(ti), dtype=paddle.get_default_dtype()), x)
            )
        )
    tx = np.vstack(tx)
    return tx


def cartesian_product(*arrays: np.ndarray) -> np.ndarray:
    """Cartesian product for input sequence of array(s).

    Reference: https://stackoverflow.com/questions/11144513/cartesian-product-of-x-and-y-array-points-into-single-array-of-2d-points

    Assume shapes of input arrays are: $(N_1,), (N_2,), (N_3,), ..., (N_M,)$,
    then the cartesian product result will be shape of $(N_1xN_2xN_3x...xN_M, M)$.

    Args:
        arrays (np.ndarray): Input arrays.

    Returns:
        np.ndarray: Cartesian product result of shape $(N_1xN_2xN_3x...xN_M, M)$.

    Examples:
        >>> t = np.array([1, 2])
        >>> x = np.array([10, 20])
        >>> y = np.array([100, 200])
        >>> txy = cartesian_product(t, x, y)
        >>> print(txy)
        [[  1  10 100]
         [  1  10 200]
         [  1  20 100]
         [  1  20 200]
         [  2  10 100]
         [  2  10 200]
         [  2  20 100]
         [  2  20 200]]
    """
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[..., i] = a
    return arr.reshape(-1, la)


def set_random_seed(seed: int):
    """Set numpy, random, paddle random_seed to given seed.

    Args:
        seed (int): Random seed.
    """
    paddle.seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def run_on_eval_mode(func: Callable) -> Callable:
    """A decorator automatically running given class method in eval mode and keep
    training state unchanged after function finished.

    Args:
        func (Callable): Class method which is expected running in eval mode.

    Returns:
        Callable: Decorated class method.
    """

    @functools.wraps(func)
    def function_with_eval_state(self, *args, **kwargs):
        # log original state
        train_state = self.model.training

        # switch to eval mode
        if train_state:
            self.model.eval()

        # run func in eval mode
        result = func(self, *args, **kwargs)

        # restore state
        if train_state:
            self.model.train()

        return result

    return function_with_eval_state


def run_at_rank0(func: Callable) -> Callable:
    """A decorator that allow given function run only at rank 0 to avoid
    multiple logs or other events. Usually effected in distributed environment.

    Args:
        func (Callable): Given function.

    Returns:
        Callable: Wrapped function which will only run at at rank 0,
            skipped at other rank.

    Examples:
        >>> import paddle
        >>> from ppsci.utils import misc
        >>> @misc.run_at_rank0
        ... def func():
        ...     print(f"now_rank is {paddle.distributed.get_rank()}")
        >>> func()
        now_rank is 0
    """

    @functools.wraps(func)
    def wrapped_func(*args, **kwargs):
        if dist.get_rank() == 0:
            return func(*args, **kwargs)

    return wrapped_func


def ragged_range(sizes: paddle.Tensor) -> paddle.Tensor:
    """Multiple concatenated ranges.

    Examples
    --------
        sizes = [1 4 2 3]
        Return: [0  0 1 2 3  0 1  0 1 2]
    """
    assert sizes.dim() == 1
    if sizes.sum() == 0:
        return paddle.empty(shape=[0], dtype=sizes.dtype)
    sizes_nonzero = sizes > 0
    if not paddle.all(x=sizes_nonzero):
        sizes = paddle.masked_select(x=sizes, mask=sizes_nonzero)
    id_steps = paddle.ones(shape=sizes.sum(), dtype="int64")
    id_steps[0] = 0
    insert_index = sizes[:-1].cumsum(axis=0)
    insert_val = (1 - sizes)[:-1]
    id_steps[insert_index] = insert_val
    res = id_steps.cumsum(axis=0)
    return res


def repeat_blocks(
    sizes: paddle.Tensor,
    repeats: paddle.Tensor,
    continuous_indexing: bool = True,
    start_idx: int = 0,
    block_inc: int = 0,
    repeat_inc: int = 0,
) -> paddle.Tensor:
    """Repeat blocks of indices.
    Adapted from https://stackoverflow.com/questions/51154989/numpy-vectorized-function-to-repeat-blocks-of-consecutive-elements

    continuous_indexing: Whether to keep increasing the index after each block
    start_idx: Starting index
    block_inc: Number to increment by after each block,
               either global or per block. Shape: len(sizes) - 1
    repeat_inc: Number to increment by after each repetition,
                either global or per block

    Examples
    --------
        sizes = [1,3,2] ; repeats = [3,2,3] ; continuous_indexing = False
        Return: [0 0 0  0 1 2 0 1 2  0 1 0 1 0 1]
        sizes = [1,3,2] ; repeats = [3,2,3] ; continuous_indexing = True
        Return: [0 0 0  1 2 3 1 2 3  4 5 4 5 4 5]
        sizes = [1,3,2] ; repeats = [3,2,3] ; continuous_indexing = True ;
        repeat_inc = 4
        Return: [0 4 8  1 2 3 5 6 7  4 5 8 9 12 13]
        sizes = [1,3,2] ; repeats = [3,2,3] ; continuous_indexing = True ;
        start_idx = 5
        Return: [5 5 5  6 7 8 6 7 8  9 10 9 10 9 10]
        sizes = [1,3,2] ; repeats = [3,2,3] ; continuous_indexing = True ;
        block_inc = 1
        Return: [0 0 0  2 3 4 2 3 4  6 7 6 7 6 7]
        sizes = [0,3,2] ; repeats = [3,2,3] ; continuous_indexing = True
        Return: [0 1 2 0 1 2  3 4 3 4 3 4]
        sizes = [2,3,2] ; repeats = [2,0,2] ; continuous_indexing = True
        Return: [0 1 0 1  5 6 5 6]
    """
    assert sizes.dim() == 1
    assert all(sizes >= 0)
    sizes_nonzero = sizes > 0
    if not paddle.all(x=sizes_nonzero):
        assert block_inc == 0
        sizes = paddle.masked_select(x=sizes, mask=sizes_nonzero)
        if isinstance(repeats, paddle.Tensor):
            repeats = paddle.masked_select(x=repeats, mask=sizes_nonzero)
        if isinstance(repeat_inc, paddle.Tensor):
            repeat_inc = paddle.masked_select(x=repeat_inc, mask=sizes_nonzero)
    if isinstance(repeats, paddle.Tensor):
        assert all(repeats >= 0)
        insert_dummy = repeats[0] == 0
        if insert_dummy:
            one = paddle.ones(shape=[1], dtype=sizes.dtype)
            zero = paddle.zeros(shape=[1], dtype=sizes.dtype)
            sizes = paddle.concat(x=(one, sizes))
            repeats = paddle.concat(x=(one, repeats))
            if isinstance(block_inc, paddle.Tensor):
                block_inc = paddle.concat(x=(zero, block_inc))
            if isinstance(repeat_inc, paddle.Tensor):
                repeat_inc = paddle.concat(x=(zero, repeat_inc))
    else:
        assert repeats >= 0
        insert_dummy = False
    r1 = paddle.repeat_interleave(x=paddle.arange(end=len(sizes)), repeats=repeats)
    N = (sizes * repeats).sum()
    id_ar = paddle.ones(shape=N, dtype="int64")
    id_ar[0] = 0
    insert_index = sizes[r1[:-1]].cumsum(axis=0)
    insert_val = (1 - sizes)[r1[:-1]]
    if isinstance(repeats, paddle.Tensor) and paddle.any(x=repeats == 0):
        diffs = r1[1:] - r1[:-1]
        indptr = paddle.concat(
            x=(paddle.zeros(shape=[1], dtype=sizes.dtype), diffs.cumsum(axis=0))
        )
        if continuous_indexing:
            # insert_val += segment_csr(sizes[: r1[-1]], indptr, reduce="sum")
            raise NotImplementedError()
        if isinstance(block_inc, paddle.Tensor):
            # insert_val += segment_csr(block_inc[: r1[-1]], indptr, reduce="sum")
            raise NotImplementedError()
        else:
            insert_val += block_inc * (indptr[1:] - indptr[:-1])
            if insert_dummy:
                insert_val[0] -= block_inc
    else:
        idx = r1[1:] != r1[:-1]
        if continuous_indexing:
            insert_val[idx] = 1
        idx = paddle.where(condition=idx)[0].flatten()
        insert_val[idx] += block_inc
    if isinstance(repeat_inc, paddle.Tensor):
        insert_val += repeat_inc[r1[:-1]]
        if isinstance(repeats, paddle.Tensor):
            repeat_inc_inner = repeat_inc[repeats > 0][:-1]
        else:
            repeat_inc_inner = repeat_inc[:-1]
    else:
        insert_val += repeat_inc
        repeat_inc_inner = repeat_inc
    if isinstance(repeats, paddle.Tensor):
        repeats_inner = repeats[repeats > 0][:-1]
    else:
        repeats_inner = repeats
    idx = r1[1:] != r1[:-1]
    idx = paddle.where(condition=idx)[0].flatten()
    insert_val[idx] -= repeat_inc_inner * repeats_inner
    id_ar[insert_index] = insert_val
    if insert_dummy:
        id_ar = id_ar[1:]
        if continuous_indexing:
            id_ar[0] -= 1
    id_ar[0] += start_idx
    res = id_ar.cumsum(axis=0)
    return res


def aggregate_per_sample(
    data_per_row: paddle.Tensor,
    batch_idx: (paddle.Tensor | None),
    reduce: Literal["sum", "mean"],
    batch_size: int,
):
    """
    Aggregate (potentially) batched input tensor to get a scalar for each sample in the
    batch.
    E.g., (num_atoms, d1, d2, ..., dn) -> (batch_size, d1, d2, ..., dn) -> (batch_size,)
    where the first aggregation only happens when batch_idx is provided.

    Args:
        data_per_row: shape (num_nodes, any_more_dims). May contain multiple nodes per
            sample.
        batch_idx: shape (num_nodes,). Indicates which sample each row belongs to. If
            not provided, then we assume the first dimension is the batch dimension.
        reduce: determines how to aggregate over nodes within each sample. (Aggregation
            over samples and within dims for one node is always mean.)
        batch_size: number of samples in the batch.

    Returns:
        Scalar for each sample, shape (batch_size,).

    """
    data_per_row = paddle.mean(
        x=data_per_row.reshape([tuple(data_per_row.shape)[0], -1]), axis=1
    )
    if batch_idx is None:
        data_per_sample = data_per_row
    else:
        data_per_sample = scatter(
            src=data_per_row, index=batch_idx, dim_size=batch_size, reduce=reduce
        )
    return data_per_sample


def expand(a, x_shape, left=False):
    a_dim = len(tuple(a.shape))
    if left:
        return a.reshape(*((1,) * (len(x_shape) - a_dim) + tuple(a.shape)))
    else:
        return a.reshape([*(tuple(a.shape) + (1,) * (len(x_shape) - a_dim))])


def _broadcast_like(x, like):
    """
    add broadcast dimensions to x so that it can be broadcast over ``like``
    """
    if like is None:
        return x
    return x[(...,) + (None,) * (like.ndim - x.ndim)]


def maybe_expand(
    x: paddle.Tensor, batch: Optional[paddle.Tensor], like: paddle.Tensor = None
) -> paddle.Tensor:
    """

    Args:
        x: shape (batch_size, ...)
        batch: shape (num_thingies,) with integer entries in the range [0, batch_size),
            indicating which sample each thingy belongs to
        like: shape x.shape + potential additional dimensions
    Returns:
        expanded x with shape (num_thingies,), or if given like.shape, containing value
             of x for each thingy.
        If `batch` is None, just returns `x` unmodified, to avoid pointless work if you
        have exactly one thingy per sample.
    """
    x = _broadcast_like(x, like)
    if batch is None:
        return x
    else:
        return x[batch]


def make_noise_symmetric_preserve_variance(noise: paddle.Tensor) -> paddle.Tensor:
    """Makes the noise matrix symmetric, preserving the variance. Assumes i.i.d. noise
    for each dimension.

    Args:
        noise (paddle.Tensor): Input noise matrix, must be a batched square matrix,
            i.e., have shape (batch_size, dim, dim).

    Returns:
        paddle.Tensor: The symmetric noise matrix, with the same variance as the input.
    """
    assert (
        len(tuple(noise.shape)) == 3 and tuple(noise.shape)[1] == tuple(noise.shape)[2]
    ), "Symmetric noise only works for square-matrix-shaped data."
    return (
        1
        / 2**0.5
        * (1 - paddle.eye(num_rows=3)[None])
        * (noise + noise.transpose(perm=dim2perm(noise.ndim, 1, 2)))
        + paddle.eye(num_rows=3)[None] * noise
    )


def is_equal(dict1, dict2):
    """Recursively compares two potentially nested structures for deep equality

    This function performs a thorough comparison of dictionaries, lists, tuples,
    and other basic data types, handling nested structures at any depth. It supports:

    - Dictionary comparison (order-insensitive for keys)
    - List/tuple comparison (order-sensitive)
    - Type-sensitive comparisons (e.g., int vs float, list vs tuple)
    - Mixed structure comparisons (dicts containing lists containing dicts, etc.)

    Args:
        dict1 (dict/list/tuple/any): First structure to compare
        dict2 (dict/list/tuple/any): Second structure to compare

    Returns:
        bool: True if structures are deeply equal, False otherwise

    Notes:
        - For dictionaries: Key order doesn't matter, but key-value pairs must match
        - For sequences: Element order matters (lists/tuples are order-sensitive)
        - Basic types (int, str, etc.) are compared using normal equality
        - Different container types are considered unequal (e.g., list vs tuple)
        - Recursive structures (circular references) will cause infinite recursion
    """

    # type check
    if type(dict1) != type(dict2):
        return False

    # compare dicts
    if isinstance(dict1, dict):
        if len(dict1) != len(dict2):
            return False
        for key in dict1:
            if key not in dict2:
                return False
            if not is_equal(dict1[key], dict2[key]):
                return False
        return True

    # compare lists/tuples
    elif isinstance(dict1, (list, tuple)):
        if len(dict1) != len(dict2):
            return False
        for i in range(len(dict1)):
            if not is_equal(dict1[i], dict2[i]):
                return False
        return True

    # compare basic types
    else:
        return dict1 == dict2

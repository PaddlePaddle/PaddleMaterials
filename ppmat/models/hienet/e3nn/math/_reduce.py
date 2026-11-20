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

import itertools

import paddle
from . import perm


def germinate_formulas(formula):
    formulas = [
        (-1 if f.startswith("-") else 1, f.replace("-", "")) for f in formula.split("=")
    ]
    s0, f0 = formulas[0]
    assert s0 == 1
    for _s, f in formulas:
        if len(set(f)) != len(f) or set(f) != set(f0):
            raise RuntimeError(f"{f} is not a permutation of {f0}")
        if len(f0) != len(f):
            raise RuntimeError(f"{f0} and {f} don't have the same number of indices")
    formulas = {(s, tuple(f.index(i) for i in f0)) for s, f in formulas}
    while True:
        n = len(formulas)
        formulas = formulas.union([(s, perm.inverse(p)) for s, p in formulas])
        formulas = formulas.union(
            [
                (s1 * s2, perm.compose(p1, p2))
                for s1, p1 in formulas
                for s2, p2 in formulas
            ]
        )
        if len(formulas) == n:
            break
    return f0, formulas


def reduce_permutation(f0, formulas, dtype=None, device=None, **dims):
    """
    Parameters
    ----------
    f0 : str

    formulas : list of tuple (int, str)

    dims : dict of str -> int

    Examples
    --------
    >>> Q, ret = reduce_permutation(*germinate_formulas("ij=-ji"), i=2)
    >>> Q.shape, len(ret)
    (torch.Size([1, 2, 2]), 1)
    """
    for _s, p in formulas:
        f = "".join(f0[i] for i in p)
        for i, j in zip(f0, f):
            if i in dims and j in dims and dims[i] != dims[j]:
                raise RuntimeError(f"dimension of {i} and {j} should be the same")
            if i in dims:
                dims[j] = dims[i]
            if j in dims:
                dims[i] = dims[j]
    for i in f0:
        if i not in dims:
            raise RuntimeError(f"index {i} has no dimension associated to it")
    dims = [dims[i] for i in f0]
    full_base = list(itertools.product(*(range(d) for d in dims)))
    base = set()
    for x in full_base:
        xs = {(s, tuple(x[i] for i in p)) for s, p in formulas}
        if (-1, x) not in xs:
            base.add(frozenset({frozenset(xs), frozenset({(-s, x) for s, x in xs})}))
    base = sorted([sorted([sorted(xs) for xs in x]) for x in base])
    d_sym = len(base)
    Q = paddle.zeros(d_sym, len(full_base), dtype=dtype, device=device)
    ret = []
    for i, x in enumerate(base):
        x = max(x, key=lambda xs: sum(s for s, x in xs))
        ret.append(x)
        for s, e in x:
            j = 0
            for k, d in zip(e, dims):
                j *= d
                j += k
            Q[i, j] = s / len(x) ** 0.5
    Q = Q.reshape(d_sym, *dims)
    return Q, ret

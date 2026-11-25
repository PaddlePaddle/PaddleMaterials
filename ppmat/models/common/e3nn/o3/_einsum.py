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

"""An ``einsum`` built from ``transpose``/``reshape``/``bmm``.

Paddle's ``einsum`` backward does not register a gradient for ``einsum_grad``,
so expressions that are differentiated twice (energy -> forces -> loss ->
parameters, see ``ppmat/models/mace/utils.py``) fail with::

    The Op einsum_grad doesn't have any grad op.

``matmul``/``bmm`` do support the second-order autograd needed by force and
stress losses, therefore the e3nn tensor product and linear layers use this
helper instead of ``paddle.einsum``. The contraction order is chosen at
runtime from the equation, which is sufficient for the fixed set of e3nn
equations; unsupported forms fall back to ``paddle.einsum``.
"""

from math import prod

import paddle


def _prod_shape(labels, shapes):
    size = 1
    for label in labels:
        size *= shapes[label]
    return size


def _merge(a, labels_a, b, labels_b, out, later):
    """Contract the labels of ``a`` and ``b`` that disappear from ``out``."""
    contract = [
        label
        for label in labels_a
        if label in labels_b and label not in out and label not in later
    ]
    batch = [
        label for label in labels_a if label in labels_b and label not in contract
    ]
    a_keep = [label for label in labels_a if label not in batch and label not in contract]
    b_keep = [label for label in labels_b if label not in batch and label not in contract]

    shapes = dict(zip(labels_a + labels_b, tuple(a.shape) + tuple(b.shape)))
    batch_dims = [shapes[label] for label in batch]
    batch_size = prod(batch_dims) if batch_dims else 1

    a_perm = (
        [labels_a.index(label) for label in batch]
        + [labels_a.index(label) for label in a_keep]
        + [labels_a.index(label) for label in contract]
    )
    b_perm = (
        [labels_b.index(label) for label in batch]
        + [labels_b.index(label) for label in contract]
        + [labels_b.index(label) for label in b_keep]
    )
    if a_perm != list(range(len(labels_a))):
        a = paddle.transpose(a, a_perm)
    if b_perm != list(range(len(labels_b))):
        b = paddle.transpose(b, b_perm)

    a_rows = _prod_shape(a_keep, shapes) if a_keep else 1
    a_contract = _prod_shape(contract, shapes) if contract else 1
    b_cols = _prod_shape(b_keep, shapes) if b_keep else 1

    a = paddle.reshape(a, [batch_size, a_rows, a_contract])
    b = paddle.reshape(b, [batch_size, a_contract, b_cols])
    merged = paddle.bmm(a, b).reshape(
        batch_dims + [shapes[label] for label in a_keep + b_keep]
    )
    return merged, batch + a_keep + b_keep


def einsum(equation, *operands):
    if len(operands) == 0 or "..." in equation:
        return paddle.einsum(equation, *operands)
    if "->" in equation:
        lhs, rhs = equation.split("->")
        out = list(rhs)
    else:
        lhs = equation
        flat = lhs.replace(",", "")
        out = [label for label in flat if flat.count(label) == 1]
    terms = lhs.split(",")
    if len(terms) != len(operands):
        return paddle.einsum(equation, *operands)
    labels = [list(term) for term in terms]
    # repeated or missing labels inside a single term are not handled here
    if any(len(set(term)) != len(term) for term in labels):
        return paddle.einsum(equation, *operands)
    if any(len(operand.shape) != len(term) for operand, term in zip(operands, labels)):
        return paddle.einsum(equation, *operands)

    result = operands[0]
    result_labels = labels[0]
    for index in range(1, len(operands)):
        later = set()
        for term in labels[index + 1 :]:
            later.update(term)
        result, result_labels = _merge(
            result,
            result_labels,
            operands[index],
            labels[index],
            out,
            later,
        )

    # sum labels that do not appear in the output
    reduce_axes = [
        axis for axis, label in enumerate(result_labels) if label not in out
    ]
    if reduce_axes:
        result = paddle.sum(result, axis=reduce_axes)
        result_labels = [label for label in result_labels if label in out]

    perm = [result_labels.index(label) for label in out]
    if perm != list(range(len(out))):
        result = paddle.transpose(result, perm)
    return result
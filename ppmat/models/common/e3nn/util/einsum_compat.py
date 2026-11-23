"""Double-gradient-compatible contractions used by SevenNet.

Paddle 3.1 does not provide a gradient of ``einsum_grad``. Atomistic force
training requires second-order automatic differentiation because forces are
first derivatives of energy.

Only contraction equations that are exercised by SevenNet are lowered here.
Other equations continue to use :func:`paddle.einsum`.
"""

from __future__ import annotations

import paddle


def _linear_uw_zui(
    w: paddle.Tensor,
    x: paddle.Tensor,
) -> paddle.Tensor:
    # uw,zui->zwi
    #
    # [z, u, i] -> [z, i, u]
    # [z, i, u] @ [u, w] -> [z, i, w]
    # -> [z, w, i]
    return paddle.matmul(
        paddle.transpose(
            x,
            perm=[0, 2, 1],
        ),
        w,
    ).transpose(
        [0, 2, 1]
    )


def _outer_zui_zuj(
    x1: paddle.Tensor,
    x2: paddle.Tensor,
) -> paddle.Tensor:
    # zui,zuj->zuij
    return (
        x1.unsqueeze(-1)
        * x2.unsqueeze(-2)
    )


def _outer_zui_zvj(
    x1: paddle.Tensor,
    x2: paddle.Tensor,
) -> paddle.Tensor:
    # zui,zvj->zuvij
    return (
        x1.unsqueeze(2)
        .unsqueeze(-1)
        *
        x2.unsqueeze(1)
        .unsqueeze(-2)
    )


def _contract_zuv_ijk_zuvij(
    w: paddle.Tensor,
    w3j: paddle.Tensor,
    xx: paddle.Tensor,
) -> paddle.Tensor:
    # zuv,ijk,zuvij->zuk
    #
    # First contract v with the per-edge/path weights:
    # [z,u,v,i,j] -> [z,u,i,j]
    weighted = paddle.sum(
        xx
        * w.unsqueeze(-1)
        .unsqueeze(-1),
        axis=2,
    )

    z = weighted.shape[0]
    u = weighted.shape[1]
    i = weighted.shape[2]
    j = weighted.shape[3]
    k = w3j.shape[2]

    lhs = weighted.reshape(
        [z * u, i * j]
    )

    rhs = w3j.reshape(
        [i * j, k]
    )

    return paddle.matmul(
        lhs,
        rhs,
    ).reshape(
        [z, u, k]
    )


def _contract_zuv_zu_zv(
    w: paddle.Tensor,
    x1: paddle.Tensor,
    x2: paddle.Tensor,
) -> paddle.Tensor:
    # zuv,zu,zv->zu
    return paddle.sum(
        w
        * x1.unsqueeze(-1)
        * x2.unsqueeze(1),
        axis=2,
    )


def _contract_zuv_zu_zvj(
    w: paddle.Tensor,
    x1: paddle.Tensor,
    x2: paddle.Tensor,
) -> paddle.Tensor:
    # zuv,zu,zvj->zuj
    return paddle.sum(
        w.unsqueeze(-1)
        * x1.unsqueeze(-1)
        .unsqueeze(-1)
        * x2.unsqueeze(1),
        axis=2,
    )


def _contract_zuv_zui_zv(
    w: paddle.Tensor,
    x1: paddle.Tensor,
    x2: paddle.Tensor,
) -> paddle.Tensor:
    # zuv,zui,zv->zui
    scale = paddle.sum(
        w
        * x2.unsqueeze(1),
        axis=2,
    )

    return (
        x1
        * scale.unsqueeze(-1)
    )


def _contract_zuv_zui_zvi(
    w: paddle.Tensor,
    x1: paddle.Tensor,
    x2: paddle.Tensor,
) -> paddle.Tensor:
    # zuv,zui,zvi->zu
    value = (
        w.unsqueeze(-1)
        * x1.unsqueeze(2)
        * x2.unsqueeze(1)
    )

    return paddle.sum(
        value,
        axis=[2, 3],
    )


def _scale_zui_zu(
    x1: paddle.Tensor,
    x2: paddle.Tensor,
) -> paddle.Tensor:
    # zui,zu->zui
    return (
        x1
        * x2.unsqueeze(-1)
    )


def einsum_compat(
    equation: str,
    *operands: paddle.Tensor,
) -> paddle.Tensor:
    """Dispatch selected contractions to double-grad-compatible Paddle ops."""

    if equation == "uw,zui->zwi":
        return _linear_uw_zui(
            operands[0],
            operands[1],
        )

    if equation == "zui,zuj->zuij":
        return _outer_zui_zuj(
            operands[0],
            operands[1],
        )

    if equation == "zui,zvj->zuvij":
        return _outer_zui_zvj(
            operands[0],
            operands[1],
        )

    if equation == "zuv,ijk,zuvij->zuk":
        return _contract_zuv_ijk_zuvij(
            operands[0],
            operands[1],
            operands[2],
        )

    if equation == "zuv,zu,zv->zu":
        return _contract_zuv_zu_zv(
            operands[0],
            operands[1],
            operands[2],
        )

    if equation == "zuv,zu,zvj->zuj":
        return _contract_zuv_zu_zvj(
            operands[0],
            operands[1],
            operands[2],
        )

    if equation == "zuv,zui,zv->zui":
        return _contract_zuv_zui_zv(
            operands[0],
            operands[1],
            operands[2],
        )

    if equation == "zuv,zui,zvi->zu":
        return _contract_zuv_zui_zvi(
            operands[0],
            operands[1],
            operands[2],
        )

    if equation == "zui,zu->zui":
        return _scale_zui_zu(
            operands[0],
            operands[1],
        )

    if equation in {"uvw,zu,zv->zw", "uvw,zui,zv->zwi"}:
        weight, features, species = operands
        u, v, w = weight.shape
        # Contract the species axis first to avoid a per-atom [u, v, w] tensor.
        mixed = paddle.matmul(
            species, weight.transpose([1, 0, 2]).reshape([v, u * w])
        ).reshape([-1, u, w])
        if equation == "uvw,zu,zv->zw":
            return paddle.bmm(features.unsqueeze(1), mixed).squeeze(1)
        return paddle.bmm(mixed.transpose([0, 2, 1]), features)

    # Other e3nn equations are untouched.
    return paddle.einsum(
        equation,
        *operands,
    )

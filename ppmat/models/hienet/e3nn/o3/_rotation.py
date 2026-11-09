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

import math

import paddle


def rand_matrix(*shape, requires_grad: bool = False, dtype=None, device=None):
    """random rotation matrix

    Parameters
    ----------
    *shape : int

    Returns
    -------
    `torch.Tensor`
        tensor of shape :math:`(\\mathrm{shape}, 3, 3)`
    """
    R = angles_to_matrix(*rand_angles(*shape, dtype=dtype, device=device))
    return R.detach().requires_grad_(requires_grad)


def identity_angles(*shape, requires_grad: bool = False, dtype=None, device=None):
    """angles of the identity rotation

    Parameters
    ----------
    *shape : int

    Returns
    -------
    alpha : `torch.Tensor`
        tensor of shape :math:`(\\mathrm{shape})`

    beta : `torch.Tensor`
        tensor of shape :math:`(\\mathrm{shape})`

    gamma : `torch.Tensor`
        tensor of shape :math:`(\\mathrm{shape})`
    """
    abc = paddle.zeros(3, *shape, dtype=dtype, device=device)
    return (
        abc[0].requires_grad_(requires_grad),
        abc[1].requires_grad_(requires_grad),
        abc[2].requires_grad_(requires_grad),
    )


def rand_angles(*shape, requires_grad: bool = False, dtype=None, device=None):
    """random rotation angles

    Parameters
    ----------
    *shape : int

    Returns
    -------
    alpha : `torch.Tensor`
        tensor of shape :math:`(\\mathrm{shape})`

    beta : `torch.Tensor`
        tensor of shape :math:`(\\mathrm{shape})`

    gamma : `torch.Tensor`
        tensor of shape :math:`(\\mathrm{shape})`
    """
    alpha, gamma = 2 * math.pi * paddle.rand(2, *shape, dtype=dtype, device=device)
    beta = paddle.rand(shape, dtype=dtype, device=device).mul(2).sub(1).acos()
    alpha = alpha.detach().requires_grad_(requires_grad)
    beta = beta.detach().requires_grad_(requires_grad)
    gamma = gamma.detach().requires_grad_(requires_grad)
    return alpha, beta, gamma


def compose_angles(a1, b1, c1, a2, b2, c2):
    """compose angles

    Computes :math:`(a, b, c)` such that :math:`R(a, b, c) = R(a_1, b_1, c_1) \\circ R(a_2, b_2, c_2)`

    Parameters
    ----------
    a1 : `torch.Tensor`
        tensor of shape :math:`(...)`, (applied second)

    b1 : `torch.Tensor`
        tensor of shape :math:`(...)`, (applied second)

    c1 : `torch.Tensor`
        tensor of shape :math:`(...)`, (applied second)

    a2 : `torch.Tensor`
        tensor of shape :math:`(...)`, (applied first)

    b2 : `torch.Tensor`
        tensor of shape :math:`(...)`, (applied first)

    c2 : `torch.Tensor`
        tensor of shape :math:`(...)`, (applied first)

    Returns
    -------
    alpha : `torch.Tensor`
        tensor of shape :math:`(...)`

    beta : `torch.Tensor`
        tensor of shape :math:`(...)`

    gamma : `torch.Tensor`
        tensor of shape :math:`(...)`
    """
    a1, b1, c1, a2, b2, c2 = paddle.broadcast_tensors([a1, b1, c1, a2, b2, c2])
    return matrix_to_angles(angles_to_matrix(a1, b1, c1) @ angles_to_matrix(a2, b2, c2))


def inverse_angles(a, b, c):
    """angles of the inverse rotation

    Parameters
    ----------
    a : `torch.Tensor`
        tensor of shape :math:`(...)`

    b : `torch.Tensor`
        tensor of shape :math:`(...)`

    c : `torch.Tensor`
        tensor of shape :math:`(...)`

    Returns
    -------
    alpha : `torch.Tensor`
        tensor of shape :math:`(...)`

    beta : `torch.Tensor`
        tensor of shape :math:`(...)`

    gamma : `torch.Tensor`
        tensor of shape :math:`(...)`
    """
    return -c, -b, -a


def identity_quaternion(*shape, requires_grad: bool = False, dtype=None, device=None):
    """quaternion of identity rotation

    Parameters
    ----------
    *shape : int

    Returns
    -------
    `torch.Tensor`
        tensor of shape :math:`(\\mathrm{shape}, 4)`
    """
    q = paddle.zeros(*shape, 4, dtype=dtype, device=device)
    q[..., 0] = 1
    q = q.detach().requires_grad_(requires_grad)
    return q


def rand_quaternion(*shape, requires_grad: bool = False, dtype=None, device=None):
    """generate random quaternion

    Parameters
    ----------
    *shape : int

    Returns
    -------
    `torch.Tensor`
        tensor of shape :math:`(\\mathrm{shape}, 4)`
    """
    q = angles_to_quaternion(*rand_angles(*shape, dtype=dtype, device=device))
    q = q.detach().requires_grad_(requires_grad)
    return q


def compose_quaternion(q1, q2) -> paddle.Tensor:
    """compose two quaternions: :math:`q_1 \\circ q_2`

    Parameters
    ----------
    q1 : `torch.Tensor`
        tensor of shape :math:`(..., 4)`, (applied second)

    q2 : `torch.Tensor`
        tensor of shape :math:`(..., 4)`, (applied first)

    Returns
    -------
    `torch.Tensor`
        tensor of shape :math:`(..., 4)`
    """
    q1, q2 = paddle.broadcast_tensors([q1, q2])
    return paddle.stack(
        [
            q1[..., 0] * q2[..., 0]
            - q1[..., 1] * q2[..., 1]
            - q1[..., 2] * q2[..., 2]
            - q1[..., 3] * q2[..., 3],
            q1[..., 1] * q2[..., 0]
            + q1[..., 0] * q2[..., 1]
            + q1[..., 2] * q2[..., 3]
            - q1[..., 3] * q2[..., 2],
            q1[..., 0] * q2[..., 2]
            - q1[..., 1] * q2[..., 3]
            + q1[..., 2] * q2[..., 0]
            + q1[..., 3] * q2[..., 1],
            q1[..., 0] * q2[..., 3]
            + q1[..., 1] * q2[..., 2]
            - q1[..., 2] * q2[..., 1]
            + q1[..., 3] * q2[..., 0],
        ],
        dim=-1,
    )


def inverse_quaternion(q):
    """inverse of a quaternion

    Works only for unit quaternions.

    Parameters
    ----------
    q : `torch.Tensor`
        tensor of shape :math:`(..., 4)`

    Returns
    -------
    `torch.Tensor`
        tensor of shape :math:`(..., 4)`
    """
    q = q.clone()
    q[..., 1:].neg_()
    return q


def rand_axis_angle(*shape, requires_grad: bool = False, dtype=None, device=None):
    """generate random rotation as axis-angle

    Parameters
    ----------
    *shape : int

    Returns
    -------
    axis : `torch.Tensor`
        tensor of shape :math:`(\\mathrm{shape}, 3)`

    angle : `torch.Tensor`
        tensor of shape :math:`(\\mathrm{shape})`
    """
    axis, angle = angles_to_axis_angle(*rand_angles(*shape, dtype=dtype, device=device))
    axis = axis.detach().requires_grad_(requires_grad)
    angle = angle.detach().requires_grad_(requires_grad)
    return axis, angle


def compose_axis_angle(axis1, angle1, axis2, angle2):
    """compose :math:`(\\vec x_1, \\alpha_1)` with :math:`(\\vec x_2, \\alpha_2)`

    Parameters
    ----------
    axis1 : `torch.Tensor`
        tensor of shape :math:`(..., 3)`, (applied second)

    angle1 : `torch.Tensor`
        tensor of shape :math:`(...)`, (applied second)

    axis2 : `torch.Tensor`
        tensor of shape :math:`(..., 3)`, (applied first)

    angle2 : `torch.Tensor`
        tensor of shape :math:`(...)`, (applied first)

    Returns
    -------
    axis : `torch.Tensor`
        tensor of shape :math:`(..., 3)`

    angle : `torch.Tensor`
        tensor of shape :math:`(...)`
    """
    return quaternion_to_axis_angle(
        compose_quaternion(
            axis_angle_to_quaternion(axis1, angle1),
            axis_angle_to_quaternion(axis2, angle2),
        )
    )


def matrix_x(angle: paddle.Tensor) -> paddle.Tensor:
    """matrix of rotation around X axis

    Parameters
    ----------
    angle : `torch.Tensor`
        tensor of any shape :math:`(...)`

    Returns
    -------
    `torch.Tensor`
        matrices of shape :math:`(..., 3, 3)`
    """
    c = angle.cos()
    s = angle.sin()
    o = paddle.ones_like(angle)
    z = paddle.zeros_like(angle)
    return paddle.stack(
        [
            paddle.stack([o, z, z], dim=-1),
            paddle.stack([z, c, -s], dim=-1),
            paddle.stack([z, s, c], dim=-1),
        ],
        dim=-2,
    )


def matrix_y(angle: paddle.Tensor) -> paddle.Tensor:
    """matrix of rotation around Y axis

    Parameters
    ----------
    angle : `torch.Tensor`
        tensor of any shape :math:`(...)`

    Returns
    -------
    `torch.Tensor`
        matrices of shape :math:`(..., 3, 3)`
    """
    c = angle.cos()
    s = angle.sin()
    o = paddle.ones_like(angle)
    z = paddle.zeros_like(angle)
    return paddle.stack(
        [
            paddle.stack([c, z, s], dim=-1),
            paddle.stack([z, o, z], dim=-1),
            paddle.stack([-s, z, c], dim=-1),
        ],
        dim=-2,
    )


def matrix_z(angle: paddle.Tensor) -> paddle.Tensor:
    """matrix of rotation around Z axis

    Parameters
    ----------
    angle : `torch.Tensor`
        tensor of any shape :math:`(...)`

    Returns
    -------
    `torch.Tensor`
        matrices of shape :math:`(..., 3, 3)`
    """
    c = angle.cos()
    s = angle.sin()
    o = paddle.ones_like(angle)
    z = paddle.zeros_like(angle)
    return paddle.stack(
        [
            paddle.stack([c, -s, z], dim=-1),
            paddle.stack([s, c, z], dim=-1),
            paddle.stack([z, z, o], dim=-1),
        ],
        dim=-2,
    )


def angles_to_matrix(alpha, beta, gamma) -> paddle.Tensor:
    """conversion from angles to matrix

    Parameters
    ----------
    alpha : `torch.Tensor`
        tensor of shape :math:`(...)`

    beta : `torch.Tensor`
        tensor of shape :math:`(...)`

    gamma : `torch.Tensor`
        tensor of shape :math:`(...)`

    Returns
    -------
    `torch.Tensor`
        matrices of shape :math:`(..., 3, 3)`
    """
    alpha, beta, gamma = paddle.broadcast_tensors([alpha, beta, gamma])
    return matrix_y(alpha) @ matrix_x(beta) @ matrix_y(gamma)


def matrix_to_angles(R):
    """conversion from matrix to angles

    Parameters
    ----------
    R : `torch.Tensor`
        matrices of shape :math:`(..., 3, 3)`

    Returns
    -------
    alpha : `torch.Tensor`
        tensor of shape :math:`(...)`

    beta : `torch.Tensor`
        tensor of shape :math:`(...)`

    gamma : `torch.Tensor`
        tensor of shape :math:`(...)`
    """
    assert paddle.compat.allclose(
        paddle.linalg.det(x=R), paddle.to_tensor(data=1, dtype=R.dtype)
    )
    x = R @ paddle.to_tensor(data=[0.0, 1.0, 0.0], dtype=R.dtype)
    a, b = xyz_to_angles(x)
    R = angles_to_matrix(a, b, paddle.zeros_like(a)).transpose(-1, -2) @ R
    c = paddle.atan2(R[..., 0, 2], R[..., 0, 0])
    return a, b, c


def angles_to_quaternion(alpha, beta, gamma) -> paddle.Tensor:
    """conversion from angles to quaternion

    Parameters
    ----------
    alpha : `torch.Tensor`
        tensor of shape :math:`(...)`

    beta : `torch.Tensor`
        tensor of shape :math:`(...)`

    gamma : `torch.Tensor`
        tensor of shape :math:`(...)`

    Returns
    -------
    `torch.Tensor`
        matrices of shape :math:`(..., 4)`
    """
    alpha, beta, gamma = paddle.broadcast_tensors([alpha, beta, gamma])
    qa = axis_angle_to_quaternion(
        paddle.to_tensor(data=[0.0, 1.0, 0.0], dtype=alpha.dtype), alpha
    )
    qb = axis_angle_to_quaternion(
        paddle.to_tensor(data=[1.0, 0.0, 0.0], dtype=beta.dtype), beta
    )
    qc = axis_angle_to_quaternion(
        paddle.to_tensor(data=[0.0, 1.0, 0.0], dtype=gamma.dtype), gamma
    )
    return compose_quaternion(qa, compose_quaternion(qb, qc))


def matrix_to_quaternion(R) -> paddle.Tensor:
    """conversion from matrix :math:`R` to quaternion :math:`q`

    Parameters
    ----------
    R : `torch.Tensor`
        tensor of shape :math:`(..., 3, 3)`

    Returns
    -------
    `torch.Tensor`
        tensor of shape :math:`(..., 4)`
    """
    return axis_angle_to_quaternion(*matrix_to_axis_angle(R))


def axis_angle_to_quaternion(xyz, angle) -> paddle.Tensor:
    """convertion from axis-angle to quaternion

    Parameters
    ----------
    xyz : `torch.Tensor`
        tensor of shape :math:`(..., 3)`

    angle : `torch.Tensor`
        tensor of shape :math:`(...)`

    Returns
    -------
    `torch.Tensor`
        tensor of shape :math:`(..., 4)`
    """
    xyz, angle = paddle.broadcast_tensors([xyz, angle[..., None]])
    xyz = paddle.nn.functional.normalize(xyz, dim=-1)
    c = paddle.cos(angle[..., :1] / 2)
    s = paddle.sin(angle / 2)
    return paddle.cat([c, xyz * s], dim=-1)


def quaternion_to_axis_angle(q):
    """convertion from quaternion to axis-angle

    Parameters
    ----------
    q : `torch.Tensor`
        tensor of shape :math:`(..., 4)`

    Returns
    -------
    axis : `torch.Tensor`
        tensor of shape :math:`(..., 3)`

    angle : `torch.Tensor`
        tensor of shape :math:`(...)`
    """
    angle = 2 * paddle.acos(q[..., 0].clamp(-1, 1))
    axis = paddle.nn.functional.normalize(q[..., 1:], dim=-1)
    return axis, angle


def matrix_to_axis_angle(R):
    """conversion from matrix to axis-angle

    Parameters
    ----------
    R : `torch.Tensor`
        tensor of shape :math:`(..., 3, 3)`

    Returns
    -------
    axis : `torch.Tensor`
        tensor of shape :math:`(..., 3)`

    angle : `torch.Tensor`
        tensor of shape :math:`(...)`
    """
    assert paddle.compat.allclose(
        paddle.linalg.det(x=R), paddle.to_tensor(data=1, dtype=R.dtype)
    )
    tr = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    angle = paddle.acos(tr.sub(1).div(2).clamp(-1, 1))
    axis = paddle.stack(
        [
            R[..., 2, 1] - R[..., 1, 2],
            R[..., 0, 2] - R[..., 2, 0],
            R[..., 1, 0] - R[..., 0, 1],
        ],
        dim=-1,
    )
    axis = paddle.nn.functional.normalize(axis, dim=-1)
    return axis, angle


def angles_to_axis_angle(alpha, beta, gamma):
    """conversion from angles to axis-angle

    Parameters
    ----------
    alpha : `torch.Tensor`
        tensor of shape :math:`(...)`

    beta : `torch.Tensor`
        tensor of shape :math:`(...)`

    gamma : `torch.Tensor`
        tensor of shape :math:`(...)`

    Returns
    -------
    axis : `torch.Tensor`
        tensor of shape :math:`(..., 3)`

    angle : `torch.Tensor`
        tensor of shape :math:`(...)`
    """
    return matrix_to_axis_angle(angles_to_matrix(alpha, beta, gamma))


def axis_angle_to_matrix(axis, angle) -> paddle.Tensor:
    """conversion from axis-angle to matrix

    Parameters
    ----------
    axis : `torch.Tensor`
        tensor of shape :math:`(..., 3)`

    angle : `torch.Tensor`
        tensor of shape :math:`(...)`

    Returns
    -------
    `torch.Tensor`
        tensor of shape :math:`(..., 3, 3)`
    """
    axis, angle = paddle.broadcast_tensors([axis, angle[..., None]])
    alpha, beta = xyz_to_angles(axis)
    R = angles_to_matrix(alpha, beta, paddle.zeros_like(beta))
    Ry = matrix_y(angle[..., 0])
    return R @ Ry @ R.transpose(-2, -1)


def quaternion_to_matrix(q) -> paddle.Tensor:
    """convertion from quaternion to matrix

    Parameters
    ----------
    q : `torch.Tensor`
        tensor of shape :math:`(..., 4)`

    Returns
    -------
    `torch.Tensor`
        tensor of shape :math:`(..., 3, 3)`
    """
    return axis_angle_to_matrix(*quaternion_to_axis_angle(q))


def quaternion_to_angles(q):
    """convertion from quaternion to angles

    Parameters
    ----------
    q : `torch.Tensor`
        tensor of shape :math:`(..., 4)`

    Returns
    -------
    alpha : `torch.Tensor`
        tensor of shape :math:`(...)`

    beta : `torch.Tensor`
        tensor of shape :math:`(...)`

    gamma : `torch.Tensor`
        tensor of shape :math:`(...)`
    """
    return matrix_to_angles(quaternion_to_matrix(q))


def axis_angle_to_angles(axis, angle):
    """convertion from axis-angle to angles

    Parameters
    ----------
    axis : `torch.Tensor`
        tensor of shape :math:`(..., 3)`

    angle : `torch.Tensor`
        tensor of shape :math:`(...)`

    Returns
    -------
    alpha : `torch.Tensor`
        tensor of shape :math:`(...)`

    beta : `torch.Tensor`
        tensor of shape :math:`(...)`

    gamma : `torch.Tensor`
        tensor of shape :math:`(...)`
    """
    return matrix_to_angles(axis_angle_to_matrix(axis, angle))


def angles_to_xyz(alpha, beta) -> paddle.Tensor:
    """convert :math:`(\\alpha, \\beta)` into a point :math:`(x, y, z)` on the sphere

    Parameters
    ----------
    alpha : `torch.Tensor`
        tensor of shape :math:`(...)`

    beta : `torch.Tensor`
        tensor of shape :math:`(...)`

    Returns
    -------
    `torch.Tensor`
        tensor of shape :math:`(..., 3)`

    Examples
    --------

    >>> angles_to_xyz(torch.tensor(1.7), torch.tensor(0.0)).abs()
    tensor([0., 1., 0.])
    """
    alpha, beta = paddle.broadcast_tensors([alpha, beta])
    x = paddle.sin(beta) * paddle.sin(alpha)
    y = paddle.cos(beta)
    z = paddle.sin(beta) * paddle.cos(alpha)
    return paddle.stack([x, y, z], dim=-1)


def xyz_to_angles(xyz):
    """convert a point :math:`\\vec r = (x, y, z)` on the sphere into angles :math:`(\\alpha, \\beta)`

    .. math::

        \\vec r = R(\\alpha, \\beta, 0) \\vec e_z


    Parameters
    ----------
    xyz : `torch.Tensor`
        tensor of shape :math:`(..., 3)`

    Returns
    -------
    alpha : `torch.Tensor`
        tensor of shape :math:`(...)`

    beta : `torch.Tensor`
        tensor of shape :math:`(...)`
    """
    xyz = paddle.nn.functional.normalize(xyz, p=2.0, dim=-1)
    xyz = xyz.clamp(-1, 1)
    beta = paddle.acos(xyz[..., 1])
    alpha = paddle.atan2(xyz[..., 0], xyz[..., 2])
    return alpha, beta

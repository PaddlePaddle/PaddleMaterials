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

import math

import paddle

from ppmat.models.common.e3nn import o3


class GaussianOrbital(paddle.nn.Layer):
    """
    Gaussian-type orbital

    .. math::
        \\psi_{n\\ell m}(\\mathbf{r})=
        \\sqrt{\\frac{2(2a_n)^{\\ell+3/2}}{\\Gamma(\\ell+3/2)}}
        \\exp(-a_n r^2) r^\\ell Y_{\\ell}^m(\\hat{\\mathbf{r}})

    """

    def __init__(self, gauss_start, gauss_end, num_gauss, lmax=7):
        super(GaussianOrbital, self).__init__()
        self.gauss_start = gauss_start
        self.gauss_end = gauss_end
        self.num_gauss = num_gauss
        self.lmax = lmax
        self.lc2lcm = BroadcastGTOTensor(lmax, num_gauss, src="lc", dst="lcm")
        self.m2lcm = BroadcastGTOTensor(lmax, num_gauss, src="m", dst="lcm")
        self.gauss: paddle.Tensor
        self.lognorm: paddle.Tensor
        self.register_buffer(
            name="gauss",
            tensor=paddle.linspace(start=gauss_start, stop=gauss_end, num=num_gauss),
        )
        self.register_buffer(name="lognorm", tensor=self._generate_lognorm())

    def _generate_lognorm(self):
        power = (paddle.arange(end=self.lmax + 1) + 1.5).unsqueeze(axis=-1)
        numerator = power * paddle.log(x=2 * self.gauss).unsqueeze(axis=0) + math.log(2)
        denominator = paddle.lgamma(x=power)
        lognorm = (numerator - denominator) / 2
        return lognorm.reshape([-1])

    def forward(self, vec):
        """
        Evaluate the basis functions
        :param vec: un-normalized vectors of (..., 3)
        :return: basis values of (..., (l+1)^2 * c)
        """

        r = vec.norm(axis=-1) + 1e-08
        spherical = o3.spherical_harmonics(
            list(range(self.lmax + 1)),
            vec / r[..., None],
            normalize=False,
            normalization="integral",
        )
        r = r.unsqueeze(axis=-1)
        lognorm = self.lognorm * paddle.ones_like(x=r)
        exponent = -self.gauss * (r * r)
        poly = paddle.arange(dtype="float32", end=self.lmax + 1) * paddle.log(x=r)
        log = exponent.unsqueeze(axis=-2) + poly.unsqueeze(axis=-1)
        log_flat = log.reshape([*tuple(log.shape)[:-2], -1])
        radial = paddle.exp(x=paddle.add(log_flat, lognorm))
        # Use explicit elementwise multiplication to avoid potential issues
        # with Python's `*` dispatch on Tensor-like objects.
        return paddle.multiply(self.lc2lcm(radial), self.m2lcm(spherical))


class BroadcastGTOTensor(paddle.nn.Layer):
    """
    Broadcast between spherical tensors of the Gaussian Type Orbitals (GTOs):

    .. math::
        \\{a_{clm}, 1\\le c\\le c_{max}, 0\\le\\ell\\le\\ell_{max},
        -\\ell\\le m\\le\\ell\\}

    For efficiency reason, the feature tensor is indexed by l, c, m.
    For example, for lmax = 3, cmax = 2, we have a tensor of 1s2s 1p2p 1d2d 1f2f.
    Currently, we support the following broadcasting:
    lc -> lcm;
    m -> lcm.
    """

    def __init__(self, lmax, cmax, src="lc", dst="lcm"):
        super(BroadcastGTOTensor, self).__init__()
        assert src in ["lc", "m"]
        assert dst in ["lcm"]
        self.src = src
        self.dst = dst
        self.lmax = lmax
        self.cmax = cmax
        if src == "lc":
            self.src_dim = (lmax + 1) * cmax
        else:
            self.src_dim = (lmax + 1) ** 2
        self.dst_dim = (lmax + 1) ** 2 * cmax
        if src == "lc":
            indices = self._generate_lc2lcm_indices()
        else:
            indices = self._generate_m2lcm_indices()
        self.register_buffer(name="indices", tensor=indices)

    def _generate_lc2lcm_indices(self):
        """
        lc -> lcm
        .. math::
            1s2s 1p2p → 1s2s 1p_x1p_y1p_z2p_x2p_y2p_z
        [0, 1, 2, 2, 2, 3, 3, 3]

        :return: (lmax+1)^2 * cmax
        """
        indices = [
            (degree * self.cmax + c)
            for degree in range(self.lmax + 1)
            for c in range(self.cmax)
            for _ in range(2 * degree + 1)
        ]
        return paddle.to_tensor(data=indices, dtype="int64")

    def _generate_m2lcm_indices(self):
        """
        m -> lcm
        .. math::
            s p_x p_y p_z → 1s2s 1p_x1p_y1p_z2p_x2p_y2p_z
        [0, 0, 1, 2, 3, 1, 2, 3]

        :return: (lmax+1)^2 * cmax
        """
        indices = [
            (degree * degree + m)
            for degree in range(self.lmax + 1)
            for _ in range(self.cmax)
            for m in range(2 * degree + 1)
        ]
        return paddle.to_tensor(data=indices, dtype="int64")

    def forward(self, x):
        """
        Apply broadcasting to x.
        :param x: (..., src_dim)
        :return: (..., dst_dim)
        """
        assert x.shape[-1] == self.src_dim, (
            f"Input dimension mismatch! Should be {self.src_dim}, "
            f"but got {x.shape[-1]} instead!"
        )
        if self.src == self.dst:
            return x
        return x[..., self.indices]

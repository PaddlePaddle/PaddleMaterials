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
from ._irreps import Irreps
from ._tensor_product._tensor_product import TensorProduct
from ..util.jit import compile_mode


@compile_mode("trace")
class Norm(paddle.nn.Module):
    """Norm of each irrep in a direct sum of irreps.

    Parameters
    ----------
    irreps_in : `e3nn.o3.Irreps`
        representation of the input

    squared : bool, optional
        Whether to return the squared norm. ``False`` by default, i.e. the norm itself (sqrt of squared norm) is returned.

    Examples
    --------
    Compute the norms of 17 vectors.

    >>> norm = Norm("17x1o")
    >>> norm(torch.randn(17 * 3)).shape
    torch.Size([17])
    """

    squared: bool

    def __init__(self, irreps_in, squared: bool = False) -> None:
        super().__init__()
        irreps_in = Irreps(irreps_in).simplify()
        irreps_out = Irreps([(mul, "0e") for mul, _ in irreps_in])
        instr = [
            (i, i, i, "uuu", False, ir.dim) for i, (mul, ir) in enumerate(irreps_in)
        ]
        self.tp = TensorProduct(
            irreps_in, irreps_in, irreps_out, instr, irrep_normalization="component"
        )
        self.irreps_in = irreps_in
        self.irreps_out = irreps_out.simplify()
        self.squared = squared

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.irreps_in})"

    def forward(self, features):
        """Compute norms of irreps in ``features``.

        Parameters
        ----------
        features : `torch.Tensor`
            tensor of shape ``(..., irreps_in.dim)``

        Returns
        -------
        `torch.Tensor`
            tensor of shape ``(..., irreps_out.dim)``
        """
        out = self.tp(features, features)
        if self.squared:
            return out
        else:
            return out.relu().sqrt()

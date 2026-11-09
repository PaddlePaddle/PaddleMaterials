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

from ._irreps import Irrep, Irreps
from ._linear import Linear
from ._norm import Norm
from ._reduce import ReducedTensorProducts
from ._rotation import (angles_to_axis_angle, angles_to_matrix,
                        angles_to_quaternion, angles_to_xyz,
                        axis_angle_to_angles, axis_angle_to_matrix,
                        axis_angle_to_quaternion, compose_angles,
                        compose_axis_angle, compose_quaternion,
                        identity_angles, identity_quaternion, inverse_angles,
                        inverse_quaternion, matrix_to_angles,
                        matrix_to_axis_angle, matrix_to_quaternion, matrix_x,
                        matrix_y, matrix_z, quaternion_to_angles,
                        quaternion_to_axis_angle, quaternion_to_matrix,
                        rand_angles, rand_axis_angle, rand_matrix,
                        rand_quaternion, xyz_to_angles)
from ._spherical_harmonics import SphericalHarmonics, spherical_harmonics
from ._tensor_product import (ElementwiseTensorProduct, FullTensorProduct,
                              FullyConnectedTensorProduct, Instruction,
                              TensorProduct, TensorSquare)
from ._wigner import (change_basis_real_to_complex, so3_generators,
                      su2_generators, wigner_3j, wigner_D)
from .experimental import FullTensorProductv2

__all__ = [
    "rand_matrix",
    "identity_angles",
    "rand_angles",
    "compose_angles",
    "inverse_angles",
    "identity_quaternion",
    "rand_quaternion",
    "compose_quaternion",
    "inverse_quaternion",
    "rand_axis_angle",
    "compose_axis_angle",
    "matrix_x",
    "matrix_y",
    "matrix_z",
    "angles_to_matrix",
    "matrix_to_angles",
    "angles_to_quaternion",
    "matrix_to_quaternion",
    "axis_angle_to_quaternion",
    "quaternion_to_axis_angle",
    "matrix_to_axis_angle",
    "angles_to_axis_angle",
    "axis_angle_to_matrix",
    "quaternion_to_matrix",
    "quaternion_to_angles",
    "axis_angle_to_angles",
    "angles_to_xyz",
    "xyz_to_angles",
    "wigner_D",
    "wigner_3j",
    "change_basis_real_to_complex",
    "su2_generators",
    "so3_generators",
    "Irrep",
    "Irreps",
    "irrep",
    "Instruction",
    "TensorProduct",
    "FullyConnectedTensorProduct",
    "ElementwiseTensorProduct",
    "FullTensorProduct",
    "FullTensorProductv2",
    "TensorSquare",
    "SphericalHarmonics",
    "spherical_harmonics",
    "ReducedTensorProducts",
    "Linear",
    "Norm",
]

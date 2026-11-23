# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import annotations

from os import PathLike
from typing import List
from typing import Literal
from typing import Optional
from typing import Sequence
from typing import Union

import numpy as np
from p_tqdm import p_map
from PIL import Image


class BuildImage:
    """Build channel-first NumPy image arrays from image files.

    Args:
        format: Format of the input image data. Currently only ``image_file``
            is supported.
        mode: Pillow image mode used when reading the file. Defaults to ``L``.
        dtype: NumPy dtype of the output array. Defaults to ``float32``.
        num_cpus: Number of CPUs used when building a list of images. Defaults
            to 1.
    """

    def __init__(
        self,
        format: Literal["image_file"] = "image_file",
        mode: str = "L",
        dtype: str = "float32",
        num_cpus: Optional[int] = None,
    ) -> None:
        self.format = format
        self.mode = mode
        self.dtype = dtype
        self.num_cpus = 1 if num_cpus is None else int(num_cpus)

    @staticmethod
    def build_one(
        image_data: Union[str, PathLike[str]],
        format: str,
        mode: str,
        dtype: str,
    ) -> np.ndarray:
        if format != "image_file":
            raise ValueError(f"Invalid format specified: {format}")

        with Image.open(image_data) as image:
            image_array = np.asarray(image.convert(mode), dtype=dtype)

        if image_array.ndim == 2:
            return image_array[None, ...]
        if image_array.ndim == 3:
            return image_array.transpose(2, 0, 1)
        raise ValueError(
            "Expected a two- or three-dimensional image array, "
            f"but got shape {image_array.shape}."
        )

    def __call__(
        self,
        images_data: Union[
            Sequence[Union[str, PathLike[str]]],
            str,
            PathLike[str],
        ],
    ) -> Union[List[np.ndarray], np.ndarray]:
        if isinstance(images_data, (list, tuple)):
            return p_map(
                BuildImage.build_one,
                images_data,
                [self.format] * len(images_data),
                [self.mode] * len(images_data),
                [self.dtype] * len(images_data),
                num_cpus=self.num_cpus,
                desc="Building images",
                dynamic_ncols=True,
                mininterval=0.2,
            )
        return BuildImage.build_one(
            images_data,
            self.format,
            self.mode,
            self.dtype,
        )

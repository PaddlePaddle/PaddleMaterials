# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import os.path as osp
from typing import Optional
from typing import Sequence

import paddle
from PIL import Image
from tqdm import tqdm

from ppmat.datasets.build_image import BuildImage
from ppmat.predictor.base import BasePredictor
from ppmat.utils import logger


class SpectrumPredictor(BasePredictor):
    """Spectrum enhancement predictor.

    This class provides an interface for enhancing spectrum images using
    pre-trained deep learning models. Supports two initialization modes:

    1. **Automatic Model Loading**
       Specify `model_name` and `weights_name` to automatically download
       and load pre-trained weights from the `MODEL_REGISTRY`.

    2. **Custom Model Loading**
       Provide explicit `config_path` and `checkpoint_path` to load
       custom-trained models from local files.

    Args:
        model_name (Optional[str], optional): Name of the pre-defined model architecture
            from the `MODEL_REGISTRY` registry. When specified, associated weights
            will be automatically downloaded. Defaults to None.

        weights_name (Optional[str], optional): Specific pre-trained weight identifier.
            Used only when `model_name` is provided. Valid options include:
            - 'best.pdparams' (highest validation performance)
            - 'latest.pdparams' (most recent training checkpoint)
            - Custom weight files ending with '.pdparams'
            Defaults to None.

        config_path (Optional[str], optional): Path to model configuration file (YAML)
            for custom models. Required when not using predefined `model_name`.
            Defaults to None.
        checkpoint_path (Optional[str], optional): Path to a model checkpoint file
            (.pdparams) for custom models. If omitted, `Predict.checkpoint_path` from
            the config is used. Defaults to None.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        weights_name: Optional[str] = None,
        config_path: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
        device: Optional[str] = None,
        config_overrides: Optional[Sequence[str]] = None,
    ):
        super().__init__(
            model_name=model_name,
            weights_name=weights_name,
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            work_dir="",
            device=device,
            config_overrides=config_overrides,
        )
        self.load_inference_model()

    def from_image(self, image):
        return self._run_model(image)

    def from_image_file(
        self,
        image_file_path: str,
        save_path: Optional[str] = None,
    ):
        """Predict enhanced spectra from an image file or directory.

        Args:
            image_file_path: Path to one image or a directory of images.
            save_path: Optional directory for enhanced PNG images.

        Returns:
            List of prediction dictionaries.
        """
        if osp.isdir(image_file_path):
            image_files = sorted(
                osp.join(image_file_path, file_name)
                for file_name in os.listdir(image_file_path)
                if osp.splitext(file_name)[1].lower()
                in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
            )
        else:
            image_files = [image_file_path]

        results = []
        for image_path in tqdm(image_files, desc="Predict"):
            image = BuildImage(
                format="image_file",
                mode="L",
                dtype="float32",
            )(image_path)
            result = self.from_image(image)
            results.append(result)

        if save_path is not None and results:
            for image_file, result in zip(image_files, results):
                pred = (
                    paddle.clip(result[self.model.target_name], 0, 255)
                    .squeeze()
                    .cpu()
                    .numpy()
                    .astype("uint8")
                )
                Image.fromarray(pred).save(
                    osp.join(
                        save_path, f"{osp.splitext(osp.basename(image_file))[0]}.png"
                    )
                )
            logger.info(f"Saved prediction results to {save_path}")

        return results

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

from __future__ import absolute_import
from __future__ import annotations

import hashlib
import os
import os.path as osp
import pickle
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
import paddle
import paddle.distributed as dist
from paddle.io import Dataset

from ppmat.datasets.build_image import BuildImage
from ppmat.datasets.build_matched_name import build_matched_name_samples
from ppmat.datasets.build_matched_name import build_prediction_samples
from ppmat.utils import download
from ppmat.utils import io
from ppmat.utils import logger
from ppmat.utils.misc import is_equal


class SFINDataset(Dataset):
    """SFIN Dataset Handler.

    This class loads paired STEM images for the SFIN spectrum enhancement
    benchmark. Each noisy image is paired with two supervision targets:
    ``gt_enhance`` for denoised image restoration and ``gt_detect`` for
    structure/detail detection.

    **Dataset Overview**
    - **Source**: Original data from the SFIN STEM image enhancement benchmark.
    - **Preprocessed Version**:
    ```
    ┌───────────────────┬─────────┬─────────┐
    │ Dataset Partition │ Train   │ Val/Test│
    ├───────────────────┼─────────┼─────────┤
    │ HAADF             │ 1,000   │ 100     │
    │ BF                │ 1,000   │ 100     │
    └───────────────────┴─────────┴─────────┘
    ```
    Download preprocessed data:
    https://paddle-org.bj.bcebos.com/paddlematerials/datasets/SFIN/sfin_haadf.zip
    https://paddle-org.bj.bcebos.com/paddlematerials/datasets/SFIN/sfin_bf.zip

    **Data Format**
    The dataset is stored as paired PNG images. A sample is matched by file name
    across the following folders:

    | Folder       | Description                              | Example Value |
    |--------------|------------------------------------------|---------------|
    | `noisy`      | Low-dose noisy STEM input image          | 0001.png      |
    | `gt_enhance` | Ground-truth enhanced/restored image     | 0001.png      |
    | `gt_detect`  | Ground-truth structure/detail label      | 0001.png      |

    **Example Row:**
    ```python
    {
        "noisy": Tensor(shape=[1, H, W]),
        "gt_enhance": Tensor(shape=[1, H, W]),
        "gt_detect": Tensor(shape=[1, H, W]),
        "name": "0001",
        "id": 0,
    }
    ```

    Args:
        path (str, optional): The path of the dataset root. The folder name
            should be ``sfin_haadf`` or ``sfin_bf`` for automatic download.
            Defaults to ``./sfin_haadf``.
        split (str, optional): Dataset split, selected from ``train``, ``val``
            and ``test``. ``val`` uses the same files as ``test``. Defaults to
            ``train``.
        target_subdir (Optional[str], optional): Target folder used as the
            training label. Set to ``None`` for prediction-only input loading.
            Defaults to ``gt_enhance``.
        data_count (Optional[int], optional): If set, only the first
            ``data_count`` samples are used. Defaults to None.
        build_samples_cfg (Optional[Dict[str, Any]], optional): Configuration
            for matching noisy and target files by name. Defaults to indexed
            matching.
        build_image_cfg (Optional[Dict[str, Any]], optional): Configuration for
            building channel-first NumPy image arrays. Defaults to grayscale
            ``float32`` image-file loading.
        transforms (Optional[Callable], optional): Preprocess transforms for
            each sample. Defaults to None.
        cache_path (Optional[str], optional): Explicit path for the cache
            directory. Defaults to None.
        overwrite (bool, optional): Whether to rebuild existing cache files.
            Defaults to False.
    """

    name = "sfin"
    url = (
        "https://paddle-org.bj.bcebos.com/paddlematerials/datasets/"
        "SFIN/sfin_haadf.zip"
    )
    md5 = "f96dea9ac1f722d6ca55c7e49c1b3a41"
    bf_url = (
        "https://paddle-org.bj.bcebos.com/paddlematerials/datasets/" "SFIN/sfin_bf.zip"
    )
    bf_md5 = "74ec1c1959e162669cc8cbbc8713bda0"
    label_subdirs = ("gt_enhance", "gt_detect")

    def __init__(
        self,
        path: str = "./sfin_haadf",
        split: str = "train",
        target_subdir: Optional[str] = "gt_enhance",
        data_count: Optional[int] = None,
        build_samples_cfg: Optional[Dict[str, Any]] = None,
        build_image_cfg: Optional[Dict[str, Any]] = None,
        transforms: Optional[Callable] = None,
        cache_path: Optional[str] = None,
        overwrite: bool = False,
    ):
        super().__init__()

        if split not in ("train", "val", "test"):
            raise ValueError(
                f"Unsupported split '{split}', expected 'train', 'val' or 'test'."
            )
        if target_subdir is not None and target_subdir not in self.label_subdirs:
            raise ValueError(
                f"Unsupported target_subdir '{target_subdir}', expected "
                "'gt_enhance', 'gt_detect', or None."
            )
        if data_count is not None and int(data_count) < 0:
            raise ValueError("data_count must be None or a non-negative integer.")

        self.path = path
        self.dataset_name = osp.basename(osp.normpath(path))
        self.split = "test" if split == "val" else split
        self.target_subdir = target_subdir
        self.data_count = int(data_count) if data_count is not None else None
        self.transforms = transforms
        self.cache_path = cache_path
        self.overwrite = overwrite
        self.url, self.md5 = self._get_dataset_url_md5()

        if not osp.exists(path):
            if self.url is None:
                raise FileNotFoundError(f"Dataset root not found: {path}")
            logger.message("The dataset is not found. Will download it now.")
            root_path = download.get_datasets_path_from_url(self.url, self.md5)
            path = self._get_downloaded_dataset_path(root_path)
            self.path = path

        if build_samples_cfg is None:
            build_samples_cfg = {"match_mode": "indexed"}
            logger.message(
                "The build_samples_cfg is not set, will use the default "
                f"configs: {build_samples_cfg}"
            )
        self.build_samples_cfg = build_samples_cfg
        self.sample_builder = build_matched_name_samples(build_samples_cfg)

        if build_image_cfg is None:
            build_image_cfg = {
                "format": "image_file",
                "mode": "L",
                "dtype": "float32",
                "num_cpus": 1,
            }
            logger.message(
                "The build_image_cfg is not set, will use the default "
                f"configs: {build_image_cfg}"
            )
        self.build_image_cfg = build_image_cfg
        self.image_builder = BuildImage(**build_image_cfg)

        self.data_root = osp.join(self.path, self.split)
        self.noisy_root = osp.join(self.data_root, "noisy")
        self.target_roots = {
            name: osp.join(self.data_root, name) for name in self.label_subdirs
        }
        self.target_root = (
            self.target_roots[self.target_subdir]
            if self.target_subdir is not None
            else None
        )

        self.row_data, self.num_samples = self.read_data(self.path)
        self.samples = self.row_data["samples"]
        self.file_names = self.row_data["name"]
        self._prepare_cache()
        logger.info(f"Load {self.num_samples} samples from {self.path}")

    def _get_dataset_url_md5(self) -> Tuple[Optional[str], Optional[str]]:
        """Return download metadata for known SFIN dataset folders."""
        if self.dataset_name == "sfin_haadf":
            return self.url, self.md5
        if self.dataset_name == "sfin_bf":
            return self.bf_url, self.bf_md5
        return None, None

    def _get_downloaded_dataset_path(self, root_path: str) -> str:
        """Resolve the dataset root returned by the shared download utility."""
        for candidate in (root_path, osp.join(root_path, self.dataset_name)):
            if osp.isdir(osp.join(candidate, self.split, "noisy")):
                return candidate
        return osp.join(root_path, self.dataset_name)

    def read_data(self, path: str) -> Tuple[Dict[str, List[Any]], int]:
        """Read image file names and build matched sample metadata."""
        if not osp.isdir(self.noisy_root):
            raise FileNotFoundError(f"Noisy directory not found: {self.noisy_root}")

        noisy_files = io.list_files_by_suffix(self.noisy_root, ".png")
        if self.target_subdir is None:
            samples = build_prediction_samples(noisy_files)
        else:
            for label_name, target_root in self.target_roots.items():
                if not osp.isdir(target_root):
                    raise FileNotFoundError(
                        f"Target directory not found: {target_root}"
                    )

            target_files = io.list_files_by_suffix(self.target_root, ".png")
            samples = self.sample_builder(
                noisy_files,
                target_files,
                self.noisy_root,
                self.target_root,
                ".png",
            )
            self._add_label_files(samples, noisy_files)

        if self.data_count is not None:
            samples = samples[: self.data_count]
        if not samples and self.data_count != 0:
            raise FileNotFoundError(f"No samples found under {self.noisy_root}.")

        row_data = {
            "samples": samples,
            "noisy": [sample["noisy"] for sample in samples],
            "name": [sample["name"] for sample in samples],
        }
        if self.target_subdir is not None:
            row_data["target"] = [sample["target"] for sample in samples]
            for label_name in self.label_subdirs:
                row_data[label_name] = [sample[label_name] for sample in samples]
        return row_data, len(samples)

    def _add_label_files(self, samples: List[Dict[str, str]], noisy_files: List[str]):
        """Attach both SFIN label paths to matched samples by image name."""
        for label_name, target_root in self.target_roots.items():
            if label_name == self.target_subdir:
                for sample in samples:
                    sample[label_name] = sample["target"]
                continue

            label_files = io.list_files_by_suffix(target_root, ".png")
            label_samples = self.sample_builder(
                noisy_files,
                label_files,
                self.noisy_root,
                target_root,
                ".png",
            )
            label_file_by_name = {
                sample["name"]: sample["target"] for sample in label_samples
            }
            for sample in samples:
                sample[label_name] = label_file_by_name[sample["name"]]

    def _prepare_cache(self):
        if self.cache_path is None:
            target_name = (
                self.target_subdir if self.target_subdir is not None else "predict"
            )
            self.cache_path = osp.join(
                f"{self.path}_cache", self.split, str(target_name)
            )
        logger.info(f"Cache path: {self.cache_path}")

        sample_cache_path = osp.join(self.cache_path, "samples")
        sample_done_flag = osp.join(sample_cache_path, "completed.flag")
        cache_cfg = self._cache_cfg()
        cache_cfg_path = osp.join(self.cache_path, "cache_cfg.pkl")

        cache_exists = osp.exists(self.cache_path)
        if cache_exists and not self.overwrite:
            logger.warning(
                "Cache enabled. If a cache file exists, it will be automatically "
                "read and current settings will be ignored. Please ensure that the "
                "settings used in match your current settings."
            )

        rebuild_cache = (
            self.overwrite
            or not cache_exists
            or not self._is_cache_valid(
                sample_cache_path, sample_done_flag, cache_cfg_path, cache_cfg
            )
        )
        if rebuild_cache and self._rank() == 0:
            self._build_cache(sample_cache_path, sample_done_flag, cache_cfg_path)

        if dist.is_initialized():
            dist.barrier()

        self.cache_files = [
            osp.join(sample_cache_path, f"{idx:010d}.pkl")
            for idx in range(self.num_samples)
        ]
        if not all(osp.exists(cache_file) for cache_file in self.cache_files):
            raise RuntimeError(
                f"No complete cached SFIN samples found under {sample_cache_path}."
            )

    @staticmethod
    def _file_fingerprint(path: str) -> Dict[str, Any]:
        path = osp.abspath(path)
        digest = hashlib.sha256()
        with open(path, "rb") as file_obj:
            for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
                digest.update(chunk)
        stat = os.stat(path)
        return {
            "path": path,
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "sha256": digest.hexdigest(),
        }

    def _source_fingerprints(self) -> List[Dict[str, Any]]:
        paths = set()
        for sample in self.samples:
            for key, value in sample.items():
                if key == "name" or not isinstance(value, (str, os.PathLike)):
                    continue
                if key == "noisy":
                    paths.add(osp.join(self.noisy_root, value))
                elif key in self.target_roots:
                    paths.add(osp.join(self.target_roots[key], value))
        return [self._file_fingerprint(path) for path in sorted(paths)]

    def _cache_cfg(self) -> Dict[str, Any]:
        return {
            "build_image_cfg": self.build_image_cfg,
            "build_samples_cfg": self.build_samples_cfg,
            "path": osp.abspath(self.path),
            "split": self.split,
            "target_subdir": self.target_subdir,
            "sample_names": self.file_names,
            "num_samples": self.num_samples,
            "source_files": self._source_fingerprints(),
        }

    def _is_cache_valid(
        self,
        sample_cache_path: str,
        sample_done_flag: str,
        cache_cfg_path: str,
        cache_cfg: Dict[str, Any],
    ) -> bool:
        try:
            cache_cfg_saved = self.load_from_cache(cache_cfg_path)
        except Exception as e:
            logger.warning(e)
            return False

        if not is_equal(cache_cfg_saved, cache_cfg):
            logger.warning("cache_cfg is different from current config.")
            return False

        num_cached = self._count_cache_files(sample_cache_path)
        if osp.exists(sample_done_flag) and num_cached == self.num_samples:
            logger.info(f"Using cached SFIN samples ({num_cached}).")
            return True

        logger.warning(
            f"Cached SFIN samples are incomplete "
            f"(cached={num_cached}, expected={self.num_samples})."
        )
        return False

    def _build_cache(
        self,
        sample_cache_path: str,
        sample_done_flag: str,
        cache_cfg_path: str,
    ):
        os.makedirs(self.cache_path, exist_ok=True)
        os.makedirs(sample_cache_path, exist_ok=True)
        self._clean_cache_dir(sample_cache_path)
        self.save_to_cache(cache_cfg_path, self._cache_cfg())

        logger.message(
            f"Caching {self.num_samples} SFIN samples to {sample_cache_path}"
        )
        for idx in range(self.num_samples):
            self.save_to_cache(
                osp.join(sample_cache_path, f"{idx:010d}.pkl"),
                self._serialize_item(self._build_item(idx)),
            )
        with open(sample_done_flag, "w") as f:
            f.write("done")

    @staticmethod
    def _rank() -> int:
        return dist.get_rank() if dist.is_initialized() else 0

    @staticmethod
    def _count_cache_files(cache_path: str) -> int:
        if not osp.isdir(cache_path):
            return 0
        return len([name for name in os.listdir(cache_path) if name.endswith(".pkl")])

    @staticmethod
    def _clean_cache_dir(cache_path: str):
        for file_name in os.listdir(cache_path):
            if file_name.endswith(".pkl") or file_name.endswith(".flag"):
                os.remove(osp.join(cache_path, file_name))

    def _build_item(self, idx: int) -> Dict[str, Any]:
        data = {
            "noisy": self.image_builder(
                osp.join(self.noisy_root, self.row_data["noisy"][idx])
            ),
            "name": self.row_data["name"][idx],
            "id": idx,
        }

        if self.target_subdir is not None:
            for label_name, target_root in self.target_roots.items():
                data[label_name] = self.image_builder(
                    osp.join(target_root, self.row_data[label_name][idx])
                )
        return data

    @staticmethod
    def _serialize_item(data: Dict[str, Any]) -> Dict[str, Any]:
        return {
            key: value.detach().cpu().numpy()
            if isinstance(value, paddle.Tensor)
            else value
            for key, value in data.items()
        }

    @staticmethod
    def _deserialize_item(data: Dict[str, Any]) -> Dict[str, Any]:
        return {
            key: paddle.to_tensor(value) if isinstance(value, np.ndarray) else value
            for key, value in data.items()
        }

    def save_to_cache(self, cache_path: str, data: Any):
        with open(cache_path, "wb") as f:
            pickle.dump(data, f)

    def load_from_cache(self, cache_path: str):
        if not osp.exists(cache_path):
            raise FileNotFoundError(f"No such file or directory: {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    def __getitem__(self, idx: int):
        data = self._deserialize_item(self.load_from_cache(self.cache_files[idx]))
        data = self.transforms(data) if self.transforms is not None else data
        return data

    def __len__(self):
        return self.num_samples

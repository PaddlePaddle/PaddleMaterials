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

import hashlib
import pickle
from pathlib import Path

import numpy as np

from ppmat.datasets.build_structure import BuildStructure
from ppmat.datasets.custom_data_type import ConcatData
from ppmat.utils import download
from ppmat.utils import logger


class AsymmetricUnitTransform:
    """Add cached ASU supervision to a crystal dataset's structure array.

    Args:
        symprec (float): Symmetry tolerance in angstroms. Defaults to 0.1.
        angle_tolerance (float): Angular tolerance in degrees. Defaults to 5.0.
        mapping_tolerance (float): Fractional ASU membership tolerance.
            Defaults to 1e-6.
        cif_key (str, optional): Top-level CIF property used for symmetry
            analysis at its original precision. Defaults to structure_array.
        cache_dir (str, optional): Derived-feature cache directory. Defaults to
            the unified dataset cache under sgequidiff/asu.
    """

    def __init__(
        self,
        symprec=0.1,
        angle_tolerance=5.0,
        mapping_tolerance=1e-6,
        cif_key=None,
        cache_dir=None,
    ):
        from ppmat.models.sgequidiff.asu import AsymmetricUnitConverter

        self.converter = AsymmetricUnitConverter(
            symprec=symprec,
            angle_tolerance=angle_tolerance,
            mapping_tolerance=mapping_tolerance,
        )
        self.cif_key = cif_key
        self.cache_dir = (
            Path(cache_dir)
            if cache_dir is not None
            else Path(download.DATASETS_HOME) / "sgequidiff" / "asu"
        )

    def __call__(self, data):
        from ppmat.models.sgequidiff.asu import save_asu_cache

        digest = hashlib.sha256(self.converter.cache_key.encode())
        if self.cif_key is None:
            array = data["structure_array"]
            source = {
                "atom_types": array["atom_types"].data,
                "frac_coords": array["frac_coords"].data,
                "lengths": array["lengths"].data.reshape(3),
                "angles": array["angles"].data.reshape(3),
            }
            input_format = "array"
            for key, value in source.items():
                digest.update(str((key, value.shape, value.dtype.str)).encode())
                digest.update(value.tobytes())
        else:
            source = data[self.cif_key]
            input_format = "cif_str"
            digest.update(source.encode())
        cache_path = self.cache_dir / f"{digest.hexdigest()}.pkl"
        crystal = None
        if cache_path.is_file():
            try:
                with cache_path.open("rb") as file_obj:
                    crystal = pickle.load(file_obj)
            except (EOFError, OSError, pickle.UnpicklingError):
                logger.warning(f"Rebuilding invalid ASU cache: {cache_path}")
        if crystal is None:
            structure = BuildStructure.build_one(
                source, input_format, niggli=False, canocial=False
            )
            crystal = self.converter(structure)
            save_asu_cache(cache_path, crystal)
        for key, value in crystal.items():
            data[key] = value.copy() if isinstance(value, np.ndarray) else value
        for key in (
            "element_indices",
            "wyckoff_indices",
            "wyckoff_shape_indices",
            "frac_coords",
        ):
            data[key] = ConcatData(data[key])
        return data

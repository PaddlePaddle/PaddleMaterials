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

"""Wyckoff shape decomposition cache builder and cache-location management.

Builds ``wyckoff_shape_decomposition.pkl`` from the registered ASU sites
vocabulary role (see ``ppmat.vocab.build_vocab``) and manages the on-disk
cache directory shared by the SGEquiDiff derived-cache files.
"""

import pickle
from fractions import Fraction
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull

from ppmat.models.sgequidiff.vocabs import VOCAB_NAME
from ppmat.utils import download
from ppmat.utils import logger
from ppmat.vocab import build_vocab


def get_data_directory() -> Path:
    """Writable cache directory for SGEquiDiff derived assets.

    Resolved under ``~/.paddlemat/datasets`` (the unified dataset cache
    root) and created on demand.
    """
    directory = Path(download.DATASETS_HOME) / "sgequidiff"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def get_shape_decomp_dict_path() -> Path:
    """Path of the cached wyckoff_shape_decomposition.pkl under the data directory."""
    return get_data_directory() / "wyckoff_shape_decomposition.pkl"


def ensure_wyckoff_shape_decomp() -> None:
    """Ensure wyckoff_shape_decomposition.pkl exists
    (generated from the registered ASU sites vocabulary role).
    """
    shape_decomp_path = get_shape_decomp_dict_path()
    if shape_decomp_path.exists():
        return

    asu_dict = build_vocab(VOCAB_NAME)["asu_sites"]["data"]
    build_shape_decomp_dict(str(shape_decomp_path), asu_dict)


def _to_array(region) -> np.ndarray:
    """Convert vertex list to numpy array, supporting fraction strings like "1/2"."""
    if isinstance(region, np.ndarray):
        return region.astype(np.float64)
    result = []
    for pt in region:
        if isinstance(pt, (list, tuple)):
            coords = []
            for c in pt:
                if isinstance(c, str):
                    coords.append(float(Fraction(c)))
                else:
                    coords.append(float(c))
            result.append(coords)
        else:
            result.append([float(pt)])
    return np.array(result, dtype=np.float64)


def _fan_triangulate_convex_polygon_3d(vertices_3d: np.ndarray):
    """Fan triangulation of convex polygon in 3D (from centroid)."""
    n = len(vertices_3d)
    if n < 3:
        return np.zeros((0, 3, 3), dtype=np.float64), np.zeros(0, dtype=np.float64)

    centroid = vertices_3d.mean(axis=0)
    triangles = []
    areas = []
    for i in range(n):
        v0 = centroid
        v1 = vertices_3d[i]
        v2 = vertices_3d[(i + 1) % n]
        tri = np.stack([v0, v1, v2], axis=0)  # (3, 3)
        # triangle area = 0.5 * |cross(v1-v0, v2-v0)|
        area = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))
        triangles.append(tri)
        areas.append(area)
    return np.array(triangles, dtype=np.float64), np.array(areas, dtype=np.float64)


def _compute_3d_convex_hull_volume(vertices: np.ndarray) -> float:
    """Compute volume of a 3D convex polytope."""
    if len(vertices) < 4:
        return 0.0
    try:
        hull = ConvexHull(vertices)
        return float(hull.volume)
    except Exception:
        # degenerate case (all points coplanar, etc.)
        return 0.0


def build_shape_decomp_dict(
    output_path: str,
    asu_dict: dict,
) -> None:
    """Build Wyckoff shape decomposition dict and persist to pickle."""
    output = Path(output_path)
    if output.exists():
        return

    output.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"[wyckoff_shape_decomp] building {output_path} ...")

    shape_decomposition_dict = {}

    for sg, sg_dict in asu_dict.items():
        sg_shape_info = {}

        for wyckoff_letter in sg_dict["ordered_wyckoff_letters"]:
            geometry = sg_dict[wyckoff_letter]
            site_dim = int(geometry["dim"])

            if site_dim == 0:
                wyckoff_shapes_info = {"dim": 0, "volumes": None}

            elif site_dim == 1:
                wyckoff_shapes_info = {"dim": 1, "volumes": []}
                for interval in geometry["vertices"]:
                    arr = _to_array(interval)  # (2, 3)
                    if arr.shape[0] >= 2:
                        diff = arr[1] - arr[0]  # (3,)
                        length = float(np.linalg.norm(diff))
                    else:
                        length = 0.0
                    wyckoff_shapes_info["volumes"].append(length)

            elif site_dim == 2:
                wyckoff_shapes_info = {
                    "dim": 2,
                    "volumes": [],
                    "facet_triangles": [],
                    "facet_triangle_areas": [],
                    "max_triangles_per_facet": 0,
                }
                max_tri = 0
                for facet in geometry["vertices"]:
                    arr = _to_array(facet)  # (n_v, 3)
                    triangles, areas = _fan_triangulate_convex_polygon_3d(arr)
                    total_area = float(areas.sum()) if len(areas) > 0 else 0.0
                    wyckoff_shapes_info["volumes"].append(total_area)
                    wyckoff_shapes_info["facet_triangles"].append(triangles)
                    wyckoff_shapes_info["facet_triangle_areas"].append(areas)
                    max_tri = max(max_tri, len(triangles))
                wyckoff_shapes_info["max_triangles_per_facet"] = max_tri

            elif site_dim == 3:
                wyckoff_shapes_info = {"dim": 3, "volumes": []}
                arr = _to_array(geometry["vertices"])  # (n_v, 3)
                vol = _compute_3d_convex_hull_volume(arr)
                wyckoff_shapes_info["volumes"].append(vol)

            else:
                raise ValueError(
                    f"Wyckoff site dimensionality must be in [0, 1, 2, 3], "
                    f"got: {site_dim}"
                )

            sg_shape_info[wyckoff_letter] = wyckoff_shapes_info

        shape_decomposition_dict[sg] = sg_shape_info

    with open(output, "wb") as f:
        pickle.dump(shape_decomposition_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

    logger.info(f"[wyckoff_shape_decomp] done -> {output}")

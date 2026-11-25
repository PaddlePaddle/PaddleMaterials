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

"""Build packed ASU samples and vocabulary-derived crystallographic resources."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import pickle
import tempfile
from fractions import Fraction
from functools import partial
from importlib.metadata import version
from pathlib import Path
from typing import Tuple

import numpy as np
import paddle
from p_tqdm import p_map
from pymatgen.core import Element
from pymatgen.symmetry.groups import SpaceGroup as PymatgenSpaceGroup
from scipy import __version__ as scipy_version
from scipy.spatial import ConvexHull

from ppmat.utils import download
from ppmat.utils import logger
from ppmat.vocab import build_vocab

# Empirical lattice-parameter bounds per dataset, taken from the upstream
# SGEquiDiff reference configuration. Each range encloses the observed
# min/max over the corresponding bundled train split (verified for
# mp_20: lengths [2.281, 132.382] A, angles [60.003, 134.969] deg;
# mpts_52: lengths [0.986, 189.494] A, angles [60.062, 134.898] deg).
# Used for normalization domains and sampling clamps.
lattice_parameter_ranges = {
    "mp_20": {
        "min_lattice_length": 2.0,
        "max_lattice_length": 133.0,
        "min_lattice_angle": 60.0,
        "max_lattice_angle": 135.0,
    },
    "mpts_52": {
        "min_lattice_length": 0.98,
        "max_lattice_length": 189.5,
        "min_lattice_angle": 60.0,
        "max_lattice_angle": 135.0,
    },
}
# Maximum number of atoms in the full unit cell for each dataset. Used to bound
# autoregressive Wyckoff/element generation and to size padded tensors.
max_atoms_per_dataset = {
    "mp_20": 20,
    "mpts_52": 52,
}

# Number of crystallographic space groups, indexed 1..230.
NUM_CRYSTALLOGRAPHIC_SPACE_GROUPS: int = 230

# Number of lattice parameters per crystal: 3 lengths + 3 angles.
NUM_LATTICE_PARAMS: int = 6

# Maximum length of ``ordered_wyckoff_letters`` observed across the 230
# bundled space-group entries (``clean_wyckoffs_in_asu_v6.json``); bound for
# tensor padding.
MAX_WYCKOFF_POSITIONS: int = 27

# Element-encoding table size. Not the periodic-table total (118); covers the
# elements in the supported ASU datasets (max Z=94) with margin. Index 0 is a
# placeholder. Used by the ASU data layout and the model embeddings.
ELEMENT_ENCODING_SIZE: int = 98

# Symbol table for 0-indexed atomic numbers in [0, ELEMENT_ENCODING_SIZE);
# built from pymatgen to avoid a hand-written table.
chemical_symbols = ["X"] + [
    Element.from_Z(atomic_number).symbol
    for atomic_number in range(1, ELEMENT_ENCODING_SIZE + 1)
]


VOCAB_NAME = "sgequidiff"


class EmbeddingTools:
    """Element / space-group / Wyckoff embedding tables.

    Construct explicitly via ``EmbeddingTools(vocab)`` or
    ``build_embedding_tools()``. No global singleton.
    """

    @paddle.no_grad()
    def __init__(self, vocab: dict):
        element_vectors = vocab["element"]["vectors"][: ELEMENT_ENCODING_SIZE + 1]
        if len(element_vectors) != ELEMENT_ENCODING_SIZE + 1:
            raise ValueError(
                "element vocabulary has "
                f"{len(element_vectors)} rows, expected "
                f"{ELEMENT_ENCODING_SIZE + 1} (index 0 placeholder + Z=1.."
                f"{ELEMENT_ENCODING_SIZE})"
            )
        self.element_embedding_length = len(element_vectors[0])
        self.element_embedding_tensor = paddle.to_tensor(
            element_vectors,
            dtype=paddle.float32,
        )

        space_group_vectors = vocab["space_group"]["vectors"]
        self.space_group_embedding_length = len(space_group_vectors[0])
        self.space_group_embedding_tensor = paddle.to_tensor(
            space_group_vectors,
            dtype=paddle.float32,
        )

        wyckoff_dict = vocab["wyckoff"]["vectors"]
        self.wyckoff_embedding_length = len(wyckoff_dict["1"]["a"])
        asu_sites = vocab["asu_sites"]["data"]

        wyckoff_emb_list = []
        for sg_num in range(1, NUM_CRYSTALLOGRAPHIC_SPACE_GROUPS + 1):
            wyckoff_dict_of_sg = wyckoff_dict[str(sg_num)]
            letters = list(wyckoff_dict_of_sg.keys())

            # Map letters to sort keys: lowercase "a".."z" -> 0..25 (ASCII 97
            # is "a"), uppercase "A".."Z" -> 26..51 (ASCII 65 is "A", ranked
            # after all lowercase letters). The sorted result must reproduce
            # the JSON order of ``ordered_wyckoff_letters`` used by the model
            # side; verified below per space group.
            wyckoff_ascii = [ord(letter) for letter in letters]
            wyckoff_idxs = [
                ai - 97 if ai >= 97 else ai - 65 + 26 for ai in wyckoff_ascii
            ]
            sorted_letters = [
                letter
                for letter, _ in sorted(
                    zip(letters, wyckoff_idxs), key=lambda pair: pair[1]
                )
            ]
            ordered_letters = list(asu_sites[str(sg_num)]["ordered_wyckoff_letters"])
            if sorted_letters != ordered_letters:
                raise ValueError(
                    f"space group {sg_num}: wyckoff vocabulary keys sorted to "
                    f"{sorted_letters} but asu_sites ordered_wyckoff_letters is "
                    f"{ordered_letters}; embeddings and site geometry would be "
                    "silently misaligned"
                )

            emb_array = paddle.to_tensor(
                [wyckoff_dict_of_sg[letter] for letter in sorted_letters],
                dtype=paddle.float32,
            )
            padding = paddle.zeros(
                [
                    MAX_WYCKOFF_POSITIONS - len(letters),
                    self.wyckoff_embedding_length,
                ]
            )
            wyckoff_emb_list.append(paddle.concat([emb_array, padding], axis=0))

        self.wyckoff_embedding_tensor = paddle.stack(wyckoff_emb_list, axis=0)

    @paddle.no_grad()
    def get_space_group_embedding(
        self, space_group_index: paddle.Tensor
    ) -> paddle.Tensor:
        """Get space group embedding."""
        if space_group_index.dtype != paddle.int64:
            raise TypeError(
                "space_group_index must be int64, got " f"{space_group_index.dtype}"
            )
        return self.space_group_embedding_tensor[space_group_index]

    @paddle.no_grad()
    def get_element_embedding(self, atomic_number: paddle.Tensor) -> paddle.Tensor:
        """Get element embedding."""
        if atomic_number.dtype != paddle.int64:
            raise TypeError(f"atomic_number must be int64, got {atomic_number.dtype}")
        return self.element_embedding_tensor[atomic_number]

    @paddle.no_grad()
    def get_wyckoff_embedding(
        self,
        wyckoff_index: paddle.Tensor,
        space_group_index: paddle.Tensor,
    ) -> paddle.Tensor:
        """Get Wyckoff embedding by index and space group.

        Callers must pass valid Wyckoff indices for the space group (the model
        derives them from ``valid_wyckoff_positions_mask``). Out-of-range
        indices read zero padding silently; the per-step validity check was
        removed from this hot path to avoid a GPU sync on every training step.
        """
        return self.wyckoff_embedding_tensor[space_group_index, wyckoff_index, :]


def build_embedding_tools(vocab: dict | None = None) -> EmbeddingTools:
    """Build EmbeddingTools from the registered vocabulary."""
    if vocab is None:
        vocab = build_vocab(VOCAB_NAME)
    return EmbeddingTools(vocab)


# Max number of geometric shapes a single Wyckoff site is decomposed into
# (0D point / 1D line / 2D facet); used to pad shape-related tensors.
_max_shapes_per_wyckoff = 4
# Max number of simplicial facets of an ASU convex hull; used to pad hull
# equation tensors to a fixed shape across all space groups.
_max_simplicial_hull_facets = 16
# Max number of symmetry operations of any space group (192 = 48 * 4).
MAX_GENERAL_WYCKOFF_OPS = 192

# Fixed random vector keeps 1D hull equations reproducible across runs.
_deterministic_random_vector = np.random.RandomState(0).rand(3)


def _load_wyckoff_sites_dict(wyckoffs_dict: dict) -> dict:
    """Normalize the Wyckoff site dictionary (fractions and tensors)."""
    to_fraction = np.vectorize(Fraction)
    for space_group_number in wyckoffs_dict.keys():
        for wyckoff_letter in wyckoffs_dict[space_group_number][
            "ordered_wyckoff_letters"
        ]:
            site = wyckoffs_dict[space_group_number][wyckoff_letter]
            dim = int(site["dim"])

            if dim != 2:
                vertices = to_fraction(site["vertices"])
                site["vertices"] = vertices
                site["vertices_tensor"] = paddle.to_tensor(vertices.astype("float32"))
            else:
                faces = []
                face_tensors = []
                for face in site["vertices"]:
                    face_array = to_fraction(face)  # (n_face_vertices, 3)
                    faces.append(face_array)
                    face_tensors.append(paddle.to_tensor(face_array.astype("float32")))
                site["vertices"] = faces
                # Participates in the derived-asset cache digest.
                site["vertices_tensors"] = face_tensors

            # JSON stores "dim" as a string; normalize in place.
            site["dim"] = dim

    return wyckoffs_dict


def _cache_json_value(value):
    if isinstance(value, Fraction):
        return str(value)
    return value.tolist()


class WyckoffGeometry:
    """All Wyckoff/ASU precomputed data, derived from the ASU sites vocabulary.

    Instances are owned by their callers (no global cache or lock); build one
    via ``build_wyckoff_geometry`` and pass it to downstream modules.
    """

    cache_version = 1

    def __init__(self, vocab: dict | None = None) -> None:
        if vocab is None:
            vocab = build_vocab(VOCAB_NAME)
        self.asu_wyckoff_dict = _load_wyckoff_sites_dict(
            copy.deepcopy(vocab["asu_sites"]["data"])
        )

        self._build_symmetry_ops()
        self._build_wyckoff_dimension_tensor()
        self._build_shape_projections()
        self._build_asu_hull_equations()
        self._build_wyckoff_shape_hull_equations()
        identity = {
            "version": self.cache_version,
            "sites": self.asu_wyckoff_dict,
            "numpy": np.__version__,
            "paddle": paddle.__version__,
            "pymatgen": version("pymatgen"),
            "scipy": scipy_version,
        }
        digest = hashlib.sha256(
            json.dumps(identity, sort_keys=True, default=_cache_json_value).encode()
        )
        digest.update(self.padded_general_wyckoff_matrices.numpy().tobytes())
        self.cache_key = digest.hexdigest()

    @staticmethod
    def _project_onto_1d_subspace(direction: paddle.Tensor) -> paddle.Tensor:
        """Orthogonal projection onto a line, returned as a (3,3) matrix."""
        return (direction.T @ direction) / (direction**2).sum()

    @staticmethod
    def _project_onto_2d_subspace(
        facet_vertices: paddle.Tensor,
    ) -> Tuple[paddle.Tensor, paddle.Tensor]:
        """Orthogonal projection onto a plane and its unit normal."""
        ab = facet_vertices[0] - facet_vertices[1]
        bc = facet_vertices[1] - facet_vertices[2]
        plane_normal = paddle.linalg.cross(ab, bc)

        v1 = paddle.linalg.cross(ab, plane_normal).reshape([1, 3])
        v2 = ab.reshape([1, 3])

        projection_matrix = (v1.T @ v1) / (v1**2).sum() + (v2.T @ v2) / (
            v2**2
        ).sum()
        return projection_matrix, plane_normal

    def _build_symmetry_ops(self) -> None:
        n_sgs = NUM_CRYSTALLOGRAPHIC_SPACE_GROUPS
        padded_matrices = paddle.zeros([n_sgs, MAX_GENERAL_WYCKOFF_OPS, 3, 3])
        padded_inverse_matrices = paddle.zeros([n_sgs, MAX_GENERAL_WYCKOFF_OPS, 3, 3])
        padded_translations = paddle.zeros([n_sgs, MAX_GENERAL_WYCKOFF_OPS, 1, 3])
        padded_ops_mask = paddle.zeros(
            [n_sgs, MAX_GENERAL_WYCKOFF_OPS], dtype=paddle.bool
        )

        for sg_num in range(1, n_sgs + 1):
            ops = list(PymatgenSpaceGroup.from_int_number(sg_num).symmetry_ops)

            identity_idx = next(
                (
                    i
                    for i, op in enumerate(ops)
                    if np.allclose(op.rotation_matrix, np.eye(3), atol=1e-6)
                    and np.allclose(op.translation_vector, np.zeros(3), atol=1e-6)
                ),
                0,
            )
            if identity_idx != 0:
                ops = [ops[identity_idx]] + [
                    ops[j] for j in range(len(ops)) if j != identity_idx
                ]

            rotations = []
            inverse_rotations = []
            translations = []
            for symmetry_rep in ops:
                rotation = paddle.to_tensor(
                    symmetry_rep.rotation_matrix, dtype=paddle.float32
                )
                rotations.append(rotation.T)
                inverse_rotations.append(paddle.linalg.inv(rotation).T)
                translations.append(
                    paddle.to_tensor(
                        symmetry_rep.translation_vector, dtype=paddle.float32
                    ).unsqueeze(0)
                )

            rotations = paddle.stack(rotations, axis=0)
            inverse_rotations = paddle.stack(inverse_rotations, axis=0)
            translations = paddle.stack(translations, axis=0)

            n_ops = rotations.shape[0]
            padded_matrices[sg_num - 1, :n_ops] = rotations
            padded_inverse_matrices[sg_num - 1, :n_ops] = inverse_rotations
            padded_translations[sg_num - 1, :n_ops] = translations
            padded_ops_mask[sg_num - 1, :n_ops] = True

            # First slot must be the identity operation.
            if not paddle.equal_all(padded_matrices[sg_num - 1, 0], paddle.eye(3)):
                raise ValueError(f"space group {sg_num}: first op is not identity")
            if not paddle.equal_all(
                padded_translations[sg_num - 1, 0], paddle.zeros([1, 3])
            ):
                raise ValueError(
                    f"space group {sg_num}: first op has nonzero translation"
                )
            if not bool(padded_ops_mask[sg_num - 1, 0]):
                raise ValueError(f"space group {sg_num}: first op mask not set")

        self.padded_general_wyckoff_matrices = padded_matrices
        self.padded_inverse_general_wyckoff_matrices = padded_inverse_matrices
        self.padded_general_wyckoff_translations = padded_translations
        self.padded_general_wyckoff_ops_mask = padded_ops_mask

    def _build_wyckoff_dimension_tensor(self) -> None:
        # -1 marks slots without a Wyckoff site of the space group.
        wyckoff_dimension_tensor = paddle.full(
            [NUM_CRYSTALLOGRAPHIC_SPACE_GROUPS, MAX_WYCKOFF_POSITIONS],
            fill_value=-1,
            dtype=paddle.int64,
        )

        for sg_num in range(1, NUM_CRYSTALLOGRAPHIC_SPACE_GROUPS + 1):
            sg_dict = self.asu_wyckoff_dict[str(sg_num)]
            for index, letter in enumerate(sg_dict["ordered_wyckoff_letters"]):
                wyckoff_dimension_tensor[sg_num - 1, index] = int(
                    sg_dict[letter]["dim"]
                )

        self.wyckoff_dimension_tensor = wyckoff_dimension_tensor

    def _build_shape_projections(self) -> None:
        n_sgs = NUM_CRYSTALLOGRAPHIC_SPACE_GROUPS
        shape = [n_sgs, MAX_WYCKOFF_POSITIONS, _max_shapes_per_wyckoff]
        noise_projection_matrices = paddle.zeros(shape + [3, 3])
        point_per_1d_wyckoff_line = paddle.zeros(shape + [3])
        line_directions_of_1d_wyckoffs = paddle.zeros(shape + [3])
        point_per_2d_wyckoff_plane = paddle.zeros(shape + [3])
        plane_normals_of_2d_wyckoffs = paddle.zeros(shape + [3])

        for sg_num in range(1, n_sgs + 1):
            sg_dict = self.asu_wyckoff_dict[str(sg_num)]
            projection_matrices = paddle.zeros([_max_shapes_per_wyckoff, 3, 3])
            for index, letter in enumerate(sg_dict["ordered_wyckoff_letters"]):
                site = sg_dict[letter]

                dim = int(site["dim"])
                if dim == 0:
                    projection_matrices[0] = paddle.zeros([3, 3])
                elif dim == 1:
                    for shape_idx, segment in enumerate(site["vertices"]):
                        segment = segment.astype("float32")
                        direction = paddle.to_tensor(
                            segment[1] - segment[0], dtype=paddle.float32
                        ).reshape([1, 3])
                        projection_matrices[shape_idx] = self._project_onto_1d_subspace(
                            direction
                        )
                        point_per_1d_wyckoff_line[
                            sg_num - 1, index, shape_idx
                        ] = paddle.to_tensor(segment[1], dtype=paddle.float32)
                        line_directions_of_1d_wyckoffs[
                            sg_num - 1, index, shape_idx
                        ] = direction.reshape([3])
                elif dim == 2:
                    for shape_idx, facet_vertices in enumerate(site["vertices"]):
                        facet_vertices = facet_vertices.astype("float32")
                        projection, plane_normal = self._project_onto_2d_subspace(
                            paddle.to_tensor(facet_vertices)
                        )
                        projection_matrices[shape_idx] = projection
                        point_per_2d_wyckoff_plane[
                            sg_num - 1, index, shape_idx
                        ] = paddle.to_tensor(facet_vertices[0])
                        plane_normals_of_2d_wyckoffs[
                            sg_num - 1, index, shape_idx
                        ] = plane_normal
                elif dim == 3:
                    projection_matrices[0] = paddle.eye(3)
                else:
                    raise ValueError(
                        f"space group {sg_num} site {letter}: "
                        f"unexpected Wyckoff dimension {dim}"
                    )
                noise_projection_matrices[sg_num - 1, index] = projection_matrices

        self.noise_projection_matrices = noise_projection_matrices
        self.point_per_1d_wyckoff_line = point_per_1d_wyckoff_line
        self.line_directions_of_1d_wyckoffs = line_directions_of_1d_wyckoffs
        self.point_per_2d_wyckoff_plane = point_per_2d_wyckoff_plane
        self.plane_normals_of_2d_wyckoffs = plane_normals_of_2d_wyckoffs

    def _build_asu_hull_equations(self) -> None:
        asu_hull_equations = paddle.full(
            [NUM_CRYSTALLOGRAPHIC_SPACE_GROUPS, _max_simplicial_hull_facets, 4],
            fill_value=0.0,
            dtype=paddle.float32,
        )
        asu_hull_equations[..., 3] = -1.0

        for sg in range(1, NUM_CRYSTALLOGRAPHIC_SPACE_GROUPS + 1):
            sg_dict = self.asu_wyckoff_dict[str(sg)]
            general_site = sg_dict[sg_dict["ordered_wyckoff_letters"][-1]]

            if int(general_site["dim"]) != 3:
                raise ValueError(f"space group {sg}: general site is not 3D")
            vertices = general_site["vertices"].astype("float64")

            hull = ConvexHull(vertices)
            equations = hull.equations
            n_facets = equations.shape[0]
            if n_facets > _max_simplicial_hull_facets:
                raise ValueError(
                    f"space group {sg}: ASU hull has {n_facets} facets, "
                    f"exceeding the padding budget "
                    f"{_max_simplicial_hull_facets}; hull constraints would "
                    "be silently truncated"
                )
            asu_hull_equations[sg - 1, :n_facets] = paddle.to_tensor(
                equations, dtype=paddle.float32
            )

        self.asu_hull_equations = asu_hull_equations

    def _build_wyckoff_shape_hull_equations(self) -> None:
        """Hull equations for each 0/1/2/3D Wyckoff shape (bounded shapes included)."""
        # The hull-facet budget covers every Wyckoff shape's bound count
        # observed in the bundled 230 space-group entries.
        max_num_shape_bounds = _max_simplicial_hull_facets

        padded_hull_equations = paddle.full(
            [
                NUM_CRYSTALLOGRAPHIC_SPACE_GROUPS,
                MAX_WYCKOFF_POSITIONS,
                _max_shapes_per_wyckoff,
                max_num_shape_bounds,
                4,
            ],
            fill_value=0.0,
            dtype=paddle.float32,
        )
        padded_hull_equations[..., 3] = -1.0

        mask_padded_hull_equations = paddle.zeros(
            [
                NUM_CRYSTALLOGRAPHIC_SPACE_GROUPS,
                MAX_WYCKOFF_POSITIONS,
                _max_shapes_per_wyckoff,
            ],
            dtype=paddle.bool,
        )

        for sg_num in range(1, NUM_CRYSTALLOGRAPHIC_SPACE_GROUPS + 1):
            sg_dict = self.asu_wyckoff_dict[str(sg_num)]
            for wp_idx, letter in enumerate(sg_dict["ordered_wyckoff_letters"]):
                site = sg_dict[letter]
                dim = int(site["dim"])

                if dim == 0:
                    shape_idx = 0
                    mask_padded_hull_equations[sg_num - 1, wp_idx, shape_idx] = True
                    vertex = site["vertices"].astype("float64").reshape(-1)
                    eps = 1e-4
                    hull_equations = np.zeros((6, 4))
                    hull_equations[:3, :3] = np.eye(3)
                    hull_equations[3:, :3] = -np.eye(3)
                    hull_equations[:3, -1] = -(vertex + eps)
                    hull_equations[3:, -1] = -(-vertex + eps)
                    padded_hull_equations[
                        sg_num - 1, wp_idx, shape_idx, :6
                    ] = paddle.to_tensor(hull_equations, dtype=paddle.float32)

                elif dim == 1:
                    for shape_idx, segment in enumerate(
                        site["vertices"].astype("float64")
                    ):
                        mask_padded_hull_equations[sg_num - 1, wp_idx, shape_idx] = True
                        eps = 1e-4
                        line_dir = segment[1] - segment[0]
                        line_dir /= np.linalg.norm(line_dir)
                        # Fixed random vector keeps hull equations reproducible.
                        normal1 = np.cross(line_dir, _deterministic_random_vector)[
                            np.newaxis, :
                        ]
                        normal1 /= np.linalg.norm(normal1)
                        normal2 = np.cross(line_dir, normal1)
                        normal2 /= np.linalg.norm(normal2)
                        p1 = eps * normal1
                        p2 = eps * normal2
                        bounding_polytope = np.concatenate(
                            [
                                segment + (p1 + p2),
                                segment + (p1 - p2),
                                segment + (-p1 + p2),
                                segment + (-p1 - p2),
                            ],
                            axis=0,
                        )
                        hull = ConvexHull(bounding_polytope)
                        n_facets = hull.equations.shape[0]
                        padded_hull_equations[
                            sg_num - 1, wp_idx, shape_idx, :n_facets
                        ] = paddle.to_tensor(hull.equations, dtype=paddle.float32)

                elif dim == 2:
                    for shape_idx, polygon_vertices in enumerate(site["vertices"]):
                        mask_padded_hull_equations[sg_num - 1, wp_idx, shape_idx] = True
                        polygon_vertices = polygon_vertices.astype("float64")
                        ab = polygon_vertices[1] - polygon_vertices[0]
                        ac = polygon_vertices[2] - polygon_vertices[0]
                        normal = np.cross(ab, ac)
                        normal /= np.linalg.norm(normal)
                        bounding = np.concatenate(
                            [
                                polygon_vertices + 1e-4 * normal,
                                polygon_vertices - 1e-4 * normal,
                            ],
                            axis=0,
                        )
                        hull = ConvexHull(bounding)
                        n_facets = hull.equations.shape[0]
                        padded_hull_equations[
                            sg_num - 1, wp_idx, shape_idx, :n_facets
                        ] = paddle.to_tensor(hull.equations, dtype=paddle.float32)

                elif dim == 3:
                    shape_idx = 0
                    mask_padded_hull_equations[sg_num - 1, wp_idx, shape_idx] = True
                    hull = ConvexHull(site["vertices"].astype("float64"))
                    n_facets = hull.equations.shape[0]
                    padded_hull_equations[
                        sg_num - 1, wp_idx, shape_idx, :n_facets
                    ] = paddle.to_tensor(hull.equations, dtype=paddle.float32)

        self.padded_hull_equations = padded_hull_equations
        self.padded_hull_equations_mask = mask_padded_hull_equations


def build_wyckoff_geometry(vocab: dict | None = None) -> WyckoffGeometry:
    """Build a WyckoffGeometry instance owned by the caller (no global cache)."""
    return WyckoffGeometry(vocab)


def get_data_directory() -> Path:
    """Writable cache directory for SGEquiDiff derived assets.

    Resolved under ``~/.paddlemat/datasets`` (the unified dataset cache
    root) and created on demand.
    """
    directory = Path(download.DATASETS_HOME) / "sgequidiff"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def save_asu_cache(path, data):
    """Atomically publish a complete derived ASU cache."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as file_obj:
            temporary_path = file_obj.name
            pickle.dump(data, file_obj, protocol=pickle.HIGHEST_PROTOCOL)
            file_obj.flush()
            os.fsync(file_obj.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and os.path.exists(temporary_path):
            os.unlink(temporary_path)


def load_wyckoff_shapes(geometry):
    """Build or load the shape decomposition of the supplied geometry."""
    path = get_data_directory() / f"wyckoff_shapes-{geometry.cache_key}.pkl"
    if path.is_file():
        try:
            with path.open("rb") as file_obj:
                return pickle.load(file_obj)
        except (EOFError, OSError, pickle.UnpicklingError):
            logger.warning(f"Rebuilding invalid Wyckoff shape cache: {path}")
    return build_shape_decomp_dict(str(path), geometry.asu_wyckoff_dict)


def _to_array(region) -> np.ndarray:
    """Convert Wyckoff vertices to a float64 array."""
    return np.asarray(region, dtype=np.float64)


def _fan_triangulate_convex_polygon_3d(vertices_3d: np.ndarray):
    """Fan triangulation of convex polygon in 3D (from centroid)."""
    n = len(vertices_3d)
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
    return float(ConvexHull(vertices).volume)


def build_shape_decomp_dict(
    output_path: str,
    asu_dict: dict,
) -> dict:
    """Build and atomically cache Wyckoff shape decomposition metadata."""
    output = Path(output_path)

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
                    diff = arr[1] - arr[0]  # (3,)
                    length = float(np.linalg.norm(diff))
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
                    wyckoff_shapes_info["volumes"].append(float(areas.sum()))
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

    save_asu_cache(output, shape_decomposition_dict)
    logger.info(f"[wyckoff_shape_decomp] done -> {output}")
    return shape_decomposition_dict


class BuildAsuCrystal:
    """Build one ASU crystal field dict from a packed flat array.

    The packed layout (NE = element encoding size, n = num atoms) is defined
    in the ``AsymmetricUnitDataset`` docstring. Archives without the optional
    trailing wyckoff-shape segment are normalized with zero shape indices
    (no shape decomposition). Fields whose order depends on the atom order
    are stably sorted by (wyckoff index, element index), so the built dict is
    the final per-sample representation.

    Args:
        element_encoding_size (int): Size of the element encoding NE.
        num_cpus (int, optional): Number of CPUs used when building a list
            of arrays. Defaults to 1.
    """

    cache_version = 1

    def __init__(self, element_encoding_size: int, num_cpus: int = 1):
        self.num_cpus = num_cpus

        # Packed-layout offsets that depend on the element encoding size.
        self._idx_lengths = 2 + element_encoding_size
        self._idx_angles = 5 + element_encoding_size
        self._idx_atoms = 8 + element_encoding_size

    @staticmethod
    def build_one(
        flat: np.ndarray,
        idx_lengths: int,
        idx_angles: int,
        idx_atoms: int,
    ) -> dict:
        """Build one ASU crystal field dict from one packed flat array.

        ``np.lexsort`` takes the keys in reverse order, so the last key of
        the tuple is the primary one.
        """
        n = int(flat[0])
        if len(flat) > idx_atoms + 5 * n:
            shape_indices = flat[idx_atoms + 5 * n : idx_atoms + 6 * n].astype(np.int64)
        else:
            shape_indices = np.zeros(n, dtype=np.int64)
        element_indices = flat[idx_atoms : idx_atoms + n].astype(np.int64)
        wyckoff_indices = flat[idx_atoms + n : idx_atoms + 2 * n].astype(np.int64)
        frac_coords = (
            flat[idx_atoms + 2 * n : idx_atoms + 5 * n].reshape(n, 3).astype(np.float32)
        )
        order = np.lexsort((element_indices, wyckoff_indices))
        return {
            "n_atoms_per_asu": n,
            "space_group_indices": int(flat[1]) - 1,
            "batch_chemistries": flat[2:idx_lengths].astype(np.float32),
            "lattice_lengths": flat[idx_lengths:idx_angles].astype(np.float32),
            "lattice_angles": flat[idx_angles:idx_atoms].astype(np.float32),
            "element_indices": element_indices[order],
            "wyckoff_indices": wyckoff_indices[order],
            "wyckoff_shape_indices": shape_indices[order],
            "frac_coords": frac_coords[order],
        }

    def __call__(self, flat_crystals):
        """Build ASU crystal field dicts from packed flat array(s)."""
        build_one = partial(
            self.build_one,
            idx_lengths=self._idx_lengths,
            idx_angles=self._idx_angles,
            idx_atoms=self._idx_atoms,
        )
        if isinstance(flat_crystals, list):
            if self.num_cpus == 1:
                return list(map(build_one, flat_crystals))
            return p_map(build_one, flat_crystals, num_cpus=self.num_cpus)
        return build_one(flat_crystals)

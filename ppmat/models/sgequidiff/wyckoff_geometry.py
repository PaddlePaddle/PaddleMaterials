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

"""Wyckoff site / space group precomputed data for the asymmetric unit (ASU).

``WyckoffGeometry`` loads the Wyckoff site JSON payload (upstream
``clean_wyckoffs_in_asu_v6.json``) from the registered ``sgequidiff``
vocabulary (``vocab["asu_sites"]["data"]``, see ``ppmat.models.sgequidiff.vocabs``)
and derives all tensors from it. Construct explicitly via ``WyckoffGeometry()`` or
``build_wyckoff_geometry()``; pass the instance to downstream modules.
"""

from fractions import Fraction
from typing import Dict
from typing import List
from typing import Tuple

import numpy as np
import paddle
from pymatgen.symmetry.groups import SpaceGroup as PymatgenSpaceGroup
from scipy.spatial import ConvexHull

from ppmat.models.sgequidiff.sgequidiff_meta import MAX_WYCKOFF_POSITIONS
from ppmat.models.sgequidiff.sgequidiff_meta import NUM_CRYSTALLOGRAPHIC_SPACE_GROUPS
from ppmat.models.sgequidiff.sgequidiff_meta import spgroup_data
from ppmat.models.sgequidiff.vocabs import VOCAB_NAME
from ppmat.vocab import build_vocab

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

_identity = paddle.eye(3)


def _transform_pair(
    p: List[List[float]], inv_p: List[List[float]], scale: float = 1.0
) -> Dict[str, paddle.Tensor]:
    """Conventional -> primitive transform pair: ``P`` maps conventional
    fractional coordinates onto the primitive cell, ``invP`` is its inverse."""
    return {
        "P": scale * paddle.to_tensor(p, dtype=paddle.float32),
        "invP": paddle.to_tensor(inv_p, dtype=paddle.float32),
    }


# Face-centered (F) and body-centered (I) lattices need real transforms; every
# other lattice (cP/tP/hP/oP/mP/aP) is already primitive and uses identity.
_face_centered = _transform_pair(
    [[-0.5, -0.5, 0.0], [-0.5, 0.0, -0.5], [0.0, -0.5, -0.5]],
    [[-1.0, -1.0, 1.0], [-1.0, 1.0, -1.0], [1.0, -1.0, -1.0]],
)
_body_centered = _transform_pair(
    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-0.5, -0.5, 0.5]],
    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 2.0]],
)

conventional_to_primitive_transforms: dict = {
    "cP": {"P": _identity, "invP": _identity},
    "tP": {"P": _identity, "invP": _identity},
    "hP": {"P": _identity, "invP": _identity},
    "oP": {"P": _identity, "invP": _identity},
    "mP": {"P": _identity, "invP": _identity},
    "aP": {"P": _identity, "invP": _identity},
    "cF": _face_centered,
    "oF": _face_centered,
    "cI": _body_centered,
    "tI": _body_centered,
    "oI": _body_centered,
    "hR": _transform_pair(
        [[-3.0, -3.0, 0.0], [-3.0, 0.0, 0.0], [-2.0, -1.0, -1.0]],
        [[0.0, -1.0, 0.0], [-1.0, 1.0, 0.0], [1.0, 1.0, -3.0]],
        scale=1.0 / 3.0,
    ),
    "oC": _transform_pair(
        [[-0.5, -0.5, 0.0], [-0.5, 0.5, 0.0], [0.0, 0.0, -1.0]],
        [[-1.0, -1.0, 0.0], [-1.0, 1.0, 0.0], [0.0, 0.0, -1.0]],
    ),
    "oA": _transform_pair(
        [[0.0, -0.5, -0.5], [-1.0, 0.0, 0.0], [0.0, 0.5, -0.5]],
        [[0.0, -1.0, 0.0], [-1.0, 0.0, 1.0], [-1.0, 0.0, -1.0]],
    ),
    "mC": _transform_pair(
        [[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.5, -0.5, 0.0]],
        [[0.0, -1.0, 0.0], [0.0, -1.0, -2.0], [1.0, 0.0, 0.0]],
    ),
}


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
                site["vertices_tensors"] = face_tensors

            # JSON stores "dim" as a string; normalize in place.
            site["dim"] = dim

    return wyckoffs_dict


class WyckoffGeometry:
    """All Wyckoff/ASU precomputed data, derived from the ASU sites vocabulary.

    Instances are owned by their callers (no global cache or lock); build one
    via ``build_wyckoff_geometry`` and pass it to downstream modules.
    """

    def __init__(self, vocab: dict | None = None) -> None:
        if vocab is None:
            vocab = build_vocab(VOCAB_NAME)
        self.asu_wyckoff_dict = _load_wyckoff_sites_dict(vocab["asu_sites"]["data"])

        self._build_symmetry_ops()
        self._build_conventional_to_primitive_matrices()
        self._build_wyckoff_dimension_tensor()
        self._build_shape_projections()
        self._build_asu_hull_equations()
        self._build_wyckoff_shape_hull_equations()

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

    def _build_conventional_to_primitive_matrices(self) -> None:
        p_matrices = []
        inv_p_matrices = []
        for space_group_number in range(1, NUM_CRYSTALLOGRAPHIC_SPACE_GROUPS + 1):
            bravais = spgroup_data[space_group_number]
            p_matrices.append(conventional_to_primitive_transforms[bravais]["P"])
            inv_p_matrices.append(conventional_to_primitive_transforms[bravais]["invP"])
        self.conventional_to_primitive_P_matrices = paddle.stack(p_matrices, axis=0)
        self.conventional_to_primitive_invP_matrices = paddle.stack(
            inv_p_matrices, axis=0
        )

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

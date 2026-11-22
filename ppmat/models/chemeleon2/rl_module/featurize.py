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

import numpy as np
import paddle

from ppmat.models.chemeleon2.common.utils import build_structure_array
from ppmat.utils.scatter import scatter_mean
from ppmat.utils.scatter import scatter_sum


def structures_to_structure_array(structures):
    """Convert pymatgen structures to the array dict consumed by the VAE.

    Values follow the MP20Dataset convention: ``atom_types`` holds atomic
    numbers ``Z`` and lattice angles are in degrees.
    """

    atom_types = []
    frac_coords = []
    lengths = []
    angles = []
    num_atoms = []
    for structure in structures:
        atom_types.append(np.array([site.specie.Z for site in structure]))
        frac_coords.append(np.asarray(structure.frac_coords))
        lengths.append(np.asarray(structure.lattice.abc))
        angles.append(np.asarray(structure.lattice.angles))
        num_atoms.append(len(structure))

    return {
        "atom_types": paddle.to_tensor(np.concatenate(atom_types), dtype="int64"),
        "frac_coords": paddle.to_tensor(np.concatenate(frac_coords), dtype="float32"),
        "lengths": paddle.to_tensor(np.stack(lengths), dtype="float32"),
        "angles": paddle.to_tensor(np.stack(angles), dtype="float32"),
        "num_atoms": paddle.to_tensor(num_atoms, dtype="int64"),
    }


def featurize(
    structures,
    vae,
    batch_size=2000,
    reduce="mean",
    use_encoder_features=False,
):
    """Featurize structures into structure/composition embeddings with the VAE.

    Mirrors the upstream Chemeleon2 featurizer: structure features are the
    posterior mean latent vectors reduced per graph, composition features are
    the per-graph reduced atom-type embeddings. Everything runs in Paddle.

    Args:
        structures: list of pymatgen ``Structure`` objects.
        vae: a ``Chemeleon2VAEModule`` (frozen) used as the encoder.
        batch_size: number of structures encoded per forward pass.
        reduce: "mean" or "sum" reduction from atom to structure features.
        use_encoder_features: append the encoder hidden states to the latent.

    Returns:
        dict with ``structure_features`` and ``composition_features`` as
        float32 tensors of shape ``(num_structures, dim)``.
    """

    if len(structures) == 0:
        raise ValueError("featurize requires at least one structure.")
    reduce_fn = scatter_mean if reduce == "mean" else scatter_sum
    batch_size = min(batch_size, len(structures))
    vae.eval()

    structure_features = []
    composition_features = []
    with paddle.no_grad():
        for start in range(0, len(structures), batch_size):
            chunk = structures[start : start + batch_size]
            batch = build_structure_array(structures_to_structure_array(chunk))
            encoded = vae.encode(batch)
            latent_vector = encoded["posterior"].mode()
            if use_encoder_features:
                latent_vector = paddle.concat([latent_vector, encoded["x"]], axis=-1)
            composition_vector = vae.encoder.atom_type_embedder(batch["atom_types"])

            structure_features.append(reduce_fn(latent_vector, batch["batch"], dim=0))
            composition_features.append(
                reduce_fn(composition_vector, batch["batch"], dim=0)
            )

    return {
        "structure_features": paddle.concat(structure_features, axis=0),
        "composition_features": paddle.concat(composition_features, axis=0),
    }

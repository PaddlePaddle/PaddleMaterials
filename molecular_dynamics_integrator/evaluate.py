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

import argparse
import json
from pathlib import Path

import numpy as np


def calculate_msd(
    traj: np.ndarray,
    atom_mask: np.ndarray | None = None,
    final_only: bool = True,
) -> np.ndarray | float:
    traj = np.asarray(traj, dtype=np.float64)
    squared_displacements = np.sum((traj - traj[0]) ** 2, axis=-1)
    if atom_mask is not None:
        squared_displacements = squared_displacements[:, atom_mask]
    if final_only:
        return np.mean(squared_displacements[-1])
    return np.mean(squared_displacements, axis=-1)


def calculate_average_rdf(
    traj: np.ndarray,
    lattice: np.ndarray,
    rmax: float = 5.0,
    nbins: int = 50,
) -> tuple[np.ndarray, np.ndarray]:
    from ase import Atoms
    from ase import neighborlist

    traj = np.asarray(traj, dtype=np.float64)
    lattice = np.asarray(lattice, dtype=np.float64)
    n_atoms = traj.shape[1]
    vol = abs(np.linalg.det(lattice))
    r_bins = np.linspace(0, rmax, nbins + 1)
    r_centers = (r_bins[:-1] + r_bins[1:]) / 2
    dr = r_bins[1] - r_bins[0]
    den = 4 * np.pi * r_centers**2 * dr * (n_atoms / vol) * n_atoms

    rdf_list = []
    for positions in traj:
        atoms = Atoms(
            numbers=np.ones(n_atoms, dtype=np.int64),
            positions=positions,
            cell=lattice,
            pbc=True,
        )
        src, dst, image = neighborlist.neighbor_list(
            "ijS", atoms, rmax, self_interaction=False
        )
        shifts = image @ lattice
        vectors = positions[dst] - positions[src] + shifts
        distances = np.linalg.norm(vectors, axis=-1)
        hist, _ = np.histogram(distances, bins=r_bins)
        rdf_list.append(hist / den)
    rdf = np.mean(rdf_list, axis=0)
    return r_centers, rdf


def _structure_from_npz(path: str):
    data = np.load(path)
    return (
        np.asarray(data["atomic_numbers"], dtype=np.int64),
        np.asarray(data["lattice"], dtype=np.float64),
    )


def evaluate_trajectory(
    prediction: np.ndarray,
    reference: np.ndarray,
    atomic_numbers: np.ndarray,
    lattice: np.ndarray,
    rmax: float = 5.0,
    nbins: int = 50,
) -> dict:
    atomic_numbers = np.asarray(atomic_numbers, dtype=np.int64)
    li_mask = atomic_numbers == 3
    frame_mask = atomic_numbers != 3
    prediction = np.asarray(prediction, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)

    msd_li = calculate_msd(prediction, li_mask)
    msd_li_ref = calculate_msd(reference, li_mask)
    msd_frame = calculate_msd(prediction, frame_mask)
    msd_frame_ref = calculate_msd(reference, frame_mask)
    final_step = int(prediction.shape[0] - 1)

    if prediction.shape[0] <= 5:
        traj_rdf = np.array(prediction[-1])[None]
    else:
        traj_rdf = prediction[5:]
    traj_ref = reference[500::100]
    _, rdf = calculate_average_rdf(traj_rdf, lattice, rmax=rmax, nbins=nbins)
    _, rdf_ref = calculate_average_rdf(traj_ref, lattice, rmax=rmax, nbins=nbins)
    rdf_mae = float(np.mean(np.abs(rdf - rdf_ref)))

    return {
        "msd_Li": float(msd_li),
        "msd_Li_ref": float(msd_li_ref),
        "msd_frame": float(msd_frame),
        "msd_frame_ref": float(msd_frame_ref),
        "rdf_mae": rdf_mae,
        "final_step": final_step,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction", type=str, required=True)
    parser.add_argument("--reference", type=str, required=True)
    parser.add_argument("--structure", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    atomic_numbers, lattice = _structure_from_npz(args.structure)
    prediction = np.load(args.prediction)
    reference = np.load(args.reference)
    metrics = evaluate_trajectory(prediction, reference, atomic_numbers, lattice)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

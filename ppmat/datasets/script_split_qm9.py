#!/usr/bin/env python3
"""Export the official QM9 archive into train, validation, and test CSVs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


TARGETS = (
    "mu",
    "alpha",
    "homo",
    "lumo",
    "gap",
    "r2",
    "zpve",
    "U0",
    "U",
    "H",
    "G",
    "Cv",
)
ELEMENTS = {1: "H", 6: "C", 7: "N", 8: "O", 9: "F"}
SPLIT_SIZES = {"train": 110_000, "val": 10_000}


def xyz_block(archive, offsets, index):
    start, end = offsets[index : index + 2]
    molecule_id = int(archive["id"][index]) + 1
    lines = [str(end - start), f"gdb {molecule_id}"]
    for atomic_number, position in zip(
        archive["Z"][start:end], archive["R"][start:end]
    ):
        coordinates = "\t".join(f"{float(value):.10f}" for value in position)
        lines.append(f"{ELEMENTS[int(atomic_number)]}\t{coordinates}")
    return "\n".join(lines), molecule_id


def split_indices(num_samples, seed):
    indices = np.random.RandomState(seed).permutation(num_samples)
    train_end = SPLIT_SIZES["train"]
    val_end = train_end + SPLIT_SIZES["val"]
    return {
        "train": indices[:train_end],
        "val": indices[train_end:val_end],
        "test": indices[val_end:],
    }


def export(input_path: Path, output_dir: Path, seed: int):
    output_dir.mkdir(parents=True, exist_ok=True)
    with np.load(input_path, allow_pickle=True) as packed:
        archive = {key: packed[key] for key in packed.files}

    missing_keys = {"N", "Z", "R", "id", *TARGETS} - archive.keys()
    if missing_keys:
        raise KeyError(f"Missing QM9 archive fields: {sorted(missing_keys)}")

    num_samples = len(archive["N"])
    offsets = np.concatenate(
        [
            np.zeros(1, dtype=np.int64),
            np.cumsum(archive["N"], dtype=np.int64),
        ]
    )
    counts = {}
    for split, indices in split_indices(num_samples, seed).items():
        with (output_dir / f"{split}.csv").open(
            "w", newline="", encoding="utf-8"
        ) as stream:
            writer = csv.writer(stream)
            writer.writerow(["standard_xyz", "molecule_id", *TARGETS])
            for index in indices:
                molecule, molecule_id = xyz_block(archive, offsets, index)
                writer.writerow(
                    [
                        molecule,
                        molecule_id,
                        *(float(archive[target][index]) for target in TARGETS),
                    ]
                )
        counts[split] = len(indices)
    return counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", type=Path, default=Path("./data/qm9/qm9_eV.npz")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("./data/qm9")
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(export(args.input, args.output_dir, args.seed))


if __name__ == "__main__":
    main()

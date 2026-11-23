"""Check converted SevenNet-0 predictions against upstream reference outputs."""

import argparse
from pathlib import Path

import numpy as np
from ase.io import read
from pymatgen.io.ase import AseAtomsAdaptor

from ppmat.predictor import PotentialPredictor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--reference-dir", required=True, type=Path)
    parser.add_argument("--device", default="gpu:0")
    args = parser.parse_args()
    predictor = PotentialPredictor(
        config_path=str(args.model_dir / "sevennet0.yaml"),
        checkpoint_path=str(args.model_dir / "checkpoints/latest.pdparams"),
        device=args.device,
    )
    tolerances = {"energy": 1e-4, "force": 1e-5, "stress": 1e-6}
    maximum = dict.fromkeys(tolerances, 0.0)
    structures = sorted(args.reference_dir.glob("*.extxyz"))
    if len(structures) != 8:
        parser.error("the reference directory must contain eight .extxyz files")
    for path in structures:
        structure = AseAtomsAdaptor.get_structure(read(path))
        predictions = predictor.from_structures(structure)
        with np.load(path.with_name(path.stem + "_torch.npz")) as reference:
            for name in tolerances:
                prediction = np.asarray(predictions[name])
                if not np.isfinite(prediction).all():
                    raise ValueError(f"{path.name}: {name} contains nonfinite values")
                target = reference[name].reshape(prediction.shape)
                maximum[name] = max(maximum[name], float(np.max(np.abs(prediction - target))))
    print(f"{len(structures)} structures. Maximum absolute errors: {maximum}")
    for name, tolerance in tolerances.items():
        if maximum[name] >= tolerance:
            raise AssertionError(f"{name}: {maximum[name]} >= {tolerance}")


if __name__ == "__main__":
    main()

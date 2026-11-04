# LiFlow example data

`initial_structure.npz` is a single-frame, future-free initial structure used
to exercise the checkpointed trajectory inference
(`molecular_dynamics_integrator/predict.py`) and the scientific evaluation
(`molecular_dynamics_integrator/evaluate.py`) without needing the full dataset.

## Contents

The archive holds four top-level arrays (no ground-truth future frames, no
labels):

| Key | Dtype | Shape | Meaning |
| --- | --- | --- | --- |
| `positions` | `float32` | `(4, 3)` | Cartesian positions of the four atoms in Angstrom. |
| `atomic_numbers` | `int64` | `(4,)` | Atomic numbers; `3` = Li, `8` = O (Li<sub>2</sub>O<sub>2</sub>). |
| `lattice` | `float32` | `(3, 3)` | Orthorhombic cell matrix, 8.0 Angstrom on each side. |
| `element_index` | `int64` | `(18,)` | `Z -> element-embedding index` table used by LiFlow; mirrors `test/fixtures/liflow/dataset_mini/element_index.npy` (`index = Z - 1`). |

## Source and license

`positions`, `atomic_numbers` and `lattice` are identical to the tracked test
fixture `test/fixtures/liflow/initial_structure.npz` (Li<sub>2</sub>O<sub>2</sub>-type
universal electrolyte structure). `element_index` is copied from
`test/fixtures/liflow/dataset_mini/element_index.npy` so the archive can be
passed to `predict.py` directly. It is redistributed here as a minimal sample
under the same repository license (Apache-2.0). This is deliberately *not* the
full Li<sub>2</sub>O<sub>2</sub> trajectory dataset, which must be downloaded
separately (see `molecular_dynamics_integrator/liflow/README.md`, section
*Data sources*).

`element_index` must match the mapping used when the checkpoint was trained.
For production runs, always prefer the `element_index.npy` shipped with the
training data over this 18-entry example table.

## Usage

Pass the archive to the inference entrypoint:

```bash
python molecular_dynamics_integrator/predict.py \
    --config molecular_dynamics_integrator/liflow/configs/liflow_universal_inference.yaml \
    --input molecular_dynamics_integrator/liflow/example_data/initial_structure.npz \
    --output_dir output/liflow_prediction
```

and to the evaluator:

```bash
python molecular_dynamics_integrator/evaluate.py \
    --prediction output/liflow_prediction/trajectory.npy \
    --reference test/fixtures/liflow/reference_mini.npy \
    --structure molecular_dynamics_integrator/liflow/example_data/initial_structure.npz \
    --output output/liflow_prediction/metrics.json
```

The evaluator reads `atomic_numbers` (to separate Li from the frame) and
`lattice` (for the periodic radial distribution function).

# LiFlow

[LiFlow](https://github.com/learningmatter-mit/liflow) — flow matching for
accelerated simulation of atomic transport.

This directory integrates the original PyTorch LiFlow implementation into
PaddleMaterials: the dual message-passing network, the trajectory dataset, the
flow-matching objective, the propagator/corrector training recipes, the
future-free trajectory integrator, and the MSD/RDF evaluation.

Reference implementation: `learningmatter-mit/liflow`, pinned commit
[`e6fc475361d046865f12cae1aee11c4f56c48d87`](https://github.com/learningmatter-mit/liflow/tree/e6fc475361d046865f12cae1aee11c4f56c48d87)
(MIT License, see [License](#11-license)).

## Contents

| # | Section |
| --- | --- |
| 1 | [Introduction](#1-introduction) |
| 2 | [Environment](#2-environment) |
| 3 | [Dataset](#3-dataset) |
| 4 | [Models](#4-models) |
| 5 | [Checkpoints](#5-checkpoints) |
| 6 | [Training](#6-training) |
| 7 | [Validation and testing](#7-validation-and-testing) |
| 8 | [Prediction](#8-prediction) |
| 9 | [MSD / RDF evaluation](#9-msd--rdf-evaluation) |
| 10 | [Reproduction status](#10-reproduction-status) |
| 11 | [License](#11-license) |
| 12 | [Citation](#12-citation) |

---

## 1. Introduction

LiFlow models molecular dynamics as a continuous normalizing flow. Two
`DualPaiNN` (equivariant message-passing) networks are trained with the same
flow-matching objective but on different conditional distributions:

- **Propagator** — learns the velocity field that transports a *time-delayed*
  start frame `x_start` plus an adaptive Maxwell-Boltzmann prior sample to a
  later end frame `x_end` (`TimeDelayedPairDataset` in the original code).
- **Corrector** — refines a *single* frame: the input is a noised copy of the
  frame (`NoisedFrameDataset`), the neighbour list is built from the noisy
  positions, and the target is the noise-free frame.

Both share the original `FlowModule` target:

```
x_0       = x_start + prior
x_t       = (1 - t) * x_0 + t * x_end        # t ~ U(0, 1) during training
velocity  = x_end - x_0                      # prediction_mode: velocity
loss      = mean( sum( (velocity_pred - velocity)^2, axis=-1 ) )
```

Trajectory generation lives outside the models in
[`IntegratorPredictor`](../../ppmat/predictor/integrator_predictor.py). It only
sees the initial structure, the lattice, the temperature, the element table and
a seed — never future frames — and integrates the propagator velocity field with
Euler or Heun steps, optionally applying the corrector every `corrector_every`
steps.

```
initial_structure.npz
   positions / atomic_numbers / lattice / element_index
                     |
                     v
IntegratorPredictor -- propagator  (velocity field, Euler/Heun)
                     |-- corrector  (optional, every N steps)
                     v
trajectory.npy   trajectory.xyz   run_metadata.json
```

Source layout:

| Path | Role |
| --- | --- |
| [`ppmat/models/liflow/dual_painn.py`](../../ppmat/models/liflow/dual_painn.py) | `DualPaiNN` backbone (radial basis, message passing, readout) |
| [`ppmat/models/liflow/liflow.py`](../../ppmat/models/liflow/liflow.py) | `LiFlow` PaddleMaterials model: `forward` / `predict` / flow-matching loss |
| [`ppmat/models/liflow/geometry.py`](../../ppmat/models/liflow/geometry.py) | Periodic neighbour list and trajectory unwrapping |
| [`ppmat/models/liflow/prior.py`](../../ppmat/models/liflow/prior.py) | `AdaptiveMaxwellBoltzmannPrior`, `MaxwellBoltzmannPrior`, `UniformScaleNormalPrior`, `NormalPrior` |
| [`ppmat/datasets/liflow_dataset.py`](../../ppmat/datasets/liflow_dataset.py) | `LiFlowDataset` (propagator, `TimeDelayedPairDataset`) |
| [`ppmat/datasets/liflow_noised_frame_dataset.py`](../../ppmat/datasets/liflow_noised_frame_dataset.py) | `NoisedFrameDataset` (corrector) |
| [`ppmat/predictor/integrator_predictor.py`](../../ppmat/predictor/integrator_predictor.py) | `IntegratorPredictor` |
| [`molecular_dynamics_integrator/train.py`](../train.py) | Training/validation entrypoint (shared `BaseTrainer`) |
| [`molecular_dynamics_integrator/predict.py`](../predict.py) | Trajectory inference entrypoint |
| [`molecular_dynamics_integrator/evaluate.py`](../evaluate.py) | MSD/RDF metrics |
| [`convert_weights.py`](convert_weights.py) | PyTorch Lightning `.ckpt` → Paddle `.pdparams` |
| [`run_universal_evaluation.py`](run_universal_evaluation.py) | Temperature/seed matrix driver for `evaluate.py` |

---

## 2. Environment

Verified in this workspace (CPU only):

| Package | Version |
| --- | --- |
| Python | 3.10.13 |
| paddlepaddle | 3.0.0 (CPU) |
| numpy | 1.26.4 |
| pandas | 2.x |
| ase | 3.26.0 |
| omegaconf | 2.3.0 |
| pgl | 2.2.6 |
| pymatgen | 2024.10.29 |
| pytest | 8.4.1 |
| torch | 2.6.0 (only needed for the reference fixtures and `convert_weights.py`) |

The `DualPaiNN` implementation is pure Paddle tensor code and has no Cython
dependency. The Cython extension under `ppmat/models/mattersim/` is shared with
MatterSim (it is imported when `ppmat.models` is loaded), so build it once if
your checkout does not ship the compiled module:

```bash
SETUPTOOLS_SCM_PRETEND_VERSION_FOR_PPMAT=0.0.0 python setup.py build_ext --inplace
```

Sanity check:

```bash
python -c "from ppmat.models import LiFlow; from ppmat.predictor import IntegratorPredictor; print('OK')"
```

Run all commands below from the repository root.

---

## 3. Dataset

### 3.1 Layout

Both datasets read a directory with this layout (the tiny
`test/fixtures/liflow/dataset_mini/` follows it exactly):

```
<data_dir>/
├── train.csv  val.csv  test.csv          # split manifests
├── positions_800K.npz                    # one file per temperature
├── atomic_numbers.npy                    # dict: name -> (n_atoms,) int
├── lattice.npy                           # dict: name -> (3, 3) float
└── element_index.npy                     # (max_Z,) int, Z -> embedding index
```

Each `positions_<T>K.npz` stores `name -> (n_frames, n_atoms, 3)` positions.
Each split manifest must contain the columns `name`, `temp`, `t_start`,
`t_end`, `prior_Li`, `prior_frame`, `comp`.

### 3.2 `element_index`

`element_index` is the table used by the original code as
`elements = element_index[atomic_numbers]`. Indices must be smaller than
`num_elements` in the model config (77 for `liflow_universal`). The shipped
fixture uses the contiguous convention `index = Z - 1`; the example structure
in `example_data/` embeds the same 18-entry table so that one-click inference
works without extra files.

**The table must be the one used when the checkpoint was trained.** For
production runs always pass the `element_index.npy` that ships with the
training data (place it next to the input `.npz`, or set
`Predict.element_index`).

### 3.3 Downloading

There is **no public download URL configured for the full Li₂O₂ trajectory
dataset**. If `path` does not exist, both datasets raise `FileNotFoundError`
unless you pass both `url` and `md5`:

```yaml
Dataset:
  train:
    dataset:
      __class_name__: LiFlowDataset
      __init_params__:
        path: data/liflow
        url: https://<your-host>/liflow_data.zip
        md5: <md5-of-the-archive>
        split: train
```

The archive is fetched with `ppmat.utils.download` into the standard dataset
cache. The repository does not bundle the full trajectories. License terms for
the original data are those of the upstream LiFlow project and are not
re-licensed here.

### 3.4 Sampling policy

The train split resamples the time window, the flow time `t`, the prior and
(for the corrector) the noise **on every `__getitem__` call**. None of these is
cached: freezing them would turn the stochastic flow-matching objective into a
fixed one and silently change the trained model. Only raw position arrays are
optionally cached with `in_memory=True`; both universal configs ship
`in_memory: false`, because a full trajectory archive (all `positions_{T}K.npz`
arrays, four temperatures) does not fit in RAM and would additionally be copied
into every DataLoader worker through pickling. With the default the datasets
keep one lazy `np.load` handle per file and per process
(`__getstate__` drops the handles before workers fork).

The validation split follows the original `FlowModule.validation_step`: the
same `(start, end, prior)` triple is reused for 11 evenly spaced flow times in
`[0, 1]`, derived deterministically from the row index.

### 3.5 Propagator vs. corrector samples

| | `LiFlowDataset` (propagator) | `NoisedFrameDataset` (corrector) |
| --- | --- | --- |
| Sample | frames `t` and `t + time_delay_steps` | a single frame at `t` |
| Input | start frame | noised frame (`UniformScaleNormalPrior(0.25)`) |
| Target | end frame | noise-free frame |
| Neighbour list | start frame (`neighbor_list_both_ends=False`) | **noisy** positions |
| Prior | `AdaptiveMaxwellBoltzmannPrior` from `prior_Li`/`prior_frame` | `MaxwellBoltzmannPrior(0.1)` |

The two datasets share only the file loading and caching layer
(`LiFlowTrajectoryDataset`); sample construction is implemented independently,
and the corrector never reuses propagator pairs.

---

## 4. Models

### 4.1 `DualPaiNN`

`num_features=64`, `num_radial_basis=20`, `num_layers=3`, `num_elements=77`,
`r_max=5.0`, `r_offset=0.5`, `ref_temp=1000.0`.

- Distances are expanded on `num_radial_basis` **Bessel** radial basis
  functions, `BesselBasis(num_radial_basis, r_max)`. The input is the shifted
  and clipped distance `clip(r + r_offset, max=r_max)`; the basis itself only
  depends on `r_max`, `r_offset` is the shift applied before it. A cosine cutoff
  at `r_max` multiplies the message.
- Each layer updates scalar and vector features with equivariant message
  passing; messages use the radial embedding and the directional unit vector.
- The flow time `t` and the **reduced temperature `T / ref_temp`** (not an
  inverse temperature) are each expanded with `GaussianFourierBasis` and
  concatenated to the scalar features before message passing;
  `ref_temp = 1000.0` in both universal configs.
- `prediction_mode: velocity` (default) predicts the flow velocity
  `x_end - x_0 = x_end - (x_start + prior)`. `prediction_mode: data` predicts
  the **displacement** `x_end - x_start` from the start frame; it is not an
  absolute position, and the integrator reconstructs
  `x_end = x_start + displacement`.

### 4.2 `LiFlow`

`LiFlow` subclasses `DualPaiNN` and adds the PaddleMaterials protocol:

- `forward(batch_data, return_loss=True, return_prediction=True)` returns
  `{"loss_dict": {...}, "pred_dict": {...}}`. The prediction key is `velocity`
  for `prediction_mode: velocity` and `displacement` for `prediction_mode: data`.
  `pred_dict` is empty when `return_prediction=False`; `loss_dict` is empty when
  `return_loss=False` and otherwise holds both the keyed loss and the aggregate
  `loss_dict["loss"]` used by the trainer;
- `predict(batch_data)` is the inference path used by `IntegratorPredictor` and
  returns the same single key (`velocity` or `displacement`);
- the metric `LiFlowMSE` sums the three xyz components and averages over nodes,
  matching the original `LiFlowMSE` for the `velocity` key.

---

## 5. Checkpoints

| Name | Status | URL | Content |
| --- | --- | --- | --- |
| `liflow_universal` | **Published** | `https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/molecular_dynamics_integrator/liflow/liflow_universal.zip` | Universal **propagator** weights (`num_features=64`, `num_layers=3`, `num_elements=77`), strict-loadable |
| `liflow_universal_corrector` | **Not published** | — | — |

Notes:

- Only `liflow_universal.zip` exists on the object store (verified HTTP 200).
  The previously proposed `liflow_universal_propagator.zip` and
  `liflow_universal_corrector.zip` assets return HTTP 404 and are therefore not
  registered in `MODEL_REGISTRY`.
- The published package is tensor-for-tensor identical to the previously cached
  propagator archive; it contains no corrector weights.
- The published package YAML carries an older `Dataset`/`Predict` section from
  before the dataset rewrite. `build_model_from_name` only reads `Model` and the
  checkpoint, so registered inference works; use the repo configs in
  `configs/` for training.
- All loading goes through `save_load.load_pretrain(..., strict=True)`: a
  missing, unexpected or wrongly shaped tensor raises instead of warning.

To point inference at your own trained weights instead, use a local
config/checkpoint pair (see [Prediction](#8-prediction)).

---

## 6. Training

Training uses the shared PaddleMaterials `BaseTrainer`:

```bash
# propagator (full dataset at data/liflow)
python molecular_dynamics_integrator/train.py \
    -c molecular_dynamics_integrator/liflow/configs/liflow_universal_propagator.yaml

# corrector (NoisedFrameDataset semantics)
python molecular_dynamics_integrator/train.py \
    -c molecular_dynamics_integrator/liflow/configs/liflow_universal_corrector.yaml
```

Both configs use `batch_size: 16`, `Adam(lr=3e-4)`, gradient clipping at
`10.0`, `prediction_mode: velocity`, and `best_metric_indicator: eval_loss`.
`BaseTrainer` has no patience-based early stopping, so `max_epochs: 500` is
fixed and the best checkpoint is selected by `eval_loss`.

Outputs land in `output/liflow_universal_propagator/` and
`output/liflow_universal_corrector/` as timestamp-suffixed run directories with
`run.log` and `checkpoints/best.pdparams`.

### 6.1 Smoke test (CPU, minutes)

```bash
python molecular_dynamics_integrator/train.py \
    -c molecular_dynamics_integrator/liflow/configs/liflow_universal_propagator.yaml \
    Trainer.max_epochs=1 Trainer.max_iter=10 \
    Trainer.output_dir=output/liflow_smoke \
    Dataset.train.dataset.__init_params__.path=test/fixtures/liflow/dataset_mini \
    Dataset.val.dataset.__init_params__.path=test/fixtures/liflow/dataset_mini \
    Dataset.train.loader.num_workers=0 Dataset.val.loader.num_workers=0
```

Same overrides with `liflow_universal_corrector.yaml` exercise the corrector
path. Success criteria: exit code 0, a `run.log` under the timestamp-suffixed
output directory, and a finite `loss` / `velocity` metric in the log.

---

## 7. Validation and testing

```bash
# eval only
python molecular_dynamics_integrator/train.py \
    -c molecular_dynamics_integrator/liflow/configs/liflow_universal_propagator.yaml \
    "Global.do_train=false" "Global.do_eval=true" \
    Trainer.pretrained_model_path=output/liflow_universal_propagator/checkpoints/best.pdparams

# test split
python molecular_dynamics_integrator/train.py \
    -c molecular_dynamics_integrator/liflow/configs/liflow_universal_propagator.yaml \
    "Global.do_train=false" "Global.do_eval=false" "Global.do_test=true" \
    Trainer.pretrained_model_path=output/liflow_universal_propagator/checkpoints/best.pdparams
```

The logged metric is `velocity` (`LiFlowMSE`).

---

## 8. Prediction

### 8.1 One-click with registered weights

```bash
python molecular_dynamics_integrator/predict.py \
    -c molecular_dynamics_integrator/liflow/configs/liflow_universal_inference.yaml \
    -i molecular_dynamics_integrator/liflow/example_data/initial_structure.npz \
    -o output/liflow_prediction
```

`Predict.propagator_model: liflow_universal` downloads (once, cached in
`~/.paddlemat/weights/`), strict-loads the checkpoint, and integrates a
future-free trajectory. The corrector is disabled because no corrector
checkpoint is published (`corrector_model`, `corrector_config`,
`corrector_checkpoint` are all `null`).

Outputs in `-o`:

| File | Content |
| --- | --- |
| `trajectory.npy` | `(steps + 1, n_atoms, 3)` float32 |
| `trajectory.xyz` | one frame per step, ASE-compatible |
| `run_metadata.json` | `input`, `temperature`, `steps`, `flow_steps`, `solver`, `seed`, `trajectory_shape`, `generated_at_utc` |

CLI overrides: `--temperature`, `--steps`, `--flow_steps`, `--solver`,
`--seed`.

Input `.npz` keys: `positions` `(n_atoms, 3)`, `atomic_numbers` `(n_atoms,)`,
`lattice` `(3, 3)`, `element_index` (optional; otherwise taken from
`Predict.element_index` or `element_index.npy` next to the input), and
optionally `prior_Li` / `prior_frame`.

### 8.2 Local config + checkpoint

`Predict.propagator_model` and the pair
`Predict.propagator_config` + `Predict.propagator_checkpoint` are mutually
exclusive. To use your own weights:

```yaml
Predict:
  propagator_model: null
  propagator_config: molecular_dynamics_integrator/liflow/configs/liflow_universal_propagator.yaml
  propagator_checkpoint: output/liflow_universal_propagator/checkpoints/best.pdparams
```

Add `corrector_config` + `corrector_checkpoint` (or `corrector_model`) to
enable the correction step.

### 8.3 Integrator semantics

| Argument | Meaning |
| --- | --- |
| `steps` | number of macro steps; output has `steps + 1` frames |
| `flow_steps` | Euler/Heun sub-steps per macro step |
| `solver` | `euler` or `heun` |
| `corrector_every` | apply the corrector every N macro steps (`1` = every step) |
| `fix_com` | subtract the centre-of-mass displacement of each step |
| `num_retries` | resample attempts when a step is numerically unstable |
| `seed` | seeds the prior sampling only; the trajectory is otherwise deterministic |

Each macro step samples the prior, integrates the propagator from `t=0` to
`t=1` over `flow_steps` sub-steps, and (if enabled) integrates the corrector
from the noisy prediction back to a clean frame. No ground-truth frame is read.

### 8.4 Converting original PyTorch checkpoints (optional)

```bash
# stage 1 (torch env): dump the Lightning state dict
python molecular_dynamics_integrator/liflow/convert_weights.py \
    --input /path/to/universal.ckpt --dump-state \
    --state-npz /tmp/universal_state.npz

# stage 2 (paddle env): transpose Linear weights and write .pdparams
python molecular_dynamics_integrator/liflow/convert_weights.py \
    --state-npz /tmp/universal_state.npz \
    --output output/liflow_universal_propagator/checkpoints/best.pdparams \
    --audit output/weight_conversion.json
```

The converter strips `model.` / `network.` prefixes, transposes 2-D
`*.weight` tensors (torch `[out, in]` → paddle `[in, out]`) while leaving
`atom_embedding.weight` and scalar buffers untouched, and raises on any key or
shape mismatch. The audit JSON records the source hash, per-key
shape/dtype equality, the number of transposed weights and an aggregate
checksum.

---

## 9. MSD / RDF evaluation

```bash
python molecular_dynamics_integrator/evaluate.py \
    --prediction output/liflow_prediction/trajectory.npy \
    --reference /path/to/reference_trajectory.npy \
    --structure molecular_dynamics_integrator/liflow/example_data/initial_structure.npz \
    --output output/liflow_prediction/metrics.json
```

| Metric | Definition |
| --- | --- |
| `msd_Li` | final-frame mean squared displacement (`Σ_xyz`) over Li atoms (`Z == 3`) |
| `msd_frame` | same over the non-Li atoms |
| `msd_*_ref` | the same metric on the reference trajectory |
| `rdf_mae` | mean absolute error between predicted and reference RDF |
| `final_step` | last frame index used |

RDF uses a periodic minimum-image neighbour list (`rmax=5.0`, `nbins=50`);
frames `[5:]` of the prediction are used when more than five frames exist,
otherwise the last frame only. The reference RDF uses `[500::100]`, so the
reference trajectory must contain at least ~1000 frames for the original
protocol.

The temperature/seed matrix driver:

```bash
python molecular_dynamics_integrator/liflow/run_universal_evaluation.py \
    --prediction_dir output/liflow_matrix/pred \
    --reference_dir output/liflow_matrix/ref \
    --structure molecular_dynamics_integrator/liflow/example_data/initial_structure.npz \
    --output output/liflow_universal_eval \
    --temperatures 600,800,1000,1200 --seeds 1,2,3 \
    --modes propagator,propagator_corrector
```

Files are matched by name:

```
<prediction_dir>/temp_<T>_seed_<s>_<mode>.npy
<reference_dir>/temp_<T>_seed_<s>.npy
```

It writes one CSV per temperature/mode plus an aggregate `summary.json` with
per-metric mean/std and the failed-combination rate.

---

## 10. Reproduction status

### 10.1 Automated tests

Command:

```bash
python -m pytest -q \
  test/test_liflow_dataset.py test/test_liflow_model.py \
  test/test_liflow_predictor.py test/test_liflow_metric.py \
  test/test_liflow_evaluation.py test/test_liflow_alignment.py \
  test/test_liflow_layers.py test/test_liflow_public_imports.py \
  test/test_liflow_weight_conversion.py test/test_liflow_model_package.py \
  test/test_liflow_train_entrypoint.py test/test_liflow_predict_entrypoint.py
```

Result in the CPU workspace above: **82 passed**.

| Suite | Covers |
| --- | --- |
| `test_liflow_dataset.py` | split loading, dynamic resampling (no frozen cache), corrector noised-frame construction, variable-length batching, zero-edge graphs, and the opt-in `as_dict` loader flag (a non-LiFlow dataset keeps the default collator) |
| `test_liflow_model.py` | forward shapes, loss reduction, gradient flow, `prediction_mode` |
| `test_liflow_layers.py`, `test_liflow_alignment.py` | numerical alignment against `test/fixtures/liflow/reference_model.npz` / `reference_layers.npz` |
| `test_liflow_predictor.py` | integrator shape, solver, centre-of-mass, no future information |
| `test_liflow_metric.py`, `test_liflow_evaluation.py` | `LiFlowMSE`, MSD/RDF |
| `test_liflow_weight_conversion.py` | transpose rules, audit JSON, strict-load contract |
| `test_liflow_model_package.py` | registry entry, download, strict load, config/weights match |
| `test_liflow_train_entrypoint.py` | real 2-step propagator and corrector training subprocesses |
| `test_liflow_predict_entrypoint.py` | real `predict.py` subprocess with a local checkpoint; registry/disable/mutual-exclusion branches |

Alignment fixtures were generated with the original PyTorch implementation at
commit `e6fc475361d046865f12cae1aee11c4f56c48d87` (torch 2.5.1+cpu, seed
20260910); see `test/fixtures/liflow/reference_manifest.json`.

### 10.2 Verified local runs

| Run | Result |
| --- | --- |
| One-click registered inference (`liflow_universal`, 25 steps × 10 Euler sub-steps, seed 42, T=800 K) | `trajectory.npy` `(26, 4, 3)` float32, all finite; `trajectory.xyz` has 26 frames; `run_metadata.json` records `trajectory_shape: [26, 4, 3]` |
| `evaluate.py` on that trajectory with `test/fixtures/liflow/reference_mini.npy` | Pipeline executes and writes finite metrics. **Not a model-accuracy result**: `reference_mini.npy` was produced by a 4-feature toy model, so the predicted and reference trajectories are not comparable |

### 10.3 Results table (to be filled from real runs)

The table below is intentionally empty. It must only be populated from real
runs against the full Li₂O₂ reference trajectories, with the exact config,
seed and checkpoint recorded. `PENDING` means "not measured yet", not
"supported".

| Model | Dataset | T (K) | Seed | MSD Li (Å²) | MSD frame (Å²) | RDF MAE | Config |
| --- | --- | --- | --- | ---: | ---: | ---: | --- |
| `liflow_universal` | Li₂O₂ trajectories | 600 | — | PENDING | PENDING | PENDING | `configs/liflow_universal_inference.yaml` |
| `liflow_universal` | Li₂O₂ trajectories | 800 | — | PENDING | PENDING | PENDING | `configs/liflow_universal_inference.yaml` |
| `liflow_universal` | Li₂O₂ trajectories | 1000 | — | PENDING | PENDING | PENDING | `configs/liflow_universal_inference.yaml` |
| `liflow_universal` | Li₂O₂ trajectories | 1200 | — | PENDING | PENDING | PENDING | `configs/liflow_universal_inference.yaml` |
| `liflow_universal` + corrector | Li₂O₂ trajectories | — | — | PENDING | PENDING | PENDING | corrector checkpoint not published |

### 10.4 NOT RUN in the current environment

| Item | Why | Command to run elsewhere |
| --- | --- | --- |
| `test/test_liflow_cinn_workflows.py` (5 tests) | Requires Linux + CUDA and a CINN-capable Paddle build; this workspace is CPU-only | `python -m pytest -q test/test_liflow_cinn_workflows.py` on a GPU host |
| Full-dataset training (500 epochs) | Needs the external Li₂O₂ dataset and a GPU | [Training](#6-training) commands |
| Full-dataset MSD/RDF matrix | Needs the full reference trajectories | [Evaluation matrix](#9-msd--rdf-evaluation) |
| Corrector inference | No published corrector checkpoint | Train the corrector, then set `corrector_config`/`corrector_checkpoint` |

---

## 11. License

- Original LiFlow reference implementation
  (`learningmatter-mit/liflow`): **MIT License**, `Copyright (c) 2024
  Learning Matter @ MIT`. The upstream repository does ship a `LICENSE` file
  (verified HTTP 200 on `raw.githubusercontent.com`); the full text is preserved
  in [`LICENSE.liflow`](LICENSE.liflow) next to this README, as required by the
  MIT "include in all copies or substantial portions" clause. The local git
  checkout used while porting did not contain the file, which is why it is
  reproduced from the published repository rather than copied from disk.
- This PaddleMaterials integration: Apache License 2.0 (see the file headers
  and the repository root `LICENSE`).
- The dataset, the checkpoints and any derivative weights remain subject to the
  terms of their original publishers.

---

## 12. Citation

```bibtex
@misc{liflow,
  title  = {LiFlow: Flow matching for accelerated simulation of atomic transport},
  author = {Learning Matter, MIT},
  year   = {2025},
  url    = {https://github.com/learningmatter-mit/liflow}
}
```

Please cite the original LiFlow work when using these models. This integration
is part of PaddleMaterials.
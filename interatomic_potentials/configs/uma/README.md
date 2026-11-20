# UMA

[UMA: A Family of Universal Models for Atoms](https://github.com/facebookresearch/fairchem/tree/main/src/fairchem/core/models/uma)

## Abstract

UMA is a family of equivariant machine-learning interatomic potentials. This
implementation ports the single-task eSCN architecture to Paddle and integrates
it with the native PaddleMaterials data, training, and prediction workflow. It
supports direct energy and force prediction; FairChem multi-task, MoE, stress,
Hessian, graph-parallel, and optimized execution backends are outside the scope
of this implementation.

## Datasets

The supplied configurations use OMat24 rattled structures and OC20 S2EF
structures with total-energy and atomic-force labels. Data are read through
the corresponding `OMat24Dataset` and `UMAOC20Dataset`, converted to
`pgl.Graph`, and batched by `DefaultCollator`.

| Field | Description |
| :-- | :-- |
| `energy` | Total structure energy in eV, stored as float32 |
| `forces` | Atomic forces in eV/Å, float32 with shape `[num_atoms, 3]` |

OMat24Dataset downloads the official OMat24 rattled-500 train or val archive
independently, with separate cache directories for each split. The OMat24
configuration evaluates on val; no separate test split is configured. A local
path may point directly to an ASE-LMDB file or a directory of shards. Both
datasets download automatically when the configured local path is missing.
The 1M and sAlex packages are not configured here. OC20 currently retains the
previous prepared package until its official replacement is provided.

Both datasets support the standard `path`, `property_names`,
`build_structure_cfg`, `build_graph_cfg`, `transforms`, `cache_path`,
`overwrite`, and `filter_unvalid` options. The optional cache is populated
per sample on first access; it does not precompute the full dataset. With
`filter_unvalid=True`, records with nonfinite selected labels are excluded.
Filtering is disabled by default to preserve the supplied training splits.

## Models

The model embeds atomic numbers and radial edge distances, rotates spherical
node features into the local edge frame, and updates them with eSCN SO(2)
convolutions and equivariant interaction blocks. A scalar head predicts energy
and an equivariant L=1 head predicts forces directly.

`predict` accepts a graph, a batched graph, or a list of graphs and runs without
gradient tracking. Energy outputs have shape `[num_structures, 1]`; force
outputs retain all atoms with shape `[sum(num_atoms), 3]`. A graph input returns
a dictionary, while a list input returns a list of dictionaries.

## Recorded Paddle results

The following are recorded Paddle-only measurements, not a matched-epoch
PyTorch comparison. Both runs used eight RTX 3090 GPUs, a batch size of four
per GPU (global batch size 32), and full-precision training. The native
single-task model has 6.26M trainable parameters.

| Dataset | Train structures | Validation structures | Epochs | Energy MAE (eV/atom) | Force vector L2 (eV/Å) | Training time |
| :-- | --: | --: | --: | --: | --: | --: |
| OC20 `train_200K` / `val_id` | 200,000 | 25,000 | 3 | 0.008893 | 0.100804 | 2 h 56 min |
| OMat24 `rattled-500` | 6,922,197 | 71,522 | 1 | 0.015218 | 0.179509 | 5 h 15 min |

The OMat24 run used the complete training split without a sample limit. With
`drop_last: True`, only the final 21 structures that did not fill a global batch
were omitted.

Energy MAE averages `abs(E_pred / N - E_label / N)` over structures, where `N`
is the number of atoms in each structure. Force vector L2 averages the
Euclidean force error over atoms within each structure, then over structures.
These are logged as `energy(loss)` and `forces(loss)`. The default
`energy(metric)` measures total-energy L1 error and `forces(metric)` measures
component-wise force L1 error; neither is interchangeable with the table's
metrics. The weighted training objective is `10 * energy(loss) + 30 * forces(loss)`.

## Matched-epoch framework comparison

The OC20 comparison below uses 200,000 training structures and the same 25,000
validation structures for three epochs (18,750 optimizer steps). Both frameworks
start from identical weights and use a global batch size of 32, FP32 without
AMP, AdamW, and the same learning-rate schedule, sample order, and rotation
schedule. Each framework uses two RTX 3090 GPUs, with eight structures per GPU
and two gradient-accumulation steps. PyTorch TF32 is disabled.

| Framework | Energy MAE (eV/atom) | Force vector L2 (eV/Å) | Training and validation time |
| :-- | --: | --: | --: |
| Paddle | 0.008046 | 0.099372 | 2 h 15 min |
| PyTorch single-task reference | 0.008249 | 0.098975 | 2 h 7 min |

These measurements use the final epoch checkpoints and the metrics defined
above, reduced over the complete configured validation split without padding.
The PyTorch reference matches this single-task architecture using FairChem core
operators; it is not the full multi-task UMA model. These are single-seed results,
not a statistical equivalence assessment.

The matched one-epoch OMat24 comparison is still running; its final validation
results are not yet available. The recorded OMat24 result above is from the
earlier Paddle-only run and is not part of this framework comparison.

## Training

Run from the repository root and set the dataset paths in the configuration
or through command-line overrides. The examples below use eight GPUs to retain
the recorded global batch size of 32. The OMat24 YAML defaults to six epochs;
the explicit override selects the one-epoch budget reported above.

```bash
python -m paddle.distributed.launch --gpus=0,1,2,3,4,5,6,7 \
  interatomic_potentials/train.py \
  -c interatomic_potentials/configs/uma/uma_omat24_r500_s2ef.yaml \
  Trainer.max_epochs=1 Trainer.eval_freq=1
```

```bash
python -m paddle.distributed.launch --gpus=0,1,2,3,4,5,6,7 \
  interatomic_potentials/train.py \
  -c interatomic_potentials/configs/uma/uma_oc20_200k_s2ef.yaml \
  Trainer.max_epochs=3 Global.do_test=False
```

## Evaluation

Use a single evaluation process to cover the complete configured validation
split, including its final partial batch. For a matched-epoch comparison,
select the end-of-epoch checkpoint rather than a best checkpoint from a
different epoch.

```bash
CUDA_VISIBLE_DEVICES=0 python interatomic_potentials/train.py \
  -c interatomic_potentials/configs/uma/uma_omat24_r500_s2ef.yaml \
  Global.do_train=False Global.do_eval=True Global.do_test=False \
  Trainer.pretrained_model_path="path/to/epoch_1.pdparams"
```

```bash
CUDA_VISIBLE_DEVICES=0 python interatomic_potentials/train.py \
  -c interatomic_potentials/configs/uma/uma_oc20_200k_s2ef.yaml \
  Global.do_train=False Global.do_eval=True Global.do_test=False \
  Trainer.pretrained_model_path="path/to/epoch_3.pdparams"
```

## Prediction

Registered packages are available as `uma_omat24_r500_s2ef` and
`uma_oc20_200k_s2ef`. The default weight selection prefers `best.pdparams`.

```bash
python interatomic_potentials/predict.py \
  --model_name uma_omat24_r500_s2ef \
  --input_path path/to/structure.cif \
  --input_format cif \
  --output_path results
```

To use a local checkpoint instead of a registered package:

```bash
python interatomic_potentials/predict.py \
  --config_path interatomic_potentials/configs/uma/uma_omat24_r500_s2ef.yaml \
  --checkpoint_path path/to/epoch_1.pdparams \
  --device cpu \
  --input_path path/to/structure.cif \
  --input_format cif \
  --output_path results
```

## Citation

```bibtex
@misc{fairchem_uma,
  title = {UMA: A Family of Universal Models for Atoms},
  author = {FAIR Chemistry Team},
  howpublished = {\url{https://github.com/facebookresearch/fairchem}},
  year = {2025}
}
```

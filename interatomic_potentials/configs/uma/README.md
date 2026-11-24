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

| Dataset | Download |
| :-- | :-- |
| OC20 | [Data](https://paddle-org.bj.bcebos.com/paddlematerials/datasets/UMA/uma_datasets.zip) |
| OMat24 | [Train](https://paddle-org.bj.bcebos.com/paddlematerials/datasets/OMat24/train/rattled-500.tar.gz), [Validation](https://paddle-org.bj.bcebos.com/paddlematerials/datasets/OMat24/val/rattled-500.tar.gz) |

## Models

The model embeds atomic numbers and radial edge distances, rotates spherical
node features into the local edge frame, and updates them with eSCN SO(2)
convolutions and equivariant interaction blocks. A scalar head predicts energy
and an equivariant L=1 head predicts forces directly.

## Results

Both Paddle runs used eight RTX 3090 GPUs, a batch size of four
per GPU (global batch size 32), and full-precision training. The native
single-task model has 6.26M trainable parameters.

| Dataset | Train structures | Validation structures | Epochs | Energy MAE (eV/atom) | Force vector L2 (eV/Å) | Training time |
| :-- | --: | --: | --: | --: | --: | --: |
| OC20 `train_200K` / `val_id` | 200,000 | 25,000 | 3 | 0.008893 | 0.100804 | 2 h 56 min |
| OMat24 `rattled-500` | 6,922,197 | 71,522 | 1 | 0.015218 | 0.179509 | 5 h 15 min |

Configurations: [OC20](uma_oc20_200k_s2ef.yaml),
[OMat24](uma_omat24_r500_s2ef.yaml).
Pretrained weights: [Checkpoint](https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/interatomic_potentials/uma/uma_checkpoints.zip).

Energy MAE averages `abs(E_pred / N - E_label / N)` over structures, where `N`
is the number of atoms in each structure. Force vector L2 averages the
Euclidean force error over atoms within each structure, then over structures.
Force vector L2 is not component-wise force MAE.

## Training

Run from the repository root and set the dataset paths in the configuration
or through command-line overrides. The examples below use eight GPUs and a
global batch size of 32, with the epoch counts reported above.

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
split, including its final partial batch.

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
  --input_path interatomic_potentials/example_data/cifs/mp-18767-LiMnO2.cif \
  --input_format cif \
  --output_path results
```

To use a local checkpoint instead of a registered package:

```bash
python interatomic_potentials/predict.py \
  --config_path interatomic_potentials/configs/uma/uma_omat24_r500_s2ef.yaml \
  --checkpoint_path path/to/epoch_1.pdparams \
  --device cpu \
  --input_path interatomic_potentials/example_data/cifs/mp-18767-LiMnO2.cif \
  --input_format cif \
  --output_path results
```

Predicted energies and atomic forces are saved to `results/result.csv`.

## Citation

```bibtex
@misc{fairchem_uma,
  title = {UMA: A Family of Universal Models for Atoms},
  author = {FAIR Chemistry Team},
  howpublished = {\url{https://github.com/facebookresearch/fairchem}},
  year = {2025}
}
```

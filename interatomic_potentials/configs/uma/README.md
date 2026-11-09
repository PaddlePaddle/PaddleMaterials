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
| `energy` | Structure energy |
| `forces` | Atomic forces with shape `[num_atoms, 3]` |

OMat24Dataset downloads the official OMat24 rattled-500 train or val archive
independently, with separate cache directories for each split. The OMat24
configuration evaluates on val; no separate test split is configured. A local
path may point directly to an ASE-LMDB file or a directory of shards. Both
datasets download automatically when the configured local path is missing.
The 1M and sAlex packages are not configured here. OC20 currently retains the
previous prepared package until its official replacement is provided.

## Models

The model embeds atomic numbers and radial edge distances, rotates spherical
node features into the local edge frame, and updates them with eSCN SO(2)
convolutions and equivariant interaction blocks. A scalar head predicts energy
and an equivariant L=1 head predicts forces directly.

## Results

The native single-task model was trained on OC20 S2EF and OMat24
`rattled-500`. Both runs used eight RTX 3090 GPUs, a batch size of four per GPU,
and full-precision training. The model has 6.26M trainable parameters.

| Dataset | Train structures | Validation structures | Epochs | Energy MAE (eV/atom) | Force vector L2 (eV/A) | Training time |
| :-- | --: | --: | --: | --: | --: | --: |
| OC20 `train_200K` / `val_id` | 200,000 | 25,000 | 3 | 0.008893 | 0.100804 | 2 h 56 min |
| OMat24 `rattled-500` | 6,922,197 | 71,522 | 1 | 0.015218 | 0.179509 | 5 h 15 min |

The OMat24 run used the complete training split without a sample limit. With
`drop_last: True`, only the final 21 structures that did not fill a global batch
were omitted.

## Training

```bash
python interatomic_potentials/train.py \
  -c interatomic_potentials/configs/uma/uma_omat24_r500_s2ef.yaml
```

```bash
python interatomic_potentials/train.py \
  -c interatomic_potentials/configs/uma/uma_oc20_200k_s2ef.yaml
```

## Evaluation

```bash
python interatomic_potentials/train.py \
  -c interatomic_potentials/configs/uma/uma_omat24_r500_s2ef.yaml \
  Global.do_train=False Global.do_eval=True Global.do_test=False \
  Trainer.pretrained_model_path="path/to/checkpoint.pdparams"
```

## Prediction

```bash
python interatomic_potentials/predict.py \
  --model_name uma_omat24_r500_s2ef \
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

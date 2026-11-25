# MACE-MP-0

[MACE: Higher Order Equivariant Message Passing Neural Networks for Fast and Accurate Force Fields](https://arxiv.org/abs/2401.00096)

MACE is an equivariant interatomic potential that predicts energies, forces and
stresses with higher-order message passing. This directory provides the
PaddlePaddle implementation of MACE-MP-0 medium and the released
`mace_mp0_medium` weights, which cover 89 elements (H to Bi).

## Datasets

MACE-MP-0 is trained on the Materials Project Trajectory (MPtrj, 2022.9)
dataset: 145,923 compounds / 1,580,395 structures at the GGA/GGA+U level, split
into fixed train / val / test JSON files. The configs read the same MPtrj files
as the CHGNet config (`./data/MPtrj_2022.9_full/{train,val,test}.json`), with
energies taken from `uncorrected_total_energy`, forces as stored and stresses
converted once from kBar with `-0.1 * stress_kBar`.

## Models

`mace_mp0_medium` follows the released MACE architecture: atomic numbers are
embedded as node features, distances are expanded with a Bessel radial basis,
angles with spherical harmonics, and two interaction blocks pass higher-order
(3-body) messages through symmetric tensor contractions; atomic energies are
summed and forces/stresses are obtained by differentiating the total energy.

Hyper-parameters are those of `mace_mp0_medium.yaml`: `r_max = 6.0 Å`,
`num_bessel = 10`, `max_ell = 3`, `num_interactions = 2`,
`hidden_irreps = 128x0e+128x1o`, `radial_MLP = [64, 64, 64]`, `correlation = 3`,
`num_elements = 89`.

## Results

All numbers below were produced by this Paddle implementation, on 1x
Tesla V100-PCIE-16GB with PaddlePaddle 3.1.0 in float32, evaluated on a fixed
validation subset of 754 frames / 60 materials from the MPtrj validation split.

- **energy**: `uncorrected_total_energy / natoms`, MAE in meV/atom
- **force**: MPtrj forces, MAE in eV/Å
- **stress**: `-0.1 * stress_kBar`, MAE over the 6 independent components of the
  symmetric tensor, in GPa

| Checkpoint | Energy MAE (meV/atom) | Force MAE (eV/Å) | Stress MAE (GPa) |
| --- | ---: | ---: | ---: |
| converted MACE-MP-0 medium | 17.0261 | 0.061427 | 0.42247 |
| **`mace_mp0_medium` (2000 steps, released)** | **16.9411** | **0.061174** | **0.42211** |

The released weights lower all three validation errors with respect to the
converted starting point. A combined score
`E / 17.0261 + F / 0.0614272 + S9 / 0.310582` (lower is better) was used to
select step 2000 of the run. Per-step values, the checkpoint-selection rule and
the validation of the reloaded archive are collected in the
[validation report](mace_mp0_medium_validation_report.md).

## Checkpoint and prediction

`--model_name=mace_mp0_medium` downloads the released archive:

https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/interatomic_potentials/mace/mace_mp0_medium.zip

| Artifact | SHA256 |
| --- | --- |
| `mace_mp0_medium.zip` | `ef2eef766e4d485ed99507a1014d3cab704dd7a73e7cf5ee4393bf5a65dbbb08` |
| packaged `checkpoints/best.pdparams` | `6182192a55304de567c9c544dce8568f2a37d8881e113e91a96c76ce461ccb96` |

Run from the repository root after installing the project dependencies:

```bash
python interatomic_potentials/predict.py \
  --model_name=mace_mp0_medium --device=cpu \
  --input_format=cif --input_path=./interatomic_potentials/example_data/cifs/

# Alternatively, after extracting mace_mp0_medium.zip into the repository root:
python interatomic_potentials/predict.py \
  --config_path=mace_mp0_medium/mace_mp0_medium.yaml \
  --checkpoint_path=mace_mp0_medium/checkpoints/best.pdparams \
  --device=cpu --input_format=cif \
  --input_path=./interatomic_potentials/example_data/cifs/
```

The archive ships its own YAML and checkpoint, so both entry points run as is.

## Training

`mace_mp0_medium` is a Paddle training run of 2000 steps that starts from the
official MACE-MP-0 medium weights converted to Paddle and trains them further
on MPtrj. The run uses Adam (weight decay 0), a constant learning rate of `1e-6`,
the model's own SmoothL1 loss with E : F : S weights `1 : 1 : 0.1`, gradient
clipping at 10, seed 42, and batches packed to at most 112 atoms and 8 frames.
It used 1x V100-PCIE-16GB, peak GPU memory 7,012 MB, and 11,791 s including
in-loop validation. The 40,010-frame training subset is drawn from the MPtrj
train split and excludes the validation and test splits.

[mace_mp0_medium_finetune.yaml](mace_mp0_medium_finetune.yaml) is the
configuration for this training setup:

```bash
python interatomic_potentials/train.py \
  -c interatomic_potentials/configs/mace/mace_mp0_medium_finetune.yaml
```

The converted starting weights are expected at
`./weights/mace_mp0_medium_converted.pdparams`.

## Citation

```bibtex
@article{batatia2024mace,
  title={MACE: Higher Order Equivariant Message Passing Neural Networks for Fast and Accurate Force Fields},
  author={Batatia, Ilyes and Kov{\'a}cs, D{\'a}vid P{\'e}ter and Simm, Gregor N. C. and Ortner, Christoph and Cs{\'a}nyi, G{\'a}bor},
  journal={arXiv preprint arXiv:2401.00096},
  year={2024}
}
```
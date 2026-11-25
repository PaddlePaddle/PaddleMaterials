# MACE-MP-0

[MACE: Higher Order Equivariant Message Passing Neural Networks for Fast and Accurate Force Fields](https://arxiv.org/abs/2401.00096)

MACE is an equivariant interatomic potential using higher-order message passing
to predict energies, forces and stresses. MACE-MP-0 was pretrained on the
Materials Project Trajectory (MPtrj) dataset and supports 89 elements.

## Datasets

MACE-MP-0 is pretrained on the Materials Project Trajectory (MPtrj, 2022.9)
dataset: 145,923 compounds / 1,580,395 structures at the GGA/GGA+U level, split
into fixed train / val / test JSON files. The repository configs read the same
MPtrj files as the CHGNet config
(`./data/MPtrj_2022.9_full/{train,val,test}.json`), with energies taken from
`uncorrected_total_energy`, forces as stored and stresses converted once from
kBar with `-0.1 * stress_kBar`.

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

## Released model

The current `mace_mp0_medium` checkpoint is a **2000-optimisation-step Paddle
fine-tune of converted official PyTorch weights**, not a 200-epoch training run
from scratch and not a reproduction of the paper's accuracy. The historical run
used 500 steps followed by a 1500-step continuation with restored model parameters
and a **fresh Adam state**, at constant learning rate `1e-6`.

### Historical results and label protocol

These are previously recorded results; **the full datasets were not re-evaluated
for this documentation update**. Validation used a fixed subset of 754 frames /
60 materials, not the full official split. Labels are
`uncorrected_total_energy / natoms`, stored MPtrj forces, and
`-0.1 * stress_kBar` (converted once in the dataset pipeline). Stress below uses
the 6 independent components of the symmetric tensor.

| Subset | Checkpoint | Energy MAE (meV/atom) | Force MAE (eV/Å) | Stress MAE (GPa, 6 comp.) |
| --- | --- | ---: | ---: | ---: |
| val754 | step 2000 | 16.9411 | 0.061174 | 0.422105 |
| test326/340 | converted weights | 13.9885 | 0.062960 | 0.458858 |
| test326/340 | step 2000 | 14.0511 | 0.062653 | 0.460272 |

The checkpoint was selected by a combined validation score, not by test metrics
or by being best on every metric: step 1000 has lower validation stress MAE.
The test summary excludes the same 14 large frames for both checkpoints because
of memory limits; it is not a full 340-frame result. See the
[validation report](mace_mp0_medium_validation_report.md) for the selection rule,
limitations and artifact hashes.

## Checkpoint and prediction

`--model_name=mace_mp0_medium` downloads the BCE archive:

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

**The old YAML inside the BCE archive is for inference only.** It predates the
energy/stress label-protocol fixes. Training must use the updated repository
YAML, not the packaged YAML.

## Training configuration (approximation, not the released run)

[mace_mp0_medium_finetune.yaml](mace_mp0_medium_finetune.yaml) is an approximate,
epoch-based configuration for the repository trainer. It does **not** reproduce
the original step-based harness, atom-packed batches, 500+1500-step fresh-Adam
continuation or combined checkpoint-selection rule. The original harness is not
published here; this command must not be described as the original release run.
The `1e-6` fine-tuning rate is **three orders of magnitude** below the `1e-3`
from-scratch rate.

After preparing the dataset files referenced by the YAML and the converted
weights at `./weights/mace_mp0_medium_converted.pdparams`:

```bash
python interatomic_potentials/train.py \
  -c interatomic_potentials/configs/mace/mace_mp0_medium_finetune.yaml
```

The YAML's `max_epochs: 1`, fixed 8-frame batches and energy-only best-checkpoint
tracking are trainer settings, not a recipe for recovering the historical
2000-step checkpoint or all reported numerical results.

## Citation

```bibtex
@article{batatia2024mace,
  title={MACE: Higher Order Equivariant Message Passing Neural Networks for Fast and Accurate Force Fields},
  author={Batatia, Ilyes and Kov{\'a}cs, D{\'a}vid P{\'e}ter and Simm, Gregor N. C. and Ortner, Christoph and Cs{\'a}nyi, G{\'a}bor},
  journal={arXiv preprint arXiv:2401.00096},
  year={2024}
}
```

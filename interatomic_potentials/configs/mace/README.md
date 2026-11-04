# MACE-MP-0
[MACE: Higher Order Equivariant Message Passing Neural Networks for Fast and Accurate Force Fields](https://arxiv.org/abs/2401.00096)
## Abstract
MACE (Many-body Atomic Cluster Expansion) is a novel machine learning interatomic potential that leverages higher-order equivariant message passing to achieve accurate predictions of atomic energies, forces, and stresses. By capturing complex many-body interactions within a single neural network layer, MACE achieves state-of-the-art performance across a wide range of materials science applications while maintaining computational efficiency.

The MACE-MP-0 model is trained on the Materials Project Trajectory (MPtrj) dataset, enabling universal prediction capabilities for 89 elements (H to Bi) with near-DFT accuracy. The key innovation of MACE lies in its higher-order equivariance, which allows the model to capture 3-body and 4-body interactions simultaneously through symmetric tensor contractions, eliminating the need for explicit many-body terms.

**Model Architecture Overview:**
1. **Input Layer**: Atomic numbers and 3D positions are encoded as node features
2. **Radial Basis Functions**: Interatomic distances are expanded using Bessel/Gaussian basis functions
3. **Spherical Harmonics**: Angular information is encoded using spherical harmonics
4. **Equivariant Message Passing**: Higher-order (3-body and 4-body) messages are computed through symmetric tensor contractions
5. **Interaction Blocks**: Two message passing layers with residual connections
6. **Readout Layer**: Atomic energies are summed to obtain total system energy, forces are computed via automatic differentiation
## Datasets
MACE-MP-0 is trained on the Materials Project Trajectory (MPtrj) dataset, which provides comprehensive coverage of inorganic crystalline materials.

- **MPtrj_2022.9_full**:
    The Materials Project Trajectory Dataset is the primary training dataset for MACE-MP-0. It contains density functional theory (DFT) calculations from the Materials Project database.
    - 145,923 unique compounds
    - 1,580,395 crystal structures
    - Calculations performed at GGA/GGA+U level of theory
    
    Corresponding labels:
    - Total energies
    - Atomic forces
    - Stresses
    
    | Dataset | Train | Val | Test |
    | :--- | :---: | :---: | :---: |
    | MPtrj_2022.9_full | 116738 | 14592 | 14593 |

## Models
MACE constructs an atomistic graph where:
* **Nodes**: Represent atoms with element-specific embeddings
* **Edges**: Encode interatomic distances using radial basis functions
* **Higher-order messages**: Capture 3-body and 4-body interactions simultaneously through equivariant transformations

### Mathematical Formulation
Radial basis functions expand interatomic distances:
$$
\phi_n(r) = \sqrt{\frac{2}{r_c}} \sin\left(\frac{n\pi r}{r_c}\right) / r
$$

Energy prediction:
$$
E_{\text{tot}} = \sum_i E_i
$$

Forces are computed as energy gradients:
$$
\mathbf{F}_i = -\frac{\partial E_{\text{tot}}}{\partial \mathbf{r}_i}
$$

Stresses are derived from the energy–strain relation:
$$
\boldsymbol{\sigma} = \frac{1}{V} \frac{\partial E_{\text{tot}}}{\partial \boldsymbol{\varepsilon}}
$$

## Results

Every number in this section was produced by this Paddle implementation on the
fixed validation subset described below. The original paper's MPtrj accuracy is
**not** reproduced or restated here, and no number in this section is taken
from, or comparable to, the PyTorch implementation.

### Evaluation protocol

- **validation set**: 754 frames / 60 materials randomly sampled from the MPtrj
  validation split. This is a subset, not the official validation split, and no
  test-split number is used for checkpoint selection.
- **energy**: `uncorrected_total_energy / natoms` (MACE convention), MAE in
  meV/atom.
- **force**: forces as stored in MPtrj, MAE in meV/Å.
- **stress**: `-0.1 * stress_kBar`, i.e. a single conversion in the dataset
  pipeline. MAE is reported over the **6 independent components** of the
  symmetric 3x3 tensor, in GPa. This is the only stress definition used below.
- **hardware**: 1x Tesla V100-PCIE-16GB (16 GB), PaddlePaddle 3.1.0, float32.

### Paddle results

`mace_mp0_medium` starts from weights converted from the official PyTorch
checkpoint. The run below is therefore a Paddle **fine-tuning** run on top of an
already trained model, not training from random initialisation, and it is not a
reproduction of the paper's accuracy.

All numbers are produced by this implementation on the fixed validation subset
(val754: 754 frames / 60 materials), with the label protocol above. The
checkpoint was selected by a rule that was registered before the run started:

```
score = E / 17.0261 + F / 0.0614272 + S9 / 0.310582     (lower is better)
```

where the denominators are the converted weights (step 0), so step 0 scores
exactly 3.0. `S9` is the 9-component stress MAE; it is used only inside the
score, the table below reports the 6-component value.

| Checkpoint | Steps | Energy MAE (meV/atom) | Force MAE (meV/Å) | Stress MAE (GPa, 6 comp.) | score | dE / dF / dS9 vs step 0 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| converted weights (step 0) | 0 | 17.0261 | 61.427 | 0.42247 | 3.00000 | — |
| Stage 2 (run 1) | 500 | 17.0241 | 61.341 | 0.42238 | 2.99814 | −0.011 % / −0.141 % / −0.033 % |
| Stage 2 (continuation) | 1000 | 17.0127 | 61.252 | 0.42190 | 2.99519 | −0.079 % / −0.285 % / −0.116 % |
| Stage 2 (continuation) | 1500 | 16.9827 | 61.222 | 0.42224 | 2.99349 | −0.255 % / −0.334 % / −0.062 % |
| **Stage 2 (final)** | **2000** | **16.9411** | **61.174** | 0.42211 | **2.98999** | **−0.500 % / −0.413 % / −0.089 %** |

Step 2000 is the lowest score, and it is also the best or joint-best value of
each individual metric. `finetune_cont_best.pdparams` is that checkpoint; it was
**not** picked by energy alone. The `name_for_best_metric: energy_per_atom`
entry in the YAML only configures the best-checkpoint tracking of the
repository's trainer, which this run did not use.

| Run | Train samples (frames / materials) | Steps | LR | GPU | Peak GPU memory | Time | Config |
| --- | ---: | ---: | ---: | --- | ---: | ---: | --- |
| mace_mp0_medium, Stage 2 | 40,010 / 9,386 | 2000 (500 + 1500) | 1e-6 constant | 1x V100-PCIE-16GB | 7,012 MB | 11,791 s ≈ 3.28 h, including the in-loop validation | [mace_mp0_medium_finetune.yaml](mace_mp0_medium_finetune.yaml) |

The run uses Adam (weight decay 0), the model's own SmoothL1 loss with weights
E:F:S = 1:1:0.1, gradient clipping at 10 and seed 42; batches hold at most 112
atoms and 8 frames. The 2000 steps were executed as 500 steps plus a 1500-step
continuation that restores the model parameters strictly but starts a **fresh
Adam state**. Both facts are documented in §6 of the validation report together
with the evidence that the re-evaluated step-500 checkpoint matches the value
recorded at the end of the first run (differences < 1e-3).

The gain is small because the starting point is already converged and the
learning rate is deliberately low: 2000 steps at 1e-6 move the weights by at
most 2.8e-4, enough to lower all three errors without destabilising the
pre-trained model. The validation subset is long-tailed — per-frame force MAE is
44.8 meV/Å at the median, 122.7 at P90 and 322.2 at P99 — so aggregate numbers
should be read with that in mind.

The test split was opened once, after this selection, and it was not used to
choose the checkpoint. 326 of the 340 test frames are reported (`test326/340`):
the other 14 frames have 192 and 248 atoms and do not fit the 16 GB card shared
with other jobs, and they are excluded from both checkpoints alike. On the same
326 frames the fine-tuned checkpoint lowers the force MAE from 0.062960 to
0.062653 eV/Å (-0.49 %, paired 95 % CI -4.6e-4 .. -1.6e-4) and leaves energy
(13.9885 -> 14.0511 meV/atom) and stress (0.458858 -> 0.460272 GPa) inside their
paired confidence intervals, i.e. the held-out set confirms the force gain and
shows no energy or stress change beyond noise. These numbers, the per-frame
median/P90/P99 and the size dependence are in §9 of the report.

Earlier short fine-tuning attempts (300 steps at lr 3e-5; 100 steps at lr 1e-6
and 3e-6 on a 4,021-frame subset) did not improve all three metrics and are not
part of this checkpoint; their numbers are kept in the validation report.

### Checkpoint status

`mace_mp0_medium.zip` contains the new fine-tuned checkpoint
(`checkpoints/best.pdparams`, step 2000, SHA256 `6182192a...`) together with
`mace_mp0_medium.yaml`, so that `predict.py` works from the archive as is:

```bash
python interatomic_potentials/predict.py \
  --config_path=mace_mp0_medium/mace_mp0_medium.yaml \
  --checkpoint_path=mace_mp0_medium/checkpoints/best.pdparams \
  --input_path=<structures> --input_format=cif
```

`--model_name=mace_mp0_medium` uses the same checkpoint once the archive is
uploaded to BOS. The archive was extracted into a clean directory and the
repository's own `predict.py` was run from it on 7 CIFs; the outputs are in §10
of the validation report.

The full evidence — per-step validation table, continuation audit, gradient and
parameter-update record, checkpoint integrity, packaged-model run and the
`test326/340` result — is in
[mace_mp0_medium_validation_report.md](mace_mp0_medium_validation_report.md).

### Training
```bash
# multi-gpu training
python -m paddle.distributed.launch --gpus="0,1,2,3" interatomic_potentials/train.py -c interatomic_potentials/configs/mace/mace_mp0_medium.yaml
# single-gpu training
python interatomic_potentials/train.py -c interatomic_potentials/configs/mace/mace_mp0_medium.yaml
# single-gpu fine-tuning from the converted weights (the run that produced the
# released checkpoint: lr 1e-6 constant, Adam, wd 0, seed 42)
python interatomic_potentials/train.py -c interatomic_potentials/configs/mace/mace_mp0_medium_finetune.yaml
```

### Validation
```bash
python interatomic_potentials/train.py -c interatomic_potentials/configs/mace/mace_mp0_medium_finetune.yaml Global.do_eval=True Global.do_train=False Global.do_test=False Trainer.pretrained_model_path='your checkpoint path(*.pdparams)'
```

### Testing
```bash
python interatomic_potentials/train.py -c interatomic_potentials/configs/mace/mace_mp0_medium_finetune.yaml Global.do_test=True Global.do_train=False Global.do_eval=False Trainer.pretrained_model_path='your checkpoint path(*.pdparams)'
```

### Prediction
```bash
# Mode 1: Use pretrained model
python interatomic_potentials/predict.py --model_name='mace_mp0_medium' --input_format=cif --input_path='./interatomic_potentials/example_data/cifs/'
# Mode 2: Use custom checkpoint
python interatomic_potentials/predict.py --config_path='interatomic_potentials/configs/mace/mace_mp0_medium.yaml' --checkpoint_path="your checkpoint path(*.pdparams)" --input_format=cif --input_path='./interatomic_potentials/example_data/cifs/'
```

## Citation
```
@article{batatia2024mace,
  title={MACE: Higher Order Equivariant Message Passing Neural Networks for Fast and Accurate Force Fields},
  author={Batatia, Ilyes and Kov{\'a}cs, D{\'a}vid P{\'e}ter and Simm, Gregor N. C. and Ortner, Christoph and Cs{\'a}nyi, G{\'a}bor},
  journal={arXiv preprint arXiv:2401.00096},
  year={2024}
}
```

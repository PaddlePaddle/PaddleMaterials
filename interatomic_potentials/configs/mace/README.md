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
checkpoint. Both Stage 1 runs below are therefore short Paddle **fine-tuning**
runs on top of an already trained model, not training from random
initialisation, and they are not a reproduction of the paper's accuracy.

Reference point — the converted weights before any Paddle update (0 steps):
Energy 17.03 meV/atom, Force 61.43 meV/Å, Stress 0.4225 GPa (6 components).

| Run | Dataset | Train samples (frames / materials) | Steps | LR | Energy MAE (meV/atom) | Force MAE (meV/Å) | Stress MAE (GPa, 6 comp.) | GPU | Peak GPU memory | Training time | Config | Checkpoint / log |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- | --- | --- |
| mace_mp0_medium, Stage 1-A | MPtrj train subset | 4,021 / 938 | 100 | 1e-6 | **17.01** | **61.42** | 0.4231 | 1x V100-PCIE-16GB | 8,230 MB | 259 s wall: 47.3 s fwd/bwd/opt (0.47 s/step) + 103.9 s cold data load, plus in-loop eval | [mace_mp0_medium.yaml](mace_mp0_medium.yaml) | not released (local only) |
| mace_mp0_medium, Stage 1-B | MPtrj train subset | 4,021 / 938 | 100 | 3e-6 | 16.98 | 61.41 | 0.4244 | 1x V100-PCIE-16GB | 8,230 MB | 211 s wall: 64.4 s fwd/bwd/opt (0.64 s/step) + 0.2 s data load, plus in-loop eval | [mace_mp0_medium.yaml](mace_mp0_medium.yaml) | not released (local only) |

Both runs use Adam (no weight decay), the model's own SmoothL1 loss with
weights E:F:S = 1:1:0.1, gradient clipping at 10, seed 42, and the same batch
schedule. "fwd/bwd/opt" is the per-step compute sum over the 100 optimizer
steps; "wall" also includes batch collation and the in-loop validation of the
fixed 754 frames. Stage 1-A paid 103.9 s of first-epoch data loading because
its cache was cold, which is why its wall time is longer. No Stage 1 checkpoint
has been published to BOS; `mace_mp0_medium.zip` still ships the converted
checkpoint.

The 60-material validation subset is long-tailed: per-frame force MAE is
44.8 meV/Å at the median, 122.7 at P90, 322.2 at P99 and 598.3 at the maximum,
and the worst 5% of the frames carry 20.6% of the total force error (36.9% of
the stress error). Aggregate numbers above should be read with that in mind.

### Stage 0: 300 steps on the converted weights (negative result)

A first attempt with 300 Adam steps at lr 3e-5 on a 368-frame subset updated all
29 trainable tensors (max absolute change 3.1e-3), so Paddle training does run,
but every validation metric degraded monotonically: energy 17.03 → 19.91
meV/atom, force 61.43 → 62.74 meV/Å, stress 0.4225 → 0.4401 GPa. The best
Stage 0 checkpoint is therefore step 0, i.e. the initial weights, and none is
kept. The full per-checkpoint record is preserved in the project validation
report rather than repeated here.

### Stage 1: short fine-tuning at two learning rates

Stage 1 uses a 4,021-frame subset drawn from the official train split by
`md5(mp_id)` stratified sampling (938 materials, 31.8 atoms/frame against 31.1
for the full train split, so the atom-count histogram matches the full split to
within 1 percentage point; material overlap with the validation and test
subsets is 0). Both candidates use the same initial weights, seed and batch
order; the only difference is the learning rate.

| candidate | steps | Energy MAE (meV/atom) | Force MAE (meV/Å) | Stress MAE (GPa, 6 components) | relative change vs step 0 (E / F / S) |
| --- | ---: | ---: | ---: | ---: | ---: |
| initial weights | 0 | 17.026 | 61.427 | 0.4225 | — |
| Stage 1-A, lr 1e-6 | 50 | 17.017 | 61.420 | 0.4228 | −0.05% / −0.01% / +0.09% |
| Stage 1-A, lr 1e-6 | 100 | **17.006** | **61.420** | 0.4231 | −0.12% / −0.01% / +0.14% |
| Stage 1-B, lr 3e-6 | 50 | 17.004 | 61.416 | 0.4237 | −0.13% / −0.02% / +0.28% |
| Stage 1-B, lr 3e-6 | 100 | 16.978 | 61.409 | 0.4244 | −0.28% / −0.03% / +0.45% |

Checkpoints were selected by a pre-registered rule: the sum of the E/F/S MAE
ratios against step 0 (lower is better). Stage 1-A step 100 is the only
checkpoint below the baseline (2.99993 against 3.00000), and it reloads
strictly (63/63 tensors, no missing or unexpected keys; predictions reproduce to
1.9e-6 eV/Å on forces). Stage 1-B selects step 0 because its stress error grew
faster than its energy/force error fell.

Both Stage 1 runs are small: energy improves by 0.12%, force by 0.01%, and
stress is still 0.14% worse than the initial weights. This is **not** a
reproduction of the paper accuracy, and no Stage 1 checkpoint is designated as
the final model.

### Checkpoint status

`mace_mp0_medium.zip` contains the converted checkpoint (`checkpoints/best.pdparams`)
together with `mace_mp0_medium.yaml`, so that
`predict.py --model_name=mace_mp0_medium` works out of the box. It has **not**
been replaced by any Stage 0 or Stage 1 checkpoint: none of them is better than
the initial weights on all three metrics. Stage 0 and Stage 1 checkpoints are
local artifacts and are not published.

### Training
```bash
# multi-gpu training
python -m paddle.distributed.launch --gpus="0,1,2,3" interatomic_potentials/train.py -c interatomic_potentials/configs/mace/mace_mp0_medium.yaml
# single-gpu training
python interatomic_potentials/train.py -c interatomic_potentials/configs/mace/mace_mp0_medium.yaml
```

### Validation
```bash
python interatomic_potentials/train.py -c interatomic_potentials/configs/mace/mace_mp0_medium.yaml Global.do_eval=True Global.do_train=False Global.do_test=False Trainer.pretrained_model_path='your checkpoint path(*.pdparams)'
```

### Testing
```bash
python interatomic_potentials/train.py -c interatomic_potentials/configs/mace/mace_mp0_medium.yaml Global.do_test=True Global.do_train=False Global.do_eval=False Trainer.pretrained_model_path='your checkpoint path(*.pdparams)'
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

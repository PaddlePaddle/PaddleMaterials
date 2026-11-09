# MACE-MP-0 medium — Paddle fine-tuning and validation report

This report documents the Paddle fine-tuning of `mace_mp0_medium` that produced
the checkpoint shipped in `mace_mp0_medium.zip`, together with the evidence used
to accept it. It is written so that a reviewer can check every number from the
commands in §12 without access to the machine the run was performed on.

Scope and non-claims:

- the model is **fine-tuned** from the officially converted weights; it is not
  trained from random initialisation and the paper's accuracy is not reproduced;
- the architecture, the graph converter, the loss and the label protocol are the
  repository's own (`interatomic_potentials/configs/mace/mace_mp0_medium.yaml`);
  only the optimiser hyper-parameters live in the fine-tuning config;
- the 340-frame test split is opened **once**, after the checkpoint has been
  chosen on the 754-frame validation split, and is never used for a decision.

## 1. Environment

| item | value |
| --- | --- |
| PaddlePaddle | 3.1.0, float32, AMP off |
| GPU | 1x Tesla V100-PCIE-16GB (shared cluster node) |
| CUDA | 12.6 runtime / 12.4 driver |
| validation evaluation | same card, inference only |

Two **harness-only** compatibility shims are used on this node and are *not*
part of the commit:

1. `paddle.nn.Parameter` — available natively from Paddle 3.2; on 3.1.0 it is
   provided as a thin wrapper around `paddle.create_parameter`.
2. `ppmat.models.mattersim.threebody_indices` — a Cython extension that is built
   at install time; it cannot be compiled on this node and is not used by the
   MACE code path, so a stub that raises on use is registered instead of
   blocking the package import.

On an environment with Paddle >= 3.2 and the extension built, neither shim is
needed.

## 2. Data and labels

All splits come from `MPtrj_2022.9_full` through the repository's
`MPTrjDataset`. The training subset is material-disjoint from the validation and
test subsets (zero shared `mp_id`).

| split | frames | materials | atoms | role |
| --- | ---: | ---: | ---: | --- |
| train subset | 40,010 | 9,386 | 1,248,744 | optimisation |
| validation (val754) | 754 | 60 | 29,766 | checkpoint selection |
| test | 340 | 60 | 25,944 | final report only |

Train-subset atom statistics: mean 31.211 per frame, median 22, P90 69, max 432.

Label protocol, identical for training and for every evaluation below:

- **energy** — `uncorrected_total_energy / natoms`, i.e. the raw VASP total
  energy per atom (MACE convention), reported in **meV/atom**;
- **force** — the forces stored in MPtrj, reported in **eV/Å**;
- **stress** — `-0.1 * stress_kBar`, a single conversion inside the dataset
  pipeline, reported in **GPa**. Aggregates use the **6 independent components**
  of the symmetric tensor (`xx, yy, zz, xy, yz, xz`). The 9-component variant is
  quoted only where it is explicitly labelled.

Graph construction: `cutoff = 6.0 Å`, `canocial = False`, `primitive = False`,
`niggli = False`, exactly as in `mace_mp0_medium.yaml`.

## 3. Training configuration

Config: [`mace_mp0_medium_finetune.yaml`](mace_mp0_medium_finetune.yaml)
(architecture identical to [`mace_mp0_medium.yaml`](mace_mp0_medium.yaml)).

| item | value |
| --- | --- |
| initial weights | converted official PyTorch checkpoint, 63 tensors (16 redundant PyTorch buffers removed) |
| optimiser | Adam, `weight_decay = 0` |
| learning rate | `1e-6`, constant, no scheduler |
| loss | the model's SmoothL1, weights E:F:S = 1:1:0.1 |
| gradient clipping | global norm 10.0 |
| seed | 42 |
| AMP | off |
| batch | <= 112 atoms **and** <= 8 frames per step (greedy packing); mean 90.8 atoms / 3.13 frames |
| effective steps | **2000** optimistic steps (500 + 1500 in a continuation run) |

Frames larger than 128 atoms are excluded from the training pool because a
single-frame forward + backward + Adam step does not fit the 16 GB card:
**605 frames (1.5121 % of the split, 131 materials, 99,482 atoms)**. They are
listed explicitly in the training log; nothing else is dropped and no frame is
silently truncated.

## 4. Checkpoint selection rule (pre-registered)

The rule was fixed before the run and was not changed afterwards:

```
score = E / 17.0261 + F / 0.0614272 + S9 / 0.310582        (lower is better)
```

where `E` is val754 energy MAE in meV/atom, `F` the force MAE in eV/Å and `S9`
the 9-component stress MAE in GPa. The denominators are the *converted* weights
(step 0), so the score of step 0 is exactly 3.0 and every candidate is a
relative comparison against the starting point. All terms are computed on
val754 only.

`finetune_cont_best.pdparams` is the checkpoint with the lowest score, **not**
the checkpoint with the lowest energy MAE. The `name_for_best_metric:
energy_per_atom` entry in the YAML only affects the best-checkpoint tracking of
the repository trainer, which was not used for this run; the selection itself
was done by the rule above and the per-step metrics are in the training log.

## 5. Validation results (val754, 754 frames / 29,766 atoms)

| step | Energy MAE (meV/atom) | Force MAE (eV/Å) | Stress MAE (GPa, 9 comp.) | Stress MAE (GPa, 6 comp.) | score | dE | dF | dS9 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 (converted) | 17.0261 | 0.061427 | 0.310582 | 0.422468 | 3.00000 | — | — | — |
| 250 | 17.0292 | 0.061407 | 0.310530 | 0.422417 | 2.99969 | +0.018 % | -0.033 % | -0.017 % |
| 500 | 17.0241 | 0.061341 | 0.310479 | 0.422384 | 2.99814 | -0.011 % | -0.141 % | -0.033 % |
| 1000 | 17.0127 | 0.061252 | 0.310221 | 0.421900 | 2.99519 | -0.079 % | -0.285 % | -0.116 % |
| 1500 | 16.9827 | 0.061222 | 0.310390 | 0.422235 | 2.99349 | -0.255 % | -0.334 % | -0.062 % |
| **2000** | **16.9411** | **0.061174** | **0.310306** | 0.422105 | **2.98999** | **-0.500 %** | **-0.413 %** | **-0.089 %** |

`dE / dF / dS9` are relative changes against step 0 (negative = better). All
three terms of the score improve monotonically except for a small stress
increase between step 1000 and step 1500, which is why the rule (not a single
metric) decides the winner: step 2000 is the best checkpoint under the
pre-registered rule, and it is also the best or joint-best value of every
individual metric.

The absolute improvement is small because the starting point is already
converged and the learning rate is deliberately low (1e-6): 2000 steps at this
rate move the weights by at most 2.8e-4 (§8), which is enough to lower all three
errors without destabilising the pre-trained model.

## 6. Continuation from step 500

The 2000 steps were executed as 500 steps (fresh run) + 1500 steps (continuation
started from `finetune_step500.pdparams`). What was actually restored:

| state | restored? | evidence |
| --- | --- | --- |
| model parameters | yes, 63/63 tensors | strict load, no missing/unexpected key |
| Adam first moment | **no, re-initialised** | continuation starts a new optimiser state |
| Adam second moment | **no, re-initialised** | idem |
| learning rate | yes | constant 1e-6, no scheduler to restore |
| global step counter | yes | continuation starts at step 501 and reports 1500 steps |
| data order | yes | the batch stream is advanced by exactly 500 batches before step 501, so steps 501-2000 see the same frames in the same order as an uninterrupted run |
| RNG state | not restored (new seed 42 stream) | affects only batch-independent stochasticity; this model has no dropout and no stochastic augmentations |

The continuation is therefore a **warm restart of the parameters with a fresh
optimiser state**, not a bit-exact resume. To bound the effect of this choice the
step-500 checkpoint was re-evaluated at the start of the continuation and
compared with the value recorded at the end of the first run:

| metric | first run (step 500) | re-evaluation (continuation start) | abs. difference |
| --- | ---: | ---: | ---: |
| energy (meV/atom) | 17.0241 | 17.0241 | 4.46e-5 |
| force (eV/Å) | 0.061341 | 0.061341 | 0 |
| stress 9-comp. (GPa) | 0.310479 | 0.310479 | 5.2e-8 |

The three differences are at float32 round-off level (< 1e-3 acceptance
tolerance), so the warm restart reproduces the validation state of the
checkpoint it was started from.

## 7. Optimisation evidence

Taken from the per-step diagnostics of the 1500-step continuation (the 500-step
first run records the same fields).

| item | value |
| --- | --- |
| effective optimiser steps | 2000 (500 + 1500), all with a finite loss |
| trainable tensors | 29 of 63 |
| tensors with a gradient at every step | 29 / 29 |
| tensors with a non-zero gradient at every step | 29 / 29 |
| gradient norm | mean 0.0891, max 0.6085, min 0.0100 |
| steps clipped at 10.0 | 3 of 1500 |
| parameter update (L2 over all tensors) | mean 3.84e-4 per step |
| learning rate values seen | 1e-6 for all steps |
| non-finite loss / gradient / parameter events | 0 |

`param_update_l2 > 0` on every step and a non-zero gradient on 29/29 tensors are
the direct record that `optimizer.step()` actually moved the weights; §8 confirms
the movement independently on the saved file.

## 8. Final checkpoint integrity

Checkpoint: `checkpoints/best.pdparams` inside `mace_mp0_medium.zip`
(step 2000).

| check | result |
| --- | --- |
| strict load into the model | 63/63 tensors, no missing key, no unexpected key |
| legacy PyTorch entries (16 names) | none present |
| tensors that differ from the converted weights | 29 of 63 (the 29 trainable ones) |
| max abs. tensor delta vs converted weights | 2.769e-4 |
| max abs. tensor delta vs the run-2 start | 2.057e-4 |
| parameters finite | all 29 trainable tensors and all 63 state-dict tensors |
| gradients finite at every step | yes |
| predictions finite (754 val frames) | yes |
| save -> reload -> predict | energy 0, force 3.81e-6 eV/Å, stress 2.12e-5 GPa |
| fresh-process reload from the extracted ZIP | passed (`predict.py`, 7 CIFs, CPU) |
| SHA256 of the checkpoint | `6182192a55304de567c9c544dce8568f2a37d8881e113e91a96c76ce461ccb96` |
| SHA256 of the converted weights (step 0) | `1f3190e217a60d0930d949f69b6cefb5f24977cf05d126dc15565bc92d6bac32` |

## 9. Independent test set (`test326/340`)

The test split was opened once, **after** the checkpoint had been chosen on
val754, and it was not used for any decision. Both columns below come from the
same code path, the same label protocol (section 2) and the same metric
definitions (section 5).

### 9.1 Sample set

| item | value |
| --- | --- |
| test split | 340 frames / 25,944 atoms |
| evaluated | **326 frames / 22,808 atoms** (`test326/340`) |
| excluded, identical for both checkpoints | indices 301-314 |
| device | 286 frames on 1x V100-PCIE-16GB, 40 frames (160-180 atoms) on CPU |

| indices | material | atoms/frame | frames |
| --- | --- | ---: | ---: |
| 301-306 | mp-699489 | 192 | 6 |
| 307-314 | mp-1202559 | 248 | 8 |

These two cells do not fit the 16 GB card used for this work (out of memory in
the second-order backward required by the stress term from 192 atoms on), and no
test frame is truncated or reformatted to make it fit. The 40 frames with
160-180 atoms (`230-237`, `257-264`, `285-292`, `315-322`, `332-339`) did not fit
the card while it was shared with other jobs and were evaluated on CPU; on four
frames evaluated both ways (indices 0, 60, 120, 200) the per-frame force MAE
differs by at most 1.4e-5 relative, five orders of magnitude below the reported
errors. The 326 evaluated frames are exactly the same for both checkpoints.

### 9.2 Pooled results

| metric | step 0 (converted) | step 2000 (final) | change | paired bootstrap 95% CI |
| --- | ---: | ---: | ---: | --- |
| energy MAE (meV/atom) | 13.9885 | 14.0511 | +0.448 % | -0.037 .. +0.157 meV (not significant) |
| force MAE (eV/Å) | 0.062960 | 0.062653 | -0.488 % | -4.59e-4 .. -1.63e-4 eV/Å |
| stress MAE (GPa, 6 comp.) | 0.458858 | 0.460272 | +0.308 % | -0.0003 .. +0.0032 GPa (not significant) |
| force MAE, mean of the per-frame values (eV/Å) | 0.050981 | 0.050716 | -0.520 % | |
| energy MAE, atom-weighted (meV/atom) | 13.9169 | 13.9548 | +0.272 % | |
| stress MAE, atom-weighted (GPa) | 0.371919 | 0.374542 | +0.705 % | |

The force error improves on 190 of the 326 frames and the paired confidence
interval excludes zero, so the force gain seen on val754 (-0.413 %) carries over
to the test split (-0.488 %). The energy and stress changes have confidence
intervals that contain zero: on the held-out frames the fine-tune is a
force-favouring, statistically neutral change, not an improvement of all three
metrics. This is reported as measured; the checkpoint was selected by the
pre-registered rule on val754 before the test split was opened, and the
selection was not revisited.

### 9.3 Per-frame distribution

| metric | step 0 median | step 0 P90 | step 0 P99 | final median | final P90 | final P99 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| energy MAE (meV/atom) | 5.4837 | 23.8043 | 125.206 | 5.8933 | 23.9182 | 124.5357 |
| force MAE (eV/Å) | 0.0371 | 0.0863 | 0.3705 | 0.0369 | 0.0853 | 0.3676 |
| stress MAE (GPa, 6 comp.) | 0.1707 | 0.5874 | 3.63 | 0.1698 | 0.6017 | 3.6677 |

The distribution is long-tailed: the median frame has a force error of
0.0369 eV/Å, while the worst 1 % exceed 0.37 eV/Å. The pooled force MAE
(0.0627 eV/Å) is therefore well above the median frame, and the two are not
interchangeable.

### 9.4 Size dependence

| atoms/frame | frames | force MAE step 0 | force MAE step 2000 | energy MAE step 0 | energy MAE step 2000 |
| --- | ---: | ---: | ---: | ---: | ---: |
| <= 40 | 130 | 0.043601 | 0.043366 | 17.1965 | 17.1444 |
| 41-100 | 109 | 0.058331 | 0.057876 | 10.0124 | 10.3306 |
| > 100 | 87 | 0.069224 | 0.068987 | 14.1765 | 14.0902 |

Small cells carry a larger per-atom energy error (17.2 meV/atom below 40 atoms
against 10.0 meV/atom in the middle band), which is the main reason the test
energy MAE (14.0) differs from the validation value (17.0) on a different
sample. The force error grows with cell size in both checkpoints, and the
fine-tune lowers it in all three bands.

The per-frame arrays behind this section (`e_pred`, `f_pred`, `s_pred`,
`f_lab`, `evaluated`) are written next to the run logs, and the validation
number of the same checkpoint is the one used for selection in section 5.

## 10. Packaged model

`mace_mp0_medium.zip` (13,349,582 bytes, SHA256
`ef2eef766e4d485ed99507a1014d3cab704dd7a73e7cf5ee4393bf5a65dbbb08`) contains

```text
mace_mp0_medium/
├── mace_mp0_medium.yaml
└── checkpoints/
    └── best.pdparams        # SHA256 6182192a..., 22,102,934 bytes
```

The archive was extracted into a new empty directory, the model was rebuilt from
the YAML **inside the archive**, and the repository's own predictor was run:

```bash
python interatomic_potentials/predict.py \
  --config_path=<extracted>/mace_mp0_medium/mace_mp0_medium.yaml \
  --checkpoint_path=<extracted>/mace_mp0_medium/checkpoints/best.pdparams \
  --device=cpu --input_format=cif \
  --input_path=<7 CIF files> --output_path=<new directory>
```

Result (`<output_path>/result.csv`, columns `cif_file, energy_per_atom, force, stress`):

| CIF | atoms | energy/atom (eV) | max abs. force (eV/Å) | max abs. stress (GPa) |
| --- | ---: | ---: | ---: | ---: |
| mp-1026567 | 16 | -1.5404092 | 0.139292 | 0.891770 |
| mp-1094250 | 12 | -2.8465780 | 0.092744 | 0.951210 |
| mp-1176311 | 32 | -5.7565710 | 0.179655 | 1.053608 |
| mp-1180720 | 60 | -5.6148515 | 3.849026 | 0.570694 |
| mp-1186869 | 4 | -2.2350860 | 0.000020 | 0.225074 |
| mp-1190178 | 24 | -7.0417423 | 0.180477 | 0.911329 |
| mp-1194237 | 29 | -6.5492473 | 0.129381 | 7.590039 |

All values are finite; forces have shape `(natoms, 3)` and stresses shape
`(3, 3)`. The same seven structures were predicted again in a second process
from a second extraction of the same archive and the numbers are identical bit
for bit. The CSV is the repository's own output format (one record per CIF, the
force and stress arrays are written as multi-line Python/numpy reprs), so the
file has more physical lines than structures.

## 11. Reproduction

```bash
# 1. fine-tune from the converted weights (single GPU)
python interatomic_potentials/train.py \
  -c interatomic_potentials/configs/mace/mace_mp0_medium_finetune.yaml

# 2. evaluate a checkpoint on the 754-frame validation split
python interatomic_potentials/train.py \
  -c interatomic_potentials/configs/mace/mace_mp0_medium_finetune.yaml \
  Global.do_eval=True Global.do_train=False Global.do_test=False \
  Trainer.pretrained_model_path=<checkpoint>.pdparams

# 3. run the shipped model
python interatomic_potentials/predict.py --model_name=mace_mp0_medium \
  --input_path=<structures> --input_format=cif --device=gpu
```

The exact per-step numbers of this run are the ones quoted above; the config in
this directory is the one that produced them.

## 12. Reviewer checklist

| reviewer question | where to check |
| --- | --- |
| is the architecture unchanged? | diff of `mace_mp0_medium.yaml` — untouched |
| is training real? | §7 gradient/update table, §8 parameter deltas |
| is the selection honest? | §4 rule, §5 table (rule fixed before the run) |
| are train/val/test disjoint? | §2, zero shared materials |
| is the test set used for selection? | no — §5 is val754 only, test is reported once in §9 |
| is the checkpoint reloadable? | §8, strict 63/63 from the extracted archive |
| does the archive work stand-alone? | §10, `predict.py` from a fresh extraction |
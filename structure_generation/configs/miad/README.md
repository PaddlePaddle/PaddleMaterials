# MiAD

[Mirage Atom Diffusion for De Novo Crystal Generation](https://arxiv.org/abs/2511.14426)

## Abstract

MiAD introduces mirage infusion, which lets a diffusion model change the number of atoms during crystal generation. Atoms can switch between real elements and a non-existent mirage state. The model jointly diffuses lattice matrices, fractional coordinates and atom types, using DiffCSP's CSPNet as its denoising backbone.

![MiAD Method Scheme](../../docs/miad_method_scheme.png)

---

## Model Description

### Overview

A crystal is represented by three components within its unit cell:

- atom types: $A = (a_1,\ldots,a_N)$
- fractional coordinates: $F = (f_1,\ldots,f_N),\; f_i \in [0,1)^3$
- lattice matrix: $L \in \mathbb{R}^{3 \times 3}$

MiAD defines separate forward corruption processes for $(L, F, A)$ and trains a CSPNet-based denoiser to reverse them. During sampling, mirage atoms (type 0) can be transformed into real elements or discarded, allowing the model to adjust atom counts dynamically.

### Method

#### 1) Lattice diffusion (DDPM on $L$)

The lattice is diffused with a DDPM Gaussian process:

$$
L_t = \sqrt{\bar{\alpha}_t}\,L_0 + \sqrt{1 - \bar{\alpha}_t}\,\epsilon,\quad \epsilon \sim \mathcal{N}(0, I)
$$

Lattice matrices are unbounded, so DDPM sample clipping is disabled.
`CosineDDPMScheduler` constructs the clipped cosine table and reuses the shared
DDPM implementation through `trained_betas`. The three diffusion components use
`lattice_noise_scheduler_cfg`, `coord_noise_scheduler_cfg` and
`atom_noise_scheduler_cfg`.

#### 2) Fractional-coordinate diffusion on a torus (wrapped Normal / score matching)

Fractional coordinates live on a 3D torus $[0,1)^3$. Wrapped Normal noise is used:

$$
x_t = (x_0 + \sigma(t)\,\epsilon) \bmod 1,\quad \epsilon \sim \mathcal{N}(0, I)
$$

`WrappedVESDEPCScheduler` owns the score target and predictor-corrector updates.
Its indexed predictor clamps the previous sigma to `sigma_min`, so the final
step has zero drift and diffusion.

#### 3) Periodic E(3)-aware denoiser (CSPNet + periodic Fourier features)

MiAD reuses the CSPNet denoiser on a fully connected atom graph. Periodic
Fourier features encode fractional-coordinate differences, and the network
predicts lattice noise, coordinate scores and atom-type logits.

Atom types use D3PM with a uniform transition kernel and a cosine schedule.
`D3PMUniformScheduler` implements the transition tables, forward noising,
reverse sampling, prior sampling and per-atom KL loss.

During sampling, atoms with type 0 are treated as mirage atoms. The model dynamically decides which mirage atoms should materialize into real elements, allowing the final structure to have fewer atoms than the initial maximum. This is the core innovation enabling variable-atom-count generation.
If every predicted atom is a mirage atom, the implementation returns hydrogen
at all candidate sites instead of an empty structure.

During training, every crystal is padded to `model_cfg.mirage_num_atoms` atoms (configured as 25, matching the official MiAD `add_mirage_atoms_upto25`) by appending type-0 mirage atoms with random fractional coordinates. The coordinate loss masks out mirage atoms and rescales to keep the loss scale.

The in-repo `SUNMetric` reports validity, uniqueness and novelty. Stability and
S.U.N. require energies above hull from an external CHGNet/FIRE relaxation
pipeline (fmax 0.05 eV/Angstrom, at most 1500 steps, 2023-02-07 MP hull) and are
omitted when those energies are not provided. The Stable, S.U.N. and RMSD values
below were obtained with that external pipeline. RMSD is the mean per-structure
root mean square of same-index atomic displacements before and after relaxation,
using the minimum image under the relaxed cell.

---

## Dataset Description

#### MP-20 split (download link)

| Dataset | Train | Val | Test |
| --- | --- | --- | --- |
| [MP-20](https://paddle-org.bj.bcebos.com/paddlematerial/datasets/mp_20/mp_20.zip) | 27136 | 9047 | 9046 |

---

## Results

| Model | Dataset | S.U.N. (%) | Stable (%) | Unique (%) | Novel (%) | RMSD (Å) | GPUs | Training Time | Config | Checkpoint / Log |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| miad_mp20 | mp20 | 13.90 | 19.60 | 99.40 | 76.60 | 0.229 | 1 | ~93 s/epoch | [miad_mp20.yaml](miad_mp20.yaml) | [registry](../../../ppmat/models/__init__.py) |

The checkpoint is converted from the official MP-20 epoch-8000 release. Metrics
come from 1000 structures generated with 25 initial candidates and 1000
diffusion steps, relaxed with CHGNet (MPtrj) and FIRE (fmax 0.05 eV/Angstrom, up
to 1500 steps, 2023-02-07 MP hull). All 1000 samples enter the stability/SUN
denominators and 997 relaxations converged. Raw record:
[results_20261120205054.json](../../../logs/miad_benchmark_20261120205054/results_20261120205054.json).

Training time is measured on one GPU with the released YAML: one epoch over the
full MP-20 train split (6784 steps, batch size 4) takes ~93 s, and a full run
including validation finishes in 173 s. Log:
[miad_train_timing_20261127010612.log](../../logs/miad_train_timing_20261127010612.log).

---

## Command

### Training
```bash
# multi-gpu training (example with 4 GPUs)
python -m paddle.distributed.launch --gpus="0,1,2,3" structure_generation/train.py -c structure_generation/configs/miad/miad_mp20.yaml Trainer.pretrained_model_path=null
# single-gpu training
python structure_generation/train.py -c structure_generation/configs/miad/miad_mp20.yaml Trainer.pretrained_model_path=null
```

### Validation
```bash
# Adjust program behavior on the fly using command-line parameters without modifying the configuration file directly.
# Example: --Global.do_eval=True
# The checkpoint configured in the YAML is downloaded automatically.
python structure_generation/train.py -c structure_generation/configs/miad/miad_mp20.yaml Global.do_eval=True Global.do_train=False Global.do_test=False
```

### Testing
```bash
# Evaluate the model on the test dataset.
python structure_generation/train.py -c structure_generation/configs/miad/miad_mp20.yaml Global.do_eval=False Global.do_train=False Global.do_test=True
```

### Sample
```bash
# Predict crystal structures using a trained model.
# Mode 1: Load the checkpoint registered under MODEL_REGISTRY.
# Mode 2: Use a custom configuration file and local checkpoint.
# Results are saved to the folder specified by --output_path (default: results).
# MiAD is unconditional: use by_dataloader or by_num_atoms.
# by_chemical_formula only fixes the number of atoms and does not constrain the composition.

# Mode 1: pre-trained model
python structure_generation/sample.py --model_name='miad_mp20' --weights_name='best.pdparams' --output_path='result_miad_mp20/' --mode='by_dataloader'

# Mode 2: custom config + checkpoint
python structure_generation/sample.py --config_path='structure_generation/configs/miad/miad_mp20.yaml' --checkpoint_path='./output/miad_mp20/checkpoints/best.pdparams' --output_path='result_miad_mp20/' --mode='by_dataloader'

# sample a fixed number of atoms
python structure_generation/sample.py --config_path='structure_generation/configs/miad/miad_mp20.yaml' --checkpoint_path='./output/miad_mp20/checkpoints/best.pdparams' --output_path='result_miad_num_atoms/' --mode='by_num_atoms' --num_atoms=20
```

### Evaluate
```bash
# Sample with the model and report S.U.N. metrics in one command.
python structure_generation/sample.py --config_path='structure_generation/configs/miad/miad_mp20.yaml' --checkpoint_path='./output/miad_mp20/checkpoints/best.pdparams' --output_path='result_miad_eval/' --mode='compute_metric'
```

---

## Citation
```
@article{okhotin2025miad,
  title={MiAD: Mirage Atom Diffusion for De Novo Crystal Generation},
  author={Okhotin, Andrey and Nakhodnov, Maksim and Kazeev, Nikita and Lazarev, Mikhail and Ustyuzhanin, Andrey E and Vetrov, Dmitry},
  journal={arXiv preprint arXiv:2511.14426},
  year={2025},
  url={https://arxiv.org/abs/2511.14426}
}
```

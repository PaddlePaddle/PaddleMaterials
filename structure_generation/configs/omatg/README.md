# OMatG

[Open Materials Generation with Stochastic Interpolants](https://arxiv.org/abs/2502.02582)

## Abstract

The discovery of new materials is essential for enabling technological advancements.
Computational approaches for predicting novel materials must effectively learn the
manifold of stable crystal structures within an infinite design space. We introduce
Open Materials Generation (OMatG), a unifying framework for the generative design and
discovery of inorganic crystalline materials. OMatG employs stochastic interpolants
(SI) to bridge an arbitrary base distribution to the target distribution of inorganic
crystals via a broad class of tunable stochastic processes, encompassing both
diffusion models and flow matching as special cases. In this work, we adapt the SI
framework by integrating an equivariant graph representation of crystal structures
and extending it to account for periodic boundary conditions in unit cell
representations. Additionally, we couple the SI flow over spatial coordinates and
lattice vectors with discrete flow matching for atomic species. We benchmark OMatG's
performance on two tasks: Crystal Structure Prediction (CSP) for specified
compositions, and "de novo" generation (DNG) aimed at discovering stable, novel, and
unique structures. In our ground-up implementation of OMatG, we refine and extend
both CSP and DNG metrics compared to previous works. OMatG establishes a new state of
the art in generative modeling for materials discovery, outperforming purely
flow-based and diffusion-based implementations.

![OMatG CSP](../../docs/omatg_csp_movie.gif)
![OMatG DNG](../../docs/omatg_dng_movie.gif)

---

## Model Description

### Overview

A periodic crystal unit cell is represented as:

- atom types (composition): $A = (a_1,\ldots,a_N)$
- fractional coordinates: $F = (f_1,\ldots,f_N),\; f_i \in [0,1)^3$
- lattice matrix: $L \in \mathbb{R}^{3\times 3}$

OMatG generates all three fields jointly. Instead of a diffusion noise process, it
uses stochastic interpolants between a base sample $(A_0, F_0, L_0)$ and the target
$(A_1, F_1, L_1)$:

$$
x_t = \alpha(t)\,x_0 + \beta(t)\,x_1 + \gamma(t)\,z,
\qquad
z \sim \mathcal{N}(0, I),
$$

with $\alpha(0)=\beta(1)=1$ and $\alpha(1)=\beta(0)=\gamma(0)=\gamma(1)=0$. The
network learns the velocity $b^\theta(t,x)$ and, when $\gamma(t)>0$, the denoiser
$\eta^\theta(t,x)$.

### Method

#### 1) Interpolant losses

$$
\mathcal{L}_b = \mathbb{E}_{t,z}\left[\left\|\dot{\alpha}(t)x_0 + \dot{\beta}(t)x_1
+ \dot{\gamma}(t)z - b^\theta(t,x_t)\right\|^2\right],
\qquad
\mathcal{L}_\eta = \mathbb{E}_{t,z}\left[\left\|z - \eta^\theta(t,x_t)\right\|^2\right].
$$

Species use discrete flow matching instead of a continuous interpolant:
$A_t$ starts from a mask token and each atom jumps to its target species with a rate
that grows with $t$. The CSP variant additionally keeps the input composition fixed
($A_t = A_1$).

#### 2) Integration

Both deterministic and stochastic sampling evolve the same learned fields:

$$
\text{ODE:}\quad \mathrm{d}x = (1 + st)\,b^\theta(t,x)\,\mathrm{d}t,
$$

$$
\text{SDE:}\quad \mathrm{d}x = \left[(1 + st)\,b^\theta(t,x) -
\frac{\epsilon(t)}{\gamma(t)}\,\eta^\theta(t,x)\right]\mathrm{d}t
+ \sqrt{2\epsilon(t)}\,\mathrm{d}W,
$$

where $s$ is the configured velocity-annealing factor and $\epsilon(t)$ a vanishing
noise schedule. Fractional coordinates are periodic: their interpolant follows the
shortest path on the torus and the result is wrapped back into the unit cell. The
lattice evolves in Cartesian space. Every field is updated from the same previous
state, and integrations use a fixed Euler step per interval.

#### 3) Backbone

The denoiser is the DiffCSP message-passing network with a sinusoidal time embedding,
dual output heads per field, and periodic Fourier features on fractional-coordinate
differences. Only fully connected atom graphs are supported.

---

## Dataset Description

- **MP-20**: 45,231 inorganic crystals from the Materials Project with at most 20
  atoms per unit cell; the standard benchmark for crystal generation and CSP.
- CSP uses one structure per test entry, with its composition given as input.
- DNG draws the number of atoms from the dataset distribution and generates both the
  composition and the structure.

#### MP-20 split (download link)

| Dataset | Train | Val | Test |
| --- | --- | --- | --- |
| [MP-20](https://paddle-org.bj.bcebos.com/paddlematerial/datasets/mp_20/mp_20.zip) | 27136 | 9047 | 9046 |

The dataset is loaded with the shared `MP20Dataset`; CIF strings are cached under
`data/mp_20_cache/` and reused across runs.

---

## Results

CSP on the MP-20 test split (210 integration steps):

| Model | Dataset | Match Rate (%) | RMS Dist | GPUs | Training Time | Config | Checkpoint / Log |
| --- | --- | --- | --- | --- | --- | --- | --- |
| omatg_mp20_csp_linear_ode | mp20 | 63.65 | 0.0712 | 1 | ~23.7 h (2000 epochs) | [omatg_mp20_csp.yaml](omatg_mp20_csp.yaml) | [checkpoint](https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/OMatG/omatg_mp20_csp_linear_ode.zip) |

De novo generation on MP-20 (1000 structures, 710 integration steps):

| Model | Dataset | Valid (%) | Struct. Valid (%) | COV-P (%) | COV-R (%) | METRe (%) | wDist (rho) | wDist (N-ary) | wDist (CN) | GPUs | Training Time | Config | Checkpoint / Log |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| omatg_mp20_dng_linear_sde_gamma | mp20 | 83.60 | 99.60 | 99.90 | 95.25 | 0.23 | 0.1953 | 0.0523 | 0.4954 | 1 | ~37.5 h (2000 epochs) | [omatg_mp20_dng.yaml](omatg_mp20_dng.yaml) | [checkpoint](https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/OMatG/omatg_mp20_dng_linear_sde_gamma.zip) |

Both rows are produced by the released checkpoints with the evaluation commands below
on a single GPU: CSP generates one structure for every MP-20 test entry (9046
structures, 210 integration steps), while DNG generates 1000 structures (710
integration steps) and scores them against the MP-20 test split. The CSP row uses the
shared `CSPMetric`; the DNG row uses `OMatGMetric`, cross-checked against the
upstream `ValidAtoms`, `get_cov` and `metre_rmsds` implementations. `Valid` follows
the upstream validity conjunction (SMACT composition validity, pairwise-distance
structural validity, non-degenerate polar sine and computable CrystalNN/Magpie
fingerprints), while `Struct. Valid` reports the pairwise-distance structural
validity alone, the definition used by the original generation benchmark; the MP-20
test split scores 100.0% structural validity under the same definition.
`COV-P`/`COV-R` are coverage precision/recall at the published MP-20 cutoffs, the
`wDist` columns are the density, N-ary and mean-coordination-number Wasserstein
distances, and `METRe` is the fraction of reference structures matched by at least
one generated structure. Training Time is the configured 2000-epoch schedule at the
measured per-epoch wall-clock on one GPU (CSP: 42.7 s/epoch measured over 50 epochs,
DNG: 67.5 s/epoch measured over 20 epochs; both include the per-epoch validation pass
and checkpoint writes). The configurations keep the upstream effective batch sizes
(1024 for CSP, 512 for DNG) through gradient accumulation over four micro-batches so
that training fits a 32 GB GPU.

---

## Command

### Training

```bash
# multi-gpu training (example with 4 GPUs)
python -m paddle.distributed.launch --gpus="0,1,2,3" structure_generation/train.py -c structure_generation/configs/omatg/omatg_mp20_csp.yaml
# single-gpu training
python structure_generation/train.py -c structure_generation/configs/omatg/omatg_mp20_csp.yaml

# de novo generation (atom types are generated as well)
python structure_generation/train.py -c structure_generation/configs/omatg/omatg_mp20_dng.yaml
```

### Validation

```bash
# Override configuration fields on the command line; do not edit the yaml.
python structure_generation/train.py -c structure_generation/configs/omatg/omatg_mp20_csp.yaml Global.do_eval=True Global.do_train=False Global.do_test=False Trainer.pretrained_model_path='path/to/model.pdparams'
```

### Testing

```bash
python structure_generation/train.py -c structure_generation/configs/omatg/omatg_mp20_csp.yaml Global.do_eval=False Global.do_train=False Global.do_test=True Trainer.pretrained_model_path='path/to/model.pdparams'
```

### Sample

```bash
# Mode 1: pre-trained model (downloads automatically).
python structure_generation/sample.py --model_name='omatg_mp20_csp_linear_ode' --mode='by_chemical_formula' --chemical_formula='LiMnO2' --output_path='result_omatg_mp20_csp/'
python structure_generation/sample.py --model_name='omatg_mp20_dng_linear_sde_gamma' --mode='by_num_atoms' --num_atoms=8 --output_path='result_omatg_mp20_dng/'

# Mode 2: custom config + checkpoint.
python structure_generation/sample.py --config_path='structure_generation/configs/omatg/omatg_mp20_csp.yaml' --checkpoint_path='path/to/model.pdparams' --mode='by_chemical_formula' --chemical_formula='LiMnO2' --output_path='result_omatg_mp20_csp/'
python structure_generation/sample.py --config_path='structure_generation/configs/omatg/omatg_mp20_dng.yaml' --checkpoint_path='path/to/model.pdparams' --mode='by_num_atoms' --num_atoms=8 --output_path='result_omatg_mp20_dng/'
```

### Evaluation

```bash
# CSP: one structure per MP-20 test entry, scored with the shared CSPMetric.
python structure_generation/sample.py --config_path='structure_generation/configs/omatg/omatg_mp20_csp.yaml' --checkpoint_path='path/to/model.pdparams' --mode='compute_metric' --output_path='result_omatg_mp20_csp_metric/'

# DNG: 1000 structures scored with OMatGMetric (validity, METRe, coverage).
python structure_generation/sample.py --config_path='structure_generation/configs/omatg/omatg_mp20_dng.yaml' --checkpoint_path='path/to/model.pdparams' --mode='compute_metric' --output_path='result_omatg_mp20_dng_metric/'
```

---

## Citation

```
@article{hoellmer2025open,
  title={Open Materials Generation with Stochastic Interpolants},
  author={Hoellmer, Philipp and Egg, Thomas and Martirossyan, Maya M. and Fuemmeler, Eric and Shui, Zeren and Gupta, Amit and Prakash, Pawan and Roitberg, Adrian and Liu, Mingjie and Karypis, George and Transtrum, Mark and Hennig, Richard G. and Tadmor, Ellad B. and Martiniani, Stefano},
  journal={arXiv preprint arXiv:2502.02582},
  year={2025}
}
```

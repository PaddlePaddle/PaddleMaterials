# OMatG

[Open Materials Generation with Stochastic Interpolants](https://openreview.net/forum?id=gHGrzxFujU) (ICML 2025) /
[All that structure matches does not glitter](https://openreview.net/forum?id=ig9ujp50D4) (NeurIPS 2025)

## Abstract

A state-of-the-art generative model for crystal structure prediction and *de novo* generation of inorganic crystals.
OMatG implements the [stochastic interpolants (SIs) framework](https://arxiv.org/abs/2303.08797) that bridges samples
from a base distribution to the target data distribution. A stochastic interpolant
$x_t = \alpha(t)\,x_0 + \beta(t)\,x_1 + \gamma(t)\,z$ evolves base samples $x_0$ into data samples $x_1$ over time
$t\in[0,1]$. The time-dependent density is realized via deterministic (ODE) or stochastic (SDE) sampling, requiring
only a learned velocity field $b^\theta(t, x)$ (and optionally a denoiser $z^\theta(t, x)$ for SDE).

OMatG defines a crystalline material by its unit cell ($\mathbf{L}\in\mathbb{R}^{3\times3}$), fractional coordinates
($\mathbf{X}\in[0,1)^{3\times N}$ with periodic boundary conditions), and discrete atomic species
($\mathbf{A}\in\mathbb{Z}^N_{>0}$). The SI framework handles the continuous variables $\{\mathbf{X}, \mathbf{L}\}$
while discrete species $\mathbf{A}$ are treated with [discrete flow matching](https://arxiv.org/abs/2402.04997).

Two crystal generation modes are supported:
1. **CSP** (crystal structure prediction): atomic species are fixed; only coordinates and lattice vectors evolve.
2. **DNG** (*de novo* generation): all species are masked at start and evolve together with structure.

## Model Description

### Overview

A crystalline material is represented by its unit cell ($\mathbf{L}\in\mathbb{R}^{3\times3}$),
fractional coordinates ($\mathbf{X}\in[0,1)^{3\times N}$ with periodic boundary conditions),
and discrete atomic species ($\mathbf{A}\in\mathbb{Z}^N_{>0}$).

OMatG evolves base samples $x_0$ into data samples $x_1$ through a stochastic interpolant
$x_t = \alpha(t)\,x_0 + \beta(t)\,x_1 + \gamma(t)\,z$ over $t\in[0,1]$, learned via velocity
matching. Sampling integrates the learned velocity field with a deterministic (ODE) or
stochastic (SDE) scheme. Continuous variables $\{\mathbf{X}, \mathbf{L}\}$ use the SI
framework; discrete species $\mathbf{A}$ use [discrete flow matching](https://arxiv.org/abs/2402.04997).

### Crystal structure prediction of GaTe

<img src="../../docs/omatg_csp_movie.gif" alt="csp movie" width="60%">

### *De novo* generation of Pd<sub>3</sub>Te<sub>2</sub>I<sub>3</sub>

<img src="../../docs/omatg_dng_movie.gif" alt="dng movie" width="60%">


### Method

#### 1) CSPNet backbone
Message-passing GNN with fully-connected (fc) or k-NN (knn) edge construction,
internal time embedding and dual output heads.

#### 2) Stochastic Interpolants
ODE/SDE integration with multiple interpolant schedules (Linear / Trig / EncDec /
VESBD / VPSBD); continuous variables $\{\mathbf{X}, \mathbf{L}\}$ evolve via velocity
matching.

#### 3) IndependentSampler
Base distribution sampler for position / lattice / species.

#### 4) Discrete flow matching
Masked-species evolution for *de novo* (DNG) generation.


## Dataset Description

### Included Datasets

Several standard material datasets are included as LMDB files:

| Dataset | Structures | Max Atoms | Description |
|---------|-----------:|:---------:|-------------|
| MP-20 | 45,229 | 20 | [Materials Project](https://pubs.aip.org/aip/apm/article/1/1/011002/119685) structures |
| MPTS-52 | 40,476 | 52 | [Chronological MP split](https://joss.theoj.org/papers/10.21105/joss.05618) |
| Perov-5 | 18,928 | 5 | [Perovskite dataset](https://pubs.rsc.org/en/content/articlelanding/2012/ee/c2ee22341d) |
| Alex-MP-20 | 675,204 | - | Consolidated [Alexandria](https://arxiv.org/abs/2210.00579) + MP-20 |

### Data Preparation

`OMATGStructureDataset` reads the three input formats of the original OMatG
release, so `Dataset.*.dataset.__init_params__.file_path` (and
`Sample.metrics.__init_params__.gt_file_path`) can point at any of them:

| Format | Expected content | Notes |
|--------|------------------|-------|
| `.lmdb` | one record per structure holding `cell` (3x3), `atomic_numbers` (1D) and Cartesian `pos` (Nx3); record keys must not start with `__` | the configs' default; records are read lazily and converted to fractional coordinates when `convert_to_fractional=True` |
| `.csv` | a `cif` column | fully materialized at init time (`lazy_storage` is ignored) |
| `.parquet` | `cell`, `positions` (Nx3 Cartesian) and `atomic_numbers` columns | the format the OMatG release ships on HuggingFace |

The MP-20 CSVs ship in the standard `mp_20` data package (`mp_20.zip`, md5
`73371948155aa9da609436e291142d7e`, the same package used by `MP20Dataset`) —
the only MP-20 source downloadable through this repository:

```bash
mkdir -p ./data/mp_20
wget -O ./data/mp_20/mp_20.zip https://paddle-org.bj.bcebos.com/paddlematerial/datasets/mp_20/mp_20.zip
unzip -j ./data/mp_20/mp_20.zip -d ./data/mp_20   # yields {train,val,test}.csv
```

For a quick start, train directly on those CSVs by pointing
`Dataset.*.dataset.__init_params__.file_path` at `./data/mp_20/train.csv` (and
`val.csv`, `test.csv`); `Sample.metrics` already reads `./data/mp_20/test.csv` as
the reference ground-truth set. CSV input is materialized at init time, so for the
full-scale splits (MP-20 45,229 structures, Alex-MP-20 675,204) provide the LMDB
or Parquet input of the original OMatG release instead:
`https://huggingface.co/OMatG/datasets` (Parquet, fetched by the upstream
`omg_load` command) or the `omg/data` directory of the upstream repository (LMDB).
The same three formats apply to the other datasets (MPTS-52 / Perov-5 /
Alex-MP-20).



### Supported Datasets

Pretrained weights are available for 5 dataset × mode combinations.
Training configs are provided for `mp_20` (CSP + DNG); other datasets
(`perov_5_csp`, `mpts_52_csp`, `alex_mp_20_csp`) can be used by changing
the `file_path` in the dataset section of the config.

| Dataset | Mode | Variants | Weight Index |
|---------|:----:|:--------:|--------------|
| mp_20_csp | CSP | 11 | `build_omatg_model("mp_20_csp", variant, mode="csp")` |
| mp_20_dng | DNG | 11 | `build_omatg_model("mp_20_dng", variant, mode="dng")` |
| perov_5_csp | CSP | 11 | `build_omatg_model("perov_5_csp", variant, mode="csp")` |
| mpts_52_csp | CSP | 8 | `build_omatg_model("mpts_52_csp", variant, mode="csp")` |
| alex_mp_20_csp | CSP | 11 | `build_omatg_model("alex_mp_20_csp", variant, mode="csp")` |


## Configuration Files

| File | Mode | Dataset | Description |
|------|:----:|---------|-------------|
| `omatg_mp20_csp.yaml` | CSP | MP-20 | Training (Linear-ODE, Cosine LR) + sampling |
| `omatg_mp20_dng.yaml` | DNG | MP-20 | Training (Linear-SDE, species prediction + WeightDecay) + sampling |

## Results

<table>
    <tr>
        <th nowrap>Model</th>
        <th nowrap>Dataset</th>
        <th nowrap>Mode</th>
        <th nowrap>Interpolant</th>
        <th nowrap>Config (Train / Sample)</th>
        <th nowrap>Weight</th>
    </tr>
    <tr>
        <td nowrap>mp_20_csp</td><td nowrap>MP-20</td><td nowrap>CSP</td><td nowrap>Linear-ODE</td>
        <td nowrap><a href="omatg_mp20_csp.yaml">config</a></td>
        <td nowrap><a href="https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/OMatG/omatg_mp_20_csp/Linear-ODE.pdparams">weight</a></td>
    </tr>
    <tr>
        <td nowrap>mp_20_dng</td><td nowrap>MP-20</td><td nowrap>DNG</td><td nowrap>Linear-SDE</td>
        <td nowrap><a href="omatg_mp20_dng.yaml">config</a></td>
        <td nowrap><a href="https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/OMatG/omatg_mp_20_dng/Linear-SDE-Gamma.pdparams">weight</a></td>
    </tr>
    <tr>
        <td nowrap>perov_5_csp</td><td nowrap>Perov-5</td><td nowrap>CSP</td><td nowrap>EncDec-ODE-Gamma</td>
        <td nowrap>use mp_20 CSP config, change <code>file_path</code> to <code>./data/perov_5/</code></td>
        <td nowrap><a href="https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/OMatG/omatg_perov_5_csp/EncDec-ODE-Gamma.pdparams">weight</a></td>
    </tr>
    <tr>
        <td nowrap>mpts_52_csp</td><td nowrap>MPTS-52</td><td nowrap>CSP</td><td nowrap>EncDec-ODE-Gamma</td>
        <td nowrap>use mp_20 CSP config, change <code>file_path</code> to <code>./data/mpts_52/</code></td>
        <td nowrap><a href="https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/OMatG/omatg_mpts_52_csp/EncDec-ODE-Gamma.pdparams">weight</a></td>
    </tr>
    <tr>
        <td nowrap>alex_mp_20_csp</td><td nowrap>Alex-MP-20</td><td nowrap>CSP</td><td nowrap>EncDec-ODE-Gamma</td>
        <td nowrap>use mp_20 CSP config, change <code>file_path</code> to <code>./data/alex_mp_20/</code></td>
        <td nowrap><a href="https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/OMatG/omatg_alex_mp_20_csp/EncDec-ODE-Gamma.pdparams">weight</a></td>
    </tr>
</table>

## Pretrained Weights

Pre-trained weights hosted on Baidu BOS (52 files, 5 dataset x variant tables below). Use `build_omatg_model(dataset, variant, mode=...)`
for automatic download. Cached to `~/.paddlemat/weights/omatg_{dataset}/` after first download.

```python
from ppmat.models.omatg import build_omatg_model
model = build_omatg_model("mp_20_csp", "encdec_ode_gamma", mode="csp")  # CSP
model = build_omatg_model("mp_20_dng", "encdec_ode_gamma", mode="dng")  # DNG
```

The two headline weights from the Results table are also registered in
`MODEL_REGISTRY` (standard `<key>.zip` packages with config + `best.pdparams`),
so `build_model_from_name` and `sample.py --model_name` work out of the box:

- `omatg_mp20_csp_linear_ode` (CSP / Linear-ODE)
- `omatg_mp20_dng_linear_sde_gamma` (DNG / Linear-SDE-Gamma)

### Perov-5

| Variant | Weight |
|---------|--------|
| encdec_ode_gamma | [EncDec-ODE-Gamma.pdparams](https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/OMatG/omatg_perov_5_csp/EncDec-ODE-Gamma.pdparams) |
| encdec_sde_gamma | [EncDec-SDE-Gamma.pdparams](https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/OMatG/omatg_perov_5_csp/EncDec-SDE-Gamma.pdparams) |
| linear_ode | [Linear-ODE.pdparams](https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/OMatG/omatg_perov_5_csp/Linear-ODE.pdparams) |
| linear_ode_gamma | [Linear-ODE-Gamma.pdparams](https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/OMatG/omatg_perov_5_csp/Linear-ODE-Gamma.pdparams) |
| linear_sde_gamma | [Linear-SDE-Gamma.pdparams](https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/OMatG/omatg_perov_5_csp/Linear-SDE-Gamma.pdparams) |
| trig_ode | [Trig-ODE.pdparams](https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/OMatG/omatg_perov_5_csp/Trig-ODE.pdparams) |
| trig_ode_gamma | [Trig-ODE-Gamma.pdparams](https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/OMatG/omatg_perov_5_csp/Trig-ODE-Gamma.pdparams) |
| trig_sde_gamma | [Trig-SDE-Gamma.pdparams](https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/OMatG/omatg_perov_5_csp/Trig-SDE-Gamma.pdparams) |
| vesbd_ode | [VESBD-ODE.pdparams](https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/OMatG/omatg_perov_5_csp/VESBD-ODE.pdparams) |
| vpsbd_ode | [VPSBD-ODE.pdparams](https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/OMatG/omatg_perov_5_csp/VPSBD-ODE.pdparams) |
| vpsbd_sde | [VPSBD-SDE.pdparams](https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/OMatG/omatg_perov_5_csp/VPSBD-SDE.pdparams) |

### MPTS-52

| Variant | Weight |
|---------|--------|
| encdec_ode_gamma | [EncDec-ODE-Gamma.pdparams](https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/OMatG/omatg_mpts_52_csp/EncDec-ODE-Gamma.pdparams) |
| encdec_sde_gamma | [EncDec-SDE-Gamma.pdparams](https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/OMatG/omatg_mpts_52_csp/EncDec-SDE-Gamma.pdparams) |
| linear_ode | [Linear-ODE.pdparams](https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/OMatG/omatg_mpts_52_csp/Linear-ODE.pdparams) |
| linear_ode_gamma | [Linear-ODE-Gamma.pdparams](https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/OMatG/omatg_mpts_52_csp/Linear-ODE-Gamma.pdparams) |
| linear_sde_gamma | [Linear-SDE-Gamma.pdparams](https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/OMatG/omatg_mpts_52_csp/Linear-SDE-Gamma.pdparams) |
| trig_ode | [Trig-ODE.pdparams](https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/OMatG/omatg_mpts_52_csp/Trig-ODE.pdparams) |
| trig_ode_gamma | [Trig-ODE-Gamma.pdparams](https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/OMatG/omatg_mpts_52_csp/Trig-ODE-Gamma.pdparams) |
| trig_sde_gamma | [Trig-SDE-Gamma.pdparams](https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/OMatG/omatg_mpts_52_csp/Trig-SDE-Gamma.pdparams) |

### MP-20 (DNG)

| Variant | Weight |
|---------|--------|
| encdec_ode_gamma | [EncDec-ODE-Gamma.pdparams](https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/OMatG/omatg_mp_20_dng/EncDec-ODE-Gamma.pdparams) |
| encdec_sde_gamma | [EncDec-SDE-Gamma.pdparams](https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/OMatG/omatg_mp_20_dng/EncDec-SDE-Gamma.pdparams) |
| linear_ode | [Linear-ODE.pdparams](https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/OMatG/omatg_mp_20_dng/Linear-ODE.pdparams) |
| linear_ode_gamma | [Linear-ODE-Gamma.pdparams](https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/OMatG/omatg_mp_20_dng/Linear-ODE-Gamma.pdparams) |
| linear_sde_gamma | [Linear-SDE-Gamma.pdparams](https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/OMatG/omatg_mp_20_dng/Linear-SDE-Gamma.pdparams) |
| trig_ode | [Trig-ODE.pdparams](https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/OMatG/omatg_mp_20_dng/Trig-ODE.pdparams) |
| trig_ode_gamma | [Trig-ODE-Gamma.pdparams](https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/OMatG/omatg_mp_20_dng/Trig-ODE-Gamma.pdparams) |
| trig_sde_gamma | [Trig-SDE-Gamma.pdparams](https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/OMatG/omatg_mp_20_dng/Trig-SDE-Gamma.pdparams) |
| vesbd_ode | [VESBD-ODE.pdparams](https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/OMatG/omatg_mp_20_dng/VESBD-ODE.pdparams) |
| vpsbd_ode | [VPSBD-ODE.pdparams](https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/OMatG/omatg_mp_20_dng/VPSBD-ODE.pdparams) |
| vpsbd_sde | [VPSBD-SDE.pdparams](https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/OMatG/omatg_mp_20_dng/VPSBD-SDE.pdparams) |

### MP-20 (CSP)

| Variant | Weight |
|---------|--------|
| encdec_ode_gamma | [EncDec-ODE-Gamma.pdparams](https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/OMatG/omatg_mp_20_csp/EncDec-ODE-Gamma.pdparams) |
| encdec_sde_gamma | [EncDec-SDE-Gamma.pdparams](https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/OMatG/omatg_mp_20_csp/EncDec-SDE-Gamma.pdparams) |
| linear_ode | [Linear-ODE.pdparams](https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/OMatG/omatg_mp_20_csp/Linear-ODE.pdparams) |
| linear_ode_gamma | [Linear-ODE-Gamma.pdparams](https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/OMatG/omatg_mp_20_csp/Linear-ODE-Gamma.pdparams) |
| linear_sde_gamma | [Linear-SDE-Gamma.pdparams](https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/OMatG/omatg_mp_20_csp/Linear-SDE-Gamma.pdparams) |
| trig_ode | [Trig-ODE.pdparams](https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/OMatG/omatg_mp_20_csp/Trig-ODE.pdparams) |
| trig_ode_gamma | [Trig-ODE-Gamma.pdparams](https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/OMatG/omatg_mp_20_csp/Trig-ODE-Gamma.pdparams) |
| trig_sde_gamma | [Trig-SDE-Gamma.pdparams](https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/OMatG/omatg_mp_20_csp/Trig-SDE-Gamma.pdparams) |
| vesbd_ode | [VESBD-ODE.pdparams](https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/OMatG/omatg_mp_20_csp/VESBD-ODE.pdparams) |
| vpsbd_ode | [VPSBD-ODE.pdparams](https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/OMatG/omatg_mp_20_csp/VPSBD-ODE.pdparams) |
| vpsbd_sde | [VPSBD-SDE.pdparams](https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/OMatG/omatg_mp_20_csp/VPSBD-SDE.pdparams) |

### Alex-MP-20

| Variant | Weight |
|---------|--------|
| encdec_ode_gamma | [EncDec-ODE-Gamma.pdparams](https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/OMatG/omatg_alex_mp_20_csp/EncDec-ODE-Gamma.pdparams) |
| encdec_sde_gamma | [EncDec-SDE-Gamma.pdparams](https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/OMatG/omatg_alex_mp_20_csp/EncDec-SDE-Gamma.pdparams) |
| linear_ode | [Linear-ODE.pdparams](https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/OMatG/omatg_alex_mp_20_csp/Linear-ODE.pdparams) |
| linear_ode_gamma | [Linear-ODE-Gamma.pdparams](https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/OMatG/omatg_alex_mp_20_csp/Linear-ODE-Gamma.pdparams) |
| linear_sde_gamma | [Linear-SDE-Gamma.pdparams](https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/OMatG/omatg_alex_mp_20_csp/Linear-SDE-Gamma.pdparams) |
| trig_ode | [Trig-ODE.pdparams](https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/OMatG/omatg_alex_mp_20_csp/Trig-ODE.pdparams) |
| trig_ode_gamma | [Trig-ODE-Gamma.pdparams](https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/OMatG/omatg_alex_mp_20_csp/Trig-ODE-Gamma.pdparams) |
| trig_sde_gamma | [Trig-SDE-Gamma.pdparams](https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/OMatG/omatg_alex_mp_20_csp/Trig-SDE-Gamma.pdparams) |
| vesbd_ode | [VESBD-ODE.pdparams](https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/OMatG/omatg_alex_mp_20_csp/VESBD-ODE.pdparams) |
| vpsbd_ode | [VPSBD-ODE.pdparams](https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/OMatG/omatg_alex_mp_20_csp/VPSBD-ODE.pdparams) |
| vpsbd_sde | [VPSBD-SDE.pdparams](https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/OMatG/omatg_alex_mp_20_csp/VPSBD-SDE.pdparams) |

## Command

### Training

```bash
# Smoke training (1 epoch, small batch; periodic in-training eval still
# runs per eval_freq, Global.do_eval only skips the final standalone eval)
python structure_generation/train.py \
    -c structure_generation/configs/omatg/omatg_mp20_csp.yaml \
    Trainer.max_epochs=1 \
    Global.do_eval=False \
    Dataset.train.sampler.__init_params__.batch_size=32 \
    Dataset.val.sampler.__init_params__.batch_size=32

# Full training (requires data/mp_20/ dataset)
python structure_generation/train.py \
    -c structure_generation/configs/omatg/omatg_mp20_csp.yaml \
    Trainer.max_epochs=500 \
    Trainer.output_dir=./output/omatg_mp20_csp
```

### Fine-tuning

Load the released weights (or a previous run's checkpoint) and continue
training:

```bash
# Fine-tune from the released CSP Linear-ODE weights
python structure_generation/train.py \
    -c structure_generation/configs/omatg/omatg_mp20_csp.yaml \
    Trainer.pretrained_model_path=https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/OMatG/omatg_mp_20_csp/Linear-ODE.pdparams \
    Trainer.max_epochs=50 \
    Trainer.output_dir=./output/omatg_mp20_csp_finetune
```

### Validation

```bash
# NOTE: output_dir is automatically suffixed with _t_<timestamp>_s_<seed>
# at runtime (see Trainer docs), so point to the suffixed directory here.
# Evaluate on the validation split using a saved checkpoint
python structure_generation/train.py \
    -c structure_generation/configs/omatg/omatg_mp20_csp.yaml \
    Global.do_train=False \
    Global.do_eval=True \
    Trainer.pretrained_model_path=./output/omatg_mp20_csp_t_<timestamp>_s_42/checkpoints
```

### Testing

```bash
# Evaluate on the test split with a saved checkpoint
python structure_generation/train.py \
    -c structure_generation/configs/omatg/omatg_mp20_csp.yaml \
    Global.do_train=False \
    Global.do_test=True \
    Global.do_eval=False \
    Trainer.pretrained_model_path=./output/omatg_mp20_csp_t_<timestamp>_s_42/checkpoints
```

### Unit Tests

The OMatG unit tests live in `test/omatg/test_omatg.py` and cover the core
end-to-end flows (project convention 4.2): CSP/DNG forward and loss, the
SI velocity-matching training path (Linear-ODE and DNG Linear-SDE), the
config-driven build (YAML -> model -> forward -> sample), SI sampling
stability, `OMatGMetric` (match/dng modes) and the dataset-to-collator-to-
model pipeline (CSV -> `OMATGStructureDataset` -> `DefaultCollator` (with
`ConcatData`) -> SI forward).

```bash
python -m pytest test/omatg/test_omatg.py -v
```

### Sample

```bash
# Sample with a registered MODEL_REGISTRY package (auto-download).
python structure_generation/sample.py \
    --model_name omatg_mp20_csp_linear_ode \
    --mode by_num_atoms \
    --num_atoms 8 \
    --output_path ./outputs/omatg_samples
```

`--mode by_num_atoms` fixes only the atom count: without a chemical formula the
species are drawn uniformly from the model's atomic-number range, and a CSP config
then mirrors those sampled species. Use `--mode by_chemical_formula` to sample a
fixed composition.

```bash
# Sample by number of atoms (with local checkpoint)
python structure_generation/sample.py \
    --config_path structure_generation/configs/omatg/omatg_mp20_csp.yaml \
    --checkpoint_path ./output/omatg_mp20_csp/checkpoints/best.pdparams \
    --mode by_num_atoms \
    --num_atoms 8 \
    --output_path ./outputs/omatg_samples

# Sample by chemical formula (with local checkpoint)
python structure_generation/sample.py \
    --config_path structure_generation/configs/omatg/omatg_mp20_csp.yaml \
    --checkpoint_path ./output/omatg_mp20_csp/checkpoints/best.pdparams \
    --mode by_chemical_formula \
    --chemical_formula LiMnO2 \
    --output_path ./outputs/omatg_samples

# Sample by dataloader (with local checkpoint)
python structure_generation/sample.py \
    --config_path structure_generation/configs/omatg/omatg_mp20_csp.yaml \
    --checkpoint_path ./output/omatg_mp20_csp/checkpoints/best.pdparams \
    --mode by_dataloader \
    --output_path ./outputs/omatg_samples
```

## Reproduction Status

This migration reproduces the OMatG **encoder forward pass** against the released
checkpoints. All 52 weight files (5 datasets × variants) load and run a forward pass with
per-attribute MAE well below `1e-6` relative to the PyTorch reference. The trained-loss /
sampling-defect numbers of the paper (`match_rate` / COV / METRe / `dng_eval`) have **not
yet been re-evaluated end-to-end for this migration**, and doing so requires the datasets
prepared in the preceding section. The pretrained-weight path here reproduces the model
architecture, weight loading and forward behaviour; paper-level sampling metrics remain to
be validated and reported.

To run the metrics path today, provide a ground-truth set of generated structures and target
the `--mode compute_metric` sampling config, which computes CSP `match_rate` / DNG
`valid_rate`, Wasserstein, COV and `dng_eval` via `OMatGMetric`.

## Citation

```bibtex
@inproceedings{
    hoellmer2025,
    title={Open Materials Generation with Stochastic Interpolants},
    author={Philipp H{\"o}llmer and Thomas Egg and Maya Martirossyan and Eric
    Fuemmeler and Zeren Shui and Amit Gupta and Pawan Prakash and Adrian
    Roitberg and Mingjie Liu and George Karypis and Mark Transtrum and Richard
    Hennig and Ellad B. Tadmor and Stefano Martiniani},
    booktitle={Forty-second International Conference on Machine Learning},
    year={2025},
    url={https://openreview.net/forum?id=gHGrzxFujU},
    archivePrefix={arXiv},
    eprint={2502.02582},
    primaryClass={cs.LG},
}
```

```bibtex
@inproceedings{
    martirossyan2025,
    title={All that structure matches does not glitter},
    author={Maya Martirossyan and Thomas Egg and Philipp H{\"o}llmer
    and George Karypis and Mark Transtrum and Adrian Roitberg
    and Mingjie Liu and Richard Hennig and Ellad B. Tadmor and Stefano Martiniani},
    booktitle={Thirty-Ninth Annual Conference on Neural Information Processing Systems},
    year={2025},
    url={https://openreview.net/forum?id=ig9ujp50D4},
    archivePrefix={arXiv},
    eprint={2509.12178},
    primaryClass={cs.LG},
}
```

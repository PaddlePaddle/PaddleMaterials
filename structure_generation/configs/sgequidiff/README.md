# SGEquiDiff

[Space Group Equivariant Crystal Diffusion](https://arxiv.org/abs/2505.10994)

## Abstract

Crystal structures are governed by space group symmetry, which constrains the positions of atoms within the unit cell and determines the repeating patterns throughout three-dimensional space. Existing generative models for crystals often ignore these fundamental symmetry constraints, producing structures that violate crystallographic rules or fail to explore the full diversity of symmetry-distinct configurations. In this paper, we introduce SGEquiDiff, a hierarchical generative model that explicitly incorporates space group equivariance at every stage of crystal generation. Our approach sequentially samples space groups, lattice parameters subject to Bravais lattice constraints, elements and Wyckoff positions via an autoregressive transformer, and finally fractional coordinates using a score-based diffusion process on the asymmetric unit. Equivariance of the coordinate diffusion model is achieved through symmetrization: averaging inverse-transformed predictions over all space group operations. Experimental results on the MP-20 and MPTS-52 benchmarks demonstrate that SGEquiDiff generates structurally valid, diverse, and novel crystal structures while faithfully reproducing the space group and Wyckoff position distributions of the training data.

![SGEquiDiff Overview](../../docs/sgequidiff_fig_overview.png)

---

## Model Description

### Overview

A crystal structure in the asymmetric unit (ASU) representation is described by:
- space group index: $sg \in \{0, \ldots, 229\}$ (one of 230 crystallographic space groups)
- lattice lengths: $(a, b, c) \in \mathbb{R}^3_{>0}$, and lattice angles: $(\alpha, \beta, \gamma) \in (0°, 180°)^3$
- element indices: $E = (e_1, \ldots, e_N)$ for $N$ ASU atoms
- Wyckoff indices: $W = (w_1, \ldots, w_N)$ specifying the Wyckoff position of each ASU atom
- fractional coordinates: $F = (f_1, \ldots, f_N),\; f_i \in [0,1)^3$ within the ASU

### Method

`SGEQuiDiff` orchestrates four specialized modules in sequence:

1. **Space group sampling**: a learnable categorical distribution over all 230 space groups, trained with negative log-likelihood.
2. **Lattice parameter sampling** (`TelescopingDiscreteLatticeSampler`): lattice parameters conditioned on the space group, with Bravais lattice constraints enforced by pre-computed linear transforms and angle offsets; noisy training targets are produced by rejection sampling.
3. **Element and Wyckoff position sampling** (`WyckoffElementTransformer`, hidden dim 256, 4 layers, 2 heads): autoregressive generation of (element, Wyckoff) pairs conditioned on the space group and lattice, until a termination token is emitted.
4. **Fractional coordinate diffusion** (`EquivariantDiffusionModel`): a variance-exploding SDE on the ASU, trained with score matching and sampled with a predictor-corrector scheme using Wyckoff-projected Gaussian noise.

The total training loss combines the negative log-likelihood terms of the discrete components with the score-matching loss, balanced by gradient reweighting:

$$
\mathcal{L} = w_{sg}\,\mathcal{L}_{sg} + w_{L}\,\mathcal{L}_{L} + w_{EW}\,\mathcal{L}_{EW} + w_{F}\,\mathcal{L}_{F}
$$

**Space group equivariance via symmetrization.** The score network symmetrizes a non-equivariant backbone $f_\theta$ over all space group operations $\{(A_g, t_g)\}_{g \in G}$:

$$
\hat{s}_\theta(x) = \frac{1}{|G|} \sum_{g \in G} A_g^{-1}\, f_\theta(A_g x + t_g)
$$

Wyckoff shape constraints are additionally enforced by projecting the score onto pre-computed noise projection matrices for each Wyckoff site.

Three backbone architectures are supported for $f_\theta$, selected via `model_type`:
- **GNN** (default): message-passing network with periodic boundary conditions, plane-wave Fourier features on fractional differences, and variance-preserving aggregation.
- **TorusMLP**: an MLP over plane-wave Fourier features of fractional coordinates with time embeddings.
- **CSPNet**: a DiffCSP-style fully-connected architecture with sinusoidal distance embeddings, included for comparison.

---

## Dataset Description

- **MP-20**: 45,231 inorganic crystals from the Materials Project with at most 20 atoms per unit cell; widely used for crystal generation benchmarking.
- **MPTS-52 (Materials Project Time Split)**: 40,476 crystals with up to 52 atoms per cell; chronological split for evaluating temporal generalization.

Each sample stores the ASU fields consumed by `AsymmetricUnitDataset`: `space_group_number`, `conventional_lattice_lengths` $(3,)$, `conventional_lattice_angles` $(3,)$, `element_indices` $(N,)$, `wyckoff_indices` $(N,)$, and `conventional_frac_coords` $(N, 3)$.

#### Dataset preparation

Both datasets are provided as preprocessed NPZ archives (`train/val/test.npz`).
Each config selects the dataset class per source (`MP20ASUDataset` for
MP-20, `MPTS52ASUDataset` for MPTS-52) and sets an explicit `path` per split
(defaulting to `data/data/<name>/<split>.npz`); when the path does not exist,
the archive is fetched automatically through the unified download pipeline
(MD5-verified, cached under `~/.paddlemat/datasets`). Point `path` at a custom
location to use your own copy.
---

| Model | Dataset | Config | Checkpoint |
| --- | --- | --- | --- |
| sgequidiff | mp_20 | [sgequidiff_mp20.yaml](sgequidiff_mp20.yaml) | [sgequidiff_mp20.zip](https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/SGEquiDiff/sgequidiff_mp20.zip) |
| sgequidiff | mpts_52 | [sgequidiff_mpts_52.yaml](sgequidiff_mpts_52.yaml) | [sgequidiff_mpts_52.zip](https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/SGEquiDiff/sgequidiff_mpts_52.zip) |

---

## Command

### Training
```bash
python structure_generation/train.py -c structure_generation/configs/sgequidiff/sgequidiff_mp20.yaml
```

### Evaluation
```bash
python structure_generation/train.py -c structure_generation/configs/sgequidiff/sgequidiff_mp20.yaml Global.do_train=False Global.do_eval=True Trainer.pretrained_model_path=/path/to/checkpoints/best.pdparams
```

Note: this command reports the validation loss. Generation-quality metrics
(validity / uniqueness / novelty / coverage / distances, computed by
`SGEQuiDiffMetric`) are produced by the Generation-Quality Evaluation command
below.

### Sample
```bash
# Mode 1: use a registered pretrained model (weights downloaded automatically)
python structure_generation/sample.py --model_name='sgequidiff_mp20' --weights_name='best.pdparams' --mode='by_dataloader' --output_path='./sgequidiff_samples'

# Mode 2: use a local config and checkpoint
python structure_generation/sample.py --config_path=structure_generation/configs/sgequidiff/sgequidiff_mp20.yaml --checkpoint_path=/path/to/checkpoints/best.pdparams --mode=by_dataloader --output_path=./sgequidiff_samples
```

Note: SGEQuiDiff is an unconditional generator and only supports
`--mode=by_dataloader`. The requested condition in `--mode=by_num_atoms` and
`--mode=by_condition` is rejected with a `NotImplementedError`; the formula
passed to `--mode=by_chemical_formula` is ignored, so do not use it.

### Generation-Quality Evaluation
```bash
# Mode 1: use a registered pretrained model (weights downloaded automatically)
python structure_generation/sample.py --model_name='sgequidiff_mp20' --mode='compute_metric' --output_path='./sgequidiff_samples'

# Mode 2: use a local config and checkpoint
python structure_generation/sample.py --config_path=structure_generation/configs/sgequidiff/sgequidiff_mp20.yaml --checkpoint_path=/path/to/checkpoints/best.pdparams --mode=compute_metric --output_path=./sgequidiff_samples
```

---

## Citation
```
@misc{chang2025spacegroupequivariantcrystal,
      title={Space Group Equivariant Crystal Diffusion},
      author={Rees Chang and Angela Pak and Alex Guerra and Ni Zhan and Nick Richardson and Elif Ertekin and Ryan P. Adams},
      year={2025},
      eprint={2505.10994},
      archivePrefix={arXiv},
      url={https://arxiv.org/abs/2505.10994},
}
```

# Chemeleon2

[Guiding Generative Models to Uncover Diverse and Novel Crystals via Reinforcement Learning](https://arxiv.org/abs/2511.07158)

## Abstract

Chemeleon2 combines a variational autoencoder (VAE), latent diffusion and
reinforcement learning to generate periodic crystal structures. The VAE learns
continuous representations of crystal geometry and composition, a diffusion
Transformer learns to generate these representations, and Group Relative Policy
Optimization (GRPO) adapts the generator using rewards for novelty, diversity or
material properties. PaddleMaterials provides all three stages through the
unified `Chemeleon2` model.

![Chemeleon2 architecture and GRPO fine-tuning](../../docs/chemeleon2_arch.png)

## Model Description

### Overview

A crystal is represented by atomic numbers $A$, a lattice matrix
$L \in \mathbb{R}^{3 \times 3}$ and fractional coordinates
$X \in [0,1)^{N \times 3}$. The VAE maps these inputs to eight latent values
per atom. Generation denoises a latent sample and decodes it into $(A,L,X)$;
the requested atom count is fixed while atom types and geometry are generated.

| Stage | Objective | Trainable components | Configuration |
| --- | --- | --- | --- |
| VAE | Crystal reconstruction and latent regularization | Encoder and decoder | [chemeleon2_mp20_vae.yaml](chemeleon2_mp20_vae.yaml) |
| Diffusion | Noise prediction in VAE latent space | Diffusion Transformer; VAE frozen | [chemeleon2_mp20_ldm.yaml](chemeleon2_mp20_ldm.yaml) |
| RL | Reward-guided generation with GRPO | Diffusion Transformer; VAE and condition embeddings frozen | [chemeleon2_mp20_rl.yaml](chemeleon2_mp20_rl.yaml) |

The `stage` configuration selects `vae`, `diffusion` or `rl`. VAE training alone
does not provide de novo sampling. The released vocabulary supports atomic
numbers 1 through 99; index zero is reserved for corruption.

### Method

#### 1) Crystal representation learning

The VAE encoder produces a Gaussian posterior for each atom's latent vector.
The decoder reconstructs atomic numbers, fractional coordinates, lattice lengths
normalized by $N^{1/3}$, and lattice angles. Training combines atom classification,
coordinate and lattice reconstruction losses, and KL regularization. Translation,
rotation and coordinate/type corruption can be enabled during training.

#### 2) Latent diffusion and sampling

The diffusion stage freezes the VAE. For a posterior sample $z$ and the stored
latent scale $s$, the normalized representation is $z_0=z/s$. Gaussian corruption
is applied in this continuous latent space:

$$
z_t = \sqrt{\bar{\alpha}_t}\,z_0
    + \sqrt{1-\bar{\alpha}_t}\,\epsilon,
\qquad \epsilon \sim \mathcal{N}(0,I).
$$

The diffusion Transformer predicts $\epsilon$ with a mean squared error loss over
unpadded atomic latents. Sampling reverses this process, rescales the generated
latents by $s$ and decodes the crystal. DDIM with 50 inference steps is the
default; fixed-variance DDPM is also supported. Atom types and fractional
coordinates are reconstructed by the VAE rather than diffused separately.

Conditional models use `condition_names` and `property_embeddings_adapt_cfg`.
The supported encoders are `ValueEncoder`, `CategoricalEncoder`,
`CompositionEncoder` and `ChemicalSystemEncoder`. `p_unconditional` controls
joint condition dropout, and `cfg_scale` controls classifier-free guidance.
The supplied LDM checkpoint is unconditional. Fixed-composition sampling via
`by_chemical_formula` is unsupported.

#### 3) Reward-guided fine-tuning

As illustrated above, the diffusion model acts as the policy and decoded
crystals are scored by the reward function. GRPO normalizes rewards within
rollout groups and optimizes a clipped policy objective with KL and entropy
regularization. RL uses stochastic DDIM trajectories (`eta > 0`); only the
denoiser is updated.

| Reward | Purpose |
| --- | --- |
| `CreativityReward` | Encourage unique and novel crystals |
| `StructureDiversityReward` | Encourage geometric diversity using frozen-VAE features |
| `CompositionDiversityReward` | Encourage compositional diversity using frozen-VAE features |
| `EnergyReward` | Penalize energy above the convex hull |
| `CustomReward` | Provide an extension point for custom objectives |

The supplied RL configuration combines creativity and the two diversity rewards.
Energy rewards require an ASE calculator supplied through
`build_model(..., energy_calculator=calculator)` and a gzip-compressed pickle of
a pymatgen phase diagram configured by `phase_diagram_path`. They are not enabled
by default.

The port uses an epsilon-only diffusion objective and fixed DDPM variance;
`learn_sigma=True` retains the released output layout without learning variance.
For creativity rewards, AMD distances are used when available; otherwise the
implementation uses a lattice-parameter distance. Diversity reference features
are computed with the frozen VAE. These choices differ from upstream settings,
so learned-range DDPM equivalence and upstream RL benchmark results are not
claimed. CINN support for diffusion-stage denoising is described in
[the execution-backend documentation](../../../docs/cinn.md).

#### 4) Evaluation metrics

`CrystalVUNMetric` reports three population statistics:

- **Validity:** the fraction of all generated crystals passing composition and
  geometric validity checks.
- **Uniqueness:** the fraction of distinct structures among valid crystals,
  including comparisons across generated batches.
- **Novelty:** the fraction of unique valid structures that match no structure
  in the reference population. This result is omitted without a reference set.

These metrics do not measure thermodynamic stability or S.U.N.

## Dataset Description

| Dataset | Train | Val | Test |
| --- | --- | --- | --- |
| [MP20](https://paddle-org.bj.bcebos.com/paddlematerial/datasets/mp_20/mp_20.zip) | 27136 | 9047 | 9046 |

### RL reference population

RL requires `data/mp_20/novelty_reference.pkl` for novelty and diversity rewards.
This file is not included in the MP20 archive. Prepare it from the training
split, preserving row order and excluding validation and test structures:

```bash
python - <<'PY'
import pickle
from pathlib import Path

import pandas as pd

from ppmat.datasets.build_structure import BuildStructure
from ppmat.datasets.mp20_dataset import MP20Dataset
from ppmat.utils import download

source = Path("data/mp_20/train.csv")
if not source.is_file():
    root = download.get_datasets_path_from_url(MP20Dataset.url, MP20Dataset.md5)
    source = Path(root) / MP20Dataset.name / source.name
structures = [
    BuildStructure.build_one(cif, "cif_str", niggli=False, canocial=False)
    for cif in pd.read_csv(source)["cif"]
]
target = Path("data/mp_20/novelty_reference.pkl")
target.parent.mkdir(parents=True, exist_ok=True)
with target.open("wb") as handle:
    pickle.dump(structures, handle, protocol=4)
PY
```

## Results

The previously reported small-sample evaluation uses the converted pretrained
LDM with DDIM 50 steps, 50 structures of 20 atoms, and seed 2024. Novelty is measured
against the MP20 training structures in `novelty_reference.pkl`.

| Model | Dataset | Valid (%) | Unique (%) | Novel (%) | Config | Checkpoint |
| --- | --- | --- | --- | --- | --- | --- |
| Chemeleon2 (diffusion) | MP20 | 98.0 | 100.0 | 100.0 | [chemeleon2_mp20_ldm.yaml](chemeleon2_mp20_ldm.yaml) | [Pretrained LDM](https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/Chemeleon2/chemeleon2_ldm.zip) |

This evaluates the pretrained diffusion checkpoint, not a full training run with
the supplied YAML or an RL-fine-tuned model. RL has integration-test coverage;
no converged RL generation result is reported here.

## Command

### Training

Train the diffusion Transformer from scratch with the registered VAE frozen:

```bash
python structure_generation/train.py -c structure_generation/configs/chemeleon2/chemeleon2_mp20_ldm.yaml
```

To train the VAE, select its configuration from the stage table. It loads the
registered VAE by default; add
`Model.__init_params__.vae_pretrained_model_name=null` for training from scratch.
For RL, prepare the reference population above and select the RL configuration;
it starts from the registered LDM.

To use a custom VAE in diffusion training, set `vae_pretrained_model_name=null`
and set `vae_ckpt_path` to a unified VAE training checkpoint. These checkpoints
use `core.` parameter keys; older local bare VAE checkpoints need that prefix
added before loading. Registered VAE component packages load directly.
The inference-only [sample configuration](chemeleon2_mp20_sample.yaml) is used
with a complete LDM checkpoint.

### Validation

Evaluate VAE reconstruction on the validation split using the registered VAE:

```bash
python structure_generation/train.py -c structure_generation/configs/chemeleon2/chemeleon2_mp20_vae.yaml Global.do_train=False Global.do_eval=True
```

### Testing

Evaluate VAE reconstruction on the test split using the registered VAE:

```bash
python structure_generation/train.py -c structure_generation/configs/chemeleon2/chemeleon2_mp20_vae.yaml Global.do_train=False Global.do_eval=False Global.do_test=True
```

For a custom unified VAE checkpoint, disable `vae_pretrained_model_name` and set
`Trainer.pretrained_model_path`. Reconstruction losses are distinct from the
generation metrics reported above.

### Sample

Generate a 20-atom crystal with the registered LDM:

```bash
python structure_generation/sample.py --model_name=chemeleon2_ldm --mode=by_num_atoms --num_atoms=20
```

Generated structures are saved as CIF files under the default output directory.

## Citation

[Original implementation](https://github.com/hspark1212/chemeleon2) and
[upstream documentation](https://hspark1212.github.io/chemeleon2/) provide further
background on the method.

```bibtex
@article{Park2025chemeleon2,
  title={Guiding Generative Models to Uncover Diverse and Novel Crystals via Reinforcement Learning},
  author={Hyunsoo Park and Aron Walsh},
  year={2025},
  url={https://arxiv.org/abs/2511.07158}
}
```

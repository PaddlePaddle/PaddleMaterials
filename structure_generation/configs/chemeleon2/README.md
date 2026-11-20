# Chemeleon2

[Guiding Generative Models to Uncover Diverse and Novel Crystals via Reinforcement Learning](https://arxiv.org/abs/2511.07158)

## Abstract

The discovery of novel crystalline materials with targeted properties remains a central challenge in computational materials science. Generative models offer a promising route to accelerate this process, yet existing approaches often struggle to simultaneously achieve high novelty, thermodynamic stability, and compositional diversity. Here, we present **Chemeleon2**, a reinforcement learning framework built on latent diffusion models for crystal structure generation. Chemeleon2 implements a three-stage sequential pipeline—Variational Autoencoder (VAE), Latent Diffusion Model (LDM), and Reinforcement Learning (RL)—where each stage builds upon the learned representations of the previous. The RL stage employs Group Relative Policy Optimization (GRPO) with a modular, multi-objective reward system to steer generation toward desired material properties. Chemeleon2 supports de novo generation (DNG) and composition-specified prediction (CSP), and provides a simple Python interface for defining custom reward functions targeting arbitrary material properties such as band gap, density, or thermodynamic stability.

---

## Model Description

### Overview

Chemeleon2 represents a crystal structure by its unit cell:
- atom types: $A = (a_1, \ldots, a_N)$, where $a_i \in \{1, \ldots, 100\}$ (atomic number)
- fractional coordinates: $X = (x_1, \ldots, x_N)$, $x_i \in [0,1)^3$
- lattice parameters: lengths $(a, b, c)$ and angles $(\alpha, \beta, \gamma)$

The three-stage pipeline is strictly sequential: the VAE is trained first to learn a continuous latent space, the LDM is then trained to generate in that latent space, and finally RL fine-tuning shapes generation toward target properties. This repo ports the VAE and LDM stages; the RL stage is not yet ported (see Stage 3 below).

### Method

#### 1) Stage 1: Variational Autoencoder (VAE)

The VAE compresses crystal structures into a continuous latent space of dimension $L=8$ per atom, enabling the LDM to operate in a low-dimensional, well-structured space.

**Encoder** (`Chemeleon2TransformerEncoder`, 8 layers, $d_\text{model}=512$, 8 heads): Atom type embeddings and fractional coordinate projections are summed and passed through transformer self-attention layers to produce per-atom representations of shape $(B_n, d_\text{model})$. A linear projection `quant_conv` maps these to mean $\mu$ and log-variance $\log\sigma^2$:

$$
(\mu, \log\sigma^2) = \text{quant\_conv}(\text{Encoder}(A, X)) \in \mathbb{R}^{B_n \times 2L}
$$

**Reparameterization**: Latent vectors are sampled via:

$$
z = \mu + \sigma \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I), \quad z \in \mathbb{R}^{B_n \times L}
$$

**Decoder** (`Chemeleon2TransformerDecoder`, 8 layers, $d_\text{model}=512$, 8 heads): The latent $z$ is projected back to $d_\text{model}$ via `post_quant_conv` and decoded into four prediction heads: atom type logits, lattice lengths $(a,b,c)$, lattice angles $(\alpha,\beta,\gamma)$, and fractional coordinates $(x,y,z)$.

**Training objective**:

$$
\mathcal{L}_\text{VAE} = \lambda_A \mathcal{L}_\text{CE}(A, \hat{A}) + \lambda_L \mathcal{L}_\text{MSE}(L, \hat{L}) + \lambda_\alpha \mathcal{L}_\text{MSE}(\alpha, \hat{\alpha}) + \lambda_X \mathcal{L}_\text{MSE}(X, \hat{X}) + \lambda_\text{KL} \cdot \text{KL}(q(z|x) \| \mathcal{N}(0,I))
$$

where $\lambda_\text{KL} = 10^{-5}$ is kept small to prevent posterior collapse. The VAE is frozen (parameters fixed, gradients disabled) for all subsequent stages.

#### 2) Stage 2: Latent Diffusion Model (LDM)

The LDM learns to generate crystal structures by performing Gaussian diffusion entirely within the VAE's latent space, avoiding the complexity of diffusing directly over discrete atom types and periodic coordinates.

**Forward process**: Latent vectors $z_0 \in \mathbb{R}^{B_n \times L}$ are reshaped to dense batches $(B, N, L)$ with a padding mask, then corrupted over $T=1000$ timesteps with a linear noise schedule:

$$
q(z_t | z_0) = \mathcal{N}\!\left(z_t;\, \sqrt{\bar{\alpha}_t}\, z_0,\, (1 - \bar{\alpha}_t) I\right)
$$

**Denoiser** (`Chemeleon2DiT`, Diffusion Transformer): A Vision Transformer-based architecture with `hidden_size=768`, `depth=12`, `num_heads=12` processes the noisy dense latent $(B, N, L)$ with timestep $t$ and optional condition $y$, using Adaptive LayerNorm (AdaLN) conditioning and masked self-attention to handle variable-length structures:

$$
\epsilon_\theta = \text{DiT}(z_t,\, t,\, \text{mask},\, y)
$$

**Training objective** (simple diffusion loss):

$$
\mathcal{L}_\text{LDM} = \mathbb{E}_{z_0, \epsilon, t}\!\left[\|\epsilon - \epsilon_\theta(z_t, t, \text{mask}, y)\|^2\right]
$$

**Sampling**: Both DDPM and DDIM samplers are supported. DDIM with 50 steps is the default for efficient generation. The final latent $z_0$ is decoded by the frozen VAE decoder to recover the crystal structure.

**Port deviation note**: The upstream model trains its variance head with a
variational-bound term (`ModelVarType.LEARNED_RANGE`) in addition to the
epsilon MSE. This port trains only the epsilon head with the objective
above: the variance output channels created by `learn_sigma` receive no
training signal, and `sampler=ddpm` uses the scheduler's fixed variance
instead of the upstream learned-range variance. DDIM (the default sampler)
does not consume variance and is unaffected; keep `sampler=ddim` when
comparing against upstream results.

**Conditional generation**: A `Chemeleon2ConditionModule` embeds composition (CSP) or scalar property conditions into a vector $y \in \mathbb{R}^{L_y}$. Classifier-Free Guidance (CFG) is applied at sampling time:

$$
\epsilon_\text{cfg} = \epsilon_\theta(\cdot \mid \varnothing) + w \cdot \bigl(\epsilon_\theta(\cdot \mid y) - \epsilon_\theta(\cdot \mid \varnothing)\bigr)
$$

where $w$ is the guidance scale (default 2.0).

#### 3) Stage 3: Reinforcement Learning (RL) with GRPO (not yet ported)

The original paper fine-tunes the LDM denoiser with Group Relative Policy
Optimization (GRPO) and a modular, multi-objective reward system to steer
generation toward desired material properties. This RL stage is described in
the paper for reference only and is **not yet ported** in PaddleMaterials; the
current release covers the VAE and LDM stages only.

---

## Dataset Description

- **MP-20**: A benchmark subset of Materials Project structures containing **up to 20 atoms per unit cell**. It is widely used for fair comparison across crystal generative models. The dataset includes property labels and is split into train/val/test sets.

Recommended data fields for each sample:
- `atom_types`: length-$N$ list of atomic numbers
- `frac_coords`: $N \times 3$ fractional coordinates in $[0,1)$
- `lengths`: lattice vector lengths $(a, b, c)$
- `angles`: lattice angles $(\alpha, \beta, \gamma)$

Optional fields include `num_atoms`, `band_gap`, `e_above_hull`, and other property labels.

#### MP-20 split (download link)
| Dataset | Train | Val | Test |
| --- | --- | --- | --- |
| [MP-20](https://paddle-org.bj.bcebos.com/paddlematerial/datasets/mp_20/mp_20.zip) | 27136 | 9047 | 9046 |

#### Novelty reference
The `metrics.reference_file_path` field of `chemeleon2_mp20_sample.yaml` expects
`novelty_reference.pkl`: a pickle file holding the list of training-set
pymatgen `Structure` objects used for the novelty computation. It is not part
of the MP-20 zip; build it locally from the downloaded `train.csv` (parse the
cif strings with `BuildStructure.build_one`) or set `reference_file_path` to
null to skip the novelty metric.

---

## Results

| Model | Dataset | Val (loss) | Valid (%) | Unique (%) | Novel (%) | GPUs | Training Time | Config | Checkpoint / Log |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| chemeleon2_vae | mp20 | 0.2593 | - | - | - | 1 | - | [chemeleon2_mp20_vae.yaml](chemeleon2_mp20_vae.yaml) | [checkpoint](https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/Chemeleon2/chemeleon2_vae.zip) |
| chemeleon2_ldm | mp20 | 0.6704 | 95.97 | 86.12 | 99.89 | 1 | - | [chemeleon2_mp20_ldm.yaml](chemeleon2_mp20_ldm.yaml) | [checkpoint](https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/Chemeleon2/chemeleon2_ldm.zip) |


---

## Command

### Training

```bash
# Stage 1: VAE training (mp20 dataset)
python structure_generation/train.py -c structure_generation/configs/chemeleon2/chemeleon2_mp20_vae.yaml

# Stage 2: LDM training (mp20 dataset, requires a pre-trained VAE)
# Set vae_ckpt_path in chemeleon2_mp20_ldm.yaml to the VAE checkpoint
# (e.g. the best.pdparams extracted from chemeleon2_vae.zip); training
# on a randomly initialized frozen VAE is meaningless.
python structure_generation/train.py -c structure_generation/configs/chemeleon2/chemeleon2_mp20_ldm.yaml
```

### Validation

```bash
# Adjust program behavior on the fly using command-line parameters without modifying the configuration file directly.
# Example: --Global.do_eval=True

# VAE validation
python structure_generation/train.py -c structure_generation/configs/chemeleon2/chemeleon2_mp20_vae.yaml Global.do_eval=True Global.do_train=False Global.do_test=False Trainer.pretrained_model_path='path/to/vae_model.pdparams'

# LDM validation
python structure_generation/train.py -c structure_generation/configs/chemeleon2/chemeleon2_mp20_ldm.yaml Global.do_eval=True Global.do_train=False Global.do_test=False Trainer.pretrained_model_path='path/to/ldm_model.pdparams'
```

### Testing

```bash
# This command is used to evaluate the model's performance on the test dataset.

# VAE testing
python structure_generation/train.py -c structure_generation/configs/chemeleon2/chemeleon2_mp20_vae.yaml Global.do_eval=False Global.do_train=False Global.do_test=True Trainer.pretrained_model_path='path/to/vae_model.pdparams'

# LDM testing
python structure_generation/train.py -c structure_generation/configs/chemeleon2/chemeleon2_mp20_ldm.yaml Global.do_eval=False Global.do_train=False Global.do_test=True Trainer.pretrained_model_path='path/to/ldm_model.pdparams'
```

### Sample

```bash
# This command is used to predict the crystal structure using a trained model.
# Mode 1: Use a pre-trained model (downloads automatically via MODEL_REGISTRY).
# Mode 2: Use a custom configuration file and checkpoint.
# Results are saved to the folder specified by --output_path (default: results).
#

# Mode 1: pre-trained model
python structure_generation/sample.py --model_name='chemeleon2_ldm' --weights_name='best.pdparams' --output_path='result_chemeleon2_ldm/' --mode='by_num_atoms' --num_atoms=20

# Mode 2: custom config + checkpoint
python structure_generation/sample.py --config_path='structure_generation/configs/chemeleon2/chemeleon2_mp20_sample.yaml' --checkpoint_path='./output/chemeleon2_ldm/checkpoints/best.pdparams' --output_path='result_chemeleon2_ldm/' --mode='by_num_atoms' --num_atoms=20

# Quick forward pass test (no training data required)
python -c "
from ppmat.models import build_model_from_name
import paddle
model, config = build_model_from_name('chemeleon2_ldm')
batch = {
    'structure_array': {
        'atom_types': paddle.randint(0, 100, [640]),
        'num_atoms': paddle.full([32], 20, dtype='int64'),
        'frac_coords': paddle.rand([640, 3]),
        'lengths': paddle.rand([32, 3]) * 10 + 5,
        'angles': paddle.rand([32, 3]) * 60 + 60,
    }
}
out = model(batch)
print(f'LDM forward loss: {float(out[\"loss_dict\"][\"total_loss\"]):.4f}')
"
```

---

## Citation

```
@article{Park2025chemeleon2,
  title={Guiding Generative Models to Uncover Diverse and Novel Crystals via Reinforcement Learning},
  author={Hyunsoo Park and Aron Walsh},
  year={2025},
  url={https://arxiv.org/abs/2511.07158}
}
```

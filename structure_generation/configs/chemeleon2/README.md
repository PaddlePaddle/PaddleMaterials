# Chemeleon2

[Guiding Generative Models to Uncover Diverse and Novel Crystals via Reinforcement Learning](https://arxiv.org/abs/2511.07158)

## Abstract

Chemeleon2 generates periodic crystal structures with a variational autoencoder
(VAE), a diffusion Transformer in the learned latent space, and a reinforcement
learning (RL) stage that fine-tunes the diffusion denoiser with Group Relative
Policy Optimization (GRPO). PaddleMaterials implements all three stages.

## Model Description

### Overview

`Chemeleon2` is the single entry point for both training (`forward`, with `stage`
selecting `vae`, `diffusion` or `rl`) and generation (`sample`/`predict`). The VAE,
the DiT denoiser and the GRPO wrapper are internal sub-modules.

The VAE encodes atomic numbers, fractional coordinates and the lattice matrix
into eight latent values per atom. Its decoder reconstructs atom types,
fractional coordinates, atom-count-scaled lengths and lattice angles. The LDM
learns Gaussian diffusion in this continuous latent space; it does not diffuse
atom types and fractional coordinates separately. The RL stage keeps the VAE
frozen and optimizes only the LDM denoiser against a composable reward.

### Method

VAE training combines atom classification, coordinate and lattice reconstruction,
and KL losses. Translation, rotation and coordinate/type corruption are optional.
The LDM freezes the VAE and trains a masked epsilon predictor. Its scheduler is
configured through `noise_scheduler_cfg` and built by `build_scheduler`.
DDIM with 50 inference steps is the default; DDPM is also supported. Neither
sampler clips the latent prediction when `clip_denoised=False`.

The RL stage rolls out a DDIM sampling trajectory, decodes the final latent with
the frozen VAE, scores each structure with the reward components, and recomputes
the per-step Gaussian log-probabilities with gradients enabled to form the
clipped GRPO objective (`clip_ratio`, `kl_weight`, `entropy_weight`). Groups of
`num_group_samples` rollouts share an advantage normalization controlled by
`group_reward_norm`. Only the denoiser is optimized; the VAE and the condition
embedding are frozen. The reward function composes pluggable components:
`CreativityReward` (unique AND novel), `EnergyReward` (energy above hull),
`StructureDiversityReward` / `CompositionDiversityReward` (per-sample MMD over
frozen-VAE features) and `CustomReward`. RL training uses DDIM because the
learned-variance DDPM path does not expose per-step means. `Chemeleon2RLModule`
return meets the framework training contract (`forward` ->`{"loss_dict": {"loss"}}`)
and delegates `sample`/`predict` to the wrapped LDM, so it is a drop-in model for
`BaseTrainer` and `StructureSampler`.

Port deviations from the upstream RL stage:

- `amd` is not a project dependency. `CreativityReward` falls back to an inverse
  lattice-parameter distance when a structure is unique but not novel (or vice
  versa); with `amd` installed the AMD distance is used.
- Upstream uses a MACE calculator for energy above hull. MACE is PyTorch-only and
  is not used here. `EnergyReward` requires an injectable ASE `calculator` plus a
  pickled pymatgen phase diagram; it is omitted from the supplied RL config, which
  runs on CPU-only metrics.
- Diversity reference features are computed at runtime from a configurable
  reference subset with the frozen VAE, instead of shipping precomputed tensors.

The upstream learned-variance objective is not implemented: `learn_sigma=True`
retains the released parameter layout, but the additional output channels are
excluded from the epsilon loss. DDPM uses fixed variance. This implementation
must not be described as numerically equivalent to upstream learned-range DDPM.

The released VAE vocabulary has 100 entries indexed by atomic number, including
index zero for corruption. Its supported physical atomic numbers are 1 through
99. Supporting atomic number 100 requires a larger VAE vocabulary and compatible
weights. Composition embeddings likewise require `in_dim` greater than every
atomic number they encode; composition vectors are normalized to unit sum.

Conditional training uses `condition_names` and `property_embeddings_adapt_cfg`.
Each embedding is configured with `__class_name__` and `__init_params__`; supported
encoders are `ValueEncoder`, `CategoricalEncoder`, `CompositionEncoder` and
`ChemicalSystemEncoder`. `p_unconditional` controls joint condition dropout and
`cfg_scale` controls classifier-free guidance at sampling. The revised conditional
configuration is incompatible with earlier conditional checkpoints.

Configurations use `*_cfg` names built by `build_model`. The released
`chemeleon2_ldm` and `chemeleon2_vae` packages share this layout: `chemeleon2_ldm`
is a unified `Chemeleon2` checkpoint with `core.`-prefixed weights, and
`chemeleon2_vae` keeps bare `Chemeleon2VAEModule` weights.

CINN dispatch is available at `denoise_step`; eager/CINN parity for structure
sampling is verified on GPU, for both a single denoise step and full DDIM
sampling (atol 2e-5).

## Dataset Description

| Dataset | Train | Val | Test |
| --- | --- | --- | --- |
| [MP20](https://paddle-org.bj.bcebos.com/paddlematerial/datasets/mp_20/mp_20.zip) | 27136 | 9047 | 9046 |

The supplied configurations reuse `MP20Dataset` and its automatic download.
Stage-two training loads the registered `chemeleon2_vae` weights by default.
For a custom VAE, set `vae_pretrained_model_name=null` and provide `vae_ckpt_path`.
The RL stage draws `num_atoms` from the training batches and starts from the
registered `chemeleon2_ldm` denoiser.

`StructGenMetric` reports validity and uniqueness. Validity uses composition and
structure checks, without fingerprint extraction. Uniqueness compares valid
structures across the complete generated population, including cross-batch
duplicates. Novelty is returned only when `reference_file_path` or
`reference_structures` provides a reference population. A reference pickle must
contain training-set pymatgen structures and is not included in the MP20 archive;
`data/mp_20/novelty_reference.pkl` holds the MP20 training structures used below.

## Results

`StructGenMetric` on `data/mp_20/novelty_reference.pkl` (MP20 training
structures, not shipped with the MP20 archive), 50 samples of 20 atoms with
DDIM 50 steps (eval seed 2024):

| Model | Dataset | Valid (%) | Unique (%) | Novel (%) | Config | Checkpoint |
| --- | --- | --- | --- | --- | --- | --- |
| Chemeleon2 (diffusion) | MP20 | 98.0 | 100.0 | 100.0 | [chemeleon2_mp20_ldm.yaml](chemeleon2_mp20_ldm.yaml) | [Registered LDM](https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/Chemeleon2/chemeleon2_ldm.zip) |

These values come from the released LDM denoiser, not an RL-fine-tuned result.
The RL stage is a functional smoke run (reward 1.41-1.65 over 30 steps; see the
[RL config](chemeleon2_mp20_rl.yaml)), not a converged result.

## Command

### Training

```bash
# Stage 1: VAE
python structure_generation/train.py -c structure_generation/configs/chemeleon2/chemeleon2_mp20_vae.yaml Trainer.pretrained_model_path=null
# Stage 2: LDM (uses the registered frozen VAE)
python structure_generation/train.py -c structure_generation/configs/chemeleon2/chemeleon2_mp20_ldm.yaml
# Stage 3: RL (uses the registered frozen LDM denoiser)
python structure_generation/train.py -c structure_generation/configs/chemeleon2/chemeleon2_mp20_rl.yaml
```

To train on band gaps, select a conditional configuration built from
`chemeleon2_mp20_ldm.yaml` (see the conditional-training notes above). The RL
config accepts `Global.num_group_samples`, `Global.sampling_steps` and the reward
components through `Model.__init_params__.reward_fn_cfg`.

### Validation

```bash
python structure_generation/train.py -c structure_generation/configs/chemeleon2/chemeleon2_mp20_vae.yaml Global.do_train=False Global.do_eval=True
```

Validation and testing load the registered VAE through the YAML by default.
Custom checkpoint paths can be supplied in `Trainer.pretrained_model_path`.

### Testing

```bash
python structure_generation/train.py -c structure_generation/configs/chemeleon2/chemeleon2_mp20_vae.yaml Global.do_train=False Global.do_eval=False Global.do_test=True
```

### Sample

```bash
python structure_generation/sample.py --model_name=chemeleon2_ldm --mode=by_num_atoms --num_atoms=20
```

The sampler writes generated CIF files under its default output directory.

## Citation

```bibtex
@article{Park2025chemeleon2,
  title={Guiding Generative Models to Uncover Diverse and Novel Crystals via Reinforcement Learning},
  author={Hyunsoo Park and Aron Walsh},
  year={2025},
  url={https://arxiv.org/abs/2511.07158}
}
```

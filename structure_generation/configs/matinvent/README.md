# MatInvent

[MatInvent: Accelerating inverse materials design using generative diffusion models with reinforcement learning](https://arxiv.org/abs/2511.03112)

## Abstract

MatInvent is a general and efficient reinforcement learning workflow that optimizes diffusion models for goal-directed crystal generation. MatInvent enables robust optimization for inverse material design tasks with single or multiple target properties. Compatible with diverse diffusion model architectures and property constraints, MatInvent could offer broad applicability in materials discovery.

![MatInvent RL Pipeline](../../docs/matinvent_RL_pipeline.png)

---

## Model Description

### Overview

MatInvent adapts pretrained crystal diffusion models for goal-directed materials generation through reinforcement learning. Each round generates crystal structures, evaluates their properties, and fine-tunes the generator using the resulting rewards. The implementation supports MatterGen and DiffCSP backbones, and both are registered models of the structure generation task.

| Backbone | Generation task | Config |
| --- | --- | --- |
| MatterGen | Joint generation of atom types, coordinates, and lattice | [matinvent_mattergen_mp20.yaml](matinvent_mattergen_mp20.yaml) |
| DiffCSP | Coordinate and lattice generation with fixed atom types | [matinvent_diffcsp_mp20.yaml](matinvent_diffcsp_mp20.yaml) |
| DiffCSP | Joint generation of atom types, coordinates, and lattice | [matinvent_diffcsp_mp20_joint.yaml](matinvent_diffcsp_mp20_joint.yaml) |

The MatterGen and fixed-composition DiffCSP configurations optimize density. The joint DiffCSP configuration optimizes the Herfindahl-Hirschman Index (HHI) for supply risk, with relaxation, stability, novelty, and uniqueness screening. Reward properties and targets are configurable.

### Framework integration

| Config block | Content |
| --- | --- |
| `Model` | `MatinventMatterGen` or `MatinventDiffCSP`: the diffusion backbone itself, with one noise scheduler per data space |
| `Optimizer` | The reward-weighted fine-tuning optimizer |
| `Dataset` | MP-20 train (denoising anchors), val (validation and early-stop probe) and test splits |
| `Trainer` | `ppmat.models.matinvent.trainer.MatInventTrainer` plus the RL recipe |
| `Sample` | Sampling dataset and diffusion step budget |

One epoch is one RL round, so `Trainer.max_epochs` is the number of rounds and `save_freq`/`eval_freq` keep their framework meaning. `Trainer.train()` owns the optimizer, the checkpoints (`output_dir/checkpoints`), the learning-rate schedule and validation; the replay buffer and the long-term memory are stored in `output_dir/rl_state.pkl` and restored by `Trainer.resume_from_checkpoint`. Sampling and reward scoring run in a single process, and mixed precision is not supported, so the configurations reject `use_amp` and distributed launches.

### Method

For a generated structure with reward $r$, fine-tuning minimizes the reward-weighted diffusion objective with regularization toward a frozen pretrained prior:

$$
\mathcal{L} = r\,\mathcal{L}_{\mathrm{diff}} + \sigma\,(1.1-r)\,\mathcal{L}_{\mathrm{prior}}.
$$

Both DiffCSP configurations additionally use a denoising loss on MP-20 training structures to reduce distribution drift. Fixed-composition DiffCSP uses a fixed anchor batch; joint DiffCSP resamples training anchors at each anchor loss computation while keeping its validation probe fixed.

Here, $\mathcal{L}_{\mathrm{diff}}$ combines the coordinate, lattice, and, for joint generation, atom-type denoising losses. $\mathcal{L}_{\mathrm{prior}}$ measures the mean squared difference between the agent and prior predictions; $\sigma$ controls its weight.

Structures are ranked by reward before fine-tuning. A replay buffer retains high-reward structures, while a composition-based diversity filter reduces rewards for repeatedly generated compositions. The agent and frozen prior are initialized from the same pretrained backbone.

---

## Dataset Description

RL training uses generated structures and property rewards. Both DiffCSP configurations additionally use MP-20 training structures as denoising anchors. Density tasks draw atom counts from the Alex-MP-20 distribution; joint DiffCSP uses the MP-20 distribution.

| Data | Usage | Source |
| --- | --- | --- |
| Generated crystals | Online reward evaluation and fine-tuning | Current diffusion model |
| Pretrained backbones | Agent and frozen-prior initialization | Registered MatInvent packages; original checkpoints are available on [Hugging Face](https://huggingface.co/jwchen25/MatInvent) |
| MP-20 training structures | DiffCSP denoising anchors | [MP-20](https://paddle-org.bj.bcebos.com/paddlematerial/datasets/mp_20/mp_20.zip) |
| MP-20 validation structures | Validation loss and early-stop probe | [MP-20](https://paddle-org.bj.bcebos.com/paddlematerial/datasets/mp_20/mp_20.zip) |
| MP2020-corrected reference database | Joint DiffCSP stability and novelty screening | [reference_MP2020correction.gz](https://huggingface.co/jwchen25/MatInvent/blob/main/reference_MP2020correction.gz) |

The MP-20 splits are declared in the `Dataset` block. For joint DiffCSP training, place the reference archive at `data/matinvent/reference_MP2020correction.gz` or update `qualification_cfg.reference_path` under `Trainer.__init_params__`. Relaxation uses the registered `mattersim_5M` model.

---

## Results

| Model | Val loss | Density (g/cm^3) | Valid (%) | Target success (%) | Mean reward | Config | Initialization |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| matinvent_mattergen_mp20 | 0.3854 | 7.571 | 80.21 | 49.48 | 0.5322 | [config](matinvent_mattergen_mp20.yaml) | [pretrained checkpoint](https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/MatInvert/matinvent_mattergen_mp20.zip) |
| matinvent_diffcsp_mp20 | 0.9735 | 11.522 | 30.73 | 29.69 | 0.2843 | [config](matinvent_diffcsp_mp20.yaml) | [pretrained checkpoint](https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/MatInvert/matinvent_diffcsp_mp20.zip) |

| Model | Val loss | HHI | Valid (%) | Qualified (%) | Target success (%) | Config | Initialization |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| matinvent_diffcsp_mp20_joint | 1.0039 | 3231.84 | 82.81 | 15.10 | 0.52 | [config](matinvent_diffcsp_mp20_joint.yaml) | [pretrained checkpoint](https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/MatInvert/matinvent_diffcsp_mp20_joint.zip) |

RL checkpoint results. Links provide initialization weights.

Rates use all requested samples; density/HHI means use valid/qualified structures. Success: density >= 5 g/cm^3 or qualified HHI <= 1250.

---

## Command

### Training
```bash
# MatterGen backbone
python structure_generation/train.py -c structure_generation/configs/matinvent/matinvent_mattergen_mp20.yaml

# DiffCSP backbone with fixed atom types
python structure_generation/train.py -c structure_generation/configs/matinvent/matinvent_diffcsp_mp20.yaml

# Joint DiffCSP generation; prepare the reference database first
python structure_generation/train.py -c structure_generation/configs/matinvent/matinvent_diffcsp_mp20_joint.yaml
```

RL checkpoints are written by the framework trainer as `output_dir/checkpoints/epoch_*.pdparams` and can be resumed with `Trainer.resume_from_checkpoint`.

### Validation
Set `Global.do_eval` to `True` to evaluate the diffusion loss on the `Dataset.val` split. Each round also reports the reward, the number of scored structures, the density/HHI statistics and the diversity metrics through the framework logger and the enabled visualisation backends.

### Testing
Set `Global.do_test` to `True` to evaluate the diffusion loss on the `Dataset.test` split. The configurations do not define a task metric, so `sample.py --mode=compute_metric` is not available.

### Sample
```bash
# Predict crystal structures using a trained model.
# Mode 1: Use a pre-trained model (downloads automatically).
# Mode 2: Use a custom configuration file and checkpoint.
# Results are saved to the folder specified by --output_path (default: results).

# Mode 1: pre-trained MatterGen backbone
python structure_generation/sample.py --model_name='matinvent_mattergen_mp20' --output_path='result_matinvent_mattergen/' --mode='by_dataloader'

# Mode 2: MatterGen checkpoint produced by an RL run
python structure_generation/sample.py --config_path='structure_generation/configs/matinvent/matinvent_mattergen_mp20.yaml' --checkpoint_path='<run_dir>/checkpoints/epoch_10.pdparams' --output_path='result_matinvent_mattergen/' --mode='by_dataloader'

# Mode 2: fixed-composition DiffCSP checkpoint; the formula supplies the atom types
python structure_generation/sample.py --config_path='structure_generation/configs/matinvent/matinvent_diffcsp_mp20.yaml' --checkpoint_path='<run_dir>/checkpoints/epoch_10.pdparams' --output_path='result_matinvent_diffcsp/' --mode='by_chemical_formula' --chemical_formula='LiMnO2'
```

---

## Citation
```
@article{matinvent,
  title={Accelerating inverse materials design using generative diffusion models with reinforcement learning},
  author={Chen, Junwu and Guo, Jeff and Fako, Edvin and Schwaller, Philippe},
  journal={arXiv preprint arXiv:2511.03112},
  year={2025}
}
```

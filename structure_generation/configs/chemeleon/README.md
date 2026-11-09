# Chemeleon

[Exploration of crystal chemical space using text-guided generative artificial intelligence](https://doi.org/10.1038/s41467-025-59636-y)

## Abstract

Chemeleon is a text-conditioned diffusion model for crystal structure generation: given a text prompt such as a composition, it denoises atom types, the lattice matrix, and fractional coordinates into a periodic crystal. Three diffusion processes run in lockstep — an absorbing-state discrete diffusion (D3PM) over atom types, a variance-preserving diffusion over the lattice, and a variance-exploding diffusion over fractional coordinates on a torus — and share a single CSPNet decoder pass per step. Text conditions enter the decoder through a FiLM layer, and sampling combines a predictor-corrector scheme with classifier-free guidance. The architecture follows the official work; this directory contains the PaddlePaddle implementation, verified in fp64.

## Model Description

A crystal is represented as atom types $A = (a_1,\ldots,a_N)$ (vocabulary of 104 entries: 103 elements plus one absorbing dummy state), fractional coordinates $F \in [0,1)^{3 \times N}$, and a lattice matrix $L \in \mathbb{R}^{3 \times 3}$ (upper-triangular effective freedom of 6 entries). Each process corrupts its own channel:

- **Atom types (D3PM):** an absorbing-state transition matrix $Q_t$ progressively replaces tokens with the dummy state; the model learns $q(x_{t-1} \mid x_t, x_0)$ with a hybrid VB + CE objective.
- **Lattice (VP diffusion):** standard DDPM noising $L_t = \sqrt{\bar\alpha_t}\,L_0 + \sqrt{1-\bar\alpha_t}\,\epsilon$, with cosine $\beta$ over 1000 timesteps.
- **Coordinates (VE diffusion):** wrapped-normal noising $F_t = w(F_0 + \sigma_t \epsilon)$ with a score-matching target, sampled by an ancestral predictor plus a Langevin corrector.

The denoiser is a CSPNet over the fully-connected atom graph: periodic Fourier features of fractional-coordinate differences feed message passing layers, a FiLM layer injects the time and text embeddings into every layer, and three readout heads predict the atom-type logits, the lattice noise, and the coordinate score. The text front-end (WordPiece tokenizer + BERT-base encoder + projection MLP) is bundled with the model under `ppmat/models/chemeleon/`, so no extra NLP packages are required; the transformer backbone is frozen during diffusion training. This config conditions on the composition string (e.g. `Li2O`, from the `composition` csv column, `text_targets: ["composition"]`); the original release also describes a two-stage recipe with a contrastively trained text encoder for richer text prompts.

## Dataset Description

`mp-40` contains Materials Project structures with fewer than 40 atoms per unit cell, split into 28,636 / 3,181 / 708 structures for train / val / test. Each record carries a `cif` string plus text-target columns (the original descriptions were generated with an LLM, the OpenAI API in the original preparation). No dataset is shipped with this repository: point `Dataset.*.dataset.__init_params__.data_dir` to a directory holding `{split}.csv` files. The original release provides the processed csv files and a preparation notebook:

- csv files: [data/mp-40](https://github.com/hspark1212/chemeleon/tree/main/data/mp-40)
- preparation: [data_preparation.ipynb](https://github.com/hspark1212/chemeleon/blob/main/data/mp-40/data_preparation.ipynb)

## Commands

### Training

```bash
# multi-gpu training (example with 4 GPUs)
python -m paddle.distributed.launch --gpus="0,1,2,3" structure_generation/train.py -c structure_generation/configs/chemeleon/chemeleon_mp40.yaml
# single-gpu training
python structure_generation/train.py -c structure_generation/configs/chemeleon/chemeleon_mp40.yaml
```

### Sample

```bash
# Generate structures from a text condition (composition prompt).
# --num_atoms must match the stoichiometry of the composition (Li2O -> 3).
# Results are saved to the folder specified by --output_path (default: results).
python structure_generation/sample.py --config_path='structure_generation/configs/chemeleon/chemeleon_mp40.yaml' --checkpoint_path='./output/chemeleon_mp40/checkpoints/latest.pdparams' --mode=by_condition --condition text=Li2O --num_atoms=3 --output_path='result_chemeleon_mp40/'
```

## Accuracy Alignment

This model was verified against the official implementation (fp64 on CPU).

| Gate              | Comparison                                                  | Result                                        |
| :---------------- | :---------------------------------------------------------- | :-------------------------------------------- |
| diffusion buffers | scheduler / D3PM buffers vs the torch checkpoint            | bitwise identical                             |
| text encoder      | bundled MinimalBert CLS / projected embeddings (16 texts)   | max_abs 8.88e-15 (CLS) / 3.73e-14 (projected) |
| forward           | full forward, condition-kept and condition-dropped variants | worst max_abs 2.29e-7                         |
| sampling          | first 10 reverse steps, injected noises                     | worst 5.35e-7, zero atom-type flips           |

Training loss parity (fp32 CPU, 512-structure train subset, batch 32, 2 epochs with identical batch order and injected noises / text vectors, Adam lr 1e-3), mean relative difference per loss term:

| vb     | ce     | lattice | coords | total  |
| :----- | :----- | :------ | :----- | :----- |
| 1.5e-4 | 2.6e-6 | 2.0e-5  | 3.4e-4 | 1.7e-4 |

The workflow tests can be reproduced from the repository root:

```bash
python -m pytest test/test_chemeleon_cinn_workflows.py -v   # 4 passed
```

## CINN Acceleration

Inference can be switched to compiled execution by setting `Execution.backend=cinn` in the config yaml. The recommended tier is AST (`full_graph=True`). Compiled and eager results agree on the 16-structure set: worst forward difference 1.9e-5, worst sampling difference 5.8e-5 with zero atom-type flips (paddle allclose, atol = rtol = 3e-4).

Inference speed (fixed batch of 4 graphs x 20 atoms, 100 sampling steps, 2 repeats, fp32):

| Mode                        | s/step | Speedup |
| :-------------------------- | :----- | :------ |
| eager (default)             | 0.0489 | —       |
| cinn (AST, full_graph=True) | 0.0305 | +60.5%  |

The SOT tier (`full_graph=False`) yields +8.9%. Compilation costs about half minute once per distinct shape.

For training, CINN brings about +1.7% (0.119 -> 0.117 s/step, fixed batch shape) speedup. However, if the batch size is variable, each time a new shape takes about one minune to re-compile.

## Citation

```bibtex
@article{park2025exploration,
  title={Exploration of crystal chemical space using text-guided generative artificial intelligence},
  author={Park, Hyunsoo and Onwuli, Anthony and Walsh, Aron},
  journal={Nature Communications},
  volume={16},
  number={1},
  pages={1--14},
  year={2025},
  publisher={Nature Publishing Group}
}
```

# DiffCSP

[COMPLETE AND EFFICIENT GRAPH TRANSFORMERS FOR CRYSTAL MATERIAL PROPERTY PREDICTION](https://arxiv.org/abs/2309.04475)

## Abstract

Crystal structures are characterized by atomic bases within a primitive unit cell that repeats along a regular lattice throughout 3D space. The periodic and infinite nature of crystals poses unique challenges for geometric graph representation learning. Specifically, constructing graphs that effectively capture the complete geometric information of crystals and handle chiral crystals remains an unsolved and challenging problem. In this paper, we introduce a novel approach that utilizes the periodic patterns of unit cells to establish the lattice-based representation for each atom, enabling efficient and expressive graph representations of crystals. Furthermore, we propose ComFormer, a SE(3) transformer designed specifically for crystalline materials. ComFormer includes two variants; namely, iComFormer that employs invariant geometric descriptors of Euclidean distances and angles, and eComFormer that utilizes equivariant vector representations. Experimental results demonstrate the state-of-the-art predictive accuracy of ComFormer variants on various tasks across three widely used crystal benchmarks.

![DiffCSP Overview](../../docs/diffcsp_overview.png)

---

## Model Description

### Overview
A periodic crystal unit cell is represented as:
- atom types (composition): $A = (a_1,\ldots,a_N)$
- fractional coordinates: $F = (f_1,\ldots,f_N),\; f_i \in [0,1)^3$ (stacked as $F \in [0,1)^{3 \times N}$)
- lattice matrix: $L \in \mathbb{R}^{3 \times 3}$

DiffCSP designs separate forward corruption processes for $(L, F)$: the lattice is diffused with a standard DDPM Gaussian process, while fractional coordinates are diffused on a 3D torus using wrapped Gaussian noise. The denoiser $\phi(L, F, A, t)$ is an EGNN-style model with periodic Fourier features on fractional coordinate differences.

### Method

#### 1) Lattice diffusion (DDPM on $L$)
Forward diffusion:

$$
q(L_t \mid L_0) = \mathcal{N}\!\left(L_t \mid \sqrt{\bar{\alpha}_t}\,L_0,\; (1-\bar{\alpha}_t) I\right),
\qquad
\bar{\alpha}_t = \prod_{s=1}^{t} (1 - \beta_s).
$$

Reparameterized sampling:

$$
L_t = \sqrt{\bar{\alpha}_t}\,L_0 + \sqrt{1 - \bar{\alpha}_t}\,\epsilon_L,
\qquad
\epsilon_L \sim \mathcal{N}(0, I).
$$

Reverse (ancestral) step:

$$
p_\theta(L_{t-1} \mid M_t) = \mathcal{N}\!\left(L_{t-1} \mid \mu_\theta(M_t, t),\; \sigma_t^2 I\right),
\qquad
M_t = (L_t, F_t, A).
$$

With mean:

$$
\mu_\theta(M_t, t) = \frac{1}{\sqrt{\alpha_t}}
\left(L_t - \frac{\sqrt{\beta_t}}{\sqrt{1 - \bar{\alpha}_t}}\,\hat{\epsilon}_L(M_t, t)\right).
$$

Lattice denoising loss:

$$
\mathcal{L}_L = \mathbb{E}_{t,\epsilon_L}\left[\left\|\epsilon_L - \hat{\epsilon}_L(M_t, t)\right\|_F^2\right].
$$

#### 2) Fractional-coordinate diffusion on a torus (wrapped Normal / score matching)
Because $F \in [0,1)^{3 \times N}$ is periodic, DiffCSP corrupts coordinates by adding Gaussian noise then wrapping back into the unit cell via a truncation/wrapping operator $w(\cdot)$:

$$
F_t = w(F_0 + \sigma_t \epsilon_F),
\qquad
\epsilon_F \sim \mathcal{N}(0, I).
$$

This implies the wrapped Normal transition density:

$$
q(F_t \mid F_0) \propto
\sum_{Z \in \mathbb{Z}^{3 \times N}}
\exp\left(
-\frac{\left\|F_t - F_0 + Z\right\|_F^2}{2 \sigma_t^2}
\right).
$$

As $\sigma_t$ increases sufficiently, $q(F_t \mid F_0)$ approaches the uniform distribution over $[0,1)^{3 \times N}$.

Score-matching objective:

$$
\mathcal{L}_F =
\mathbb{E}_{t, F_t}\left[
\lambda_t \left\|
\nabla_{F_t} \log q(F_t \mid F_0) - \hat{\epsilon}_F(M_t, t)
\right\|_F^2
\right].
$$

Sampling typically uses a predictor-corrector scheme: an ancestral predictor combined with a Langevin corrector driven by $\hat{\epsilon}_F(M_t, t)$.

#### 3) Periodic E(3)-aware denoiser (EGNN + periodic Fourier features)
DiffCSP builds $\phi(L, F, A, t)$ on a fully connected atom graph. Node initialization:

$$
h_i^{(0)} = \rho\left(f_{\text{atom}}(a_i),, f_{\text{pos}}(t)\right)
$$

Message passing at layer $s$:

$$
m_{ij}^{(s)} = \phi_m\left(
h_i^{(s-1)},, h_j^{(s-1)},, L^\top L,, \psi_{\mathrm{FT}}(f_j - f_i)
\right)
$$

$$
m_i^{(s)} = \sum_{j=1}^{N} m_{ij}^{(s)}
$$

$$
h_i^{(s)} = h_i^{(s-1)} + \phi_h\left(h_i^{(s-1)},, m_i^{(s)}\right)
$$

Periodic Fourier features for relative fractional coordinates $f = [f_1, f_2, f_3]^\top$:

$$
\psi_{\mathrm{FT}}(f)[c, k] =
\begin{cases}
\sin(2 \pi m f_c), & k = 2m \
\cos(2 \pi m f_c), & k = 2m + 1
\end{cases}
$$

which is periodic-translation invariant under wrapping.

Outputs (noise/score predictions):

$$
\hat{\epsilon}L = L,\phi_L\left(\frac{1}{N} \sum{i=1}^{N} h_i^{(S)}\right)
$$

$$
\hat{\epsilon}_F[:, i] = \phi_F\left(h_i^{(S)}\right)
$$

---

## Dataset Description

- **Perov-5**: 18,928 perovskite structures; each unit cell contains 5 atoms (ABX$_3$-like), forming a structured CSP benchmark.
- **Carbon-24**: 10,153 carbon structures; unit cells contain 6-24 atoms with all-carbon composition, useful for assessing one-to-many structure diversity.
- **MP-20**: 45,231 inorganic crystals (Materials Project subset) with at most 20 atoms per unit cell; widely used for crystal generation and CSP.
- **MPTS-52 (Materials Project Time Split)**: 40,476 crystals with up to 52 atoms per cell; chronological split for temporal generalization. A common split is 27,380 / 5,000 / 8,096 for train/val/test.

Recommended data fields for each sample:
- `atom_types`: length-$N$ atomic numbers or element indices
- `frac_coords`: $N \times 3$ fractional coordinates in $[0,1)$
- `lattice`: $3 \times 3$ lattice matrix

Optional fields include `material_id`, `spacegroup`, `energy`, and dataset split tags.

#### MP-20 split (download link)
| Dataset | Train | Val | Test |
| --- | --- | --- | --- |
| [MP-20](https://paddle-org.bj.bcebos.com/paddlematerial/datasets/mp_20/mp_20.zip) | 27136 | 9047 | 9046 |

---

## Results

| Model | Dataset | Match Rate (%) | RMS Dist | GPUs | Training Time | Config | Checkpoint / Log |
| --- | --- | --- | --- | --- | --- | --- | --- |
| diffcsp_mp20 | mp20 | 51.72 | 0.0591 | 1 | ~13.5 hours | [diffcsp_mp20.yaml](diffcsp_mp20.yaml) | [checkpoint / log](https://paddle-org.bj.bcebos.com/paddlematerial/checkpoints/structure_generation/diffcsp/diffcsp_mp20.zip) |

---

## Command

### Training
```bash
# multi-gpu training (example with 4 GPUs)
python -m paddle.distributed.launch --gpus="0,1,2,3" structure_generation/train.py -c structure_generation/configs/diffcsp/diffcsp_mp20.yaml
# single-gpu training
python structure_generation/train.py -c structure_generation/configs/diffcsp/diffcsp_mp20.yaml
```

### Validation
```bash
# Adjust program behavior on the fly using command-line parameters without modifying the configuration file directly.
# Example: --Global.do_eval=True
python structure_generation/train.py -c structure_generation/configs/diffcsp/diffcsp_mp20.yaml Global.do_eval=True Global.do_train=False Global.do_test=False Trainer.pretrained_model_path='path/to/model.pdparams'
```

### Testing
```bash
# Evaluate the model on the test dataset.
python structure_generation/train.py -c structure_generation/configs/diffcsp/diffcsp_mp20.yaml Global.do_eval=False Global.do_train=False Global.do_test=True Trainer.pretrained_model_path='path/to/model.pdparams'
```

### Sample
```bash
# Predict crystal structures using a trained model.
# Mode 1: Use a pre-trained model (downloads automatically).
# Mode 2: Use a custom configuration file and checkpoint.
# Results are saved to the folder specified by --output_path (default: results).

# Mode 1: pre-trained model
python structure_generation/sample.py --model_name='diffcsp_mp20' --weights_name='latest.pdparams' --output_path='result_diffcsp_mp20/' --chemical_formula='LiMnO2'

# Mode 2: custom config + checkpoint
python structure_generation/sample.py --config_path='structure_generation/configs/diffcsp/diffcsp_mp20.yaml' --checkpoint_path='./output/diffcsp_mp20/checkpoints/latest.pdparams' --output_path='result_diffcsp_mp20/' --chemical_formula='LiMnO2'
```

### Ab initio generation
`diffcsp_mp20_ab_initio.yaml` trains `DiffCSPWithType`, the DiffCSP variant that also diffuses
the atom types. Sampling draws the number of atoms from the MP-20 training distribution and
writes one CIF per structure.
```bash
python structure_generation/train.py -c structure_generation/configs/diffcsp/diffcsp_mp20_ab_initio.yaml
python structure_generation/sample.py --config_path='structure_generation/configs/diffcsp/diffcsp_mp20_ab_initio.yaml' --checkpoint_path='./output/diffcsp_mp20_ab_initio_t_<timestamp>_s_42/checkpoints/best.pdparams' --output_path='result_diffcsp_mp20_ab_initio/' --mode='by_dataloader'
```

---

## Citation
```
@article{jiao2023crystal,
  title={Crystal structure prediction by joint equivariant diffusion},
  author={Jiao, Rui and Huang, Wenbing and Lin, Peijia and Han, Jiaqi and Chen, Pin and Lu, Yutong and Liu, Yang},
  journal={arXiv preprint arXiv:2309.04475},
  year={2023}
}
```

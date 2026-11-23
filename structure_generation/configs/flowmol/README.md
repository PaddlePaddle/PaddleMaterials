# FlowMol

[FlowMol3: Flow Matching for 3D De Novo Small-Molecule Generation](https://arxiv.org/abs/2508.12629)

## Abstract

FlowMol jointly generates molecular topology and three-dimensional coordinates
with mixed continuous and categorical flow matching. A Geometric Vector
Perceptron network predicts atomic positions, elements, formal charges and bond
orders from an interpolated molecular graph. FlowMol3 adds self-conditioning,
fake atoms and training-time geometry distortion to improve generation of
drug-like molecules with explicit hydrogens. This implementation provides QM9
training through the existing dataset pipeline and sampling with the converted
upstream FlowMol3 GEOM-Drugs checkpoint.

![FlowMol Overview](../../docs/flowmol_fig1.png)

---

## Model Description

### Overview

A molecule is represented as:

- atom positions: $X = (x_1, \ldots, x_N),\; x_i \in \mathbb{R}^3$
- atom types (one-hot): $A \in \{0,1\}^{N \times K}$
- formal charges (one-hot): $C \in \{0,1\}^{N \times 6}$
- bond orders (one-hot, upper triangle of the complete graph): $E \in \{0,1\}^{M \times 4}$

The training objective interpolates between a prior (a centered Gaussian for
positions, the mask token for every categorical feature) and the data
endpoint, and trains a GVP message passing network to predict the endpoint
$x_1$ from the interpolated state $x_t$.

### Method

#### 1) CTMC probability paths for categorical features
Every categorical feature follows the CTMC probability path of discrete flow
matching. With the mask token as the prior state, the feature is revealed
node-by-node with probability $\alpha(t)$:

$$
p(a_t \mid a_1) = (1 - \alpha_t)\,\delta_{\mathrm{mask}} + \alpha_t\,\delta_{a_1}.
$$

The training target is the true endpoint category, restricted to nodes whose
state is still masked (`ignore_index=-100`); already revealed nodes carry no
supervision signal.

#### 2) Linear interpolation paths for positions
Positions follow the linear probability path

$$
x_t = (1 - \alpha_t)\,x_0 + \alpha_t\,x_1,
$$

where $x_0$ is a centered Gaussian prior aligned to $x_1$ with an optimal
assignment (Hungarian) followed by a rigid Kabsch alignment. The training
target is the endpoint $x_1$ (the "endpoint" parameterization of the velocity
field).

#### 3) GVP denoiser
The endpoint predictor is a Geometric Vector Perceptron (GVP) network over
the batched complete graph: scalar node features (atom/charge/time
embeddings), vector features, edge features and radial basis function
distance embeddings are processed by repeated rounds of message passing,
node position updates and edge updates. An optional self-conditioning
residual layer feeds the previous endpoint prediction back into the
network.

The `denoise_step` execution boundary supports CINN compilation of the GVP
core. Eager execution is the default; feature embeddings and the
self-conditioning residual remain eager in both modes. Training and sampling
use the same parameters across backends.

#### 4) Loss

$$
\mathcal{L} = \sum_{f \in \{x, a, c, e\}} w_f\, \mathcal{L}_f,
$$

with MSE for positions and cross entropy for the categorical features, both
optionally scaled by the time-dependent weights
$\mathrm{clip}(\alpha_t / (1-\alpha_t), 0.05, 1.5)$.

#### 5) Sampling
Sampling integrates the learned endpoint field with an Euler scheme over
`tspan = [0, 1]` (250 time points by default). At every step, masked nodes are
stochastically unmasked towards sampled endpoint categories (Campbell
discrete flow, with optional purity sampling that favors high-confidence
predictions) and can be re-masked; positions follow the Euler update.

---

## Dataset Description

The QM9 training configuration reuses `QM9Dataset` and its existing CSV splits:

| Dataset | Train | Val | Test |
| --- | --- | --- | --- |
| [QM9](https://paddle-org.bj.bcebos.com/paddlematerials/datasets/qm9/qm9.zip) | 110000 | 10000 | 10831 |

`FlowMolMolConverter` infers formal charges and kekulized bond orders from
`standard_xyz` with RDKit. `FlowMolStructure` supplies centered coordinates,
categorical features, augmentation and aligned position priors. Geometry-derived
topology can differ from the CSV SMILES. The split also differs from upstream
FlowMol's 96,334 / 19,775 / 12,903 partition, so evaluations of models trained
with this configuration are not directly comparable to upstream QM9 benchmarks.

The registered `flowmol3` checkpoint was trained upstream on GEOM-Drugs.
Its sampling dataset uses the released GEOM training atom-count histogram and
requires no additional data download. GEOM training is not provided by this
configuration, and the registered checkpoint is not a pretrained QM9 model.

---

## Results

The converted upstream GEOM-Drugs checkpoint was evaluated through
`MolecularSampler` with 1,024 generated molecules, 250 integration time points, batch
size 16 and seed 42 for Python, NumPy, Paddle and the atom-count sampler. All
other sampling settings match the linked configuration. These are Paddle
inference results, not results from training the QM9 configuration.

| Model | Dataset | Samples | Valid (%) | Unique (%) | Connected (%) | Stable Atoms (%) | Stable Molecules (%) | Config | Checkpoint |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| flowmol3 | GEOM-Drugs | 1024 | 100.00 | 99.90 | 98.93 | 100.00 | 100.00 | [flowmol3.yaml](./flowmol3.yaml) | [flowmol3.zip](https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/flowmol/flowmol3.zip) |

- **Valid**: the complete generated molecule passes RDKit sanitization; no
  largest-fragment selection or chemical repair is applied. All 1,024 molecules
  passed, and all were free of radical electrons after sanitization.
- **Unique**: 1,023 distinct canonical isomeric SMILES among 1,024 valid
  molecules, after removing explicit hydrogens. Deduplication covers all batches.
- **Connected**: 1,013 of 1,024 generated molecules have exactly one connected
  component. Disconnected molecules remain in the validity and stability counts.
- **Stable Atoms / Molecules**: bond-order sums agree with the charge-dependent
  valences observed in the upstream GEOM training data. Atom stability uses all
  generated atoms; molecule stability requires every atom in a generated molecule
  to pass. These are valence checks, not thermodynamic stability measurements.

Molecular properties below use all 1,013 valid, connected molecules, with
hydrogens removed for descriptor calculation. Values are mean +/- population
standard deviation; no ranking or property-based filtering is applied.

| Model | Molecules | QED | Molecular Weight (g/mol) | cLogP | Heavy Atoms |
| --- | --- | --- | --- | --- | --- |
| flowmol3 | 1013 | 0.7276 +/- 0.1381 | 312.35 +/- 67.63 | 2.6575 +/- 1.0932 | 21.95 +/- 4.83 |

Geometry was evaluated by MMFF94 relaxation from the original generated
coordinates, without embedding a new conformer. All 1,013 valid, connected
molecules had force-field parameters. Both iteration budgets below start from
the same original coordinates; every supported molecule contributes to each
mean, including optimizations that did not converge within the budget.

| Model | MMFF94 Iteration Limit | Converged (%) | Heavy-Atom RMSD (Angstrom) | Energy Decrease (kcal/mol/atom) |
| --- | --- | --- | --- | --- |
| flowmol3 | 200 | 72.46 | 0.6328 | 0.4285 |
| flowmol3 | 1000 | 100.00 | 0.6552 | 0.4307 |

RMSD measures displacement from generated to relaxed coordinates after rigid
alignment with fixed atom correspondence and no reflection. Energy decrease is
the generated-minus-relaxed MMFF94 energy divided by the number of atoms,
including hydrogens. The longer budget resolves optimizer nonconvergence; it
does not change the generated samples. These force-field diagnostics are not
the paper's GFN2-xTB energy or crystal reconstruction RMSD.

Novelty is not reported because the complete GEOM training molecule reference
set is unavailable in this integration. Crystal match rate and S.U.N. are not
applicable to unconditional molecular generation. This sample does not establish
reproduction of the paper's full benchmark.

---

## Command

### Training

```bash
python structure_generation/train.py -c structure_generation/configs/flowmol/flowmol_qm9.yaml
```

### Validation

```bash
python structure_generation/train.py -c structure_generation/configs/flowmol/flowmol_qm9.yaml Global.do_eval=True Global.do_train=False Global.do_test=False Trainer.pretrained_model_path='path/to/model.pdparams'
```

### Testing

```bash
python structure_generation/train.py -c structure_generation/configs/flowmol/flowmol_qm9.yaml Global.do_eval=False Global.do_train=False Global.do_test=True Trainer.pretrained_model_path='path/to/model.pdparams'
```

### Sample

```python
from ppmat.sampler import MolecularSampler

sampler = MolecularSampler(model_name="flowmol3")
results = sampler.sample_by_dataloader()
sampler.model.save_samples(results["pred"], sampler.output_dir)
```

Validation and testing require a checkpoint trained with `flowmol_qm9.yaml`.
For sampling that checkpoint, construct `MolecularSampler` with `config_path`
and `checkpoint_path`, and set `config_overrides=["Dataset.train=null"]`.
The registered GEOM checkpoint uses a different atom vocabulary.

Interpolation uses the generic `InterpolantScheduler` with linear or
power-cosine schedules and analytic time derivatives. Time-dependent loss
weighting remains in the model. Older configurations must replace
`FlowMolInterpolantScheduler` with `InterpolantScheduler`; the updated package
already contains this configuration change.

Older FlowMol weight files include unused updater-zero tensors when
`separate_mol_updaters=True` and `convs_per_update=1`. Remove keys starting with
`vector_field.node_position_updaters.0.` or `vector_field.edge_updaters.0.`
when migrating those weights for pretrained initialization. All other keys and
tensor values remain unchanged. The updated package uses
`checkpoints/best.pdparams`.

---

## Citation

```
@article{dunn2025flowmol3,
  title={FlowMol3: Flow Matching for 3D De Novo Small-Molecule Generation},
  author={Dunn, Ian and Koes, David Ryan},
  journal={arXiv preprint arXiv:2508.12629},
  year={2025}
}
```

# InfGCN

[InfGCN: Equivariant Neural Operator Learning with Graphon Convolution](https://arxiv.org/abs/2311.10908)

## Abstract

We propose a general architecture that combines a coefficient-learning scheme with a residual operator layer for learning mappings between continuous functions in 3D Euclidean space. The model is SE(3)-equivariant by design. From a graph-spectrum view, the method can be interpreted as convolution on graphons (dense graphs with infinitely many nodes), which we term InfGCN. By leveraging both the continuous graphon structure and the discrete graph structure of the input data, the model effectively captures geometric information while preserving equivariance. On large-scale electron-density datasets, InfGCN outperforms current state-of-the-art architectures, and ablation studies confirm the effectiveness of the design.

![InfGCN Overview](../../docs/infgcn.png)

---

## Model Description

### Overview
InfGCN is an operator-learning model for **electron density prediction**. Given atom types $Z = (z_1,\ldots,z_N)$ and Cartesian coordinates $R = (r_1,\ldots,r_N) \in \mathbb{R}^{N \times 3}$, the model predicts a continuous electron-density field $\rho(x)$ (typically evaluated on a 3D grid). The core idea is:
- **Atom-centered basis expansion** to represent $\rho(x)$
- **SE(3)-equivariant graphon convolution** to learn expansion coefficients
- Optional **residual operator layer** to refine global details

### Method

#### 1) Atom-centered basis expansion
(1) Atom-centered basis expansion

The density field is expanded as a sum of atom-centered basis functions:

$$
\hat{\rho}(x) = \sum_{i=1}^{N} \sum_{n=1}^{N_r} \sum_{l=0}^{l_{\max}} \sum_{m=-l}^{l}
c_{i,nlm},\phi_{nlm}(x - r_i)
$$

A common choice for $\phi_{nlm}$ is a separable radial-angular basis:

$$
\phi_{nlm}(r) = g_n(|r|),Y_{lm}!\left(\widehat{r}\right),
\qquad r = x - r_i
$$

where $g_n(\cdot)$ is a radial basis and $Y_{lm}$ are spherical harmonics. All learnable information is in the coefficients $c_{i,nlm}$.

#### 2) SE(3)-equivariant coefficient learning
Coefficients are updated with equivariant message passing:

$$
C_i^{(s)} = \sum_{j \in \mathcal{N}(i)} W_{ij}^{(s)} \odot C_j^{(s-1)}, \quad s = 1,\ldots,S
$$

Edge weights $W_{ij}^{(s)}$ depend on distance and angle features (radial basis on $\lVert r_{ij}\rVert$, spherical harmonics on $\widehat{r_{ij}}$, and an MLP). This yields rotation equivariance, permutation invariance, and physically meaningful local-to-global aggregation.

#### 3) Residual operator layer (optional)

A lightweight refinement adds a learnable correction on top of the base expansion:

$$
\hat{\rho}(x) = \hat{\rho}{\text{base}}(x) + \Delta \rho{\theta}(x)
$$

where $\Delta \rho_{\theta}$ is produced by an extra operator acting on intermediate features (for example, grid features or learned coefficients).

#### 4) Training objective and metrics

A standard regression objective minimizes an $L_2$ error over the 3D domain:

$$
\mathcal{L} = \mathbb{E}!\left[\left|\hat{\rho} - \rho\right|_2^2\right]
$$

The density is discretized on an $n \times n \times n$ grid; grid points can be subsampled for memory efficiency. A common metric is **Normalized Mean Absolute Error (NMAE)**:

$$
\mathrm{NMAE} =
\frac{\sum_{i=1}^{n^3}\left|\hat{\rho}(x_i) - \rho(x_i)\right|}
{\sum_{i=1}^{n^3}\left|\rho(x_i)\right|}
$$

---

## Dataset Description

### Datasets
- **QM9_ES**: Electron densities stored as `*.CHGCAR.lz4` under the configured `./data/data_qm9` path (train 123,835 / val 50 / test 10,000).
- **MP_ES (cubic)**: Materials Project-style crystals serialized as `.json.xz` under the configured `./data/data_cubic` path (train 14,421 / val 1,000 / test 1,000).
- **OMol25_ES**: Metal-complex cubes stored under `data/dataset_OMol25_MC_5k` (full release: train 3,943 / val 493 / test 493). The InfGCN config uses the filtered split (train 3,933 / val 491 / test 491).
- **MD17_ES**: Small molecules (for example, ethanol, benzene, phenol, resorcinol) from the MD17 electron-density release under `./data/data_md`; each config points directly to its molecule/split directory. The default config trains on ethanol, and its validation split uses the published test samples.

---

## Results

<table>
    <thead>
        <tr>
            <th nowrap="nowrap">Model Name</th>
            <th nowrap="nowrap">Dataset</th>
            <th nowrap="nowrap">Normalized MAE of Density</th>
            <th nowrap="nowrap">GPUs</th>
            <th nowrap="nowrap">Training time</th>
            <th nowrap="nowrap">Config</th>
            <th nowrap="nowrap">Checkpoint | Log</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td nowrap="nowrap">infgcn_md17_benzene</td>
            <td nowrap="nowrap">MD17_EC_Benzene</td>
            <td nowrap="nowrap">3.5418%</td>
            <td nowrap="nowrap">1</td>
            <td nowrap="nowrap">35min</td>
            <td nowrap="nowrap"><a href="../../../electronic_structure/configs/infgcn/infgcn_md17_benzene.yaml">infgcn_md17_benzene</a></td>
            <td nowrap="nowrap"><a href="https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/electronic_structure/infgcn/infgcn_md17_benzene.zip">checkpoint</td>
        </tr>
        <tr>
            <td nowrap="nowrap">infgcn_md17_ethane</td>
            <td nowrap="nowrap">MD17_EC_Ethane</td>
            <td nowrap="nowrap">4.0910%</td>
            <td nowrap="nowrap">1</td>
            <td nowrap="nowrap">27min</td>
            <td nowrap="nowrap"><a href="../../../electronic_structure/configs/infgcn/infgcn_md17_ethane.yaml">infgcn_md17_ethane</a></td>
            <td nowrap="nowrap"><a href="https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/electronic_structure/infgcn/infgcn_md17_ethane.zip">checkpoint</td>
        </tr>
        <tr>
            <td nowrap="nowrap">infgcn_md17_ethanol</td>
            <td nowrap="nowrap">MD17_EC_Ethanol</td>
            <td nowrap="nowrap">4.4674%</td>
            <td nowrap="nowrap">1</td>
            <td nowrap="nowrap">41min</td>
            <td nowrap="nowrap"><a href="../../../electronic_structure/configs/infgcn/infgcn_md17_ethanol.yaml">infgcn_md17_ethanol</a></td>
            <td nowrap="nowrap"><a href="https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/electronic_structure/infgcn/infgcn_md17_ethanol.zip">checkpoint</td>
        </tr>
        <tr>
            <td nowrap="nowrap">infgcn_md17_malonaldehyde</td>
            <td nowrap="nowrap">MD17_EC_Malonaldehyde</td>
            <td nowrap="nowrap">6.2800%</td>
            <td nowrap="nowrap">1</td>
            <td nowrap="nowrap">33min</td>
            <td nowrap="nowrap"><a href="../../../electronic_structure/configs/infgcn/infgcn_md17_malonaldehyde.yaml">infgcn_md17_malonaldehyde</a></td>
            <td nowrap="nowrap"><a href="https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/electronic_structure/infgcn/infgcn_md17_malonaldehyde.zip">checkpoint</td>
        </tr>
        <tr>
            <td nowrap="nowrap">infgcn_md17_phenol</td>
            <td nowrap="nowrap">MD17_EC_Phenol</td>
            <td nowrap="nowrap">3.5976%</td>
            <td nowrap="nowrap">1</td>
            <td nowrap="nowrap">54min</td>
            <td nowrap="nowrap"><a href="../../../electronic_structure/configs/infgcn/infgcn_md17_phenol.yaml">infgcn_md17_phenol</a></td>
            <td nowrap="nowrap"><a href="https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/electronic_structure/infgcn/infgcn_md17_phenol.zip">checkpoint</td>
        </tr>
        <tr>
            <td nowrap="nowrap">infgcn_md17_resorcinol</td>
            <td nowrap="nowrap">MD17_EC_Resorcinol</td>
            <td nowrap="nowrap">3.4942%</td>
            <td nowrap="nowrap">1</td>
            <td nowrap="nowrap">43min</td>
            <td nowrap="nowrap"><a href="../../../electronic_structure/configs/infgcn/infgcn_md17_resorcinol.yaml">infgcn_md17_resorcinol</a></td>
            <td nowrap="nowrap"><a href="https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/electronic_structure/infgcn/infgcn_md17_resorcinol.zip">checkpoint</td>
        </tr>
        <tr>
            <td nowrap="nowrap">infgcn_qm9</td>
            <td nowrap="nowrap">QM9_EC</td>
            <td nowrap="nowrap">0.9315%</td>
            <td nowrap="nowrap">1</td>
            <td nowrap="nowrap">75hour41min</td>
            <td nowrap="nowrap"><a href="../../../electronic_structure/configs/infgcn/infgcn_qm9.yaml">infgcn_qm9</a></td>
            <td nowrap="nowrap"><a href="https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/electronic_structure/infgcn/infgcn_qm9.zip">checkpoint</td>
        </tr>
        <tr>
            <td nowrap="nowrap">infgcn_mp</td>
            <td nowrap="nowrap">MP_EC (cubic)</td>
            <td nowrap="nowrap">7.7784%</td>
            <td nowrap="nowrap">1</td>
            <td nowrap="nowrap">61hour26min</td>
            <td nowrap="nowrap"><a href="../../../electronic_structure/configs/infgcn/infgcn_mp.yaml">infgcn_mp</a></td>
            <td nowrap="nowrap"><a href="https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/electronic_structure/infgcn/infgcn_mp.zip">checkpoint</td>
        </tr>
        <tr>
            <td nowrap="nowrap">infgcn_omol25_mc_5k</td>
            <td nowrap="nowrap">OMol25_EC_5k</td>
            <td nowrap="nowrap">12.6260%</td>
            <td nowrap="nowrap">4</td>
            <td nowrap="nowrap">66hour28min</td>
            <td nowrap="nowrap"><a href="../../../electronic_structure/configs/infgcn/infgcn_omol25_MC_5k_trimmed.yaml">infgcn_omol25</a></td>
            <td nowrap="nowrap"><a href="https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/electronic_structure/infgcn/infgcn_omol25_mc_5k_trimmed.zip">checkpoint</td>
        </tr>
    </tbody>
</table>

QM9_EC is evaluated on the last 1,600 test molecules without rotation, MP_EC (cubic) on all 1,000 test structures and each MD17_EC molecule on its full test split without rotation; all use every grid point with a positive reference density and average the NMAE per sample.

---

## Command

### Training
```bash
# multi-gpu training (example with 4 GPUs)
python -m paddle.distributed.launch --gpus="0,1,2,3" electronic_structure/train.py -c electronic_structure/configs/infgcn/infgcn_qm9.yaml
# single-gpu training
python electronic_structure/train.py -c electronic_structure/configs/infgcn/infgcn_qm9.yaml
```

### Validation
```bash
# Enable eval-only mode with a saved checkpoint.
python electronic_structure/train.py -c electronic_structure/configs/infgcn/infgcn_qm9.yaml Global.do_eval=True Global.do_train=False Global.do_test=False Trainer.pretrained_model_path='path/to/model.pdparams'
```

### Testing
```bash
# Evaluate on the test split using a pretrained checkpoint.
python electronic_structure/train.py -c electronic_structure/configs/infgcn/infgcn_qm9.yaml Global.do_eval=False Global.do_train=False Global.do_test=True Trainer.pretrained_model_path='path/to/model.pdparams'
```

### Prediction
```bash
# 1) One-click inference with a registered pretrained model and bundled molecule.
python electronic_structure/predict.py \
  --model_name infgcn_qm9 \
  --weights_name best.pdparams \
  --input_format=mol --input_path electronic_structure/example_data/methane.mol \
  --grid_shape 8 \
  --grid_batch_size 4096 \
  --output_path output/infgcn_qm9/methane \
  --save_html

# 2) MOL-file inference (single file or directory).
# This mode predicts electron density from molecular structure files (*.mol).
python electronic_structure/predict.py \
  --config_path electronic_structure/configs/infgcn/infgcn_omol25_MC_5k_trimmed.yaml \
  --checkpoint_path path/to/infgcn_omol25.pdparams \
  --input_format=mol --input_path path/to/mols_or_mol_file \
  --output_path output/infgcn_mol

# 3) XYZ-file inference on a molecular bounding-box grid.
python electronic_structure/predict.py \
  --model_name infgcn_omol25_mc_5k_trimmed \
  --weights_name best.pdparams \
  --input_format=xyz --input_path property_prediction/example_data/molecules/isoguvacine.xyz \
  --grid_shape 80 \
  --output_path output/infgcn_omol25/xyz

# 4) Predict a periodic crystal field from CIF on a full-cell grid.
python electronic_structure/predict.py \
  --model_name infgcn_mp \
  --weights_name best.pdparams \
  --input_format=cif --input_path property_prediction/example_data/cifs/mp-18767-LiMnO2.cif \
  --grid_shape 80 \
  --output_path output/infgcn_mp/cif

# 5) Reuse the structure and native grid from a QM9 CHGCAR test sample.
python electronic_structure/predict.py \
  --model_name infgcn_qm9 \
  --weights_name best.pdparams \
  --input_format=chgcar --input_path electronic_structure/example_data/ammonia.CHGCAR.lz4 \
  --output_path output/infgcn_qm9/chgcar

# 6) Reuse the structure and periodic grid from an MP density JSON test sample.
python electronic_structure/predict.py \
  --model_name infgcn_mp \
  --weights_name best.pdparams \
  --input_format=json --input_path electronic_structure/example_data/mg3dy_mp-1546.json.xz \
  --output_path output/infgcn_mp/json

# 7) Reuse the structure and native grid from an OMol25 CUBE test sample.
python electronic_structure/predict.py \
  --model_name infgcn_omol25_mc_5k_trimmed \
  --weights_name best.pdparams \
  --input_format=cube --input_path electronic_structure/example_data/c38h40eun9op.cube.lz4 \
  --output_path output/infgcn_omol25/cube
```

Notes:
- Replace `path/to/*.pdparams` with a downloaded pretrained checkpoint or a checkpoint produced by training.
- Model configs keep the recommended `grid_batch_size` and share the configured
  field builder between Dataset and Predict. Input selection, output
  paths, CUBE export, and MOL grid settings are runtime command-line options.
- `--input_path` with `--input_format=mol` supports either one `.mol` file or a directory of `.mol`
  files. The atom vocabulary declared by `Vocabulary` is downloaded and used
  automatically.
- `--input_path` with `--input_format=xyz` supports one `.xyz` file or a directory and uses the same
  molecular bounding-box grid controls as MOL input.
- `--input_path` with `--input_format=cif` supports one `.cif` file or a directory and builds a periodic
  `GridSpec` spanning the complete unit cell.
- `--input_path` with `--input_format=cube`, `chgcar`, or `json` also accepts
  one file or a directory. These paths reuse the source structure and exact
  `GridSpec`; the reference density values are parsed but are not model inputs.
- `--output_path` is an output directory and always receives the predicted CUBE.
- The task-level `predict.py` supports `--visualize` for PNG, `--save_html` for
  interactive HTML, and `--show_plot` for an interactive Plotly window. It invokes
  the reusable `ppmat.visualization` package after prediction, while
  `FieldPredictor` remains independent of visualization.
- Static PNG export uses Plotly Kaleido and requires Chrome; run
  `plotly_get_chrome` once if Chrome is not already available.
- `--grid_shape` controls MOL, XYZ, and CIF grids (default `80,80,80`), while
  `--grid_padding` applies to molecular bounding-box grids (default `6.0` Angstrom).
- MOL/XYZ input coordinates and `--grid_padding` are interpreted in Angstrom.
- File and directory inputs both return a list of prediction dictionaries.

The bundled field examples are the first entries of their published test splits:

| Dataset | Test entry | Bundled input |
|---|---:|---|
| QM9 ES | `1` → `000002` | `ammonia.CHGCAR.lz4` |
| MP ES cubic | `mp-1546` | `mg3dy_mp-1546.json.xz` |
| OMol25 MC 5k | `4437` | `c38h40eun9op.cube.lz4` |

---

## Citation
```
@article{cheng2023infgcn,
  title={Equivariant neural operator learning with graphon convolution},
  author={Cheng, Chaoran and Peng, Jian},
  journal={arXiv preprint arXiv:2311.10908},
  year={2023}
}
```

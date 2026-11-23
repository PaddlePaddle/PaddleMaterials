# PaddleMaterials

<p align="center">
  <img src="docs/ppmat_logo.png" alt="PaddleMaterials" width="400">
</p>

<p align="center">
  <a href="https://paddlepaddle.github.io/PaddleMaterials/">
    <img alt="Homepage" src="https://img.shields.io/badge/Homepage-PaddleMaterials-3B82F6?style=flat">
  </a>
  <a href="Install.md">
    <img alt="Python 3.10+" src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&amp;logoColor=white">
  </a>
  <a href="https://pypi.org/project/ppmat/">
    <img alt="PyPI version" src="https://img.shields.io/pypi/v/ppmat?logo=pypi&amp;logoColor=white&amp;cacheSeconds=300">
  </a>
  <a href="LICENSE">
    <img alt="Apache 2.0 License" src="https://img.shields.io/github/license/PaddlePaddle/PaddleMaterials">
  </a>
  <a href="https://github.com/PaddlePaddle/PaddleMaterials/stargazers">
    <img alt="GitHub Stars" src="https://img.shields.io/github/stars/PaddlePaddle/PaddleMaterials?style=flat&amp;logo=github">
  </a>
</p>

<p align="center">
  <a href="README.md"><img src="https://img.shields.io/badge/English-blue" alt="English"></a>
  <a href="README_zh.md"><img src="https://img.shields.io/badge/简体中文-lightgrey" alt="简体中文"></a>
  <a href="README_ja.md"><img src="https://img.shields.io/badge/日本語-lightgrey" alt="日本語"></a>
</p>

## 🚀 Introduction

**PaddleMaterials** is an end-to-end AI4Materials toolkit built on the **PaddlePaddle** deep learning framework. Designed as a data-mechanism dual-driven platform for developing and deploying foundation models in materials science, **PPMat** enables researchers to efficiently build AI models and accelerate material discovery using pretrained models.

<p align="left">
 <img src="docs/overview_en.png" align="middle" width = "1000"/>
<p align="left">

### 🧩 Core Capabilities

| Task | Description | Typical Applications |
|------|-------------|---------------------|
| **Property Prediction (PP)** | Predict material properties from structure | Forward design or predict formation energy, band gap, elastic moduli etc. |
| **Structure Generation (SG)** | Generate novel crystal structures | Inverse design or structure generation |
| **Machine Learning Interatomic Potential (MLIP)** | Surrogate Model for DFT as ML potentials | Molecular dynamics simulations |
| **Electronic Structure (ES)** | Surrogate Model for DFT to predict physical field  | Predict electronic density |
| **Spectrum Elucidation (SE)** | Reconstruct structures from spectra | NMR structure elucidation |
| **Spectrum Enhancement (SPEN)** | Enhance microscopy and spectrum signals | STEM image enhancement, denoising |

### 🧱 Supported Materials

- **Inorganic Crystals** - Well-supported with multiple datasets and pretrained models
- **Organic Molecules** - Support for multiple datasets and pretrained models including small molecules and partial polymers

### ✨ Why PaddleMaterials?

- ✅ **Rich Pretrained Models & AI-ready Datasets** - 50+ pretrained models ready for inference and Multiple curated datasets for training
- ✅ **Multi-Task Integration** - Unified framework across tasks of PP, SG, MLIP, ES, SE, SPEN etc.
- ✅ **Multi-Hardware Support** - Full support for NVIDIA GPUs and MetaX GPUs and Intel CPUs
- ✅ **Production-Ready** - Easy to use with standandlize design & distributed training, mixed precision, checkpoint recovery

### 📑 Support Tasks

| Task | Description | Link |
|------|-------------|------|
| **Property Prediction (PP)** | Predict formation energy, band gap, elastic properties | [README](property_prediction/README.md) |
| **Structure Generation (SG)** | Generate new crystal structures with diffusion models | [README](structure_generation/README.md) |
| **Machine Learning Interatomic Potential (MLIP)** | DFT-accurate potentials for molecular dynamics | [README](interatomic_potentials/README.md) |
| **Electronic Structure (ES)** | Predict electronic structure properties | [README](electronic_structure/README.md) |
| **Spectrum Elucidation (SE)** | Reconstruct molecular structures from NMR spectra | [README](spectrum_elucidation/README.md) |
| **Spectrum Enhancement (SPEN)** | Enhance microscopy and spectral signals | [README](spectrum_enhancement/README.md) |

### 🤖 Available Pretrained Models

| Task | Models | Dataset |
|------|--------|---------|
| **Property Prediction**                       | MEGNet, iComformer, DimeNet++, SphereNet | MP2018, MP2024, JARVIS, QM9, etc.|
| **Structure Generation**                      | MatterGen, DiffCSP                       | MP20, ALEX, etc.|
| **Machine Learning Interatomic Potential**    | CHGNet, MatterSim, SphereNet             | MPTRJ, MD17, etc.|
| **Electronic Structure**                      | InfGCN                                   | QM9_ES, MP_ES, OMol25_MC_ES, etc.|
| **Spectrum Elucidation**                      | DiffNMR                                  | MSD_NMR, etc.|
| **Spectrum Enhancement**                      | SFIN                                     | SFIN-HAADF/BF, etc.|

Full model list: See [MODEL_REGISTRY](ppmat/models/__init__.py#L75)

---

## 🚀 Get Started

### 🔧 Installation

Please refer to the installation [document](Install.md) for your hardware environment. See [SupportedHardwareList](./docs/multi_device.md) for more multi-hardware adaptation information.

---

### ⚡ Easy Inference

#### Property Prediction

Predict material formation energy using a pretrained MEGNet model:

```bash
python property_prediction/predict.py \
    --model_name='megnet_mp2018_train_60k_e_form' \
    --weights_name='best.pdparams' \
    --input_format='cif' \
    --input_path='./property_prediction/example_data/cifs/' \
    --output_path='result_property_prediction/'
```

#### Structure Generation

Generate novel crystal structures using a pretrained MatterGen model:

```bash
python structure_generation/sample.py \
    --model_name='mattergen_mp20' \
    --weights_name='latest.pdparams' \
    --output_path='result_mattergen_mp20/' \
    --mode='by_num_atoms' \
    --num_atoms=4
```

#### Interatomic Potentials

Predict energy and forces using a pretrained MatterSim model:

```bash
python interatomic_potentials/predict.py \
    --model_name='mattersim_1M' \
    --weights_name='mattersim-v1.0.0-1M_model.pdparams' \
    --input_format='cif' \
    --input_path='./interatomic_potentials/example_data/cifs/' \
    --output_path='result_interatomic_potentials/'
```

#### Electronic Structure

Predict electron density from the bundled methane example using a pretrained InfGCN
model:

```bash
python electronic_structure/predict.py \
    --model_name='infgcn_qm9' \
    --weights_name='best.pdparams' \
    --input_format='mol' \
    --input_path='electronic_structure/example_data/methane.mol' \
    --grid_shape=8 \
    --grid_batch_size=4096 \
    --output_path='output/infgcn_qm9/methane'
```

See the [InfGCN prediction guide](electronic_structure/configs/infgcn/README.md#prediction)
for registered-model and local-checkpoint inference from MOL, CUBE, CHGCAR, and
density JSON inputs. Real test-split field examples are bundled under
`electronic_structure/example_data/`.

#### Spectrum Elucidation

Run NMR spectrum elucidation using the bundled example and a pretrained DiffNMR
model:

```bash
python spectrum_elucidation/sample.py \
    --model_name='diffnmr_msdnmr_nless15' \
    --weights_name='best.pdparams' \
    --output_path='result_diffnmr_nless15/'
```

#### Spectrum Enhancement

Enhance STEM images using a pretrained SFIN model:

```bash
python spectrum_enhancement/predict.py \
    --model_name='sfin_haadf_enhance' \
    --weights_name='best.pdparams' \
    --input_path='spectrum_enhancement/example_data/sfin_haadf.png' \
    --output_path='result_sfin/'
```

---

### 🏋️ Start Training

For training and fine-tuning, refer to the [documentation](get_started.md).

---

## 🤝 Contributors & Cooperation & Community

[![Star History Chart](https://api.star-history.com/svg?repos=PaddlePaddle/PaddleMaterials&type=date&legend=top-left)](https://www.star-history.com/#PaddlePaddle/PaddleMaterials&type=date&legend=top-left)

Thanks to all contributors who have helped build PaddleMaterials！
<a href="https://github.com/PaddlePaddle/PaddleMaterials/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=PaddlePaddle/PaddleMaterials" />
</a>

Thanks for the following organiziton for cooprative support!
<p align="left">
 <img src="docs/logo_SZNL_2.jpeg" align="middle" width = "180"/>
 <img src="docs/logo_SinochemDI_2.jpeg" align="middle" width = "180"/>
 <img src="docs/logo_MetaX.png" align="middle" width = "140"/>
<p align="left">

Join the PaddleMaterials WeChat group to discuss with us!
<p align="left">
 <img src="docs/wechat_group.png" align="middle" width = "100"/>
<p align="left">

## 🛠️ Contribute to PaddleMaterials

For developer, please refer to [architecture](docs/ARCHITECTURE_ch.md).

---

## 📜 License

PaddleMaterials is licensed under the [Apache License 2.0](LICENSE).

---

## 🎓 Citation

```bibtex
@misc{liu2026paddlematerialsendtoenddeeplearning,
  title={PaddleMaterials: An End-to-End Deep Learning Toolkit for Materials Science},
  author={Xuwei Liu and Zhimin Zhang and Xiaoguang Hu and Zhongwei Liu and Yuanhui Sun and Jun Huo and Hua Wu and Haifeng Wang},
  year={2026},
  eprint={2611.03154},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2611.03154},
}
```

---

## 🙏 Acknowledgements

This repository references code from the following projects:

[PaddleScience](https://github.com/PaddlePaddle/PaddleScience) |
[Matgl](https://github.com/materialsvirtuallab/matgl) |
[CDVAE](https://github.com/txie-93/cdvae) |
[DiffCSP](https://github.com/jiaor17/DiffCSP) |
[MatterGen](https://github.com/microsoft/mattergen) |
[MatterSim](https://github.com/microsoft/mattersim) |
[CHGNet](https://github.com/CederGroupHub/chgnet) |
[AIRS](https://github.com/divelab/AIRS)

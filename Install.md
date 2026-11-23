# Installation 🔧

[简体中文](./Install_cn.md)

## 1. Installation Instructions

We recommend using a conda virtual environment to manage dependencies. You can install conda via [Miniforge](https://github.com/conda-forge/miniforge).

### 1.1 Create Virtual Environment
Create and activate a new conda virtual environment:

    conda create -n ppmat python=3.10
    conda activate ppmat

We currently develop under Python 3.10 environment and recommend using Python 3.10 or newer.

### 1.2 Install PaddlePaddle
Install the appropriate PaddlePaddle version based on your CUDA version. Refer to the [PaddlePaddle Official Website](https://www.paddlepaddle.org.cn/install/quick) for installation commands. We recommend installing PaddlePaddle version >= 3.1 or the develop version.

For example, in a CUDA 12.6 environment, install the paddlepaddle-gpu version:

    python -m pip install paddlepaddle-gpu==3.1.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/

After installation, verify the installation with:

    python -c "import paddle; paddle.utils.run_check()"

If you see "PaddlePaddle is installed successfully! Let's start deep learning with PaddlePaddle now.", the installation was successful.

### 1.3 Install PaddleMaterials

Install the released package from PyPI:

    python -m pip install ppmat

Verify that PaddleMaterials is imported from the installed package:

    python -c "import ppmat; print(ppmat.__version__); print(ppmat.__file__)"

For development, install PaddleMaterials from source:

    # Clone PaddleMaterials repository
    git clone https://github.com/PaddlePaddle/PaddleMaterials.git

    # Navigate to PaddleMaterials directory
    cd PaddleMaterials

    # Install dependencies
    pip install --upgrade pip setuptools==68.2.2 wheel
    pip install setuptools_scm
    pip install Cython

    # Install in editable mode
    pip install -e . --no-build-isolation
    # pip install -e . --no-build-isolation -i https://pypi.tuna.tsinghua.edu.cn/simple recommended if you are in China


## 2. Run Examples

The task scripts, configuration files, and example data are maintained in the source
repository and are not included in the `ppmat` wheel. Clone the repository and run
the following commands from its root directory.

### 2.1 Property Prediction

Predict material formation energy using a pretrained MEGNet model:

```bash
python property_prediction/predict.py \
    --model_name='megnet_mp2018_train_60k_e_form' \
    --weights_name='best.pdparams' \
    --input_format=cif --input_path='./property_prediction/example_data/cifs/' \
    --output_path='results/'
```

### 2.2 Structure Generation

Generate crystal structures with four atoms using a pretrained MatterGen model:

```bash
python structure_generation/sample.py \
    --model_name='mattergen_mp20' \
    --weights_name='latest.pdparams' \
    --output_path='result_mattergen_mp20/' \
    --mode='by_num_atoms' \
    --num_atoms=4
```

### 2.3 Interatomic Potentials

Predict energy and forces using a pretrained MatterSim model:

```bash
python interatomic_potentials/predict.py \
    --model_name='mattersim_1M' \
    --weights_name='mattersim-v1.0.0-1M_model.pdparams' \
    --input_format=cif --input_path='./interatomic_potentials/example_data/cifs/' \
    --output_path='results/'
```

### 2.4 Electronic Structure

Predict electron density using a trained InfGCN checkpoint:

```bash
python electronic_structure/predict.py \
    --model_name='infgcn_qm9' \
    --weights_name='best.pdparams' \
    --input_format=mol --input_path='electronic_structure/example_data/methane.mol' \
    --grid_shape=8 \
    --grid_batch_size=4096 \
    --output_path='output/infgcn_qm9/methane'
```

Prepare the dataset and checkpoint as described in the
[InfGCN prediction guide](electronic_structure/configs/infgcn/README.md#prediction).

### 2.5 Spectrum Elucidation

Run NMR spectrum elucidation using a trained DiffNMR checkpoint:

```bash
python spectrum_elucidation/sample.py \
    --model_name='diffnmr_msdnmr_nless15' \
    --weights_name='best.pdparams' \
    --output_path='result_diffnmr_nless15/'
```

### 2.6 Spectrum Enhancement

Enhance STEM images using a pretrained SFIN model:

```bash
python spectrum_enhancement/predict.py \
    --model_name='sfin_haadf_enhance' \
    --weights_name='best.pdparams' \
    --input_path='path/to/noisy_image.png' \
    --output_path='result_sfin/'
```

For more usage instructions, refer to the task-specific README files or the
[Get Started](./get_started.md) documentation.

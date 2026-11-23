
# Installation 🔧

[English](./Install.md)

## 1. 安装说明

我们推荐使用conda虚拟环境来管理依赖包，你可以通过安装[Miniforge](https://github.com/conda-forge/miniforge)使用conda。

### 1.1 创建虚拟环境
创建一个新的conda虚拟环境，并激活环境：

    conda create -n ppmat python=3.10
    conda activate ppmat

目前我们在python 3.10环境下进行开发，因此建议使用python 3.10或者更高的版本。

### 1.2 安装PaddlePaddle
根据你的cuda版本安装对应版本的PaddlePaddle，具体安装命令可参考[PaddlePaddle官网](https://www.paddlepaddle.org.cn/install/quick)。我们推荐安装PaddlePaddle >= 3.1或者develop版本。

例如，对于cuda12.6环境，安装paddlepaddle-gpu版本：

     python -m pip install paddlepaddle-gpu==3.1.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/

安装完毕之后，运行以下命令，验证 Paddle 是否安装成功。

    python -c "import paddle; paddle.utils.run_check()"

如果出现 PaddlePaddle is installed successfully! Let's start deep learning with PaddlePaddle now. 信息，说明已成功安装。

### 1.3 安装 PaddleMaterials

从 PyPI 安装已发布的软件包：

    python -m pip install ppmat

验证 PaddleMaterials 是否从已安装的软件包导入：

    python -c "import ppmat; print(ppmat.__version__); print(ppmat.__file__)"

如需开发 PaddleMaterials，请从源码安装：

    # clone PaddleMaterials
    git clone https://github.com/PaddlePaddle/PaddleMaterials.git

    # 切换到PaddleMaterials目录
    cd PaddleMaterials

    # 安装依赖
    pip install --upgrade pip setuptools==68.2.2 wheel
    pip install setuptools_scm
    pip install Cython

    # 以可编辑模式安装PaddleMaterials
    pip install -e . --no-build-isolation -i https://pypi.tuna.tsinghua.edu.cn/simple


## 2. 运行示例

任务脚本、配置文件和示例数据保存在源码仓库中，不包含在 `ppmat` wheel 内。
请先克隆源码仓库，并在仓库根目录执行以下命令。

### 2.1 分子与材料性质预测

使用预训练 MEGNet 模型预测材料形成能：

```bash
python property_prediction/predict.py \
    --model_name='megnet_mp2018_train_60k_e_form' \
    --weights_name='best.pdparams' \
    --input_format=cif --input_path='./property_prediction/example_data/cifs/' \
    --output_path='results/'
```

### 2.2 结构生成

使用预训练 MatterGen 模型生成包含四个原子的晶体结构：

```bash
python structure_generation/sample.py \
    --model_name='mattergen_mp20' \
    --weights_name='latest.pdparams' \
    --output_path='result_mattergen_mp20/' \
    --mode='by_num_atoms' \
    --num_atoms=4
```

### 2.3 机器学习势函数

使用预训练 MatterSim 模型预测能量和力：

```bash
python interatomic_potentials/predict.py \
    --model_name='mattersim_1M' \
    --weights_name='mattersim-v1.0.0-1M_model.pdparams' \
    --input_format=cif --input_path='./interatomic_potentials/example_data/cifs/' \
    --output_path='results/'
```

### 2.4 电子结构预测

使用训练完成的 InfGCN 权重预测电子密度：

```bash
python electronic_structure/predict.py \
    --model_name='infgcn_qm9' \
    --weights_name='best.pdparams' \
    --input_format=mol --input_path='electronic_structure/example_data/methane.mol' \
    --grid_shape=8 \
    --grid_batch_size=4096 \
    --output_path='output/infgcn_qm9/methane'
```

数据集和权重准备方式请参考
[InfGCN 预测文档](electronic_structure/configs/infgcn/README.md#prediction)。

### 2.5 谱图解析

使用训练完成的 DiffNMR 权重进行 NMR 谱图解析：

```bash
python spectrum_elucidation/sample.py \
    --model_name='diffnmr_msdnmr_nless15' \
    --weights_name='best.pdparams' \
    --output_path='result_diffnmr_nless15/'
```

### 2.6 谱图增强

使用预训练 SFIN 模型增强 STEM 图像：

```bash
python spectrum_enhancement/predict.py \
    --model_name='sfin_haadf_enhance' \
    --weights_name='best.pdparams' \
    --input_path='path/to/noisy_image.png' \
    --output_path='result_sfin/'
```

更多使用说明请参考各任务 README 或 [Get Started](./get_started.md)。

# PaddleMaterials 架构说明

## 项目概述

**PaddleMaterials**（简称PPMat）是一个基于飞桨（PaddlePaddle）深度学习框架的AI4Materials（人工智能驱动材料科学）端到端工具包。它是一个数据-机理双驱动的材料科学基础模型开发部署平台，支持无机材料、有机分子、聚合物等多种材料类型的研究与开发。

---

## 目录结构

```
PaddleMaterials/
├── ppmat/                          # 核心Python包
│   ├── calculator/                 # ASE计算器集成
│   ├── datasets/                   # 数据集处理模块
│   ├── losses/                     # 损失函数
│   ├── metrics/                    # 评估指标
│   ├── models/                     # 核心模型实现
│   ├── optimizer/                  # 优化器
│   ├── predictor/                  # 预测器
│   ├── sampler/                    # 采样器
│   ├── schedulers/                 # 扩散调度器
│   ├── trainer/                    # 训练框架
│   └── utils/                      # 工具函数
│
├── property_prediction/            # 性质预测任务
├── structure_generation/           # 结构生成任务
├── interatomic_potentials/         # 机器学习原子间势(MLIP)
├── electronic_structure/           # 机器学习电子结构(MLES)
├── spectrum_elucidation/           # 谱图解析任务(SE)
├── ppmatSim/                       # 分子动力学模拟工具
├── research/                       # 研究项目
├── jointContribution/              # 联合贡献项目
├── docs/                           # 文档和资源
├── test/                           # 测试文件
├── setup.py                        # 安装配置
├── requirements.txt                # 依赖列表
└── README.md                       # 主说明文档
```

---

## 核心模块详解

### 1. ppmat/models/ - 深度学习模型库

| 模型 | 路径 | 功能描述 |
|------|------|----------|
| **MEGNetPlus** | `ppmat/models/megnet/` | 等变图网络，材料性质预测 |
| **iComformer** | `ppmat/models/comformer/` | 改进版ComFormer，晶体性质预测 |
| **DimeNet++** | `ppmat/models/dimenetpp/` | 方向消息传递网络 |
| **MatterGen** | `ppmat/models/mattergen/` | 条件扩散模型，晶体结构生成 |
| **DiffCSP** | `ppmat/models/diffcsp/` | 扩散晶体结构预测 |
| **CHGNet** | `ppmat/models/chgnet/` | 电荷图神经网络，原子间势 |
| **MatterSim** | `ppmat/models/mattersim/` | 通用原子间势函数 |
| **InfGCN** | `ppmat/models/infgcn/` | 电子结构预测 |
| **DiffNMR** | `ppmat/models/diffnmr/` | NMR谱图解析 |

### 2. ppmat/datasets/ - 数据集处理

- **MP2018Dataset** - Materials Project 2018数据集
- **MP2024Dataset** - Materials Project 2024数据集
- **MP20Dataset** - 材料结构生成基准数据集
- **JarvisDataset** - JARVIS材料数据集
- **QM9Dataset** - 有机分子数据集
- **OC20S2EFDataset** - Open Catalyst 2020数据集
- **MSDnmrDataset** - NMR谱图数据集

### 3. ppmat/trainer/ - 训练框架

- **base_trainer.py** - 基础训练器，支持：
  - 分布式训练（多GPU）
  - 混合精度训练
  - 断点续训
  - 学习率调度
  - 早停机制

### 4. ppmat/utils/ - 工具函数

包含晶体结构处理、可视化、模型保存/加载、日志记录等通用工具。

---

## 任务模块

### property_prediction/ - 性质预测

**功能**：预测材料的各种物理化学性质

**支持任务**：
- 形成能（Formation Energy）
- 带隙（Band Gap）
- 剪切模量（Shear Modulus）
- 体积模量（Bulk Modulus）

**配置文件**：`property_prediction/configs/`（包含 megnet、comformer、dimenet++ 配置）

### structure_generation/ - 结构生成

**功能**：生成新型晶体结构

**模型**：
- MatterGen - 无条件/条件结构生成
- DiffCSP - 扩散晶体结构预测

**配置文件**：`structure_generation/configs/`

### interatomic_potentials/ - 原子间势

**功能**：机器学习原子间势（MLIP）计算

**模型**：
- CHGNet - 电荷图神经网络势
- MatterSim - 通用原子间势

**配置文件**：`interatomic_potentials/configs/`

### electronic_structure/ - 电子结构

**功能**：机器学习电子结构（MLES）预测

**模型**：
- InfGCN - 推断图卷积网络

**配置文件**：`electronic_structure/configs/`

### spectrum_elucidation/ - 谱图解析

**功能**：谱图到结构的解析

**模型**：
- DiffNMR - NMR谱图到结构解析

**配置文件**：`spectrum_elucidation/configs/`

---

## 技术栈

### 深度学习框架
- **PaddlePaddle >= 3.1** - 核心深度学习框架

### 主要依赖

| 类别 | 依赖包 | 用途 |
|------|--------|------|
| 图神经网络 | pgl 2.2.6 | 图学习库 |
| 科学计算 | numpy 1.26.4, scipy 1.13.1 | 数值计算 |
| 材料科学 | pymatgen 2024.10.29 | 材料分析 |
| | ase 3.23.0 | 原子模拟环境 |
| | matminer 0.9.2 | 材料数据挖掘 |
| 分子处理 | rdkit 2024.9.1 | 分子化学信息学 |
| 配置管理 | hydra-core 1.3.2 | 配置管理 |
| 可视化 | tensorboardX, visualdl, wandb | 训练可视化 |
| 数据处理 | pandas, pyarrow, lmdb | 数据处理与存储 |

### 支持的硬件
- NVIDIA GPU（CUDA 12.x）
- MetaX GPU（国产GPU）
- CPU

---

## 配置系统

项目使用 **Hydra** 进行配置管理，所有任务都通过YAML配置文件进行设置。

### 配置继承

支持配置继承和组合，例如：

```yaml
defaults:
  - model: megnet
  - dataset: mp2018
  - optimizer: adam
```

---

## 预训练模型

项目提供丰富的预训练模型（MODEL_REGISTRY中定义）：

### 性质预测模型
- `megnet_mp2018_train_60k_e_form` - 形成能预测
- `comformer_mp2018_train_60k_band_gap` - 带隙预测
- `dimenetpp_mp2018_train_60k_G` - 剪切模量预测

### 结构生成模型
- `mattergen_mp20` - 无条件结构生成
- `diffcsp_mp20` - 扩散晶体结构预测

### 原子间势模型
- `chgnet_mptrj` - 通用势函数
- `mattersim_1M/5M` - MatterSim势函数

---

## 使用示例

### 性质预测

```bash
python property_prediction/predict.py \n    --model_name='megnet_mp2018_train_60k_e_form' \n    --weights_name='best.pdparams' \n    --input_format=cif --input_path='./property_prediction/example_data/cifs/'
```

### 多GPU训练

```bash
python -m paddle.distributed.launch --gpus="0,1,2,3" \n    property_prediction/train.py \n    -c property_prediction/configs/megnet/megnet_mp2018_train_60k_e_form.yaml
```

---

## 开发指南

### 添加新模型

1. 在 `ppmat/models/` 下创建新模型目录
2. 继承基础模型类，实现 `forward` 方法
3. 在 `ppmat/models/__init__.py` 中注册模型
4. 创建对应的配置文件

### 添加新数据集

1. 在 `ppmat/datasets/` 下创建数据集类
2. 继承 `BaseDataset`，实现 `get_data` 方法
3. 在 `ppmat/datasets/__init__.py` 中注册数据集

### 添加新任务

1. 在根目录创建新任务目录
2. 实现 `train.py`, `predict.py` 等入口脚本
3. 创建 `configs/` 目录存放配置
4. 在 `README.md` 中添加任务说明

---

## 文档索引

| 文档 | 路径 | 说明 |
|------|------|------|
| 安装说明 | `Install.md` | 英文安装指南 |
| 中文安装说明 | `Install_cn.md` | 中文安装指南 |
| 快速开始 | `get_started.md` | 快速入门教程 |
| 配置说明 | `about_configs.md` | Hydra配置详解 |
| 多硬件支持 | `docs/multi_device.md` | 多硬件适配说明 |
| MetaX支持 | `docs/MetaX/` | MetaX GPU适配文档 |

---

## 许可证

本项目采用 [Apache License 2.0](LICENSE) 开源许可证。

---

## 联系方式

- 项目主页：https://github.com/PaddlePaddle/PaddleMaterials
- 问题反馈：https://github.com/PaddlePaddle/PaddleMaterials/issues

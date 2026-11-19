# LiFlow

[Flow Matching for Accelerated Simulation of Atomic Transport in Crystalline Materials](https://doi.org/10.1038/s42256-025-01125-4)

移植自 [learningmatter-mit/liflow](https://github.com/learningmatter-mit/liflow)，参考实现固定于 commit [`e6fc475`](https://github.com/learningmatter-mit/liflow/tree/e6fc475361d046865f12cae1aee11c4f56c48d87)。原实现为 MIT License，版权与许可声明随移植代码保留在各文件头部（见 [License](#license)）。

## Abstract

分子动力学模拟中，原子输运（如锂离子导体中的 Li 扩散）需要大量短时间步的 SDE 采样，计算代价高。LiFlow 把体系随时间的演化看成一个在时间延迟帧对之间搬运概率密度的连续流，用流匹配（flow matching）直接学习速度场，一次积分即可从当前帧推进到时间延迟后的目标帧，替代逐步 SDE 采样。

实现包含两个共享同一 `DualPaiNN` 主干、但条件分布不同的模型：

- **propagator**：以时间延迟帧对的起始帧 `x_start` 加一个自适应的 Maxwell-Boltzmann 先验样本为输入，学习输运到 `t` 时刻后的终止帧 `x_end` 的速度场（对应原实现的 `TimeDelayedPairDataset`）。
- **corrector**：对单帧加噪后修正：输入是加噪副本，邻居表由加噪位置构建，目标是无噪帧，先验为 `UniformScaleNormalPrior(0.25)` 与 `MaxwellBoltzmannPrior(0.1)`（对应原实现的 `NoisedFrameDataset`）。

轨迹积分由 [`IntegratorPredictor`](../../../ppmat/predictor/integrator_predictor.py) 完成：只看初始结构、晶格、温度、元素表和随机种子，不使用任何未来帧，用 Euler 或 Heun 步积分 propagator 速度场，每 `corrector_every` 步可选施加一次 corrector。

## Datasets

数据集按"一个温度一份轨迹文件、一条轨迹一行索引"组织。数据目录（两个训练配置中的 `path` 均为 `data/liflow`）包含：

| 文件 | 内容 |
| --- | --- |
| `train.csv` / `val.csv` / `test.csv` | 划分索引，每条轨迹一行，列为 `name, temp, t_start, t_end, prior_Li, prior_frame, comp` |
| `positions_{temp}K.npz` | 各温度下的轨迹帧，键为轨迹名，值为 `(T, N, 3)` 的坐标数组 |
| `atomic_numbers.npy` | 键为轨迹名、值为该轨迹各原子原子序数 `(N,)` 的字典 |
| `lattice.npy` | 键为轨迹名、值为 `(3, 3)` 晶格的字典 |
| `element_index.npy` | 长度 18 的元素查找表，`element_index[Z]` 把原子序数映射为训练时的元素索引 |

原版按温度索引轨迹（每个温度单独一个文件），转为本仓库 CSV 的规则：

- 同一温度下的所有轨迹写入同一个 `positions_{temp}K.npz`，`temp` 列记录该行温度。
- `t_start` / `t_end` 是该行在轨迹帧数组中的起止索引，即时间延迟帧对；`prior_Li` / `prior_frame` 选择 Maxwell-Boltzmann 先验帧。
- 拓扑信息（原子数、元素序数、晶格）不随温度变化，按轨迹名存入 `atomic_numbers.npy` / `lattice.npy`。
- `element_index.npy` 与 checkpoint 训练时一致：Li(Z=3)→2、O(Z=8)→7、P(Z=15)→14。随仓库提供的 `example_data/initial_structure.npz` 中 `atomic_numbers=[3,8,3,8]` 查表得元素索引 `[2,7,2,7]`，与训练 fixture 的表完全一致。

划分规则：按整条轨迹分配（一条轨迹只出现在一个 split），同一轨迹的帧不会跨 split，避免帧泄漏；`seed=42`，validation 对每行用 `default_rng(seed + row_idx)` 确定性抽取 `num_validation_times` 个流时间。

随仓库提供的小样本 `test/fixtures/liflow/dataset_mini/`：

| Dataset | Train | Val | Test |
| --- | --- | --- | --- |
| dataset_mini（800 K） | 3 | 2 | 1 |

共 6 条轨迹（`train_a/b/c`、`val_a/b`、`test_a`），每条 120 帧；`train_c` 含 5 个原子（其余 3–4 个原子）用于覆盖不同原子数。

完整训练数据（论文中的 800 K 等温度数据集）未随仓库发布。复现完整训练需先按上表规则整理上游数据，放入 `data/liflow/`；在此之前可用 `dataset_mini` 做加载与最小训练验证。

采样策略：propagator 保留时间延迟帧对采样与自适应 Maxwell-Boltzmann 先验；corrector 保留单帧加噪采样，先验为 `UniformScaleNormalPrior(0.25)` 与 `MaxwellBoltzmannPrior(0.1)`。

## Models

给定原子坐标与晶格，`DualPaiNN` 在截断半径 `r_max=5.0 Å` 内构建原子图，用 `num_radial_basis=20` 个径向基编码原子间距，经 `num_layers=3` 层消息传递更新节点特征，`num_elements=77` 个元素嵌入，参考温度 `ref_temp=1000.0 K`，`r_offset=0.5`。

流匹配目标（与原实现 `FlowModule` 一致）：

```
x_0       = x_start + prior
x_t       = (1 - t) * x_0 + t * x_end      # t ~ U(0, 1)
velocity  = x_end - x_0                    # prediction_mode: velocity
loss      = mean_nodes( sum_xyz( (v_pred - v_target)^2 ) )
```

`prediction_mode` 可选 `velocity`（预测速度场）或 `data`（预测 `x_end`）。模型参数直接挂在 `LiFlow`（`ppmat/models/liflow/liflow.py`）上而不包一层网络，使 state dict 与发布的 `best.pdparams` 权重键名一致。

数据加载使用 [`LiFlowCollator`](../../../ppmat/datasets/collate_fn.py)（三个配置的 `collate_fn` 均显式指定）：Paddle `DataLoader` 的 worker 协议只识别 sequence 与 mapping，原生 `Batch` 对象会被压成 0 个槽位导致 loader 产出 0 个 batch，因此该 collator 复用 `DefaultCollator` 的图拼接与 `edge_index` 偏移逻辑后返回纯 `dict`；全空边 batch 先用占位自环拼接、再在原位置清空，规避 `paddle.concat` 对空列表报错。

## Results

以下结果均来自本仓库测试与实测记录，无外部引用值。

| 项目 | 数据/权重 | 指标 | 结果 |
| --- | --- | --- | --- |
| 推理跑通（公开权重） | `liflow_universal.zip`，空权重缓存 | 25 步轨迹 `(26, 4, 3)`，数值全部有限 | 通过 |
| Paddle / PyTorch 数值对齐（模型输出） | `reference_model.npz`（PyTorch 2.5.1+cpu） | rtol=1e-5, atol=1e-6 | 通过 |
| Paddle / PyTorch 数值对齐（中间层） | `reference_layers.npz` | rtol=2e-4, atol=2e-5 | 通过 |
| 单元测试 | `test/fixtures/liflow/` | `pytest test/test_liflow_*.py` | 86 passed, 2 skipped |
| 轨迹评估（完整数据集 MSD / RDF） | 未随仓库发布 | — | 待补 |

推理输出文件：`trajectory.npy`（初始帧 + 25 步）、`trajectory.xyz`、`run_metadata.json`。

发布的通用权重包：

```bash
wget https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/molecular_dynamics_integrator/liflow/liflow_universal.zip
```

包内为 `liflow_universal.yaml` 与 `checkpoints/best.pdparams`、`checkpoints/best.json`。注意当前 BCE 上的权重包条目使用 Windows 反斜杠路径，下载解包逻辑已按安全规则展开（实测通过）；后续由维护者重新打包为斜杠条目后此兼容仍保留但不再触发。

### Training

```bash
# propagator
python molecular_dynamics_integrator/train.py -c molecular_dynamics_integrator/liflow/configs/liflow_universal_propagator.yaml
# corrector
python molecular_dynamics_integrator/train.py -c molecular_dynamics_integrator/liflow/configs/liflow_universal_corrector.yaml
```

### Validation

```bash
python molecular_dynamics_integrator/train.py -c molecular_dynamics_integrator/liflow/configs/liflow_universal_propagator.yaml Global.do_eval=True Global.do_train=False Global.do_test=False Trainer.pretrained_model_path='your checkpoint path(*.pdparams)'
```

### Testing

```bash
python molecular_dynamics_integrator/train.py -c molecular_dynamics_integrator/liflow/configs/liflow_universal_propagator.yaml Global.do_test=True Global.do_train=False Global.do_eval=False Trainer.pretrained_model_path='your checkpoint path(*.pdparams)'
```

### Prediction

```bash
# 一键推理：自动下载权重，输入为结构文件（.npz，含 positions/atomic_numbers/lattice/element_index）
python molecular_dynamics_integrator/predict.py -c molecular_dynamics_integrator/liflow/configs/liflow_universal_inference.yaml -i molecular_dynamics_integrator/liflow/example_data/initial_structure.npz -o output

# 可选参数：--temperature <K> --steps <n> --flow_steps <n> --solver <euler|heun> --seed <int>
```

MSD / RDF 评估使用 `molecular_dynamics_integrator/liflow/run_universal_evaluation.py` 对预测轨迹与参考轨迹计算均方位移和径向分布函数（需完整数据集，当前未提供数据，结果待补）。

## License

原实现为 MIT License（Copyright (c) 2024 Learning Matter @ MIT）。仓库未单独放置第三方 License 文件，按本仓库惯例，MIT 版权与许可声明以文件头形式随移植代码保留，涉及文件：`ppmat/models/liflow/`（`dual_painn.py`、`geometry.py`、`layers.py`、`liflow.py`、`prior.py`、`__init__.py`）、`ppmat/datasets/liflow_dataset.py`、`ppmat/datasets/liflow_noised_frame_dataset.py`、`ppmat/metrics/liflow_metric.py`、`molecular_dynamics_integrator/train.py`、`molecular_dynamics_integrator/predict.py`、`molecular_dynamics_integrator/evaluate.py`、`molecular_dynamics_integrator/liflow/run_universal_evaluation.py`。各文件头部格式为：

```
# This code is ported from https://github.com/learningmatter-mit/liflow
#
# Copyright (c) 2024 Learning Matter @ MIT
# Licensed under the MIT License:
#     https://github.com/learningmatter-mit/liflow/blob/main/LICENSE
```

## Citation

```bibtex
@article{nam2025flow,
  title={Flow Matching for Accelerated Simulation of Atomic Transport in Crystalline Materials},
  author={Juno Nam and Sulin Liu and Gavin Winter and KyuJung Jun and Soojung Yang and Rafael Gómez-Bombarelli},
  journal={Nature Machine Intelligence},
  year={2025},
  doi={10.1038/s42256-025-01125-4},
}
```
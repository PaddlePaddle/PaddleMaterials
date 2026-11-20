# PaddleMaterials 性能与精度对比

本目录提供 eager 与 CINN 的性能和精度对比脚本，覆盖注册权重中具备 CINN
路径的 79 个权重。InfGCN 仅支持 eager，不在测试范围内。

## 目录

```text
benchmark/
├── README.md
└── cinn/
    ├── benchmark_registered.py
    ├── compare_backends.py
    ├── run_registry_matrix.py
    └── fixtures/
```

- `benchmark_registered.py`：在独立进程中测试单个权重，记录首次调用和
  warm 耗时以及模型输出。
- `run_registry_matrix.py`：一条命令依次运行 eager 和 CINN，并自动生成
  对比结果。
- `compare_backends.py`：合并两种后端的结果，计算加速比和核心精度误差。
- `fixtures/`：SphereNet MD17 使用的小型固定输入。

## 环境准备

测试需要 NVIDIA GPU，以及同时启用 CUDA 和 CINN 的 PaddlePaddle。先安装项目：

```bash
python -m pip install -r requirements.txt
python setup.py build_ext --inplace
python -m pip install -e .
```

确认运行环境：

```bash
python -c "import paddle; print(paddle.__version__, paddle.is_compiled_with_cuda(), paddle.base.is_compiled_with_cinn())"
```

后两项应输出 `True True`。首次运行会自动下载注册权重，因此还需要网络连接和
足够的磁盘空间。

## 快速开始

默认覆盖全部 79 个 CINN 权重，分别执行 eager 和 CINN，每个后端 warm 运行 3
次，最后自动生成对比结果：

```bash
python benchmark/cinn/run_registry_matrix.py \
  --output_dir output/cinn_benchmark
```

先用一个权重做环境冒烟测试，验证环境、权重下载和 CINN 编译链路：

```bash
python benchmark/cinn/run_registry_matrix.py \
  --output_dir output/cinn_benchmark_smoke \
  --repeats 1 \
  --models chgnet_mptrj
```

如果需要分别查看某个模型或某次调用，可以直接调用
`benchmark_registered.py`：

```bash
python benchmark/cinn/benchmark_registered.py \
  --model_name chgnet_mptrj \
  --backend cinn \
  --repeats 3 \
  --output output/cinn_benchmark/cinn/chgnet_mptrj.json
```

## 常用参数

`run_registry_matrix.py` 只保留少量常用参数：

- `--models`：指定一个或多个注册权重，默认全部支持权重。
- `--gpu`：使用的 CUDA 设备编号，默认 `0`。
- `--repeats`：每个后端 warm 运行次数，默认 `3`。
- `--output_dir`：输出目录，默认 `output/cinn_benchmark`。
- `--backend`：`eager`、`cinn` 或 `both`，默认 `both`。

脚本按权重串行运行；不同 GPU 可以分别启动一条命令。例如：

```bash
python benchmark/cinn/run_registry_matrix.py \
  --gpu 0 --models chgnet_mptrj spherenet_qm9_mu \
  --output_dir output/cinn_benchmark_gpu0
python benchmark/cinn/run_registry_matrix.py \
  --gpu 1 --models comformer_mp2018_train_60k_e_form \
  --output_dir output/cinn_benchmark_gpu1
```

## 输出

```text
output/cinn_benchmark/
├── eager/<model>.json
├── cinn/<model>.json
├── logs/eager/<model>.log
├── logs/cinn/<model>.log
├── comparison.json
└── comparison.csv
```

每个权重的 JSON 记录：

- `first_seconds`：首次调用耗时；CINN 模式下包含编译；
- `warm_median_seconds`：后续 warm 调用耗时中位数；
- `runtime_keys`：本次调用生成的 CINN 运行时缓存键；
- `prediction`：首次调用的模型输出，用于 eager/CINN 精度对比；
- `loss`：模型自身损失函数得到的标量，以及可复用的指标值，例如 SFIN 的
  `psnr`、`ssim`；输入不带 ground-truth 标签时留空；
- `paddle_version`、`compiled_with_cinn` 等环境信息。

`comparison.json` 和 `comparison.csv` 包含每个权重的效率和精度结果。加速比
定义为 `eager warm / CINN warm`，大于 1 表示 CINN 更快。

## 精度对比

eager 和 CINN 都会在首次调用前重置随机种子，因此可以直接比较同一个输入下的
输出。每次比较会扫描预测中的数值字段，并给出：

- `precision_status`：`passed`、`failed`、`no_arrays` 或 `not_compared`；
- `precision_max_abs_error`：所有匹配字段中的最大绝对误差；
- `precision_max_relative_error`：所有匹配字段中的最大相对误差；
- `precision_relative_l2_error`：所有匹配字段中的最大相对 L2 误差；
- `precision_fields` / `precision_matched_fields`：参与比较的字段数和形状
  匹配的字段数。

除预测张量外，还会逐字段比较 `loss` 中的标量，并给出
`loss_metric_status`、`loss_metric`。每个字段记录 eager 与 CINN 的取值、
`abs_error` 和 `relative_error`；汇总里的 `loss_metric_compared` 与
`loss_metric_failed` 给出通过和失败的数量。loss 与指标在计时区间之外计算，
不影响效率数字。

相对误差按 `abs(eager - cinn) / max(abs(eager), 1e-12)` 计算，相对 L2 误差
按 `norm(eager - cinn) / max(norm(eager), 1e-12)` 计算，其中 eager 输出是
参考值。当前不设置统一的通过阈值；CINN 与 eager 使用相同权重和输入，但浮点
运算顺序可能不同，因此误差通常很小，不保证逐位一致，需要结合具体物理量的
可接受误差判断。

生成类模型（如 DiffCSP、MatterGen、DiffNMR）包含随机采样流程，比较时会重置
随机种子，但如果 CINN 的编译路径改变了随机操作顺序，输出仍可能产生明显差异；
这类结果应结合同一环境下的重复运行来解释。

## 参考实测结果

以下结果来自 2026-11-19 在单机 8×NVIDIA A100-SXM4-40GB 上的测试。环境为
PaddlePaddle 3.1.0，CUDA 和 CINN 均开启；覆盖 79 个具备 CINN 路径的注册
权重，eager/CINN 各执行 1 次首次调用和 5 次 warm 调用。结果仅用于本次机器和
环境的参考，不用于跨机器结论。

- eager 全部 79/79 通过；CINN 77/79 通过。
- 77 个双后端通过的权重全部完成精度比较，`precision_status` 均为
  `passed`，没有字段缺失、形状不匹配或未比较的项。
- 加速比中位数 0.997，最小 0.517，最大 3.164；其中 38 个权重 CINN 更快。
  加速比按 `eager warm / CINN warm` 计算，小于 1 表示 eager 更快。
- 整体最大绝对误差 93.0，最大相对误差 691.55，最大相对 L2 误差 3.34。
  最大误差主要来自 MatterGen 的随机结构采样；SFIN detect 的最大相对误差
  则是小值分母造成的，对应相对 L2 误差仅 5.49e-05。

| 模型族 | 权重数 | CINN 通过 | 加速比中位数 | 最大绝对误差 | 最大相对误差 |
| --- | ---: | ---: | ---: | ---: | ---: |
| CHGNet | 1 | 1 | 1.704 | 3.26e-03 | 43.64 |
| ComFormer | 8 | 8 | 0.849 | 1.17e-01 | 2.47e-04 |
| DiffCSP | 2 | 2 | 0.917 | 3.87e-04 | 3.12e-03 |
| DiffNMR | 1 | 1 | 1.087 | 0 | 0 |
| DimeNet++ | 16 | 16 | 1.484 | 1.81e-02 | 6.31e-04 |
| MatterGen | 17 | 17 | 0.677 | 93.0 | 691.55 |
| MatterSim | 2 | 0 | - | - | - |
| MEGNet | 8 | 8 | 0.552 | 6.06e-05 | 5.42e-05 |
| SFIN | 4 | 4 | 2.332 | 9.72e-02 | 497.2 |
| SphereNet | 20 | 20 | 1.475 | 6.25e-02 | 9.97e-02 |

MatterSim 的 2 个权重未进入精度比较。它们在 CINN 编译阶段报
`Required in_value_dim and out_value_dim have same size`
（`in_value_dim.size()==0`、`out_value_dim.size()==1`），属于当前
PaddlePaddle 3.1.0 CINN 编译期错误，而不是 benchmark 输入或模型参数问题。

实测还发现 MatterGen、MatterSim 源码中的 `reshape(1, -1)` 在 CINN SOT 下
不可被识别。本次已将相关调用改为 `reshape([1, -1])`；MatterGen 的 4 个此前
CINN 失败的权重在修改后通过，MatterSim 仍被上述编译期错误阻塞。

测试输入由 `benchmark_registered.py` 固定。结果用于比较同一权重的 eager 和
CINN，不用于不同模型之间的横向性能比较。输出目录位于 `output/`，不会提交到
Git。

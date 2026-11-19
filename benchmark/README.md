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

相对误差按 `abs(eager - cinn) / max(abs(eager), 1e-12)` 计算，相对 L2 误差
按 `norm(eager - cinn) / max(norm(eager), 1e-12)` 计算，其中 eager 输出是
参考值。当前不设置统一的通过阈值；CINN 与 eager 使用相同权重和输入，但浮点
运算顺序可能不同，因此误差通常很小，不保证逐位一致，需要结合具体物理量的
可接受误差判断。

生成类模型（如 DiffCSP、MatterGen、DiffNMR）包含随机采样流程，比较时会重置
随机种子，但如果 CINN 的编译路径改变了随机操作顺序，输出仍可能产生明显差异；
这类结果应结合同一环境下的重复运行来解释。

测试输入由 `benchmark_registered.py` 固定。结果用于比较同一权重的 eager 和
CINN，不用于不同模型之间的横向性能比较。输出目录位于 `output/`，不会提交到
Git。

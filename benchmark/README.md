# PaddleMaterials 性能测试

本目录提供可复现的性能测试脚本。当前包含 CINN 注册权重性能测试，覆盖具备
CINN 路径的 79 个权重。InfGCN 仅支持 eager，不在测试范围内。

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

- `benchmark_registered.py`：在独立进程中测试单个权重，记录首次调用和 warm
  耗时。
- `run_registry_matrix.py`：一条命令依次调度 eager 和 CINN 测试；也可只测
  单个后端或指定权重。
- `compare_backends.py`：合并两种后端的结果，计算加速比和模型系列汇总。
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
  --gpus 0 \
  --output_dir output/cinn_benchmark
```

首次执行会逐个下载尚未缓存的模型包。运行中断后可复用已成功的结果：

```bash
python benchmark/cinn/run_registry_matrix.py \
  --gpus 0 \
  --output_dir output/cinn_benchmark \
  --resume
```

建议先用一个权重做环境冒烟测试：

```bash
python benchmark/cinn/run_registry_matrix.py \
  --output_dir output/cinn_benchmark_smoke \
  --gpus 0 \
  --repeats 1 \
  --models chgnet_mptrj
```

`--list-models` 可查看当前由 `MODEL_REGISTRY` 解析出的完整权重清单；
`--dry-run` 可在不下载权重、不启动 GPU 任务的情况下确认本次选择。单独测试一个
后端时使用 `--backend eager` 或 `--backend cinn`。

## 全量测试

需要更稳定的正式数据时，建议每个后端 warm 执行 10 次：

```bash
python benchmark/cinn/run_registry_matrix.py \
  --output_dir output/cinn_benchmark \
  --gpus 0 \
  --repeats 10
```

单 GPU 会串行运行。多 GPU 可通过逗号分隔，例如 `--gpus 0,1,2,3`；每张 GPU
同一时间只运行一个权重。比较两次测试时应使用相同的 GPU、并发数和系统负载。

## 输出

```text
output/cinn_benchmark/
├── eager/<model>.json
├── cinn/<model>.json
├── logs/eager/<model>.log
├── logs/cinn/<model>.log
├── eager_launcher_summary.json
├── cinn_launcher_summary.json
├── comparison.json
└── comparison.csv
```

单权重结果包含：

- `first_seconds`：首次调用耗时；CINN 模式下包含编译；
- `warm_seconds`：后续各次调用耗时；
- `warm_median_seconds`：warm 耗时中位数；
- `runtime_keys`：本次调用生成的运行时缓存键。

`comparison.json` 包含逐权重结果和模型系列汇总，`comparison.csv` 便于直接用
表格工具分析。加速比定义为
`eager warm / CINN warm`，大于 1 表示 CINN 更快。

测试输入由 `benchmark_registered.py` 固定。结果用于比较同一权重的 eager 和
CINN，不用于不同模型之间的横向性能比较。输出目录位于 `output/`，不会提交到
Git。

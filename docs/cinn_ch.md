# CINN 执行后端

[English](cinn.md)

PaddleMaterials 默认使用 eager。支持 CINN 的模型可以通过
`Execution.backend=cinn` 启用编译执行。

## 配置

```yaml
Execution:
  backend: cinn
  __init_params__:
    full_graph: false
```

`Execution.__init_params__` 可以省略，`full_graph` 默认为 `false`。命令行也可以
使用相同配置，例如 `Execution.backend=cinn`。

- `full_graph: false` 使用 SOT 捕获，允许断图；
- `full_graph: true` 使用 AST 转换，要求整个边界成为一个静态程序，只适用于
  下表标记为 **AST** 的模型。

CINN 要求 CUDA 设备以及启用 CINN 的 Paddle。当前运行时不支持 AMP 和分布式
执行。

## 运行机制

- `RuntimeMixin` 管理后端、运行选项和缓存；
- `runtime_boundary` 标记需要编译的数值函数；
- `CinnBackend` 校验环境并调用 `paddle.jit.to_static`。

编译在第一次调用边界时发生。运行时按后端、训练/评估模式和边界名缓存；切换后端
或修改运行选项会清空缓存。

边界名表示稳定的计算角色：

- `forward`：标准数值前向；
- `denoise_step`：单次扩散步骤；
- `graph_encoder` 和 `spectrum_encoder`：独立编译的编码器。

## 模型支持

下表记录使用 Paddle 3.3 完成注册 checkpoint 验证的流程。测试环境和耗时见
[性能矩阵](cinn_performance_ch.md)。

| 模型 | 边界 | 捕获方式 | 已验证流程 |
| --- | --- | --- | --- |
| `MEGNetPlus` | `forward` | SOT | 预测和训练 |
| `iComformer` | `forward` | SOT | 预测 |
| `DimeNetPlusPlus` | `forward` | SOT | 预测和训练 |
| `SphereNet` | `forward` | 属性 SOT；势模型 AST | 属性和能量/力推理 |
| `SFIN` | `forward` | AST | 图像推理 |
| `M3GNet` | `forward` | AST | 能量/力推理 |
| `CHGNet` | `forward` | AST | 能量/力/应力/磁矩推理 |
| `DiffCSP` | `denoise_step` | SOT | 结构采样 |
| `MatterGen`、`MatterGenWithCondition` | `denoise_step` | SOT | 无条件和条件采样 |
| `MolecularGraphFormer` | `graph_encoder`、`denoise_step` | SOT | 无独立注册流程 |
| `NMRNetCLIP` | `graph_encoder`、`spectrum_encoder` | SOT | 无独立注册流程 |
| `DiffPrior` | `denoise_step` | SOT | 无独立注册流程 |
| `DiffNMR` | `spectrum_encoder`、`denoise_step` | SOT | 反向扩散 |
| `InfGCN` | 无 | 仅 eager | CINN 不可用 |

## 接入 CINN

保留现有 `forward -> _forward -> predict` 协议，初始化 `RuntimeMixin`，并用
`runtime_boundary` 标记数值函数：

```python
from ppmat.models.common.runtime import RuntimeMixin, runtime_boundary


class NewModel(RuntimeMixin, paddle.nn.Layer):
    def __init__(self, execution_backend="eager", runtime_options=None):
        super().__init__()
        self._init_runtime(execution_backend, runtime_options)
        self.network = paddle.nn.Linear(16, 32)

    def _forward(self, data):
        return self._runtime_forward(data["x"])

    @runtime_boundary("forward")
    def _runtime_forward(self, x):
        return self.network(x)
```

如果模型输入已经是 Tensor，可以直接装饰 `_forward`。不要增加模型专属 CINN
wrapper，也不要复制一份 Tensor 实现。

模型标记为已验证前，需要完成：

1. eager 数值回归；
2. 运行时缓存测试；
3. 公开训练、预测或采样流程测试；
4. 使用代表性 checkpoint 和 GPU 输入进行 eager/CINN 对齐。

首次编译可能耗时较长或导致进程退出，应在隔离进程中运行，并将编译时间与 warm
执行耗时分开测量。

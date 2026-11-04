# CINN execution backend

[中文版](cinn_ch.md)

PaddleMaterials uses eager execution by default. Set `Execution.backend=cinn`
to enable CINN for a supported model.

## Configuration

```yaml
Execution:
  backend: cinn
  __init_params__:
    full_graph: false
```

`Execution.__init_params__` is optional, and `full_graph` defaults to `false`.
The same values can be supplied as command-line overrides, for example
`Execution.backend=cinn`.

- `full_graph: false` uses SOT capture and allows graph breaks.
- `full_graph: true` uses AST conversion and requires the whole boundary to
  become one static program. Use it only for models marked **AST** below.

CINN requires a CUDA device and a Paddle build with CINN enabled. AMP and
distributed execution are not supported by this runtime.

## Runtime

- `RuntimeMixin` manages backend selection, options, and runtime caches.
- `runtime_boundary` marks a numerical function for compilation.
- `CinnBackend` validates the environment and calls `paddle.jit.to_static`.

Compilation happens on the first boundary call. Runtimes are cached by backend,
train/eval mode, and boundary name. Changing the backend or its options clears
the cache.

Boundary names describe stable roles:

- `forward`: standard numerical forward;
- `denoise_step`: one diffusion step;
- `graph_encoder` and `spectrum_encoder`: independently compiled encoders.

## Model support

The table records the registered-checkpoint workflows qualified with Paddle
3.3. See the [performance matrix](cinn_performance.md) for the test environment
and timings.

| Model | Boundary | Capture | Qualified workflow |
| --- | --- | --- | --- |
| `MEGNetPlus` | `forward` | SOT | prediction and training |
| `iComformer` | `forward` | SOT | prediction |
| `DimeNetPlusPlus` | `forward` | SOT | prediction and training |
| `SphereNet` | `forward` | SOT for properties; AST for potentials | property and energy/force inference |
| `SFIN` | `forward` | AST | image inference |
| `M3GNet` | `forward` | AST | energy/force inference |
| `CHGNet` | `forward` | AST | energy/force/stress/magnetic-moment inference |
| `DiffCSP` | `denoise_step` | SOT | structure sampling |
| `MatterGen`, `MatterGenWithCondition` | `denoise_step` | SOT | unconditional and conditional sampling |
| `MolecularGraphFormer` | `graph_encoder`, `denoise_step` | SOT | no standalone registered workflow |
| `NMRNetCLIP` | `graph_encoder`, `spectrum_encoder` | SOT | no standalone registered workflow |
| `DiffPrior` | `denoise_step` | SOT | no standalone registered workflow |
| `DiffNMR` | `spectrum_encoder`, `denoise_step` | SOT | reverse diffusion |
| `InfGCN` | none | eager only | CINN unavailable |

## Adding CINN support

Keep the existing `forward -> _forward -> predict` protocol. Initialize
`RuntimeMixin` and mark the numerical function with `runtime_boundary`:

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

For a tensor-ready model, the decorator can be placed directly on `_forward`.
Do not add a model-specific CINN wrapper or a duplicate tensor implementation.

Before listing a model as qualified, verify:

1. eager numerical regression;
2. runtime cache behavior;
3. public training, prediction, or sampling workflows;
4. eager/CINN parity with a representative checkpoint and GPU input.

Run first-time compilation in an isolated process because compilation can be
slow or terminate the process. Measure warm execution separately from compile
time.

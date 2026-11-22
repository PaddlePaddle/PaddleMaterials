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
    dynamic_shape: true
```

`Execution.__init_params__` is optional; `full_graph` defaults to `false` and
`dynamic_shape` to `true`. The same values can be supplied as command-line
overrides, for example `Execution.backend=cinn`.

- `full_graph: false` uses SOT capture and allows graph breaks.
- `full_graph: true` uses AST conversion and requires the whole boundary to
  become one static program. Use it only for models marked **AST** below.
  Force and stress models need it, because SOT capture can split `paddle.grad`
  from its input; see the model README for the exact setting.
- `dynamic_shape: true` compiles the axes that a boundary declares as dynamic
  (see [Dynamic axes](#dynamic-axes)), so that one program serves every edge or
  triplet count. It applies to AST capture in eval mode.
  `dynamic_shape: false` compiles one program per input shape.

CINN requires a CUDA device and a Paddle build with CINN enabled. AMP and
distributed execution are not supported by this runtime.

## Runtime

- `RuntimeMixin` manages backend selection, options, and runtime caches.
- `runtime_boundary` marks a numerical function for compilation.
- `CinnBackend` validates the environment and calls `paddle.jit.to_static`.

Compilation happens on the first boundary call. Runtimes are cached by backend,
train/eval mode, and boundary name. Changing the backend or its options clears
the cache.

### Dynamic axes

AST capture traces one static program per input shape, and CINN compiles each
of them. A force model sees a new edge and triplet count at almost every MD or
relaxation step, so without dynamic axes it recompiles at almost every step.
A boundary declares the axes whose sizes change between calls:

```python
@runtime_boundary("forward", dynamic_dims={"edge_index": 1, "idx_kj": 0})
def _runtime_forward(self, pos, edge_index, idx_kj):
    ...
```

With `full_graph: true` and `dynamic_shape: true`, an eval-mode call passes
`paddle.jit.to_static` an `InputSpec` in which these axes are `None`. Calls that
differ only in declared axes share one program. A change in any other axis, such
as the atom count, or in a non-tensor argument compiles a new program. `M3GNet`,
`CHGNet` and the `SphereNet` potentials declare their edge, triplet and angle
axes.

Training keeps static shapes. A training batch also changes the atom count,
which stays static, so dynamic edge axes would not remove its recompilation.
The second-order force gradient of `SphereNet` training also relies on an
empty-triplet branch that a static shape resolves while tracing.

Inside a boundary with dynamic axes, a size such as `x.shape[0]` is a tensor
while tracing. Do not branch on it (`if x.shape[0] == 0:`): AST conversion turns
the branch into a `pd_op.if`, whose backward gave wrong `M3GNet` forces and
which CINN failed to compile for `SphereNet`. Keep such shortcuts for sizes that
are Python ints, and handle empty inputs without a branch, for example with a
mask.

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

# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Dynamic-shape compilation of the potential boundaries.

Every MD step moves atoms, so the edge and triplet counts that enter a force
model's boundary change while the atom count does not. Without dynamic axes the
AST capture traces, and CINN compiles, a new program for every new count. These
tests pin the contract of ``runtime_boundary(..., dynamic_dims=...)``: changing
a declared axis reuses one program and keeps the eager numbers, changing any
other axis or a non-tensor argument builds a new program, empty triplet sets
match eager, and ``dynamic_shape=False`` restores one program per shape.

The capture runs on CPU (``to_static`` with ``backend=None``), which exercises
the same InputSpec and traced graph that CINN compiles; the last test repeats
the core check with CINN on a GPU.
"""

from __future__ import annotations

import numpy as np
import paddle
import pytest
from cinn_workflow_harness import gpu_cinn_or_skip
from cinn_workflow_harness import requires_gpu_cinn
from pymatgen.core import Lattice
from pymatgen.core import Structure

import ppmat.models.common.cinn as cinn_module
from ppmat.models.chgnet.chgnet import CHGNet
from ppmat.models.chgnet.chgnet_graph_converter import CHGNetGraphConverter
from ppmat.models.common.graph_converter import RadiusGraphConverter
from ppmat.models.common.runtime import DynamicShapeRuntime
from ppmat.models.common.runtime import RuntimeMixin
from ppmat.models.common.runtime import boundary_dynamic_dims
from ppmat.models.common.runtime import runtime_boundary
from ppmat.models.mattersim.m3gnet import M3GNet
from ppmat.models.mattersim.m3gnet_graph_converter import M3GNetGraphConvertor
from ppmat.models.spherenet.spherenet import SphereNet
from ppmat.utils.scatter import scatter
from ppmat.utils.scatter import scatter_sum_first_order


@pytest.fixture
def programs(monkeypatch):
    """Capture boundaries with AST on CPU and record every static program built."""

    from paddle.jit.dy2static import pir_partial_program

    built = []
    init = pir_partial_program.PartialProgramLayer.__init__

    def counting_init(self, *args, **kwargs):
        built.append(self)
        init(self, *args, **kwargs)

    def compile_on_cpu(function, *, full_graph=False, input_spec=None):
        return paddle.jit.to_static(
            function, input_spec=input_spec, backend=None, full_graph=full_graph
        )

    monkeypatch.setattr(
        pir_partial_program.PartialProgramLayer, "__init__", counting_init
    )
    monkeypatch.setattr(cinn_module, "compile_cinn", compile_on_cpu)
    monkeypatch.setattr(cinn_module.CINN_BACKEND, "validate", lambda *a, **k: None)
    return built


# ---------------------------------------------------------------- runtime contract
class _EdgeModel(RuntimeMixin, paddle.nn.Layer):
    """Minimal boundary with an edge axis, a static atom axis and a constant."""

    def __init__(self, **options):
        super().__init__()
        self.linear = paddle.nn.Linear(3, 2)
        self._init_runtime("cinn", {"cinn": {"full_graph": True, **options}})

    @runtime_boundary("forward", dynamic_dims={"edge_vec": 0})
    def _runtime_forward(self, pos, edge_vec, scale):
        return self.linear(edge_vec).sum(axis=0) * scale + pos.sum()


def _expected(model, pos, edge_vec, scale):
    return (model.linear(edge_vec).sum(axis=0) * scale + pos.sum()).numpy()


def _tensor(shape, seed):
    return paddle.to_tensor(np.random.RandomState(seed).rand(*shape).astype("float32"))


def test_dynamic_axis_reuses_one_program(programs):
    model = _EdgeModel()
    model.eval()
    pos = _tensor([4, 3], 0)
    for seed, num_edges in enumerate((3, 7, 5)):
        edge_vec = _tensor([num_edges, 3], seed + 1)
        actual = model._runtime_forward(pos, edge_vec, 2.0)
        np.testing.assert_allclose(
            actual.numpy(), _expected(model, pos, edge_vec, 2.0), rtol=1e-6
        )
    assert len(programs) == 1
    runtime = model._runtime_cache[("cinn", "eval", "forward")]
    assert isinstance(runtime, DynamicShapeRuntime)
    assert list(model._runtime_cache) == [("cinn", "eval", "forward")]


def test_static_axes_and_constants_stay_in_the_cache_key(programs):
    model = _EdgeModel()
    model.eval()
    edge_vec = _tensor([3, 3], 1)
    cases = [
        (_tensor([4, 3], 0), 2.0),
        (_tensor([5, 3], 2), 2.0),
        (_tensor([5, 3], 2), 3.0),
    ]
    for pos, scale in cases:
        actual = model._runtime_forward(pos, edge_vec, scale)
        np.testing.assert_allclose(
            actual.numpy(), _expected(model, pos, edge_vec, scale), rtol=1e-6
        )
    # A new atom count and a new constant each need their own program.
    assert len(programs) == 3


def test_dynamic_shape_false_restores_one_program_per_shape(programs):
    model = _EdgeModel(dynamic_shape=False)
    model.eval()
    pos = _tensor([4, 3], 0)
    for seed, num_edges in enumerate((3, 7)):
        model._runtime_forward(pos, _tensor([num_edges, 3], seed + 1), 2.0)
    assert len(programs) == 2
    runtime = model._runtime_cache[("cinn", "eval", "forward")]
    assert not isinstance(runtime, DynamicShapeRuntime)


@pytest.mark.parametrize("mode", ["train", "sot"])
def test_training_and_sot_boundaries_get_no_input_spec(monkeypatch, mode):
    calls = []

    def fake_compile(function, **kwargs):
        calls.append(kwargs)
        return function

    monkeypatch.setattr(cinn_module, "compile_cinn", fake_compile)
    monkeypatch.setattr(cinn_module.CINN_BACKEND, "validate", lambda *a, **k: None)
    model = _EdgeModel(full_graph=mode != "sot")
    if mode == "train":
        model.train()
    else:
        model.eval()
    model._runtime_forward(_tensor([4, 3], 0), _tensor([3, 3], 1), 1.0)
    assert calls == [{"full_graph": mode != "sot"}]


def test_dynamic_dims_are_validated():
    with pytest.raises(ValueError, match="unknown arguments"):

        @runtime_boundary("forward", dynamic_dims={"missing": 0})
        def _missing(self, x):
            return x

    with pytest.raises(ValueError, match="non-negative axes"):
        runtime_boundary("forward", dynamic_dims={"x": -1})

    assert boundary_dynamic_dims(_EdgeModel._runtime_forward) == {"edge_vec": (0,)}


def test_cinn_dynamic_shape_option_is_normalized():
    backend = cinn_module.CINN_BACKEND
    assert backend.normalize_options(None) == {
        "full_graph": False,
        "dynamic_shape": True,
    }
    assert backend.normalize_options({"full_graph": True, "dynamic_shape": False}) == {
        "full_graph": True,
        "dynamic_shape": False,
    }
    with pytest.raises(TypeError, match="dynamic_shape"):
        backend.normalize_options({"dynamic_shape": "false"})
    assert backend.accepts_input_spec({"full_graph": True, "dynamic_shape": True})
    assert not backend.accepts_input_spec({"full_graph": False, "dynamic_shape": True})
    assert not backend.accepts_input_spec({"full_graph": True, "dynamic_shape": False})


# ---------------------------------------------------------------- the three potentials
def _crystal(lattice, coords):
    return Structure(lattice, ["Li", "O"], coords, coords_are_cartesian=True)


# Two atoms each, off their symmetric sites so that the forces do not vanish:
# two cells with different edge and triplet (CHGNet angle) counts, and a dimer
# in a large box, which has edges but no triplets and no angles.
CRYSTALS = {
    "a": _crystal(Lattice.cubic(3.2), [[0.0, 0.0, 0.0], [1.5, 1.6, 1.7]]),
    "b": _crystal(Lattice.cubic(4.0), [[0.0, 0.0, 0.0], [1.6, 1.8, 2.3]]),
    "no_triplet": _crystal(Lattice.cubic(20.0), [[0.0, 0.0, 0.0], [1.9, 0.0, 0.0]]),
}
# Four atoms each: a compact molecule, the same atoms with one moved away (fewer
# edges and triplets), and two separated pairs, which have edges but no triplets.
MOLECULES = {
    "a": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.2, 1.1, 0.1], [0.1, 0.3, 1.2]],
    "b": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.2, 1.1, 0.1], [5.5, 0.3, 1.2]],
    "no_triplet": [
        [0.0, 0.0, 0.0],
        [1.1, 0.0, 0.0],
        [10.0, 0.0, 0.0],
        [11.1, 0.0, 0.0],
    ],
}


def _m3gnet(**options):
    with paddle.utils.unique_name.guard():
        paddle.seed(2026)
        return M3GNet(
            num_blocks=1,
            units=8,
            max_l=2,
            max_n=2,
            cutoff=5.0,
            max_z=94,
            threebody_cutoff=4.0,
            energy_key="energy",
            force_key="force",
            stress_key=None,
            execution_backend="cinn",
            runtime_options={"cinn": {"full_graph": True, **options}},
        )


def _chgnet(**options):
    with paddle.utils.unique_name.guard():
        paddle.seed(2026)
        return CHGNet(
            atom_fea_dim=8,
            bond_fea_dim=8,
            angle_fea_dim=8,
            composition_model=None,
            num_radial=5,
            num_angular=5,
            n_conv=2,
            atom_conv_hidden_dim=8,
            bond_conv_hidden_dim=8,
            angle_layer_hidden_dim=0,
            mlp_hidden_dims=(8,),
            property_names=("energy_per_atom", "force"),
            execution_backend="cinn",
            runtime_options={"cinn": {"full_graph": True, **options}},
        )


def _spherenet(**options):
    with paddle.utils.unique_name.guard():
        paddle.seed(2026)
        return SphereNet(
            derivative_properties=("force",),
            num_layers=1,
            hidden_channels=8,
            out_channels=1,
            int_emb_size=4,
            basis_emb_size_dist=4,
            basis_emb_size_angle=4,
            basis_emb_size_torsion=4,
            out_emb_channels=8,
            num_spherical=2,
            num_radial=3,
            num_before_skip=1,
            num_after_skip=1,
            num_output_layers=1,
            execution_backend="cinn",
            runtime_options={"cinn": {"full_graph": True, **options}},
        )


def _m3gnet_graph(name):
    return M3GNetGraphConvertor()(CRYSTALS[name])


def _chgnet_graph(name):
    converter = CHGNetGraphConverter(
        atom_graph_cutoff=6.0, bond_graph_cutoff=3.0, num_cpus=1
    )
    return converter(CRYSTALS[name])


def _spherenet_graph(name):
    converter = RadiusGraphConverter(
        cutoff=5.0, return_triplet_indices=True, num_cpus=1
    )
    return converter.from_arrays(
        np.asarray([6, 1, 7, 8], dtype=np.int64),
        np.asarray(MOLECULES[name], dtype=np.float32),
    )


POTENTIALS = {
    "m3gnet": (_m3gnet, _m3gnet_graph, "three_body_indices", "energy"),
    "chgnet": (_chgnet, _chgnet_graph, "bond_graph", "energy_per_atom"),
    "spherenet": (_spherenet, _spherenet_graph, "ti_idx_kj", "energy"),
}


def _sizes(graph, triplet_key):
    num_edges = int(np.asarray(graph.num_edges).reshape(-1)[0])
    return num_edges, int(np.asarray(graph.edge_feat[triplet_key]).shape[0])


def _predict(model, graph):
    prediction = model.predict(graph)
    return {key: np.asarray(value) for key, value in prediction.items()}


def _pair(factory, **options):
    """An eager reference and a compiled model with the same parameters."""

    eager = factory()
    eager.set_execution_backend("eager")
    eager.eval()
    compiled = factory(**options)
    compiled.set_state_dict(eager.state_dict())
    compiled.eval()
    return eager, compiled


def _assert_matches_eager(eager, compiled, graph, energy_key):
    expected = _predict(eager, graph)
    actual = _predict(compiled, graph)
    np.testing.assert_allclose(
        actual[energy_key], expected[energy_key], rtol=1e-5, atol=1e-6
    )
    # Relative to the largest force, because the small test models give forces
    # between 1e-5 and 1; float32 noise stays below 1e-3 of that.
    scale = float(np.abs(expected["force"]).max())
    error = float(np.abs(actual["force"] - expected["force"]).max())
    assert error <= 1e-3 * scale + 1e-8, (error, scale)


def _control_flow_ops(model):
    """Control-flow ops in the one dynamic-shape program of a compiled model."""

    runtime = model._runtime_cache[("cinn", "eval", "forward")]
    assert isinstance(runtime, DynamicShapeRuntime)
    (program,) = runtime.programs.values()
    block = program.concrete_program.main_program.global_block()
    return [op.name() for op in block.ops if op.name() in ("pd_op.if", "pd_op.while")]


@pytest.mark.parametrize("name", sorted(POTENTIALS))
def test_two_edge_counts_share_one_program_and_match_eager(programs, name):
    factory, make_graph, triplet_key, energy_key = POTENTIALS[name]
    eager, compiled = _pair(factory)
    graphs = [make_graph("a"), make_graph("b")]
    assert _sizes(graphs[0], triplet_key) != _sizes(graphs[1], triplet_key)
    for graph in graphs:
        _assert_matches_eager(eager, compiled, graph, energy_key)
    assert len(programs) == 1
    # Shape checks inside the boundary must not trace a cond: with a dynamic
    # axis its backward gave wrong M3GNet forces and CINN rejected SphereNet's.
    assert _control_flow_ops(compiled) == []


@pytest.mark.parametrize("name", sorted(POTENTIALS))
def test_no_triplets_match_eager_in_the_dynamic_program(programs, name):
    factory, make_graph, triplet_key, energy_key = POTENTIALS[name]
    eager, compiled = _pair(factory)
    empty = make_graph("no_triplet")
    num_edges, num_triplets = _sizes(empty, triplet_key)
    assert num_edges > 0 and num_triplets == 0
    # The first call traces with dynamic axes; the empty set then runs through
    # the same program, where it cannot take the eager early return.
    _assert_matches_eager(eager, compiled, make_graph("a"), energy_key)
    _assert_matches_eager(eager, compiled, empty, energy_key)
    assert len(programs) == 1


def test_dynamic_shape_false_traces_one_program_per_edge_count(programs):
    eager, compiled = _pair(_m3gnet, dynamic_shape=False)
    for name in ("a", "b"):
        _assert_matches_eager(eager, compiled, _m3gnet_graph(name), "energy")
    assert len(programs) == 2


@pytest.mark.parametrize("reduce_fn", ["scatter", "scatter_sum_first_order"])
def test_scatter_accepts_empty_rows_with_a_dynamic_axis(reduce_fn):
    def reduce(src, index):
        if reduce_fn == "scatter":
            return scatter(src, index, dim=0, dim_size=3)
        return scatter_sum_first_order(src, index, dim_size=3)

    compiled = paddle.jit.to_static(
        reduce,
        input_spec=[
            paddle.static.InputSpec([None, 2], "float32"),
            paddle.static.InputSpec([None], "int64"),
        ],
        backend=None,
        full_graph=True,
    )
    for rows in (4, 0):
        src = paddle.ones([rows, 2], dtype="float32")
        index = paddle.to_tensor(np.arange(rows) % 3, dtype="int64")
        np.testing.assert_allclose(
            compiled(src, index).numpy(), reduce(src, index).numpy()
        )


@requires_gpu_cinn
def test_cinn_compiles_one_program_for_md_like_inputs(monkeypatch):
    """With CINN on a GPU: edge counts and an empty triplet set share one program."""

    from paddle.jit.dy2static import pir_partial_program

    gpu_cinn_or_skip()
    built = []
    init = pir_partial_program.PartialProgramLayer.__init__

    def counting_init(self, *args, **kwargs):
        built.append(self)
        init(self, *args, **kwargs)

    monkeypatch.setattr(
        pir_partial_program.PartialProgramLayer, "__init__", counting_init
    )
    eager, compiled = _pair(_m3gnet)
    for name in ("a", "b", "no_triplet"):
        _assert_matches_eager(eager, compiled, _m3gnet_graph(name), "energy")
    assert len(built) == 1

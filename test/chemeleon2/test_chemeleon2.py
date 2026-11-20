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

import copy
import os

import numpy as np
import paddle
import pytest
from omegaconf import OmegaConf

from ppmat.models import build_model

_CONFIG_DIR = os.path.join(
    os.path.dirname(__file__),
    "../../structure_generation/configs/chemeleon2",
)


def _load_cfg(name):
    path = os.path.join(_CONFIG_DIR, f"chemeleon2_mp20_{name}.yaml")
    return OmegaConf.to_container(OmegaConf.load(path).Model, resolve=True)


def _build_vae():
    return build_model(_load_cfg("vae"))


def _build_ldm():
    return build_model(_load_cfg("ldm"))


def _structure_array(num_atoms_list, seed=42):
    # Plain-dict dataset batch as produced by MP20Dataset + the collator.
    paddle.seed(seed)
    total = sum(num_atoms_list)
    return {
        "atom_types": paddle.randint(1, 90, [total]),
        "num_atoms": paddle.to_tensor(num_atoms_list),
        "frac_coords": paddle.rand([total, 3]),
        "lengths": paddle.to_tensor([[5.0, 5.0, 5.0] for _ in num_atoms_list]),
        "angles": paddle.to_tensor([[90.0, 90.0, 90.0] for _ in num_atoms_list]),
    }


def _batch(num_atoms_list, seed=42):
    from ppmat.models.chemeleon2.common.schema import CrystalBatch

    structure_array = _structure_array(num_atoms_list, seed=seed)
    total = sum(num_atoms_list)
    b = CrystalBatch()
    b.atom_types = structure_array["atom_types"]
    b.frac_coords = structure_array["frac_coords"]
    b.cart_coords = paddle.rand([total, 3])
    b.lattices = paddle.stack([paddle.eye(3) * 5 for _ in num_atoms_list])
    b.lengths = structure_array["lengths"]
    b.angles = structure_array["angles"]
    b.lengths_scaled = b.lengths / paddle.to_tensor(
        num_atoms_list, dtype="float32"
    ).unsqueeze(-1) ** (1 / 3)
    b.angles_radians = paddle.deg2rad(b.angles)
    b.num_atoms = paddle.to_tensor(num_atoms_list)
    b.batch = paddle.repeat_interleave(
        paddle.arange(len(num_atoms_list)), paddle.to_tensor(num_atoms_list)
    )
    b.token_idx = paddle.concat([paddle.arange(n) for n in num_atoms_list])
    b.num_graphs = len(num_atoms_list)
    b.mask = paddle.ones([len(num_atoms_list), max(num_atoms_list)], dtype="bool")
    return b


def test_vae_pipeline():
    vae = _build_vae()
    b = _batch([8, 12])

    paddle.seed(42)
    loss = vae.calculate_loss(b, training=False)
    assert loss["total_loss"].item() > 0
    assert not paddle.isnan(loss["total_loss"]).item()

    encoded = vae.encode(b)
    assert "x" in encoded and "moments" in encoded and "posterior" in encoded

    encoded["x"] = encoded["posterior"].sample()
    decoded = vae.decode(encoded)
    assert decoded["atom_types"].shape[0] == 20

    rec = vae.reconstruct(decoded, b)
    assert rec.lattices is not None and rec.lengths is not None


def test_forward_loss_contract():
    # BaseTrainer contract: forward() consumes the plain-dict dataset batch
    # ({"structure_array": ...}) and returns {"loss_dict": {"loss": ...}}
    # with a finite scalar suitable for gradient backprop.
    structure_array = _structure_array([8, 12])
    paddle.seed(42)
    with paddle.no_grad():
        outputs = {
            "vae": _build_vae()({"structure_array": structure_array}),
            "ldm": _build_ldm()({"structure_array": structure_array}),
        }
    for out in outputs.values():
        assert "loss_dict" in out
        assert "loss" in out["loss_dict"]
        assert paddle.is_tensor(out["loss_dict"]["loss"])
        assert not paddle.isnan(out["loss_dict"]["loss"]).item()


def test_ldm_pipeline():
    ldm = _build_ldm()
    b = _batch([8, 12])

    paddle.seed(42)
    loss = ldm.calculate_loss(b, training=False)
    assert loss["total_loss"].item() > 0

    with paddle.no_grad():
        for sampler in ("ddpm", "ddim"):
            r = ldm.sample(b, sampler=sampler, num_inference_steps=5, progress=False)
            assert "result" in r and len(r["result"]) == 2

    with paddle.no_grad():
        r = ldm.predict(
            {"num_samples": 2, "batch_size": 2, "num_atoms": 8},
            num_inference_steps=5,
            sampler="ddim",
        )
    assert "result" in r

    # An unconditional build must reject condition payloads loudly instead
    # of silently ignoring them.
    with pytest.raises(ValueError):
        ldm.predict(
            {"num_samples": 1, "num_atoms": 8, "condition": "MgO"},
            num_inference_steps=1,
            sampler="ddim",
            progress=False,
        )

    # Generation-to-evaluation contract: the sample() output schema must be
    # directly consumable by the generation metrics layer. Deterministic
    # structures are used because the tiny random model can emit degenerate
    # lattices that stall the structure matcher.
    from ppmat.metrics import StructGenMetric
    from ppmat.metrics.utils import Crystal

    def make_pred(atom_types, frac_coords, lattice):
        return {
            "atom_types": np.array(atom_types),
            "frac_coords": np.array(frac_coords),
            "lattice": np.array(lattice),
        }

    schema_pred = {
        k: np.asarray(v)
        for k, v in r["result"][0].items()
        if k in ("atom_types", "frac_coords", "lattice")
    }
    assert sorted(schema_pred) == ["atom_types", "frac_coords", "lattice"]

    a = [11.0, 11.0, 8.0]
    coords = [[0.1, 0.1, 0.1], [0.6, 0.1, 0.1], [0.6, 0.6, 0.1]]
    latt = [[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]]
    valid_pred = make_pred(a, coords, latt)
    # A deterministically invalid variant: a non-positive lattice is always
    # rejected by the metric layer.
    bad_pred = make_pred(a, coords, [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    metric = StructGenMetric()
    res = metric([valid_pred, valid_pred, bad_pred])
    assert 0.0 <= res["validity"] <= 1.0 and res["validity"] > 0.0
    assert 0.0 <= res["uniqueness"] <= 1.0 and res["uniqueness"] < 1.0

    ref_metric = StructGenMetric()
    ref_metric.reference_structures = [Crystal(valid_pred).structure]
    res2 = ref_metric([valid_pred])
    assert res2["novelty"] == 0.0


def test_ldm_spaced_scheduler_alphas():
    # Regression for the spaced-sampling math: the SpacedDiffusion scheduler
    # must be rebuilt from the spaced betas so that alphas_cumprod decays
    # monotonically over the spaced chain (a scheduler still carrying the
    # original 1000-step alphas_cumprod would stay near 1.0 and make the
    # DDIM/DDPM update equations meaningless).
    from ppmat.models.chemeleon2.ldm_module.diffusion import create_diffusion

    full = create_diffusion(
        timestep_respacing="", noise_schedule="linear", diffusion_steps=1000
    )
    spaced = create_diffusion(
        timestep_respacing="ddim50", noise_schedule="linear", diffusion_steps=1000
    )
    assert spaced.num_timesteps == 50
    assert spaced.timestep_map == list(range(0, 1000, 20))

    orig = full.scheduler.alphas_cumprod.numpy()
    new = spaced.scheduler.alphas_cumprod.numpy()
    assert new.shape[0] == 50
    assert new[-1] < 1e-3 and new[0] > 0.5
    assert np.allclose(new, orig[spaced.timestep_map], atol=1e-6)

    # DDPM respacing shares the same invariant (no "ddim" prefix)
    ddpm_spaced = create_diffusion(
        timestep_respacing="5", noise_schedule="linear", diffusion_steps=1000
    )
    ddpm_new = ddpm_spaced.scheduler.alphas_cumprod.numpy()
    assert ddpm_new.shape[0] == 5 and ddpm_new[-1] < 1e-3
    assert np.allclose(ddpm_new, orig[ddpm_spaced.timestep_map], atol=1e-6)

    # t=0 must fall back to the identity update (alpha_bar_prev=1.0),
    # matching the upstream alphas_cumprod_prev convention.
    diffusion = create_diffusion(
        timestep_respacing="ddim50", noise_schedule="linear", diffusion_steps=1000
    )
    x = paddle.randn([2, 4, 8])
    t0 = paddle.to_tensor([0, 0], dtype="int64")

    def zero_model(x, t, **kwargs):
        # learn_sigma=True: the denoiser emits (B, 2N, L) before split_epsilon
        return paddle.zeros([x.shape[0], 2 * x.shape[1], x.shape[2]])

    with paddle.no_grad():
        out = diffusion.ddim_sample(
            zero_model, x, t0, clip_denoised=False, model_kwargs=None, eta=0.0
        )
    assert paddle.allclose(out["sample"], out["pred_xstart"])


def test_ldm_conditional_pipeline():
    # Small end-to-end CFG pipeline build: forward, loss and sampling with a
    # condition module wired into the DiT through the condition_dim channel.
    cfg = copy.deepcopy(_load_cfg("ldm"))
    params = cfg["__init_params__"]
    params["denoiser"]["__init_params__"].update(
        hidden_dim=64, num_layers=2, num_heads=4, condition_dim=64
    )
    params["vae"]["__init_params__"]["encoder"]["__init_params__"].update(
        d_model=64, dim_feedforward=128, num_layers=2
    )
    params["vae"]["__init_params__"]["decoder"]["__init_params__"].update(
        d_model=64, dim_feedforward=128, num_layers=2
    )
    params["condition_module"] = {
        "__class_name__": "Chemeleon2ConditionModule",
        "__init_params__": {
            "condition_type": {"condition": "composition"},
            "hidden_dim": 64,
            "drop_prob": 0.1,
        },
    }

    ldm = build_model(cfg)
    assert ldm.use_cfg is True

    b = _batch([8, 12])
    b.y = {"condition": ["MgO", "NaCl"]}
    paddle.seed(42)
    loss = ldm.calculate_loss(b, training=False)
    assert not paddle.isnan(loss["total_loss"]).item()

    with paddle.no_grad():
        r = ldm.sample(b, sampler="ddim", num_inference_steps=3, progress=False)
        assert "result" in r and len(r["result"]) == 2

    # ``StructureSampler.sample_by_condition`` contract: conditions arrive as
    # top-level dict keys, routed through the model's ``condition_names``.
    assert ldm.condition_names == ["condition"]
    data_by_condition = {
        "structure_array": {"num_atoms": paddle.to_tensor([8], dtype="int64")},
        "condition": "MgO",
    }
    with paddle.no_grad():
        r = ldm.sample(
            data_by_condition, sampler="ddim", num_inference_steps=3, progress=False
        )
        assert "result" in r and len(r["result"]) == 1

    # predict() follows the same top-level-key convention; a scalar condition
    # must be routed through condition_module instead of crashing on it.
    with paddle.no_grad():
        r = ldm.predict(
            {"num_samples": 1, "num_atoms": 8, "condition": "MgO"},
            num_inference_steps=3,
            sampler="ddim",
            progress=False,
        )
    assert "result" in r and len(r["result"]) == 1


def test_ldm_cinn_dispatch_and_parity(monkeypatch):
    # Single runtime-contract check: hook protocol, weight invariance under
    # backend switches, boundary funneling and eager/cinn numeric parity.
    from ppmat.models.common.runtime import RuntimeMixin

    ldm = _build_ldm()
    assert isinstance(ldm, RuntimeMixin)
    for hook in (
        "set_execution_backend",
        "set_runtime_options",
        "validate_execution_backend",
    ):
        assert callable(getattr(ldm, hook, None))

    keys_eager = sorted(ldm.state_dict().keys())
    ldm.set_execution_backend("cinn")
    assert ldm.execution_backend == "cinn"
    assert sorted(ldm.state_dict().keys()) == keys_eager
    ldm.set_execution_backend("eager")
    assert ldm.execution_backend == "eager"

    ldm.eval()
    batch = _batch([8, 12])

    paddle.seed(7)
    with paddle.no_grad():
        eager_result = ldm.sample(
            batch, sampler="ddim", num_inference_steps=4, progress=False
        )

    seen_boundaries = []

    def eager_dispatch(name, function, *args, **kwargs):
        seen_boundaries.append(name)
        return function(*args, **kwargs)

    monkeypatch.setattr(ldm, "_run_runtime", eager_dispatch)
    ldm.set_execution_backend("cinn")
    paddle.seed(7)
    with paddle.no_grad():
        cinn_result = ldm.sample(
            batch, sampler="ddim", num_inference_steps=4, progress=False
        )

    assert seen_boundaries and set(seen_boundaries) == {"denoise_step"}
    for eager_struct, cinn_struct in zip(eager_result["result"], cinn_result["result"]):
        for key in ("atom_types", "frac_coords", "lattice"):
            assert np.array_equal(
                np.asarray(eager_struct[key]), np.asarray(cinn_struct[key])
            )


@pytest.mark.skipif(
    os.environ.get("PPMAT_RUN_CINN_WORKFLOW_TESTS") != "1",
    reason="set PPMAT_RUN_CINN_WORKFLOW_TESTS=1 to run the CINN GPU compile test",
)
def test_ldm_cinn_gpu_compile_parity():
    # Real CINN compilation on GPU: one compiled denoise step must match eager.
    if not (paddle.is_compiled_with_cuda() and paddle.base.is_compiled_with_cinn()):
        pytest.skip("CUDA Paddle built with CINN is required")

    cfg = copy.deepcopy(_load_cfg("ldm"))
    params = cfg["__init_params__"]
    params["denoiser"]["__init_params__"].update(
        hidden_dim=64, num_layers=2, num_heads=4
    )
    for part in ("encoder", "decoder"):
        params["vae"]["__init_params__"][part]["__init_params__"].update(
            d_model=64, dim_feedforward=128, num_layers=2
        )

    paddle.set_device("gpu")
    try:
        ldm = build_model(cfg)
        ldm.eval()
        x = paddle.randn([2, 8, 8])
        t = paddle.to_tensor([3, 3], dtype="int64")
        mask = paddle.ones([2, 8], dtype="bool")
        with paddle.no_grad():
            eager = ldm._denoise_step(x, t, mask=mask)
            ldm.set_execution_backend("cinn")
            cinn = ldm._denoise_step(x, t, mask=mask)
    finally:
        paddle.set_device("cpu")

    assert bool(paddle.allclose(eager, cinn, atol=2e-5).item())

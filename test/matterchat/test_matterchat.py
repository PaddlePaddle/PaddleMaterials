# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Integration tests for the MatterChat inference model."""

import json

import numpy as np
import paddle
import pytest
from cinn_workflow_harness import gpu_cinn_or_skip
from cinn_workflow_harness import requires_gpu_cinn
from pymatgen.core import Lattice
from pymatgen.core import Structure

from ppmat.models import MODEL_REGISTRY
from ppmat.models.matterchat.mistral.configuration_mistral import MistralConfig
from ppmat.models.matterchat.mistral.modeling_mistral import MistralForCausalLM
from ppmat.models.matterchat.q_former.q_former_complete import Blip2MistralInstruct

# Tiny Mistral config; the default 7B setting does not fit on one GPU.
_SMALL_LLM_CONFIG = {
    "vocab_size": 1000,
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "max_position_embeddings": 512,
}

# Minimal WordLevel tokenizer.json accepted by MistralTokenizerWrapper.
_TOK_JSON = {
    "version": "1.0",
    "truncation": None,
    "padding": None,
    "added_tokens": [
        {
            "id": 1,
            "content": "<s>",
            "single_word": False,
            "lstrip": False,
            "rstrip": False,
            "normalized": False,
            "special": True,
        },
        {
            "id": 2,
            "content": "</s>",
            "single_word": False,
            "lstrip": False,
            "rstrip": False,
            "normalized": False,
            "special": True,
        },
    ],
    "normalizer": None,
    "pre_tokenizer": {"type": "Whitespace"},
    "post_processor": None,
    "decoder": None,
    "model": {
        "type": "WordLevel",
        "vocab": {
            "<unk_0>": 0,
            "<s>": 1,
            "</s>": 2,
            "a": 3,
            "b": 4,
            "Si": 5,
        },
        "unk_token": "<unk_0>",
    },
}


def _write_tokenizer_json(path):
    path.write_text(json.dumps(_TOK_JSON))


def _build_tiny_model(tokenizer_path=None, **settings):
    paddle.seed(0)
    return Blip2MistralInstruct(
        num_query_token=4,
        prompt="test",
        max_txt_len=32,
        max_output_txt_len=32,
        qformer_text_input=False,
        llm_tokenizer=None,
        llm_model=None,
        tokenizer_path=tokenizer_path or "",
        llm_config=dict(_SMALL_LLM_CONFIG),
        **settings,
    )


@pytest.fixture
def small_model():
    """Build a tiny Blip2MistralInstruct without an LLM tokenizer."""
    model = _build_tiny_model()
    model.eval()
    return model


@pytest.fixture
def chat_model(tmp_path_factory):
    """Tiny Blip2MistralInstruct with a generated tokenizer for chat()."""
    tok_dir = tmp_path_factory.mktemp("tokenizer")
    tok_path = tok_dir / "tokenizer.json"
    _write_tokenizer_json(tok_path)
    model = _build_tiny_model(tokenizer_path=str(tok_path))
    model.eval()
    return model


@pytest.fixture
def si_structure():
    return Structure(Lattice.cubic(5.43), ["Si"] * 2, [[0, 0, 0], [0.25, 0.25, 0.25]])


@pytest.fixture
def gan_structure():
    return Structure(Lattice.cubic(4.5), ["Ga", "N"], [[0, 0, 0], [0.5, 0.5, 0.5]])


def test_model_configuration_and_structure_encoding(
    small_model, si_structure, gan_structure
):
    from dataclasses import is_dataclass

    from omegaconf import OmegaConf

    from ppmat.models.chgnet.chgnet import CHGNet
    from ppmat.models.chgnet.chgnet_graph_converter import CHGNetGraphConverter

    assert isinstance(small_model.material_encoder, CHGNet)
    assert isinstance(
        small_model.material_encoder.graph_converter, CHGNetGraphConverter
    )
    config = OmegaConf.load(
        "structure_generation/configs/matterchat/matterchat_full.yaml"
    )
    settings = OmegaConf.to_container(
        config.Model.__init_params__.llm_config, resolve=True
    )
    assert is_dataclass(small_model.llm_model.config)
    assert is_dataclass(small_model.Qformer.config)
    assert small_model._build_llm_config(settings) == small_model._build_llm_config(
        None
    )
    with pytest.raises(TypeError, match="hidden_szie"):
        small_model._build_llm_config({"hidden_szie": 64})
    with pytest.raises(ValueError, match="num_key_value_heads"):
        small_model._build_llm_config({"num_key_value_heads": 3})
    with pytest.raises(ValueError, match="qformer_text_input"):
        Blip2MistralInstruct(qformer_text_input=True)
    embeddings = []
    for structure in (si_structure, gan_structure):
        embedding = small_model.material_encoder.predict_structure_embedding(structure)
        embeddings.append(embedding)
        assert embedding.shape == [len(structure), 64]
        assert bool(paddle.isfinite(embedding).all())
        query = small_model.query_tokens.expand([1, -1, -1])
        mask = paddle.ones([1, len(structure)], dtype="int64")
        output = small_model._run_qformer(query, embedding.unsqueeze(0), mask)
        projected = small_model.llm_proj(output.last_hidden_state)
        assert projected.shape == [1, 4, _SMALL_LLM_CONFIG["hidden_size"]]
        assert bool(paddle.isfinite(projected).all())
    assert not paddle.allclose(embeddings[0], embeddings[1])
    params = config.Model.__init_params__
    configured = _build_tiny_model(
        material_encoder_config=OmegaConf.to_container(params.material_encoder_config),
        qformer_config=OmegaConf.to_container(params.qformer_config),
    )
    assert configured.state_dict().keys() == small_model.state_dict().keys()
    for name, value in small_model.state_dict().items():
        np.testing.assert_array_equal(
            value.numpy(), configured.state_dict()[name].numpy()
        )
    configured.eval()
    with paddle.no_grad():
        actual = configured.material_encoder.predict_structure_embedding(si_structure)
    np.testing.assert_array_equal(actual.numpy(), embeddings[0].numpy())
    with pytest.raises(ValueError, match="encoder_width"):
        _build_tiny_model(qformer_config={"encoder_width": 32})
    with pytest.raises(TypeError, match="hidden_szie"):
        _build_tiny_model(qformer_config={"hidden_szie": 32})
    assert "matterchat_full" not in MODEL_REGISTRY
    assert Blip2MistralInstruct.pretrained_url.startswith("https://")


def test_chat_and_generation_workflows(
    chat_model, si_structure, gan_structure, monkeypatch
):
    for structure in (si_structure, gan_structure):
        answer = chat_model.chat(structure, "a b", max_new_tokens=4)
        assert isinstance(answer, str)
    samples = {"material_sample": si_structure, "prompt": "a b"}
    generated = chat_model.generate(samples, max_length=8)
    followup = chat_model.generate_followup(samples, max_length=8)
    sampled = chat_model.generate(
        samples, num_beams=1, max_length=8, use_nucleus_sampling=True, top_p=0.3
    )
    assert len(generated) == len(followup) == 1
    assert len(sampled) == 1 and isinstance(sampled[0], str)
    assert isinstance(generated[0], str) and isinstance(followup[0], str)
    with pytest.raises(ValueError, match="one response"):
        chat_model.generate(samples, num_beams=1, max_length=8, num_captions=2)
    for layer in (chat_model.material_encoder, chat_model.llm_model):
        assert all(parameter.stop_gradient for parameter in layer.parameters())

    def prefer_eos(hidden_states):
        logits = paddle.full(
            [*hidden_states.shape[:-1], _SMALL_LLM_CONFIG["vocab_size"]], -20.0
        )
        logits[..., 2] = 10.0
        logits[..., 3] = 9.0
        logits[..., 4] = 8.0
        return logits

    llm = chat_model.llm_model
    prefix = paddle.to_tensor([[1, 5]], dtype="int64")
    with monkeypatch.context() as patch:
        patch.setattr(llm.lm_head, "forward", prefer_eos)
        for beams in (1, 5):
            for minimum, expected in (
                (0, [2]),
                (1, [3, 2]),
                (2, [3, 3, 2]),
            ):
                for budget in (max(1, minimum), minimum + 2):
                    tokens = llm.generate(
                        input_ids=prefix,
                        num_beams=beams,
                        min_length=minimum,
                        max_length=prefix.shape[1] + budget,
                        repetition_penalty=1.0,
                    )
                    assert tokens.tolist() == [expected[:budget]]

        head_calls = 0

        def prefer_continuation(hidden_states):
            nonlocal head_calls
            logits = paddle.full(
                [*hidden_states.shape[:-1], _SMALL_LLM_CONFIG["vocab_size"]], -20.0
            )
            if head_calls == 0:
                logits[..., 3] = 10.0
                logits[..., 2] = 8.0
            else:
                logits[..., 4] = 10.0
            head_calls += 1
            return logits

        patch.setattr(llm.lm_head, "forward", prefer_continuation)
        tokens = llm.generate(
            input_ids=prefix,
            num_beams=5,
            min_length=0,
            max_length=prefix.shape[1] + 2,
            repetition_penalty=1.0,
        )
        assert tokens.tolist() == [[3, 4]]


@pytest.mark.parametrize(
    "use_gpu",
    [
        pytest.param(False, id="cpu"),
        pytest.param(True, marks=requires_gpu_cinn, id="gpu"),
    ],
)
def test_attention_backends_training_and_cached_generation(use_gpu):
    if use_gpu:
        gpu_cinn_or_skip()
    paddle.seed(2026)
    settings = dict(
        _SMALL_LLM_CONFIG,
        hidden_size=256,
        intermediate_size=512,
        num_attention_heads=2,
        num_key_value_heads=1,
    )
    eager = MistralForCausalLM(MistralConfig(**settings))
    sdpa = MistralForCausalLM(MistralConfig(**settings, attn_implementation="sdpa"))
    sdpa.set_state_dict(eager.state_dict())
    assert eager.state_dict().keys() == sdpa.state_dict().keys()
    inputs = paddle.to_tensor(
        [[1, 3, 4, 5, 6, 7, 8, 9], [1, 4, 5, 3, 4, 6, 7, 8]], dtype="int64"
    )
    mask = paddle.to_tensor(
        [[1, 1, 1, 1, 1, 0, 0, 0], [1, 0, 1, 1, 0, 1, 1, 1]], dtype="int64"
    )
    labels = inputs.masked_fill(mask == 0, -100)
    outputs = []
    for model in (eager, sdpa):
        model.train()
        output = model(
            input_ids=inputs,
            attention_mask=mask,
            labels=labels,
            use_cache=False,
        )
        output.loss.backward()
        for name, parameter in model.named_parameters():
            assert parameter.grad is not None, name
            assert bool(paddle.isfinite(parameter.grad).all()), name
        outputs.append(output)
    np.testing.assert_allclose(
        outputs[0].logits.numpy(), outputs[1].logits.numpy(), rtol=1e-5, atol=1e-6
    )
    np.testing.assert_allclose(
        outputs[0].loss.numpy(), outputs[1].loss.numpy(), rtol=1e-5, atol=1e-6
    )
    for name, parameter in eager.named_parameters():
        np.testing.assert_allclose(
            parameter.grad.numpy(),
            sdpa.state_dict()[name].grad.numpy(),
            rtol=1e-4,
            atol=1e-6,
        )

    for model, output in zip((eager, sdpa), outputs):
        model.eval()
        with paddle.no_grad():
            changed = inputs.masked_fill(mask == 0, 9)
            masked = model(input_ids=changed, attention_mask=mask, use_cache=False)
            np.testing.assert_allclose(
                masked.logits.numpy()[mask.numpy().astype(bool)],
                output.logits.numpy()[mask.numpy().astype(bool)],
                rtol=1e-5,
                atol=1e-6,
            )
            prefill = model(
                input_ids=inputs[:, :3], attention_mask=mask[:, :3], use_cache=True
            )
            decoded = model(
                input_ids=inputs[:, 3:],
                attention_mask=mask,
                past_key_values=prefill.past_key_values,
                use_cache=True,
            )
            np.testing.assert_allclose(
                decoded.logits.numpy(),
                output.logits[:, 3:].numpy(),
                rtol=1e-5,
                atol=1e-6,
            )
            inspected = model(
                input_ids=inputs,
                attention_mask=mask,
                output_attentions=True,
                use_cache=False,
            )
            assert len(inspected.attentions) == _SMALL_LLM_CONFIG["num_hidden_layers"]
            np.testing.assert_allclose(
                inspected.logits.numpy(), output.logits.numpy(), rtol=1e-5, atol=1e-6
            )


def _registry_layout(tmp_path, model):
    """Materialize a complete local model package for checkpoint integration."""
    import os

    config = {
        "Model": {
            "__class_name__": "Blip2MistralInstruct",
            "__init_params__": {
                "num_query_token": 4,
                "prompt": "test",
                "max_txt_len": 32,
                "max_output_txt_len": 32,
                "qformer_text_input": False,
                "llm_tokenizer": None,
                "llm_model": None,
                "tokenizer_path": "",
                "llm_config": dict(_SMALL_LLM_CONFIG),
            },
        }
    }
    try:
        from omegaconf import OmegaConf

        OmegaConf.save(config, str(tmp_path / "matterchat_full.yaml"))
    except ImportError:
        import yaml

        with open(tmp_path / "matterchat_full.yaml", "w") as f:
            yaml.safe_dump(config, f)
    os.makedirs(tmp_path / "checkpoints", exist_ok=True)
    _write_tokenizer_json(tmp_path / "tokenizer.json")
    paddle.save(model.state_dict(), str(tmp_path / "checkpoints" / "best.pdparams"))
    return tmp_path


def test_pretrained_package_roundtrip(small_model, tmp_path, monkeypatch):
    """from_pretrained rebuilds the model and restores saved weights."""
    with pytest.raises(FileNotFoundError, match="yaml"):
        Blip2MistralInstruct.from_pretrained(str(tmp_path))
    with pytest.raises(FileNotFoundError, match="No weights"):
        small_model._load_package_weights(str(tmp_path))
    path = _registry_layout(tmp_path, small_model)
    downloads = []

    def download_package(url):
        downloads.append(url)
        return str(path)

    monkeypatch.setattr(
        "ppmat.utils.download.get_weights_path_from_url", download_package
    )
    loaded = Blip2MistralInstruct.from_pretrained("matterchat_full")
    assert downloads == [Blip2MistralInstruct.pretrained_url]
    ref_sd = small_model.state_dict()
    loaded_sd = loaded.state_dict()
    assert set(ref_sd.keys()) == set(loaded_sd.keys())
    for key in ("llm_proj.weight", "llm_proj.bias", "query_tokens"):
        assert paddle.allclose(ref_sd[key], loaded_sd[key])

    checkpoints = path / "checkpoints"
    partial = checkpoints / "partial.pdparams"
    paddle.save({"llm_proj.bias": ref_sd["llm_proj.bias"]}, str(partial))
    with pytest.raises(ValueError, match="Checkpoint keys mismatch"):
        loaded._load_single_weight(str(partial))
    partial.unlink()

    items = list(ref_sd.items())
    midpoint = len(items) // 2
    shards = (dict(items[:midpoint]), dict(items[midpoint:]))
    weight_map = {}
    for number, shard in enumerate(shards):
        name = f"shard_{number}.pdparams"
        paddle.save(shard, str(checkpoints / name))
        weight_map.update({key: name for key in shard})
    index_path = checkpoints / "model.pdparams.index.json"
    index_path.write_text(json.dumps({"weight_map": weight_map}))
    loaded._load_package_weights(str(path))
    for key in ref_sd:
        np.testing.assert_array_equal(
            loaded.state_dict()[key].numpy(), ref_sd[key].numpy()
        )
    del weight_map["llm_proj.bias"]
    index_path.write_text(json.dumps({"weight_map": weight_map}))
    with pytest.raises(ValueError, match="Checkpoint keys mismatch"):
        loaded._load_package_weights(str(path))


def test_cinn_chat_dispatch_preserves_eager_output(
    chat_model, si_structure, monkeypatch
):
    keys = tuple(chat_model.state_dict())
    expected = chat_model.chat(si_structure, "a b", max_new_tokens=4)
    recorded = []

    def run_runtime(owner, name, function, *args, **kwargs):
        recorded.append(name)
        return function(*args, **kwargs)

    monkeypatch.setattr(type(chat_model.llm_model.model), "_run_runtime", run_runtime)
    try:
        chat_model.set_execution_backend("cinn")
        assert chat_model.execution_backend == "cinn"
        assert chat_model.llm_model.model.execution_backend == "cinn"
        assert chat_model.chat(si_structure, "a b", max_new_tokens=4) == expected
        assert recorded and set(recorded) == {"forward"}
        assert tuple(chat_model.state_dict()) == keys
    finally:
        chat_model.set_execution_backend("eager")


@requires_gpu_cinn
def test_gpu_cinn_chat_matches_eager(chat_model, si_structure):
    gpu_cinn_or_skip()
    chat_model.to(paddle.get_device())
    owner = chat_model.llm_model.model
    observed = []

    def collect_hidden(layer, inputs, output):
        observed.append(output[0].numpy().copy())

    hook = owner.register_forward_post_hook(collect_hidden)
    try:
        keys = tuple(chat_model.state_dict())
        chat_model.set_execution_backend("eager")
        expected = chat_model.chat(si_structure, "a b", max_new_tokens=4)
        eager_hidden = list(observed)
        observed.clear()
        chat_model.set_execution_backend("cinn")
        actual = chat_model.chat(si_structure, "a b", max_new_tokens=4)
        assert actual == expected
        assert len(observed) == len(eager_hidden)
        for compiled, eager in zip(observed, eager_hidden):
            np.testing.assert_allclose(compiled, eager, atol=2e-5, rtol=2e-5)
        assert ("cinn", "eval", "forward") in owner._runtime_cache
        assert tuple(chat_model.state_dict()) == keys
    finally:
        hook.remove()
        chat_model.set_execution_backend("eager")
        chat_model.to("cpu")
        paddle.set_device("cpu")


def test_official_trainer_training_and_resume(tmp_path, si_structure, monkeypatch):
    from cinn_workflow_harness import assert_state_dict_close
    from cinn_workflow_harness import trainer_config
    from omegaconf import OmegaConf

    from ppmat.datasets import build_dataloader
    from ppmat.models import build_model
    from ppmat.optimizer import build_optimizer
    from ppmat.trainer import BaseTrainer

    config = OmegaConf.load(
        "structure_generation/configs/matterchat/matterchat_train.yaml"
    )
    tokenizer_path = tmp_path / "tokenizer.json"
    _write_tokenizer_json(tokenizer_path)
    config.Model.__init_params__.tokenizer_path = str(tokenizer_path)
    llm_config = dict(_SMALL_LLM_CONFIG, vocab_size=32769)
    config.Model.__init_params__.llm_config = llm_config
    config.Model.__init_params__.num_query_token = 4
    config.Model.__init_params__.material_encoder_config = {"atom_fea_dim": 32}
    config.Dataset.train.dataset.__init_params__.embedding_dim = 32
    config.Model.__init_params__.qformer_config = {
        "hidden_size": 32,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "intermediate_size": 64,
    }
    model_config = OmegaConf.to_container(config.Model, resolve=True)
    paddle.seed(2026)
    with paddle.utils.unique_name.guard():
        model = build_model(model_config)
    model.eval()
    larger_structure = si_structure.copy()
    larger_structure.make_supercell([2, 1, 1])
    with paddle.no_grad():
        features = [
            model.material_encoder.predict_structure_embedding(structure).numpy()
            for structure in (si_structure, larger_structure)
        ]
    data_path = tmp_path / "train.npz"
    np.savez(
        data_path,
        material_embedding=np.concatenate(features),
        num_atoms=np.array([len(feature) for feature in features]),
        text_input=np.array(["a", "a b"]),
        text=np.array(["b", "a b"]),
    )
    config.Dataset.train.dataset.__init_params__.path = str(data_path)
    config.Dataset.train.sampler.__init_params__.shuffle = False
    loader = build_dataloader(
        OmegaConf.to_container(config.Dataset.train, resolve=True)
    )
    batch = next(iter(loader))
    assert batch["num_atoms"].tolist() == [2, 4]
    assert batch["material_embedding"].shape == [6, 32]
    model.train()
    with pytest.raises(ValueError, match="width must match"):
        model(dict(batch, material_embedding=batch["material_embedding"][:, :-1]))
    captured_labels = []
    llm_forward = model.llm_model.forward

    def capture_labels(*args, **kwargs):
        captured_labels.append(kwargs["labels"].numpy().copy())
        return llm_forward(*args, **kwargs)

    monkeypatch.setattr(model.llm_model, "forward", capture_labels)
    loss = model(batch)["loss_dict"]["loss"]
    monkeypatch.setattr(model.llm_model, "forward", llm_forward)
    np.testing.assert_array_equal(
        captured_labels[0],
        [
            [-100, -100, -100, -100, -100, 4, 2, -100, -100],
            [-100, -100, -100, -100, -100, -100, 3, 4, 2],
        ],
    )
    assert bool(paddle.isfinite(loss))
    loss.backward()
    for name, parameter in model.named_parameters():
        if parameter.stop_gradient:
            assert parameter.grad is None, name
        else:
            assert parameter.grad is not None, name
            assert bool(paddle.isfinite(parameter.grad).all()), name
    model.clear_gradients()
    initial_projection = model.llm_proj.weight.numpy().copy()
    frozen_embedding = model.llm_model.model.embed_tokens.weight.numpy().copy()
    optimizer_config = OmegaConf.to_container(config.Optimizer, resolve=True)
    optimizer_config["__init_params__"]["lr"] = {
        "__class_name__": "ExponentialDecay",
        "__init_params__": {
            "learning_rate": 0.0001,
            "gamma": 0.5,
            "decay_steps": 1,
            "by_epoch": False,
        },
    }
    initial_state = {name: value.clone() for name, value in model.state_dict().items()}
    optimizer, scheduler = build_optimizer(optimizer_config, model, 2, len(loader))
    trainer = BaseTrainer(
        trainer_config(tmp_path / "first", 1),
        model,
        train_dataloader=loader,
        val_dataloader=loader,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        execution_config={"backend": "eager"},
    )
    with paddle.utils.unique_name.guard():
        trainer.train()
    assert trainer.state.global_step == 1
    assert not np.array_equal(initial_projection, model.llm_proj.weight.numpy())
    np.testing.assert_array_equal(
        frozen_embedding, model.llm_model.model.embed_tokens.weight.numpy()
    )
    checkpoint = tmp_path / "first" / "checkpoints" / "latest"
    for suffix in ("pdparams", "pdopt", "pdstates"):
        assert checkpoint.with_suffix("." + suffix).is_file()
    for name, parameter in model.named_parameters():
        assert bool(paddle.isfinite(parameter).all()), name
    expected_keys = tuple(model.state_dict())
    with paddle.utils.unique_name.guard():
        restored = build_model(model_config)
    restored_optimizer, restored_scheduler = build_optimizer(
        optimizer_config, restored, 2, len(loader)
    )
    resumed = BaseTrainer(
        trainer_config(tmp_path / "resumed", 2),
        restored,
        train_dataloader=loader,
        val_dataloader=loader,
        optimizer=restored_optimizer,
        lr_scheduler=restored_scheduler,
        execution_config={"backend": "eager"},
    )
    # Zero epochs after restoration isolate checkpoint restoration from updates.
    resumed.max_epochs = 1
    with paddle.utils.unique_name.guard():
        resumed.train(resume_from_checkpoint=str(checkpoint))
    assert resumed.state.global_step == 1
    assert tuple(restored.state_dict()) == expected_keys
    assert_state_dict_close(restored.state_dict(), model.state_dict(), 0)
    expected_optimizer = optimizer.state_dict()
    actual_optimizer = restored_optimizer.state_dict()
    assert actual_optimizer.pop("LR_Scheduler") == expected_optimizer.pop(
        "LR_Scheduler"
    )
    assert_state_dict_close(actual_optimizer, expected_optimizer, 0)
    assert restored_scheduler.state_dict() == scheduler.state_dict()
    assert restored_optimizer.get_lr() == optimizer.get_lr() == 0.00005
    assert resumed.state.epoch == trainer.state.epoch == 1
    resumed.max_epochs = 2
    with paddle.utils.unique_name.guard():
        resumed.train(resume_from_checkpoint=str(checkpoint))
    assert resumed.state.global_step == 2
    for name, parameter in restored.named_parameters():
        assert bool(paddle.isfinite(parameter).all()), name

    with paddle.utils.unique_name.guard():
        straight_model = build_model(model_config)
        straight_model.set_state_dict(initial_state)
        straight_optimizer, straight_scheduler = build_optimizer(
            optimizer_config, straight_model, 2, len(loader)
        )
    straight = BaseTrainer(
        trainer_config(tmp_path / "straight", 2),
        straight_model,
        train_dataloader=loader,
        val_dataloader=loader,
        optimizer=straight_optimizer,
        lr_scheduler=straight_scheduler,
        execution_config={"backend": "eager"},
    )
    with paddle.utils.unique_name.guard():
        straight.train()
    assert resumed.state.epoch == straight.state.epoch == 2
    assert resumed.state.global_step == straight.state.global_step == 2
    assert restored_scheduler.state_dict() == straight_scheduler.state_dict()
    assert restored_optimizer.get_lr() == straight_optimizer.get_lr() == 0.000025
    assert restored.state_dict().keys() == straight_model.state_dict().keys()
    resumed_state = restored_optimizer.state_dict()
    straight_state = straight_optimizer.state_dict()
    assert resumed_state.keys() == straight_state.keys()
    assert resumed_state.pop("LR_Scheduler") == straight_state.pop("LR_Scheduler")
    for name in resumed_state:
        assert resumed_state[name].shape == straight_state[name].shape

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

"""Complete MatterChat model: Blip2MistralInstruct."""

import json
import os

import paddle
import paddle.nn as nn
from omegaconf import OmegaConf

from ppmat.models.common.runtime import RuntimeMixin
from ppmat.models.matterchat.mistral.configuration_mistral import MistralConfig
from ppmat.models.matterchat.mistral.modeling_mistral import MistralForCausalLM
from ppmat.models.matterchat.q_former.q_former_base import Blip2Base
from ppmat.utils import download
from ppmat.utils.model_package import resolve_model_package_dir


class Blip2MistralInstruct(RuntimeMixin, Blip2Base):
    pretrained_model_name = "matterchat_full"
    pretrained_url = "https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/llm/MatterChat/matterchat_full.zip"  # noqa

    def __init__(
        self,
        num_query_token=32,
        prompt="test",
        max_txt_len=512,
        max_output_txt_len=1024,
        qformer_text_input=False,
        llm_tokenizer=None,
        llm_model=None,
        tokenizer_path=None,
        llm_config=None,
        material_encoder_config=None,
        qformer_config=None,
        pretrained_package=None,
        execution_backend="eager",
        runtime_options=None,
    ):
        super().__init__()
        if num_query_token < 1:
            raise ValueError("num_query_token must be positive.")
        if qformer_text_input:
            raise ValueError(
                "Only qformer_text_input=False is supported by MatterChat."
            )

        if (
            pretrained_package is not None
            and llm_tokenizer is None
            and not tokenizer_path
        ):
            tokenizer_path = os.path.join(pretrained_package, "tokenizer.json")
        if llm_tokenizer is None and tokenizer_path is not None and tokenizer_path:
            from ppmat.models.matterchat.utils.tokenizer import MistralTokenizerWrapper

            _resolved = os.path.expanduser(tokenizer_path)
            llm_tokenizer = MistralTokenizerWrapper(_resolved)

        self.material_encoder = self.init_material_encoder(material_encoder_config)
        for param in self.material_encoder.parameters():
            param.stop_gradient = True

        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token, self.material_encoder.atom_fea_dim, qformer_config
        )

        self.Qformer.bert.embeddings.word_embeddings = None
        self.Qformer.bert.embeddings.position_embeddings = None
        for layer in self.Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None
        self.Qformer.cls = None

        self.llm_tokenizer = llm_tokenizer
        self.llm_model = (
            llm_model
            if llm_model is not None
            else MistralForCausalLM(self._build_llm_config(llm_config))
        )

        self.set_execution_backend(execution_backend)
        self.set_runtime_options(runtime_options or {})
        for param in self.llm_model.parameters():
            param.stop_gradient = True

        self.llm_proj = nn.Linear(
            self.Qformer.config.hidden_size, self.llm_model.config.hidden_size
        )

        self.max_txt_len = max_txt_len
        self.max_output_txt_len = max_output_txt_len
        self.prompt = prompt

        if self.llm_tokenizer is not None:
            prompt_tokens = self.llm_tokenizer(self.prompt, return_tensors="pd")
            self.prompt_length = prompt_tokens.attention_mask.sum(1).item()
        else:
            self.prompt_length = 0
        if pretrained_package is not None:
            self._load_package_weights(pretrained_package)

    @property
    def execution_backend(self):
        return self.llm_model.model.execution_backend

    def set_execution_backend(self, backend):
        self.llm_model.model.set_execution_backend(backend)

    def set_runtime_options(self, runtime_options):
        self.llm_model.model.set_runtime_options(runtime_options)

    def validate_execution_backend(self, *, use_amp=False, world_size=1):
        self.llm_model.model.validate_execution_backend(
            use_amp=use_amp, world_size=world_size
        )

    @staticmethod
    def _build_llm_config(llm_config):
        """Build the Mistral config; a dict selects a smaller custom LLM."""
        return MistralConfig(**({} if llm_config is None else llm_config))

    @classmethod
    def from_pretrained(cls, path):
        """Load the published model name or an explicit local model package."""
        if path == cls.pretrained_model_name:
            extracted_path = download.get_weights_path_from_url(cls.pretrained_url)
            path = resolve_model_package_dir(cls.pretrained_model_name, extracted_path)

        yaml_files = [
            f for f in os.listdir(path) if f.endswith(".yaml") or f.endswith(".yml")
        ]
        if not yaml_files:
            raise FileNotFoundError(f"No yaml config found in {path}")
        if len(yaml_files) != 1:
            raise ValueError(f"Expected exactly one model yaml config in {path}.")
        config_path = os.path.join(path, yaml_files[0])
        config = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)

        init_params = config["Model"]["__init_params__"]

        tokenizer_path = os.path.join(path, "tokenizer.json")
        if os.path.exists(tokenizer_path):
            init_params["tokenizer_path"] = tokenizer_path

        init_params["pretrained_package"] = path
        return cls(**init_params)

    def _load_package_weights(self, weight_dir):
        ckpt_dir = os.path.join(weight_dir, "checkpoints")
        if not os.path.isdir(ckpt_dir):
            ckpt_dir = weight_dir

        index_path = os.path.join(ckpt_dir, "model.pdparams.index.json")
        if os.path.exists(index_path):
            self._load_sharded_weights(ckpt_dir, index_path)
            return

        best_path = os.path.join(ckpt_dir, "best.pdparams")
        if os.path.exists(best_path):
            self._load_single_weight(best_path)
            return

        raise FileNotFoundError(
            f"No weights found in {ckpt_dir}: "
            f"expected best.pdparams or model.pdparams.index.json"
        )

    def _load_sharded_weights(self, ckpt_dir, index_path):
        with open(index_path) as f:
            idx = json.load(f)

        model_sd = self._validate_checkpoint_keys(idx["weight_map"])
        shard_files = sorted(set(idx["weight_map"].values()))
        for shard_file in shard_files:
            shard_path = os.path.join(ckpt_dir, shard_file)
            shard = paddle.load(shard_path, return_numpy=True)
            for key, val in shard.items():
                if idx["weight_map"].get(key) != shard_file:
                    raise ValueError(
                        f"Unexpected checkpoint tensor {key} in {shard_file}."
                    )
                self._assign_checkpoint_tensor(key, val, model_sd[key])
            expected_keys = {
                key for key, name in idx["weight_map"].items() if name == shard_file
            }
            if set(shard) != expected_keys:
                raise ValueError(f"Missing checkpoint tensors in {shard_file}.")
            del shard

    def _load_single_weight(self, weight_path):
        state_dict = paddle.load(weight_path, return_numpy=True)
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        model_sd = self._validate_checkpoint_keys(state_dict)
        for key, value in state_dict.items():
            self._assign_checkpoint_tensor(key, value, model_sd[key])

    @staticmethod
    def _assign_checkpoint_tensor(key, value, target):
        if list(value.shape) != list(target.shape):
            raise ValueError(f"Checkpoint shape mismatch for {key}.")
        target.set_value(
            paddle.to_tensor(value, dtype=target.dtype, place=target.place)
        )

    def _validate_checkpoint_keys(self, checkpoint):
        model_sd = self.state_dict()
        missing = sorted(set(model_sd) - set(checkpoint))
        unexpected = sorted(set(checkpoint) - set(model_sd))
        if missing or unexpected:
            raise ValueError(
                f"Checkpoint keys mismatch: missing={missing}, unexpected={unexpected}."
            )
        return model_sd

    def _run_qformer(self, query_tokens, material_embeds, material_atts):
        """Encode crystal features using the released query-only Q-Former."""
        return self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=material_embeds,
            encoder_attention_mask=material_atts,
            return_dict=True,
        )

    def _encode_for_generate(self, samples):
        """Shared encoding pipeline for generate / generate_followup."""
        self.llm_tokenizer.padding_side = "left"
        prompt = samples.get("prompt", self.prompt)
        if isinstance(prompt, str):
            prompt = [prompt]

        material_embed = self.material_encoder.predict_structure_embedding(
            samples["material_sample"]
        ).unsqueeze(0)
        material_att = paddle.ones(material_embed.shape[:-1], dtype=paddle.int64)

        query_tokens = self.query_tokens.expand([1, -1, -1])
        query_output = self._run_qformer(query_tokens, material_embed, material_att)

        inputs_llm = self.llm_proj(
            query_output.last_hidden_state[:, : query_tokens.shape[1], :]
        )
        atts_llm = paddle.ones(inputs_llm.shape[:-1], dtype=paddle.int64)

        prompt_tokens = self.llm_tokenizer(
            prompt, return_tensors="pd", padding="longest"
        )
        input_embeds = self.llm_model.get_input_embeddings()(prompt_tokens.input_ids)
        inputs_embeds = paddle.concat([inputs_llm, input_embeds], axis=1)
        attention_mask = paddle.concat([atts_llm, prompt_tokens.attention_mask], axis=1)
        return inputs_embeds, attention_mask, prompt

    def concat_text_input_output(self, input_ids, input_atts, output_ids, output_atts):
        input_part_targets_len = []
        input_ids_list, attention_mask_list = [], []

        for i in range(input_ids.shape[0]):
            this_input_len = input_atts[i].sum()
            input_part_targets_len.append(this_input_len)

            input_ids_list.append(
                paddle.concat(
                    [
                        input_ids[i][:this_input_len],
                        output_ids[i],
                        input_ids[i][this_input_len:],
                    ]
                )
            )

            attention_mask_list.append(
                paddle.concat(
                    [
                        input_atts[i][:this_input_len],
                        output_atts[i],
                        input_atts[i][this_input_len:],
                    ]
                )
            )

        return {
            "input_ids": paddle.stack(input_ids_list),
            "attention_mask": paddle.stack(attention_mask_list),
        }, input_part_targets_len

    def forward(self, samples):
        """Train the Q-Former bridge on packed frozen-encoder features and text."""
        if self.llm_tokenizer is None:
            raise ValueError("Training requires a tokenizer_path or llm_tokenizer.")
        num_atoms = samples["num_atoms"].tolist()
        embeddings = samples["material_embedding"]
        if (
            embeddings.ndim != 2
            or embeddings.shape[1] != self.material_encoder.atom_fea_dim
        ):
            raise ValueError(
                "material_embedding width must match the material encoder."
            )
        embedding_list = paddle.split(embeddings, num_atoms, axis=0)
        max_atoms = max(num_atoms)
        material_embeds = paddle.stack(
            [
                paddle.concat(
                    [
                        embedding,
                        paddle.zeros(
                            [max_atoms - count, embedding.shape[1]],
                            dtype=embedding.dtype,
                        ),
                    ],
                    axis=0,
                )
                for embedding, count in zip(embedding_list, num_atoms)
            ]
        )
        material_atts = (
            paddle.arange(max_atoms).unsqueeze(0)
            < paddle.to_tensor(num_atoms).unsqueeze(1)
        ).cast("int64")
        query_tokens = self.query_tokens.expand([material_embeds.shape[0], -1, -1])

        query_output = self._run_qformer(query_tokens, material_embeds, material_atts)

        inputs_llm = self.llm_proj(
            query_output.last_hidden_state[:, : query_tokens.shape[1], :]
        )
        atts_llm = paddle.ones(inputs_llm.shape[:-1], dtype=paddle.int64)

        self.llm_tokenizer.padding_side = "right"
        self.llm_tokenizer.truncation_side = "left"

        if "text_input" in samples:
            text_input_tokens = self.llm_tokenizer(
                samples["text_input"],
                return_tensors="pd",
                padding="longest",
                truncation=True,
                max_length=self.max_txt_len,
            )
        else:
            bos_token = self.llm_tokenizer.bos_token
            input_text = [bos_token] * material_embeds.shape[0]
            text_input_tokens = self.llm_tokenizer(
                input_text,
                return_tensors="pd",
                padding="longest",
                truncation=True,
                max_length=self.max_txt_len,
            )

        self.llm_tokenizer.truncation_side = "right"
        text_output_tokens = self.llm_tokenizer(
            [t + self.llm_tokenizer.eos_token for t in samples["text"]],
            return_tensors="pd",
            padding="longest",
            truncation=True,
            max_length=self.max_output_txt_len,
            add_special_tokens=False,
        )

        llm_tokens, input_target_lengths = self.concat_text_input_output(
            text_input_tokens.input_ids,
            text_input_tokens.attention_mask,
            text_output_tokens.input_ids,
            text_output_tokens.attention_mask,
        )

        targets = llm_tokens["input_ids"].masked_fill(
            llm_tokens["attention_mask"] == 0, -100
        )
        for i, length in enumerate(input_target_lengths):
            targets[i, :length] = -100

        empty_targets = paddle.ones(atts_llm.shape, dtype=paddle.int64).fill_(-100)
        targets = paddle.concat([empty_targets, targets], axis=1)

        inputs_embeds = self.llm_model.get_input_embeddings()(llm_tokens["input_ids"])
        inputs_embeds = paddle.concat(
            [inputs_llm.cast(inputs_embeds.dtype), inputs_embeds], axis=1
        )
        attention_mask = paddle.concat([atts_llm, llm_tokens["attention_mask"]], axis=1)

        outputs = self.llm_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
            labels=targets,
            use_cache=False,
        )

        return {"loss_dict": {"loss": outputs.loss}, "pred_dict": {}}

    @paddle.no_grad()
    def chat(self, structure, prompt, max_new_tokens=64):
        """Single-turn dialogue over a crystal structure and a text prompt."""
        embed = self.material_encoder.predict_structure_embedding(structure).unsqueeze(
            0
        )
        mask = paddle.ones(embed.shape[:-1], dtype=paddle.int64)
        qt = self.query_tokens.expand([1, -1, -1])
        qo = self._run_qformer(qt, embed, mask)
        inputs_llm = self.llm_proj(qo.last_hidden_state[:, : qt.shape[1], :])
        atts_llm = paddle.ones(inputs_llm.shape[:-1], dtype=paddle.int64)
        self.llm_tokenizer.padding_side = "left"
        pt = self.llm_tokenizer([prompt], return_tensors="pd", padding="longest")
        ie = self.llm_model.get_input_embeddings()(pt.input_ids)
        ie = paddle.concat([inputs_llm, ie], axis=1)
        am = paddle.concat([atts_llm, pt.attention_mask], axis=1)
        out = self.llm_model.generate(
            inputs_embeds=ie,
            attention_mask=am,
            do_sample=False,
            max_length=ie.shape[1] + max_new_tokens,
        )
        out[out == 0] = 2
        return self.llm_tokenizer.decode(out[0], skip_special_tokens=True).strip()

    @paddle.no_grad()
    def generate(
        self,
        samples,
        use_nucleus_sampling=False,
        num_beams=5,
        max_length=256,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.5,
        length_penalty=1,
        num_captions=1,
        temperature=1,
    ):
        inputs_embeds, attention_mask, _prompt = self._encode_for_generate(samples)

        with self.maybe_autocast():
            outputs = self.llm_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                do_sample=use_nucleus_sampling,
                top_p=top_p,
                temperature=temperature,
                num_beams=num_beams,
                max_length=max_length,
                min_length=min_length,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                num_return_sequences=num_captions,
            )

        outputs[outputs == 0] = 2  # sanitize output
        return [
            self.llm_tokenizer.decode(o, skip_special_tokens=True).strip()
            for o in outputs
        ]

    @paddle.no_grad()
    def generate_followup(
        self,
        samples,
        use_nucleus_sampling=False,
        num_beams=5,
        max_length=256,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.5,
        length_penalty=1,
        num_captions=1,
        temperature=1,
    ):
        inputs_embeds, attention_mask, prompt = self._encode_for_generate(samples)

        with self.maybe_autocast():
            outputs = self.llm_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                do_sample=use_nucleus_sampling,
                top_p=top_p,
                temperature=temperature,
                num_beams=num_beams,
                max_length=max_length,
                min_length=min_length,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                num_return_sequences=num_captions,
            )

        outputs[outputs == 0] = 2  # sanitize output
        decoded_outputs = [
            self.llm_tokenizer.decode(o, skip_special_tokens=True).strip()
            for o in outputs
        ]

        cleaned_outputs = []
        for out, p in zip(decoded_outputs, prompt * num_captions):
            if out.lower().startswith(p.lower()):
                out = out[len(p) :].lstrip(":,.- \n")
            cleaned_outputs.append(out)

        return cleaned_outputs

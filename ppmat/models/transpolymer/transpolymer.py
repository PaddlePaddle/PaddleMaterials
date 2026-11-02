import json
import os
import shutil
import subprocess
import tempfile

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import pandas as pd

from ppmat.models.transpolymer.modeling import RobertaModel
from ppmat.models.transpolymer.tokenizer import PolymerSmilesTokenizer
from ppmat.utils import download


TRANS_POLYMER_VOCAB_URL = (
    "https://paddle-org.bj.bcebos.com/paddlematerials/assets/vocabs/"
    "transpolymer_vocab.tar.gz"
)
TRANS_POLYMER_VOCAB_MD5 = "4983d7128d9f7dbd587cf3ed0bd8ff6e"

ROBERTA_BACKBONE_CONFIG_KEYS = (
    "vocab_size",
    "hidden_size",
    "num_hidden_layers",
    "num_attention_heads",
    "intermediate_size",
    "hidden_dropout_prob",
    "attention_probs_dropout_prob",
    "max_position_embeddings",
    "type_vocab_size",
    "initializer_range",
    "layer_norm_eps",
    "pad_token_id",
    "position_embedding_type",
)


class TransPolymerRegressor(nn.Layer):
    """TransPolymer encoder with a regression head for polymer property prediction."""

    @staticmethod
    def _is_pretrained_checkpoint_ready(pretrained_model_path):
        if pretrained_model_path is None:
            return False
        config_path = os.path.join(pretrained_model_path, "config.json")
        state_path = os.path.join(pretrained_model_path, "model_state.pdparams")
        return (
            os.path.exists(config_path)
            and os.path.exists(state_path)
            and os.path.getsize(state_path) > 1024 * 1024
        )

    @classmethod
    def _prepare_pretrained_checkpoint(cls, pretrained_model_path, pretrained_model_url):
        if cls._is_pretrained_checkpoint_ready(pretrained_model_path):
            return
        if pretrained_model_url is None:
            return
        if shutil.which("git") is None:
            raise RuntimeError(
                "TransPolymer pretrained checkpoint is missing and git is not installed. "
                f"Please download {pretrained_model_url} to {pretrained_model_path}."
            )

        os.makedirs(pretrained_model_path, exist_ok=True)
        with tempfile.TemporaryDirectory() as tmp_dir:
            repo_dir = os.path.join(tmp_dir, "transpolymer_pretrain")
            subprocess.run(["git", "lfs", "install"], check=True)
            subprocess.run(
                ["git", "clone", pretrained_model_url, repo_dir],
                check=True,
            )
            for filename in ("config.json", "model_state.pdparams"):
                src = os.path.join(repo_dir, filename)
                if not os.path.exists(src):
                    raise FileNotFoundError(
                        f"Missing {filename} in downloaded TransPolymer checkpoint."
                    )
                shutil.copy2(src, os.path.join(pretrained_model_path, filename))
            if not cls._is_pretrained_checkpoint_ready(pretrained_model_path):
                raise RuntimeError(
                    "Downloaded TransPolymer checkpoint is incomplete. "
                    "Please check that Git LFS is installed and enabled."
                )

    @staticmethod
    def _load_pretrained_encoder(pretrained_model_path):
        """Build the backbone from checkpoint metadata and load matching tensors."""
        config_path = os.path.join(pretrained_model_path, "config.json")
        with open(config_path, "r", encoding="utf-8") as f:
            checkpoint_config = json.load(f)
        backbone_params = {
            key: checkpoint_config[key]
            for key in ROBERTA_BACKBONE_CONFIG_KEYS
            if key in checkpoint_config
        }
        encoder = RobertaModel(**backbone_params)

        state_path = os.path.join(pretrained_model_path, "model_state.pdparams")
        raw_state = paddle.load(state_path)
        current_state = encoder.state_dict()
        loaded_state = {}
        for key, value in raw_state.items():
            model_key = key[len("roberta.") :] if key.startswith("roberta.") else key
            if model_key in current_state and list(current_state[model_key].shape) == list(
                value.shape
            ):
                loaded_state[model_key] = value
        current_state.update(loaded_state)
        encoder.set_state_dict(current_state)
        print(f"Loaded {len(loaded_state)} RobertaModel tensors from {state_path}")
        return encoder

    @staticmethod
    def _resolve_vocab_file(vocab_sup_file, vocab_url, vocab_md5):
        if vocab_sup_file is None or os.path.exists(vocab_sup_file):
            return vocab_sup_file

        root_path = download.get_datasets_path_from_url(vocab_url, vocab_md5)
        downloaded_path = os.path.join(root_path, os.path.basename(vocab_sup_file))
        if not os.path.exists(downloaded_path):
            raise FileNotFoundError(
                f"{os.path.basename(vocab_sup_file)} is missing from downloaded "
                f"TransPolymer vocabulary archive at {root_path}."
            )
        return downloaded_path

    def __init__(
        self,
        pretrained_model_path=None,
        pretrained_model_url=None,
        vocab_size=50265,
        hidden_size=768,
        intermediate_size=3072,
        max_position_embeddings=514,
        num_attention_heads=12,
        num_hidden_layers=6,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        drop_rate=0.1,
        resize_vocab_size=None,
        tokenizer_name_or_path="roberta-base",
        tokenizer_cache_dir=None,
        vocab_sup_file=None,
        vocab_url=TRANS_POLYMER_VOCAB_URL,
        vocab_md5=TRANS_POLYMER_VOCAB_MD5,
        blocksize=411,
        smiles_key="smiles",
        use_token_cache=True,
        property_name="Conductivity [S/cm]",
        data_mean=0.0,
        data_std=1.0,
        loss_type="mse_loss",
    ):
        super().__init__()
        if pretrained_model_path:
            self._prepare_pretrained_checkpoint(
                pretrained_model_path, pretrained_model_url
            )
            encoder = self._load_pretrained_encoder(pretrained_model_path)
        else:
            encoder = RobertaModel(
                vocab_size=vocab_size,
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                max_position_embeddings=max_position_embeddings,
                num_attention_heads=num_attention_heads,
                num_hidden_layers=num_hidden_layers,
                hidden_dropout_prob=hidden_dropout_prob,
                attention_probs_dropout_prob=attention_probs_dropout_prob,
            )

        self.encoder = encoder
        tokenizer_kwargs = {"max_len": blocksize}
        if tokenizer_cache_dir is not None:
            tokenizer_kwargs["cache_dir"] = tokenizer_cache_dir
        self.tokenizer = PolymerSmilesTokenizer.from_pretrained(
            tokenizer_name_or_path, **tokenizer_kwargs
        )
        vocab_sup_file = self._resolve_vocab_file(
            vocab_sup_file, vocab_url, vocab_md5
        )
        if vocab_sup_file is not None:
            vocab_sup = pd.read_csv(vocab_sup_file, header=None).values.flatten()
            self.tokenizer.add_tokens(vocab_sup.tolist())
        if resize_vocab_size is None:
            resize_vocab_size = len(self.tokenizer)
        if resize_vocab_size is not None:
            self.encoder.resize_token_embeddings(resize_vocab_size)
        self.blocksize = blocksize
        self.smiles_key = smiles_key
        self.use_token_cache = use_token_cache
        self._token_cache = {}
        if isinstance(property_name, list):
            self.property_name = property_name[0]
        else:
            self.property_name = property_name
        self.register_buffer(tensor=paddle.to_tensor(data_mean), name="data_mean")
        self.register_buffer(tensor=paddle.to_tensor(data_std), name="data_std")
        if loss_type == "mse_loss":
            self.loss_fn = F.mse_loss
        elif loss_type == "l1_loss":
            self.loss_fn = F.l1_loss
        else:
            raise ValueError(f"Unknown loss type {loss_type}.")

        hidden_size = self.encoder.hidden_size
        self.regressor = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(hidden_size, hidden_size),
            nn.Silu(),
            nn.Linear(hidden_size, 1),
        )

    def normalize(self, tensor):
        return (tensor - self.data_mean) / self.data_std

    def unnormalize(self, tensor):
        return tensor * self.data_std + self.data_mean

    def _forward(self, input_ids, attention_mask):
        hidden_states = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = hidden_states[:, 0, :]
        return self.regressor(cls_embedding)

    def _encode_smiles(self, smiles):
        if self.use_token_cache and smiles in self._token_cache:
            return self._token_cache[smiles]

        encoding = self.tokenizer(
            smiles,
            add_special_tokens=True,
            max_length=self.blocksize,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
        )
        if self.use_token_cache:
            self._token_cache[smiles] = encoding
        return encoding

    def _build_model_inputs(self, data):
        if "input_ids" in data and "attention_mask" in data:
            return data["input_ids"], data["attention_mask"]

        smiles_batch = data[self.smiles_key]
        if isinstance(smiles_batch, str):
            smiles_batch = [smiles_batch]
        encodings = [self._encode_smiles(str(smiles)) for smiles in smiles_batch]
        input_ids = paddle.to_tensor(
            [encoding["input_ids"] for encoding in encodings], dtype="int64"
        )
        attention_mask = paddle.to_tensor(
            [encoding["attention_mask"] for encoding in encodings], dtype="int64"
        )
        return input_ids, attention_mask

    def forward(
        self,
        data,
        attention_mask=None,
        return_loss=True,
        return_prediction=True,
    ):
        if not isinstance(data, dict):
            return self._forward(data, attention_mask)

        assert (
            return_loss or return_prediction
        ), "At least one of return_loss or return_prediction must be True."
        input_ids, attention_mask = self._build_model_inputs(data)
        pred = self._forward(input_ids, attention_mask)

        loss_dict = {}
        if return_loss:
            label = self.normalize(data[self.property_name])
            loss_dict["loss"] = self.loss_fn(input=pred, label=label)

        pred_dict = {}
        if return_prediction:
            pred_dict[self.property_name] = self.unnormalize(pred)

        return {"loss_dict": loss_dict, "pred_dict": pred_dict}

    def predict(self, data):
        if isinstance(data, str):
            data = {self.smiles_key: [data]}
        elif isinstance(data, (list, tuple)):
            data = {self.smiles_key: list(data)}

        output = self.forward(data, return_loss=False, return_prediction=True)
        return output["pred_dict"]

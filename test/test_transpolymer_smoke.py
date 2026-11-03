import sys
from zipfile import ZipFile

import paddle
import pandas as pd
import pytest
import yaml
from sklearn.model_selection import KFold

from ppmat.datasets.transpolymer_dataset import TransPolymerDataset
from ppmat.models.transpolymer.modeling import RobertaModel
from ppmat.models.transpolymer.transpolymer import TransPolymerRegressor
from property_prediction.pack_transpolymer_registry import package_task
from property_prediction import predict as predict_module
from property_prediction.predict import predict_csv


def test_transpolymer_model_forward():
    model = RobertaModel(
        vocab_size=128,
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        intermediate_size=64,
        max_position_embeddings=32,
        type_vocab_size=1,
    )
    input_ids = paddle.randint(0, 128, shape=[2, 16], dtype="int64")
    attention_mask = paddle.ones([2, 16], dtype="int64")
    hidden_states = model(input_ids=input_ids, attention_mask=attention_mask)
    assert list(hidden_states.shape) == [2, 16, 32]


def test_transpolymer_regressor_forward():
    model = TransPolymerRegressor(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        max_position_embeddings=32,
        num_attention_heads=4,
        num_hidden_layers=1,
        resize_vocab_size=128,
    )
    input_ids = paddle.randint(0, 128, shape=[2, 16], dtype="int64")
    attention_mask = paddle.ones([2, 16], dtype="int64")
    pred = model(input_ids, attention_mask)
    assert list(pred.shape) == [2, 1]


def test_transpolymer_csv_prediction(tmp_path):
    class Predictor:
        def from_structures(self, smiles):
            assert smiles == ["[*]CC[*]", "[*]CO[*]"]
            return {"value": paddle.to_tensor([[0.1], [0.2]])}

    input_path = tmp_path / "input.csv"
    output_path = tmp_path / "output.csv"
    pd.DataFrame({"polymer": ["[*]CC[*]", "[*]CO[*]"]}).to_csv(input_path, index=False)

    predict_csv(Predictor(), str(input_path), "polymer", str(output_path))

    result = pd.read_csv(output_path)
    assert result["polymer"].tolist() == ["[*]CC[*]", "[*]CO[*]"]
    assert result["value"].tolist() == pytest.approx([0.1, 0.2])


def test_transpolymer_csv_requires_smiles_column(tmp_path):
    input_path = tmp_path / "input.csv"
    pd.DataFrame({"polymer": ["[*]CC[*]"]}).to_csv(input_path, index=False)

    with pytest.raises(ValueError, match="SMILES column 'smiles' is missing"):
        predict_csv(None, str(input_path), save_path=str(tmp_path / "output.csv"))


def test_predict_cli_routes_csv_to_transpolymer(tmp_path, monkeypatch):
    class Model:
        pass

    class Predictor:
        model = Model()

    input_path = str(tmp_path / "input.csv")
    output_path = str(tmp_path / "output.csv")
    calls = []
    monkeypatch.setattr(predict_module, "TransPolymerRegressor", Model)
    monkeypatch.setattr(predict_module, "PropertyPredictor", lambda **kwargs: Predictor())
    monkeypatch.setattr(
        predict_module,
        "predict_csv",
        lambda predictor, path, column, save_path: calls.append(
            (path, column, save_path)
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["predict.py", "--csv_file_path", input_path, "--save_path", output_path],
    )

    predict_module.main()

    assert calls == [(input_path, "smiles", output_path)]


def test_predict_cli_keeps_default_cif_input(monkeypatch):
    calls = []

    class Predictor:
        def from_cif_file(self, path, save_path):
            calls.append((path, save_path))

    monkeypatch.setattr(predict_module, "PropertyPredictor", lambda **kwargs: Predictor())
    monkeypatch.setattr(sys, "argv", ["predict.py"])

    predict_module.main()

    assert calls == [("./property_prediction/example_data/cifs/", "result.csv")]


def test_transpolymer_cv_matches_original_kfold():
    rows = {"id": list(range(12)), "smiles": [f"C{i}" for i in range(12)]}
    held_out = []

    for fold, (train_indices, val_indices) in enumerate(
        KFold(n_splits=5, shuffle=True, random_state=1).split(range(12)), start=1
    ):
        train_rows, _ = TransPolymerDataset.select_fold(rows, fold, "train")
        val_rows, _ = TransPolymerDataset.select_fold(rows, fold, "val")
        assert train_rows["id"] == train_indices.tolist()
        assert val_rows["id"] == val_indices.tolist()
        assert not set(train_rows["id"]) & set(val_rows["id"])
        held_out.extend(val_rows["id"])

    assert sorted(held_out) == rows["id"]


def test_transpolymer_cv_filters_after_split_and_separates_caches(tmp_path):
    path = tmp_path / "polymers.csv"
    pd.DataFrame(
        {
            "id": list(range(12)),
            "smiles": [f"C{i}" for i in range(12)],
            "value": [None] + list(range(1, 12)),
        }
    ).to_csv(path, index=False)

    train = TransPolymerDataset(
        path=str(path), property_names=["value"], cv_fold=1, cv_split="train"
    )
    val = TransPolymerDataset(
        path=str(path), property_names=["value"], cv_fold=1, cv_split="val"
    )
    raw_rows, _ = TransPolymerDataset.read_data(str(path))
    raw_train, _ = TransPolymerDataset.select_fold(raw_rows, 1, "train")
    raw_val, _ = TransPolymerDataset.select_fold(raw_rows, 1, "val")

    assert train.row_data["id"] == [i for i in raw_train["id"] if i != 0]
    assert val.row_data["id"] == [i for i in raw_val["id"] if i != 0]
    assert train.cache_path != val.cache_path
    assert train.sequences == train.row_data["smiles"]
    assert val.sequences == val.row_data["smiles"]


def test_transpolymer_registry_package(tmp_path):
    source = tmp_path / "source" / "transpolymer_eea"
    checkpoint_dir = source / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "best.pdparams").write_bytes(b"x" * (1024 * 1024 + 1))
    config_path = source / "transpolymer_eea_finetune.yaml"
    config_path.write_text(
        "Model:\n"
        "  __class_name__: TransPolymerRegressor\n"
        "  __init_params__:\n"
        "    pretrained_model_path: ./ckpt/pretrain.pt\n"
        "    pretrained_model_url: https://example.com/pretrain\n",
        encoding="utf-8",
    )

    package = package_task(tmp_path / "source", tmp_path / "packages", "eea")
    with ZipFile(package) as archive:
        assert archive.namelist() == [
            "transpolymer_eea/transpolymer_eea.yaml",
            "transpolymer_eea/checkpoints/best.pdparams",
        ]
        config = yaml.safe_load(archive.read("transpolymer_eea/transpolymer_eea.yaml"))
        assert config["Model"]["__init_params__"]["pretrained_model_path"] is None
        assert config["Model"]["__init_params__"]["pretrained_model_url"] is None


def test_transpolymer_registered_package_path(tmp_path, monkeypatch):
    import ppmat.models as models

    package_dir = tmp_path / "transpolymer_eea"
    package_dir.mkdir()
    (package_dir / "transpolymer_eea.yaml").write_text(
        "Model:\n  __class_name__: TransPolymerRegressor\n  __init_params__: {}\n",
        encoding="utf-8",
    )

    loaded = {}
    monkeypatch.setattr(
        models.download,
        "get_weights_path_from_url",
        lambda url: str(package_dir),
    )
    monkeypatch.setattr(models, "build_model", lambda config: object())
    monkeypatch.setattr(
        models.save_load,
        "load_pretrain",
        lambda model, path, weights_name: loaded.update(
            path=path, weights_name=weights_name
        ),
    )

    models.build_model_from_name("transpolymer_eea", "best.pdparams")

    assert loaded == {"path": str(package_dir), "weights_name": "best.pdparams"}

import zipfile
from pathlib import Path

import pytest

from ppmat.utils import download
from ppmat.utils.download import _uncompress_file_zip
from ppmat.utils.model_package import get_model_config_path
from ppmat.utils.model_package import resolve_model_package_dir


@pytest.mark.parametrize("directory_entry", [False, True])
@pytest.mark.parametrize("weights_first", [False, True])
def test_uncompress_single_directory_preserves_archive_root(
    tmp_path, directory_entry, weights_first
):
    archive_path = tmp_path / "model.zip"
    members = [
        ("model/checkpoints/best.pdparams", b"weights"),
        ("model/model.yaml", b"Model: {}"),
    ]
    if not weights_first:
        members.reverse()
    with zipfile.ZipFile(archive_path, "w") as archive:
        if directory_entry:
            archive.writestr("model/", b"")
        for name, content in members:
            archive.writestr(name, content)

    extracted_path = _uncompress_file_zip(str(archive_path))

    assert Path(extracted_path) == tmp_path / "model"
    package_dir = Path(resolve_model_package_dir("model", extracted_path))
    assert package_dir == tmp_path / "model" / "model"
    assert (package_dir / "model.yaml").is_file()
    assert (package_dir / "checkpoints" / "best.pdparams").is_file()


@pytest.mark.parametrize("rooted", [False, True])
@pytest.mark.parametrize("yaml_suffix", [".yaml", ".yml"])
def test_cached_weight_package_supports_flat_and_rooted_layouts(
    tmp_path, monkeypatch, rooted, yaml_suffix
):
    monkeypatch.setattr(download, "WEIGHTS_HOME", str(tmp_path))
    archive_path = tmp_path / "model.zip"
    prefix = "model/" if rooted else ""
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr(f"{prefix}checkpoints/best.pdparams", b"weights")
        archive.writestr(f"{prefix}model{yaml_suffix}", b"Model: {}")

    for _ in range(2):
        extracted_path = download.get_weights_path_from_url(
            "https://example.com/model.zip"
        )
        package_dir = resolve_model_package_dir("model", extracted_path)
        config_path = get_model_config_path("model", package_dir)
        assert Path(config_path).read_bytes() == b"Model: {}"
        assert (
            Path(package_dir) / "checkpoints" / "best.pdparams"
        ).read_bytes() == b"weights"


@pytest.mark.parametrize("directory_entry", [False, True])
def test_cached_dataset_archive_preserves_root_directory(
    tmp_path, monkeypatch, directory_entry
):
    monkeypatch.setattr(download, "DATASETS_HOME", str(tmp_path))
    archive_path = tmp_path / "dataset.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        if directory_entry:
            archive.writestr("dataset/", b"")
        archive.writestr("dataset/train/data.json", b"[]")
        archive.writestr("dataset/val/data.json", b"[]")

    extracted_path = download.get_datasets_path_from_url(
        "https://example.com/dataset.zip"
    )
    assert Path(extracted_path) == tmp_path / "dataset"
    assert (Path(extracted_path) / "dataset" / "train" / "data.json").is_file()
    assert (Path(extracted_path) / "dataset" / "val" / "data.json").is_file()

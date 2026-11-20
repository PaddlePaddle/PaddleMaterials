import importlib.util
import zipfile
from pathlib import Path

import pytest

from ppmat.utils import download
from ppmat.utils.model_package import get_model_config_path
from ppmat.utils.model_package import resolve_model_package_dir

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "electronic_structure/configs/gpwno/prepare_model_packages.py"
)
spec = importlib.util.spec_from_file_location("prepare_gpwno_packages", SCRIPT)
packages = importlib.util.module_from_spec(spec)
spec.loader.exec_module(packages)


@pytest.mark.parametrize("name", packages.MODEL_NAMES)
def test_repacked_gpwno_download_preserves_payload(tmp_path, monkeypatch, name):
    source = tmp_path / "source" / f"{name}.zip"
    source.parent.mkdir()
    payload = {
        f"{name}/checkpoints/best.pdparams": b"original checkpoint",
        f"{name}/{name}.yaml": b"Model: {}",
    }
    with zipfile.ZipFile(source, "w", zipfile.ZIP_DEFLATED) as archive:
        for member, content in payload.items():
            archive.writestr(member, content)
    original = source.read_bytes()
    destination = tmp_path / "weights" / source.name
    packages.prepare_package(source, destination)

    assert source.read_bytes() == original
    with zipfile.ZipFile(destination) as archive:
        assert archive.namelist()[0] == f"{name}/"
        assert archive.getinfo(f"{name}/").is_dir()
        assert {
            member.filename: archive.read(member)
            for member in archive.infolist()
            if not member.is_dir()
        } == payload

    monkeypatch.setattr(download, "WEIGHTS_HOME", str(destination.parent))
    for _ in range(2):
        extracted = download.get_weights_path_from_url(
            f"https://example.com/{name}.zip"
        )
        package = Path(resolve_model_package_dir(name, extracted))
        assert (
            Path(get_model_config_path(name, str(package))).read_bytes() == b"Model: {}"
        )
        assert (package / "checkpoints/best.pdparams").read_bytes() == (
            b"original checkpoint"
        )


def test_repacking_does_not_overwrite_existing_package(tmp_path):
    name = packages.MODEL_NAMES[0]
    source = tmp_path / f"{name}.zip"
    with zipfile.ZipFile(source, "w") as archive:
        archive.writestr(f"{name}/{name}.yaml", b"Model: {}")
        archive.writestr(f"{name}/checkpoints/best.pdparams", b"weights")
    original = source.read_bytes()
    with pytest.raises(FileExistsError):
        packages.prepare_package(source, source)
    assert source.read_bytes() == original


def test_repacking_rejects_incomplete_package(tmp_path):
    name = packages.MODEL_NAMES[0]
    source = tmp_path / f"{name}.zip"
    destination = tmp_path / "output" / source.name
    with zipfile.ZipFile(source, "w") as archive:
        archive.writestr(f"{name}/{name}.yaml", b"Model: {}")
    with pytest.raises(ValueError, match="Unexpected model package contents"):
        packages.prepare_package(source, destination)
    assert not destination.exists()

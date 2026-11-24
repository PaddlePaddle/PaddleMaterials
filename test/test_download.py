import io
import tarfile
import zipfile
from pathlib import Path

import pytest

from ppmat.utils.download import _uncompress_file_tar
from ppmat.utils.download import _uncompress_file_zip


@pytest.mark.parametrize("directory_entry", [True, False])
def test_uncompress_single_directory_preserves_archive_root(tmp_path, directory_entry):
    archive_path = tmp_path / "delivery.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        if directory_entry:
            archive.writestr("model/", b"")
        archive.writestr("model/checkpoints/best.pdparams", b"weights")
        archive.writestr("model/model.yaml", b"Model: {}")

    extracted_path = _uncompress_file_zip(str(archive_path))

    assert Path(extracted_path) == tmp_path / "delivery" / "model"
    assert (Path(extracted_path) / "model.yaml").read_bytes() == b"Model: {}"
    assert (
        Path(extracted_path) / "checkpoints/best.pdparams"
    ).read_bytes() == b"weights"


@pytest.mark.parametrize("directory_entry", [True, False])
def test_uncompress_dataset_root_without_explicit_directory(tmp_path, directory_entry):
    archive_path = tmp_path / "delivery.tar"
    with tarfile.open(archive_path, "w") as archive:
        if directory_entry:
            root = tarfile.TarInfo("graphene/")
            root.type = tarfile.DIRTYPE
            root.mode = 0o755
            archive.addfile(root)
        for name, data in (
            ("graphene/graphene_split.json", b"{}"),
            ("graphene/processed/1000/rc.npz", b"sample"),
        ):
            entry = tarfile.TarInfo(name)
            entry.size = len(data)
            archive.addfile(entry, io.BytesIO(data))

    extracted_path = Path(_uncompress_file_tar(str(archive_path)))

    assert extracted_path == tmp_path / "delivery" / "graphene"
    assert (extracted_path / "processed/1000/rc.npz").read_bytes() == b"sample"

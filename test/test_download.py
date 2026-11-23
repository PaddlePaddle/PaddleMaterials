import zipfile
from pathlib import Path

from ppmat.utils.download import _uncompress_file_zip


def test_uncompress_single_directory_preserves_archive_root(tmp_path):
    archive_path = tmp_path / "model.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("model/", b"")
        archive.writestr("model/checkpoints/best.pdparams", b"weights")
        archive.writestr("model/model.yaml", b"Model: {}")

    extracted_path = _uncompress_file_zip(str(archive_path))

    assert Path(extracted_path) == tmp_path / "model"
    assert (tmp_path / "model" / "model" / "model.yaml").exists()

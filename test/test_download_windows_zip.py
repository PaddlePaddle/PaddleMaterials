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

"""Archive-layout regression tests for ``ppmat.utils.download``.

Windows-made zip files store ``checkpoints\\best.pdparams`` as a single file
name.  ``_expand_windows_separators`` relocates such entries, and these tests
pin down both the happy path and the malicious cases (path traversal, absolute
paths, drive letters, silent overwrite) that must be refused.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from ppmat.utils import download


def _write_zip(path: Path, entries) -> Path:
    with zipfile.ZipFile(path, "w") as archive:
        for name, data in entries:
            archive.writestr(name, data)
    return path


def test_windows_separators_are_expanded(tmp_path):
    archive = _write_zip(
        tmp_path / "pkg.zip",
        [
            ("liflow_universal.yaml", b"yaml"),
            ("checkpoints\\best.pdparams", b"weights"),
        ],
    )

    root = Path(download._uncompress_file_zip(str(archive)))

    assert (root / "checkpoints" / "best.pdparams").read_bytes() == b"weights"
    assert not (root / "checkpoints\\best.pdparams").exists()


def test_parent_component_is_rejected(tmp_path):
    archive = _write_zip(
        tmp_path / "pkg.zip",
        [
            ("liflow_universal.yaml", b"yaml"),
            ("..\\evil.pdparams", b"payload"),
        ],
    )

    with pytest.raises(RuntimeError, match="invalid path component"):
        download._uncompress_file_zip(str(archive))
    assert not (tmp_path / "evil.pdparams").exists()
    assert not any(
        p.name == "evil.pdparams" for p in tmp_path.rglob("*") if p.is_file()
    )


@pytest.mark.parametrize("entry", ["\\etc\\evil.pdparams", "C:\\evil.pdparams"])
def test_absolute_and_drive_paths_are_rejected(tmp_path, entry):
    archive = _write_zip(
        tmp_path / "pkg.zip",
        [("liflow_universal.yaml", b"yaml"), (entry, b"payload")],
    )

    with pytest.raises(RuntimeError):
        download._uncompress_file_zip(str(archive))


def test_existing_target_is_not_overwritten(tmp_path):
    archive = _write_zip(
        tmp_path / "pkg.zip",
        [
            ("checkpoints/best.pdparams", b"nested"),
            ("checkpoints\\best.pdparams", b"flat"),
        ],
    )

    with pytest.raises(RuntimeError, match="refusing to overwrite"):
        download._uncompress_file_zip(str(archive))


def test_standard_slash_zip_is_untouched(tmp_path):
    archive = _write_zip(
        tmp_path / "pkg.zip",
        [
            ("liflow_universal.yaml", b"yaml"),
            ("checkpoints/best.pdparams", b"weights"),
        ],
    )

    root = Path(download._uncompress_file_zip(str(archive)))

    assert (root / "checkpoints" / "best.pdparams").read_bytes() == b"weights"


@pytest.mark.parametrize("as_directory", [True, False])
def test_symlinked_entries_are_rejected(tmp_path, as_directory):
    root = tmp_path / "root"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    link = root / "checkpoints\\best.pdparams"
    if as_directory:
        link.symlink_to(outside, target_is_directory=True)
    else:
        (outside / "payload").write_bytes(b"payload")
        link.symlink_to(outside / "payload")

    with pytest.raises(RuntimeError, match="symlinked"):
        download._expand_windows_separators(str(root))
    assert not (root / "checkpoints").exists()

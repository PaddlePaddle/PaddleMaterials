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

"""Prepare GPWNO ZIPs for the unchanged PaddleMaterials download utility."""

import argparse
import zipfile
from pathlib import Path

MODEL_NAMES = (
    "gpwno_md17_benzene",
    "gpwno_md17_ethane",
    "gpwno_qm9",
    "gpwno_mp",
)


def prepare_package(source: Path, destination: Path) -> None:
    """Add an explicit root entry first, preserving every payload file's bytes."""
    name = source.stem
    if name not in MODEL_NAMES or destination.name != source.name:
        raise ValueError("Source and destination must use the same GPWNO model name")
    required = {f"{name}/{name}.yaml", f"{name}/checkpoints/best.pdparams"}
    with zipfile.ZipFile(source) as archive:
        members = archive.infolist()
        payload = [member for member in members if not member.is_dir()]
        if len(payload) != len(required) or {m.filename for m in payload} != required:
            raise ValueError(f"Unexpected model package contents: {source}")
        # Read and CRC-check both files before creating the destination.
        contents = [(member, archive.read(member)) for member in payload]
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "x") as archive:
        archive.writestr(f"{name}/", b"")
        for member, content in contents:
            archive.writestr(member, content)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    for name in MODEL_NAMES:
        filename = f"{name}.zip"
        prepare_package(args.source_dir / filename, args.output_dir / filename)
        print(args.output_dir / filename)


if __name__ == "__main__":
    main()

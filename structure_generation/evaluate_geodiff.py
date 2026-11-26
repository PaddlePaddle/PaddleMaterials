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

import argparse
import json
import zipfile
from pathlib import Path

from ppmat.metrics.geodiff_conformers import read_covmat_inputs
from ppmat.metrics.geodiff_covmat import compute_covmat


def main():
    parser = argparse.ArgumentParser(description="Evaluate GeoDiff conformer ensembles")
    parser.add_argument("metric", choices=["covmat"])
    parser.add_argument(
        "--reference", type=Path, required=True, help="GEOM reference NPZ"
    )
    parser.add_argument(
        "--generated",
        type=Path,
        required=True,
        help="GeoDiff SDF records with source tags",
    )
    parser.add_argument(
        "--ratio",
        type=int,
        default=2,
        help="Generated conformers per reference record (default: 2)",
    )
    parser.add_argument(
        "--output", type=Path, help="New JSON result file (default: stdout)"
    )
    args = parser.parse_args()
    if args.ratio < 1:
        parser.error("--ratio must be a positive integer")
    if args.output is not None and (
        args.output.exists() or not args.output.parent.is_dir()
    ):
        parser.error("--output must be a new file in an existing directory")
    try:
        groups, stereo = read_covmat_inputs(args.reference, args.generated)
        result = compute_covmat(groups, ratio=args.ratio)
        result["stereochemistry"] = stereo
        if args.output is None:
            print(json.dumps(result, allow_nan=False))
        else:
            with args.output.open("x", encoding="utf-8") as stream:
                json.dump(result, stream, allow_nan=False)
                stream.write("\n")
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        parser.exit(2, f"{parser.prog}: {exc}\n")


if __name__ == "__main__":
    main()

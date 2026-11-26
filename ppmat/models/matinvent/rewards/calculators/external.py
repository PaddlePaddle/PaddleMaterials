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

import hashlib
import json
import os

import numpy as np
from pymatgen.io.cif import CifWriter


class ExternalResults:
    """Export a batch for an external property calculation and read its results.

    The batch identity includes the task and ordered structures. Results belong
    in ``results.txt``, with one float per structure and NaN for failed entries.
    """

    def __init__(self, root_dir: str, task: str):
        self.root_dir = root_dir
        self.task = task

    def calc(self, structures, label="tmp"):
        payload = json.dumps(
            {
                "task": self.task,
                "structures": [structure.as_dict() for structure in structures],
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        fingerprint = hashlib.sha256(payload).hexdigest()
        directory = os.path.abspath(os.path.join(self.root_dir, label, fingerprint))
        out_path = os.path.join(directory, "results.txt")
        if not os.path.isfile(out_path):
            os.makedirs(directory, exist_ok=True)
            for index, structure in enumerate(structures):
                CifWriter(structure).write_file(os.path.join(directory, f"{index}.cif"))
            raise FileNotFoundError(
                f"External property results are missing: {out_path}. "
                f"Run the {self.task} calculation on the exported CIF files and "
                "write one value per structure before rerunning."
            )
        results = np.loadtxt(out_path, dtype=float, ndmin=1)
        if results.shape != (len(structures),):
            raise ValueError(
                f"{out_path} has shape {results.shape}, expected ({len(structures)},)"
            )
        return results

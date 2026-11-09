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


class Calculator:
    """Base class of the property calculators consumed by ``Reward``.

    Subclasses return one float per structure and use NaN to mark failures.
    """

    def __init__(self, root_dir: str, task: str):
        self.root_dir = root_dir
        self.task = task
        os.makedirs(root_dir, exist_ok=True)

    def calc(self, samples, label="tmp"):
        raise NotImplementedError

    def external_result_dir(self, structures, label):
        payload = json.dumps(
            [structure.as_dict() for structure in structures],
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        fingerprint = hashlib.sha256(payload).hexdigest()
        directory = os.path.join(self.root_dir, label, fingerprint)
        os.makedirs(directory, exist_ok=True)
        return directory

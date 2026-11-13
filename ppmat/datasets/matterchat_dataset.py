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

"""Local frozen-CHGNet features paired with crystal question-answer records."""

import numpy as np
from paddle.io import Dataset

from ppmat.datasets.custom_data_type import ConcatData


class MatterChatDataset(Dataset):
    """Read packed per-atom embeddings and text from an explicit NPZ file."""

    def __init__(self, path, embedding_dim=64):
        super().__init__()
        if not isinstance(embedding_dim, int) or embedding_dim < 1:
            raise ValueError("embedding_dim must be a positive integer.")
        with np.load(path, allow_pickle=False) as data:
            self.embeddings = data["material_embedding"].astype("float32")
            self.num_atoms = data["num_atoms"]
            self.questions = data["text_input"]
            self.answers = data["text"]
        if self.num_atoms.ndim != 1 or self.num_atoms.dtype.kind not in "iu":
            raise ValueError("num_atoms must be a one-dimensional integer array.")
        if len(self.num_atoms) == 0 or np.any(self.num_atoms <= 0):
            raise ValueError("Every record must contain at least one atom.")
        if self.embeddings.shape != (int(self.num_atoms.sum()), embedding_dim):
            raise ValueError(
                "material_embedding must have shape [sum(num_atoms), embedding_dim]."
            )
        if not np.isfinite(self.embeddings).all():
            raise ValueError("material_embedding must contain finite values.")
        for name, values in (("text_input", self.questions), ("text", self.answers)):
            if values.shape != self.num_atoms.shape or values.dtype.kind != "U":
                raise ValueError(
                    f"{name} must be a Unicode string array matching num_atoms."
                )
        if any(not answer.strip() for answer in self.answers):
            raise ValueError("Every record must contain a nonempty answer.")
        self.offsets = np.concatenate(([0], np.cumsum(self.num_atoms)))

    def __len__(self):
        return len(self.num_atoms)

    def __getitem__(self, index):
        start, end = self.offsets[index : index + 2]
        return {
            "num_atoms": np.int64(self.num_atoms[index]),
            "material_embedding": ConcatData(self.embeddings[start:end]),
            "text_input": str(self.questions[index]),
            "text": str(self.answers[index]),
        }

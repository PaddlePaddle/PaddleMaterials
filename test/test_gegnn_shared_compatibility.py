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

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from rdkit import Chem

from ppmat.models.common.graph_converter import MolecularGraphConverter
from ppmat.vocab import _build_vocab_from_name
from ppmat.vocab import build_vocab


def raw_vocab():
    return {
        "atom": {"type": "categorical", "tokens": ["C", "O"], "num_embeddings": 2},
        "bond": {
            "type": "categorical",
            "tokens": ["SINGLE", "DOUBLE", "TRIPLE", "AROMATIC", "NO_BOND"],
            "num_embeddings": 5,
        },
    }


class TestSharedGEGNNCompatibility(unittest.TestCase):
    def test_registered_and_inline_vocabulary_match_without_mutation(self):
        raw = raw_vocab()
        original = copy.deepcopy(raw)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "vocab.json"
            path.write_text(json.dumps(raw))
            _build_vocab_from_name.cache_clear()
            try:
                with patch(
                    "ppmat.vocab.download.get_datasets_path_from_url",
                    return_value=str(path),
                ):
                    registered = build_vocab("infgcn_md17")
                inline = build_vocab(raw)
                self.assertEqual(registered, inline)
                self.assertEqual(build_vocab(inline), inline)
                self.assertEqual(raw, original)
                self.assertIsNone(build_vocab(None))
            finally:
                _build_vocab_from_name.cache_clear()

    def test_default_atom_only_features_and_optional_key(self):
        vocab = build_vocab(raw_vocab())
        molecule = Chem.MolFromSmiles("CO")
        default = MolecularGraphConverter(vocab=vocab, num_cpus=1)(molecule)
        renamed = MolecularGraphConverter(
            vocab=vocab, node_feature_key="h", num_cpus=1
        )(molecule)
        self.assertEqual(set(default.node_feat), {"feat"})
        self.assertEqual(set(renamed.node_feat), {"h"})
        np.testing.assert_array_equal(
            default.node_feat["feat"], np.eye(2, dtype="float32")
        )
        np.testing.assert_array_equal(default.node_feat["feat"], renamed.node_feat["h"])
        np.testing.assert_array_equal(default.edges, renamed.edges)
        np.testing.assert_array_equal(
            default.edge_feat["feat"], renamed.edge_feat["feat"]
        )

    def test_unknown_atom_keeps_existing_rejection_behavior(self):
        converter = MolecularGraphConverter(vocab=build_vocab(raw_vocab()), num_cpus=1)
        self.assertIsNone(converter(Chem.MolFromSmiles("N")))

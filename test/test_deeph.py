# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

import os
import tempfile
import unittest

import numpy as np
import paddle
from ase import Atoms
from omegaconf import OmegaConf
from pymatgen.core.structure import Structure

from ppmat.datasets.collate_fn import DefaultCollator
from ppmat.datasets.deeph_dataset import DeepHData
from ppmat.datasets.deeph_dataset import DeepHDataset
from ppmat.models.deeph import DeepHGraphConverter
from ppmat.models.deeph import DeepHHamiltonian
from ppmat.predictor.deeph import DeepHPredictor
from ppmat.utils.io import build_deeph_hamiltonian_blocks
from ppmat.utils.io import save_deeph_hamiltonian


class TestDeepH(unittest.TestCase):
    def test_deeph_config_can_be_loaded(self):
        config_path = os.path.join(
            "electronic_structure",
            "configs",
            "deeph",
            "deeph_graphene.yaml",
        )
        config = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)

        self.assertEqual(config["Model"]["__class_name__"], "DeepHHamiltonian")
        self.assertEqual(
            config["Dataset"]["train"]["dataset"]["__class_name__"],
            "DeepHDataset",
        )
        init_params = config["Dataset"]["train"]["dataset"]["__init_params__"]
        self.assertIn("config", init_params)
        self.assertNotIn("config_files", init_params)
        self.assertEqual(len(init_params["config"]["basic"]["orbital"]), 169)

    def test_deeph_forward(self):
        paddle.seed(42)
        model = DeepHHamiltonian(
            num_species=1,
            in_atom_fea_len=16,
            in_edge_fea_len=32,
            num_orbital=9,
            num_l=3,
            gauss_stop=6.0,
            if_exp=True,
            normalization="LayerNorm",
            target_name="label",
        )

        num_nodes = 3
        num_edges = 4
        num_sub_edges = 8
        batch = {
            "x": paddle.zeros([num_nodes], dtype="int64"),
            "edge_index": paddle.to_tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype="int64"),
            "edge_attr": paddle.ones([num_edges, 1], dtype="float32"),
            "batch": paddle.zeros([num_nodes], dtype="int64"),
            "sub_atom_idx": paddle.to_tensor(
                [
                    [0, 1],
                    [1, 0],
                    [1, 2],
                    [2, 1],
                    [0, 1],
                    [1, 0],
                    [1, 2],
                    [2, 1],
                ],
                dtype="int64",
            ),
            "sub_edge_idx": paddle.to_tensor([0, 1, 2, 3, 0, 1, 2, 3], dtype="int64"),
            "sub_edge_ang": paddle.zeros([num_sub_edges, 9], dtype="float32"),
            "sub_index": paddle.to_tensor([0, 1, 2, 3, 4, 5, 6, 7], dtype="int64"),
            "label": paddle.zeros([num_edges, 9], dtype="float32"),
            "mask": paddle.ones([num_edges, 9], dtype="bool"),
        }

        output = model(batch)

        self.assertIn("loss", output["loss_dict"])
        self.assertIn("label", output["pred_dict"])
        self.assertEqual(list(output["pred_dict"]["label"].shape), [num_edges, 9])

    def test_deeph_data_uses_default_collator(self):
        def sample(num_nodes, edge_index, sub_atom_idx, sub_edge_idx, sub_index):
            num_edges = edge_index.shape[1]
            return DeepHData(
                num_nodes=num_nodes,
                x=np.zeros([num_nodes], dtype=np.int64),
                edge_index=edge_index,
                edge_attr=np.ones([num_edges, 1], dtype=np.float32),
                label=np.zeros([num_edges, 9], dtype=np.float32),
                mask=np.ones([num_edges, 9], dtype=bool),
                sub_atom_idx=sub_atom_idx,
                sub_edge_idx=sub_edge_idx,
                sub_edge_ang=np.zeros([len(sub_edge_idx), 9], dtype=np.float32),
                sub_index=sub_index,
            )

        first = sample(
            2,
            np.asarray([[0, 1], [1, 0]], dtype=np.int64),
            np.asarray([[0, 1], [1, 0]], dtype=np.int64),
            np.asarray([0, 1], dtype=np.int64),
            np.asarray([0, 1], dtype=np.int64),
        )
        second = sample(
            3,
            np.asarray([[0, 1, 2], [1, 2, 0]], dtype=np.int64),
            np.asarray([[0, 1], [1, 2], [2, 0]], dtype=np.int64),
            np.asarray([0, 1, 2], dtype=np.int64),
            np.asarray([0, 2, 5], dtype=np.int64),
        )

        collated = DefaultCollator()([first, second])

        self.assertEqual(collated["batch"].tolist(), [0, 0, 1, 1, 1])
        self.assertEqual(
            collated["edge_index"].tolist(), [[0, 1, 2, 3, 4], [1, 0, 3, 4, 2]]
        )
        self.assertEqual(
            collated["sub_atom_idx"].tolist(),
            [[0, 1], [1, 0], [2, 3], [3, 4], [4, 2]],
        )
        self.assertEqual(collated["sub_edge_idx"].tolist(), [0, 1, 2, 3, 4])
        self.assertEqual(collated["sub_index"].tolist(), [0, 1, 4, 6, 9])

        model = DeepHHamiltonian(
            num_species=1,
            in_atom_fea_len=16,
            in_edge_fea_len=32,
            num_orbital=9,
            num_l=3,
        )
        output = model(collated)
        self.assertEqual(output["pred_dict"]["label"].shape, [5, 9])

    def test_deeph_prediction_can_export_hamiltonian_npz(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "orbital_types.dat"), "w") as f:
                f.write("0 1\n")
                f.write("0 1\n")

            batch = {
                "edge_index": np.asarray([[0], [1]], dtype=np.int64),
                "edge_attr": np.asarray(
                    [[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0]],
                    dtype=np.float32,
                ),
                "structure_lattice": np.eye(3, dtype=np.float32).reshape(1, 3, 3),
                "structure_atomic_numbers": np.asarray([6, 6], dtype=np.int64),
                "structure_folder": tmpdir,
            }
            pred = np.asarray([[0.25]], dtype=np.float32)
            orbital = [{"6 6": [0, 1]}]
            hamiltonian = build_deeph_hamiltonian_blocks(pred, batch, orbital)

            self.assertEqual(list(hamiltonian.keys()), ["[0, 0, 0, 1, 2]"])
            self.assertEqual(hamiltonian["[0, 0, 0, 1, 2]"].shape, (4, 4))
            self.assertAlmostEqual(
                float(hamiltonian["[0, 0, 0, 1, 2]"][0, 1]),
                0.25,
                places=6,
            )

            output_path = os.path.join(tmpdir, "rh_pred.npz")
            save_deeph_hamiltonian(output_path, pred, batch, orbital)
            loaded = np.load(output_path)
            self.assertIn("[0, 0, 0, 1, 2]", loaded.files)
            self.assertAlmostEqual(
                float(loaded["[0, 0, 0, 1, 2]"][0, 1]),
                0.25,
                places=6,
            )

    def test_deeph_graph_converter_accepts_ase_atoms(self):
        atoms = Atoms(
            symbols=["C", "C"],
            positions=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            cell=np.eye(3) * 4.0,
            pbc=True,
        )
        structure = DeepHGraphConverter._to_structure(atoms)

        self.assertIsInstance(structure, Structure)
        self.assertEqual(len(structure), 2)

    def test_deeph_graph_converter_returns_periodic_image_ranges(self):
        atoms = Atoms(
            symbols=["C"],
            positions=[[0.0, 0.0, 0.0]],
            cell=np.eye(3) * 2.0,
            pbc=True,
        )
        converter = DeepHGraphConverter(
            radius=2.1,
            max_num_nbr=0,
            default_dtype=np.float32,
            interface="npz",
            num_l=2,
        )

        images = converter(atoms, None, only_periodic_images=True)

        self.assertIn((0, 0, 0), images)
        self.assertIn((1, 0, 0), images)
        self.assertIn((-1, 0, 0), images)

    def test_deeph_graph_converter_builds_dft_graph(self):
        atoms = Atoms(
            symbols=["C", "C"],
            positions=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            cell=np.eye(3) * 4.0,
            pbc=True,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "orbital_types.dat"), "w") as file:
                file.write("0\n0\n")
            matrix_keys = {
                "[0, 0, 0, 1, 2]": np.eye(3, dtype=np.float32),
                "[0, 0, 0, 2, 1]": np.eye(3, dtype=np.float32),
            }
            np.savez(os.path.join(tmpdir, "rc.npz"), **matrix_keys)
            np.savez(
                os.path.join(tmpdir, "rh.npz"),
                **{
                    key: np.asarray([[index]], dtype=np.float32)
                    for index, key in enumerate(matrix_keys, start=1)
                },
            )
            converter = DeepHGraphConverter(
                radius=6.0,
                max_num_nbr=0,
                default_dtype=np.float32,
                interface="npz",
                num_l=2,
                create_from_DFT=True,
            )

            graph = converter(atoms, tmpdir)

        self.assertEqual(graph.edge_index.tolist(), [[0, 1], [1, 0]])
        self.assertEqual(graph.term_real.shape, (2, 1, 1))
        self.assertEqual(graph.subgraph_dict["subgraph_edge_ang"].shape, (4, 4))

    def test_deeph_dataset_builds_and_reuses_cache(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sample_dir = os.path.join(tmpdir, "raw", "sample")
            os.makedirs(sample_dir)
            raw_lattice = np.asarray(
                [[4.0, 0.2, 0.0], [0.0, 5.0, 0.3], [0.0, 0.0, 6.0]]
            )
            raw_positions = np.asarray([[0.0, 1.0], [0.0, 0.2], [0.0, 0.3]])
            np.savetxt(os.path.join(sample_dir, "lat.dat"), raw_lattice)
            np.savetxt(
                os.path.join(sample_dir, "site_positions.dat"),
                raw_positions,
            )
            np.savetxt(os.path.join(sample_dir, "element.dat"), [6, 6], fmt="%d")
            with open(os.path.join(sample_dir, "orbital_types.dat"), "w") as file:
                file.write("0\n0\n")
            rotations = {
                "[0, 0, 0, 1, 2]": np.eye(3, dtype=np.float32),
                "[0, 0, 0, 2, 1]": np.eye(3, dtype=np.float32),
            }
            np.savez(os.path.join(sample_dir, "rc.npz"), **rotations)
            np.savez(
                os.path.join(sample_dir, "rh.npz"),
                **{
                    key: np.asarray([[index]], dtype=np.float32)
                    for index, key in enumerate(rotations, start=1)
                },
            )
            config = {
                "basic": {
                    "raw_dir": os.path.join(tmpdir, "raw"),
                    "graph_dir": os.path.join(tmpdir, "cache"),
                    "dataset_name": "test",
                    "interface": "npz",
                    "target": "hamiltonian",
                    "orbital": [{"6 6": [0, 0]}],
                    "seed": 0,
                },
                "graph": {
                    "radius": 6.0,
                    "max_num_nbr": 0,
                    "create_from_dft": True,
                    "if_lcmp_graph": True,
                },
                "train": {"train_ratio": 1.0, "val_ratio": 0.0, "test_ratio": 0.0},
                "hyperparameter": {"dtype": "float32"},
                "network": {"num_l": 2},
            }

            dataset = DeepHDataset(split="train", config=config)
            cached_dataset = DeepHDataset(split="train", config=config)

            self.assertEqual(len(dataset), 1)
            self.assertEqual(dataset[0]["label"].shape, (2, 1))
            np.testing.assert_allclose(
                dataset[0]["structure_lattice"], raw_lattice.T[None]
            )
            np.testing.assert_allclose(dataset[0]["pos"], raw_positions.T)
            self.assertEqual(len(cached_dataset), 1)

    def test_deeph_predictor_accepts_dft_interface(self):
        predictor = DeepHPredictor()
        model_config = {"__class_name__": "DeepHHamiltonian"}

        self.assertIs(
            predictor.modify_model_config("deeph_dft", model_config), model_config
        )


if __name__ == "__main__":
    unittest.main()

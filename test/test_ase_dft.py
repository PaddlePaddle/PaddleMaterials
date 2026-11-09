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

import json
import os
import tempfile
import unittest
from types import SimpleNamespace

import h5py
import numpy as np
from ase import Atoms

from ppmat.calculator.deeph import DeepHDFTCalculator
from ppmat.calculator.deeph import preprocess_openmx_overlap
from ppmat.models.deeph import DeepHHamiltonian
from ppmat.predictor.deeph import DeepHPredictor


class TestASEDFT(unittest.TestCase):
    @staticmethod
    def _write_openmx_overlap_fixture(raw_dir):
        os.makedirs(os.path.join(raw_dir, "output"), exist_ok=True)
        with open(os.path.join(raw_dir, "openmx.out"), "w") as file:
            file.write(
                """<Definition.of.Atomic.Species
C C6.0-s1p1 C_PBE
Definition.of.Atomic.Species>
Atoms.UnitVectors.Unit Ang
<Atoms.UnitVectors
4.0 0.0 0.0
0.0 4.0 0.0
0.0 0.0 4.0
Atoms.UnitVectors>
Fractional coordinates of the final structure
index element x y z
---
---
1 C 0.0 0.0 0.0
2 C 0.25 0.0 0.0
3 C 0.0 0.25 0.0

"""
            )
        overlap_path = os.path.join(raw_dir, "output", "overlaps_0.h5")
        with h5py.File(overlap_path, "w") as overlap_file:
            for atom_i in range(1, 4):
                for atom_j in range(1, 4):
                    key = json.dumps([0, 0, 0, atom_i, atom_j])
                    overlap_file[key] = np.eye(4, dtype=np.float64)

    def test_ase_calculator_schedules_backend(self):
        atoms = Atoms(
            symbols=["Cu"],
            positions=[[0.0, 0.0, 0.0]],
            cell=np.eye(3) * 4.0,
            pbc=True,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            calculator = DeepHDFTCalculator(
                predictor=SimpleNamespace(),
                calculator_cls="ase.calculators.emt.EMT",
            )
            calculator._run_dft(atoms, tmpdir)
            self.assertTrue(os.path.exists(os.path.join(tmpdir, "structure.xyz")))

    def test_openmx_overlap_to_deeph_inference(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            raw_dir = os.path.join(tmpdir, "raw")
            processed_dir = os.path.join(tmpdir, "processed")
            self._write_openmx_overlap_fixture(raw_dir)

            preprocess_openmx_overlap(raw_dir, processed_dir)
            predictor = DeepHPredictor()
            batch = predictor.build_inference_batch(
                processed_dir,
                species=[6],
                num_l=2,
            )
            self.assertEqual(batch["edge_index"].shape, (2, 9))
            self.assertEqual(batch["sub_edge_ang"].shape[-1], 4)

            orbital = [
                {"6 6": [orbital_i, orbital_j]}
                for orbital_i in range(4)
                for orbital_j in range(4)
            ]

            model = DeepHHamiltonian(
                num_species=1,
                in_atom_fea_len=16,
                in_edge_fea_len=32,
                num_orbital=16,
                num_l=2,
                gauss_stop=6.0,
                if_exp=True,
                normalization="LayerNorm",
                target_name="label",
            )
            predictor.model = model
            predictor.eval_with_no_grad = True
            predictor.post_process = lambda data: data
            calculator = DeepHDFTCalculator(predictor=predictor)
            calculator.run_deeph_inference(
                structures=[Atoms("C3")],
                orbital=orbital,
                species=[6],
                num_l=2,
                output_dir=os.path.join(tmpdir, "result"),
                dtype="float32",
                run_dft=False,
                processed_dirs=[processed_dir],
            )

            output_path = os.path.join(tmpdir, "result", "000000", "rh_pred.npz")
            with np.load(output_path) as prediction:
                self.assertEqual(len(prediction.files), 9)
                self.assertEqual(prediction[prediction.files[0]].shape, (4, 4))


if __name__ == "__main__":
    unittest.main()

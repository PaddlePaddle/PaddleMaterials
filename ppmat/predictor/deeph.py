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

from __future__ import annotations

import json
from configparser import ConfigParser

import numpy as np

from ppmat.models import build_graph_converter
from ppmat.predictor.base import BasePredictor
from ppmat.utils.io import load_deeph_structure
from ppmat.utils.io import save_deeph_hamiltonian


class DeepHPredictor(BasePredictor):
    """Predict DeepH Hamiltonian blocks and export them in DeepH npz format."""

    def modify_model_config(self, interface_type, model_config):
        """Keep the DeepH model contract for the ppmatSim DFT interface."""
        if interface_type == "deeph_dft":
            return model_config
        return super().modify_model_config(interface_type, model_config)

    @staticmethod
    def load_orbital_config(config_path):
        config = ConfigParser()
        if not config.read(config_path):
            raise FileNotFoundError(f"Can not read DeepH config: {config_path}")
        return json.loads(config.get("basic", "orbital"))

    def build_inference_batch(
        self,
        processed_dir,
        species,
        num_l,
        dtype="float32",
    ):
        """Build a label-free DeepH batch from overlap-derived local coordinates."""
        structure = load_deeph_structure(processed_dir)
        np_dtype = np.dtype(dtype).type
        graph_converter = build_graph_converter(
            {
                "__class_name__": "DeepHGraphConverter",
                "__init_params__": {
                    "radius": -1.0,
                    "max_num_nbr": 0,
                    "default_dtype": np_dtype,
                    "interface": "npz_rc_only",
                    "num_l": num_l,
                    "create_from_DFT": True,
                    "if_lcmp_graph": True,
                    "separate_onsite": False,
                    "target": "hamiltonian",
                },
            }
        )
        graph = graph_converter(structure, processed_dir)
        model_input = graph_converter.to_model_input(
            graph, structure, processed_dir, species=species
        )
        model_input["batch"] = np.zeros(model_input["x"].shape[0], dtype=np.int64)
        return model_input

    def predict_hamiltonian(
        self,
        processed_dir,
        output_path,
        orbital,
        species,
        num_l,
        dtype="float32",
    ):
        """Run Paddle DeepH inference and save predicted Hamiltonian blocks."""
        batch = self.build_inference_batch(
            processed_dir=processed_dir,
            species=species,
            num_l=num_l,
            dtype=dtype,
        )
        prediction = self._run_model(batch)
        pred = prediction[self.model.target_name]
        save_deeph_hamiltonian(
            output_path,
            pred,
            batch,
            orbital,
            spinful=False,
        )
        return output_path

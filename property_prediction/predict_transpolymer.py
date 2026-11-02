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

import argparse

import pandas as pd

from ppmat.models.transpolymer.transpolymer import TransPolymerRegressor
from ppmat.utils import logger


def predict_csv(predictor, csv_file_path, smiles_column="smiles", save_path="result.csv"):
    if not save_path.endswith(".csv"):
        raise ValueError("save_path must end with .csv")

    data = pd.read_csv(csv_file_path)
    if smiles_column not in data.columns:
        raise ValueError(
            f"SMILES column '{smiles_column}' is missing from {csv_file_path}"
        )
    if data.empty or data[smiles_column].isna().any():
        raise ValueError("Input CSV must contain non-empty SMILES values")

    predictions = predictor.from_structures(data[smiles_column].tolist())
    for name, values in predictions.items():
        if hasattr(values, "numpy"):
            values = values.numpy()
        if hasattr(values, "reshape"):
            values = values.reshape([-1]).tolist()
        if len(values) != len(data):
            raise ValueError(f"Prediction count for '{name}' does not match input rows")
        data[name] = values

    data.to_csv(save_path, index=False)
    logger.info(f"Saved the prediction result to {save_path}")
    return predictions


def main():
    from property_prediction.predict import PropertyPredictor

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default=None)
    parser.add_argument("--weights_name", default=None)
    parser.add_argument("--config_path", default=None)
    parser.add_argument("--checkpoint_path", default=None)
    parser.add_argument("--csv_file_path", required=True)
    parser.add_argument("--smiles_column", default="smiles")
    parser.add_argument("--save_path", default="result.csv")
    args = parser.parse_args()

    predictor = PropertyPredictor(
        model_name=args.model_name,
        weights_name=args.weights_name,
        config_path=args.config_path,
        checkpoint_path=args.checkpoint_path,
    )
    if not isinstance(predictor.model, TransPolymerRegressor):
        raise ValueError("This entry point requires a TransPolymerRegressor model")
    predict_csv(predictor, args.csv_file_path, args.smiles_column, args.save_path)


if __name__ == "__main__":
    main()

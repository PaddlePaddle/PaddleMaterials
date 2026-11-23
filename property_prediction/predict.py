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
import os
import os.path as osp

from ppmat.predictor import PropertyPredictor


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Material property prediction")
    # Select exactly one model source: a registered package or a local config.
    model_source = parser.add_mutually_exclusive_group(required=True)
    model_source.add_argument("--model_name", help="Registered model name.")
    model_source.add_argument("--config_path", help="Path to a local config file.")
    parser.add_argument("--weights_name", help="Weight filename in a model package.")
    parser.add_argument(
        "--checkpoint_path",
        help="Checkpoint path or URL; defaults to Predict.checkpoint_path in config.",
    )
    parser.add_argument("--device", help="Paddle device, for example cpu or gpu:0.")
    parser.add_argument("--input_path", required=True, help="Input file or directory.")
    parser.add_argument(
        "--input_format",
        required=True,
        choices=["cif", "xyz"],
        help="Input file format.",
    )
    parser.add_argument(
        "--output_path",
        default="results",
        help="Directory used to save result.csv.",
    )
    args, config_overrides = parser.parse_known_args(argv)
    if args.model_name is not None and args.checkpoint_path is not None:
        parser.error("--checkpoint_path cannot be combined with --model_name")
    if args.config_path is not None and args.weights_name is not None:
        parser.error("--weights_name can only be used with --model_name")
    if any(value.startswith("-") or "=" not in value for value in config_overrides):
        parser.error("unrecognized arguments: " + " ".join(config_overrides))
    return args, config_overrides


def main():
    args, config_overrides = parse_args()
    predictor = PropertyPredictor(
        model_name=args.model_name,
        weights_name=args.weights_name,
        config_path=args.config_path,
        checkpoint_path=args.checkpoint_path,
        device=args.device,
        config_overrides=config_overrides,
    )
    os.makedirs(args.output_path, exist_ok=True)
    save_path = osp.join(args.output_path, "result.csv")
    if args.input_format == "xyz":
        results = predictor.from_xyz_file(args.input_path, save_path)
    else:
        results = predictor.from_cif_file(args.input_path, save_path)
    print(results)


if __name__ == "__main__":
    main()

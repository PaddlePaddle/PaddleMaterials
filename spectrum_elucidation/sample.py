# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse

from ppmat.sampler import MolecularSampler
from ppmat.utils import logger


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Molecular structure sampling")
    # Select exactly one model source: a registered package or a local config.
    model_source = parser.add_mutually_exclusive_group(required=True)
    model_source.add_argument("--model_name", help="Registered model name.")
    model_source.add_argument("--config_path", help="Path to a local config file.")
    parser.add_argument("--weights_name", help="Weight filename in a model package.")
    parser.add_argument("--checkpoint_path", help="Path to a local checkpoint.")
    parser.add_argument(
        "--input_path",
        help="Dataset path used by by_dataloader or compute_metric mode.",
    )
    parser.add_argument(
        "--output_path",
        default="results",
        help="Directory in which sampling results are saved.",
    )
    parser.add_argument(
        "--mode",
        choices=["by_dataloader", "compute_metric"],
        default="by_dataloader",
    )
    args, config_overrides = parser.parse_known_args(argv)
    if (args.config_path is None) != (args.checkpoint_path is None):
        parser.error("--config_path and --checkpoint_path must be provided together")
    if any(value.startswith("-") or "=" not in value for value in config_overrides):
        parser.error("unrecognized arguments: " + " ".join(config_overrides))
    if args.input_path is not None:
        config_overrides.append(
            f"Sampler.data.dataset.__init_params__.path={args.input_path}"
        )
    return args, config_overrides


def main():
    args, config_overrides = parse_args()

    sampler = MolecularSampler(
        model_name=args.model_name,
        weights_name=args.weights_name,
        config_path=args.config_path,
        checkpoint_path=args.checkpoint_path,
        config_overrides=config_overrides,
    )
    if args.mode == "compute_metric":
        metric_result = sampler.compute_metric(save_path=args.output_path)
        for metric_name, metric_value in metric_result.items():
            logger.info(f"{metric_name}: {metric_value}")
    else:
        sampler.sample_by_dataloader(save_path=args.output_path)


if __name__ == "__main__":
    main()

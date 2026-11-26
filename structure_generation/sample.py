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
from pathlib import Path

import paddle.distributed as dist
from omegaconf import OmegaConf

from ppmat.sampler import MolecularSampler
from ppmat.sampler import StructureSampler
from ppmat.utils import logger
from ppmat.utils.molecular_conformers import write_molecular_conformers

__all__ = ["StructureSampler", "build_parser", "main"]


def build_parser():
    parser = argparse.ArgumentParser(
        description="Crystal or molecular structure sampling"
    )
    model_source = parser.add_mutually_exclusive_group(required=True)
    model_source.add_argument("--model_name", help="Registered model name.")
    model_source.add_argument("--config_path", help="Path to a local config file.")
    parser.add_argument(
        "--weights_name",
        help="Weight filename in a registered model package.",
    )
    parser.add_argument(
        "--checkpoint_path",
        help="Path to a local checkpoint file or directory.",
    )
    parser.add_argument(
        "--input_path",
        help="Dataset path for by_dataloader, compute_metric, or by_molecular_graph.",
    )
    parser.add_argument(
        "--output_path",
        default="results",
        help="Directory in which generated structures are saved.",
    )
    parser.add_argument("--chemical_formula", default="LiMnO2")
    parser.add_argument("--num_atoms", type=int, default=4)
    parser.add_argument(
        "--condition",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="Conditional property value; repeat for multi-property models.",
    )
    parser.add_argument(
        "--mode",
        choices=[
            "by_chemical_formula",
            "by_num_atoms",
            "by_condition",
            "by_dataloader",
            "by_molecular_graph",
            "compute_metric",
        ],
        default="by_chemical_formula",
    )
    return parser


def main():
    parser = build_parser()
    args, config_overrides = parser.parse_known_args()
    if args.model_name is not None and args.checkpoint_path is not None:
        parser.error("--checkpoint_path cannot be combined with --model_name")
    if args.config_path is not None and args.weights_name is not None:
        parser.error("--weights_name can only be used with --model_name")
    if args.config_path is not None and args.checkpoint_path is None:
        parser.error("--checkpoint_path is required with --config_path")
    invalid_overrides = [
        value for value in config_overrides if value.startswith("-") or "=" not in value
    ]
    if invalid_overrides:
        parser.error("unrecognized arguments: " + " ".join(invalid_overrides))
    by_molecular_graph = args.mode == "by_molecular_graph"
    data_section = "Sampler" if by_molecular_graph else "Sample"
    if args.input_path is not None:
        config_overrides.append(
            f"{data_section}.data.dataset.__init_params__.path={args.input_path}"
        )
    if by_molecular_graph:
        if dist.get_world_size() != 1:
            parser.error("by_molecular_graph requires a single process")
        config_overrides.extend(
            (
                "Dataset.train=null",
                "Metric=null",
                f"Sampler.output_dir={args.output_path}",
            )
        )

    sampler_class = MolecularSampler if by_molecular_graph else StructureSampler
    sampler = sampler_class(
        model_name=args.model_name,
        weights_name=args.weights_name,
        config_path=args.config_path,
        checkpoint_path=args.checkpoint_path,
        config_overrides=config_overrides,
    )
    if by_molecular_graph:
        samples = sampler.sample_by_dataloader(save_path=args.output_path)
        write_molecular_conformers(
            Path(args.output_path) / "conformers.sdf", samples["pred"]
        )
    elif args.mode == "compute_metric":
        metric_result = sampler.compute_metric(save_path=args.output_path)
        for metric_name, metric_value in metric_result.items():
            logger.info(f"{metric_name}: {metric_value}")
    elif args.mode == "by_chemical_formula":
        sampler.sample_by_chemical_formula(
            chemical_formula=args.chemical_formula,
            save_path=args.output_path,
        )
    elif args.mode == "by_num_atoms":
        sampler.sample_by_num_atoms(
            num_atoms=args.num_atoms,
            save_path=args.output_path,
        )
    elif args.mode == "by_condition":
        conditions = OmegaConf.to_container(
            OmegaConf.from_dotlist(args.condition), resolve=True
        )
        sampler.sample_by_condition(
            num_atoms=args.num_atoms,
            conditions=conditions,
            save_path=args.output_path,
        )
    else:
        sampler.sample_by_dataloader(save_path=args.output_path)


if __name__ == "__main__":
    main()

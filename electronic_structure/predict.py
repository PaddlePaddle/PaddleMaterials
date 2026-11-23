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

from ppmat.datasets.build_field import BuildField
from ppmat.predictor import FieldPredictor
from ppmat.utils import logger
from ppmat.visualization import VolumeVisualizer


def parse_grid_shape(value):
    shape = list(map(int, value.split(",")))
    if len(shape) == 1:
        shape *= 3
    if len(shape) != 3 or min(shape) <= 1:
        raise argparse.ArgumentTypeError(f"Invalid grid shape: {value}")
    return shape


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Electron-density field prediction")
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

    parser.add_argument(
        "--input_path",
        required=True,
        help="Path to one input file or a directory of input files.",
    )
    parser.add_argument(
        "--input_format",
        required=True,
        choices=["mol", "xyz", "cif", "cube", "chgcar", "json"],
        help="Input file format.",
    )

    parser.add_argument(
        "--output_path",
        default="./results",
        help="Directory used for predicted CUBE files.",
    )
    parser.add_argument(
        "--grid_batch_size",
        type=int,
        help="Maximum grid points per forward pass; defaults to Predict config.",
    )
    parser.add_argument(
        "--grid_shape",
        type=parse_grid_shape,
        default=[80, 80, 80],
        help="MOL grid shape as N or Nx,Ny,Nz.",
    )
    parser.add_argument(
        "--grid_padding",
        default=6.0,
        type=float,
        help="Padding around MOL coordinates in Angstrom.",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Write the predicted density visualization as a PNG file.",
    )
    parser.add_argument(
        "--save_html",
        action="store_true",
        help="Write the predicted density visualization as interactive HTML.",
    )
    parser.add_argument(
        "--show_plot",
        action="store_true",
        help="Display the predicted density in a Plotly window.",
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
    predictor = FieldPredictor(
        model_name=args.model_name,
        weights_name=args.weights_name,
        config_path=args.config_path,
        checkpoint_path=args.checkpoint_path,
        device=args.device,
        config_overrides=config_overrides,
    )
    if args.input_format == "mol":
        predictor.from_mol_file(
            mol_file_path=args.input_path,
            save_path=args.output_path,
            grid_shape=args.grid_shape,
            grid_padding=args.grid_padding,
            grid_batch_size=args.grid_batch_size,
        )
    elif args.input_format == "xyz":
        predictor.from_xyz_file(
            xyz_file_path=args.input_path,
            save_path=args.output_path,
            grid_shape=args.grid_shape,
            grid_padding=args.grid_padding,
            grid_batch_size=args.grid_batch_size,
        )
    elif args.input_format == "cif":
        predictor.from_cif_file(
            cif_file_path=args.input_path,
            save_path=args.output_path,
            grid_shape=args.grid_shape,
            grid_batch_size=args.grid_batch_size,
        )
    else:
        getattr(predictor, f"from_{args.input_format}_file")(
            args.input_path,
            save_path=args.output_path,
            grid_batch_size=args.grid_batch_size,
        )

    if args.visualize or args.save_html or args.show_plot:
        output_dir = Path(args.output_path)
        field_builder = BuildField(format="cube", name=predictor.model.target_name)
        visualizer = VolumeVisualizer()
        for cube_path in sorted(output_dir.glob("*_pred.cube")):
            figure = visualizer.render(
                field_builder(cube_path),
                isomin=0.05,
                isomax=3.5,
                surface_count=5,
                title="Predicted Electron Density",
            )
            if args.save_html:
                html_path = cube_path.with_suffix(".html")
                visualizer.save_html(figure, html_path)
                logger.info(f"Saved field visualization to: {html_path}")
            if args.visualize:
                image_path = cube_path.with_suffix(".png")
                visualizer.save_png(figure, image_path)
                logger.info(f"Saved field visualization to: {image_path}")
            if args.show_plot:
                figure.show()


if __name__ == "__main__":
    main()

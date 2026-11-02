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
from zipfile import ZIP_STORED
from zipfile import ZipFile

import yaml


TASKS = ("eea", "egb", "egc", "ei", "eps", "nc", "xc", "opv", "pe_i", "pe_ii")


def make_inference_config(config_path):
    with config_path.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    model_config = config["Model"]
    if model_config["__class_name__"] != "TransPolymerRegressor":
        raise ValueError(f"Not a TransPolymer config: {config_path}")

    params = model_config["__init_params__"]
    params["pretrained_model_path"] = None
    params["pretrained_model_url"] = None
    return yaml.safe_dump(config, sort_keys=False, allow_unicode=True)


def package_task(source_root, output_dir, task):
    model_name = f"transpolymer_{task}"
    source_dir = source_root / model_name
    config_path = source_dir / f"{model_name}_finetune.yaml"
    checkpoint_path = source_dir / "checkpoints" / "best.pdparams"
    if not config_path.is_file() or not checkpoint_path.is_file():
        raise FileNotFoundError(f"Missing config or best.pdparams in {source_dir}")
    if checkpoint_path.stat().st_size < 1024 * 1024:
        raise ValueError(f"Checkpoint is too small to be model weights: {checkpoint_path}")

    config = make_inference_config(config_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    archive_path = output_dir / f"{model_name}.zip"
    if archive_path.exists():
        raise FileExistsError(f"Archive already exists: {archive_path}")

    with ZipFile(
        archive_path, "w", compression=ZIP_STORED, allowZip64=True
    ) as archive:
        archive.writestr(f"{model_name}/{model_name}.yaml", config)
        archive.write(checkpoint_path, f"{model_name}/checkpoints/best.pdparams")
    return archive_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("output/transpolymer_full_train_artifacts"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/transpolymer_registry_packages"),
    )
    args = parser.parse_args()
    print(package_task(args.source_root, args.output_dir, args.task))


if __name__ == "__main__":
    main()

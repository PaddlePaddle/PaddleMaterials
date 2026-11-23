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

from pathlib import Path


def resolve_model_package_dir(model_name: str, extracted_path: str) -> str:
    """Resolve either a flat or one-directory-deep model package."""

    root = Path(extracted_path)
    candidates = (root, root / model_name)
    valid = [
        candidate
        for candidate in candidates
        if candidate.is_dir()
        and any(
            (candidate / f"{model_name}{suffix}").is_file()
            for suffix in (".yaml", ".yml")
        )
    ]
    if len(valid) == 1:
        return str(valid[0])
    if len(valid) > 1:
        raise ValueError(
            f"Ambiguous package layout for '{model_name}' under {extracted_path}: "
            f"{[str(path) for path in valid]}"
        )
    raise FileNotFoundError(
        f"Invalid package for '{model_name}' under {extracted_path}. Expected "
        f"'{model_name}.yaml' directly in the extracted directory or in its "
        f"'{model_name}' child directory."
    )


def get_model_config_path(model_name: str, package_dir: str) -> str:
    """Return the model YAML from a validated package."""

    package_path = Path(package_dir)
    matches = [
        package_path / f"{model_name}{suffix}"
        for suffix in (".yaml", ".yml")
        if (package_path / f"{model_name}{suffix}").is_file()
    ]
    if len(matches) == 1:
        return str(matches[0])
    if len(matches) > 1:
        raise ValueError(
            f"Package {package_dir} contains both YAML variants for '{model_name}'."
        )
    raise FileNotFoundError(
        f"Package {package_dir} does not contain '{model_name}.yaml' or "
        f"'{model_name}.yml'."
    )

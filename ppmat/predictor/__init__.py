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

from ppmat.predictor.base import BasePredictor

__all__ = [
    "BasePredictor",
    "FieldPredictor",
    "PotentialPredictor",
    "PropertyPredictor",
    "SpectrumPredictor",
]


def __getattr__(name):
    if name == "FieldPredictor":
        from ppmat.predictor.field_predictor import FieldPredictor

        predictor_class = FieldPredictor
    elif name == "PotentialPredictor":
        from ppmat.predictor.potential_predictor import PotentialPredictor

        predictor_class = PotentialPredictor
    elif name == "PropertyPredictor":
        from ppmat.predictor.property_predictor import PropertyPredictor

        predictor_class = PropertyPredictor
    elif name == "SpectrumPredictor":
        from ppmat.predictor.spectrum_predictor import SpectrumPredictor

        predictor_class = SpectrumPredictor
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    globals()[name] = predictor_class
    return predictor_class

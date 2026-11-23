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
from ppmat import calculator  # noqa
from ppmat import datasets  # noqa
from ppmat import losses  # noqa
from ppmat import metrics  # noqa
from ppmat import models  # noqa
from ppmat import optimizer  # noqa
from ppmat import predictor  # noqa
from ppmat import sampler  # noqa
from ppmat import schedulers  # noqa
from ppmat import trainer  # noqa
from ppmat import utils  # noqa
from ppmat import vocab  # noqa

__all__ = [
    "calculator",
    "models",
    "trainer",
    "predictor",
    "sampler",
    "vocab",
]

try:
    # import auto-generated version information from '._version' file, using
    # setuptools_scm via 'pip install'. Details of versioning rule can be referd to:
    # https://peps.python.org/pep-0440/#public-version-identifiers
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown version"

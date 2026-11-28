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

"""Regression guard for the existing PaddleMaterials public import surface.

LiFlow registers extra Datasets, Models, Metrics and Predictors through the
public ``__init__`` files.  Every incremental registration in this PR must keep
the pre-existing public symbols importable, so this baseline is protected by a
dedicated test that runs after every registration task.
"""

from __future__ import annotations


def test_existing_public_symbols_are_preserved():
    from ppmat.datasets import DensityDataset
    from ppmat.datasets import MP20Dataset
    from ppmat.datasets import SFINDataset
    from ppmat.models import CHGNet
    from ppmat.models import MEGNetPlus
    from ppmat.models import SphereNet
    from ppmat.predictor import BasePredictor
    from ppmat.predictor import FieldPredictor
    from ppmat.predictor import PotentialPredictor

    assert all(
        symbol is not None
        for symbol in (
            DensityDataset,
            MP20Dataset,
            SFINDataset,
            CHGNet,
            MEGNetPlus,
            SphereNet,
            BasePredictor,
            FieldPredictor,
            PotentialPredictor,
        )
    )


def test_liflow_models_are_public():
    from ppmat.models import DualPaiNN
    from ppmat.models import LiFlow

    assert DualPaiNN is not None
    assert LiFlow is not None


def test_liflow_dataset_is_public():
    from ppmat.datasets import LiFlowDataset
    from ppmat.datasets import NoisedFrameDataset

    assert LiFlowDataset is not None
    assert NoisedFrameDataset is not None


def test_liflow_metric_is_public():
    from ppmat.metrics import LiFlowMSE

    assert LiFlowMSE is not None


def test_integrator_predictor_is_public():
    from ppmat.predictor import IntegratorPredictor

    assert IntegratorPredictor is not None

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

from typing import Any
from typing import Dict
from typing import Iterable
from typing import Optional

import paddle

from ppmat.metrics.diffnmr_metric import DiffNMRMetric
from ppmat.metrics.diffnmr_metric import SamplingMolecularMetrics
from ppmat.metrics.streaming_base import StreamingMetricBase
from ppmat.schedulers import scheduling_diffnmr


class DiffNMRStreamingAdapter(StreamingMetricBase):
    """
    Minimal, robust streaming adapter for DiffNMR.

    Contract expected from model.forward(...):
      result["pred_dict"]  : {"masked_pred_X", "masked_pred_E", "pred_y"}
      result["label_dict"] : {"true_X", "true_E", "true_y"}
      result or batch      : {"node_mask"}
      result (optional)    : {"noisy_data"}; if missing, adapter recomputes.

    Train: accumulates CE/Top-k via DiffNMRMetric (no logging noise).
    Eval : accumulates NLL (and KL/logp if compute_val_loss supports it).
    Sample :
        exact SMILES match accuracy
        RDKit quality metrics (Validity/Uniqueness/Novelty/ConnComp + stats)
        Histogram MAE on n-nodes / atom-types / bond-types / valency
        Optional retrieval top-k (molVec-nmrVec), with CSV similarity dump
    """

    def __init__(
        self,
        *,
        t_scale: float = 1.0,  # multiply eval NLL/terms by T if you want
        dataset_infos: Any = None,
        sample_metrics: Optional[Iterable[str]] = None,
    ):
        self.t_scale = float(t_scale)
        self.train_core = DiffNMRMetric(mode="train", dataset_infos=dataset_infos)
        self.model: Optional[paddle.nn.Layer] = None
        self.eval_acc = EvalAccumulator()
        self.sample_core = None
        self.dataset_infos = None
        self.clip = None
        self.num_candidate = 1
        if isinstance(sample_metrics, dict):
            sample_metrics = sample_metrics.keys()
        elif isinstance(sample_metrics, str):
            sample_metrics = [sample_metrics]
        self.sample_metrics = None if sample_metrics is None else set(sample_metrics)
        self.reset()

    # ---- lifecycle ----
    def bind(self, **runtime_objs):
        """Receive runtime objects (model, dataset_infos, clip, ...)."""
        self.model = runtime_objs.get("model", self.model)
        if hasattr(self.train_core, "bind"):
            # Only pass what DiffNMRMetric.bind accepts (keep it simple)
            try:
                self.train_core.bind(
                    **{k: v for k, v in runtime_objs.items() if k != "model"}
                )
            except TypeError:
                pass
        self.clip = runtime_objs.get("clip", self.clip)
        self.num_candidate = runtime_objs.get("num_candidate", self.num_candidate)
        self.dataset_infos = runtime_objs.get("dataset_infos", self.dataset_infos)

    def reset(self):
        if hasattr(self.train_core, "reset"):
            self.train_core.reset()
        self.eval_acc.reset()
        self._right, self._total = 0, 0
        if self.sample_core is not None:
            self.sample_core.reset()

    # ---- streaming entrypoints ----
    def update_step(self, *, result: Dict[str, Any], batch: Any, stage: str):
        if stage == "train":
            self._update_train(result, batch)
        elif stage == "eval":
            self._update_eval(result, batch)
        elif stage == "sample":
            self._update_sample(result, batch)

    def compute_epoch(self, *, stage: str) -> Dict[str, float]:
        if stage == "train":
            return self._finalize_train()
        if stage == "eval":
            return self.eval_acc.finalize()
        if stage == "sample":
            return self._finalize_sample()
        return {}

    # ---- train path ----
    def _update_train(self, result: Dict[str, Any], batch: Any):
        pred = result.get("pred_dict") or {}
        lab = result.get("label_dict") or {}
        # Require exactly four tensors; skip silently if anything missing
        masked_pred_X = pred.get("masked_pred_X", None)
        masked_pred_E = pred.get("masked_pred_E", None)
        true_X = lab.get("true_X", None)
        true_E = lab.get("true_E", None)
        if any(v is None for v in (masked_pred_X, masked_pred_E, true_X, true_E)):
            return
        self.train_core(
            pred={"masked_pred_X": masked_pred_X, "masked_pred_E": masked_pred_E},
            label={"true_X": true_X, "true_E": true_E},
            log=False,
        )

    def _finalize_train(self) -> Dict[str, float]:
        out: Dict[str, float] = {}
        impl = getattr(self.train_core, "impl", None)
        if impl is not None and hasattr(impl, "log_epoch_metrics"):
            res = impl.log_epoch_metrics() or {}
            # Already returns flat dict of scalar metrics; cast to float
            for k, v in res.items():
                out[str(k)] = float(v)
        # reset train core for next epoch
        if hasattr(self.train_core, "reset"):
            self.train_core.reset()
        return out

    def _finalize_sample(self) -> Dict[str, float]:
        if self._total == 0:
            return {}
        # out = {
        #     "Accuracy": self._right / float(self._total),
        #     "basic_metrics/n_mae": float(self.sample_core.mae_n.accumulate()),
        #     "basic_metrics/node_mae": float(self.sample_core.mae_nodes.accumulate()),
        #     "basic_metrics/edge_mae": float(self.sample_core.mae_edges.accumulate()),
        #     "basic_metrics/valency_mae": float(self.sample_core.mae_val.accumulate()),
        # }
        return self.sample_metric_dict

    # ---- eval path ----
    def _update_eval(self, result: Dict[str, Any], batch: Any):
        assert (
            self.model is not None
        ), "Adapter missing model. Call adapter.bind(model=...) first."
        pred = result.get("pred_dict") or {}
        lab = result.get("label_dict") or {}
        node_mask = _coalesce(
            result.get("node_mask", None),
            batch.get("node_mask", None) if isinstance(batch, dict) else None,
        )

        # Need all six tensors
        pred_X = pred.get("masked_pred_X", None)
        pred_E = pred.get("masked_pred_E", None)
        pred_y = pred.get("pred_y", None)
        true_X = lab.get("true_X", None)
        true_E = lab.get("true_E", None)
        true_y = lab.get("true_y", None)
        if any(
            v is None
            for v in (pred_X, pred_E, pred_y, true_X, true_E, true_y, node_mask)
        ):
            return

        # Prefer cache; else recompute noise
        noisy = result.get("noisy_data", None)
        if noisy is None:
            flag = bool(getattr(self.model, "flag_use_formula", False))
            noisy = scheduling_diffnmr.apply_noise(
                self.model, true_X, true_E, true_y, node_mask, flag
            )

        # Pack predictions to match compute_val_loss signature
        Pred = type("Pred", (), {})
        pred_obj = Pred()
        pred_obj.X, pred_obj.E, pred_obj.y = pred_X, pred_E, pred_y

        # Try to get detailed terms if supported; fallback to scalar
        batch_spectrum = batch["spectrum"]
        condition_H1nmr = paddle.to_tensor(batch_spectrum["H_nmr"])
        condition_C13nmr = paddle.to_tensor(batch_spectrum["C_nmr"])
        num_H_peak = paddle.to_tensor(batch_spectrum["num_H_peak"])
        num_C_peak = paddle.to_tensor(batch_spectrum["num_C_peak"])
        condition_Spectrum = [condition_H1nmr, num_H_peak, condition_C13nmr, num_C_peak]
        try:
            terms = scheduling_diffnmr.compute_val_loss(
                self.model,
                pred_obj,
                noisy,
                true_X,
                true_E,
                true_y,
                node_mask,
                condition=condition_Spectrum,
                return_terms=True,  # or True in test mode
            )
        except TypeError:
            # Older signature without 'test' or different arg order
            terms = scheduling_diffnmr.compute_val_loss(
                self.model, pred_obj, noisy, true_X, true_E, true_y, node_mask, []
            )

        B = int(true_X.shape[0])
        # Accept: dict with pieces OR scalar tensor/float
        if isinstance(terms, dict):
            # Optional scale by T (if you want NLL per trajectory step)
            for k in ("nll", "X_kl", "E_kl", "X_logp", "E_logp"):
                if k in terms and terms[k] is not None:
                    self.eval_acc.add(
                        key=_name_map(k),
                        value=terms[k],
                        batch_size=B,
                        scale=self.t_scale,
                    )
        else:
            self.eval_acc.add(
                key="val_nll", value=terms, batch_size=B, scale=self.t_scale
            )

    def _update_sample(self, result: Dict[str, Any], batch: Any):
        if self.sample_core is None:
            assert self.dataset_infos is not None, "Sampling requires dataset infos."
            train_smiles = None
            if self.sample_metrics is None or "Novelty" in self.sample_metrics:
                train_smiles = self.dataset_infos.load_train_smiles()
            self.sample_core = SamplingMolecularMetrics(
                dataset_infos=self.dataset_infos,
                train_smiles=train_smiles,
                clip=self.clip,
                num_candidate=self.num_candidate,
            )
        self.sample_metric_dict = self.sample_core(
            samples=result["samples"],
            current_epoch=result.get("epoch_id", 0),
            local_rank=result.get("local_rank", 0),
            output_dir=result.get("output_dir", "."),
            flag_test=False,
        )
        self._right += int(self.sample_metric_dict.get("Right Number", 0))
        self._total += int(self.sample_metric_dict.get("Total Number", 0))


# -------------------------
# small helpers / accumulator
# -------------------------


def _name_map(k: str) -> str:
    return {
        "nll": "val_nll",
        "X_kl": "val_X_kl",
        "E_kl": "val_E_kl",
        "X_logp": "val_X_logp",
        "E_logp": "val_E_logp",
    }.get(k, k)


class EvalAccumulator:
    """
    Pre-key accumulator:
      - supports scalar (mean) or vector ([B]) tensors/number
      - keeps {key: sum} and {key: denom} separately
      - optional scale (e.g., multiply by T)
      - optional custom denom (override batch size for that key)
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self._sum: Dict[str, float] = {}
        self._den: Dict[str, int] = {}

    @staticmethod
    def _to_float(x) -> float:
        if isinstance(x, paddle.Tensor):
            return (
                float(x.numpy().item())
                if x.numel() == 1
                else float(paddle.sum(x).numpy().item())
            )
        return float(x)

    def add(
        self,
        *,
        key: str,
        value,
        batch_size: Optional[int] = None,
        scale: float = 1.0,
        denom: Optional[float] = None,
    ):
        """
        value:
          - tensor [B] → treat as sum(vector)
          - tensor scalar / python number → treat as mean and multiply by denom (or
            batch_size)
        denom:
          - if provided, use it as denominator for this (key, step)
          - else: if value is vector → len(vector)
                 if value is scalar → batch_size (fallback 1)
        """
        if value is None:
            return

        if isinstance(value, paddle.Tensor) and value.numel() > 1:
            # vector → sum directly; denom = length
            v_sum = float(paddle.sum(value).numpy().item()) * float(scale)
            d = float(value.shape[0]) if denom is None else float(denom)
        else:
            # scalar → assume it's mean; multiply by denom/batch_size
            mean_val = self._to_float(value) * float(scale)
            if denom is None:
                d = float(batch_size if batch_size is not None else 1.0)
            else:
                d = float(denom)
            v_sum = mean_val * d

        self._sum[key] = self._sum.get(key, 0.0) + v_sum
        self._den[key] = self._den.get(key, 0.0) + d

    def finalize(self) -> Dict[str, float]:
        out = {}
        for k, s in self._sum.items():
            d = max(self._den.get(k, 0.0), 1.0)
            out[k] = s / d
        return out


def _coalesce(*vals):
    """Return first value that is not None (NO truthiness on tensors)."""
    for v in vals:
        if v is not None:
            return v
    return None

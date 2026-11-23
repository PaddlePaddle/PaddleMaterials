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

from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from rdkit import Chem
from rdkit.Chem import DataStructs
from rdkit.Chem import RDKFingerprint

from ppmat.models.diffnmr.utils import diffgraphformer_utils as utils
from ppmat.schedulers import scheduling_diffnmr
from ppmat.utils.ext_rdkit import compute_molecular_metrics
from ppmat.utils.ext_rdkit import mol_from_graphs

# =========================
# Utilities
# =========================


def _is_dist():
    try:
        import paddle.distributed as dist

        return dist.is_initialized() and dist.get_world_size() > 1
    except Exception:
        return False


def _all_reduce_sum_(t: paddle.Tensor) -> paddle.Tensor:
    """Inplace SUM all_reduce if distributed; returns t."""
    if _is_dist():
        import paddle.distributed as dist

        dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t


def _to_f32(x) -> paddle.Tensor:
    return (
        paddle.to_tensor(x, dtype="float32")
        if not isinstance(x, paddle.Tensor)
        else x.astype("float32")
    )


def _l1_normalize(x: paddle.Tensor, eps: float = 1e-12) -> paddle.Tensor:
    x = x.astype("float32")
    s = paddle.sum(x)
    return x / paddle.maximum(s, _to_f32(eps))


# =========================
# Scalar Metrics (fixed bugs)
# =========================


class SumExceptBatchMetric(paddle.metric.Metric):
    """Sum over all dims then average per-sample (batch)."""

    def __init__(self):
        super().__init__()
        self.reset()

    def name(self):
        return self.__class__.__name__

    def reset(self):
        self._sum = _to_f32(0.0)
        self._n = _to_f32(0.0)

    def update(self, values: paddle.Tensor):
        self._sum += paddle.sum(values)
        self._n += _to_f32(values.shape[0])

    def accumulate(self):
        num = _all_reduce_sum_(self._sum.clone())
        den = _all_reduce_sum_(self._n.clone())
        return num / paddle.maximum(den, _to_f32(1.0))

    def __call__(self, values: paddle.Tensor):
        self.reset()
        self.update(values)
        return self.accumulate()


class SumExceptBatchKL(paddle.metric.Metric):
    """
    KL with robust handling of logits/probs:
      update(p, q) treats p,q as logits by default (safer).
      You can pass log_input/log_target=True if providing log-probs.
    """

    def __init__(self, log_input: bool = False, log_target: bool = False):
        super().__init__()
        self.log_input = log_input
        self.log_target = log_target
        self.reset()

    def name(self):
        return self.__class__.__name__

    def reset(self):
        self._sum = _to_f32(0.0)
        self._n = _to_f32(0.0)

    def update(self, p: paddle.Tensor, q: paddle.Tensor):
        x = p if self.log_input else F.log_softmax(p, axis=-1)
        y = q if self.log_target else F.softmax(q, axis=-1)
        kl = F.kl_div(x, y, reduction="sum", log_target=self.log_target)
        self._sum += kl
        self._n += _to_f32(p.shape[0])

    def accumulate(self):
        num = _all_reduce_sum_(self._sum.clone())
        den = _all_reduce_sum_(self._n.clone())
        return num / paddle.maximum(den, _to_f32(1.0))

    def __call__(self, p: paddle.Tensor, q: paddle.Tensor):
        self.reset()
        self.update(p, q)
        return self.accumulate()


class CrossEntropyMetric(paddle.metric.Metric):
    """Average CE for one-hot targets; fixed accumulate and no unintended resets."""

    def __init__(self):
        super().__init__()
        self.reset()

    def name(self):
        return self.__class__.__name__

    def reset(self):
        self._sum = _to_f32(0.0)
        self._n = _to_f32(0.0)

    def update(self, logits: paddle.Tensor, target_onehot: paddle.Tensor):
        target = paddle.argmax(target_onehot, axis=-1)
        ce = F.cross_entropy(logits, target, reduction="sum")
        self._sum += ce
        self._n += _to_f32(logits.shape[0])

    def accumulate(self):
        num = _all_reduce_sum_(self._sum.clone())
        den = _all_reduce_sum_(self._n.clone())
        return num / paddle.maximum(den, _to_f32(1.0))

    def __call__(self, logits: paddle.Tensor, target_onehot: paddle.Tensor):
        self.reset()
        self.update(logits, target_onehot)
        return self.accumulate()


class NLL(paddle.metric.Metric):
    """Streaming Negative Log-Likelihood mean."""

    def __init__(self):
        super().__init__()
        self.reset()

    def name(self):
        return self.__class__.__name__

    def reset(self):
        self._sum = _to_f32(0.0)
        self._num = _to_f32(0.0)

    def update(self, batch_nll: paddle.Tensor):
        self._sum += paddle.sum(batch_nll)
        self._num += _to_f32(batch_nll.numel())

    def accumulate(self):
        num = _all_reduce_sum_(self._sum.clone())
        den = _all_reduce_sum_(self._num.clone())
        out = num / paddle.maximum(den, _to_f32(1.0))
        self.reset()
        return out

    def __call__(self, batch_nll: paddle.Tensor):
        self.update(batch_nll)
        return self.accumulate()


# =========================
# CE Refactor (vectorized + per-class stats + top1)
# =========================


class _CETracker:
    """
    Overall CE on masked positions + per-class CE and counts + top1 acc.
    Vectorized, device-side accumulation, distributed-safe.
    """

    def __init__(
        self,
        num_classes: int,
        class_names: Optional[List[str]] = None,
        prefix: str = "",
    ):
        self.C = num_classes
        self.class_names = class_names or [str(i) for i in range(num_classes)]
        self.prefix = prefix.rstrip("/")
        self.reset()

    def reset(self):
        self._loss_sum = _to_f32(0.0)
        self._count = _to_f32(0.0)
        self._correct_sum = _to_f32(0.0)
        self._loss_per_cls = paddle.zeros([self.C], dtype="float32")
        self._cnt_per_cls = paddle.zeros([self.C], dtype="float32")

    @staticmethod
    def _flatten_logits_targets(
        logits: paddle.Tensor, target_onehot: paddle.Tensor
    ) -> Tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]:
        """
        Flatten to [N, C] and build valid mask from one-hot (any non-zero).
        """
        C = target_onehot.shape[-1]
        t = paddle.reshape(target_onehot, [-1, C])
        mask = paddle.any(t != 0.0, axis=-1)  # [N]
        ll = paddle.reshape(logits, [-1, C])
        return ll, t, mask

    def update(self, logits: paddle.Tensor, target_onehot: paddle.Tensor):
        l, t, mask = self._flatten_logits_targets(logits, target_onehot)  # noqa
        if paddle.sum(mask).item() == 0:
            return
        target_idx = paddle.argmax(t, axis=-1)
        # CE per-sample
        loss_vec = F.cross_entropy(l, target_idx, reduction="none")  # [N]
        # mask
        loss_vec = paddle.masked_select(loss_vec, mask)
        target_idx = paddle.masked_select(target_idx, mask)
        # overall
        self._loss_sum += paddle.sum(loss_vec)
        self._count += _to_f32(loss_vec.shape[0])
        # accuracy
        pred_idx = paddle.argmax(l, axis=-1)
        correct = (pred_idx == paddle.argmax(t, axis=-1)).astype("float32")
        correct = paddle.masked_select(correct, mask)
        self._correct_sum += paddle.sum(correct)
        # per-class (vectorized)
        oh = F.one_hot(target_idx, num_classes=self.C).astype("float32")  # [Nm, C]
        self._loss_per_cls += paddle.sum(oh * loss_vec.unsqueeze(-1), axis=0)
        self._cnt_per_cls += paddle.sum(oh, axis=0)

    def compute(self) -> Dict[str, float]:
        # distributed reduction
        loss_sum = _all_reduce_sum_(self._loss_sum.clone())
        count = _all_reduce_sum_(self._count.clone())
        correct = _all_reduce_sum_(self._correct_sum.clone())
        loss_cls = _all_reduce_sum_(self._loss_per_cls.clone())
        cnt_cls = _all_reduce_sum_(self._cnt_per_cls.clone())

        out: Dict[str, float] = {}
        ce = (loss_sum / paddle.maximum(count, _to_f32(1.0))).item()
        acc = (correct / paddle.maximum(count, _to_f32(1.0))).item()
        if self.prefix:
            out[f"{self.prefix}/ce"] = ce
            out[f"{self.prefix}/acc_top1"] = acc
            # per-class ce
            for i, name in enumerate(self.class_names):
                denom = paddle.maximum(cnt_cls[i], _to_f32(1.0))
                val = (loss_cls[i] / denom).item()
                out[f"{self.prefix}/ce_class_{name}"] = val
        else:
            out["ce"] = ce
            out["acc_top1"] = acc
            for i, name in enumerate(self.class_names):
                denom = paddle.maximum(cnt_cls[i], _to_f32(1.0))
                val = (loss_cls[i] / denom).item()
                out[f"ce_class_{name}"] = val
        return out


class TrainMolecularMetricsDiscrete(nn.Layer):
    """
    Training/validation metrics:
      - Atom overall CE + per-class CE + top1
      - Bond overall CE + per-class CE + top1
    """

    def __init__(self, dataset_infos):
        super().__init__()
        atom_names = list(dataset_infos.atom_decoder)  # e.g. ['H','C','N',...]
        self.atom_ce = _CETracker(
            num_classes=len(atom_names), class_names=atom_names, prefix="train/atom"
        )
        # Bond: NoBond=0, Single=1, Double=2, Triple=3, Aromatic=4
        bond_names = ["NoBond", "Single", "Double", "Triple", "Aromatic"]
        self.bond_ce = _CETracker(
            num_classes=len(bond_names), class_names=bond_names, prefix="train/bond"
        )

    def forward(self, masked_pred_X, masked_pred_E, true_X, true_E, log: bool = False):
        self.atom_ce.update(masked_pred_X, true_X)
        self.bond_ce.update(masked_pred_E, true_E)
        if not log:
            return {}
        out = {}
        out.update(self.atom_ce.compute())
        out.update(self.bond_ce.compute())
        return out

    def reset(self):
        self.atom_ce.reset()
        self.bond_ce.reset()

    def log_epoch_metrics(self) -> Dict[str, float]:
        out = {}
        out.update(
            {
                k.replace("train/", "train_epoch/"): v
                for k, v in self.atom_ce.compute().items()
            }
        )
        out.update(
            {
                k.replace("train/", "train_epoch/"): v
                for k, v in self.bond_ce.compute().items()
            }
        )
        return out


# =========================
# Sampling-time Metrics (vectorized counters + stable bins)
# =========================


class _NHistogram:
    def __init__(self, max_n: int):
        self.max_n = max_n
        self.hist = paddle.zeros([max_n + 1], "float32")

    def update(self, molecules):
        # gather node counts then bincount
        ns = [int(atom_types.shape[0]) for atom_types, _ in molecules]
        if len(ns) == 0:
            return
        idx = paddle.to_tensor(ns, dtype="int64")
        self.hist += paddle.bincount(idx, minlength=self.max_n + 1).astype("float32")

    def accumulate(self):
        return _l1_normalize(self.hist)

    def reset(self):
        self.hist = paddle.zeros_like(self.hist)

    def __call__(self, molecules):
        self.update(molecules)
        return self.accumulate()

    def __getitem__(self, i):
        return self.hist[i]


class _NodeHistogram:
    def __init__(self, num_atom_types: int):
        self.C = num_atom_types
        self.hist = paddle.zeros([num_atom_types], "float32")

    def update(self, molecules):
        # concat all atom types -> bincount
        arrs = []
        for atom_types, _ in molecules:
            a = paddle.to_tensor(atom_types, dtype="int64")
            arrs.append(a)
        if not arrs:
            return
        flat = paddle.concat(arrs, axis=0)
        self.hist += paddle.bincount(flat, minlength=self.C).astype("float32")

    def accumulate(self):
        return _l1_normalize(self.hist)

    def reset(self):
        self.hist = paddle.zeros_like(self.hist)

    def __call__(self, molecules):
        self.update(molecules)
        return self.accumulate()

    def __getitem__(self, i):
        return self.hist[i]


class _EdgeHistogram:
    def __init__(self, num_edge_types: int):
        self.C = num_edge_types
        self.hist = paddle.zeros([num_edge_types], "float32")

    def update(self, molecules):
        # collect upper-triangular edge types of all mols, then bincount
        arrs = []
        for _, edge_types in molecules:
            e = paddle.to_tensor(edge_types)
            mask = paddle.triu(paddle.ones_like(e), diagonal=1).astype("bool")
            arrs.append(e[mask])
        if not arrs:
            return
        flat = paddle.concat(arrs).astype("int64")
        self.hist += paddle.bincount(flat, minlength=self.C).astype("float32")

    def accumulate(self):
        return _l1_normalize(self.hist)

    def reset(self):
        self.hist = paddle.zeros_like(self.hist)

    def __call__(self, molecules):
        self.update(molecules)
        return self.accumulate()

    def __getitem__(self, i):
        return self.hist[i]


class _ValencyHistogram:
    """Integer valency bins matching the MSD-NMR dataset statistics."""

    def __init__(self, max_n, bond_decoder=None, num_edge_types=5):
        self.L = 3 * max_n - 2
        self.hist = paddle.zeros([self.L], "float32")
        orders = [0.0, 1.0, 2.0, 3.0, 1.5]
        if bond_decoder is not None:
            order_by_token = {
                "NO_BOND": 0.0,
                "SINGLE": 1.0,
                "DOUBLE": 2.0,
                "TRIPLE": 3.0,
                "AROMATIC": 1.5,
            }
            orders = [0.0] * num_edge_types
            for token, index in bond_decoder["token_to_id"].items():
                orders[index] = order_by_token[token]
        self.bond_orders = paddle.to_tensor(orders, dtype="float32")

    def update(self, molecules):
        arrs = []
        for _, edge_types in molecules:
            edge_ids = paddle.to_tensor(edge_types, dtype="int64")
            orders = self.bond_orders[edge_ids]
            forward = paddle.triu(orders, diagonal=1)
            reverse = paddle.triu(orders.transpose([1, 0]), diagonal=1)
            pair_orders = paddle.where(forward > 0, forward, reverse)
            idx = (pair_orders.sum(axis=0) + pair_orders.sum(axis=1)).astype("int64")
            arrs.append(idx)
        if not arrs:
            return
        flat = paddle.concat(arrs)
        self.hist += paddle.bincount(flat, minlength=self.L).astype("float32")

    def accumulate(self):
        return _l1_normalize(self.hist)

    def reset(self):
        self.hist = paddle.zeros_like(self.hist)

    def __call__(self, molecules):
        self.update(molecules)
        return self.accumulate()

    def __getitem__(self, i):
        return self.hist[i]


class _HistMAE(paddle.metric.Metric):
    """MAE between predicted histogram and a fixed target histogram (streaming)."""

    def __init__(self, target_hist: paddle.Tensor, name_prefix: str = ""):
        super().__init__()
        assert paddle.abs(paddle.sum(target_hist) - 1.0) < 1e-3
        self.target = target_hist.astype("float32")
        self.prefix = name_prefix
        self.reset()

    def name(self):
        return f"{self.prefix}HistMAE" if self.prefix else "HistMAE"

    def reset(self):
        self._sum = _to_f32(0.0)
        self._n = _to_f32(0.0)

    def update(self, pred_hist: paddle.Tensor):
        p = _l1_normalize(pred_hist)
        self._sum += paddle.sum(paddle.abs(p - self.target))
        self._n += _to_f32(1.0)

    def accumulate(self):
        num = _all_reduce_sum_(self._sum.clone())
        den = _all_reduce_sum_(self._n.clone())
        return num / paddle.maximum(den, _to_f32(1.0))

    def __call__(self, pred_hist: paddle.Tensor):
        self.update(pred_hist)
        return self.accumulate()


class SamplingMolecularMetrics(nn.Layer):
    """
    Sampling/eval metrics:
      1) exact SMILES match accuracy
      2) RDKit quality metrics (Validity/Uniqueness/Novelty/ConnComp + stats)
      3) Histogram MAE on n-nodes / atom-types / bond-types / valency
      4) Optional retrieval top-k (molVec-spectrumVec), with CSV similarity dump.
    """

    def __init__(
        self,
        dataset_infos: Any,
        train_smiles: List[str],
        clip: Optional[nn.Layer] = None,
        num_candidate: int = 1,
    ):
        super().__init__()
        self.di = dataset_infos
        self.train_smiles = train_smiles
        self.num_candidate = num_candidate
        self.atom_decoder = dataset_infos.atom_decoder
        if clip:
            self.clip = clip
            self.spectrumVec = clip.spectrum_encoder
            self.molVec = clip.graph_encoder

        # target histograms
        self.register_buffer("target_n", _l1_normalize(_to_f32(dataset_infos.n_nodes)))
        self.register_buffer(
            "target_nodes", _l1_normalize(_to_f32(dataset_infos.node_types))
        )
        self.register_buffer(
            "target_edges", _l1_normalize(_to_f32(dataset_infos.edge_types))
        )
        self.register_buffer(
            "target_val", _l1_normalize(_to_f32(dataset_infos.valency_distribution))
        )

        # online counters
        self.gen_n = _NHistogram(dataset_infos.max_n_nodes)
        self.gen_nodes = _NodeHistogram(dataset_infos.num_atom_types)
        self.gen_edges = _EdgeHistogram(int(dataset_infos.edge_types.shape[0]))
        self.gen_val = _ValencyHistogram(
            dataset_infos.max_n_nodes,
            bond_decoder=dataset_infos.vocab["bond"],
            num_edge_types=int(dataset_infos.edge_types.shape[0]),
        )

        # MAEs
        self.mae_n = _HistMAE(self.target_n, "n/")
        self.mae_nodes = _HistMAE(self.target_nodes, "node/")
        self.mae_edges = _HistMAE(self.target_edges, "edge/")
        self.mae_val = _HistMAE(self.target_val, "valency/")

    def forward(
        self,
        samples: Dict[str, Any],
        current_epoch: int,
        local_rank: int,
        output_dir: str,
        flag_test=False,
        log_each_molecule=False,
    ) -> Dict[str, Any]:
        to_log: Dict[str, Any] = {}
        pred, true = samples["pred"], samples["true"]
        total = samples["n_all"]

        # 1) exact match
        hit = 0
        for p, t in zip(pred, true):
            try:
                mg = mol_from_graphs(self.atom_decoder, *p)
                mt = mol_from_graphs(self.atom_decoder, *t)
                if Chem.MolToSmiles(mg, True) == Chem.MolToSmiles(mt, True):
                    hit += 1
            except Exception:
                continue
        to_log.update(
            {"Accuracy": hit / total, "Right Number": hit, "Total Number": total}
        )

        # 2) RDKit global metrics
        stability, rdkit_metrics, all_smiles = compute_molecular_metrics(
            pred, self.train_smiles, self.di
        )
        if local_rank == 0:
            to_log.update(stability)
            val, uniq, nov, conn = rdkit_metrics[0]
            to_log.update(
                {
                    "Validity": val,
                    "Uniqueness": uniq,
                    "Novelty": nov,
                    "Connected Components": conn,
                }
            )
            for k, v in rdkit_metrics[2].items():
                to_log[k] = v

        # 3) Histogram MAE (streaming, cross-batch safe)
        g_n = self.gen_n(pred)
        self.mae_n(g_n)
        g_nd = self.gen_nodes(pred)
        self.mae_nodes(g_nd)
        g_ed = self.gen_edges(pred)
        self.mae_edges(g_ed)
        g_val = self.gen_val(pred)
        self.mae_val(g_val)

        if local_rank == 0:
            to_log["Gen n distribution"] = g_n
            to_log["Gen node distribution"] = g_nd
            to_log["Gen edge distribution"] = g_ed
            to_log["Gen valency distribution"] = g_val
            to_log["basic_metrics/n_mae"] = self.mae_n.accumulate()
            to_log["basic_metrics/node_mae"] = self.mae_nodes.accumulate()
            to_log["basic_metrics/edge_mae"] = self.mae_edges.accumulate()
            to_log["basic_metrics/valency_mae"] = self.mae_val.accumulate()

        # 4) per-type deltas (diagnostics)
        for i, atom_type in enumerate(self.atom_decoder):
            to_log[f"molecular_metrics/{atom_type}_dist"] = float(
                (g_nd[i] - self.target_nodes[i]).item()
            )
        for j, bond_type in enumerate(
            ["No bond", "Single", "Double", "Triple", "Aromatic"]
        ):
            to_log[f"molecular_metrics/bond_{bond_type}_dist"] = float(
                (g_ed[j] - self.target_edges[j]).item()
            )
        for k in range(min(6, g_val.shape[0])):
            to_log[f"molecular_metrics/valency_{k}_dist"] = float(
                (g_val[k] - self.target_val[k]).item()
            )

        # 5) retrieval top-k (optional)
        if samples.get("candidates") and samples.get("batch_condition"):
            to_log.update(
                self._retrieval_metrics(
                    samples,
                    output_dir,
                    current_epoch,
                    local_rank,
                    verbose=log_each_molecule,
                )
            )

        # 6) dump SMILES
        if flag_test and local_rank == 0:
            file = Path(output_dir) / "graphs" / f"final_smiles_e_{current_epoch}.txt"
            file.parent.mkdir(parents=True, exist_ok=True)
            file.write_text("\n".join(s if s is not None else "" for s in all_smiles))
        return to_log

    # Vectorized retrieval top-k (same logic, streamlined)
    def _retrieval_metrics(
        self,
        samples: Dict[str, Any],
        output_dir: str,
        epoch: int,
        local_rank: int,
        *,
        verbose=False,
    ) -> Dict[str, float]:
        cand_lists = samples["candidates"]  # List[C][B]
        cand_X = samples["candidates_X"]  # List[C][B, n_max, d_x] or np arrays
        cand_E = samples["candidates_E"]  # List[C][B, n_max, n_max, d_e]
        cond_y = samples["batch_condition"]  # list(4) of tensors for NMR encoder
        atom_counts = samples["node_mask_meta"]  # [B]
        true_list = samples["true"]
        B, C = len(true_list), len(cand_lists)

        if isinstance(cand_X, list):
            cand_X = [paddle.to_tensor(x) for x in cand_X]
            cand_X = paddle.stack(cand_X, axis=0)  # [C,B,n_max,d_x]
            cand_E = [paddle.to_tensor(x) for x in cand_E]
            cand_E = paddle.stack(cand_E, axis=0)  # [C,B,n_max,n_max,d_e]

        # node mask
        n_max = int(paddle.max(paddle.stack(atom_counts)).item())
        arange = paddle.arange(n_max, dtype="int64")
        node_mask = arange.unsqueeze(0).expand([B, n_max]) < paddle.stack(
            atom_counts
        ).unsqueeze(1)

        with paddle.no_grad():
            # text/NMR embedding once
            nmr_emb = self.spectrumVec(cond_y)  # [B, d]
            # flatten candidates
            X_flat = cand_X.reshape([C * B, *cand_X.shape[2:]])  # [C·B,n_max,d_x]
            E_flat = cand_E.reshape([C * B, *cand_E.shape[2:]])  # [C·B,n_max,n_max,d_e]
            node_mask_flat = node_mask.tile([C, 1])  # [C·B, n_max]
            y_flat = paddle.zeros([C * B, 0], dtype=X_flat.dtype)

            z_t = (
                utils.PlaceHolder(X=X_flat, E=E_flat, y=y_flat)
                .type_as(X_flat)
                .mask(node_mask_flat)
            )

            extra = scheduling_diffnmr.compute_extra_data(
                self.clip,
                {"X_t": z_t.X, "E_t": z_t.E, "y_t": z_t.y, "node_mask": node_mask_flat},
                isPure=True,
            )
            X_in = paddle.concat([z_t.X.astype("float32"), extra.X], axis=2)
            E_in = paddle.concat([z_t.E.astype("float32"), extra.E], axis=3)
            y_in = paddle.concat([z_t.y.astype("float32"), extra.y], axis=1)

            mol_flat: paddle.Tensor = self.molVec(
                X_in, E_in, y_in, node_mask_flat
            )  # [C·B, d]
            mol_embs = mol_flat.reshape([C, B, -1])  # [C,B,d]

        sims = F.cosine_similarity(
            nmr_emb.unsqueeze(0).expand([C, -1, -1]), mol_embs, axis=-1
        )  # [C,B]
        max_idx = paddle.argmax(sims, axis=0)  # [B]
        top5_idx = paddle.topk(sims, k=min(5, C), axis=0)[1]  # [k,B]
        top10_idx = paddle.topk(sims, k=min(10, C), axis=0)[1]  # [k,B]

        hit1 = hit5 = hit10 = 0
        csv_records: List[Dict[str, str]] = []

        for i in range(B):
            m_true = mol_from_graphs(self.atom_decoder, *true_list[i])
            s_true = Chem.MolToSmiles(m_true, True)

            # top-1
            sel = int(max_idx[i])
            m_pred = mol_from_graphs(self.atom_decoder, *cand_lists[sel][i])
            s_pred = Chem.MolToSmiles(m_pred, True)
            if s_pred == s_true:
                hit1 += 1

            # top-5 / top-10
            for sel in top5_idx[:, i].astype("int64").tolist():
                if (
                    Chem.MolToSmiles(
                        mol_from_graphs(self.atom_decoder, *cand_lists[sel][i]), True
                    )
                    == s_true
                ):
                    hit5 += 1
                    break
            for sel in top10_idx[:, i].astype("int64").tolist():
                if (
                    Chem.MolToSmiles(
                        mol_from_graphs(self.atom_decoder, *cand_lists[sel][i]), True
                    )
                    == s_true
                ):
                    hit10 += 1
                    break

            # fingerprint sim for top-1
            try:
                sim = DataStructs.FingerprintSimilarity(
                    RDKFingerprint(m_pred), RDKFingerprint(m_true)
                )
            except Exception:
                sim = 0.0
            csv_records.append({"SMILES": s_true, "Similarity": f"{sim:.4f}"})
            if verbose:
                print(
                    f"[GT {i+1}/{B}] top1={'OK' if s_pred==s_true else 'NO'} "
                    f"sim={sim:.3f}"
                )

        if local_rank == 0:
            csv_path = Path(output_dir) / f"similarity_results_e{epoch}.csv"
            import pandas as pd

            pd.DataFrame(csv_records).to_csv(csv_path, index=False)

        ks = (1, 5, 10)
        hits = (hit1, hit5, hit10)
        return {
            f"retrieval_top{k}": h / len(true_list) for k, h in zip(ks, hits) if C >= k
        }

    def reset(self):
        for m in [
            self.gen_n,
            self.gen_nodes,
            self.gen_edges,
            self.gen_val,
            self.mae_n,
            self.mae_nodes,
            self.mae_edges,
            self.mae_val,
        ]:
            m.reset()


# =========================
# Unified adapter for build_metric
# =========================


class DiffNMRMetric:
    """
    __init_params__:
      mode: "train" | "sample"
      dataset_infos: required for both
      train_smiles: required for sample
      clip: optional for sample
      num_candidate: int, default 1
    """

    def __init__(
        self,
        mode: str,
        dataset_infos: Any = None,
        train_smiles: Optional[List[str]] = None,
        clip: Optional[nn.Layer] = None,
        num_candidate: int = 1,
    ):
        self.mode = mode
        self._dataset_infos = dataset_infos
        self._train_smiles = train_smiles
        self._clip = clip
        self._num_candidate = num_candidate
        self.impl = None

        # If the object is in the configuration, construct it directly.
        if self._ready():
            self._build_impl()

    def bind(self, *, dataset_infos=None, train_smiles=None, clip=None):
        """Inject real objects at runtime within the Trainer"""
        if dataset_infos is not None:
            self._dataset_infos = dataset_infos
        if train_smiles is not None:
            self._train_smiles = train_smiles
        if clip is not None:
            self._clip = clip
        if self.impl is None and self._ready():
            self._build_impl()

    def _ready(self) -> bool:
        if self.mode == "train":
            return self._dataset_infos is not None
        if self.mode == "sample":
            return (self._dataset_infos is not None) and (
                self._train_smiles is not None
            )
        return False

    def _build_impl(self):
        if self.mode == "train":
            self.impl = TrainMolecularMetricsDiscrete(self._dataset_infos)
        elif self.mode == "sample":
            self.impl = SamplingMolecularMetrics(
                self._dataset_infos, self._train_smiles, self._clip, self._num_candidate
            )
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def __call__(self, pred, label=None, **kwargs):
        if self.impl is None:
            raise RuntimeError(
                "DiffNMRMetric is not bound yet. Call .bind(dataset_infos=..., "
                "train_smiles=..., clip=...) before using."
            )
        if self.mode == "train":
            return self.impl(
                pred["masked_pred_X"],
                pred["masked_pred_E"],
                label["true_X"],
                label["true_E"],
                log=kwargs.get("log", False),
            )
        else:
            return self.impl(
                pred,
                kwargs.get("current_epoch", 0),
                kwargs.get("local_rank", 0),
                kwargs.get("output_dir", "."),
                kwargs.get("flag_test", False),
                kwargs.get("log_each_molecule", False),
            )

    def reset(self):
        if hasattr(self.impl, "reset") and self.impl is not None:
            self.impl.reset()

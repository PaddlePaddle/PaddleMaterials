# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to on the "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Adapted from https://github.com/divelab/AIRS (OpenMat/HIENet)

"""HIENet contract main class (three-layer forward protocol).

``forward`` dispatches, ``_forward`` splits the batch dict into tensors +
differentiable leaves and assembles the embedded three-term Huber loss,
``_runtime_forward`` (runtime boundary) runs the backbone numeric chain and
the force/stress differentiation.

Contract:
  - the ``force_output`` (ForceStressOutput) module is truncated from the
    backbone; force/stress come from ``paddle.grad`` on the truncated graph;
  - the strain leaf is created in ``_forward`` and passed down; EdgePreprocess
    reuses ``data["_strain"]`` when present instead of creating its own;
  - the main class always runs the batched data path (single samples get a
    synthesized zero batch vector);
  - the stress label transform lives here (converter stores the raw 3x3):
    negate + Voigt reorder [xx,yy,zz,xy,yz,xz], applied exactly once before
    the loss.
"""

import paddle

from ppmat.models.common.runtime import RuntimeMixin
from ppmat.models.common.runtime import runtime_boundary

from . import _keys as KEY
from . import model_build as _model_build  # late-bound import

TO_KB = 1602.1766208  # eV/A^3 -> kbar
_STRESS_VOIGT_3x3 = ((0, 0), (1, 1), (2, 2), (0, 1), (1, 2), (0, 2))
# Label keys accepted as the total-energy term (e.g. the raw dataset field
# name); any other key maps to itself.
_ENERGY_LABEL_ALIASES = ("energy", "uncorrected_total_energy")


class HIENet(RuntimeMixin, paddle.nn.Layer):
    """HIENet potential with embedded loss (energy/force/stress)."""

    def __init__(
        self,
        config: dict,
        property_names=("energy", "force", "stress"),
        loss_type: str = "huber",
        huber_loss_delta: float = 0.01,
        loss_weights_dict: dict | None = None,
        execution_backend: str = "eager",
        runtime_options: dict | None = None,
    ):
        super().__init__()
        self._init_runtime(execution_backend, runtime_options)
        self.config = config
        self.cutoff = config[KEY.CUTOFF]
        self.type_map = config[KEY.TYPE_MAP]
        self.property_names = list(property_names)
        self.loss_type = loss_type
        self.huber_loss_delta = huber_loss_delta
        if loss_type in ("mse_loss", "mse"):
            self.loss_fn = paddle.nn.MSELoss()
        elif loss_type in ("huber", "smooth_l1_loss"):
            self.loss_fn = paddle.nn.HuberLoss(delta=huber_loss_delta)
        elif loss_type == "l1_loss":
            self.loss_fn = paddle.nn.L1Loss()
        else:
            raise ValueError(f"Unknown loss type {loss_type}.")
        if loss_weights_dict is None:
            loss_weights_dict = {"energy": 1.0, "force": 1.0, "stress": 0.01}
        self.loss_weights_dict = loss_weights_dict
        backbone = _model_build.build_E3_equivariant_model(config)
        # Force/stress are derived in _runtime_forward via paddle.grad on
        # the truncated graph.
        backbone.delete_module_by_key("force_output")
        # The contract layer always runs the batched data path.
        backbone.set_is_batch_data(True)
        self.backbone = backbone
        self._param_dtype = self.backbone.parameters()[0].dtype
        # Atomic-number -> one-hot index lookup table (built once at
        # construction; pure-tensor gather at runtime, no host round-trip).
        lut = [0] * (max(self.type_map) + 1)
        for z, idx in self.type_map.items():
            lut[z] = idx
        self._type_lut_data = lut

    def forward(self, data, return_loss=True, return_prediction=True):
        assert (
            return_loss or return_prediction
        ), "At least one of return_loss or return_prediction must be True."
        out = self._forward(data)
        if not return_loss:
            out["loss_dict"] = {}
        if not return_prediction:
            out["pred_dict"] = {}
        return out

    def predict(self, samples):
        is_list = isinstance(samples, list)
        samples = samples if is_list else [samples]
        results = []
        for sample in samples:
            result = self.forward(
                {"graph": sample}, return_loss=False, return_prediction=True
            )
            results.append({k: v.numpy() for k, v in result["pred_dict"].items()})
        return results if is_list else results[0]

    def _forward(self, data):
        """Split Batch/Data into tensors + leaves, run the chain, assemble
        the embedded loss and the prediction dict."""
        graph = data["graph"]

        def _tensor(v):
            if isinstance(v, paddle.Tensor):
                return v
            return paddle.to_tensor(v)

        # Inputs follow the current global device (no forced .cpu()): on GPU
        # this keeps one pure-device numeric graph; on CPU it is a no-op.
        cart = _tensor(graph["cart_coords"])
        lattice = _tensor(graph["lattice"])
        edge_index = _tensor(graph["edge_index"]).cast("int64")
        pbc_offset = _tensor(graph["pbc_offset"]).cast("int64")
        atom_types = _tensor(graph["atom_types"]).cast("int64")
        num_atoms = _tensor(graph["num_atoms"]).cast("int64")
        batch_vec = getattr(graph, "batch", None)
        if batch_vec is None:
            num_graphs = 1
            batch_idx = paddle.zeros([cart.shape[0]], dtype="int64")
        else:
            batch_idx = _tensor(batch_vec).cast("int64")
            num_graphs = int(graph.num_graphs)

        # Float graph inputs follow the backbone parameter dtype.
        if cart.dtype != self._param_dtype:
            cart = cart.cast(self._param_dtype)
        if lattice.dtype != self._param_dtype:
            lattice = lattice.cast(self._param_dtype)

        # Differentiable leaves.
        if "force" in self.property_names:
            cart = cart.detach()
            cart.stop_gradient = False
        strain_leaf = None
        if "stress" in self.property_names:
            strain_leaf = paddle.zeros([num_graphs, 3, 3], dtype=cart.dtype)
            strain_leaf.stop_gradient = False

        # Pure-tensor one-hot index mapping; a .numpy() + Python loop
        # would force a GPU->CPU sync per step.
        x_idx = paddle.gather(
            paddle.to_tensor(
                self._type_lut_data, dtype="int64", place=atom_types.place
            ),
            atom_types,
        )

        energy, force, stress = self._runtime_forward(
            cart_coords=cart,
            strain_leaf=strain_leaf,
            atom_types_onehot_idx=x_idx,
            edge_index=edge_index,
            pbc_offset=pbc_offset,
            lattice=lattice,
            batch_idx=batch_idx,
            num_graphs=num_graphs,
            num_atoms_per_graph=num_atoms,
        )
        pred_dict = {"energy": energy, "force": force, "stress": stress}

        loss_dict = {}
        total_loss = 0.0
        for prop in self.property_names:
            label_raw = data.get(prop, None)
            if label_raw is None:
                continue
            # Dataset-side key aliases (e.g. the raw MPtrj energy field)
            # feed the standard model outputs below.
            pred_key = "energy" if prop in _ENERGY_LABEL_ALIASES else prop
            pred = pred_dict[pred_key]
            label = _tensor(label_raw)
            if pred_key == "energy":
                # Per-atom energy: both sides divided by the atom count
                # (cast to pred dtype).
                label = label.reshape([num_graphs]).cast(pred.dtype)
                per_atom = num_atoms.cast(pred.dtype)
                pred = pred.reshape([num_graphs]) / per_atom
                label = label / per_atom
            elif pred_key == "stress":
                # Raw 3x3 label -> model convention (negate + Voigt
                # [xx,yy,zz,xy,yz,xz]); the single loss-side transform.
                label = label.cast(pred.dtype)
                label = -paddle.stack(
                    [label[:, i, j] for (i, j) in _STRESS_VOIGT_3x3], axis=1
                )
                pred = (pred * TO_KB).reshape([-1])
                label = (label * TO_KB).reshape([-1])
            else:
                label = label.cast(pred.dtype)
                pred = pred.reshape([-1])
                label = label.reshape([-1])
            valid = ~paddle.isnan(label)
            valid_pred = pred[valid]
            valid_label = label[valid]
            if valid_label.numel() > 0:
                loss_prop = self.loss_fn(valid_pred, valid_label)
                loss_dict[pred_key] = loss_prop
                total_loss = total_loss + loss_prop * self.loss_weights_dict[pred_key]
        loss_dict["loss"] = total_loss
        return {"loss_dict": loss_dict, "pred_dict": pred_dict}

    @runtime_boundary("forward")
    def _runtime_forward(
        self,
        cart_coords,
        strain_leaf,
        atom_types_onehot_idx,
        edge_index,
        pbc_offset,
        lattice,
        batch_idx,
        num_graphs,
        num_atoms_per_graph,
    ):
        """Run the backbone numeric chain and differentiate the total energy
        w.r.t. the differentiable leaves (one numerical graph)."""
        volume = paddle.sum(
            lattice[:, 0] * paddle.cross(lattice[:, 1], lattice[:, 2], axis=-1),
            axis=-1,
        )
        data = {
            KEY.NODE_FEATURE: atom_types_onehot_idx,
            KEY.POS: cart_coords,
            KEY.CELL: lattice,
            KEY.CELL_SHIFT: pbc_offset,
            KEY.EDGE_IDX: edge_index,
            KEY.BATCH: batch_idx,
            KEY.NUM_ATOMS: num_atoms_per_graph,
            KEY.CELL_VOLUME: volume,
        }
        if strain_leaf is not None:
            data["_strain"] = strain_leaf
        data = self.backbone(data)
        energy = data[KEY.PRED_TOTAL_ENERGY]
        e_sum = paddle.sum(energy)

        # The same graph survives the force call when stress follows.
        if "force" in self.property_names:
            force = -paddle.grad(
                e_sum,
                cart_coords,
                create_graph=self.training,
                retain_graph=self.training or "stress" in self.property_names,
            )[0]
        else:
            force = paddle.zeros_like(cart_coords)

        if "stress" in self.property_names and strain_leaf is not None:
            # HIENet stress convention: -dE/dstrain / volume, Voigt
            # [xx,yy,zz,xy,yz,xz].
            sgrad = paddle.grad(
                e_sum,
                strain_leaf,
                create_graph=self.training,
                retain_graph=True,
            )[0]
            stress = paddle.neg(sgrad / volume.reshape([-1, 1, 1]))
            stress = paddle.stack(
                [stress[:, i, j] for (i, j) in _STRESS_VOIGT_3x3], axis=1
            )
        else:
            stress = paddle.zeros([num_graphs, 6], dtype=energy.dtype)
        return energy, force, stress

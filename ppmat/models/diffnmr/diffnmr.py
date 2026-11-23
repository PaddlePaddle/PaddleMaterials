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

import copy
import random

import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from einops import rearrange
from einops import repeat
from rdkit import Chem
from tqdm import tqdm

from ppmat.losses.diffnmr_loss import TrainLossDiscrete
from ppmat.metrics.diffnmr_metric import NLL
from ppmat.metrics.diffnmr_metric import SumExceptBatchKL
from ppmat.metrics.diffnmr_metric import SumExceptBatchMetric
from ppmat.models.common import initializer
from ppmat.models.common.runtime import RuntimeMixin
from ppmat.models.diffnmr.diffusion_prior import DiffPriorNetwork
from ppmat.models.diffnmr.extra_features_graph import DummyExtraFeatures
from ppmat.models.diffnmr.extra_features_graph import ExtraFeatures
from ppmat.models.diffnmr.extra_features_molecular_graph import ExtraMolecularFeatures
from ppmat.models.diffnmr.graph_transformer import GraphTransformer
from ppmat.models.diffnmr.graph_transformer import MolecularEncoder
from ppmat.models.diffnmr.nmr_encoder import NMR_encoder
from ppmat.models.diffnmr.nmr_encoder import NMR_encoder_H
from ppmat.models.diffnmr.utils import diffgraphformer_utils
from ppmat.models.diffnmr.utils.diffprior_utils import default
from ppmat.models.diffnmr.utils.diffprior_utils import exists
from ppmat.models.diffnmr.utils.diffprior_utils import freeze_model_and_make_eval_
from ppmat.models.diffnmr.utils.diffprior_utils import l2norm
from ppmat.schedulers import scheduling_diffnmr
from ppmat.schedulers.scheduling_diffnmr import DiscreteUniformTransition
from ppmat.schedulers.scheduling_diffnmr import MarginalUniformTransition
from ppmat.schedulers.scheduling_diffnmr import PredefinedNoiseScheduleDiscrete
from ppmat.schedulers.scheduling_diffprior import NoiseScheduler
from ppmat.utils import logger


def _build_graph_features(dataset_infos, diffmodel_cfg):
    """Build model-owned graph features and their input/output dimensions."""
    feature_type = diffmodel_cfg.get("extra_features")
    extra_features = (
        ExtraFeatures(feature_type, dataset_infos)
        if feature_type is not None
        else DummyExtraFeatures()
    )
    domain_features = ExtraMolecularFeatures(dataset_infos)

    output_dims = {
        "X": dataset_infos.num_atom_types,
        "E": int(dataset_infos.edge_types.shape[0]),
        "y": 0,
    }
    input_dims = copy.deepcopy(output_dims)
    input_dims["y"] += 1 + int(diffmodel_cfg.get("conditdim", 0))

    for features in (extra_features, domain_features):
        for field, size in features.output_dims.items():
            input_dims[field] += size
    return extra_features, domain_features, input_dims, output_dims


class _DiffNMRSamplingMixin:
    @paddle.no_grad()
    def sample(
        self,
        batch,
        *,
        batch_id=0,
        number_chain_steps=0,
        keep_chain=0,
        iter_idx=0,
        flag_useformula=False,
        return_onehot=False,
        retrieval_initialization=False,
        clip=None,
        molecular_vectors=None,
        smiles_list=None,
    ):
        """Generate molecular graphs using the molecular-sampler output contract."""

        graph = batch["graph"]
        property_data = batch["property"]
        dense_data, node_mask = diffgraphformer_utils.to_dense(
            paddle.to_tensor(graph.node_feat["feat"]),
            paddle.to_tensor(graph.edges.T),
            paddle.to_tensor(graph.edge_feat["feat"]),
            paddle.to_tensor(graph.graph_node_id),
        )
        dense_data = dense_data.mask(node_mask)
        batch_X, batch_E = dense_data.X, dense_data.E
        batch_y = paddle.to_tensor(property_data["y"])
        n_nodes = paddle.to_tensor(property_data["atom_count"], dtype="int64").reshape(
            [-1]
        )
        batch_size = int(batch_X.shape[0])
        n_max = int(paddle.max(n_nodes).item())

        # The graph batch can use a wider shared padding capacity than this
        # atom-count vector. Reverse diffusion follows the declared valid range.
        batch_X = batch_X[:, :n_max]
        batch_E = batch_E[:, :n_max, :n_max]
        node_mask = node_mask[:, :n_max]

        if self.conditioning_mode == "spectrum":
            spectrum = batch["spectrum"]
            batch_condition = [
                paddle.to_tensor(spectrum["H_nmr"]),
                paddle.to_tensor(spectrum["num_H_peak"]),
                paddle.to_tensor(spectrum["C_nmr"]),
                paddle.to_tensor(spectrum["num_C_peak"]),
            ]
        else:
            batch_condition = None

        keep_chain = min(int(keep_chain), batch_size)
        z_t = scheduling_diffnmr.sample_discrete_feature_noise(
            limit_dist=self.limit_dist, node_mask=node_mask
        )
        X_t, E_t, y_t = z_t.X, z_t.E, z_t.y

        if keep_chain:
            chain_X = paddle.zeros(
                [number_chain_steps, keep_chain, n_max], dtype="int64"
            )
            chain_E = paddle.zeros(
                [number_chain_steps, keep_chain, n_max, n_max], dtype="int64"
            )

        if retrieval_initialization:
            output = clip.spectrum_encoder(batch_condition)
            if isinstance(output, (tuple, list)):
                output = output[0]

            similarities = []
            for start in range(0, molecular_vectors.shape[0], 128):
                vectors = molecular_vectors[start : start + 128]
                similarities.append(
                    F.cosine_similarity(
                        output.unsqueeze(1), vectors.unsqueeze(0), axis=-1
                    )
                )
            top_indices = paddle.topk(paddle.concat(similarities, axis=1), k=1, axis=1)[
                1
            ]

            atom_encoder = self.vocab["atom"]["token_to_id"]
            bond_encoder = self.vocab["bond"]["token_to_id"]
            bond_types = {
                Chem.rdchem.BondType.SINGLE: bond_encoder["SINGLE"],
                Chem.rdchem.BondType.DOUBLE: bond_encoder["DOUBLE"],
                Chem.rdchem.BondType.TRIPLE: bond_encoder["TRIPLE"],
                Chem.rdchem.BondType.AROMATIC: bond_encoder["AROMATIC"],
            }
            for sample_index in range(batch_size):
                mol = Chem.MolFromSmiles(
                    smiles_list[int(top_indices[sample_index].item())]
                )
                if mol is None:
                    continue

                node_ids = [atom_encoder[atom.GetSymbol()] for atom in mol.GetAtoms()]
                written_nodes = min(len(node_ids), int(n_nodes[sample_index].item()))
                if written_nodes:
                    X_t[sample_index, :written_nodes] = F.one_hot(
                        paddle.to_tensor(node_ids[:written_nodes], dtype="int64"),
                        num_classes=self.vocab["atom"]["num_embeddings"],
                    ).astype("float32")

                adjacency = np.full((n_max, n_max), -1, dtype="int64")
                adjacency[:written_nodes, :written_nodes] = bond_encoder["NO_BOND"]
                for bond in mol.GetBonds():
                    start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                    if start < written_nodes and end < written_nodes:
                        value = bond_types.get(
                            bond.GetBondType(), bond_encoder["NO_BOND"]
                        )
                        adjacency[start, end] = value
                        adjacency[end, start] = value
                adjacency = paddle.to_tensor(adjacency, dtype="int64")
                edge_hot = F.one_hot(
                    adjacency.clip(min=0),
                    num_classes=self.vocab["bond"]["num_embeddings"],
                ).astype("float32")
                edge_hot[adjacency == -1] = 0
                E_t[sample_index, :written_nodes, :written_nodes] = edge_hot[
                    :written_nodes, :written_nodes
                ]

        for step_index in tqdm(
            range(self.T - 1, -1, -1),
            desc=f"Batch {batch_id} RepeatIter {iter_idx} sampling {self.T}→0",
            unit="step",
        ):
            s_array = paddle.full([batch_size, 1], float(step_index))
            sampled_s, discrete_sampled_s = scheduling_diffnmr.step(
                self,
                s=s_array / self.T,
                t=(s_array + 1.0) / self.T,
                X_t=X_t,
                E_t=E_t,
                y_t=y_t,
                node_mask=node_mask,
                conditionVec=batch_condition,
                batch_X=batch_X,
                batch_E=batch_E,
                batch_y=batch_y,
            )
            X_t, E_t, y_t = sampled_s.X, sampled_s.E, sampled_s.y
            if flag_useformula:
                X_t = batch_X
            if keep_chain:
                write_index = (step_index * number_chain_steps) // self.T
                chain_X[write_index] = discrete_sampled_s.X[:keep_chain]
                chain_E[write_index] = discrete_sampled_s.E[:keep_chain]

        sampled_collapse = copy.deepcopy(sampled_s).mask(node_mask, collapse=True)
        X_idx, E_idx = sampled_collapse.X, sampled_collapse.E
        if flag_useformula:
            X_idx = paddle.argmax(batch_X, axis=-1)

        if return_onehot:
            X_hot = sampled_s.mask(node_mask).X.numpy()
            E_hot = sampled_s.mask(node_mask).E.numpy()
            if flag_useformula:
                X_hot = batch_X.numpy()
            X_hot = [X_hot[index] for index in range(batch_size)]
            E_hot = [E_hot[index] for index in range(batch_size)]

        n_nodes_array = n_nodes.numpy()
        batch_X_idx = paddle.argmax(batch_X, axis=-1).numpy()
        batch_E_idx = paddle.argmax(batch_E, axis=-1).numpy()
        molecule_list, molecule_true = [], []
        for index in range(batch_size):
            count = int(n_nodes_array[index])
            molecule_list.append(
                [X_idx[index, :count].numpy(), E_idx[index, :count, :count].numpy()]
            )
            molecule_true.append(
                [batch_X_idx[index, :count], batch_E_idx[index, :count, :count]]
            )

        chains = []
        if keep_chain:
            chain_X[0] = X_idx[:keep_chain]
            chain_E[0] = E_idx[:keep_chain]
            chain_X = scheduling_diffnmr.reverse_tensor(chain_X)
            chain_E = scheduling_diffnmr.reverse_tensor(chain_E)
            chains = [
                [chain_X[:, index].numpy(), chain_E[:, index].numpy()]
                for index in range(chain_X.shape[1])
            ]

        result = {
            "pred": molecule_list,
            "true": molecule_true,
            "num_samples": batch_size,
            "node_counts": n_nodes,
            "conditions": batch_condition,
            "atom_decoder": self.dataset_info.atom_decoder,
            "chains": chains,
        }
        if return_onehot:
            result["node_onehot"] = X_hot
            result["edge_onehot"] = E_hot
        return result


class MolecularGraphFormer(_DiffNMRSamplingMixin, RuntimeMixin, nn.Layer):
    def __init__(
        self,
        encoder_cfg,
        decoder_cfg,
        diffmodel_cfg,
        dataset_infos=None,
        vocab=None,
        execution_backend="eager",
        runtime_options=None,
    ) -> None:
        super().__init__()
        self._init_runtime(execution_backend, runtime_options)

        # configure general variables settings
        self.T = diffmodel_cfg["diffusion_steps"]

        # configure datasets inter-varibles
        (
            self.extra_features,
            self.domain_features,
            input_dims,
            output_dims,
        ) = _build_graph_features(dataset_infos, diffmodel_cfg)
        self.dataset_info = dataset_infos
        self.vocab = vocab
        # This model uses the clean molecular graph as its condition.  The
        # sampler uses this capability flag to avoid treating it as an NMR
        # encoder model.
        self.conditioning_mode = "graph"

        # configure noise scheduler
        self.noise_schedule = PredefinedNoiseScheduleDiscrete(
            diffmodel_cfg["diffusion_noise_schedule"],
            timesteps=self.T,
        )

        # configure model
        self.con_input_dim = copy.deepcopy(input_dims)
        self.con_input_dim["y"] = 12
        self.con_output_dim = output_dims

        self.encoder = MolecularEncoder(
            n_layers=encoder_cfg["num_layers"],
            input_dims=self.con_input_dim,
            hidden_mlp_dims=encoder_cfg["hidden_mlp_dims"],
            hidden_dims=encoder_cfg["hidden_dims"],
            output_dims=self.con_output_dim,
            act_fn_in=nn.ReLU(),
            act_fn_out=nn.ReLU(),
        )

        self.decoder = GraphTransformer(
            n_layers=decoder_cfg["num_layers"],
            input_dims=input_dims,
            hidden_mlp_dims=decoder_cfg["hidden_mlp_dims"],
            hidden_dims=decoder_cfg["hidden_dims"],
            output_dims=output_dims,
            act_fn_in=nn.ReLU(),
            act_fn_out=nn.ReLU(),
        )

        # configure loss calculation with initialization of transition model
        self.Xdim = input_dims["X"]
        self.Edim = input_dims["E"]
        self.ydim = input_dims["y"]
        self.Xdim_output = output_dims["X"]
        self.Edim_output = output_dims["E"]
        self.ydim_output = output_dims["y"]
        self.node_dist = dataset_infos.nodes_dist

        # Transition Model
        if diffmodel_cfg["transition"] == "uniform":
            self.transition_model = DiscreteUniformTransition(
                x_classes=self.Xdim_output,
                e_classes=self.Edim_output,
                y_classes=self.ydim_output,
            )
            x_limit = paddle.ones([self.Xdim_output]) / self.Xdim_output
            e_limit = paddle.ones([self.Edim_output]) / self.Edim_output
            y_limit = paddle.ones([self.ydim_output]) / self.ydim_output
            self.limit_dist = diffgraphformer_utils.PlaceHolder(
                X=x_limit, E=e_limit, y=y_limit
            )

        elif diffmodel_cfg["transition"] == "marginal":
            node_types = self.dataset_info.node_types.astype("float32")
            x_marginals = node_types / paddle.sum(node_types)

            edge_types = self.dataset_info.edge_types.astype("float32")
            e_marginals = edge_types / paddle.sum(edge_types)
            logger.info("Marginal distribution of classes:")
            logger.info(f"{x_marginals.tolist()} for nodes")
            logger.info(f"{e_marginals.tolist()} for edges")

            self.transition_model = MarginalUniformTransition(
                x_marginals=x_marginals,
                e_marginals=e_marginals,
                y_classes=self.ydim_output,
            )
            self.limit_dist = diffgraphformer_utils.PlaceHolder(
                X=x_marginals,
                E=e_marginals,
                y=paddle.ones([self.ydim_output]) / self.ydim_output,
            )

        # configure loss
        self.train_loss = TrainLossDiscrete(diffmodel_cfg["lambda_train"])

        # configure training setting and other properties
        self.best_val_nll = 1e8
        self.val_counter = 0

        # set use formula for training and sample or not
        self.flag_use_formula = diffmodel_cfg.get("flag_use_formula", False)

    def run_encoder(self, *args):
        return self._run_runtime("graph_encoder", self.encoder, *args)

    def run_decoder(self, *args):
        return self._run_runtime("denoise_step", self.decoder, *args)

    def forward(self, batch):
        batch_graph = batch["graph"]
        batch_property = batch["property"]

        # 1. Convert sparse graph to dense tensors and apply node mask
        dense_data, node_mask = diffgraphformer_utils.to_dense(
            paddle.to_tensor(batch_graph.node_feat["feat"]),
            paddle.to_tensor(batch_graph.edges.T),
            paddle.to_tensor(batch_graph.edge_feat["feat"]),
            paddle.to_tensor(batch_graph.graph_node_id),
        )
        dense_data = dense_data.mask(node_mask)
        X, E = dense_data.X, dense_data.E
        y = paddle.to_tensor(batch_property["y"])

        # 2. Add noise and compute extra features
        noisy_data = scheduling_diffnmr.apply_noise(
            self, X, E, y, node_mask, self.flag_use_formula
        )
        extra_data = scheduling_diffnmr.compute_extra_data(self, noisy_data)

        # Decoder inputs (noisy + extra features)
        input_X = paddle.concat(
            [noisy_data["X_t"].astype("float32"), extra_data.X], axis=2
        ).astype(dtype="float32")
        input_E = paddle.concat(
            [noisy_data["E_t"].astype("float32"), extra_data.E], axis=3
        ).astype(dtype="float32")
        input_y = paddle.hstack(
            [noisy_data["y_t"].astype("float32"), extra_data.y]
        ).astype(dtype="float32")

        # 3. Encoder condition vector from clean inputs + extra features
        z_t = (
            diffgraphformer_utils.PlaceHolder(X=X, E=E, y=y).type_as(X).mask(node_mask)
        )
        extra_data_pure = scheduling_diffnmr.compute_extra_data(
            self,
            {"X_t": z_t.X, "E_t": z_t.E, "y_t": z_t.y, "node_mask": node_mask},
            isPure=True,
        )
        input_X_pure = paddle.concat(
            [z_t.X.astype("float32"), extra_data_pure.X], axis=2
        ).astype(dtype="float32")
        input_E_pure = paddle.concat(
            [z_t.E.astype("float32"), extra_data_pure.E], axis=3
        ).astype(dtype="float32")
        input_y_pure = paddle.hstack(
            x=(z_t.y.astype("float32"), extra_data_pure.y)
        ).astype(dtype="float32")

        # obtain the condition vector from output of encoder
        conditionVec = self.run_encoder(
            input_X_pure, input_E_pure, input_y_pure, node_mask
        )
        # complete input_y for decoder
        input_y = paddle.hstack(x=(input_y, conditionVec)).astype(dtype="float32")

        # 4. Decoder forward
        # Convention: pred.X and pred.E are logits with shapes [B, n, Cx] and
        # [B, n, n, Ce]
        pred = self.run_decoder(input_X, input_E, input_y, node_mask)

        # 5. Compute training loss
        loss_dict = self.train_loss(
            masked_pred_X=pred.X,
            masked_pred_E=pred.E,
            pred_y=pred.y,
            true_X=X,
            true_E=E,
            true_y=paddle.to_tensor(batch_property["y"]),
        )

        # 6. Assemble outputs for Trainer & streaming metrics
        # Predictions: provide masked_pred_X/E; mirror X_logits/E_logits for legacy
        # paths
        pred_dict = {
            "masked_pred_X": pred.X,
            "masked_pred_E": pred.E,
            "pred_y": pred.y,
        }
        # Labels: provide true_X/true_E; node_mask is optional but useful for some
        # metrics
        label_dict = {
            "true_X": X,
            "true_E": E,
            "true_y": y,
        }

        result = {
            "loss_dict": loss_dict,
            "pred_dict": pred_dict,
            "label_dict": label_dict,
            "node_mask": node_mask,
            "noisy_data": noisy_data,
        }

        return result


class NMRNetCLIP(RuntimeMixin, nn.Layer):
    def __init__(
        self,
        graph_encoder: dict,
        spectrum_encoder: dict,
        vocab,
        dataset_infos=None,
        diffmodel_cfg=None,
        **kwargs,
    ):
        super().__init__()
        self._init_runtime(
            kwargs.pop("execution_backend", "eager"),
            kwargs.pop("runtime_options", None),
        )
        self.name = kwargs.get("__name__")
        self.vocab = vocab
        peakwidthemb_num = vocab["peakwidth"]["num_embeddings"]
        splitemb_num = vocab["split"]["num_embeddings"]
        integralemb_num = vocab["integral"]["num_embeddings"]

        self.dataset_info = dataset_infos
        (
            self.extra_features,
            self.domain_features,
            input_dims,
            output_dims,
        ) = _build_graph_features(dataset_infos, diffmodel_cfg or {})
        self.con_input_dim = copy.deepcopy(input_dims)
        self.con_input_dim["y"] = 12
        self.con_output_dim = output_dims

        self.graph_encoder = MolecularEncoder(
            n_layers=graph_encoder["n_layers_GT"],
            input_dims=self.con_input_dim,
            hidden_mlp_dims=graph_encoder["hidden_mlp_dims"],
            hidden_dims=graph_encoder["hidden_dims"],
            output_dims=self.con_output_dim,
            act_fn_in=paddle.nn.ReLU(),
            act_fn_out=paddle.nn.ReLU(),
        )
        if graph_encoder["pretrained_model_path"] is not None:
            # load graph encoder model from pretrained model
            state_dict = paddle.load(graph_encoder["pretrained_model_path"])
            encoder_state_dict = {
                k[len("encoder.") :]: v
                for k, v in state_dict.items()
                if k.startswith("encoder.")
            }
            self.graph_encoder.set_state_dict(encoder_state_dict)
        for param in self.graph_encoder.parameters():
            param.stop_gradient = True
        self.graph_encoder.eval()

        if kwargs.get("onlyH", False):
            self.flag_onlyH = True
            self.spectrum_encoder = NMR_encoder_H(
                dim_H=spectrum_encoder["dim_enc_H"],
                dimff_H=spectrum_encoder["dimff_enc_H"],
                dim_C=spectrum_encoder["dim_enc_C"],
                dimff_C=spectrum_encoder["dimff_enc_C"],
                hidden_dim=spectrum_encoder["ffn_hidden"],
                n_head=spectrum_encoder["n_head"],
                num_layers=spectrum_encoder["n_layers"],
                drop_prob=spectrum_encoder["drop_prob"],
                peakwidthemb_num=peakwidthemb_num,
                splitemb_num=splitemb_num,
                integralemb_num=integralemb_num,
            )
        else:
            self.flag_onlyH = False
            self.spectrum_encoder = NMR_encoder(
                dim_H=spectrum_encoder["dim_enc_H"],
                dimff_H=spectrum_encoder["dimff_enc_H"],
                dim_C=spectrum_encoder["dim_enc_C"],
                dimff_C=spectrum_encoder["dimff_enc_C"],
                hidden_dim=spectrum_encoder["ffn_hidden"],
                n_head=spectrum_encoder["n_head"],
                num_layers=spectrum_encoder["n_layers"],
                drop_prob=spectrum_encoder["drop_prob"],
                peakwidthemb_num=peakwidthemb_num,
                splitemb_num=splitemb_num,
                integralemb_num=integralemb_num,
            )
        # for init model weights
        self.spectrum_encoder.apply(self._init_weights)

        self.seq_len_H1 = dataset_infos.seq_len_H1
        self.seq_len_C13 = dataset_infos.seq_len_C13
        self.tem = 2  # TODO remove later

        # for Prior Training
        if (
            "pretrained_model_path" in spectrum_encoder
            and spectrum_encoder["pretrained_model_path"] is not None
        ):
            # load graph encoder model from pretrained model
            state_dict = paddle.load(spectrum_encoder["pretrained_model_path"])
            encoder_state_dict = {
                k[len("spectrum_encoder.") :]: v
                for k, v in state_dict.items()
                if k.startswith("spectrum_encoder.")
            }
            # encoder_state_dict = {
            #     k[len("encoder.") :]: v
            #     for k, v in state_dict.items()
            #     if k.startswith("encoder.")
            # } # TODO: prior training, revise it and check
            self.spectrum_encoder.set_state_dict(encoder_state_dict)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            initializer.linear_init_(m)
        elif isinstance(m, nn.Embedding):
            initializer.normal_(m.weight)

    def forward(self, batch):
        batch_graph = batch["graph"]
        batch_property = batch["property"]
        batch_spectrum = batch["spectrum"]

        batch_length = batch_graph.num_graph
        # transfer to dense graph from sparse graph
        if paddle.to_tensor(batch_graph.edges.T).numel() == 0:
            print("Found a batch with no edges. Skipping.")
            return None
        dense_data, node_mask = diffgraphformer_utils.to_dense(
            paddle.to_tensor(batch_graph.node_feat["feat"]),
            paddle.to_tensor(batch_graph.edges.T).contiguous(),
            paddle.to_tensor(batch_graph.edge_feat["feat"]),
            paddle.to_tensor(batch_graph.graph_node_id),
        )
        dense_data = dense_data.mask(node_mask)
        X, E = dense_data.X, dense_data.E
        y = paddle.to_tensor(batch_property["y"])

        # get NMR embedded vector
        # prepare NMR vectors
        condition_H1nmr = paddle.to_tensor(batch_spectrum["H_nmr"]).reshape(
            batch_length, self.seq_len_H1, -1
        )
        condition_C13nmr = paddle.to_tensor(batch_spectrum["C_nmr"]).reshape(
            batch_length, self.seq_len_C13
        )
        num_H_peak = paddle.to_tensor(batch_spectrum["num_H_peak"])
        num_C_peak = paddle.to_tensor(batch_spectrum["num_C_peak"])
        conditionAll = [condition_H1nmr, num_H_peak, condition_C13nmr, num_C_peak]
        if self.flag_onlyH is True:
            global_H, _ = self._run_runtime(
                "spectrum_encoder", self.spectrum_encoder, conditionAll
            )
            condition_nmr = global_H
        else:
            condition_nmr = self._run_runtime(
                "spectrum_encoder", self.spectrum_encoder, conditionAll
            )

        # get graph embedded vector
        # prepare the extra feature for encoder input without noisy
        z_t = (
            diffgraphformer_utils.PlaceHolder(X=X, E=E, y=y).type_as(X).mask(node_mask)
        )
        extra_data_pure = scheduling_diffnmr.compute_extra_data(
            self,
            {"X_t": z_t.X, "E_t": z_t.E, "y_t": z_t.y, "node_mask": node_mask},
            isPure=True,
        )
        # prepare the input data for encoder combining extra features
        input_X_pure = paddle.concat(
            [z_t.X.astype("float32"), extra_data_pure.X], axis=2
        ).astype(dtype="float32")
        input_E_pure = paddle.concat(
            [z_t.E.astype("float32"), extra_data_pure.E], axis=3
        ).astype(dtype="float32")
        input_y_pure = paddle.hstack(
            x=(z_t.y.astype("float32"), extra_data_pure.y)
        ).astype(dtype="float32")
        # obtain the condition vector from output of encoder
        condition_graph = self._run_runtime(
            "graph_encoder",
            self.graph_encoder,
            input_X_pure,
            input_E_pure,
            input_y_pure,
            node_mask,
        )

        # compute similarity between graph and NMR
        V1_f = condition_graph  # Assuming V1 is a feature obtained from molecular graph
        V2_f, _ = condition_nmr  # Assume V2 is a feature obtained from NMR spectrum
        V1_e = paddle.nn.functional.normalize(x=V1_f, p=2, axis=1)
        V2_e = paddle.nn.functional.normalize(x=V2_f, p=2, axis=1)
        logits = paddle.matmul(x=V1_e, y=V2_e.T) * paddle.exp(
            x=paddle.to_tensor(data=self.tem)
        )
        n = V1_f.shape[0]
        labels = paddle.arange(end=n)
        loss_fn = paddle.nn.CrossEntropyLoss()
        loss_v1 = loss_fn(logits, labels)
        loss_v2 = loss_fn(logits.T, labels)
        loss = (loss_v1 + loss_v2) / 2

        loss_dict = {"loss": loss}

        return {"loss_dict": loss_dict}


class DiffNMR(_DiffNMRSamplingMixin, RuntimeMixin, nn.Layer):
    def __init__(
        self,
        encoder_cfg,
        decoder_cfg,
        diffmodel_cfg,
        dataset_infos,
        vocab,
        clip=None,
        connector_cfg=None,
        execution_backend="eager",
        runtime_options=None,
    ) -> None:
        super().__init__()
        self._init_runtime(execution_backend, runtime_options)
        self.vocab = vocab
        peakwidthemb_num = vocab["peakwidth"]["num_embeddings"]
        splitemb_num = vocab["split"]["num_embeddings"]
        integralemb_num = vocab["integral"]["num_embeddings"]

        # configure general variables settings
        self.conditioning_mode = "spectrum"
        self.T = diffmodel_cfg["diffusion_steps"]

        # configure datasets inter-varibles
        (
            self.extra_features,
            self.domain_features,
            input_dims,
            output_dims,
        ) = _build_graph_features(dataset_infos, diffmodel_cfg)
        self.dataset_info = dataset_infos

        # configure noise scheduler
        self.noise_schedule = PredefinedNoiseScheduleDiscrete(
            diffmodel_cfg["diffusion_noise_schedule"],
            timesteps=self.T,
        )

        # set spectrum encoder model
        if encoder_cfg.get("onlyH", False):
            self.flag_onlyH = True
            self.encoder = NMR_encoder_H(
                dim_H=encoder_cfg["dim_enc_H"],
                dimff_H=encoder_cfg["dimff_enc_H"],
                dim_C=encoder_cfg["dim_enc_C"],
                dimff_C=encoder_cfg["dimff_enc_C"],
                hidden_dim=encoder_cfg["ffn_hidden"],
                n_head=encoder_cfg["n_head"],
                num_layers=encoder_cfg["n_layers"],
                drop_prob=encoder_cfg["drop_prob"],
                peakwidthemb_num=peakwidthemb_num,
                splitemb_num=splitemb_num,
                integralemb_num=integralemb_num,
            )
        else:
            self.flag_onlyH = False
            self.encoder = NMR_encoder(
                dim_H=encoder_cfg["dim_enc_H"],
                dimff_H=encoder_cfg["dimff_enc_H"],
                dim_C=encoder_cfg["dim_enc_C"],
                dimff_C=encoder_cfg["dimff_enc_C"],
                hidden_dim=encoder_cfg["ffn_hidden"],
                n_head=encoder_cfg["n_head"],
                num_layers=encoder_cfg["n_layers"],
                drop_prob=encoder_cfg["drop_prob"],
                peakwidthemb_num=peakwidthemb_num,
                splitemb_num=splitemb_num,
                integralemb_num=integralemb_num,
            )
        # load spectrum encoder model from pretrained model
        state_dict = paddle.load(encoder_cfg["pretrained_path"])
        prefixes = ("spectrum_encoder.", "encoder.")
        encoder_state_dict = {
            k[len(pref) :]: v
            for k, v in state_dict.items()
            for pref in prefixes
            if k.startswith(pref)
        }
        self.encoder.set_state_dict(encoder_state_dict)

        # set graph decoder model
        self.decoder = GraphTransformer(
            n_layers=decoder_cfg["num_layers"],
            input_dims=input_dims,
            hidden_mlp_dims=decoder_cfg["hidden_mlp_dims"],
            hidden_dims=decoder_cfg["hidden_dims"],
            output_dims=output_dims,
            act_fn_in=nn.ReLU(),
            act_fn_out=nn.ReLU(),
        )
        # load graph decoder model from pretrained model
        state_dict = paddle.load(decoder_cfg["pretrained_path"])
        decoder_state_dict = {
            k[len("decoder.") :]: v
            for k, v in state_dict.items()
            if k.startswith("decoder.")
        }
        self.decoder.set_state_dict(decoder_state_dict)

        # set connector model
        self.connector_flag = False
        if connector_cfg and connector_cfg["__name__"] == "DiffPrior":
            self.connector_flag = True
            self.connector = DiffPrior(
                sample_cfg=connector_cfg["sample_cfg"],
                connector_cfg=connector_cfg["model_cfg"],
                clip=clip,
                execution_backend=execution_backend,
                runtime_options=runtime_options,
            )
            state_dict = paddle.load(connector_cfg["pretrained_model_path"])
            connector_state_dict = {
                k[len("connector.") :]: v
                for k, v in state_dict.items()
                if k.startswith("connector.")
            }
            self.connector.set_state_dict(connector_state_dict)
        else:
            self.connector = nn.Identity()

        # configure loss calculation with initialization of transition model
        self.Xdim = input_dims["X"]
        self.Edim = input_dims["E"]
        self.ydim = input_dims["y"]
        self.Xdim_output = output_dims["X"]
        self.Edim_output = output_dims["E"]
        self.ydim_output = output_dims["y"]
        self.node_dist = dataset_infos.nodes_dist

        # Transition Model
        if diffmodel_cfg["transition"] == "uniform":
            self.transition_model = DiscreteUniformTransition(
                x_classes=self.Xdim_output,
                e_classes=self.Edim_output,
                y_classes=self.ydim_output,
            )
            x_limit = paddle.ones([self.Xdim_output]) / self.Xdim_output
            e_limit = paddle.ones([self.Edim_output]) / self.Edim_output
            y_limit = paddle.ones([self.ydim_output]) / self.ydim_output
            self.limit_dist = diffgraphformer_utils.PlaceHolder(
                X=x_limit, E=e_limit, y=y_limit
            )

        elif diffmodel_cfg["transition"] == "marginal":
            node_types = self.dataset_info.node_types.astype("float32")
            x_marginals = node_types / paddle.sum(node_types)

            edge_types = self.dataset_info.edge_types.astype("float32")
            e_marginals = edge_types / paddle.sum(edge_types)
            logger.info(
                f"Marginal distribution of classes: {x_marginals.tolist()} for nodes, "
            )
            logger.info(f"{e_marginals.tolist()} for edges")

            self.transition_model = MarginalUniformTransition(
                x_marginals=x_marginals,
                e_marginals=e_marginals,
                y_classes=self.ydim_output,
            )
            self.limit_dist = diffgraphformer_utils.PlaceHolder(
                X=x_marginals,
                E=e_marginals,
                y=paddle.ones([self.ydim_output]) / self.ydim_output,
            )

        self.train_loss = TrainLossDiscrete(diffmodel_cfg["lambda_train"])

        self.best_val_nll = 1e8
        self.val_counter = 0

        self.val_nll = NLL()
        self.val_X_kl = SumExceptBatchKL()
        self.val_E_kl = SumExceptBatchKL()
        self.val_X_logp = SumExceptBatchMetric()
        self.val_E_logp = SumExceptBatchMetric()

        self.test_nll = NLL()
        self.test_X_kl = SumExceptBatchKL()
        self.test_E_kl = SumExceptBatchKL()
        self.test_X_logp = SumExceptBatchMetric()
        self.test_E_logp = SumExceptBatchMetric()

        self.seq_len_H1 = dataset_infos.seq_len_H1
        self.seq_len_C13 = dataset_infos.seq_len_C13
        self.tem = 2  # TODO remove later

        # set use formula for training and sample or not
        self.flag_use_formula = diffmodel_cfg.get("flag_use_formula", False)

    def run_encoder(self, *args):
        return self._run_runtime("spectrum_encoder", self.encoder, *args)

    def run_decoder(self, *args):
        return self._run_runtime("denoise_step", self.decoder, *args)

    def make_src_mask(self, src):
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        return src_mask

    def forward(self, batch):
        batch_graph = batch["graph"]
        batch_property = batch["property"]
        batch_spectrum = batch["spectrum"]

        # 1. Convert sparse graph to dense tensors and apply node mask
        dense_data, node_mask = diffgraphformer_utils.to_dense(
            paddle.to_tensor(batch_graph.node_feat["feat"]),
            paddle.to_tensor(batch_graph.edges.T),
            paddle.to_tensor(batch_graph.edge_feat["feat"]),
            paddle.to_tensor(batch_graph.graph_node_id),
        )
        dense_data = dense_data.mask(node_mask)
        X, E = dense_data.X, dense_data.E
        y = paddle.to_tensor(batch_property["y"])

        # 2. Add noise and compute extra features
        noisy_data = scheduling_diffnmr.apply_noise(
            self, X, E, y, node_mask, self.flag_use_formula
        )
        extra_data = scheduling_diffnmr.compute_extra_data(self, noisy_data)

        # Decoder inputs (noisy + extra features)
        input_X = paddle.concat(
            [noisy_data["X_t"].astype("float32"), extra_data.X], axis=2
        ).astype(dtype="float32")
        input_E = paddle.concat(
            [noisy_data["E_t"].astype("float32"), extra_data.E], axis=3
        ).astype(dtype="float32")
        input_y = paddle.hstack(
            [noisy_data["y_t"].astype("float32"), extra_data.y]
        ).astype(dtype="float32")

        # 3. get NMR embedded vector
        # prepare NMR vectors
        batch_length = batch_graph.num_graph
        condition_H1nmr = paddle.to_tensor(batch_spectrum["H_nmr"]).reshape(
            batch_length, self.seq_len_H1, -1
        )  # TODO: optimize self.seq_len_H1
        condition_C13nmr = paddle.to_tensor(batch_spectrum["C_nmr"]).reshape(
            batch_length, self.seq_len_C13
        )
        num_H_peak = paddle.to_tensor(batch_spectrum["num_H_peak"])
        num_C_peak = paddle.to_tensor(batch_spectrum["num_C_peak"])
        condition_Spectrum = [condition_H1nmr, num_H_peak, condition_C13nmr, num_C_peak]
        encoder_output = self.run_encoder(condition_Spectrum)
        if self.flag_onlyH is True:
            # The H-only encoder returns the pooled H/C branches directly and
            # does not expose token-level conditioning for DiffPrior.
            embeddings_spectrum = encoder_output[0]
            spectrum_encodings = None
        else:
            embeddings_spectrum, spectrum_encodings = encoder_output
            # NMR_encoder returns ``(token_embeddings, token_mask)``.  The
            # prior reconstructs the mask from padded token embeddings.
            if isinstance(spectrum_encodings, (tuple, list)):
                spectrum_encodings = spectrum_encodings[0]

        if self.connector_flag is True:
            embeddings_spectrum = self.connector.sample(
                embeddings_spectrum,
                spectrum_encodings=spectrum_encodings,
                # The connector returns one embedding per input spectrum.  The
                # public sampling default keeps multiple candidates, which would
                # expand the batch and break the decoder contract here.
                num_samples_per_batch=1,
            )

        input_y = paddle.concat([input_y, embeddings_spectrum], axis=1).astype(
            "float32"
        )

        # 4. Decoder forward
        # Convention: pred.X and pred.E are logits with shapes [B, n, Cx] and
        # [B, n, n, Ce]
        pred = self.run_decoder(input_X, input_E, input_y, node_mask)

        # 5. Compute training loss
        loss_dict = self.train_loss(
            masked_pred_X=pred.X,
            masked_pred_E=pred.E,
            pred_y=pred.y,
            true_X=X,
            true_E=E,
            true_y=paddle.to_tensor(batch_property["y"]),
        )

        # 6. Assemble outputs for Trainer & streaming metrics
        # Predictions: provide masked_pred_X/E; mirror X_logits/E_logits for legacy
        # paths
        pred_dict = {
            "masked_pred_X": pred.X,
            "masked_pred_E": pred.E,
            "pred_y": pred.y,
        }
        # Labels: provide true_X/true_E; node_mask is optional but useful for some
        # metrics
        label_dict = {
            "true_X": X,
            "true_E": E,
            "true_y": y,
        }

        result = {
            "loss_dict": loss_dict,
            "pred_dict": pred_dict,
            "label_dict": label_dict,
            "node_mask": node_mask,
            "noisy_data": noisy_data,
        }

        return result


# PP-DiffNMR
class DiffPrior(RuntimeMixin, nn.Layer):
    def __init__(
        self,
        sample_cfg: dict,
        connector_cfg: dict,
        clip: nn.Layer,
        execution_backend="eager",
        runtime_options=None,
    ):
        super().__init__()
        self._init_runtime(execution_backend, runtime_options)

        self.clip = clip
        self.timesteps = sample_cfg["timesteps"]  # TODO: check
        self.sample_timesteps = default(sample_cfg["sample_timesteps"], self.timesteps)
        self.noise_scheduler = NoiseScheduler(
            beta_schedule=sample_cfg["beta_schedule"],
            timesteps=sample_cfg["timesteps"],
            loss_type=sample_cfg["loss_type"],
        )
        if exists(clip):
            freeze_model_and_make_eval_(clip)
            self.clip = clip
        else:
            self.clip = None
        self.net = DiffPriorNetwork(**connector_cfg)
        self.graph_embed_dim = sample_cfg["graph_embed_dim"]

        assert (
            self.net.dim == self.graph_embed_dim
        ), f"your diffusion prior network has a dimension of {self.net.dim}, \
            but you set your image embedding dimension (keyword graph_embed_dim) \
            on DiffPrior to {self.graph_embed_dim}"

        self.cond_drop_prob = default(sample_cfg["cond_drop_prob"], 0.0)
        self.spectrum_cond_drop_prob = default(
            sample_cfg["spectrum_cond_drop_prob"], self.cond_drop_prob
        )
        self.graph_cond_drop_prob = default(
            sample_cfg["graph_cond_drop_prob"], self.cond_drop_prob
        )

        self.can_classifier_guidance = (
            self.spectrum_cond_drop_prob > 0.0 and self.graph_cond_drop_prob > 0.0
        )
        self.condition_on_spectrum_encodings = default(
            sample_cfg["condition_on_spectrum_encodings"], True
        )

        self.predict_x_start = sample_cfg["predict_x_start"]
        self.predict_v = sample_cfg["predict_v"]

        self.graph_embed_scale = default(
            sample_cfg["graph_embed_scale"], sample_cfg["graph_embed_dim"] ** 0.5
        )

        self.sampling_clamp_l2norm = sample_cfg["sampling_clamp_l2norm"]
        self.sampling_final_clamp_l2norm = sample_cfg["sampling_final_clamp_l2norm"]

        self.training_clamp_l2norm = sample_cfg["training_clamp_l2norm"]
        self.init_graph_embed_l2norm = sample_cfg["init_graph_embed_l2norm"]

        # TODO: maybe could remove this dummy buffer
        self.register_buffer(
            name="_dummy", tensor=paddle.to_tensor(data=[True]), persistable=False
        )

    def _run_prior(self, *args, **kwargs):
        return self._run_runtime("denoise_step", self.net, *args, **kwargs)

    def forward(self, batch):

        batch_graph = batch["graph"]
        batch_property = batch.get("property", {})
        batch_spectrum = batch.get("spectrum", {})

        # 1. obtain the graph embeddings
        if "graph_embed" in batch:
            graph_embed = batch["graph_embed"]
        elif "graph" in batch:
            dense_data, node_mask = diffgraphformer_utils.to_dense(
                paddle.to_tensor(batch_graph.node_feat["feat"]),
                paddle.to_tensor(batch_graph.edges.T).contiguous(),
                paddle.to_tensor(batch_graph.edge_feat["feat"]),
                paddle.to_tensor(batch_graph.graph_node_id),
            )
            dense_data = dense_data.mask(node_mask)
            X, E = dense_data.X, dense_data.E
            y = paddle.to_tensor(batch_property["y"])
            z_t = (
                diffgraphformer_utils.PlaceHolder(X=X, E=E, y=y)
                .type_as(X)
                .mask(node_mask)
            )
            extra_data_pure = scheduling_diffnmr.compute_extra_data(
                self.clip,
                {"X_t": z_t.X, "E_t": z_t.E, "y_t": z_t.y, "node_mask": node_mask},
                isPure=True,
            )
            input_X_pure = paddle.concat(
                [z_t.X.astype("float32"), extra_data_pure.X], axis=2
            ).astype(dtype="float32")
            input_E_pure = paddle.concat(
                [z_t.E.astype("float32"), extra_data_pure.E], axis=3
            ).astype(dtype="float32")
            input_y_pure = paddle.hstack(
                x=(z_t.y.astype("float32"), extra_data_pure.y)
            ).astype(dtype="float32")
            # obtain the condition vector from output of encoder
            graph_embed = self.clip.graph_encoder(
                input_X_pure, input_E_pure, input_y_pure, node_mask
            )

        # 2. obtain the spectrum embeddings
        spectrum_cond = {}
        if "spectrum_embed" in batch:
            spectrum_cond["spectrum_embed"] = batch["spectrum_embed"]
            if self.condition_on_spectrum_encodings:
                spectrum_cond["spectrum_encodings"] = batch["spectrum_encodings"]
        elif "spectrum" in batch:
            condition_H1nmr = paddle.to_tensor(batch_spectrum["H_nmr"])
            condition_C13nmr = paddle.to_tensor(batch_spectrum["C_nmr"])
            num_H_peak = paddle.to_tensor(batch_spectrum["num_H_peak"])
            num_C_peak = paddle.to_tensor(batch_spectrum["num_C_peak"])
            condition_Spectrum = [
                condition_H1nmr,
                num_H_peak,
                condition_C13nmr,
                num_C_peak,
            ]
            spectrum_embed, spectrum_encodings = self.clip.spectrum_encoder(
                condition_Spectrum
            )
            spectrum_cond["spectrum_embed"] = spectrum_embed
            if self.condition_on_spectrum_encodings:
                spectrum_cond["spectrum_encodings"] = spectrum_encodings

        # 3. diffusion process
        batch_size = graph_embed.shape[0]
        times = self.noise_scheduler.sample_random_times(batch_size)
        graph_embed *= self.graph_embed_scale

        # 4. calculate loss
        loss_dict = self.p_losses(graph_embed, times, spectrum_cond=spectrum_cond)

        # 5. retrun restults
        return {
            "loss_dict": loss_dict,
            "pred_dict": {
                "graph_embed": graph_embed,
                "spectrum_embed": spectrum_cond.get("spectrum_embed"),
                "times": times,
            },
            "label_dict": {"graph": batch_graph, "property": batch_property},
        }

    def generate_embed_vector(self, batch):
        batch_graph, other_data = batch
        batch_length = batch_graph.num_graph
        # transfer to dense graph from sparse graph
        if batch_graph.edges.T.numel() == 0:
            print("Found a batch with no edges. Skipping.")
            return None
        dense_data, node_mask = diffgraphformer_utils.to_dense(
            batch_graph.node_feat["feat"],
            batch_graph.edges.T.contiguous(),
            batch_graph.edge_feat["feat"],
            batch_graph.graph_node_id,
        )
        dense_data = dense_data.mask(node_mask)
        graph_X, graph_E = (
            dense_data.X,
            dense_data.E,
        )
        graph_y = paddle.zeros(shape=[graph_X.shape[0], 1024]).cuda(blocking=True)

        clip_graph_embeds = self.clip.graph_encoder(
            graph_X, graph_E, graph_y, node_mask
        )

        spectrum_conditionVec = other_data["conditionVec"]
        spectrum_conditionVec = spectrum_conditionVec.reshape(
            [batch_length, self.config["CLIP"]["nmr_encoder"]["max_len"]]
        )

        assert isinstance(
            spectrum_conditionVec, paddle.Tensor
        ), "nmr_spectrum_conditionVec should be a tensor, but got type {}".format(
            type(spectrum_conditionVec)
        )
        spectrum_srcMask = self.clip.make_src_mask(spectrum_conditionVec)

        clip_spectrum_embeds = self.clip.spectrum_encoder(
            spectrum_conditionVec, spectrum_srcMask
        )
        clip_spectrum_embeds = clip_spectrum_embeds.reshape(
            [clip_spectrum_embeds.shape[0], -1]
        )
        clip_spectrum_embeds = self.clip.spectrum_encoder_projector(
            clip_spectrum_embeds
        )

        return clip_graph_embeds, clip_spectrum_embeds

    def p_losses(self, moleculargraph_embed, times, spectrum_cond, noise=None):
        noise = default(
            noise,
            lambda: paddle.randn(
                shape=moleculargraph_embed.shape, dtype=moleculargraph_embed.dtype
            ),
        )

        moleculargraph_embed_noisy = self.noise_scheduler.q_sample(
            x_start=moleculargraph_embed, t=times, noise=noise
        )

        self_cond = None
        if self.net.self_cond and random.random() < 0.5:
            with paddle.no_grad():
                self_cond = self._run_prior(
                    moleculargraph_embed_noisy, times, **spectrum_cond
                ).detach()

        pred = self._run_prior(
            moleculargraph_embed_noisy,
            times,
            self_cond=self_cond,
            spectrum_cond_drop_prob=self.spectrum_cond_drop_prob,
            graph_cond_drop_prob=self.graph_cond_drop_prob,
            **spectrum_cond,
        )

        if self.predict_x_start and self.training_clamp_l2norm:
            pred = self.l2norm_clamp_embed(pred)

        if self.predict_v:
            target = self.noise_scheduler.calculate_v(
                moleculargraph_embed, times, noise
            )
        elif self.predict_x_start:
            target = moleculargraph_embed
        else:
            target = noise

        loss = self.noise_scheduler.loss_fn(pred, target)

        return {"loss": loss}

    def l2norm_clamp_embed(self, graph):
        return l2norm(graph) * self.graph_embed_scale

    @paddle.no_grad()
    def sample(
        self,
        spectrum_embeds,
        spectrum_encodings=None,
        num_samples_per_batch=2,
        cond_scale=1.0,
        timesteps=None,  # mask
    ):
        timesteps = default(timesteps, self.sample_timesteps)

        if isinstance(spectrum_encodings, (tuple, list)):
            # NMR_encoder returns ``(token_embeddings, token_mask)``.  Accepting
            # that result directly keeps this public connector API consistent
            # with the encoder contract while DiffPriorNetwork consumes tokens.
            spectrum_encodings = spectrum_encodings[0]
        spectrum_embeds = repeat(
            spectrum_embeds, "b ... -> (b r) ...", r=num_samples_per_batch
        )
        if spectrum_encodings is not None and self.condition_on_spectrum_encodings:
            spectrum_encodings = repeat(
                spectrum_encodings,
                "b ... -> (b r) ...",
                r=num_samples_per_batch,
            )

        batch_size = tuple(spectrum_embeds.shape)[0]
        graph_embed_dim = self.graph_embed_dim

        # spectrum_embeds = self.clip.spectrum_encoder(spectrum, mask)
        # spectrum_embeds = spectrum_embeds.reshape([spectrum_embeds.shape[0], -1])
        # spectrum_embeds = self.clip.spectrum_encoder_projector(spectrum_embeds)

        spectrum_cond = dict(spectrum_embed=spectrum_embeds)

        if self.condition_on_spectrum_encodings:
            spectrum_cond = {**spectrum_cond, "spectrum_encodings": spectrum_encodings}

        graph_embeds = self.p_sample_loop(
            (batch_size, graph_embed_dim),
            spectrum_cond=spectrum_cond,
            cond_scale=cond_scale,
            timesteps=timesteps,
        )

        # retrieve original unscaled image embed

        spectrum_embeds = spectrum_cond["spectrum_embed"]

        spectrum_embeds = rearrange(
            spectrum_embeds, "(b r) d -> b r d", r=num_samples_per_batch
        )
        graph_embeds = rearrange(
            graph_embeds, "(b r) d -> b r d", r=num_samples_per_batch
        )

        spectrum_image_sims = paddle.einsum(
            "b r d, b r d -> b r", l2norm(spectrum_embeds), l2norm(graph_embeds)
        )
        top_sim_indices = spectrum_image_sims.topk(k=1)[1]

        top_sim_indices = repeat(top_sim_indices, "b 1 -> b 1 d", d=graph_embed_dim)

        top_graph_embeds = graph_embeds.take_along_axis(
            axis=1, indices=top_sim_indices, broadcast=False
        )
        return rearrange(top_graph_embeds, "b 1 d -> b d")

    @paddle.no_grad()
    def p_sample_loop(self, *args, timesteps=None, **kwargs):
        timesteps = default(timesteps, self.noise_scheduler.num_timesteps)
        assert timesteps <= self.noise_scheduler.num_timesteps
        is_ddim = timesteps < self.noise_scheduler.num_timesteps
        if not is_ddim:
            normalized_graph_embed = self.p_sample_loop_ddpm(*args, **kwargs)
        else:
            normalized_graph_embed = self.p_sample_loop_ddim(
                *args, **kwargs, timesteps=timesteps
            )
        graph_embed = normalized_graph_embed / self.graph_embed_scale
        return graph_embed

    @paddle.no_grad()
    def p_sample_loop_ddpm(self, shape, spectrum_cond, cond_scale=1.0):
        batch = shape[0]
        graph_embed = paddle.randn(shape=shape)
        x_start = None
        if self.init_graph_embed_l2norm:
            graph_embed = l2norm(graph_embed) * self.graph_embed_scale
        for i in tqdm(
            reversed(range(0, self.noise_scheduler.num_timesteps)),
            desc="diffprior sampling",
            total=self.noise_scheduler.num_timesteps,
        ):
            times = paddle.full(shape=(batch,), fill_value=i, dtype="int64")
            self_cond = x_start if self.net.self_cond else None
            graph_embed, x_start = self.p_sample(
                graph_embed,
                times,
                spectrum_cond=spectrum_cond,
                self_cond=self_cond,
                cond_scale=cond_scale,
            )
        if self.sampling_final_clamp_l2norm and self.predict_x_start:
            graph_embed = self.l2norm_clamp_embed(graph_embed)
        return graph_embed

    @paddle.no_grad()
    def p_sample_loop_ddim(
        self, shape, spectrum_cond, *, timesteps, eta=1.0, cond_scale=1.0
    ):
        alphas = self.noise_scheduler.alphas_cumprod
        batch, total_timesteps = shape[0], self.noise_scheduler.num_timesteps
        # Include both the initial diffusion index and the terminal ``-1``
        # sentinel.  Slicing the final point (the old implementation) produced
        # ``timesteps - 1`` pairs and, for ``timesteps == 1``, an empty loop that
        # returned the initial random embedding without invoking the network.
        times = paddle.linspace(
            start=-1.0,
            stop=float(total_timesteps - 1),
            num=timesteps + 1,
        )
        times = list(reversed(times.astype(dtype="int32").tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))
        graph_embed = paddle.randn(shape=shape)
        x_start = None
        if self.init_graph_embed_l2norm:
            graph_embed = l2norm(graph_embed) * self.graph_embed_scale
        for time, time_next in tqdm(time_pairs, desc="diffprior sampling"):
            alpha = alphas[time]
            time_cond = paddle.full(shape=(batch,), fill_value=time, dtype="int64")
            self_cond = x_start if self.net.self_cond else None
            pred = self.net.forward_with_cond_scale(
                graph_embed,
                time_cond,
                self_cond=self_cond,
                cond_scale=cond_scale,
                **spectrum_cond,
            )
            if self.predict_v:
                x_start = self.noise_scheduler.predict_start_from_v(
                    graph_embed, t=time_cond, v=pred
                )
            elif self.predict_x_start:
                x_start = pred
            else:
                x_start = self.noise_scheduler.predict_start_from_noise(
                    graph_embed, t=time_cond, noise=pred
                )
            if not self.predict_x_start:
                x_start.clip_(min=-1.0, max=1.0)
            if self.predict_x_start and self.sampling_clamp_l2norm:
                x_start = self.l2norm_clamp_embed(x_start)
            if time_next < 0:
                graph_embed = x_start
                continue
            alpha_next = alphas[time_next]
            pred_noise = self.noise_scheduler.predict_noise_from_start(
                graph_embed, t=time_cond, x0=x_start
            )
            c1 = (
                eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            )
            c2 = (1 - alpha_next - paddle.square(x=c1)).sqrt()
            noise = (
                paddle.randn(shape=graph_embed.shape, dtype=graph_embed.dtype)
                if time_next > 0
                else 0.0
            )
            graph_embed = x_start * alpha_next.sqrt() + c1 * noise + c2 * pred_noise
        if self.predict_x_start and self.sampling_final_clamp_l2norm:
            graph_embed = self.l2norm_clamp_embed(graph_embed)
        return graph_embed

    @paddle.no_grad()
    def p_sample(
        self,
        x,
        t,
        spectrum_cond=None,
        self_cond=None,
        clip_denoised=True,
        cond_scale=1.0,
    ):
        (
            b,
            *_,
        ) = x.shape
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(
            x=x,
            t=t,
            spectrum_cond=spectrum_cond,
            self_cond=self_cond,
            clip_denoised=clip_denoised,
            cond_scale=cond_scale,
        )
        noise = paddle.randn(shape=x.shape, dtype=x.dtype)
        nonzero_mask = (1 - (t == 0).astype(dtype="float32")).reshape(
            b, *((1,) * (len(tuple(x.shape)) - 1))
        )
        pred = model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise
        return pred, x_start

    def p_mean_variance(
        self, x, t, spectrum_cond, self_cond=None, clip_denoised=False, cond_scale=1.0
    ):
        assert not (
            cond_scale != 1.0 and not self.can_classifier_guidance
        ), "the model was not trained with conditional dropout, and thus one cannot \
            use classifier free guidance (cond_scale anything other than 1)"
        pred = self.net.forward_with_cond_scale(
            x, t, cond_scale=cond_scale, self_cond=self_cond, **spectrum_cond
        )
        if self.predict_v:
            x_start = self.noise_scheduler.predict_start_from_v(x, t=t, v=pred)
        elif self.predict_x_start:
            x_start = pred
        else:
            x_start = self.noise_scheduler.predict_start_from_noise(x, t=t, noise=pred)
        if clip_denoised and not self.predict_x_start:
            x_start.clip_(min=-1.0, max=1.0)
        if self.predict_x_start and self.sampling_clamp_l2norm:
            x_start = l2norm(x_start) * self.graph_embed_scale
        (
            model_mean,
            posterior_variance,
            posterior_log_variance,
        ) = self.noise_scheduler.q_posterior(x_start=x_start, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

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

import paddle

from ppmat.models.common.activation import NormActivation
from ppmat.models.common.activation import ScalarActivation
from ppmat.models.common.e3nn import o3
from ppmat.models.common.e3nn.math import soft_one_hot_linspace
from ppmat.models.common.e3nn.nn import FullyConnectedNet
from ppmat.models.common.orbital import GaussianOrbital
from ppmat.models.infgcn.graph_converter import AtomGridRadiusGraphConverter
from ppmat.utils.scatter import scatter_sum_first_order


class GCNLayer(paddle.nn.Layer):
    def __init__(
        self,
        irreps_in,
        irreps_out,
        irreps_edge,
        radial_embed_size,
        num_radial_layer,
        radial_hidden_size,
        is_fc=True,
        use_sc=True,
        irrep_normalization="component",
        path_normalization="element",
    ):
        """
        A single InfGCN layer for Tensor Product-based message passing.
        If the tensor product is fully connected, we have (for every path)

        .. math::
            z_w=\\sum_{uv}w_{uvw}x_u\\otimes y_v=\\sum_{u}w_{uw}x_u \\otimes y

        Else, we have

        .. math::
            z_u=x_u\\otimes \\sum_v w_{uv}y_v=w_u (x_u\\otimes y)

        Here, uvw are radial (channel) indices of the first input, second input,
        and output, respectively. Notice that in our model, the second input is
        always the spherical harmonics of the edge vector,
        so the index v can be safely ignored.

        :param irreps_in: irreducible representations of input node features
        :param irreps_out: irreducible representations of output node features
        :param irreps_edge: irreducible representations of edge features
        :param radial_embed_size: embedding size of the edge length
        :param num_radial_layer: number of hidden layers in the radial network
        :param radial_hidden_size: hidden size of the radial network
        :param is_fc: whether to use fully connected tensor product
        :param use_sc: whether to use self-connection
        :param irrep_normalization: representation normalization passed to the
            `o3.FullyConnectedTensorProduct`
        :param path_normalization: path normalization passed to the
            `o3.FullyConnectedTensorProduct`
        """
        super().__init__()
        self.irreps_in = o3.Irreps(irreps_in)
        self.irreps_out = o3.Irreps(irreps_out)
        self.irreps_edge = o3.Irreps(irreps_edge)
        self.radial_embed_size = radial_embed_size
        self.num_radial_layer = num_radial_layer
        self.radial_hidden_size = radial_hidden_size
        self.is_fc = is_fc
        self.use_sc = use_sc

        if self.is_fc:
            self.tp = o3.FullyConnectedTensorProduct(
                self.irreps_in,
                self.irreps_edge,
                self.irreps_out,
                internal_weights=False,
                shared_weights=False,
                irrep_normalization=irrep_normalization,
                path_normalization=path_normalization,
            )
        else:
            instr = [
                (i_1, i_2, i_out, "uvu", True)
                for i_1, (_, ir_1) in enumerate(self.irreps_in)
                for i_2, (_, ir_edge) in enumerate(self.irreps_edge)
                for i_out, (_, ir_out) in enumerate(self.irreps_out)
                if ir_out in ir_1 * ir_edge
            ]
            self.tp = o3.TensorProduct(
                self.irreps_in,
                self.irreps_edge,
                self.irreps_out,
                instr,
                internal_weights=False,
                shared_weights=False,
                irrep_normalization=irrep_normalization,
                path_normalization=path_normalization,
            )
        # The activation is automatically normalized by a scaling factor.
        self.fc = FullyConnectedNet(
            [radial_embed_size]
            + num_radial_layer * [radial_hidden_size]
            + [self.tp.weight_numel],
            paddle.nn.functional.silu,
        )
        self.sc = None
        if self.use_sc:
            self.sc = o3.Linear(self.irreps_in, self.irreps_out)

    def forward(self, edge_index, node_feat, edge_feat, edge_embed, dim_size=None):
        src, dst = edge_index
        weight = self.fc(edge_embed)  # FFN
        out = self.tp(
            node_feat[src], edge_feat, weight=weight
        )  # Tensor Product [num_edges, tp.irreps_out.dim]

        out = scatter_sum_first_order(out, dst, dim_size)  # message aggregation

        if self.use_sc:
            out = out + self.sc(node_feat)
        return out


class InfGCN(paddle.nn.Layer):
    def __init__(
        self,
        vocab,
        num_radial,
        num_spherical,
        radial_embed_size,
        radial_hidden_size,
        num_radial_layer=2,
        num_gcn_layer=3,
        atom_graph_cutoff=3.0,
        atom_grid_cutoff=None,
        is_fc=True,
        gauss_start=0.5,
        gauss_end=5.0,
        activation="norm",
        residual=True,
        periodic_mode="none",
        inference_grid_point_budget=None,
        target_name="density",
        loss_eps=1e-8,
        **kwargs,
    ):
        """
        Implement the InfGCN model for electron density estimation
        :param vocab: vocabularies used by the model
        :param num_radial: number of radial basis
        :param num_spherical: maximum number of spherical harmonics for each
            radial basis,
                number of spherical basis will be (num_spherical + 1)^2
        :param radial_embed_size: embedding size of the edge length
        :param radial_hidden_size: hidden size of the radial network
        :param num_radial_layer: number of hidden layers in the radial network
        :param num_gcn_layer: number of InfGCN layers
        :param atom_graph_cutoff: cutoff used by the prebuilt atom graph and its
            radial basis expansion
        :param atom_grid_cutoff: cutoff for the residual atom-grid bipartite graph;
            required only when ``residual`` is enabled
        :param is_fc: whether the InfGCN layer should use fully connected
            tensor product
        :param gauss_start: start coefficient of the Gaussian radial basis
        :param gauss_end: end coefficient of the Gaussian radial basis
        :param activation: activation type for the InfGCN layer, can be
            ['scalar', 'norm']
        :param residual: whether to use the residue prediction layer
        :param periodic_mode: ``"none"`` for molecular data or
            ``"minimum_image"`` for periodic orbital minimum-image vectors
        :param inference_grid_point_budget: optional maximum total number of grid
            points decoded by one evaluation chunk across the whole batch. Only
            affects ``trainer.eval()``; ``predict.py`` is controlled by
            ``grid_batch_size`` instead. Bounds peak decoder memory regardless
            of batch size (peak ≈ avg_atoms × budget × orbital_dims).
        """
        super().__init__()
        self.vocab = vocab
        n_atom_type = vocab["atom"]["num_embeddings"]
        self.n_atom_type = n_atom_type
        self.num_radial = num_radial
        self.num_spherical = num_spherical
        self.radial_embed_size = radial_embed_size
        self.radial_hidden_size = radial_hidden_size
        self.num_radial_layer = num_radial_layer
        self.num_gcn_layer = num_gcn_layer
        self.atom_graph_cutoff = float(atom_graph_cutoff)
        if residual and atom_grid_cutoff is None:
            raise ValueError("atom_grid_cutoff is required when residual=True.")
        self.atom_grid_cutoff = (
            None if atom_grid_cutoff is None else float(atom_grid_cutoff)
        )
        self.is_fc = is_fc
        self.gauss_start = gauss_start
        self.gauss_end = gauss_end
        self.activation = activation
        self.residual = residual
        if periodic_mode not in {"none", "minimum_image"}:
            raise ValueError("periodic_mode must be 'none' or 'minimum_image'.")
        if periodic_mode == "minimum_image" and residual:
            raise ValueError(
                "The minimum-image periodic configuration requires residual=False."
            )
        self.periodic_mode = periodic_mode
        self.inference_grid_point_budget = self._validate_optional_positive_int(
            inference_grid_point_budget, "inference_grid_point_budget"
        )
        self.target_name = target_name
        self.loss_eps = loss_eps
        assert activation in ["scalar", "norm"]
        self.embedding = paddle.nn.Embedding(
            num_embeddings=n_atom_type, embedding_dim=num_radial
        )
        self.irreps_sh = o3.Irreps.spherical_harmonics(num_spherical, p=1)
        self.irreps_feat = (self.irreps_sh * num_radial).sort().irreps.simplify()

        self.gcns = paddle.nn.LayerList(
            sublayers=[
                GCNLayer(
                    f"{num_radial}x0e" if i == 0 else self.irreps_feat,
                    self.irreps_feat,
                    self.irreps_sh,
                    radial_embed_size,
                    num_radial_layer,
                    radial_hidden_size,
                    is_fc=is_fc,
                    **kwargs,
                )
                for i in range(num_gcn_layer)
            ]
        )
        if self.activation == "scalar":
            self.act = ScalarActivation(
                self.irreps_feat,
                paddle.nn.functional.silu,
                paddle.nn.functional.sigmoid,
            )
        else:
            self.act = NormActivation(self.irreps_feat)
        self.residue = None
        self.atom_grid_graph_converter = None
        if self.residual:
            self.atom_grid_graph_converter = AtomGridRadiusGraphConverter(
                cutoff=self.atom_grid_cutoff,
                max_num_neighbors=32,
            )
            self.residue = GCNLayer(
                self.irreps_feat,
                "0e",
                self.irreps_sh,
                radial_embed_size,
                num_radial_layer,
                radial_hidden_size,
                is_fc=True,
                use_sc=False,
                **kwargs,
            )
        self.orbital = GaussianOrbital(
            gauss_start, gauss_end, num_radial, num_spherical
        )
        self._criterion = paddle.nn.MSELoss(reduction="mean")

    def _forward(self, data):
        grid = data["grid_coord"]
        info = data.get("info")

        # PGL graphs travel through the DataLoader untouched, so the graph
        # fields are converted here. ``tensor()`` is in-place and short-circuits
        # when the graph already holds tensors.
        graph = data["graph"].tensor()
        atom_types = graph.node_feat["x"]
        atom_coord = graph.node_feat["cart_coords"]
        atom_edges = graph.edges.transpose([1, 0])
        graph_batch = graph.graph_node_id.astype("int64")
        cell = self._prepare_cell(info)

        # Predict the field independently of target availability.
        atom_features = self._encode_atoms(
            atom_types,
            atom_coord,
            atom_edges,
        )
        chunk_size = data.get("grid_batch_size")
        if chunk_size is None:
            chunk_size = self._inference_chunk_size(grid.shape[0], grid.shape[1])
        if not self.training and chunk_size and grid.shape[1] > chunk_size:
            pred = paddle.concat(
                [
                    self._decode_grid_chunk(
                        atom_features,
                        atom_coord,
                        grid[:, start : start + chunk_size],
                        graph_batch,
                        cell,
                    )
                    for start in range(0, grid.shape[1], chunk_size)
                ],
                axis=1,
            )
        else:
            pred = self._decode_grid_chunk(
                atom_features,
                atom_coord,
                grid,
                graph_batch,
                cell,
            )

        return pred

    def forward(self, data, return_loss=True, return_prediction=True):
        """Run field prediction through the PaddleMaterials model protocol."""

        assert (
            return_loss or return_prediction
        ), "At least one of return_loss or return_prediction must be True."

        pred = self._forward(data)
        mask = data.get("density_mask")
        masked_pred = pred
        if mask is not None:
            mask = mask.astype(pred.dtype)
            masked_pred = pred * mask

        # Calculate loss and NMAE only when requested.
        loss_dict = {}
        if return_loss:
            density = data[self.target_name]
            if density is None:
                raise ValueError(
                    f"data[{self.target_name!r}] must not be None when "
                    "return_loss is True."
                )
            if mask is not None:
                label_masked = density * mask
                denom = paddle.sum(mask) + self.loss_eps
                loss = paddle.sum((masked_pred - label_masked) ** 2) / denom
                # Normalized MAE (original InfGCN):
                #   mae = sum(|pred - density|) / sum(density)
                mae = paddle.sum(paddle.abs(masked_pred - label_masked)) / (
                    paddle.sum(paddle.abs(label_masked)) + self.loss_eps
                )
            else:
                label_masked = density
                loss = self._criterion(pred, label_masked)
                mae = paddle.sum(paddle.abs(pred - label_masked)) / (
                    paddle.sum(paddle.abs(label_masked)) + self.loss_eps
                )
            loss_dict["loss"] = loss
            loss_dict["mae"] = mae

        pred_dict = {}
        if return_prediction:
            pred_dict[self.target_name] = masked_pred
        return {"loss_dict": loss_dict, "pred_dict": pred_dict}

    @paddle.no_grad()
    def predict(self, samples):
        is_list = isinstance(samples, list)
        samples = samples if is_list else [samples]

        results = []
        for sample in samples:
            sample = dict(sample)
            grid = sample.pop("grid", None)
            if grid is not None:
                sample["grid_coord"] = paddle.to_tensor(
                    grid.cartesian_coordinates(),
                    dtype="float32",
                ).reshape([1, -1, 3])
                sample["info"] = {
                    "cell": paddle.to_tensor(grid.cell_vectors, dtype="float32")
                }
            result = self._forward(sample).reshape([-1]).detach().cpu()
            results.append({self.target_name: result})

        return results if is_list else results[0]

    def _prepare_cell(self, info):
        if self.periodic_mode == "none":
            return None
        cell = info["cell"]
        return cell.unsqueeze(0) if cell.ndim == 2 else cell

    @staticmethod
    def _validate_optional_positive_int(value, name):
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
        ):
            raise ValueError(f"{name} must be a positive integer or None.")
        return value

    def _inference_chunk_size(self, batch_size, num_grid_points):
        if self.training or self.inference_grid_point_budget is None:
            return None
        chunk_size = max(1, self.inference_grid_point_budget // int(batch_size))
        return min(chunk_size, int(num_grid_points))

    def _encode_atoms(self, atom_types, atom_coord, atom_edges):
        """Encode the atom graph once for reuse by every grid chunk.

        :param atom_types: atom types of (N,)
        :param atom_coord: atom coordinates of (N, 3)
        :param atom_edges: candidate atom edges of (2, E)
        :return: encoded atom features of (N, irreps_feat.dim)
        """
        feat = self.embedding(atom_types)

        # Graph topology, including max_num_neighbors, belongs to the converter.
        edge_index = atom_edges
        src, dst = edge_index
        edge_vec = atom_coord[src] - atom_coord[dst]  # coord vector
        edge_len = edge_vec.norm(axis=-1) + 1e-08  # L2 norm, equal to distance

        # Angular features have shape [num_edges, 2 * degree + 1].
        edge_feat = o3.spherical_harmonics(
            list(range(self.num_spherical + 1)),  # degree of the spherical harmonics
            edge_vec / edge_len[..., None],  # e.g. edge vector
            normalize=False,  # Input vectors are already normalized.
            normalization="integral",  # normalization of the output tensors
        )

        edge_embed = (
            soft_one_hot_linspace(  # radial features, [D_edge_index, radial_embed_size]
                edge_len,
                start=0.0,
                end=self.atom_graph_cutoff,
                number=self.radial_embed_size,  # The number of radial basis functions.
                basis="gaussian",  # Uses Gaussian functions as the radial basis.
                cutoff=False,  # Disables the cutoff/smoothing function at the boundary.
            )
            * (self.radial_embed_size**0.5)
        )  # enhance signal feature due to normalization of output

        for i, gcn in enumerate(self.gcns):
            feat = gcn(
                edge_index, feat, edge_feat, edge_embed, dim_size=atom_types.shape[0]
            )
            if i != self.num_gcn_layer - 1:
                feat = self.act(feat)

        return feat

    def _decode_grid_chunk(self, feat, atom_coord, grid, batch, cell):
        """Decode one grid chunk from shared atom features."""

        n_graph, n_sample = grid.shape[0], grid.shape[1]
        if self.residual:
            # Grid chunks are sliced along the point axis and may be
            # non-contiguous. ``reshape`` supports both contiguous full grids
            # and these evaluation-only slices.
            grid_flat = grid.reshape([-1, 3])
            grid_batch = paddle.arange(end=n_graph).repeat_interleave(repeats=n_sample)
            atom_grid_edges = self.atom_grid_graph_converter(
                atom_coord,
                grid_flat,
                batch,
                grid_batch,
            )
            if atom_grid_edges.shape[0] != 0:
                node_src, grid_dst = atom_grid_edges.transpose([1, 0])
                grid_edge = grid_flat[grid_dst] - atom_coord[node_src]
                grid_len = paddle.linalg.norm(x=grid_edge, axis=-1) + 1e-08
                grid_edge_feat = o3.spherical_harmonics(
                    list(range(self.num_spherical + 1)),
                    grid_edge / (grid_len[..., None] + 1e-08),
                    normalize=False,
                    normalization="integral",
                )
                grid_edge_embed = soft_one_hot_linspace(
                    grid_len,
                    start=0.0,
                    end=self.atom_grid_cutoff,
                    number=self.radial_embed_size,
                    basis="gaussian",
                    cutoff=False,
                ) * (self.radial_embed_size**0.5)

                residue = self.residue(
                    (node_src, grid_dst),
                    feat,
                    grid_edge_feat,
                    grid_edge_embed,
                    dim_size=grid_flat.shape[0],
                )
            else:
                residue = paddle.zeros([grid_flat.shape[0], 1], dtype=feat.dtype)
        else:
            residue = 0.0

        # Displacement vectors from each atom to each sampled grid point have
        # shape [num_atoms, num_grid_points, 3].
        sample_vec = grid[batch] - atom_coord.unsqueeze(axis=-2)
        if cell is not None:
            cell = cell[batch]
            sample_vec = sample_vec @ paddle.linalg.inv(cell)
            sample_vec = (sample_vec - paddle.round(sample_vec)) @ cell

        # Expand displacement vectors in the Gaussian-type orbital basis:
        # [num_atoms, num_grid_points, (lmax + 1)^2 * num_gaussians].
        orbital = self.orbital(sample_vec)
        density = (orbital * feat.unsqueeze(axis=1)).sum(
            axis=-1
        )  # linear combination [n_atom, n_grid]
        density = scatter_sum_first_order(
            density, batch, n_graph
        )  # molecular/cell density

        if self.residual:
            density = density + residue.view(*tuple(density.shape))

        return density

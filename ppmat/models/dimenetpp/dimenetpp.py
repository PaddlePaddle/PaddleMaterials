import math
from typing import Callable
from typing import Optional

import paddle
import sympy as sym
from paddle.nn.functional import swish

from ppmat.models.common.basis_utils import bessel_basis
from ppmat.models.common.basis_utils import real_sph_harm
from ppmat.models.common.runtime import RuntimeMixin
from ppmat.models.common.runtime import runtime_boundary
from ppmat.utils.crystal import get_pbc_distances
from ppmat.utils.scatter import scatter

"""This module is adapted from https://github.com/Open-Catalyst-Project/ocp/tree/master/ocpmodels/models
"""


class Envelope(paddle.nn.Layer):
    def __init__(self, exponent: int):
        super().__init__()
        self.p = exponent + 1
        self.a = -(self.p + 1) * (self.p + 2) / 2
        self.b = self.p * (self.p + 2)
        self.c = -self.p * (self.p + 1) / 2

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        p, a, b, c = self.p, self.a, self.b, self.c
        x_pow_p0 = x.pow(y=p - 1)
        x_pow_p1 = x_pow_p0 * x
        x_pow_p2 = x_pow_p1 * x
        return (1.0 / x + a * x_pow_p0 + b * x_pow_p1 + c * x_pow_p2) * (x < 1.0).to(
            x.dtype
        )


class BesselBasisLayer(paddle.nn.Layer):
    def __init__(
        self, num_radial: int, cutoff: float = 5.0, envelope_exponent: int = 5
    ):
        super().__init__()
        self.cutoff = cutoff
        self.envelope = Envelope(envelope_exponent)
        self.freq = paddle.create_parameter(
            shape=paddle.empty(
                shape=[
                    num_radial,
                ]
            ).shape,
            dtype=paddle.empty(
                shape=[
                    num_radial,
                ]
            )
            .numpy()
            .dtype,
            default_initializer=paddle.nn.initializer.Assign(
                paddle.arange(1, num_radial + 1, dtype=paddle.get_default_dtype())
                * math.pi
            ),
        )
        self.freq.stop_gradient = False

    def forward(self, dist: paddle.Tensor) -> paddle.Tensor:
        dist = dist.unsqueeze(axis=-1) / self.cutoff
        return self.envelope(dist) * (self.freq * dist).sin()


class SphericalBasisLayer(paddle.nn.Layer):
    def __init__(
        self,
        num_spherical: int,
        num_radial: int,
        cutoff: float = 5.0,
        envelope_exponent: int = 5,
    ):
        super().__init__()
        assert num_radial <= 64
        self.num_spherical = num_spherical
        self.num_radial = num_radial
        self.cutoff = cutoff
        self.envelope = Envelope(envelope_exponent)
        bessel_forms = bessel_basis(num_spherical, num_radial)
        sph_harm_forms = real_sph_harm(num_spherical, True, True, True)
        self.sph_funcs = []
        self.bessel_funcs = []
        x, theta = sym.symbols("x theta")
        modules = {"sin": paddle.sin, "cos": paddle.cos}
        for i in range(num_spherical):
            if i == 0:
                sph1 = sym.lambdify([theta], sph_harm_forms[i][0], modules)(0)
                self.sph_funcs.append(float(sph1))
            else:
                sph = sym.lambdify([theta], sph_harm_forms[i][0], modules)
                self.sph_funcs.append(sph)
            for j in range(num_radial):
                bessel = sym.lambdify([x], bessel_forms[i][j], modules)
                self.bessel_funcs.append(bessel)

    def forward(
        self, dist: paddle.Tensor, angle: paddle.Tensor, idx_kj: paddle.Tensor
    ) -> paddle.Tensor:
        dist = dist / self.cutoff
        rbf = paddle.stack(x=[f(dist) for f in self.bessel_funcs], axis=1)
        rbf = self.envelope(dist).unsqueeze(axis=-1) * rbf
        cbf = paddle.stack(
            x=[
                paddle.zeros_like(angle) + self.sph_funcs[0],
                *(function(angle) for function in self.sph_funcs[1:]),
            ],
            axis=1,
        )
        n, k = self.num_spherical, self.num_radial
        out = (rbf[idx_kj].reshape([-1, n, k]) * cbf.reshape([-1, n, 1])).reshape(
            [-1, n * k]
        )
        return out


class EmbeddingBlock(paddle.nn.Layer):
    def __init__(
        self, num_embeddings, num_radial: int, hidden_channels: int, act: Callable
    ):
        super().__init__()
        self.act = act
        self.emb = paddle.nn.Embedding(
            num_embeddings=num_embeddings, embedding_dim=hidden_channels
        )
        self.lin_rbf = paddle.nn.Linear(
            in_features=num_radial, out_features=hidden_channels
        )
        self.lin = paddle.nn.Linear(
            in_features=3 * hidden_channels, out_features=hidden_channels
        )

    def forward(
        self, x: paddle.Tensor, rbf: paddle.Tensor, i: paddle.Tensor, j: paddle.Tensor
    ) -> paddle.Tensor:
        x = self.emb(x)
        rbf = self.act(self.lin_rbf(rbf))
        return self.act(self.lin(paddle.concat(x=[x[i], x[j], rbf], axis=-1)))


class ResidualLayer(paddle.nn.Layer):
    def __init__(self, hidden_channels: int, act: Callable):
        super().__init__()
        self.act = act
        self.lin1 = paddle.nn.Linear(
            in_features=hidden_channels, out_features=hidden_channels
        )
        self.lin2 = paddle.nn.Linear(
            in_features=hidden_channels, out_features=hidden_channels
        )

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        return x + self.act(self.lin2(self.act(self.lin1(x))))


class InteractionPPBlock(paddle.nn.Layer):
    def __init__(
        self,
        hidden_channels,
        int_emb_size,
        basis_emb_size,
        num_spherical,
        num_radial,
        num_before_skip,
        num_after_skip,
        act=swish,
    ):
        super(InteractionPPBlock, self).__init__()
        self.act = act
        self.lin_rbf1 = paddle.nn.Linear(
            in_features=num_radial, out_features=basis_emb_size, bias_attr=False
        )
        self.lin_rbf2 = paddle.nn.Linear(
            in_features=basis_emb_size, out_features=hidden_channels, bias_attr=False
        )
        self.lin_sbf1 = paddle.nn.Linear(
            in_features=num_spherical * num_radial,
            out_features=basis_emb_size,
            bias_attr=False,
        )
        self.lin_sbf2 = paddle.nn.Linear(
            in_features=basis_emb_size, out_features=int_emb_size, bias_attr=False
        )
        self.lin_kj = paddle.nn.Linear(
            in_features=hidden_channels, out_features=hidden_channels
        )
        self.lin_ji = paddle.nn.Linear(
            in_features=hidden_channels, out_features=hidden_channels
        )
        self.lin_down = paddle.nn.Linear(
            in_features=hidden_channels, out_features=int_emb_size, bias_attr=False
        )
        self.lin_up = paddle.nn.Linear(
            in_features=int_emb_size, out_features=hidden_channels, bias_attr=False
        )
        self.layers_before_skip = paddle.nn.LayerList(
            sublayers=[
                ResidualLayer(hidden_channels, act) for _ in range(num_before_skip)
            ]
        )
        self.lin = paddle.nn.Linear(
            in_features=hidden_channels, out_features=hidden_channels
        )
        self.layers_after_skip = paddle.nn.LayerList(
            sublayers=[
                ResidualLayer(hidden_channels, act) for _ in range(num_after_skip)
            ]
        )

    def forward(self, x, rbf, sbf, idx_kj, idx_ji):
        x_ji = self.act(self.lin_ji(x))
        x_kj = self.act(self.lin_kj(x))
        rbf = self.lin_rbf1(rbf)
        rbf = self.lin_rbf2(rbf)
        x_kj = x_kj * rbf
        x_kj = self.act(self.lin_down(x_kj))
        sbf = self.lin_sbf1(sbf)
        sbf = self.lin_sbf2(sbf)
        x_kj = x_kj[idx_kj] * sbf
        x_kj = scatter(x_kj, idx_ji, dim=0, dim_size=x.shape[0])
        x_kj = self.act(self.lin_up(x_kj))
        h = x_ji + x_kj
        for layer in self.layers_before_skip:
            h = layer(h)
        h = self.act(self.lin(h)) + x
        for layer in self.layers_after_skip:
            h = layer(h)
        return h


class OutputPPBlock(paddle.nn.Layer):
    def __init__(
        self,
        num_radial,
        hidden_channels,
        out_emb_channels,
        out_channels,
        num_layers,
        act=swish,
    ):
        super(OutputPPBlock, self).__init__()
        self.act = act
        self.lin_rbf = paddle.nn.Linear(
            in_features=num_radial, out_features=hidden_channels, bias_attr=False
        )
        self.lin_up = paddle.nn.Linear(
            in_features=hidden_channels, out_features=out_emb_channels, bias_attr=True
        )
        self.lins = paddle.nn.LayerList()
        for _ in range(num_layers):
            self.lins.append(
                paddle.nn.Linear(
                    in_features=out_emb_channels, out_features=out_emb_channels
                )
            )
        self.lin = paddle.nn.Linear(
            in_features=out_emb_channels, out_features=out_channels, bias_attr=False
        )

    def forward(self, x, rbf, i, num_nodes=None):
        x = self.lin_rbf(rbf) * x
        x = scatter(x, i, dim=0, dim_size=num_nodes)
        x = self.lin_up(x)
        for lin in self.lins:
            x = self.act(lin(x))
        return self.lin(x)


class DimeNetPlusPlus(RuntimeMixin, paddle.nn.Layer):
    """
    Fast and Uncertainty-Aware Directional Message Passing for
    Non-Equilibrium Molecules, https://arxiv.org/abs/2011.14115

    Args:
        out_channels (int): The number of output channels for the final prediction.
        hidden_channels (int, optional): The dimensionality of hidden feature
            vectors in each convolutional layer. Defaults to 128.
        num_blocks (int, optional): The number of interaction blocks to stack.
            Defaults to 4.
        int_emb_size (int, optional): The size of the embedding vector
            for each atom index. Defaults to 64.
        basis_emb_size (int, optional): The size of the basis embedding used
            in the interaction layers. Defaults to 8.
        out_emb_channels (int, optional): The number of channels after the final
            embedding layer before readout. Defaults to 256.
        num_spherical (int, optional): The number of spherical basis functions to use.
            Defaults to 7.
        num_embeddings (int, optional): The number of distinct atom types to embed.
            Defaults to 95.
        num_radial (int, optional): The number of radial basis functions to use.
            Defaults to 6.
        otf_graph (bool, optional): Whether to construct the interaction graph
            on-the-fly during training. Defaults to False.
        cutoff (float, optional): The cutoff distance (in Å) for neighbor interactions.
            Defaults to 10.0.
        max_num_neighbors (int, optional): The maximum number of neighbors to consider
            for each atom. Defaults to 20.
        envelope_exponent (int, optional): The exponent used in the cutoff envelope
            function to control smooth decay. Defaults to 5.
        num_before_skip (int, optional): The number of convolutional layers
            before each skip connection. Defaults to 1.
        num_after_skip (int, optional): The number of convolutional layers
            after each skip connection. Defaults to 2.
        num_output_layers (int, optional): The number of fully connected layers
            used to produce the final output. Defaults to 3.
        readout (str, optional): The method for aggregating atom features into
            a graph-level feature (“mean” or “sum”). Defaults to "mean".
        property_names (Optional[str], optional): A comma-separated list of
            target property names to predict. Defaults to "formation_energy_per_atom".
        data_mean (float, optional): The mean used for normalizing target values.
            Defaults to 0.0.
        data_std (float, optional): The standard deviation used for
            normalizing target values. Defaults to 1.0.
        loss_type (str, optional): Loss type, can be 'mse_loss' or 'l1_loss'.
            Defaults to "l1_loss".
        act (str, optional): The activation function. Defaults to swish.
        execution_backend (str, optional): Numerical execution backend. Use
            ``"eager"`` or ``"cinn"``. Defaults to ``"eager"``.
        runtime_options (dict, optional): Per-backend runtime options.
    """

    def __init__(
        self,
        out_channels: int,
        hidden_channels: int = 128,
        num_blocks: int = 4,
        int_emb_size: int = 64,
        basis_emb_size: int = 8,
        out_emb_channels: int = 256,
        num_spherical: int = 7,
        num_embeddings: int = 95,
        num_radial: int = 6,
        otf_graph: bool = False,
        cutoff: float = 10.0,
        max_num_neighbors: int = 20,
        envelope_exponent: int = 5,
        num_before_skip: int = 1,
        num_after_skip: int = 2,
        num_output_layers: int = 3,
        readout: str = "mean",
        property_names: Optional[str] = "formation_energy_per_atom",
        data_mean: float = 0.0,
        data_std: float = 1.0,
        loss_type: str = "l1_loss",
        act: str = "swish",
        execution_backend: str = "eager",
        runtime_options: dict | None = None,
    ):
        super().__init__()
        self._init_runtime(execution_backend, runtime_options)
        # store hyperparams
        self.out_channels = out_channels
        self.cutoff = cutoff
        self.max_num_neighbors = max_num_neighbors
        self.otf_graph = otf_graph
        self.readout = readout
        if isinstance(property_names, list):
            self.property_names = property_names[0]
        else:
            assert isinstance(property_names, str)
            self.property_names = property_names
        self.register_buffer(tensor=paddle.to_tensor(data_mean), name="data_mean")
        self.register_buffer(tensor=paddle.to_tensor(data_std), name="data_std")

        # basis layers
        self.rbf = BesselBasisLayer(num_radial, cutoff, envelope_exponent)
        self.sbf = SphericalBasisLayer(
            num_spherical, num_radial, cutoff, envelope_exponent
        )

        # act func
        if act == "swish":
            act = swish
        else:
            raise ValueError(f"Invalid activation function: {act}")

        # embedding and blocks
        self.emb = EmbeddingBlock(num_embeddings, num_radial, hidden_channels, act)
        self.output_blocks = paddle.nn.LayerList(
            [
                OutputPPBlock(
                    num_radial,
                    hidden_channels,
                    out_emb_channels,
                    out_channels,
                    num_output_layers,
                    act,
                )
                for _ in range(num_blocks + 1)
            ]
        )
        self.interaction_blocks = paddle.nn.LayerList(
            [
                InteractionPPBlock(
                    hidden_channels,
                    int_emb_size,
                    basis_emb_size,
                    num_spherical,
                    num_radial,
                    num_before_skip,
                    num_after_skip,
                    act,
                )
                for _ in range(num_blocks)
            ]
        )

        if loss_type == "mse_loss":
            self.loss_fn = paddle.nn.functional.mse_loss
        elif loss_type == "l1_loss":
            self.loss_fn = paddle.nn.functional.l1_loss
        else:
            raise ValueError(f"Unknown loss type {loss_type}.")

    def triplets(self, edge_index, cell_offsets, num_nodes):
        row, col = edge_index
        value = paddle.arange(1, row.shape[0] + 1, dtype="int64")
        # build matrix of edge ids per target
        n = col.shape[0]
        rows = paddle.arange(n).unsqueeze(1)
        cols = paddle.arange(n).unsqueeze(0)
        mask = (col.unsqueeze(1) == col.unsqueeze(0)) & (cols <= rows)
        col_ = mask.astype("int64").sum(axis=1) - 1
        mat = paddle.scatter_nd(
            paddle.stack([col, col_], axis=1),
            value,
            shape=[num_nodes, col_.max().item() + 1],
        )
        idx_kj = mat[row][mat[row] > 0] - 1
        tmp = paddle.nonzero(mat[row], as_tuple=False)
        idx_ji = tmp[:, 0]
        idx_k = paddle.index_select(row, idx_kj, axis=0)
        idx_j = paddle.index_select(row, idx_ji, axis=0)
        idx_i = paddle.index_select(col, idx_ji, axis=0)
        # Remove i == k triplets with the same cell offset.
        cell_offset_kji = cell_offsets[idx_kj] + cell_offsets[idx_ji]
        mask2 = (idx_i != idx_k) | paddle.any(cell_offset_kji != 0, axis=-1)
        return (
            col,
            row,
            idx_i[mask2],
            idx_j[mask2],
            idx_k[mask2],
            idx_kj[mask2],
            idx_ji[mask2],
        )

    def normalize(self, tensor):
        return (tensor - self.data_mean) / self.data_std

    def unnormalize(self, tensor):
        return tensor * self.data_std + self.data_mean

    @runtime_boundary("forward")
    def _forward(self, data):
        #  The data in data['graph'] is numpy.ndarray, convert it to paddle.Tensor
        data["graph"] = data["graph"].tensor()

        # unpack graph dict
        graph = data["graph"]
        batch = graph.graph_node_id
        lattices = graph.node_feat["lattice"]
        pos = graph.node_feat["cart_coords"]
        frac = graph.node_feat["frac_coords"]
        edge_index = graph.edges
        to_jimages = graph.edge_feat["pbc_offset"]
        num_atoms = graph.node_feat["num_atoms"]
        num_bonds = graph.edge_feat["num_edges"]
        atom_types = graph.node_feat["atom_types"]

        out = get_pbc_distances(
            frac,
            edge_index.T,
            lattices,
            to_jimages,
            num_atoms,
            num_bonds,
            return_offsets=True,
        )
        edge_index = out["edge_index"]
        dist = out["distances"]
        offsets = out["offsets"]
        i, j, idx_i, idx_j, idx_k, idx_kj, idx_ji = self.triplets(
            edge_index, to_jimages, num_nodes=atom_types.shape[0]
        )
        # compute angles
        pos_i = pos[idx_i]
        pos_j = pos[idx_j]
        pos_ji = pos_j - pos_i + offsets[idx_ji]
        pos_kj = pos[idx_k] - pos_j + offsets[idx_kj]
        a = (pos_ji * pos_kj).sum(axis=-1)
        b = paddle.cross(pos_ji, pos_kj).norm(axis=-1)
        angle = paddle.atan2(b, a)

        # basis expansions
        rbf = self.rbf(dist)
        sbf = self.sbf(dist, angle, idx_kj)
        x = self.emb(atom_types, rbf, i, j)

        # output and interactions
        P = self.output_blocks[0](x, rbf, i, num_nodes=pos.shape[0])
        for interact, out_block in zip(self.interaction_blocks, self.output_blocks[1:]):
            x = interact(x, rbf, sbf, idx_kj, idx_ji)
            P += out_block(x, rbf, i, num_nodes=pos.shape[0])

        # readout
        energy = scatter(P, batch, dim=0, reduce=self.readout)
        return energy

    def forward(self, data, return_loss=True, return_prediction=True):
        assert (
            return_loss or return_prediction
        ), "At least one of return_loss or return_prediction must be True."
        pred = self._forward(data)

        loss_dict = {}
        if return_loss:
            label = data[self.property_names]
            label = self.normalize(label)
            loss = self.loss_fn(
                input=pred,
                label=label,
            )
            loss_dict["loss"] = loss

        prediction = {}
        if return_prediction:
            pred = self.unnormalize(pred)
            prediction[self.property_names] = pred
        return {"loss_dict": loss_dict, "pred_dict": prediction}

    @paddle.no_grad()
    def predict(self, samples):
        is_list = isinstance(samples, list)
        samples = samples if is_list else [samples]

        results = []
        for sample in samples:
            result = self._forward({"graph": sample})
            result = self.unnormalize(result).numpy()[0, 0]
            results.append({self.property_names: result})
        return results if is_list else results[0]

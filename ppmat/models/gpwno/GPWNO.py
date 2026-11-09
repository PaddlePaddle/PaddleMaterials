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

import json

import numpy as np
import paddle

from ppmat.datasets.build_grid import BuildGrid
from ppmat.datasets.graph_utils.infgcn_graph_utils import radius
from ppmat.datasets.graph_utils.infgcn_graph_utils import radius_graph
from ppmat.models.common.activation import NormActivation
from ppmat.models.common.activation import ScalarActivation
from ppmat.models.common.e3nn import o3
from ppmat.models.common.e3nn.math import soft_one_hot_linspace
from ppmat.models.common.e3nn.nn import FullyConnectedNet
from ppmat.models.common.orbital import GaussianOrbital
from ppmat.utils.crystal import pbc_vec
from ppmat.utils.scatter import scatter


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
        Equivariant graph convolution for the Gaussian and plane-wave branches.
        If the tensor product is fully connected, we have (for every path)

        .. math::
            z_w=\\sum_{uv}w_{uvw}x_u\\otimes y_v=\\sum_{u}w_{uw}x_u \\otimes y

        Else, we have

        .. math::
            z_u=x_u\\otimes \\sum_v w_{uv}y_v=w_u (x_u\\otimes y)

        Here, uvw are radial (channel) indices of the first input, second input,
        and output, respectively. The second input is always the spherical
        harmonics of the edge vector, so the index v can be safely ignored.

        :param irreps_in: irreducible representations of input node features
        :param irreps_out: irreducible representations of output node features
        :param irreps_edge: irreducible representations of edge features
        :param radial_embed_size: embedding size of the edge length
        :param num_radial_layer: number of hidden layers in the radial network
        :param radial_hidden_size: hidden size of the radial network
        :param is_fc: whether to use fully connected tensor product
        :param use_sc: whether to use self-connection
        :param irrep_normalization: representation normalization passed to
            `o3.FullyConnectedTensorProduct`
        :param path_normalization: path normalization passed to
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
        self.fc = FullyConnectedNet(
            [radial_embed_size]
            + num_radial_layer * [radial_hidden_size]
            + [self.tp.weight_numel],
            paddle.nn.functional.silu,
        )
        self.sc = None
        if self.use_sc:
            self.sc = o3.Linear(self.irreps_in, self.irreps_out)

    def forward(
        self, edge_index, node_feat, edge_feat, edge_embed, dim_size=None, reduce="sum"
    ):
        src, dst = edge_index
        weight = self.fc(edge_embed)
        out = self.tp(node_feat[src], edge_feat, weight=weight)
        out = scatter(out, dst, dim=0, dim_size=dim_size, reduce=reduce)
        if self.use_sc:
            out = out + self.sc(node_feat)
        return out


class SpectralConv3d(paddle.nn.Layer):
    """Apply a learned spectral transform to a three-dimensional real field."""

    def __init__(self, in_channels, out_channels, modes1, modes2, modes3):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.scale = 1 / (in_channels * out_channels)
        weight_shape = [in_channels, out_channels, modes1, modes2, modes3]
        for name in ("weights1", "weights2", "weights3", "weights4"):
            weight = self.scale * paddle.rand(weight_shape, dtype=paddle.complex64)
            self.add_parameter(name, paddle.nn.Parameter(weight))

    def compl_mul3d(self, input, weights):
        return paddle.einsum("bixyz,ioxyz->boxyz", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        x_ft = paddle.fft.rfftn(x, dim=[-3, -2, -1], norm="ortho")
        out_ft = paddle.zeros(
            [
                batchsize,
                self.out_channels,
                x.shape[-3],
                x.shape[-2],
                x.shape[-1] // 2 + 1,
            ],
            dtype=paddle.complex64,
            device=x.device,
        )
        out_ft[:, :, : self.modes1, : self.modes2, : self.modes3] = self.compl_mul3d(
            x_ft[:, :, : self.modes1, : self.modes2, : self.modes3], self.weights1
        )
        out_ft[:, :, -self.modes1 :, : self.modes2, : self.modes3] = self.compl_mul3d(
            x_ft[:, :, -self.modes1 :, : self.modes2, : self.modes3], self.weights2
        )
        out_ft[:, :, : self.modes1, -self.modes2 :, : self.modes3] = self.compl_mul3d(
            x_ft[:, :, : self.modes1, -self.modes2 :, : self.modes3], self.weights3
        )
        out_ft[:, :, -self.modes1 :, -self.modes2 :, : self.modes3] = self.compl_mul3d(
            x_ft[:, :, -self.modes1 :, -self.modes2 :, : self.modes3], self.weights4
        )
        # Use the original spatial shape for the inverse 3D real FFT.
        x = paddle.fft.irfftn(
            out_ft,
            s=(x.shape[-3], x.shape[-2], x.shape[-1]),
            dim=[-3, -2, -1],
            norm="ortho",
        )
        return x


class SpectralConv3d_FFNO(paddle.nn.Layer):
    """Sum learned one-dimensional spectral transforms along the three axes."""

    def __init__(self, in_channels, out_channels, modes1, modes2, modes3):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes_x = modes1
        self.modes_y = modes2
        self.modes_z = modes3
        self.fourier_weight = paddle.nn.ParameterList(parameters=[])
        for n_modes in [self.modes_x, self.modes_y, self.modes_z]:
            weight = paddle.randn(
                [in_channels, out_channels, n_modes, 2], dtype=paddle.float32
            )
            paddle.nn.init.xavier_normal_(weight)
            self.fourier_weight.append(paddle.nn.Parameter(weight))

    def forward(self, x):
        batch_size, channels, size_x, size_y, size_z = x.shape

        # Spectral convolution along the z axis.
        x_ftz = paddle.fft.rfftn(x, dim=[-1], norm="ortho")
        out_ft = x_ftz.new_zeros(batch_size, channels, size_x, size_y, size_z // 2 + 1)
        out_ft[:, :, :, :, : self.modes_z] = paddle.einsum(
            "bixyz,ioz->boxyz",
            x_ftz[:, :, :, :, : self.modes_z],
            paddle.view_as_complex(self.fourier_weight[2]),
        )
        xz = paddle.fft.irfft(out_ft, n=size_z, dim=-1, norm="ortho")

        # Spectral convolution along the y axis.
        x_fty = paddle.fft.rfftn(x, dim=[-2], norm="ortho")
        out_ft = x_ftz.new_zeros(batch_size, channels, size_x, size_y // 2 + 1, size_z)
        out_ft[:, :, :, : self.modes_y, :] = paddle.einsum(
            "bixyz,ioy->boxyz",
            x_fty[:, :, :, : self.modes_y, :],
            paddle.view_as_complex(self.fourier_weight[1]),
        )
        xy = paddle.fft.irfft(out_ft, n=size_y, dim=-2, norm="ortho")

        # Spectral convolution along the x axis.
        x_ftx = paddle.fft.rfftn(x, dim=[-3], norm="ortho")
        out_ft = x_ftz.new_zeros(batch_size, channels, size_x // 2 + 1, size_y, size_z)
        out_ft[:, :, : self.modes_x, :, :] = paddle.einsum(
            "bixyz,iox->boxyz",
            x_ftx[:, :, : self.modes_x, :, :],
            paddle.view_as_complex(self.fourier_weight[0]),
        )
        xx = paddle.fft.irfft(out_ft, n=size_x, dim=-3, norm="ortho")

        x = xx + xy + xz
        return x


class PWNO(paddle.nn.Layer):
    """Refine probe features with spectral and point-wise convolutions.

    Inputs are probe features with shape ``[batch, nx, ny, nz, channels]`` and
    three normalized coordinates per probe. The output retains the spatial
    dimensions and has ``width`` channels.
    """

    def __init__(
        self,
        modes1,
        modes2,
        modes3,
        width,
        num_fourier_time,
        padding=0,
        num_layers=2,
        using_ff=False,
    ):
        super().__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.width = width
        self.num_fourier_time = num_fourier_time
        self.padding = padding
        self.fc0 = paddle.compat.nn.Linear(self.num_fourier_time + 3, self.width)
        self.num_layers = num_layers
        if using_ff:
            self.conv = paddle.nn.LayerList(
                [
                    SpectralConv3d_FFNO(
                        self.width, self.width, self.modes1, self.modes2, self.modes3
                    )
                    for _ in range(self.num_layers)
                ]
            )
        else:
            self.conv = paddle.nn.LayerList(
                [
                    SpectralConv3d(
                        self.width, self.width, self.modes1, self.modes2, self.modes3
                    )
                    for _ in range(self.num_layers)
                ]
            )
        self.w = paddle.nn.LayerList(
            [
                paddle.nn.Conv3d(self.width, self.width, 1)
                for _ in range(self.num_layers)
            ]
        )

    def forward(self, residue, fourier_grid):
        x = paddle.cat([residue, fourier_grid], dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 4, 1, 2, 3)
        if self.padding != 0:
            x = paddle.compat.nn.functional.pad(
                x, [0, self.padding, 0, self.padding, 0, self.padding]
            )
        for i in range(self.num_layers):
            x1 = self.conv[i](x)
            x2 = self.w[i](x)
            x = x1 + x2
            if i != self.num_layers - 1:
                x = paddle.nn.functional.gelu(x)
        if self.padding != 0:
            x = x[..., :, : -self.padding, : -self.padding, : -self.padding]
        x = x.permute(0, 2, 3, 4, 1)
        return x


class GPWNO(paddle.nn.Layer):
    def __init__(
        self,
        n_atom_type,
        num_radial,
        num_spherical,
        radial_embed_size,
        radial_hidden_size,
        num_fourier=64,
        num_fourier_time=4,
        num_radial_layer=2,
        num_gcn_layer=3,
        cutoff=3.0,
        grid_cutoff=3.0,
        probe_cutoff=3.0,
        is_fc=True,
        gauss_start=0.5,
        gauss_end=5.0,
        activation="norm",
        residual=True,
        pbc=False,
        width=20,
        padding=6,
        use_max_cell=True,
        max_cell_size=22,
        equivariant_frame=True,
        model_sharing=True,
        num_local_gcn_layers=None,
        input_local_density=None,
        using_ff=True,
        scalar_mask=True,
        mask_cutoff=3.0,
        scalar_inv=False,
        num_spherical_RNO=None,
        positive_output=False,
        atomic_gauss_dist=False,
        input_dist=False,
        atom_info=None,
        fourier_mode=0,
        vocab=None,
        target_name="density",
        loss_eps=1e-8,
        **kwargs,
    ):
        """
        Implement the GPWNO model for electron density estimation
        :param n_atom_type: number of atom types
        :param num_radial: number of radial basis
        :param num_spherical: maximum spherical harmonic degree for each radial
            basis; the number of spherical basis functions is
            (num_spherical + 1)^2
        :param radial_embed_size: embedding size of the edge length
        :param radial_hidden_size: hidden size of the radial network
        :param num_radial_layer: number of hidden layers in the radial network
        :param num_gcn_layer: number of GPWNO layers
        :param cutoff: cutoff distance for building the molecular graph
        :param grid_cutoff: cutoff distance for building the grid-atom graph
        :param is_fc: whether the GPWNO layer should use a fully connected
            tensor product
        :param gauss_start: start coefficient of the Gaussian radial basis
        :param gauss_end: end coefficient of the Gaussian radial basis
        :param activation: activation type, either 'scalar' or 'norm'
        :param residual: whether to use the residue prediction layer
        :param pbc: whether the data satisfy the periodic boundary condition;
            restore the boundary atom images used by the original MP reader
        :param num_local_gcn_layers: number of layers in the separate Gaussian
            coefficient network; defaults to 3
        :param input_local_density: append the Gaussian density evaluated at
            Fourier probes to the plane-wave input; defaults to False

        The local Gaussian branch uses InfGCN-style equivariant message passing,
        but does not instantiate the standalone InfGCN model. Legacy configuration
        keys ``num_infgcn_layer`` and ``input_infgcn`` remain supported.
        """
        super().__init__()
        legacy_layers = kwargs.pop("num_infgcn_layer", None)
        if num_local_gcn_layers is None:
            num_local_gcn_layers = 3 if legacy_layers is None else legacy_layers
        elif legacy_layers is not None and legacy_layers != num_local_gcn_layers:
            raise ValueError("Conflicting local GCN layer counts in the configuration.")
        legacy_density = kwargs.pop("input_infgcn", None)
        if input_local_density is None:
            input_local_density = False if legacy_density is None else legacy_density
        elif legacy_density is not None and legacy_density != input_local_density:
            raise ValueError(
                "Conflicting local density input flags in the configuration."
            )
        self.n_atom_type = n_atom_type
        self.vocab = vocab
        self.target_name = target_name
        self.loss_eps = loss_eps
        self.num_radial = num_radial
        self.num_spherical = num_spherical
        self.radial_embed_size = radial_embed_size
        self.radial_hidden_size = radial_hidden_size
        self.num_radial_layer = num_radial_layer
        self.num_gcn_layer = num_gcn_layer
        self.cutoff = cutoff
        self.grid_cutoff = grid_cutoff
        self.is_fc = is_fc
        self.gauss_start = gauss_start
        self.gauss_end = gauss_end
        self.activation = activation
        self.residual = residual
        self.pbc = pbc
        self.num_fourier = num_fourier
        self.num_fourier_time = num_fourier_time
        self.probe_cutoff = probe_cutoff
        self.use_max_cell = use_max_cell
        self.max_cell_size = max_cell_size
        self.equivariant_frame = equivariant_frame
        self.model_sharing = model_sharing
        self.num_local_gcn_layers = num_local_gcn_layers
        self.input_local_density = input_local_density
        self.scalar_mask = scalar_mask
        self.mask_cutoff = mask_cutoff
        self.scalar_inv = scalar_inv
        self.positive_output = positive_output
        self.atomic_gauss_dist = atomic_gauss_dist
        self.input_dist = input_dist
        self.atom_info = atom_info
        self.fourier_mode = fourier_mode
        self.num_spherical_RNO = num_spherical_RNO
        if model_sharing or num_spherical_RNO is None:
            self.num_spherical_RNO = num_spherical
        assert activation in ["scalar", "norm"]
        self.embedding = paddle.nn.Embedding(n_atom_type, num_radial)
        self.irreps_sh = o3.Irreps.spherical_harmonics(num_spherical, p=1)
        self.irreps_feat = (self.irreps_sh * num_radial).sort().irreps.simplify()
        self.irreps_sh_RNO = o3.Irreps.spherical_harmonics(self.num_spherical_RNO, p=1)
        self.irreps_feat_RNO = (
            (self.irreps_sh_RNO * num_radial).sort().irreps.simplify()
        )
        self.gcns_RNO = paddle.nn.LayerList(
            [
                GCNLayer(
                    f"{num_radial}x0e" if i == 0 else self.irreps_feat_RNO,
                    self.irreps_feat_RNO,
                    self.irreps_sh_RNO,
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
                self.irreps_feat, paddle.nn.functional.silu, paddle.sigmoid
            )
            self.act_RNO = ScalarActivation(
                self.irreps_feat_RNO, paddle.nn.functional.silu, paddle.sigmoid
            )
        else:
            self.act = NormActivation(self.irreps_feat)
            self.act_RNO = NormActivation(self.irreps_feat_RNO)
        if not self.model_sharing:
            self.local_gcns = paddle.nn.LayerList(
                [
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
                    for i in range(self.num_local_gcn_layers)
                ]
            )
        self.residue = None
        if self.residual:
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
        self.probe_gcn = GCNLayer(
            self.irreps_feat_RNO,
            f"{num_fourier_time}x0e",
            self.irreps_sh_RNO,
            radial_embed_size,
            num_radial_layer,
            radial_hidden_size,
            is_fc=True,
            use_sc=False,
            **kwargs,
        )
        self.act_probe = NormActivation(f"{num_fourier_time}x0e")
        self.mode = (
            self.num_fourier // 2 if self.fourier_mode == 0 else self.fourier_mode
        )
        self.width = width
        if self.input_local_density:
            self.num_fourier_time += 1
        if self.input_dist:
            self.num_fourier_time += 1
        if self.atomic_gauss_dist:
            self.num_fourier_time += 10
        self.PWNO = PWNO(
            modes1=self.mode,
            modes2=self.mode,
            modes3=self.mode,
            width=self.width,
            num_fourier_time=self.num_fourier_time,
            padding=padding,
            using_ff=using_ff,
        )
        self.scalar_field_gcn = GCNLayer(
            f"{self.width}x0e",
            "0e",
            self.irreps_sh_RNO,
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
        if self.atomic_gauss_dist and atom_info is not None:
            with open(atom_info) as f:
                atom_info = json.load(f)
            self.atom_radius = paddle.to_tensor(
                [info["radius"] for info in atom_info], dtype="float32"
            )

    def set_state_dict(self, state_dict, use_structured_name=True):
        """Load older checkpoints whose Gaussian layers were named ``infgcns``."""
        if use_structured_name:
            renamed = {}
            for name, value in state_dict.items():
                target = (
                    "local_gcns." + name[len("infgcns.") :]
                    if name.startswith("infgcns.")
                    else name
                )
                if target in renamed:
                    raise ValueError(
                        f"Duplicate checkpoint key after renaming: {target}"
                    )
                renamed[target] = value
            state_dict = renamed
        return super().set_state_dict(
            state_dict, use_structured_name=use_structured_name
        )

    def _forward(self, data):
        grid = data["grid_coord"]
        graph = data["graph"].tensor()
        atom_types = graph.node_feat["x"]
        atom_coord = graph.node_feat["cart_coords"]
        graph_batch = graph.graph_node_id.astype("int64")
        info = data.get("info")

        chunk_size = data.get("grid_batch_size")
        if not self.training and chunk_size and grid.shape[1] > chunk_size:
            predictions = []
            aux = {}
            for start in range(0, grid.shape[1], chunk_size):
                prediction, aux = self._forward_density(
                    atom_types,
                    atom_coord,
                    grid[:, start : start + chunk_size],
                    graph_batch,
                    info,
                )
                predictions.append(prediction)
            return paddle.concat(predictions, axis=1), aux

        return self._forward_density(
            atom_types,
            atom_coord,
            grid,
            graph_batch,
            info,
        )

    def forward(self, data, return_loss=True, return_prediction=True):
        """Run field prediction through the PaddleMaterials model protocol."""

        assert (
            return_loss or return_prediction
        ), "At least one of return_loss or return_prediction must be True."

        pred, aux = self._forward(data)
        density_gt = data.get(self.target_name)
        mask = data.get("density_mask")
        masked_pred = pred if mask is None else pred * mask.astype(pred.dtype)

        loss_dict = {}
        if return_loss:
            if density_gt is None:
                raise ValueError(
                    f"data[{self.target_name!r}] must not be None when "
                    "return_loss is True."
                )
            loss, mae = self._compute_loss(pred, density_gt, mask)
            loss_dict["loss"] = loss
            loss_dict["mae"] = mae

        pred_dict = {}
        if return_prediction:
            pred_dict[self.target_name] = masked_pred
        return {
            "loss_dict": loss_dict,
            "pred_dict": pred_dict,
            "aux_dict": aux,
        }

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
                    grid.cartesian_coordinates(), dtype="float32"
                ).reshape([1, -1, 3])
                sample["info"] = {
                    "cell": paddle.to_tensor(grid.cell_vectors, dtype="float32")
                }
            prediction, _ = self._forward(sample)
            results.append({self.target_name: prediction.reshape([-1]).detach().cpu()})

        return results if is_list else results[0]

    def _compute_loss(self, pred, label, mask=None):
        if mask is not None:
            pred = pred * mask
            label = label * mask
            denom = paddle.sum(mask) + self.loss_eps
            loss = paddle.sum((pred - label) ** 2) / denom
            mae = paddle.sum(paddle.abs(pred - label)) / (
                paddle.sum(paddle.abs(label)) + self.loss_eps
            )
        else:
            loss = paddle.mean((pred - label) ** 2)
            mae = paddle.sum(paddle.abs(pred - label)) / (
                paddle.sum(paddle.abs(label)) + self.loss_eps
            )
        return loss, mae

    @staticmethod
    def _expand_boundary_atoms(atom_types, atom_coord, batch_idx, cell):
        """Restore the closed-cell boundary images used to train MP weights.

        The legacy reader copied atoms on lower cell faces to the corresponding
        upper faces, not to a full 27-cell supercell. Shared field readers now
        return only the original atoms. Keep this compatibility step local to
        GPWNO and skip images already present in the input. NumPy selects the
        discrete image topology; tensor gathers retain coordinate/cell gradients.
        """
        fractional = paddle.bmm(
            atom_coord.unsqueeze(1), paddle.linalg.inv(cell)[batch_idx]
        ).squeeze(1)
        fractional_np = fractional.detach().numpy()
        types_np = atom_types.numpy()
        batches_np = batch_idx.numpy()
        directions = np.asarray(
            [
                [0, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
                [0, 1, 1],
                [1, 0, 1],
                [1, 1, 0],
                [1, 1, 1],
            ],
            dtype=np.int64,
        )
        tolerance = 1e-6

        existing = {}
        for index, position in enumerate(fractional_np):
            key = (int(batches_np[index]), int(types_np[index]))
            existing.setdefault(key, []).append(position)
        sources, shifts = [], []
        for index, position in enumerate(fractional_np):
            key = (int(batches_np[index]), int(types_np[index]))
            for direction in directions:
                shifted = position + direction
                if direction.any():
                    if np.any(shifted > 1.0 + tolerance):
                        continue
                    if np.any(
                        np.all(
                            np.isclose(existing[key], shifted, atol=tolerance, rtol=0),
                            axis=1,
                        )
                    ):
                        continue
                    existing[key].append(shifted)
                sources.append(index)
                shifts.append(direction)
        if len(sources) == atom_coord.shape[0]:
            return atom_types, atom_coord, batch_idx

        sources = paddle.to_tensor(sources, dtype="int64", place=atom_coord.place)
        shifts = paddle.to_tensor(
            np.asarray(shifts), dtype=atom_coord.dtype, place=atom_coord.place
        )
        expanded_batch = batch_idx[sources]
        offset = paddle.bmm(shifts.unsqueeze(1), cell[expanded_batch]).squeeze(1)
        return atom_types[sources], atom_coord[sources] + offset, expanded_batch

    def _forward_density(self, atom_types, atom_coord, grid, batch_idx, info):
        device = atom_coord.place

        cell = None
        if info is not None and "cell" in info:
            cell = paddle.to_tensor(info["cell"], dtype=atom_coord.dtype)
            if cell.ndim == 2:
                cell = cell.unsqueeze(0)
            cell = cell.to(device)

        atom_types = atom_types.to(device)
        grid = grid.to(device)

        if self.pbc:
            if cell is None:
                raise ValueError("Periodic GPWNO requires info['cell'].")
            atom_types, atom_coord, batch_idx = self._expand_boundary_atoms(
                atom_types, atom_coord, batch_idx, cell
            )

        feat = self.embedding(atom_types)
        n_graph, n_sample = grid.shape[0], grid.shape[1]

        if self.use_max_cell and self.equivariant_frame:
            atom_center = scatter(atom_coord, batch_idx, dim=0, reduce="mean")
            atom_coord = atom_coord - atom_center[batch_idx]
            grid = grid - atom_center.unsqueeze(1)
        edge_index = radius_graph(atom_coord, self.cutoff, batch_idx, loop=False)
        src, dst = edge_index
        edge_vec = atom_coord[src] - atom_coord[dst]
        edge_len = edge_vec.norm(dim=-1) + 1e-08
        edge_feat = o3.spherical_harmonics(
            list(range(self.num_spherical + 1)),
            edge_vec / edge_len[..., None],
            normalize=False,
            normalization="integral",
        )
        edge_embed = soft_one_hot_linspace(
            edge_len,
            start=0.0,
            end=self.cutoff,
            number=self.radial_embed_size,
            basis="gaussian",
            cutoff=False,
        ).mul(self.radial_embed_size**0.5)
        edge_feat_RNO = o3.spherical_harmonics(
            list(range(self.num_spherical_RNO + 1)),
            edge_vec / edge_len[..., None],
            normalize=False,
            normalization="integral",
        )
        if not self.model_sharing:
            local_feat = feat
            for i, gcn in enumerate(self.local_gcns):
                local_feat = gcn(
                    edge_index,
                    local_feat,
                    edge_feat,
                    edge_embed,
                    dim_size=atom_types.size(0),
                )
                if i != self.num_local_gcn_layers - 1:
                    local_feat = self.act(local_feat)
        for i, gcn in enumerate(self.gcns_RNO):
            feat = gcn(
                edge_index, feat, edge_feat_RNO, edge_embed, dim_size=atom_types.size(0)
            )
            if i != self.num_gcn_layer - 1:
                feat = self.act_RNO(feat)
        density_feat = feat if self.model_sharing else local_feat
        bins_lin = paddle.linspace(0, 1, self.num_fourier).to(atom_coord.place)
        if self.pbc:
            half = len(bins_lin) // 2
            super_bins = paddle.cat(
                [bins_lin[half:-1] - 1, bins_lin, bins_lin[1:half] + 1]
            ).to(atom_coord.place)
            super_bins = paddle.meshgrid(super_bins, super_bins, super_bins)
            super_probe = paddle.stack(super_bins, dim=-1).to(atom_coord.place)
            bins_idx = paddle.arange(len(bins_lin))
            bins_idx[-1] = 0
            super_bins_idx = paddle.cat(
                [bins_idx[half:-1], bins_idx, bins_idx[1:half]]
            ).to(atom_coord.place)
            smart_idx = (
                paddle.arange((len(bins_idx) - 1) ** 3)
                .reshape(len(bins_idx) - 1, len(bins_idx) - 1, len(bins_idx) - 1)
                .to(atom_coord.place)
            )
            super_probe_idx_help = paddle.stack(
                paddle.meshgrid(super_bins_idx, super_bins_idx, super_bins_idx), dim=-1
            ).reshape(-1, 3)
            super_probe_idx = smart_idx[
                super_probe_idx_help[:, 0],
                super_probe_idx_help[:, 1],
                super_probe_idx_help[:, 2],
            ].reshape(-1)
            del super_probe_idx_help
        if self.use_max_cell and self.equivariant_frame:
            bins_lin -= 0.5
        bins = paddle.meshgrid(bins_lin, bins_lin, bins_lin)
        probe = paddle.stack(bins, dim=-1).to(atom_coord.place)
        cell_inp = cell
        if self.use_max_cell:
            if self.max_cell_size is None:
                self.max_cell_size = 20
            if self.equivariant_frame:
                num_batch = grid.size(0)
                new_cell = []
                for i in range(num_batch):
                    atom_bat = atom_coord[batch_idx == i]
                    atoms_centered = atom_bat - atom_bat.mean(dim=0)
                    R = paddle.matmul(atoms_centered.t(), atoms_centered)
                    _, vec = paddle.linalg.eigh(x=R)
                    vec /= paddle.linalg.norm(vec, dim=0)
                    new_cell.append(vec)
                new_cell = paddle.stack(new_cell, dim=0)
                max_cell = new_cell * self.max_cell_size
            else:
                max_cell_tensor = np.eye(3) * self.max_cell_size
                max_cell = paddle.to_tensor(max_cell_tensor, dtype=atom_coord.dtype).to(
                    atom_coord.place
                )
                max_cell = max_cell.unsqueeze(0).repeat(grid.size(0), 1, 1)
            cell_inp = max_cell
        probe = paddle.einsum("ijkl,blm->bijkm", probe, cell_inp).detach()
        probe_flat = probe.reshape(-1, 3)
        probe_batch = paddle.arange(grid.size(0), device=grid.device).repeat_interleave(
            len(bins_lin) ** 3
        )
        if self.pbc:
            super_probe = paddle.einsum(
                "ijkl,blm->bijkm", super_probe, cell_inp
            ).detach()
            super_probe_flat = super_probe.reshape(-1, 3)
            super_probe_batch = paddle.arange(
                grid.size(0), device=grid.device
            ).repeat_interleave(len(super_bins_idx) ** 3)
            super_probe_idx_batch = paddle.arange(
                grid.size(0), device=grid.device
            ).repeat_interleave(len(super_probe_idx)) * len(
                bins_lin
            ) ** 3 + super_probe_idx.repeat(
                grid.size(0)
            )
        if self.pbc:
            super_probe_dst, probe_src = radius(
                atom_coord,
                super_probe_flat,
                self.probe_cutoff,
                batch_idx,
                super_probe_batch,
            )
            probe_dst = super_probe_idx_batch[super_probe_dst]
        else:
            probe_dst, probe_src = radius(
                atom_coord, probe_flat, self.probe_cutoff, batch_idx, probe_batch
            )
        avg_probe_degree = (
            probe_dst.size(0) / probe_flat.size(0) if probe_flat.size(0) != 0 else 0
        )
        avg_probe_exist_degree = (
            probe_dst.size(0) / probe_dst.unique().size(0)
            if probe_dst.unique().size(0) != 0
            else 0
        )
        avg_probe_exist_rate = (
            probe_dst.unique().size(0) / probe_flat.size(0)
            if probe_dst.unique().size(0) != 0
            else 0
        )
        if self.pbc:
            probe_edge = super_probe_flat[super_probe_dst] - atom_coord[probe_src]
        else:
            probe_edge = probe_flat[probe_dst] - atom_coord[probe_src]
        probe_len = paddle.norm(probe_edge, dim=-1) + 1e-08
        probe_edge_feat = o3.spherical_harmonics(
            list(range(self.num_spherical_RNO + 1)),
            probe_edge / (probe_len[..., None] + 1e-08),
            normalize=False,
            normalization="integral",
        )
        probe_edge_embed = soft_one_hot_linspace(
            probe_len,
            start=0.0,
            end=self.probe_cutoff,
            number=self.radial_embed_size,
            basis="gaussian",
            cutoff=False,
        ).mul(self.radial_embed_size**0.5)
        probe_feat = self.probe_gcn(
            (probe_src, probe_dst),
            feat,
            probe_edge_feat,
            probe_edge_embed,
            dim_size=probe_flat.size(0),
            reduce="mean",
        )
        probe_feat = self.act_probe(probe_feat)
        probe_feat = probe_feat.reshape(
            grid.size(0), self.num_fourier, self.num_fourier, self.num_fourier, -1
        )
        minimal_dist_grid = paddle.zeros(grid.size(0), grid.size(1))
        for b_idx in range(grid.size(0)):
            if self.pbc:
                minimal_dist_grid[b_idx] = paddle.compat.min(
                    paddle.compat.min(
                        paddle.norm(
                            grid[b_idx].unsqueeze(1).unsqueeze(1)
                            - atom_coord[batch_idx == b_idx].unsqueeze(0).unsqueeze(0)
                            + paddle.stack(
                                [
                                    (
                                        i * cell[b_idx][0]
                                        + j * cell[b_idx][1]
                                        + k * cell[b_idx][2]
                                    )
                                    for i in [-1, 0, 1]
                                    for j in [-1, 0, 1]
                                    for k in [-1, 0, 1]
                                ],
                                dim=0,
                            ).reshape(1, 27, 1, 3),
                            dim=-1,
                        ),
                        dim=-2,
                    )[0],
                    dim=-1,
                )[0]
            else:
                minimal_dist_grid[b_idx] = paddle.compat.min(
                    paddle.norm(
                        grid[b_idx].unsqueeze(1)
                        - atom_coord[batch_idx == b_idx].unsqueeze(0),
                        dim=-1,
                    ),
                    dim=-1,
                )[0]
        mask = minimal_dist_grid > self.mask_cutoff
        atomic_dist_probe = paddle.zeros(grid.size(0), self.num_fourier**3, 10).to(
            atom_coord.place
        )
        alpha = paddle.tensor([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]).to(
            atom_coord.place
        )
        alpha = alpha.reshape(1, 1, 10)
        minimal_dist_probe = paddle.zeros(grid.size(0), self.num_fourier**3).to(
            atom_coord.place
        )
        for b_idx in range(grid.size(0)):
            if self.pbc:
                pr = probe.reshape(grid.size(0), -1, 3)
                dist_min = paddle.compat.min(
                    paddle.norm(
                        pr[b_idx].unsqueeze(1).unsqueeze(1)
                        - atom_coord[batch_idx == b_idx].unsqueeze(0).unsqueeze(0)
                        + paddle.stack(
                            [
                                (
                                    i * cell[b_idx][0]
                                    + j * cell[b_idx][1]
                                    + k * cell[b_idx][2]
                                )
                                for i in [-1, 0, 1]
                                for j in [-1, 0, 1]
                                for k in [-1, 0, 1]
                            ],
                            dim=0,
                        ).reshape(1, 27, 1, 3),
                        dim=-1,
                    ),
                    dim=-2,
                )[0]
                if self.atomic_gauss_dist and self.atom_info is not None:
                    self.atom_radius = self.atom_radius.to(atom_coord.place)
                    atomic_dist_probe[b_idx] = paddle.sum(
                        paddle.exp(
                            -alpha
                            * (
                                dist_min
                                / self.atom_radius[
                                    atom_types[batch_idx == b_idx]
                                ].unsqueeze(0)
                            ).unsqueeze(-1)
                        ),
                        dim=-2,
                    )
                minimal_dist_probe[b_idx] = paddle.compat.min(dist_min, dim=-1)[0]
        if self.input_local_density:
            probe_vec = probe.reshape(grid.size(0), -1, 3)[
                batch_idx
            ] - atom_coord.unsqueeze(-2)
            if self.pbc:
                probe_vec, _ = pbc_vec(probe_vec, cell[batch_idx])
            atom_density = (self.orbital(probe_vec) * density_feat.unsqueeze(1)).sum(
                dim=-1
            )
            density_res = scatter(
                atom_density, batch_idx, dim=0, dim_size=n_graph, reduce="sum"
            )
            density_res = density_res.reshape(
                grid.size(0), self.num_fourier, self.num_fourier, self.num_fourier, -1
            )
            if self.residual:
                probe_edge_feat_res = o3.spherical_harmonics(
                    list(range(self.num_spherical + 1)),
                    probe_edge / (probe_len[..., None] + 1e-08),
                    normalize=False,
                    normalization="integral",
                )
                residue = self.residue(
                    (probe_src, probe_dst),
                    density_feat,
                    probe_edge_feat_res,
                    probe_edge_embed,
                    dim_size=probe_flat.size(0),
                )
                density_res = density_res + residue.reshape(
                    grid.size(0),
                    self.num_fourier,
                    self.num_fourier,
                    self.num_fourier,
                    -1,
                )
            probe_feat = paddle.cat([probe_feat, density_res], dim=-1)
        if self.input_dist:
            minimal_dist_probe = paddle.exp(-minimal_dist_probe * 0.5).unsqueeze(-1)
            minimal_dist_probe = minimal_dist_probe.reshape(
                grid.size(0), self.num_fourier, self.num_fourier, self.num_fourier, 1
            )
            probe_feat = paddle.cat([probe_feat, minimal_dist_probe], dim=-1)
        if self.atomic_gauss_dist:
            atomic_dist_probe = atomic_dist_probe.reshape(
                grid.size(0), self.num_fourier, self.num_fourier, self.num_fourier, 10
            )
            probe_feat = paddle.cat([probe_feat, atomic_dist_probe], dim=-1)
        fourier_grid = self.get_grid(probe_feat.shape, probe_feat.device)
        probe_feat = self.PWNO(probe_feat, fourier_grid)
        probe_feat = probe_feat.reshape(-1, probe_feat.size(-1))
        sample_flat = grid.reshape(-1, 3)
        sample_batch = (
            paddle.arange(n_graph, device=grid.device)
            .repeat_interleave(n_sample)
            .detach()
        )
        if self.pbc:
            sample_dst, super_probe_and_node_src = radius(
                super_probe_flat,
                sample_flat,
                self.grid_cutoff,
                super_probe_batch,
                sample_batch,
            )
            probe_and_node_src = super_probe_idx_batch[super_probe_and_node_src]
        else:
            sample_dst, probe_and_node_src = radius(
                probe_flat,
                sample_flat,
                self.grid_cutoff,
                probe_batch,
                sample_batch,
            )
        avg_sample_degree = (
            sample_dst.size(0) / sample_flat.size(0) if sample_flat.size(0) != 0 else 0
        )
        avg_sample_exist_degree = (
            sample_dst.size(0) / sample_dst.unique().size(0)
            if sample_dst.unique().size(0) != 0
            else 0
        )
        avg_sample_exist_rate = (
            sample_dst.unique().size(0) / sample_flat.size(0)
            if sample_dst.unique().size(0) != 0
            else 0
        )
        if self.pbc:
            sample_edge = (
                sample_flat[sample_dst] - super_probe_flat[super_probe_and_node_src]
            )
        else:
            sample_edge = sample_flat[sample_dst] - probe_flat[probe_and_node_src]
        sample_len = paddle.norm(sample_edge, dim=-1) + 1e-08
        sample_edge_feat = o3.spherical_harmonics(
            list(range(self.num_spherical_RNO + 1)),
            sample_edge / (sample_len[..., None] + 1e-08),
            normalize=False,
            normalization="integral",
        )
        sample_edge_embed = soft_one_hot_linspace(
            sample_len,
            start=0.0,
            end=self.grid_cutoff,
            number=self.radial_embed_size,
            basis="gaussian",
            cutoff=False,
        ).mul(self.radial_embed_size**0.5)
        scalar_field_gcn = self.scalar_field_gcn(
            (probe_and_node_src, sample_dst),
            probe_feat,
            sample_edge_feat,
            sample_edge_embed,
            dim_size=sample_flat.size(0),
            reduce="mean",
        )
        if self.residual:
            grid_flat = grid.view(-1, 3)
            grid_batch = paddle.arange(n_graph, device=grid.device).repeat_interleave(
                n_sample
            )
            grid_dst, node_src = radius(
                atom_coord, grid_flat, self.cutoff, batch_idx, grid_batch
            )
            grid_edge = grid_flat[grid_dst] - atom_coord[node_src]
            grid_len = paddle.norm(grid_edge, dim=-1) + 1e-08
            grid_edge_feat = o3.spherical_harmonics(
                list(range(self.num_spherical + 1)),
                grid_edge / (grid_len[..., None] + 1e-08),
                normalize=False,
                normalization="integral",
            )
            grid_edge_embed = soft_one_hot_linspace(
                grid_len,
                start=0.0,
                end=self.cutoff,
                number=self.radial_embed_size,
                basis="gaussian",
                cutoff=False,
            ).mul(self.radial_embed_size**0.5)
            residue = self.residue(
                (node_src, grid_dst),
                density_feat,
                grid_edge_feat,
                grid_edge_embed,
                dim_size=grid_flat.size(0),
            )
        sample_vec = grid[batch_idx] - atom_coord.unsqueeze(-2)
        if self.pbc:
            sample_vec, _ = pbc_vec(sample_vec, cell[batch_idx])
        orbital = self.orbital(sample_vec)
        density = (orbital * density_feat.unsqueeze(1)).sum(dim=-1)
        density = scatter(density, batch_idx, dim=0, reduce="sum")
        scalar_field = scalar_field_gcn.reshape([grid.shape[0], grid.shape[1]])
        if self.residual:
            density = density + residue.view(*density.size())
        if self.scalar_mask:
            mask_float = mask.astype(paddle.float32).to(scalar_field.place)
            scalar_field = scalar_field * mask_float

            if self.scalar_inv:
                mask_inv_float = (1 - mask_float).astype(paddle.float32)
                density = density * mask_inv_float
        if self.positive_output:
            density_res = density + scalar_field
            density_res[(density_res < 0) & (density > 0) & (scalar_field < 0)] = 0
            scalar_field[(density_res < 0) & (density > 0) & (scalar_field < 0)] = 0
            density = density_res
        else:
            density = density + scalar_field
        scalar_field_influence = (scalar_field.abs().sum() / density.sum()).detach()

        aux = {
            "avg_probe_degree": avg_probe_degree,
            "avg_probe_exist_degree": avg_probe_exist_degree,
            "avg_probe_exist_rate": avg_probe_exist_rate,
            "avg_sample_degree": avg_sample_degree,
            "avg_sample_exist_degree": avg_sample_exist_degree,
            "avg_sample_exist_rate": avg_sample_exist_rate,
            "scalar_field": scalar_field,
            "coefficient_field": density - scalar_field,
            "scalar_field_influence": scalar_field_influence,
            "probe": probe,
        }

        return density, aux

    def get_grid(self, shape, device):
        """Build normalized Fourier coordinates, including both axis endpoints."""
        spatial_shape = tuple(shape[1:4])
        grid = BuildGrid(format="array")(
            {
                "shape": spatial_shape,
                "voxel_vectors": np.diag(
                    [1.0 / max(size - 1, 1) for size in spatial_shape]
                ),
            }
        )
        coordinates = paddle.to_tensor(
            grid.cartesian_coordinates(), dtype=paddle.get_default_dtype()
        ).to(device)
        return coordinates.reshape([1, *spatial_shape, 3]).expand(
            [shape[0], *spatial_shape, 3]
        )

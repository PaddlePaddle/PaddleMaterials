import logging
import math
from typing import Dict
from typing import Optional
from typing import Union

import numpy as np
import paddle
from ase import Atoms
from ase.units import GPa
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct
from sklearn.gaussian_process.kernels import Hyperparameter
from sklearn.gaussian_process.kernels import Kernel

from ppmat.models.common.runtime import RuntimeMixin
from ppmat.models.common.runtime import runtime_boundary
from ppmat.utils.paddle_aux import dim2perm
from ppmat.utils.scatter import scatter
from ppmat.utils.scatter import scatter_mean


def solver(X, y, regressor: Optional[str] = "NormalizedGaussianProcess", **kwargs):
    if regressor == "GaussianProcess":
        return gp(X, y, **kwargs)
    elif regressor == "NormalizedGaussianProcess":
        return normalized_gp(X, y, **kwargs)
    else:
        raise NotImplementedError(f"{regressor} is not implemented")


def normalized_gp(X, y, **kwargs):
    feature_rms = 1.0 / np.sqrt(np.average(X**2, axis=0))
    feature_rms = np.nan_to_num(feature_rms, 1)
    y_mean = paddle.sum(x=y) / paddle.sum(x=X)
    mean, std = base_gp(
        X,
        y - (paddle.sum(X, axis=1) * y_mean).reshape(y.shape),
        NormalizedDotProduct,
        {"diagonal_elements": feature_rms},
        **kwargs,
    )
    return mean + y_mean, std


def gp(X, y, **kwargs):
    return base_gp(
        X, y, DotProduct, {"sigma_0": 0, "sigma_0_bounds": "fixed"}, **kwargs
    )


def base_gp(
    X,
    y,
    kernel,
    kernel_kwargs,
    alpha: Optional[float] = 0.1,
    max_iteration: int = 20,
    stride: Optional[int] = 1,
):
    if len(tuple(y.shape)) == 1:
        y = y.reshape([-1, 1])
    if stride is not None:
        X = X[::stride]
        y = y[::stride]
    not_fit = True
    iteration = 0
    mean = None
    std = None
    while not_fit:
        print(f"GP fitting iteration {iteration} {alpha}")
        try:
            _kernel = kernel(**kernel_kwargs)
            gpr = GaussianProcessRegressor(kernel=_kernel, random_state=0, alpha=alpha)
            gpr = gpr.fit(X, y)
            vec = paddle.diag(x=paddle.ones(shape=tuple(X.shape)[1]))
            mean, std = gpr.predict(vec, return_std=True)
            mean = paddle.to_tensor(
                data=mean, dtype=paddle.get_default_dtype()
            ).reshape([-1])
            std = paddle.to_tensor(data=std, dtype=paddle.get_default_dtype()).reshape(
                [-1]
            )
            likelihood = gpr.log_marginal_likelihood()
            res = paddle.sqrt(
                x=paddle.square(
                    x=paddle.matmul(x=X, y=mean.reshape([-1, 1])) - y
                ).mean()
            )
            print(
                f"""GP fitting: alpha {alpha}:
            residue {res}
            mean {mean} std {std}
            log marginal likelihood {likelihood}"""
            )
            not_fit = False
        except Exception as e:
            print(f"GP fitting failed for alpha={alpha} and {e.args}")
            if alpha == 0 or alpha is None:
                logging.info("try a non-zero alpha")
                not_fit = False
                raise ValueError(
                    f"""Please set the {alpha} to non-zero value.
The dataset energy is rank deficient to be solved with GP"""
                )
            else:
                alpha = alpha * 2
                iteration += 1
                logging.debug(f"           increase alpha to {alpha}")
            if iteration >= max_iteration or not_fit is False:
                raise ValueError(
                    """Please set the per species shift and scale to zeros and ones.
The dataset energy is to diverge to be solved with GP"""
                )
    return mean, std


class NormalizedDotProduct(Kernel):
    """Dot-Product kernel.
    .. math::
        k(x_i, x_j) = x_i \\cdot A \\cdot x_j
    """

    def __init__(self, diagonal_elements):
        self.diagonal_elements = diagonal_elements
        self.A = np.diag(diagonal_elements)

    def __call__(self, X, Y=None, eval_gradient=False):
        """Return the kernel k(X, Y) and optionally its gradient.
        Parameters
        ----------
        X : ndarray of shape (n_samples_X, n_features)
            Left argument of the returned kernel k(X, Y)
        Y : ndarray of shape (n_samples_Y, n_features), default=None
            Right argument of the returned kernel k(X, Y). If None, k(X, X)
            if evaluated instead.
        eval_gradient : bool, default=False
            Determines whether the gradient with respect to the log of
            the kernel hyperparameter is computed.
            Only supported when Y is None.
        Returns
        -------
        K : ndarray of shape (n_samples_X, n_samples_Y)
            Kernel k(X, Y)
        K_gradient : ndarray of shape (n_samples_X, n_samples_X, n_dims), optional
            The gradient of the kernel k(X, X) with respect to the log of the
            hyperparameter of the kernel. Only returned when `eval_gradient`
            is True.
        """
        X = np.atleast_2d(X)
        if Y is None:
            K = X.dot(y=self.A).dot(y=X.T)
        else:
            if eval_gradient:
                raise ValueError("Gradient can only be evaluated when Y is None.")
            K = X.dot(y=self.A).dot(y=Y.T)
        if eval_gradient:
            return K, np.empty((tuple(X.shape)[0], tuple(X.shape)[0], 0))
        else:
            return K

    def diag(self, X):
        """Returns the diagonal of the kernel k(X, X).
        The result of this method is identical to np.diag(self(X)); however,
        it can be evaluated more efficiently since only the diagonal is
        evaluated.
        Parameters
        ----------
        X : ndarray of shape (n_samples_X, n_features)
            Left argument of the returned kernel k(X, Y).
        Returns
        -------
        K_diag : ndarray of shape (n_samples_X,)
            Diagonal of kernel k(X, X).
        """
        return np.einsum("ij,ij,jj->i", X, X, self.A)

    def __repr__(self):
        return ""

    def is_stationary(self):
        """Returns whether the kernel is stationary."""
        return False

    @property
    def hyperparameter_diagonal_elements(self):
        return Hyperparameter("diagonal_elements", "numeric", "fixed")


DATA_INDEX = {
    "total_energy": 0,
    "forces": 2,
    "per_atom_energy": 1,
    "per_species_energy": 0,
}


class AtomScaling(paddle.nn.Layer):
    """
    Atomic extensive property rescaling module
    """

    def __init__(
        self,
        atoms: list[Atoms] = None,
        total_energy: list[float] = None,
        forces: list[np.ndarray] = None,
        atomic_numbers: list[np.ndarray] = None,
        num_atoms: list[float] = None,
        max_z: int = 94,
        scale_key: str = None,
        shift_key: str = None,
        init_scale: Union[paddle.Tensor, float] = None,
        init_shift: Union[paddle.Tensor, float] = None,
        trainable_scale: bool = False,
        trainable_shift: bool = False,
        verbose: bool = False,
        **kwargs,
    ):
        """
        Args:
            forces: a list of atomic forces (np.ndarray) in each graph
            max_z: (int) maximum atomic number
                - if scale_key or shift_key is specified,
                  max_z should be equal to the maximum atomic_number.
            scale_key: valid options are:
                - total_energy_std
                - per_atom_energy_std
                - per_species_energy_std
                - forces_rms
                - per_species_forces_rms (default)
            shift_key: valid options are:
                - total_energy_mean
                - per_atom_energy_mean
                - per_species_energy_mean :
                  default option is gaussian regression (NequIP)
                - per_species_energy_mean_linear_reg :
                  an alternative choice is linear regression (M3GNet)
            init_scale (paddle.Tensor or float)
            init_shift (paddle.Tensor or float)
        """
        super().__init__()
        self.max_z = max_z
        if scale_key or shift_key:
            total_energy = paddle.to_tensor(data=np.array(total_energy))
            forces = (
                paddle.to_tensor(data=np.concatenate(forces, axis=0))
                if forces is not None
                else None
            )
            if atomic_numbers is None:
                atomic_numbers = [atom.get_atomic_numbers() for atom in atoms]
            atomic_numbers = (
                paddle.to_tensor(data=np.concatenate(atomic_numbers, axis=0))
                .squeeze(axis=-1)
                .astype(dtype="int64")
            )
            if num_atoms is None:
                num_atoms = [atom.positions.shape[0] for atom in atoms]
            num_atoms = paddle.to_tensor(data=np.array(num_atoms))
            per_atom_energy = total_energy / num_atoms
            data_list = [total_energy, per_atom_energy, forces]
            assert (
                tuple(num_atoms.shape)[0] == tuple(total_energy.shape)[0]
            ), "num_atoms and total_energy should have the same size, "
            f"but got {tuple(num_atoms.shape)[0]} and {tuple(total_energy.shape)[0]}"
            if forces is not None:
                assert (
                    tuple(forces.shape)[0] == tuple(atomic_numbers.shape)[0]
                ), "forces and atomic_numbers should have the same length, "
                f"but got {tuple(forces.shape)[0]} and {tuple(atomic_numbers.shape)[0]}"
            if (
                scale_key == "per_species_energy_std"
                and shift_key == "per_species_energy_mean"
                and init_shift is None
                and init_scale is None
            ):
                init_shift, init_scale = self.get_gaussian_statistics(
                    atomic_numbers, num_atoms, total_energy
                )
            else:
                if shift_key and init_shift is None:
                    init_shift = self.get_statistics(
                        shift_key, max_z, data_list, atomic_numbers, num_atoms
                    )
                if scale_key and init_scale is None:
                    init_scale = self.get_statistics(
                        scale_key, max_z, data_list, atomic_numbers, num_atoms
                    )
        if init_scale is None:
            init_scale = paddle.ones(shape=max_z + 1)
        elif isinstance(init_scale, float):
            init_scale = paddle.to_tensor(data=init_scale).tile(repeat_times=max_z + 1)
        else:
            assert tuple(init_scale.shape)[0] == max_z + 1
        if init_shift is None:
            init_shift = paddle.zeros(shape=max_z + 1)
        elif isinstance(init_shift, float):
            init_shift = paddle.to_tensor(data=init_shift).tile(repeat_times=max_z + 1)
        else:
            assert tuple(init_shift.shape)[0] == max_z + 1
        init_shift = init_shift.astype(dtype="float32")
        init_scale = init_scale.astype(dtype="float32")
        if trainable_scale is True:
            self.scale = paddle.base.framework.EagerParamBase.from_tensor(
                tensor=init_scale
            )
        else:
            self.register_buffer(name="scale", tensor=init_scale)
        if trainable_shift is True:
            self.shift = paddle.base.framework.EagerParamBase.from_tensor(
                tensor=init_shift
            )
        else:
            self.register_buffer(name="shift", tensor=init_shift)
        if verbose is True:
            print("Current scale: ", init_scale)
            print("Current shift: ", init_shift)

    def transform(
        self, atomic_energies: paddle.Tensor, atomic_numbers: paddle.Tensor
    ) -> paddle.Tensor:
        """
        Take the origin values from model and get the transformed values
        """
        curr_shift = self.shift[atomic_numbers]
        curr_scale = self.scale[atomic_numbers]
        normalized_energies = curr_scale * atomic_energies + curr_shift
        return normalized_energies

    def inverse_transform(
        self, atomic_energies: paddle.Tensor, atomic_numbers: paddle.Tensor
    ) -> paddle.Tensor:
        """
        Take the transformed values and get the original values
        """
        curr_shift = self.shift[atomic_numbers]
        curr_scale = self.scale[atomic_numbers]
        unnormalized_energies = (atomic_energies - curr_shift) / curr_scale
        return unnormalized_energies

    def forward(
        self, atomic_energies: paddle.Tensor, atomic_numbers: paddle.Tensor
    ) -> paddle.Tensor:
        """
        Atomic_energies and atomic_numbers should have the same size
        """
        return self.transform(atomic_energies, atomic_numbers)

    def get_statistics(
        self, key, max_z, data_list, atomic_numbers, num_atoms
    ) -> paddle.Tensor:
        """
        Valid key:
            scale_key: valid options are:
                - total_energy_mean
                - per_atom_energy_mean
                - per_species_energy_mean
                - per_species_energy_mean_linear_reg :
                  an alternative choice is linear regression
            shift_key: valid options are:
                - total_energy_std
                - per_atom_energy_std
                - per_species_energy_std
                - forces_rms
                - per_species_forces_rms
        """
        data = None
        for data_key in DATA_INDEX:
            if data_key in key:
                data = data_list[DATA_INDEX[data_key]]
        assert data is not None
        statistics = None
        if "mean" in key:
            if "per_species" in key:
                n_atoms = paddle.repeat_interleave(
                    paddle.arange(0, num_atoms.numel()), repeats=num_atoms
                )
                if "linear_reg" in key:
                    features = bincount(
                        atomic_numbers, n_atoms, minlength=self.max_z + 1
                    ).numpy()
                    data = data.numpy()
                    assert features.ndim == 2
                    features = features[(features > 0).any(axis=1)]
                    statistics = np.linalg.pinv(features.T.dot(y=features)).dot(
                        features.T.dot(y=data)
                    )
                    statistics = paddle.to_tensor(data=statistics)
                else:
                    N = bincount(atomic_numbers, num_atoms, minlength=self.max_z + 1)
                    assert N.ndim == 2
                    N = N[(N > 0).astype("bool").any(axis=1)]
                    N = N.astype(paddle.get_default_dtype())
                    statistics, _ = solver(
                        N, data, regressor="NormalizedGaussianProcess"
                    )
            else:
                statistics = paddle.mean(x=data).item()
        elif "std" in key:
            if "per_species" in key:
                print(
                    "Warning: calculating per_species_energy_std for full periodic "
                    "table systems is risky, please use per_species_forces_rms instead."
                )
                n_atoms = paddle.repeat_interleave(
                    paddle.arange(0, num_atoms.numel(0)), repeats=num_atoms
                )
                N = bincount(atomic_numbers, n_atoms, minlength=self.max_z + 1)
                assert N.ndim == 2
                N = N[(N > 0).astype("bool").any(axis=1)]
                N = N.astype(paddle.get_default_dtype())
                _, statistics = solver(N, data, regressor="NormalizedGaussianProcess")
            else:
                statistics = paddle.std(x=data).item()
        elif "rms" in key:
            if "per_species" in key:
                square = scatter_mean(
                    data.square(), atomic_numbers, dim=0, dim_size=max_z + 1
                )
                statistics = square.mean(axis=-1)
            else:
                statistics = paddle.sqrt(x=paddle.mean(x=data.square())).item()
        if isinstance(statistics, paddle.Tensor) is not True:
            statistics = paddle.to_tensor(data=statistics).tile(repeat_times=max_z + 1)
        assert tuple(statistics.shape)[0] == max_z + 1
        return statistics

    def get_gaussian_statistics(
        self,
        atomic_numbers: paddle.Tensor,
        num_atoms: paddle.Tensor,
        total_energy: paddle.Tensor,
    ):
        """
        Get the gaussian process mean and variance
        """
        n_atoms = paddle.repeat_interleave(
            paddle.arange(0, num_atoms.numel()), repeats=num_atoms
        )
        N = bincount(atomic_numbers, n_atoms, minlength=self.max_z + 1)
        assert N.ndim == 2
        N = N[(N > 0).astype("bool").any(axis=1)]
        N = N.astype(paddle.get_default_dtype())
        mean, std = solver(N, total_energy, regressor="NormalizedGaussianProcess")
        assert tuple(mean.shape)[0] == self.max_z + 1
        assert tuple(std.shape)[0] == self.max_z + 1
        return mean, std


def bincount(
    input: paddle.Tensor, batch: Optional[paddle.Tensor] = None, minlength: int = 0
):
    assert input.ndim == 1
    if batch is None:
        return paddle.bincount(x=input, minlength=minlength)
    else:
        assert tuple(batch.shape) == tuple(input.shape)
        length = input.max().item() + 1
        if minlength == 0:
            minlength = length
        if length > minlength:
            raise ValueError(
                f"minlength {minlength} too small for input with integers up to and "
                f"including {length}"
            )
        input_ = input + batch * minlength
        num_batch = batch.max() + 1
        return paddle.bincount(x=input_, minlength=minlength * num_batch).reshape(
            num_batch, minlength
        )


class LinearLayer(paddle.nn.Layer):
    def __init__(self, in_dim, out_dim, bias=True):
        super().__init__()
        self.linear = paddle.nn.Linear(
            in_features=in_dim, out_features=out_dim, bias_attr=bias
        )

    def forward(self, x):
        return self.linear(x)


class SigmoidLayer(paddle.nn.Layer):
    def __init__(self, in_dim, out_dim, bias=True):
        super().__init__()
        self.linear = paddle.nn.Linear(
            in_features=in_dim, out_features=out_dim, bias_attr=bias
        )
        self.sigmoid = paddle.nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.linear(x))


class SwishLayer(paddle.nn.Layer):
    def __init__(self, in_dim, out_dim, bias=True):
        super().__init__()
        self.linear = paddle.nn.Linear(
            in_features=in_dim, out_features=out_dim, bias_attr=bias
        )
        self.sigmoid = paddle.nn.Sigmoid()

    def forward(self, x):
        x = self.linear(x)
        return x * self.sigmoid(x)


class GatedMLP(paddle.nn.Layer):
    def __init__(
        self,
        in_dim: int,
        out_dims: list,
        activation: Union[list[Union[str, None]], str] = "swish",
        use_bias: bool = True,
    ):
        super().__init__()
        input_dim = in_dim
        if isinstance(activation, str) or activation is None:
            activation = [activation] * len(out_dims)
        else:
            assert len(activation) == len(
                out_dims
            ), "activation and out_dims must have the same length"
        module_list_g = []
        for i in range(len(out_dims)):
            if activation[i] == "swish":
                module_list_g.append(SwishLayer(input_dim, out_dims[i], bias=use_bias))
            elif activation[i] == "sigmoid":
                module_list_g.append(
                    SigmoidLayer(input_dim, out_dims[i], bias=use_bias)
                )
            elif activation[i] is None:
                module_list_g.append(LinearLayer(input_dim, out_dims[i], bias=use_bias))
            input_dim = out_dims[i]
        module_list_sigma = []
        activation[-1] = "sigmoid"
        input_dim = in_dim
        for i in range(len(out_dims)):
            if activation[i] == "swish":
                module_list_sigma.append(
                    SwishLayer(input_dim, out_dims[i], bias=use_bias)
                )
            elif activation[i] == "sigmoid":
                module_list_sigma.append(
                    SigmoidLayer(input_dim, out_dims[i], bias=use_bias)
                )
            elif activation[i] is None:
                module_list_sigma.append(
                    LinearLayer(input_dim, out_dims[i], bias=use_bias)
                )
            else:
                raise NotImplementedError
            input_dim = out_dims[i]
        self.g = paddle.nn.Sequential(*module_list_g)
        self.sigma = paddle.nn.Sequential(*module_list_sigma)

    def forward(self, x):
        return self.g(x) * self.sigma(x)


class MLP(paddle.nn.Layer):
    def __init__(
        self,
        in_dim: int,
        out_dims: list,
        activation: Union[list[Union[str, None]], str, None] = "swish",
        use_bias: bool = True,
    ):
        super().__init__()
        input_dim = in_dim
        if isinstance(activation, str) or activation is None:
            activation = [activation] * len(out_dims)
        else:
            assert len(activation) == len(
                out_dims
            ), "activation and out_dims must have the same length"
        module_list = []
        for i in range(len(out_dims)):
            if activation[i] == "swish":
                module_list.append(SwishLayer(input_dim, out_dims[i], bias=use_bias))
            elif activation[i] == "sigmoid":
                module_list.append(SigmoidLayer(input_dim, out_dims[i], bias=use_bias))
            elif activation[i] is None:
                module_list.append(LinearLayer(input_dim, out_dims[i], bias=use_bias))
            else:
                raise NotImplementedError
            input_dim = out_dims[i]
        self.mlp = paddle.nn.Sequential(*module_list)

    def forward(self, x):
        return self.mlp(x)


def polynomial(r: paddle.Tensor, cutoff: float) -> paddle.Tensor:
    """
    Polynomial cutoff function
    Args:
        r (tf.Tensor): radius distance tensor
        cutoff (float): cutoff distance
    Returns: polynomial cutoff functions
    """
    ratio = paddle.divide(x=r, y=paddle.to_tensor(cutoff))
    result = (
        1
        - 6 * paddle.pow(x=ratio, y=5)
        + 15 * paddle.pow(x=ratio, y=4)
        - 10 * paddle.pow(x=ratio, y=3)
    )
    return paddle.clip(x=result, min=0.0)


class ThreeDInteraction(paddle.nn.Layer):
    def __init__(self, max_n, max_l, cutoff, units, spherecal_dim, threebody_cutoff):
        super().__init__()
        self.atom_mlp = SigmoidLayer(in_dim=units, out_dim=spherecal_dim)
        self.edge_gate_mlp = GatedMLP(
            in_dim=spherecal_dim, out_dims=[units], activation="swish", use_bias=False
        )
        self.cutoff = cutoff
        self.threebody_cutoff = threebody_cutoff

    def forward(
        self,
        edge_attr,
        three_basis,
        atom_attr,
        edge_index,
        three_body_index,
        edge_length,
        num_edges,
        num_triple_ij,
    ):
        if three_body_index.shape[0] == 0:
            return edge_attr
        atom_mask = (
            self.atom_mlp(atom_attr)[edge_index[0][three_body_index[:, 1]]]
            * polynomial(edge_length[three_body_index[:, 0]], self.threebody_cutoff)
            * polynomial(edge_length[three_body_index[:, 1]], self.threebody_cutoff)
        )
        three_basis = three_basis * atom_mask
        index_map = paddle.arange(end=paddle.sum(x=num_edges).item()).to(
            edge_length.place
        )
        index_map = paddle.repeat_interleave(x=index_map, repeats=num_triple_ij).to(
            edge_length.place
        )
        e_ij_tuda = scatter(
            three_basis,
            index_map,
            dim=0,
            reduce="sum",
            dim_size=paddle.sum(x=num_edges).item(),
        )
        edge_attr_prime = edge_attr + self.edge_gate_mlp(e_ij_tuda)
        return edge_attr_prime


class AtomLayer(paddle.nn.Layer):
    """
    v_i'=v_i+sum(phi(v+i,v_j,e_ij',u)W*e_ij^0)
    """

    def __init__(self, atom_attr_dim, edge_attr_dim, spherecal_dim):
        super().__init__()
        self.gated_mlp = GatedMLP(
            in_dim=2 * atom_attr_dim + spherecal_dim, out_dims=[128, 64, atom_attr_dim]
        )
        self.edge_layer = LinearLayer(in_dim=edge_attr_dim, out_dim=1)

    def forward(self, atom_attr, edge_attr, edge_index, edge_attr_prime, num_atoms):
        feat = paddle.concat(
            x=[atom_attr[edge_index[0]], atom_attr[edge_index[1]], edge_attr_prime],
            axis=1,
        )
        atom_attr_prime = self.gated_mlp(feat) * self.edge_layer(edge_attr)
        atom_attr_prime = scatter(
            atom_attr_prime,
            edge_index[1],
            dim=0,
            dim_size=paddle.sum(x=num_atoms).item(),
        )
        return atom_attr_prime + atom_attr


class EdgeLayer(paddle.nn.Layer):
    """e_ij'=e_ij+phi(v_i,v_j,e_ij,u)W*e_ij^0"""

    def init(self, atom_attr_dim, edge_attr_dim, spherecal_dim):
        super().__init__()
        self.gated_mlp = GatedMLP(
            in_dim=2 * atom_attr_dim + spherecal_dim, out_dims=[128, 64, edge_attr_dim]
        )
        self.edge_layer = LinearLayer(in_dim=edge_attr_dim, out_dim=1)

    def forward(self, atom_attr, edge_attr, edge_index, edge_attr_prime):
        feat = paddle.concat(
            x=[atom_attr[edge_index[0]], atom_attr[edge_index[1]], edge_attr_prime],
            axis=1,
        )
        edge_attr_prime = self.gated_mlp(feat) * self.edge_layer(edge_attr)
        return edge_attr_prime + edge_attr


class MainBlock(paddle.nn.Layer):
    """
    MainBlock for Message Passing in M3GNet
    """

    def __init__(self, max_n, max_l, cutoff, units, spherical_dim, threebody_cutoff):
        super().__init__()
        self.gated_mlp_atom = GatedMLP(
            in_dim=2 * units + units, out_dims=[units, units], activation="swish"
        )
        self.edge_layer_atom = SwishLayer(
            in_dim=spherical_dim, out_dim=units, bias=False
        )
        self.gated_mlp_edge = GatedMLP(
            in_dim=2 * units + units, out_dims=[units, units], activation="swish"
        )
        self.edge_layer_edge = LinearLayer(
            in_dim=spherical_dim, out_dim=units, bias=False
        )
        self.three_body = ThreeDInteraction(
            max_n, max_l, cutoff, units, max_n * max_l, threebody_cutoff
        )

    def forward(
        self,
        atom_attr,
        edge_attr,
        edge_attr_zero,
        edge_index,
        three_basis,
        three_body_index,
        edge_length,
        num_edges,
        num_triple_ij,
        num_atoms,
    ):
        # threebody interaction
        edge_attr = self.three_body(
            edge_attr,
            three_basis,
            atom_attr,
            edge_index,
            three_body_index,
            edge_length,
            num_edges,
            num_triple_ij.reshape([-1]),
        )
        # update bond feature
        feat = paddle.concat(
            x=[atom_attr[edge_index[0]], atom_attr[edge_index[1]], edge_attr], axis=1
        )
        edge_attr = edge_attr + self.gated_mlp_edge(feat) * self.edge_layer_edge(
            edge_attr_zero
        )

        # update atom feature
        feat = paddle.concat(
            x=[atom_attr[edge_index[0]], atom_attr[edge_index[1]], edge_attr], axis=1
        )
        atom_attr_prime = self.gated_mlp_atom(feat) * self.edge_layer_atom(
            edge_attr_zero
        )
        atom_attr = atom_attr + scatter(
            atom_attr_prime,
            edge_index[0],
            dim=0,
            dim_size=paddle.sum(x=num_atoms).item(),
        )

        return atom_attr, edge_attr


class BesselBasis(paddle.nn.Layer):
    def __init__(self, r_max, num_basis=8, trainable=True):
        """Radial Bessel Basis, as proposed in
            DimeNet: https://arxiv.org/abs/2003.03123

        Parameters
        ----------
        r_max : float
            Cutoff radius

        num_basis : int
            Number of Bessel Basis functions

        trainable : bool
            Train the :math:`n \\pi` part or not.
        """
        super(BesselBasis, self).__init__()
        self.trainable = trainable
        self.num_basis = num_basis
        self.r_max = float(r_max)
        self.prefactor = 2.0 / self.r_max
        bessel_weights = (
            paddle.linspace(start=1.0, stop=num_basis, num=num_basis) * math.pi
        )
        if self.trainable:
            self.bessel_weights = paddle.base.framework.EagerParamBase.from_tensor(
                tensor=bessel_weights
            )
        else:
            self.register_buffer(name="bessel_weights", tensor=bessel_weights)

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        """
        Evaluate Bessel Basis for input x.

        Parameters
        ----------
        x : paddle.Tensor
            Input
        """
        numerator = paddle.sin(
            x=self.bessel_weights * x.unsqueeze(axis=-1) / self.r_max
        )
        return self.prefactor * (numerator / x.unsqueeze(axis=-1))


class SmoothBesselBasis(paddle.nn.Layer):
    def __init__(self, r_max, max_n=10):
        """Smooth Radial Bessel Basis, as proposed
            in DimeNet: https://arxiv.org/abs/2003.03123
            This is an orthogonal basis with first
            and second derivative at the cutoff
            equals to zero. The function was derived from
            the order 0 spherical Bessel function,
            and was expanded by the different zero roots
        Ref:
            https://arxiv.org/pdf/1907.02374.pdf
        Args:
            r_max: paddle.Tensor distance tensor
            max_n: int, max number of basis, expanded by the zero roots
        Returns: expanded spherical harmonics with
                 derivatives smooth at boundary
        Parameters
        ----------
        """
        super(SmoothBesselBasis, self).__init__()
        self.max_n = max_n
        n = paddle.arange(start=0, end=max_n).astype(dtype="float32")[None, :]
        PI = 3.1415926535897
        SQRT2 = 1.41421356237
        fnr = (
            (-1) ** n
            * SQRT2
            * PI
            / r_max**1.5
            * (n + 1)
            * (n + 2)
            / paddle.sqrt(x=2 * n**2 + 6 * n + 5)
        )
        en = n**2 * (n + 2) ** 2 / (4 * (n + 1) ** 4 + 1)
        dn = [paddle.to_tensor(data=1.0).astype(dtype="float32")]
        for i in range(1, max_n):
            dn.append(1 - en[0, i] / dn[-1])
        dn = paddle.stack(x=dn)
        self.register_buffer(name="dn", tensor=dn)
        self.register_buffer(name="en", tensor=en)
        self.register_buffer(name="fnr_weights", tensor=fnr)
        self.register_buffer(
            name="n_1_pi_cutoff",
            tensor=(
                (paddle.arange(start=0, end=max_n).astype(dtype="float32") + 1)
                * PI
                / r_max
            ).reshape(1, -1),
        )
        self.register_buffer(
            name="n_2_pi_cutoff",
            tensor=(
                (paddle.arange(start=0, end=max_n).astype(dtype="float32") + 2)
                * PI
                / r_max
            ).reshape(1, -1),
        )
        self.register_buffer(name="r_max", tensor=paddle.to_tensor(data=r_max))

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        """
        Evaluate Smooth Bessel Basis for input x.

        Parameters
        ----------
        x : paddle.Tensor
            Input
        """
        x_1 = x.unsqueeze(axis=-1) * self.n_1_pi_cutoff
        x_2 = x.unsqueeze(axis=-1) * self.n_2_pi_cutoff
        fnr = self.fnr_weights * (paddle.sin(x=x_1) / x_1 + paddle.sin(x=x_2) / x_2)
        gn = [fnr[:, 0]]
        for i in range(1, self.max_n):
            gn.append(
                1
                / paddle.sqrt(x=self.dn[i])
                * (fnr[:, i] + paddle.sqrt(x=self.en[0, i] / self.dn[i - 1]) * gn[-1])
            )
        return paddle.transpose(
            x=paddle.stack(x=gn), perm=dim2perm(paddle.stack(x=gn).ndim, 1, 0)
        )


def _spherical_harmonics(lmax: int, x: paddle.Tensor) -> paddle.Tensor:
    sh_0_0 = paddle.ones_like(x=x) * 0.5 * math.sqrt(1.0 / math.pi)
    if lmax == 0:
        return paddle.stack(x=[sh_0_0], axis=-1)
    sh_1_1 = math.sqrt(3.0 / (4.0 * math.pi)) * x
    if lmax == 1:
        return paddle.stack(x=[sh_0_0, sh_1_1], axis=-1)
    sh_2_2 = math.sqrt(5.0 / (16.0 * math.pi)) * (3.0 * x**2 - 1.0)
    if lmax == 2:
        return paddle.stack(x=[sh_0_0, sh_1_1, sh_2_2], axis=-1)
    sh_3_3 = math.sqrt(7.0 / (16.0 * math.pi)) * x * (5.0 * x**2 - 3.0)
    if lmax == 3:
        return paddle.stack(x=[sh_0_0, sh_1_1, sh_2_2, sh_3_3], axis=-1)
    raise ValueError("lmax must be less than 8")


class SphericalBasisLayer(paddle.nn.Layer):
    def __init__(self, max_n, max_l, cutoff):
        super(SphericalBasisLayer, self).__init__()
        assert max_l <= 4, "lmax must be less than 5"
        assert max_n <= 4, "max_n must be less than 5"
        self.max_n = max_n
        self.max_l = max_l
        self.cutoff = cutoff
        self.register_buffer(
            name="factor",
            tensor=paddle.sqrt(x=paddle.to_tensor(data=2.0 / self.cutoff**3)),
        )
        self.coef = paddle.zeros(shape=[4, 9, 4])
        self.coef[0, 0, :] = paddle.to_tensor(
            data=[
                3.14159274101257,
                6.28318548202515,
                9.42477798461914,
                12.5663709640503,
            ]
        )
        self.coef[1, :4, :] = paddle.to_tensor(
            data=[
                [
                    -1.02446483277785,
                    -1.00834335996107,
                    -1.00419641763893,
                    -1.00252381898662,
                ],
                [4.49340963363647, 7.7252516746521, 10.9041213989258, 14.0661935806274],
                [
                    0.22799275301076,
                    0.130525632358311,
                    0.092093290316619,
                    0.0712718627992818,
                ],
                [4.49340963363647, 7.7252516746521, 10.9041213989258, 14.0661935806274],
            ]
        )
        self.coef[2, :6, :] = paddle.to_tensor(
            data=[
                [
                    -1.04807944170731,
                    -1.01861796359391,
                    -1.01002272174988,
                    -1.00628955560036,
                ],
                [5.76345920562744, 9.09501171112061, 12.322940826416, 15.5146026611328],
                [
                    0.545547077361439,
                    0.335992298618515,
                    0.245888396928293,
                    0.194582402961821,
                ],
                [5.76345920562744, 9.09501171112061, 12.322940826416, 15.5146026611328],
                [
                    0.0946561878721665,
                    0.0369424811413594,
                    0.0199537107571916,
                    0.0125418876146463,
                ],
                [5.76345920562744, 9.09501171112061, 12.322940826416, 15.5146026611328],
            ]
        )
        self.coef[3, :8, :] = paddle.to_tensor(
            data=[
                [1.06942831392075, 1.0292173312802, 1.01650804843248, 1.01069656069999],
                [6.9879322052002, 10.4171180725098, 13.6980228424072, 16.9236221313477],
                [
                    0.918235852195231,
                    0.592803493701152,
                    0.445250264272671,
                    0.358326327374518,
                ],
                [6.9879322052002, 10.4171180725098, 13.6980228424072, 16.9236221313477],
                [
                    0.328507713452024,
                    0.142266673367543,
                    0.0812617757677838,
                    0.0529328657590962,
                ],
                [6.9879322052002, 10.4171180725098, 13.6980228424072, 16.9236221313477],
                [
                    0.0470107184508114,
                    0.0136570088173405,
                    0.0059323726279831,
                    0.00312775039221944,
                ],
                [6.9879322052002, 10.4171180725098, 13.6980228424072, 16.9236221313477],
            ]
        )

    def forward(self, r, theta_val):
        r = r / self.cutoff
        rbfs = []
        for j in range(self.max_l):
            rbfs.append(paddle.sin(x=self.coef[0, 0, j] * r) / r)
        if self.max_n > 1:
            for j in range(self.max_l):
                rbfs.append(
                    (
                        self.coef[1, 0, j] * r * paddle.cos(x=self.coef[1, 1, j] * r)
                        + self.coef[1, 2, j] * paddle.sin(x=self.coef[1, 3, j] * r)
                    )
                    / r**2
                )
            if self.max_n > 2:
                for j in range(self.max_l):
                    rbfs.append(
                        (
                            self.coef[2, 0, j]
                            * r**2
                            * paddle.sin(x=self.coef[2, 1, j] * r)
                            - self.coef[2, 2, j]
                            * r
                            * paddle.cos(x=self.coef[2, 3, j] * r)
                            + self.coef[2, 4, j] * paddle.sin(x=self.coef[2, 5, j] * r)
                        )
                        / r**3
                    )
                if self.max_n > 3:
                    for j in range(self.max_l):
                        rbfs.append(
                            (
                                self.coef[3, 0, j]
                                * r**3
                                * paddle.cos(x=self.coef[3, 1, j] * r)
                                - self.coef[3, 2, j]
                                * r**2
                                * paddle.sin(x=self.coef[3, 3, j] * r)
                                - self.coef[3, 4, j]
                                * r
                                * paddle.cos(x=self.coef[3, 5, j] * r)
                                + self.coef[3, 6, j]
                                * paddle.sin(x=self.coef[3, 7, j] * r)
                            )
                            / r**4
                        )
        rbfs = paddle.stack(x=rbfs, axis=-1)
        rbfs = rbfs * self.factor
        cbfs = _spherical_harmonics(self.max_l - 1, paddle.cos(x=theta_val))
        cbfs = cbfs.repeat_interleave(repeats=self.max_n, axis=1)
        return rbfs * cbfs


class M3GNet(RuntimeMixin, paddle.nn.Layer):
    """
    M3GNet
    """

    def __init__(
        self,
        num_blocks: int = 4,
        units: int = 128,
        max_l: int = 4,
        max_n: int = 4,
        cutoff: float = 5.0,
        max_z: int = 94,
        threebody_cutoff: float = 4.0,
        energy_key: str = "energy",
        force_key: str = "force",
        stress_key: str = "stress",
        loss_type: str = "smooth_l1_loss",
        huber_loss_delta: float = 0.1,
        loss_weights_dict: dict | None = None,
        execution_backend: str = "eager",
        runtime_options: dict | None = None,
        **kwargs,
    ):
        super().__init__()
        self._init_runtime(execution_backend, runtime_options)
        self.energy_key = energy_key
        self.force_key = force_key
        self.stress_key = stress_key

        self.rbf = SmoothBesselBasis(r_max=cutoff, max_n=max_n)
        self.sbf = SphericalBasisLayer(max_n=max_n, max_l=max_l, cutoff=cutoff)
        self.edge_encoder = MLP(
            in_dim=max_n, out_dims=[units], activation="swish", use_bias=False
        )
        module_list = [
            MainBlock(max_n, max_l, cutoff, units, max_n, threebody_cutoff)
            for i in range(num_blocks)
        ]
        self.graph_conv = paddle.nn.LayerList(sublayers=module_list)
        self.final = GatedMLP(
            in_dim=units,
            out_dims=[units, units, 1],
            activation=["swish", "swish", None],
        )
        self.apply(self.init_weights)
        self.atom_embedding = MLP(
            in_dim=max_z + 1, out_dims=[units], activation=None, use_bias=False
        )
        self.atom_embedding.apply(self.init_weights_uniform)
        self.normalizer = AtomScaling(verbose=False, max_z=max_z)
        self.max_z = max_z
        self.model_args = {
            "num_blocks": num_blocks,
            "units": units,
            "max_l": max_l,
            "max_n": max_n,
            "cutoff": cutoff,
            "max_z": max_z,
            "threebody_cutoff": threebody_cutoff,
        }

        self.loss_type = loss_type

        self.loss_weights_dict = loss_weights_dict
        if loss_type == "mse_loss":
            self.loss_fn = paddle.nn.MSELoss()
        elif loss_type == "smooth_l1_loss" or loss_type == "huber_loss":
            self.loss_fn = paddle.nn.SmoothL1Loss(delta=huber_loss_delta)
            self.huber_loss_delta = huber_loss_delta
        elif loss_type == "l1_loss":
            self.loss_fn = paddle.nn.L1Loss()
        else:
            raise ValueError(f"Unknown loss type {loss_type}.")

    def _forward(self, batch_data: Dict[str, paddle.Tensor]) -> paddle.Tensor:
        #  The data in data['graph'] is numpy.ndarray, convert it to paddle.Tensor
        batch_data["graph"] = batch_data["graph"].tensor()
        graph = batch_data["graph"]

        atom_attr = (
            graph.node_feat["atom_types"].astype(dtype="float32").reshape([-1, 1])
        )
        return self._runtime_forward(
            pos=graph.node_feat["cart_coords"],
            cell=graph.node_feat["lattice"],
            pbc_offsets=graph.edge_feat["pbc_offset"].astype(dtype="float32"),
            atom_attr=atom_attr,
            edge_index=(
                graph.edges.astype(dtype="int64").transpose([1, 0]).contiguous()
            ),
            three_body_indices=graph.edge_feat["three_body_indices"].astype(
                dtype="int64"
            ),
            num_three_body=graph.edge_feat["num_three_body"],
            num_bonds=graph.edge_feat["num_edges"],
            num_triple_ij=graph.edge_feat["num_triple_ij"],
            num_atoms=graph.node_feat["num_atoms"],
            num_graphs=graph.num_graph,
            batch=graph.graph_node_id,
        )

    @runtime_boundary("forward")
    def _runtime_forward(
        self,
        pos,
        cell,
        pbc_offsets,
        atom_attr,
        edge_index,
        three_body_indices,
        num_three_body,
        num_bonds,
        num_triple_ij,
        num_atoms,
        num_graphs,
        batch,
    ):
        if self.force_key is not None:
            pos.stop_gradient = False

        strain = None
        volume = None
        if self.stress_key is not None:
            strain = paddle.zeros_like(x=cell)
            strain.stop_gradient = False
            cell = paddle.matmul(x=cell, y=paddle.eye(num_rows=3)[None, ...] + strain)
            strain_augment = paddle.repeat_interleave(
                x=strain, repeats=num_atoms, axis=0
            )
            pos = paddle.bmm(
                pos.unsqueeze(1),
                paddle.eye(num_rows=3)[None, ...] + strain_augment,
            ).squeeze(1)
            volume = paddle.linalg.det(x=cell)

        # -------------------------------------------------------------#
        cumsum = paddle.cumsum(x=num_bonds, axis=0) - num_bonds
        index_bias = paddle.repeat_interleave(
            x=cumsum, repeats=num_three_body, axis=0
        ).unsqueeze(axis=-1)
        three_body_indices = three_body_indices + index_bias

        # === Refer to the implementation of M3GNet,        ===
        # === we should re-compute the following attributes ===
        # edge_length, edge_vector(optional), triple_edge_length, theta_jik
        atoms_batch = paddle.repeat_interleave(
            paddle.arange(0, num_atoms.numel()), repeats=num_atoms
        )
        edge_batch = atoms_batch[edge_index[0]]
        edge_vector = pos[edge_index[0]] - (
            pos[edge_index[1]]
            + paddle.bmm(pbc_offsets.unsqueeze(1), cell[edge_batch]).squeeze(1)
        )
        edge_length = paddle.linalg.norm(x=edge_vector, axis=1)
        vij = edge_vector[three_body_indices[:, 0].clone()]
        vik = edge_vector[three_body_indices[:, 1].clone()]
        rij = edge_length[three_body_indices[:, 0].clone()]
        rik = edge_length[three_body_indices[:, 1].clone()]
        cos_jik = paddle.sum(x=vij * vik, axis=1) / (rij * rik)
        # eps = 1e-7 avoid nan in paddle.acos function
        cos_jik = paddle.clip(x=cos_jik, min=-1.0 + 1e-07, max=1.0 - 1e-07)
        triple_edge_length = rik.reshape([-1])
        edge_length = edge_length.unsqueeze(axis=-1)
        atomic_numbers = atom_attr.squeeze(axis=1).astype(dtype="int64")

        # featurize
        atom_attr = self.atom_embedding(self.one_hot_atoms(atomic_numbers))
        edge_attr = self.rbf(edge_length.reshape([-1]))
        edge_attr_zero = edge_attr  # e_ij^0
        edge_attr = self.edge_encoder(edge_attr)
        three_basis = self.sbf(triple_edge_length, paddle.acos(x=cos_jik))

        # Main Loop
        for conv in self.graph_conv:
            atom_attr, edge_attr = conv(
                atom_attr,
                edge_attr,
                edge_attr_zero,
                edge_index,
                three_basis,
                three_body_indices,
                edge_length,
                num_bonds,
                num_triple_ij,
                num_atoms,
            )
        energies_i = self.final(atom_attr).reshape([-1])
        energies_i = self.normalizer(energies_i, atomic_numbers)
        energies = scatter(energies_i, batch, dim=0, dim_size=num_graphs)
        energies = energies.unsqueeze(-1)

        forces = None
        stresses = None
        if self.force_key is not None and self.stress_key is None:
            force_grad = paddle.grad(
                outputs=[energies],
                inputs=[pos],
                grad_outputs=[paddle.ones_like(x=energies)],
                create_graph=self.training,
            )[0]
            if force_grad is not None:
                forces = paddle.neg(x=force_grad)
        if self.force_key is not None and self.stress_key is not None:
            force_grad, stress_grad = paddle.grad(
                outputs=[energies],
                inputs=[pos, strain],
                grad_outputs=[paddle.ones_like(x=energies)],
                create_graph=self.training,
                retain_graph=True,
            )
            if force_grad is not None:
                forces = paddle.neg(x=force_grad)
            if stress_grad is not None:
                stresses = 1 / volume[:, None, None] * stress_grad / GPa

        energies = energies / num_atoms.unsqueeze(-1).astype(dtype="float32")
        return energies, forces, stresses

    def forward(self, data, return_loss=True, return_prediction=True):
        assert (
            return_loss or return_prediction
        ), "At least one of return_loss or return_prediction must be True."
        (
            energy,
            force,
            stress,
        ) = self._forward(data)

        pred_dict = {}
        if self.energy_key is not None:
            pred_dict[self.energy_key] = energy
        if self.force_key is not None:
            pred_dict[self.force_key] = force
        if self.stress_key is not None:
            pred_dict[self.stress_key] = stress

        loss_dict = {}
        if return_loss:
            loss = 0.0
            for property_name in pred_dict.keys():
                label = data[property_name]
                pred = pred_dict[property_name]
                valid_value_indices = ~paddle.isnan(label)
                valid_label = label[valid_value_indices]
                valid_pred = pred[valid_value_indices]

                if valid_label.numel() > 0:
                    loss_property = self.loss_fn(
                        input=valid_pred,
                        label=valid_label,
                    )

                    loss_dict[property_name] = loss_property
                    loss += loss_property * self.loss_weights_dict[property_name]
            loss_dict["loss"] = loss

        prediction = {}
        if return_prediction:
            prediction = pred_dict

        return {"loss_dict": loss_dict, "pred_dict": prediction}

    def _prediction_to_numpy(self, prediction):
        for key in prediction.keys():
            if isinstance(prediction[key], list):
                prediction[key] = [
                    prediction[key][i].numpy() for i in range(len(prediction[key]))
                ]
            else:
                prediction[key] = prediction[key].numpy()
            if key == "stress" and len(prediction["stress"].shape) == 3:
                prediction[key] = prediction[key][0]
            if key == "magmom" and isinstance(prediction[key], list):
                prediction[key] = prediction[key][0]
            if key == "energy_pre_atom" and isinstance(prediction[key], np.ndarray):
                prediction[key] = prediction[key][0]
        return prediction

    def predict(self, samples):
        is_list = isinstance(samples, list)
        samples = samples if is_list else [samples]

        results = []
        for sample in samples:
            result = self.forward(
                {"graph": sample},
                return_loss=False,
                return_prediction=True,
            )
            prediction = result["pred_dict"]
            prediction = self._prediction_to_numpy(prediction)
            results.append(prediction)
        return results if is_list else results[0]

    def init_weights(self, m):
        if isinstance(m, paddle.nn.Linear):
            init_XavierUniform = paddle.nn.initializer.XavierUniform()
            init_XavierUniform(m.weight)

    def init_weights_uniform(self, m):
        if isinstance(m, paddle.nn.Linear):
            init_Uniform = paddle.nn.initializer.Uniform(low=-0.05, high=0.05)
            init_Uniform(m.weight)

    def one_hot_atoms(self, species):
        return paddle.nn.functional.one_hot(
            num_classes=self.max_z + 1, x=species
        ).astype(dtype="float32")

    def set_normalizer(self, normalizer: AtomScaling):
        self.normalizer = normalizer

    def get_model_args(self):
        return self.model_args

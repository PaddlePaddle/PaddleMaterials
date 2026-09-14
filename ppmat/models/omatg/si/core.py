# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Stochastic interpolants container, concrete single-SI implementations,
and config-driven factory.
"""

from enum import Enum
from typing import Any
from typing import Callable
from typing import Dict
from typing import Iterable
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union

import paddle
from scipy.optimize import linear_sum_assignment

from .interpolants import Corrector
from .interpolants import Epsilon
from .interpolants import IdentityCorrector
from .interpolants import Interpolant
from .interpolants import LatentGamma
from .interpolants import StochasticInterpolant
from .interpolants import StochasticInterpolantSpecies

# Species cardinality of the released OMatG checkpoints; also the upper
# bound of the masked-species token space.
DEFAULT_MAX_ATOMS: int = 100

# SI integration runs on the interior time grid [SMALL_TIME, BIG_TIME] so
# the singular endpoints t=0/1 (gamma/epsilon derivatives) stay out of play.
SMALL_TIME: float = 1.0e-3
BIG_TIME: float = 1.0 - 1.0e-3

# Lattice matrices are clipped elementwise during integration for numerical
# stability; |a_ij| = 100 A is far beyond any physical cell entry.
_CELL_CLIP_BOUND: float = 100.0


def _clone_dict(data: Dict[str, paddle.Tensor]) -> Dict[str, paddle.Tensor]:
    """Deep-copy the tensor fields of a sample dict."""
    return {k: v.clone() for k, v in data.items()}


def _compute_mean_velocity(
    velocity: paddle.Tensor, batch_indices: paddle.Tensor
) -> paddle.Tensor:
    from ppmat.utils.scatter import scatter_mean

    mean_vel = scatter_mean(velocity, batch_indices, dim=0)
    return mean_vel[batch_indices]


class SingleStochasticInterpolant(StochasticInterpolant):
    """Stochastic interpolant x_t = I(t, x_0, x_1) + gamma(t) * z."""

    def __init__(
        self,
        interpolant: Interpolant,
        gamma: Optional[LatentGamma] = None,
        epsilon: Optional[Epsilon] = None,
        differential_equation_type: str = "ODE",
        integrator_kwargs: Optional[Dict[str, Any]] = None,
        correct_center_of_mass_motion: bool = False,
        velocity_annealing_factor: float = 0.0,
    ) -> None:
        """Construct stochastic interpolant."""
        super().__init__()
        self._interpolant = interpolant
        self._gamma = gamma
        if self._gamma is not None:
            self._use_antithetic = self._gamma.requires_antithetic()
        else:
            self._use_antithetic = False
        self._epsilon = epsilon
        self._differential_equation_type = differential_equation_type.upper()
        self._corrector = self._interpolant.get_corrector()

        if self._differential_equation_type == "ODE":
            self.loss = self._ode_loss
            self.integrate = self._ode_integrate
            if self._epsilon is not None:
                raise ValueError("Epsilon function should not be provided for ODEs.")
        elif self._differential_equation_type == "SDE":
            self.loss = self._sde_loss
            self.integrate = self._sde_integrate
            if self._epsilon is None:
                raise ValueError("Epsilon function should be provided for SDEs.")
            if self._gamma is None:
                raise ValueError("Gamma function should be provided for SDEs.")
        else:
            raise ValueError(
                f"Unknown differential equation type: {differential_equation_type}"
            )

        self._correct_center_of_mass_motion = correct_center_of_mass_motion
        self._velocity_annealing_factor = velocity_annealing_factor

    def interpolate(
        self,
        t: paddle.Tensor,
        x_0: paddle.Tensor,
        x_1: paddle.Tensor,
        batch_indices: paddle.Tensor,
    ) -> Tuple[paddle.Tensor, paddle.Tensor]:
        """Stochastically interpolate between x_0 and x_1 at times t."""
        assert x_0.shape == x_1.shape
        interpolate = self._interpolant.interpolate(t, x_0, x_1)
        if self._gamma is not None:
            z = paddle.randn(x_0.shape)
            gamma_t = self._gamma.gamma(t)
            interpolate = self._corrector.correct(interpolate + gamma_t * z)
        else:
            z = paddle.zeros_like(x_0)
        return interpolate, z

    def loss(self, *args, **kwargs):
        raise NotImplementedError  # Overridden in __init__

    def loss_keys(self) -> Iterable[str]:
        """Yield the loss keys produced by this interpolant."""
        yield "loss_b"
        if self._differential_equation_type == "SDE":
            yield "loss_z"

    def _ode_loss(
        self,
        model_function: Callable,
        t: paddle.Tensor,
        x_0: paddle.Tensor,
        x_1: paddle.Tensor,
        x_t: paddle.Tensor,
        z: paddle.Tensor,
        batch_indices: paddle.Tensor,
    ) -> Dict[str, paddle.Tensor]:
        assert x_0.shape == x_1.shape

        if self._use_antithetic:
            assert self._gamma is not None
            x_t_without_gamma = self._interpolant.interpolate(t, x_0, x_1)
            gamma = self._gamma.gamma(t)
            x_t_p = self._corrector.correct(x_t_without_gamma + gamma * z)
            x_t_m = self._corrector.correct(x_t_without_gamma - gamma * z)

            expected_velocity_without_gamma = self._interpolant.interpolate_derivative(
                t, x_0, x_1
            )
            gamma_derivative = self._gamma.gamma_derivative(t)
            expected_velocity_p = expected_velocity_without_gamma + gamma_derivative * z
            expected_velocity_m = expected_velocity_without_gamma - gamma_derivative * z

            if self._correct_center_of_mass_motion:
                mean_velocity_p = _compute_mean_velocity(
                    expected_velocity_p, batch_indices
                )
                expected_velocity_p = expected_velocity_p - mean_velocity_p
                mean_velocity_m = _compute_mean_velocity(
                    expected_velocity_m, batch_indices
                )
                expected_velocity_m = expected_velocity_m - mean_velocity_m

            pred_b_p = model_function(x_t_p)[0]
            pred_b_m = model_function(x_t_m)[0]

            loss = (
                0.5 * paddle.mean(pred_b_p**2)
                + 0.5 * paddle.mean(pred_b_m**2)
                - paddle.mean(pred_b_p * expected_velocity_p)
                - paddle.mean(pred_b_m * expected_velocity_m)
            )
        else:
            expected_velocity = self._interpolant.interpolate_derivative(t, x_0, x_1)
            if self._gamma is not None:
                expected_velocity += self._gamma.gamma_derivative(t) * z

            pred_b = model_function(x_t)[0]

            if self._correct_center_of_mass_motion:
                mean_velocity = _compute_mean_velocity(expected_velocity, batch_indices)
                expected_velocity = expected_velocity - mean_velocity

            loss = paddle.mean(pred_b**2) - 2.0 * paddle.mean(
                pred_b * expected_velocity
            )

        return {"loss_b": loss}

    def _sde_loss(
        self,
        model_function: Callable,
        t: paddle.Tensor,
        x_0: paddle.Tensor,
        x_1: paddle.Tensor,
        x_t: paddle.Tensor,
        z: paddle.Tensor,
        batch_indices: paddle.Tensor,
    ) -> Dict[str, paddle.Tensor]:
        assert x_0.shape == x_1.shape
        assert self._gamma is not None

        if self._use_antithetic:
            x_t_without_gamma = self._interpolant.interpolate(t, x_0, x_1)
            gamma = self._gamma.gamma(t)
            x_t_p = self._corrector.correct(x_t_without_gamma + gamma * z)
            x_t_m = self._corrector.correct(x_t_without_gamma - gamma * z)

            expected_velocity_without_gamma = self._interpolant.interpolate_derivative(
                t, x_0, x_1
            )
            gamma_derivative = self._gamma.gamma_derivative(t)
            expected_velocity_p = expected_velocity_without_gamma + gamma_derivative * z
            expected_velocity_m = expected_velocity_without_gamma - gamma_derivative * z

            if self._correct_center_of_mass_motion:
                mean_velocity_p = _compute_mean_velocity(
                    expected_velocity_p, batch_indices
                )
                expected_velocity_p = expected_velocity_p - mean_velocity_p
                mean_velocity_m = _compute_mean_velocity(
                    expected_velocity_m, batch_indices
                )
                expected_velocity_m = expected_velocity_m - mean_velocity_m

            pred_b_p, pred_z = model_function(x_t_p)
            pred_b_m, _ = model_function(x_t_m)

            loss_b = (
                0.5 * paddle.mean(pred_b_p**2)
                + 0.5 * paddle.mean(pred_b_m**2)
                - paddle.mean(pred_b_p * expected_velocity_p)
                - paddle.mean(pred_b_m * expected_velocity_m)
            )
        else:
            expected_velocity = (
                self._interpolant.interpolate_derivative(t, x_0, x_1)
                + self._gamma.gamma_derivative(t) * z
            )
            pred_b, pred_z = model_function(x_t)

            if self._correct_center_of_mass_motion:
                mean_velocity = _compute_mean_velocity(expected_velocity, batch_indices)
                expected_velocity = expected_velocity - mean_velocity

            loss_b = paddle.mean(pred_b**2) - paddle.mean(pred_b * expected_velocity)

        loss_z = paddle.mean(pred_z**2) - 2.0 * paddle.mean(pred_z * z)

        return {"loss_b": loss_b, "loss_z": loss_z}

    def _ode_integrate(
        self,
        model_function: Callable,
        x_t: paddle.Tensor,
        time: paddle.Tensor,
        time_step: paddle.Tensor,
        batch_indices: paddle.Tensor,
    ) -> paddle.Tensor:
        dt = time_step.item() if hasattr(time_step, "item") else float(time_step)
        t_val = time.item() if time.numel() == 1 else float(time[0])
        model_result = model_function(time, self._corrector.correct(x_t))
        velocity = model_result[0]
        annealing_factor = 1.0 + self._velocity_annealing_factor * t_val
        x_new = x_t + dt * annealing_factor * velocity
        return self._corrector.correct(x_new)

    def _sde_integrate(
        self,
        model_function: Callable,
        x_t: paddle.Tensor,
        time: paddle.Tensor,
        time_step: paddle.Tensor,
        batch_indices: paddle.Tensor,
    ) -> paddle.Tensor:
        dt = time_step.item() if hasattr(time_step, "item") else float(time_step)
        t_val = time.item() if time.numel() == 1 else float(time[0])

        model_result = model_function(time, self._corrector.correct(x_t))
        drift = model_result[0]
        eta = model_result[1] if len(model_result) > 1 else paddle.zeros_like(drift)

        epsilon_t = (
            self._epsilon.epsilon(paddle.to_tensor([t_val])).item()
            if self._epsilon
            else 0.0
        )
        gamma_t = (
            self._gamma.gamma(paddle.to_tensor([t_val])).item() if self._gamma else 1.0
        )

        diffusion = paddle.sqrt(paddle.to_tensor(2.0 * epsilon_t * dt)) * paddle.randn(
            x_t.shape
        )
        x_new = x_t + drift * dt - (epsilon_t / gamma_t) * eta * dt + diffusion
        return self._corrector.correct(x_new)

    def get_corrector(self) -> Corrector:
        return self._corrector

    def integrate(self, *args, **kwargs):
        raise NotImplementedError  # Overridden in __init__


class SingleStochasticInterpolantIdentity(StochasticInterpolantSpecies):
    """Stochastic interpolant x_t = x_0 = x_1 (species constant)."""

    def __init__(self) -> None:
        super().__init__()

    def uses_masked_species(self) -> bool:
        return False

    def interpolate(
        self,
        t: paddle.Tensor,
        x_0: paddle.Tensor,
        x_1: paddle.Tensor,
        batch_indices: paddle.Tensor,
    ) -> Tuple[paddle.Tensor, paddle.Tensor]:
        assert bool(paddle.equal_all(x_0, x_1))
        return x_0.clone(), paddle.zeros_like(x_0)

    def loss_keys(self) -> Iterable[str]:
        yield "loss"

    def loss(
        self,
        model_function: Callable,
        t: paddle.Tensor,
        x_0: paddle.Tensor,
        x_1: paddle.Tensor,
        x_t: paddle.Tensor,
        z: paddle.Tensor,
        batch_indices: paddle.Tensor,
    ) -> Dict[str, paddle.Tensor]:
        assert bool(paddle.equal_all(x_0, x_1))
        return {"loss": paddle.to_tensor(0.0, place=x_0.place)}

    def integrate(
        self,
        model_function: Callable,
        x_t: paddle.Tensor,
        time: paddle.Tensor,
        time_step: paddle.Tensor,
        batch_indices: paddle.Tensor,
    ) -> paddle.Tensor:
        return x_t.clone()

    def get_corrector(self) -> Corrector:
        return IdentityCorrector()


class DiscreteFlowMatchingMask(StochasticInterpolantSpecies):
    """Discrete flow matching between masked base p_0 and target p_1 for species."""

    def __init__(self, noise: float = 0.0) -> None:
        super().__init__()
        if noise < 0.0:
            raise ValueError("Noise parameter must be greater than or equal to 0.")
        self._mask_index = 0
        self._noise = noise

    def uses_masked_species(self) -> bool:
        return True

    def interpolate(
        self,
        t: paddle.Tensor,
        x_0: paddle.Tensor,
        x_1: paddle.Tensor,
        batch_indices: paddle.Tensor,
    ) -> Tuple[paddle.Tensor, paddle.Tensor]:
        assert x_0.shape == x_1.shape
        assert paddle.all(x_0 == self._mask_index)
        if not bool(paddle.all(x_1 != self._mask_index)):
            raise ValueError(
                "DiscreteFlowMatchingMask target species must not contain the "
                "mask index 0; species are 1-based atomic numbers."
            )
        x_t = x_0.clone()
        mask = paddle.rand(x_0.shape) < t
        x_t[mask] = x_1[mask]
        return x_t, paddle.zeros_like(x_t)

    def loss_keys(self) -> Iterable[str]:
        yield "loss"

    def loss(
        self,
        model_function: Callable,
        t: paddle.Tensor,
        x_0: paddle.Tensor,
        x_1: paddle.Tensor,
        x_t: paddle.Tensor,
        z: paddle.Tensor,
        batch_indices: paddle.Tensor,
    ) -> Dict[str, paddle.Tensor]:
        assert x_0.shape == x_1.shape
        assert paddle.all(x_0 == self._mask_index)
        if not bool(paddle.all(x_1 != self._mask_index)):
            raise ValueError(
                "DiscreteFlowMatchingMask target species must not contain the "
                "mask index 0; species are 1-based atomic numbers."
            )
        pred = model_function(x_t)[0]
        assert pred.shape == (x_0.shape[0], DEFAULT_MAX_ATOMS)
        return {"loss": paddle.nn.functional.cross_entropy(input=pred, label=x_1 - 1)}

    def integrate(
        self,
        model_function: Callable,
        x_t: paddle.Tensor,
        time: paddle.Tensor,
        time_step: paddle.Tensor,
        batch_indices: paddle.Tensor,
    ) -> paddle.Tensor:
        """Advance masked species by one discretised CTMC step."""
        eps = paddle.finfo(paddle.float64).eps
        x_1_probs = paddle.nn.functional.softmax(model_function(time, x_t)[0], axis=-1)
        x_1_probs = x_1_probs.reshape((-1, DEFAULT_MAX_ATOMS))
        shifted_x_1 = paddle.multinomial(
            x_1_probs, num_samples=1, replacement=True
        ).squeeze(-1)
        shifted_x_t = x_t - 1
        assert shifted_x_1.shape == x_t.shape == shifted_x_t.shape
        shifted_x_1_hot = paddle.nn.functional.one_hot(
            shifted_x_1, num_classes=DEFAULT_MAX_ATOMS
        )
        dpt = shifted_x_1_hot - 1.0 / DEFAULT_MAX_ATOMS
        dpt_xt = dpt.gather(-1, shifted_x_t[:, None]).squeeze(-1)
        pt = time * shifted_x_1_hot + (1.0 - time) * (1.0 / DEFAULT_MAX_ATOMS)
        pt_xt = pt.gather(-1, shifted_x_t[:, None]).squeeze(-1)
        S = paddle.count_nonzero(x=pt, axis=-1).cast(pt.dtype)
        denom = paddle.where(pt_xt == 0.0, paddle.ones_like(pt_xt), S * pt_xt)
        rate = paddle.nn.functional.relu(x=dpt - dpt_xt[:, None]) / denom[:, None]
        rate[(pt_xt == 0.0)[:, None].expand([-1, DEFAULT_MAX_ATOMS])] = 0.0
        rate[pt == 0.0] = 0.0
        rate_db = paddle.zeros_like(rate)
        if self._noise > 0.0:
            rate_db[shifted_x_t == shifted_x_1] = 1.0
            rate_db[shifted_x_1 != shifted_x_t] = (
                DEFAULT_MAX_ATOMS * time + 1.0 - time
            ) / (1.0 - time + eps)
            rate_db *= self._noise
        rate = rate + rate_db
        step_probs = (rate * time_step).clip(max=1.0)
        step_probs[paddle.arange(len(shifted_x_t)), shifted_x_t] = 0.0
        step_probs[paddle.arange(len(shifted_x_t)), shifted_x_t] = (
            (1.0 - step_probs.sum(axis=-1, keepdim=True)).clip(min=0.0)
        ).squeeze(-1)
        step_probs = paddle.nan_to_num(step_probs, nan=0.0, posinf=1.0, neginf=0.0)
        step_probs = step_probs.clip(min=0.0, max=1.0)
        x_t = (
            paddle.multinomial(step_probs, num_samples=1, replacement=True).squeeze(-1)
            + 1
        )
        return x_t


class DataField(Enum):
    """Enum for data fields in the OMatG sample dict."""

    pos = "pos"
    cell = "cell"
    species = "species"


def _resolve_data_field(data_field: str) -> DataField:
    """Resolve a data-field name to its enum member."""
    try:
        return DataField[data_field.lower()]
    except KeyError:
        raise ValueError(
            f"Data field '{data_field}' is unknown; "
            f"available fields are {[item.value for item in DataField]}."
        )


def reshape_t(
    t: paddle.Tensor, n_atoms: paddle.Tensor, data_field: DataField
) -> paddle.Tensor:
    """Reshape times tensor for batch configurations for the given data field."""
    assert len(t.shape) == 1
    assert len(n_atoms.shape) == 1

    t_per_atom = paddle.repeat_interleave(t, n_atoms.cast("int64"))
    sum_n_atoms = int(n_atoms.sum().item())
    batch_size = len(t)

    if data_field == DataField.pos:
        return paddle.reshape(
            paddle.repeat_interleave(t_per_atom, 3),
            [sum_n_atoms, 3],
        )
    elif data_field == DataField.cell:
        return paddle.reshape(paddle.repeat_interleave(t, 9), [batch_size, 3, 3])
    else:
        return t_per_atom


class StochasticInterpolants:
    """Collection of stochastic interpolants between x_0 and x_1."""

    def __init__(
        self,
        stochastic_interpolants: Sequence[StochasticInterpolant],
        data_fields: Sequence[str],
        integration_time_steps: int,
    ) -> None:
        super().__init__()

        if not len(stochastic_interpolants) == len(data_fields):
            raise ValueError(
                "The number of stochastic interpolants and data fields must be equal."
            )

        self._data_fields = [_resolve_data_field(df) for df in data_fields]

        if not integration_time_steps > 0:
            raise ValueError("The number of integration time steps must be positive.")

        self._stochastic_interpolants = stochastic_interpolants
        self._integration_time_steps = integration_time_steps

    def __len__(self) -> int:
        return len(self._stochastic_interpolants)

    def loss_keys(self) -> List[str]:
        """Return the ``{data_field}_{loss_key}`` keys produced by :meth:`losses`."""
        return [
            f"{data_field.value}_{loss_key}"
            for data_field, stochastic_interpolant in zip(
                self._data_fields, self._stochastic_interpolants
            )
            for loss_key in stochastic_interpolant.loss_keys()
        ]

    def get_stochastic_interpolant(self, data_field: str) -> StochasticInterpolant:
        """Return the stochastic interpolant bound to ``data_field``."""
        field = _resolve_data_field(data_field)
        if field not in self._data_fields:
            raise ValueError(
                f"Data field '{data_field}' is not part of this collection; "
                f"available fields are {[item.value for item in self._data_fields]}."
            )
        return self._stochastic_interpolants[self._data_fields.index(field)]

    def _interpolate(
        self,
        t: paddle.Tensor,
        x_0: Dict[str, paddle.Tensor],
        x_1: Dict[str, paddle.Tensor],
    ) -> Tuple[Dict[str, paddle.Tensor], Dict[str, paddle.Tensor]]:
        assert bool(paddle.equal_all(x_0["batch"], x_1["batch"]))
        assert bool(paddle.equal_all(x_0["n_atoms"], x_1["n_atoms"]))

        n_atoms = x_0["n_atoms"]
        x_t = _clone_dict(x_0)
        z_data = {}

        for stochastic_interpolant, data_field in zip(
            self._stochastic_interpolants, self._data_fields
        ):
            field_name = data_field.value
            reshaped_t = reshape_t(t, n_atoms, data_field)

            if data_field == DataField.cell:
                batch_indices = paddle.arange(len(x_0["n_atoms"]))
            else:
                batch_indices = x_0["batch"]

            interpolated_x_t, z = stochastic_interpolant.interpolate(
                reshaped_t,
                x_0[field_name],
                x_1[field_name],
                batch_indices,
            )

            x_t[field_name] = interpolated_x_t
            z_data[field_name] = z

        return x_t, z_data

    def losses(
        self,
        model_function: Any,
        t: paddle.Tensor,
        x_0: Dict[str, paddle.Tensor],
        x_1: Dict[str, paddle.Tensor],
    ) -> Dict[str, paddle.Tensor]:
        x_t, z = self._interpolate(t, x_0, x_1)

        n_atoms = x_0["n_atoms"]

        losses = {}
        for stochastic_interpolant, data_field in zip(
            self._stochastic_interpolants, self._data_fields
        ):
            field_name = data_field.value
            b_data_field = field_name + "_b"
            eta_data_field = field_name + "_eta"

            reshaped_t = reshape_t(t, n_atoms, data_field)

            if data_field == DataField.cell:
                batch_indices = paddle.arange(len(x_0["n_atoms"]))
            else:
                batch_indices = x_0["batch"]

            def model_prediction_fn(x):
                x_t_clone = _clone_dict(x_t)
                x_t_clone[field_name] = x
                model_result = model_function(x_t_clone, t)
                return model_result[b_data_field], model_result[eta_data_field]

            field_losses = stochastic_interpolant.loss(
                model_prediction_fn,
                reshaped_t,
                x_0[field_name],
                x_1[field_name],
                x_t[field_name],
                z[field_name],
                batch_indices,
            )

            for loss_key, loss_value in field_losses.items():
                assert loss_key not in losses
                losses[f"{field_name}_{loss_key}"] = loss_value

        return losses

    def integrate(
        self,
        x_0: Dict[str, paddle.Tensor],
        model_function: Any,
        save_intermediate: bool = False,
        integration_time_steps: Optional[int] = None,
    ) -> Union[
        Dict[str, paddle.Tensor],
        Tuple[Dict[str, paddle.Tensor], List[Dict[str, paddle.Tensor]]],
    ]:
        num_steps = (
            self._integration_time_steps
            if integration_time_steps is None
            else integration_time_steps
        )
        if num_steps < 2:
            raise ValueError("integration_time_steps must be at least 2.")
        times = paddle.linspace(SMALL_TIME, BIG_TIME, num_steps)
        dt = (BIG_TIME - SMALL_TIME) / (num_steps - 1)

        x_t = _clone_dict(x_0)
        new_x_t = _clone_dict(x_0)

        if save_intermediate:
            inter_list = [x_t]
        else:
            inter_list = None

        with paddle.no_grad():
            for t_index in range(1, len(times)):
                t = times[t_index - 1]

                for stochastic_interpolant, data_field in zip(
                    self._stochastic_interpolants, self._data_fields
                ):
                    field_name = data_field.value

                    def model_prediction_fn(
                        time, x, _field_name=field_name, _x_t=x_t, _new_x_t=new_x_t
                    ):
                        t_val = time.item() if time.numel() == 1 else float(time[0])
                        time = paddle.full_like(
                            _x_t["n_atoms"].cast("float32"),
                            t_val,
                        )
                        x_int = _clone_dict(_new_x_t)
                        x_int[_field_name] = x
                        model_result = model_function(x_int, time)
                        return (
                            model_result[_field_name + "_b"],
                            model_result[_field_name + "_eta"],
                        )

                    if data_field == DataField.cell:
                        batch_indices = paddle.arange(len(x_0["n_atoms"]))
                    else:
                        batch_indices = x_0["batch"]

                    new_value = stochastic_interpolant.integrate(
                        model_prediction_fn,
                        x_t[field_name],
                        t,
                        dt,
                        batch_indices,
                    )

                    if data_field == DataField.cell:
                        new_value = paddle.clip(
                            new_value, -_CELL_CLIP_BOUND, _CELL_CLIP_BOUND
                        )
                    new_x_t[field_name] = new_value

                x_t = _clone_dict(new_x_t)

                if save_intermediate:
                    inter_list.append(x_t)

        if save_intermediate:
            return x_t, inter_list
        else:
            return x_t


def correct_for_minimum_permutation_distance(
    x_0: Dict[str, paddle.Tensor],
    x_1: Dict[str, paddle.Tensor],
    corrector: Corrector,
) -> None:
    """Reorder the base positions of ``x_0`` in place against ``x_1``.

    For every structure in the batch, the base positions are permuted by the
    assignment that minimises the distance to the corresponding target structure, so
    that the stochastic interpolants pair the closest atoms. Species are never
    permuted, which keeps crystal-structure prediction pairs on the same species.
    """
    if not bool(paddle.all(x_0["ptr"] == x_1["ptr"])):
        raise ValueError("x_0 and x_1 must hold the same structures in the same order.")
    if x_0["pos"].shape != x_1["pos"].shape:
        raise ValueError(
            f"x_0 and x_1 must hold the same positions, got {x_0['pos'].shape} "
            f"and {x_1['pos'].shape}."
        )

    ptr = x_0["ptr"]
    for i in range(len(ptr) - 1):
        start, end = int(ptr[i]), int(ptr[i + 1])
        sources = x_0["pos"][start:end]
        targets = x_1["pos"][start:end]
        distances = _pairwise_periodic_distances(sources, targets, corrector)
        _, assignment = linear_sum_assignment(distances.numpy())
        x_0["pos"][start:end] = sources[paddle.to_tensor(assignment, dtype="int64")]


def _pairwise_periodic_distances(
    sources: paddle.Tensor, targets: paddle.Tensor, corrector: Corrector
) -> paddle.Tensor:
    """Return the ``(len(targets), len(sources))`` matrix of periodic distances."""
    unwrapped_targets = corrector.unwrap(sources[None, :, :], targets[:, None, :])
    return paddle.norm(unwrapped_targets - sources[None, :, :], axis=-1)


def _resolve_si_class(class_name: str, default_module: str):
    """Resolve a class by name within the OMatG SI namespace."""
    import importlib

    if "." in class_name:
        module_path, cls_name = class_name.rsplit(".", 1)
        module = importlib.import_module(module_path)
        return getattr(module, cls_name)

    module = importlib.import_module(default_module)
    return getattr(module, class_name)


def _build_si_object(cfg: dict, default_module: str):
    """Build a single SI object from a {__class_name__, __init_params__} config."""
    cls = _resolve_si_class(cfg["__class_name__"], default_module)
    params = cfg.get("__init_params__", {})
    built_params = {}
    for key, val in params.items():
        if isinstance(val, dict) and "__class_name__" in val:
            built_params[key] = _build_si_object(val, default_module)
        elif isinstance(val, list):
            built_params[key] = [
                _build_si_object(item, default_module)
                if isinstance(item, dict) and "__class_name__" in item
                else item
                for item in val
            ]
        else:
            built_params[key] = val
    return cls(**built_params)


_ALLOWED_META_KEYS = {"relative_si_costs", "integration_time_steps"}


def build_si_from_cfg(si_scheduler_cfg: dict) -> StochasticInterpolants:
    """Build ``StochasticInterpolants`` from a per-field scheduler config.

    ``si_scheduler_cfg`` is a per-field dict: each key is a data field
    (``"species"``, ``"pos"`` or ``"cell"``) whose value is a
    ``__class_name__`` / ``__init_params__`` config (with arbitrarily deep
    nesting). The dict's key order defines the integration order, which is
    the same convention the YAML must preserve.

    Two metadata keys are recognised at the top level:
    ``integration_time_steps`` (required, integer) and ``relative_si_costs``
    (optional, dict). Any other top-level key is rejected to avoid silent
    typos. Classes are resolved within the OMatG SI namespace
    (``ppmat.models.omatg.si``) so the schema stays self-contained.
    """
    default_module = "ppmat.models.omatg.si"
    allowed_field_keys = {df.value for df in DataField}
    unknown_keys = [
        k
        for k in si_scheduler_cfg
        if k not in allowed_field_keys and k not in _ALLOWED_META_KEYS
    ]
    if unknown_keys:
        raise ValueError(
            f"si_scheduler_cfg has unknown top-level keys {unknown_keys}; "
            f"allowed keys are data fields {sorted(allowed_field_keys)} "
            f"and metadata {sorted(_ALLOWED_META_KEYS)}."
        )

    data_fields = [k for k in si_scheduler_cfg if k in allowed_field_keys]
    if not data_fields:
        raise ValueError(
            "si_scheduler_cfg must contain at least one data field "
            "('species', 'pos', or 'cell'); got an empty dict."
        )

    if "integration_time_steps" not in si_scheduler_cfg:
        raise ValueError(
            "si_scheduler_cfg must declare 'integration_time_steps' "
            "(int); omitting it is rejected to avoid silent drift."
        )
    integration_time_steps = si_scheduler_cfg["integration_time_steps"]
    if (
        not isinstance(integration_time_steps, int)
        or isinstance(integration_time_steps, bool)
        or integration_time_steps < 2
    ):
        raise ValueError(
            f"si_scheduler_cfg['integration_time_steps'] must be an int "
            f">= 2; got {integration_time_steps!r}."
        )

    stochastic_interpolants = [
        _build_si_object(si_scheduler_cfg[k], default_module) for k in data_fields
    ]
    return StochasticInterpolants(
        stochastic_interpolants=stochastic_interpolants,
        data_fields=data_fields,
        integration_time_steps=integration_time_steps,
    )


__all__ = [
    "BIG_TIME",
    "SMALL_TIME",
    "DataField",
    "StochasticInterpolant",
    "StochasticInterpolantSpecies",
    "StochasticInterpolants",
    "SingleStochasticInterpolant",
    "SingleStochasticInterpolantIdentity",
    "DiscreteFlowMatchingMask",
    "build_si_from_cfg",
    "correct_for_minimum_permutation_distance",
]

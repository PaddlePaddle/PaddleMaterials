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

"""Abstract base classes and concrete schedules for OMatG stochastic
interpolants.

The continuous-time SI machinery (alpha/beta/gamma schedules, correctors)
and the species-aware variants are all co-located here so the OMatG model
package stays self-contained without touching the framework-level
``ppmat.schedulers`` namespace.
"""

from abc import ABC
from abc import abstractmethod
from typing import Callable
from typing import Dict
from typing import Iterable
from typing import Tuple

import paddle


class TimeChecker:
    """Check that all times in a tensor are in [0,1]."""

    @staticmethod
    def _check_t(t: paddle.Tensor) -> paddle.Tensor:
        return paddle.all((0.0 <= t) & (t <= 1.0))


class Corrector(ABC):
    """Abstract corrector function (e.g., for PBC wrapping)."""

    @abstractmethod
    def correct(self, x: paddle.Tensor) -> paddle.Tensor:
        raise NotImplementedError

    @abstractmethod
    def unwrap(self, x_0: paddle.Tensor, x_1: paddle.Tensor) -> paddle.Tensor:
        raise NotImplementedError


class Epsilon(ABC, TimeChecker):
    """Abstract epsilon function epsilon(t)."""

    @abstractmethod
    def epsilon(self, t: paddle.Tensor) -> paddle.Tensor:
        raise NotImplementedError


class Interpolant(ABC, TimeChecker):
    """Interpolant schedule: x_t = alpha(t) * x_0 + beta(t) * x_1."""

    def interpolate(
        self, t: paddle.Tensor, x_0: paddle.Tensor, x_1: paddle.Tensor
    ) -> paddle.Tensor:
        assert bool(self._check_t(t))
        x_0prime = self.get_corrector().correct(x_0)
        x_1prime = self.get_corrector().unwrap(x_0prime, x_1)
        x_t = self.alpha(t) * x_0prime + self.beta(t) * x_1prime
        return self.get_corrector().correct(x_t)

    def interpolate_derivative(
        self, t: paddle.Tensor, x_0: paddle.Tensor, x_1: paddle.Tensor
    ) -> paddle.Tensor:
        assert bool(self._check_t(t))
        x_0prime = self.get_corrector().correct(x_0)
        x_1prime = self.get_corrector().unwrap(x_0prime, x_1)
        return self.alpha_dot(t) * x_0prime + self.beta_dot(t) * x_1prime

    @abstractmethod
    def get_corrector(self) -> Corrector:
        raise NotImplementedError

    @abstractmethod
    def alpha(self, t: paddle.Tensor) -> paddle.Tensor:
        raise NotImplementedError

    @abstractmethod
    def alpha_dot(self, t: paddle.Tensor) -> paddle.Tensor:
        raise NotImplementedError

    @abstractmethod
    def beta(self, t: paddle.Tensor) -> paddle.Tensor:
        raise NotImplementedError

    @abstractmethod
    def beta_dot(self, t: paddle.Tensor) -> paddle.Tensor:
        raise NotImplementedError


class LatentGamma(ABC, TimeChecker):
    """Abstract gamma function gamma(t) in gamma(t) * z."""

    @abstractmethod
    def gamma(self, t: paddle.Tensor) -> paddle.Tensor:
        raise NotImplementedError

    @abstractmethod
    def gamma_derivative(self, t: paddle.Tensor) -> paddle.Tensor:
        raise NotImplementedError

    @abstractmethod
    def requires_antithetic(self) -> bool:
        raise NotImplementedError


class IdentityCorrector(Corrector):
    def correct(self, x: paddle.Tensor) -> paddle.Tensor:
        return x

    def unwrap(self, x_0: paddle.Tensor, x_1: paddle.Tensor) -> paddle.Tensor:
        return x_1.clone()


class PeriodicBoundaryConditionsCorrector(Corrector):
    def __init__(self, min_value: float, max_value: float) -> None:
        super().__init__()
        if min_value >= max_value:
            raise ValueError("Minimum value must be less than maximum value.")
        self._min_value = min_value
        self._max_value = max_value
        self._range = max_value - min_value

    def correct(self, x: paddle.Tensor) -> paddle.Tensor:
        return paddle.remainder(x - self._min_value, self._range) + self._min_value

    def unwrap(self, x_0: paddle.Tensor, x_1: paddle.Tensor) -> paddle.Tensor:
        half = self._range / 2.0
        sep = x_1 - x_0
        return x_0 + paddle.remainder(sep + half, self._range) - half


class VanishingEpsilon(Epsilon):
    def __init__(self, c: float = 1.0, sigma: float = 0.01, mu: float = 0.075) -> None:
        super().__init__()
        self._c = c
        self._sigma = sigma
        self._mu = mu

    def epsilon(self, t: paddle.Tensor) -> paddle.Tensor:
        assert bool(self._check_t(t))
        f1 = paddle.sigmoid((t - self._mu) / self._sigma)
        f2 = paddle.sigmoid((1 - self._mu - t) / self._sigma)
        return self._c * f1 * f2


class LinearInterpolant(Interpolant):
    def alpha(self, t: paddle.Tensor) -> paddle.Tensor:
        return 1.0 - t

    def alpha_dot(self, t: paddle.Tensor) -> paddle.Tensor:
        return -paddle.ones_like(t)

    def beta(self, t: paddle.Tensor) -> paddle.Tensor:
        return t.clone()

    def beta_dot(self, t: paddle.Tensor) -> paddle.Tensor:
        return paddle.ones_like(t)

    def get_corrector(self) -> Corrector:
        return IdentityCorrector()


class PeriodicLinearInterpolant(LinearInterpolant):
    def get_corrector(self) -> Corrector:
        return PeriodicBoundaryConditionsCorrector(0.0, 1.0)


class LatentGammaSqrt(LatentGamma):
    def __init__(self, a: float) -> None:
        super().__init__()
        if a <= 0.0:
            raise ValueError("Constant a must be positive.")
        self._a = a

    def gamma(self, t: paddle.Tensor) -> paddle.Tensor:
        assert bool(self._check_t(t))
        return paddle.sqrt(self._a * t * (1.0 - t))

    def gamma_derivative(self, t: paddle.Tensor) -> paddle.Tensor:
        assert bool(self._check_t(t))
        return self._a * (1.0 - 2.0 * t) / (2.0 * paddle.sqrt(self._a * t * (1.0 - t)))

    def requires_antithetic(self) -> bool:
        return True


class StochasticInterpolant(ABC, TimeChecker):
    """Abstract class for a stochastic interpolant between x_0 and x_1."""

    @abstractmethod
    def interpolate(
        self,
        t: paddle.Tensor,
        x_0: paddle.Tensor,
        x_1: paddle.Tensor,
        batch_indices: paddle.Tensor,
    ) -> Tuple[paddle.Tensor, paddle.Tensor]:
        raise NotImplementedError

    @abstractmethod
    def loss_keys(self) -> Iterable[str]:
        raise NotImplementedError

    @abstractmethod
    def loss(
        self,
        model_function: Callable[[paddle.Tensor], Tuple[paddle.Tensor, paddle.Tensor]],
        t: paddle.Tensor,
        x_0: paddle.Tensor,
        x_1: paddle.Tensor,
        x_t: paddle.Tensor,
        z: paddle.Tensor,
        batch_indices: paddle.Tensor,
    ) -> Dict[str, paddle.Tensor]:
        raise NotImplementedError

    @abstractmethod
    def integrate(
        self,
        model_function: Callable[
            [paddle.Tensor, paddle.Tensor], Tuple[paddle.Tensor, paddle.Tensor]
        ],
        x_t: paddle.Tensor,
        time: paddle.Tensor,
        time_step: paddle.Tensor,
        batch_indices: paddle.Tensor,
    ) -> paddle.Tensor:
        raise NotImplementedError

    @abstractmethod
    def get_corrector(self) -> Corrector:
        raise NotImplementedError


class StochasticInterpolantSpecies(StochasticInterpolant, ABC):
    """Abstract class for a stochastic interpolant between species x_0 and x_1."""

    def get_corrector(self) -> Corrector:
        raise RuntimeError("Corrector not defined for StochasticInterpolantSpecies.")


__all__ = [
    "LinearInterpolant",
    "PeriodicLinearInterpolant",
    "VanishingEpsilon",
    "LatentGammaSqrt",
    "StochasticInterpolant",
    "StochasticInterpolantSpecies",
]

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

# This file adapted from https://github.com/PaddlePaddle/PaddleScience

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

from paddle import nn
from paddle import optimizer as optim
from paddle import regularizer
from paddle.incubate import optimizer as incubate_optim
from typing_extensions import Literal

from ppmat.utils import logger
from ppmat.utils import misc

if TYPE_CHECKING:
    import paddle

__all__ = ["SGD", "Momentum", "Adam", "RMSProp", "AdamW", "LBFGS", "OptimizerList"]


class SGD:
    """Stochastic Gradient Descent.

    Args:
        learning_rate (Union[float, optim.lr.LRScheduler], optional): The learning rate
            used to update parameter(s). Defaults to 0.001.
        weight_decay (Optional[Union[float, regularizer.L1Decay, regularizer.L2Decay]]):
            Regularization strategy. Defaults to None.
        grad_clip (Optional[Union[nn.ClipGradByNorm, nn.ClipGradByValue,
            nn.ClipGradByGlobalNorm]]): Gradient clipping strategy. Defaults to None.
    """

    def __init__(
        self,
        learning_rate: Union[float, optim.lr.LRScheduler] = 0.001,
        weight_decay: Optional[
            Union[float, regularizer.L1Decay, regularizer.L2Decay]
        ] = None,
        grad_clip: Optional[
            Union[nn.ClipGradByNorm, nn.ClipGradByValue, nn.ClipGradByGlobalNorm]
        ] = None,
    ):
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.grad_clip = grad_clip

    def __call__(self, model_list: Union[nn.Layer, Tuple[nn.Layer, ...]]):
        # model_list is None in static graph
        if not isinstance(model_list, (tuple, list)):
            model_list = (model_list,)
        parameters = (
            sum([m.parameters() for m in model_list], []) if model_list else None
        )
        opt = optim.SGD(
            learning_rate=self.learning_rate,
            parameters=parameters,
            weight_decay=self.weight_decay,
            grad_clip=self.grad_clip,
        )
        return opt


class Momentum:
    """Simple Momentum optimizer with velocity state.

    Args:
        learning_rate (Union[float, optim.lr.LRScheduler]): The learning rate
            used to update parameter(s).
        momentum (float): Momentum factor.
        weight_decay (Optional[Union[float, regularizer.L1Decay, regularizer.L2Decay]]):
            Regularization strategy. Defaults to None.
        grad_clip (Optional[Union[nn.ClipGradByNorm, nn.ClipGradByValue,
            nn.ClipGradByGlobalNorm]]): Gradient clipping strategy. Defaults to None.
        use_nesterov (bool, optional): Whether to use nesterov momentum.
            Defaults to False.
        no_weight_decay_name (Optional[str]): List of names of no weight decay
            parameters split by white space. Defaults to None.
    """

    def __init__(
        self,
        learning_rate: Union[float, optim.lr.LRScheduler],
        momentum: float,
        weight_decay: Optional[
            Union[float, regularizer.L1Decay, regularizer.L2Decay]
        ] = None,
        grad_clip: Optional[
            Union[nn.ClipGradByNorm, nn.ClipGradByValue, nn.ClipGradByGlobalNorm]
        ] = None,
        use_nesterov: bool = False,
        no_weight_decay_name: Optional[str] = None,
    ):
        super().__init__()
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.grad_clip = grad_clip
        self.use_nesterov = use_nesterov
        self.no_weight_decay_name_list = (
            no_weight_decay_name.split() if no_weight_decay_name else []
        )

    def __call__(self, model_list: Union[nn.Layer, Tuple[nn.Layer, ...]]):
        # model_list is None in static graph
        if not isinstance(model_list, (tuple, list)):
            model_list = (model_list,)
        parameters = None
        if len(self.no_weight_decay_name_list) > 0:
            params_with_decay = []
            params_without_decay = []
            for m in model_list:
                params = [
                    p
                    for n, p in m.named_parameters()
                    if not any(nd in n for nd in self.no_weight_decay_name_list)
                ]
                params_with_decay.extend(params)
                params = [
                    p
                    for n, p in m.named_parameters()
                    if any(nd in n for nd in self.no_weight_decay_name_list)
                ]
                params_without_decay.extend(params)
            parameters = [
                {"params": params_with_decay, "weight_decay": self.weight_decay},
                {"params": params_without_decay, "weight_decay": 0.0},
            ]
        else:
            parameters = (
                sum([m.parameters() for m in model_list], []) if model_list else None
            )
        opt = optim.Momentum(
            learning_rate=self.learning_rate,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
            grad_clip=self.grad_clip,
            use_nesterov=self.use_nesterov,
            parameters=parameters,
        )
        if hasattr(opt, "_use_multi_tensor"):
            opt = optim.Momentum(
                learning_rate=self.learning_rate,
                momentum=self.momentum,
                weight_decay=self.weight_decay,
                grad_clip=self.grad_clip,
                parameters=parameters,
                use_nesterov=self.use_nesterov,
                use_multi_tensor=True,
            )
        return opt


class Adam:
    """Adam: A Method for Stochastic Optimization.

    Args:
        learning_rate (Union[float, optim.lr.LRScheduler], optional): The learning rate
            used to update parameter(s). Defaults to 0.001.
        beta1 (float, optional): The exponential decay rate for the 1st moment
            estimates. Defaults to 0.9.
        beta2 (float, optional): The exponential decay rate for the 2nd moment
            estimates. Defaults to 0.999.
        epsilon (float, optional): A small float value for numerical stability.
            Defaults to 1e-08.
        weight_decay (Optional[Union[float, regularizer.L1Decay, regularizer.L2Decay]])
             : Regularization strategy. Defaults to None.
        grad_clip (Optional[Union[nn.ClipGradByNorm, nn.ClipGradByValue,
            nn.ClipGradByGlobalNorm]]): Gradient clipping strategy. Defaults to None.
        lazy_mode (bool, optional): Whether to enable lazy mode for moving-average.
            Defaults to False.
    """

    def __init__(
        self,
        learning_rate: Union[float, optim.lr.LRScheduler] = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-08,
        weight_decay: Optional[
            Union[float, regularizer.L1Decay, regularizer.L2Decay]
        ] = None,
        grad_clip: Optional[
            Union[nn.ClipGradByNorm, nn.ClipGradByValue, nn.ClipGradByGlobalNorm]
        ] = None,
        lazy_mode: bool = False,
        amsgrad: bool = False,
    ):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.grad_clip = grad_clip
        self.lazy_mode = lazy_mode
        self.amsgrad = amsgrad

    def __call__(self, model_list: Union[nn.Layer, Tuple[nn.Layer, ...]]):
        # model_list is None in static graph
        if not isinstance(model_list, (tuple, list)):
            model_list = (model_list,)
        parameters = (
            sum([m.parameters() for m in model_list], []) if model_list else None
        )
        opt = optim.Adam(
            learning_rate=self.learning_rate,
            beta1=self.beta1,
            beta2=self.beta2,
            epsilon=self.epsilon,
            weight_decay=self.weight_decay,
            grad_clip=self.grad_clip,
            lazy_mode=self.lazy_mode,
            parameters=parameters,
            amsgrad=self.amsgrad,
        )
        return opt


class LBFGS:
    """The L-BFGS is a quasi-Newton method for solving an unconstrained optimization
        problem over a differentiable function. Closely related is the Newton method
        for minimization.

    Args:
        learning_rate (float, optional): The learning rate
            used to update parameter(s). Defaults to 1.0.
        max_iter (int, optional): Maximal number of iterations per optimization step.
            Defaults to 1.
        max_eval (Optional[int]): Maximal number of function evaluations per
            optimization step. Defaults to None.
        tolerance_grad (float, optional): Termination tolerance on first order
            optimality. Defaults to 1e-07.
        tolerance_change (float, optional): Termination tolerance on function
            value/parameter changes. Defaults to 1e-09.
        history_size (int, optional): Update history size. Defaults to 100.
        line_search_fn (Optional[Literal["strong_wolfe"]]): Either 'strong_wolfe' or
            None. Defaults to "strong_wolfe".
    """

    def __init__(
        self,
        learning_rate: float = 1.0,
        max_iter: int = 1,
        max_eval: Optional[int] = None,
        tolerance_grad: float = 1e-07,
        tolerance_change: float = 1e-09,
        history_size: int = 100,
        line_search_fn: Optional[Literal["strong_wolfe"]] = "strong_wolfe",
    ):
        self.lr = learning_rate
        self.max_iter = max_iter
        self.max_eval = max_eval
        self.tolerance_grad = tolerance_grad
        self.tolerance_change = tolerance_change
        self.history_size = history_size
        self.line_search_fn = line_search_fn

    def __call__(self, model_list: Union[nn.Layer, Tuple[nn.Layer, ...]]):
        # model_list is None in static graph
        if not isinstance(model_list, (tuple, list)):
            model_list = (model_list,)
        parameters = (
            sum([m.parameters() for m in model_list], []) if model_list else None
        )
        try:
            opt = getattr(optim, "LBFGS")(
                learning_rate=self.lr,
                max_iter=self.max_iter,
                max_eval=self.max_eval,
                tolerance_grad=self.tolerance_grad,
                tolerance_change=self.tolerance_change,
                history_size=self.history_size,
                line_search_fn=self.line_search_fn,
                parameters=parameters,
            )
        except AttributeError:
            opt = getattr(incubate_optim, "LBFGS")(
                learning_rate=self.lr,
                max_iter=self.max_iter,
                max_eval=self.max_eval,
                tolerance_grad=self.tolerance_grad,
                tolerance_change=self.tolerance_change,
                history_size=self.history_size,
                line_search_fn=self.line_search_fn,
                parameters=parameters,
            )
        return opt


class RMSProp:
    """Root Mean Squared Propagation (RMSProp) is an unpublished, adaptive learning
        rate method.

    Args:
        learning_rate (Union[float, optim.lr.LRScheduler]): The learning rate
            used to update parameter(s)
        rho (float, optional): Factor ρ in equation. Defaults to 0.95.
        epsilon (float, optional): Factor ϵ in equation as a smoothing term.
            Defaults to 1e-6.
        momentum (float, optional):β in equation is the momentum term. Defaults to 0.0.
        weight_decay (Optional[Union[float, regularizer.L1Decay, regularizer.L2Decay]]):
            Regularization strategy. Defaults to None.
        grad_clip (Optional[Union[nn.ClipGradByNorm, nn.ClipGradByValue,
            nn.ClipGradByGlobalNorm]]): Gradient clipping strategy. Defaults to None.
    """

    def __init__(
        self,
        learning_rate: Union[float, optim.lr.LRScheduler],
        rho: float = 0.95,
        epsilon: float = 1e-6,
        momentum: float = 0.0,
        weight_decay: Optional[
            Union[float, regularizer.L1Decay, regularizer.L2Decay]
        ] = None,
        grad_clip: Optional[
            Union[nn.ClipGradByNorm, nn.ClipGradByValue, nn.ClipGradByGlobalNorm]
        ] = None,
    ):
        super().__init__()
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.rho = rho
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.grad_clip = grad_clip

    def __call__(self, model_list: Union[nn.Layer, Tuple[nn.Layer, ...]]):
        # model_list is None in static graph
        if not isinstance(model_list, (tuple, list)):
            model_list = (model_list,)
        parameters = (
            sum([m.parameters() for m in model_list], []) if model_list else None
        )
        opt = optim.RMSProp(
            learning_rate=self.learning_rate,
            momentum=self.momentum,
            rho=self.rho,
            epsilon=self.epsilon,
            weight_decay=self.weight_decay,
            grad_clip=self.grad_clip,
            parameters=parameters,
        )
        return opt


class AdamW:
    """AdamW is implemented based on DECOUPLED WEIGHT DECAY REGULARIZATION.

    Args:
        learning_rate (Union[float, optim.lr.LRScheduler], optional): The learning rate
            used to update parameter(s). Defaults to 0.001.
        beta1 (float, optional): The exponential decay rate for the 1st moment
            estimates. Defaults to 0.9.
        beta2 (float, optional): The exponential decay rate for the 2nd moment
            estimates. Defaults to 0.999.
        epsilon (float, optional): A small float value for numerical stability.
            Defaults to 1e-8.
        weight_decay (float, optional): Regularization coefficient. Defaults to 0.01.
        grad_clip (Optional[Union[nn.ClipGradByNorm, nn.ClipGradByValue,
            nn.ClipGradByGlobalNorm]]): Gradient clipping strategy. Defaults to None.
        no_weight_decay_name (Optional[str]): List of names of no weight decay
            parameters split by white space. Defaults to None.
        one_dim_param_no_weight_decay (bool, optional): Apply no weight decay on
            1-D parameter(s). Defaults to False.
        named_lr_groups (Optional[List[Dict[str, Union[str, float]]]], optional):
            Per-layer learning-rate (and optionally weight-decay) multipliers.
            Each entry is a dict like ``{"name": "layer_segment",
            "lr_multiplier": 0.1, "weight_decay_multiplier": 0.0}``. The
            ``name`` field matches any dot-separated segment of a parameter's
            full path (avoiding substring false positives such as
            ``"time_embedder"`` accidentally matching
            ``"time_embedder_aux.weight"``); the first matching entry wins.
            The effective learning rate of a matched parameter is
            ``learning_rate * lr_multiplier``; if ``weight_decay_multiplier``
            is given, the effective weight decay is also scaled
            (``weight_decay * weight_decay_multiplier``). Unmatched parameters
            fall back to ``lr_multiplier=1.0`` and no weight-decay multiplier.
            Defaults to None.
    """

    def __init__(
        self,
        learning_rate: Union[float, optim.lr.LRScheduler] = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
        weight_decay: float = 0.001,
        grad_clip: Optional[
            Union[nn.ClipGradByNorm, nn.ClipGradByValue, nn.ClipGradByGlobalNorm]
        ] = None,
        no_weight_decay_name: Optional[str] = None,
        one_dim_param_no_weight_decay: bool = False,
        amsgrad: bool = False,
        multi_precision: bool = False,
        named_lr_groups: Optional[List[Dict[str, Union[str, float]]]] = None,
    ):
        super().__init__()
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.grad_clip = grad_clip
        self.weight_decay = weight_decay
        self.no_weight_decay_name_list = (
            no_weight_decay_name.split() if no_weight_decay_name else []
        )
        self.one_dim_param_no_weight_decay = one_dim_param_no_weight_decay
        self.amsgrad = amsgrad
        self.multi_precision = multi_precision
        self.named_lr_groups = named_lr_groups

    def __call__(self, model_list: Union[nn.Layer, Tuple[nn.Layer, ...]]):
        # model_list is None in static graph
        if not isinstance(model_list, (tuple, list)):
            model_list = (model_list,)
        if self.named_lr_groups:
            parameters = self._build_named_lr_groups(model_list)
        else:
            parameters = (
                sum([m.parameters() for m in model_list], []) if model_list else None
            )

        # TODO(gaotingquan): Model_list is None when in static graph, "no_weight_decay"
        # not work.
        if model_list is None:
            if (
                self.one_dim_param_no_weight_decay
                or len(self.no_weight_decay_name_list) != 0
            ):
                msg = '"AdamW" does not support setting "no_weight_decay" in static '
                +"graph. Please use dynamic graph."
                logger.error(Exception(msg))
                raise Exception(msg)

        self.no_weight_decay_param_name_list = (
            [
                p.name
                for model in model_list
                for n, p in model.named_parameters()
                if any(nd in n for nd in self.no_weight_decay_name_list)
            ]
            if model_list
            else []
        )

        if self.one_dim_param_no_weight_decay:
            self.no_weight_decay_param_name_list += (
                [
                    p.name
                    for model in model_list
                    for n, p in model.named_parameters()
                    if len(p.shape) == 1
                ]
                if model_list
                else []
            )

        opt = optim.AdamW(
            learning_rate=self.learning_rate,
            beta1=self.beta1,
            beta2=self.beta2,
            epsilon=self.epsilon,
            parameters=parameters,
            weight_decay=self.weight_decay,
            grad_clip=self.grad_clip,
            apply_decay_param_fun=self._apply_decay_param_fun,
            amsgrad=self.amsgrad,
            multi_precision=self.multi_precision,
        )
        return opt

    def _build_named_lr_groups(self, model_list):
        """Group parameters by matched layer-name segment with lr/wd multipliers.

        Match semantics: the ``name`` field matches when it equals any
        dot-separated segment of the parameter's full path. Example: a
        parameter named
        ``"atom_coord_diffusion_model.wyckoff_and_element_sampler.time_embedder.weight"``
        matches pattern ``"wyckoff_and_element_sampler"``. This avoids the
        false-positive risks of substring matching (e.g. ``"time_embedder"``
        accidentally matching ``"time_embedder_aux.weight"``).

        Paddle parameter-group semantics (verified experimentally against
        ``paddle.optimizer``): a float ``learning_rate`` entry in a group is
        a **multiplier** of the optimizer's global learning rate (the global
        rate may itself be a float or an ``LRScheduler``); a ``weight_decay``
        entry in a group is an **absolute** value that overrides the
        optimizer-level weight decay for that group. Group entries whose
        ``weight_decay_multiplier`` is set therefore carry
        ``weight_decay * weight_decay_multiplier``. Note that a parameter
        matched by ``no_weight_decay_name`` never decays, regardless of any
        group ``weight_decay_multiplier`` (``apply_decay_param_fun`` wins).

        Parameters that match no entry fall back to ``lr_multiplier=1.0``
        and no weight-decay multiplier (i.e. they inherit the optimizer's
        full weight decay).

        Args:
            model_list (Tuple[nn.Layer, ...]): Tuple of model(s).

        Returns:
            List[Dict]: Paddle parameter-group spec (list of dicts, each with
            ``params``, ``learning_rate``, and ``weight_decay`` when
            ``weight_decay_multiplier`` is configured for that group).
        """
        grouped = {}
        for model in model_list:
            for name, param in model.named_parameters():
                lr_multiplier = 1.0
                wd_multiplier = None
                for group in self.named_lr_groups:
                    if "name" not in group:
                        raise ValueError(
                            "named_lr_groups entry missing 'name': "
                            f"{group}"
                        )
                    if group["name"] in name.split("."):
                        lr_multiplier = group.get("lr_multiplier", 1.0)
                        wd_multiplier = group.get("weight_decay_multiplier")
                        break
                key = (lr_multiplier, wd_multiplier)
                grouped.setdefault(key, []).append(param)

        base_wd = self.weight_decay
        result = []
        for (lr_multiplier, wd_multiplier), params in grouped.items():
            group_dict = {
                "params": params,
                "learning_rate": lr_multiplier,
            }
            if wd_multiplier is not None:
                group_dict["weight_decay"] = base_wd * wd_multiplier
            result.append(group_dict)
        return result

    def _apply_decay_param_fun(self, name):
        return name not in self.no_weight_decay_param_name_list


class OptimizerList:
    """OptimizerList which wrap more than one optimizer.
    NOTE: LBFGS is not supported yet.

    Args:
        optimizer_list (Tuple[optim.Optimizer, ...]): Optimizers listed in a tuple.
    """

    def __init__(self, optimizer_list: Tuple[optim.Optimizer, ...]):
        super().__init__()
        self._opt_list = optimizer_list
        if "LBFGS" in set(misc.typename(opt) for opt in optimizer_list):
            raise ValueError("LBFGS is not supported in OptimizerList yet.")

    def step(self):
        for opt in self._opt_list:
            opt.step()

    def clear_grad(self):
        for opt in self._opt_list:
            opt.clear_grad()

    def get_lr(self) -> float:
        """Return learning rate of first optimizer"""
        return self._opt_list[0].get_lr()

    def set_state_dict(self, state_dicts: List[Dict[str, "paddle.Tensor"]]):
        for i, opt in enumerate(self._opt_list):
            opt.set_state_dict(state_dicts[i])

    def state_dict(self) -> List[Dict[str, "paddle.Tensor"]]:
        state_dicts = [opt.state_dict() for opt in self._opt_list]
        return state_dicts

    def __len__(self) -> int:
        return len(self._opt_list)

    def __getitem__(self, idx):
        return self._opt_list[idx]

    def __setitem__(self, idx, opt):
        raise NotImplementedError("Can not modify any item in OptimizerList.")

    def __iter__(self):
        yield from iter(self._opt_list)

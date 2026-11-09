# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

"""LiFlow trajectory simulator.

This is a Paddle port of ``liflow.utils.inference.FlowSimulator`` from
https://github.com/learningmatter-mit/liflow (MIT license).  The integration
bookkeeping is intentionally kept identical to the reference implementation:

* the neighbour list is built **once** per flow integration (from the condition
  frame), not rebuilt at every Euler/Heun sub-step;
* the prior is added to the flow variable for the **propagator only**;
* :class:`MaxwellBoltzmannPrior` / :class:`AdaptiveMaxwellBoltzmannPrior`
  sampling uses the atomic masses and the requested temperature;
* ``corrector_every`` is a period, ``(i_step + 1) % corrector_every == 0``;
* the centre-of-mass correction removes the COM *displacement* of a step, so
  the absolute COM of the input frame is preserved;
* a diverged step is retried ``num_retries`` times before the run stops.

The simulator never receives future positions: only the initial frame, the
atomic numbers, the cell and the temperature.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import paddle
from ase import data as ase_data

from ppmat.models import build_model
from ppmat.models.liflow.geometry import get_neighbor_list_batch
from ppmat.models.liflow.prior import AdaptiveMaxwellBoltzmannPrior
from ppmat.models.liflow.prior import MaxwellBoltzmannPrior
from ppmat.models.liflow.prior import NormalPrior
from ppmat.models.liflow.prior import Prior
from ppmat.utils import logger
from ppmat.utils import save_load

DEFAULT_CUTOFF = 5.0
DEFAULT_PROPAGATOR_PRIOR = [[1.0, 10.0], [0.316, 3.16]]
DEFAULT_CORRECTOR_PRIOR_SCALE = 0.1


def _load_element_index(element_index) -> np.ndarray | None:
    """Normalise ``element_index`` into a ``(max(Z) + 1,)`` lookup table.

    Accepts an array, a path to ``.npy``, or a path to ``.npz`` containing the
    ``element_index`` array.
    """
    if element_index is None:
        return None
    if isinstance(element_index, (str, Path)):
        path = Path(element_index)
        if not path.exists():
            raise FileNotFoundError(f"element_index file not found: {path}")
        if path.suffix == ".npz":
            with np.load(path) as archive:
                if "element_index" not in archive:
                    raise KeyError(f"{path} does not contain an 'element_index' array.")
                array = np.asarray(archive["element_index"])
        else:
            array = np.asarray(np.load(path))
    else:
        array = np.asarray(element_index)
    array = array.astype(np.int64).reshape([-1])
    if array.size == 0:
        raise ValueError("element_index must not be empty.")
    if array.min() < 0:
        raise ValueError("element_index must be non-negative.")
    return array


class IntegratorPredictor:
    """Iterative LiFlow propagator/corrector simulator.

    Args:
        propagator_model: Instantiated propagator.  When ``None`` it is built
            from ``propagator_config`` and loaded from ``propagator_checkpoint``.
        corrector_model: Optional instantiated corrector.
        propagator_config: Training YAML of the propagator.
        propagator_checkpoint: Checkpoint loaded with ``strict=True``.
        corrector_config: Training YAML of the corrector.
        corrector_checkpoint: Checkpoint loaded with ``strict=True``.
        cutoff: Neighbour-list cutoff in Angstrom.
        periodic: Whether to use periodic boundary conditions.
        element_index: Array (or ``.npy``/``.npz`` path) mapping atomic number
            ``Z`` to the element-embedding index used during training.  It is
            required to run with a real checkpoint.
        propagator_prior: Prior added before each propagator flow.  Defaults to
            ``AdaptiveMaxwellBoltzmannPrior(scale=[[1.0, 10.0], [0.316, 3.16]])``.
        corrector_prior: Prior sampled before each corrector flow.  Defaults to
            ``MaxwellBoltzmannPrior(scale=0.1)``.
        scale_Li_index: Column index of the adaptive Li prior scale.
        scale_frame_index: Column index of the adaptive frame prior scale.
        execution_backend: Optional backend override (``"eager"``/``"cinn"``).
            ``None`` keeps the backend selected by the model itself.
    """

    def __init__(
        self,
        propagator_model: Optional[paddle.nn.Layer] = None,
        corrector_model: Optional[paddle.nn.Layer] = None,
        *,
        propagator_config: Optional[str] = None,
        propagator_checkpoint: Optional[str] = None,
        corrector_config: Optional[str] = None,
        corrector_checkpoint: Optional[str] = None,
        cutoff: float = DEFAULT_CUTOFF,
        periodic: bool = True,
        element_index=None,
        propagator_prior: Optional[Prior] = None,
        corrector_prior: Optional[Prior] = None,
        scale_Li_index: int = 0,
        scale_frame_index: int = 0,
        execution_backend: Optional[str] = None,
    ):
        if propagator_model is None:
            propagator_model = self._load_model(
                propagator_config, propagator_checkpoint, "propagator"
            )
        if corrector_model is None and (
            corrector_config is not None or corrector_checkpoint is not None
        ):
            corrector_model = self._load_model(
                corrector_config, corrector_checkpoint, "corrector"
            )

        self.propagator = propagator_model
        self.corrector = corrector_model
        self.cutoff = float(cutoff)
        self.periodic = bool(periodic)
        self.element_index = _load_element_index(element_index)

        self.propagator_prior = propagator_prior or AdaptiveMaxwellBoltzmannPrior(
            scale=DEFAULT_PROPAGATOR_PRIOR
        )
        self.corrector_prior = corrector_prior or MaxwellBoltzmannPrior(
            scale=DEFAULT_CORRECTOR_PRIOR_SCALE
        )
        self.scale_Li_index = int(scale_Li_index)
        self.scale_frame_index = int(scale_frame_index)

        from ppmat.utils.execution import configure_execution_backend

        for model in (self.propagator, self.corrector):
            if model is None:
                continue
            configure_execution_backend(
                model,
                execution_backend,
                world_size=paddle.distributed.get_world_size(),
                owner="IntegratorPredictor",
            )
            model.eval()

    # ------------------------------------------------------------------
    # construction helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _load_model(
        config_path: Optional[str],
        checkpoint_path: Optional[str],
        role: str,
    ) -> paddle.nn.Layer:
        if config_path is None or checkpoint_path is None:
            raise ValueError(
                f"{role}: both config_path and checkpoint_path are required when "
                "no model instance is provided."
            )
        from omegaconf import OmegaConf

        config = OmegaConf.load(config_path)
        config = OmegaConf.to_container(config, resolve=True)
        model_config = config.get("Model", None)
        assert model_config is not None, "Model config must be provided."
        model = build_model(model_config)
        logger.info(f"Loading {role} weights from {checkpoint_path} with strict=True")
        save_load.load_pretrain(model, checkpoint_path, strict=True)
        return model

    def _elements(self, atomic_numbers: np.ndarray) -> np.ndarray:
        """Map atomic numbers ``Z`` to the contiguous element indices."""
        if self.element_index is None:
            raise ValueError(
                "IntegratorPredictor requires `element_index` to map atomic "
                "numbers to element-embedding indices.  Pass the training "
                "`element_index.npy` (or an equivalent array)."
            )
        max_z = int(atomic_numbers.max())
        if max_z >= self.element_index.size:
            raise ValueError(
                f"element_index has {self.element_index.size} entries but the "
                f"input contains Z={max_z}."
            )
        return self.element_index[atomic_numbers]

    def _sample_prior(
        self,
        prior: Prior,
        atomic_numbers: np.ndarray,
        masses: np.ndarray,
        shape: tuple,
        temperature: float,
        rng: np.random.Generator | None,
    ) -> np.ndarray:
        if isinstance(prior, AdaptiveMaxwellBoltzmannPrior):
            return prior.sample(
                temperature=temperature,
                atomic_numbers=atomic_numbers,
                masses=masses,
                scale_Li_index=self.scale_Li_index,
                scale_frame_index=self.scale_frame_index,
                shape=shape,
                rng=rng,
            )
        if isinstance(prior, MaxwellBoltzmannPrior):
            return prior.sample(
                temperature=temperature, masses=masses, shape=shape, rng=rng
            )
        if isinstance(prior, NormalPrior):
            return prior.sample(shape=shape, rng=rng)
        raise NotImplementedError(f"Unsupported prior type: {type(prior).__name__}.")

    def _initialize_batch(
        self,
        positions: np.ndarray,
        prior: Prior,
        atomic_numbers: np.ndarray,
        elements: np.ndarray,
        masses: np.ndarray,
        lattice: np.ndarray,
        temperature: float,
        rng: np.random.Generator | None,
    ) -> dict:
        """Build the flow state used by one integration.

        Mirrors ``FlowSimulator.initialize_batch``: the neighbour list is built
        once from ``positions`` and is kept fixed during the integration, and
        the prior is sampled but not yet applied.
        """
        positions = np.asarray(positions, dtype=np.float32)
        edge_index, shifts = get_neighbor_list_batch(
            positions_batch=[positions],
            lattice=lattice,
            cutoff=self.cutoff,
            periodic=self.periodic,
        )
        sampled_prior = self._sample_prior(
            prior, atomic_numbers, masses, positions.shape, temperature, rng
        ).astype(np.float32)
        num_atoms = int(positions.shape[0])
        return {
            "condition_positions": paddle.to_tensor(positions.copy()),
            "flow_positions": paddle.to_tensor(positions.copy()),
            "prior": paddle.to_tensor(sampled_prior),
            "edge_index": paddle.to_tensor(edge_index.astype(np.int64)),
            "shifts": paddle.to_tensor(shifts.astype(np.float32)),
            "elements": paddle.to_tensor(elements.astype(np.int64)),
            "node_time": paddle.zeros([num_atoms], dtype="float32"),
            "node_temperature": paddle.full(
                [num_atoms], float(temperature), dtype="float32"
            ),
        }

    @staticmethod
    def _prediction_key(model) -> str:
        mode = getattr(model, "prediction_mode", "velocity")
        return "velocity" if mode == "velocity" else "displacement"

    def _predict(self, model, batch: dict, flow_positions, time: float):
        batch_data = dict(batch)
        batch_data["flow_positions"] = flow_positions
        batch_data["node_time"] = paddle.full(
            [int(flow_positions.shape[0])], float(time), dtype="float32"
        )
        outputs = model.predict(batch_data)
        return outputs[self._prediction_key(model)]

    def _propagate_flow(
        self,
        model,
        batch: dict,
        flow_steps: int,
        solver: str,
    ):
        """Integrate one flow from ``t=0`` to ``t=1`` (reference semantics)."""
        if solver not in {"euler", "heun"}:
            raise ValueError(f"solver must be 'euler' or 'heun', got {solver!r}.")
        if flow_steps <= 0:
            raise ValueError("flow_steps must be positive.")

        mode = getattr(model, "prediction_mode", "velocity")
        t_list = np.linspace(0.0, 1.0, int(flow_steps) + 1, dtype=np.float32)
        step_dt = float(t_list[1] - t_list[0])
        condition = batch["condition_positions"]
        flow = batch["flow_positions"]

        for i, time in enumerate(t_list[:-1]):
            time = float(time)
            prediction = self._predict(model, batch, flow, time)
            if mode == "velocity":
                dx_dt = prediction
            else:
                dx_dt = (condition + prediction - flow) / (1.0 - time)

            if solver == "euler":
                flow = flow + dx_dt * step_dt
            else:
                next_time = time + step_dt
                next_flow = flow + dx_dt * step_dt
                next_prediction = self._predict(model, batch, next_flow, next_time)
                if mode == "data" and i == int(flow_steps) - 1:
                    # ``1 - next_t`` is zero at the last step; the reference
                    # implementation returns the data prediction directly.
                    return condition + next_prediction
                if mode == "velocity":
                    next_dx_dt = next_prediction
                else:
                    next_dx_dt = (condition + next_prediction - next_flow) / (
                        1.0 - next_time
                    )
                flow = flow + 0.5 * (dx_dt + next_dx_dt) * step_dt
            batch["flow_positions"] = flow
        return flow

    def _step(
        self,
        positions: np.ndarray,
        atomic_numbers: np.ndarray,
        elements: np.ndarray,
        masses: np.ndarray,
        lattice: np.ndarray,
        temperature: float,
        flow_steps: int,
        solver: str,
        use_corrector: bool,
        rng: np.random.Generator | None,
    ) -> np.ndarray:
        # Propagator: the prior shifts the start of the flow.
        batch = self._initialize_batch(
            positions,
            self.propagator_prior,
            atomic_numbers,
            elements,
            masses,
            lattice,
            temperature,
            rng,
        )
        batch["flow_positions"] = batch["flow_positions"] + batch["prior"]
        new_positions = self._propagate_flow(self.propagator, batch, flow_steps, solver)
        new_positions = new_positions.numpy()

        if use_corrector:
            batch = self._initialize_batch(
                new_positions,
                self.corrector_prior,
                atomic_numbers,
                elements,
                masses,
                lattice,
                temperature,
                rng,
            )
            new_positions = self._propagate_flow(
                self.corrector, batch, flow_steps, solver
            ).numpy()
        return new_positions.astype(np.float32)

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def run(
        self,
        positions: np.ndarray,
        atomic_numbers: np.ndarray,
        lattice: np.ndarray,
        temperature: float,
        *,
        steps: int = 25,
        flow_steps: int = 10,
        solver: str = "euler",
        corrector_every: int = 1,
        fix_com: bool = True,
        num_retries: int = 2,
        seed: Optional[int] = 42,
    ) -> np.ndarray:
        """Generate a trajectory from a single initial frame.

        Args:
            positions: ``[n_atoms, 3]`` initial positions.
            atomic_numbers: ``[n_atoms]`` atomic numbers.
            lattice: ``[3, 3]`` cell.
            temperature: Simulation temperature in Kelvin.
            steps: Number of propagator steps (frames generated).
            flow_steps: Euler/Heun sub-steps per flow integration.
            solver: ``"euler"`` or ``"heun"``.
            corrector_every: Run the corrector every ``n`` propagation steps.
            fix_com: Remove the centre-of-mass displacement of each step.
            num_retries: Retries allowed for a diverged step.
            seed: Seed of the prior sampler.  ``None`` uses the prior's own RNG.

        Returns:
            A ``[steps + 1, n_atoms, 3]`` ``float32`` trajectory whose first
            frame is the unmodified input.
        """
        if solver not in {"euler", "heun"}:
            raise ValueError(f"solver must be 'euler' or 'heun', got {solver!r}.")
        if steps <= 0:
            raise ValueError("steps must be positive.")
        if corrector_every <= 0:
            raise ValueError("corrector_every must be positive.")
        if num_retries < 0:
            raise ValueError("num_retries must be non-negative.")

        positions = np.array(positions, dtype=np.float32)
        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError("positions must have shape [n_atoms, 3].")
        atomic_numbers = np.asarray(atomic_numbers, dtype=np.int64).reshape([-1])
        if atomic_numbers.shape[0] != positions.shape[0]:
            raise ValueError(
                "atomic_numbers and positions must contain the same number of atoms."
            )
        lattice = (
            np.zeros((3, 3), dtype=np.float32)
            if lattice is None
            else np.asarray(lattice, dtype=np.float32)
        )
        if lattice.shape != (3, 3):
            raise ValueError("lattice must have shape [3, 3].")

        elements = self._elements(atomic_numbers)
        masses = ase_data.atomic_masses[atomic_numbers].astype(np.float64)

        rng = None if seed is None else np.random.default_rng(int(seed))

        trajectory = [positions.copy()]
        current = positions.copy()
        with paddle.no_grad():
            for i_step in range(int(steps)):
                positions_prev = current.copy()
                use_corrector = (self.corrector is not None) and (
                    (i_step + 1) % int(corrector_every) == 0
                )
                success = False
                for _ in range(int(num_retries) + 1):
                    current = self._step(
                        positions_prev,
                        atomic_numbers,
                        elements,
                        masses,
                        lattice,
                        float(temperature),
                        int(flow_steps),
                        solver,
                        use_corrector,
                        rng,
                    )
                    if fix_com:
                        displacement = (current - positions_prev).astype(np.float64)
                        com_shift = np.sum(
                            displacement * masses[:, None], axis=0
                        ) / np.sum(masses)
                        current = (current - com_shift[None, :]).astype(np.float32)
                    diverged = bool(
                        np.isnan(current).any()
                        or np.abs(current - positions_prev).max() > 1e3
                    )
                    if not diverged:
                        success = True
                        break
                if not success:
                    logger.warning(
                        f"LiFlow trajectory diverged at step {i_step}; "
                        "returning the frames generated so far."
                    )
                    break
                trajectory.append(current.copy())

        return np.stack(trajectory, axis=0).astype(np.float32)

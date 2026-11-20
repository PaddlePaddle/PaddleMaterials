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

from typing import Literal
from typing import Optional
from typing import Tuple
from typing import Union

import paddle

from ppmat.utils.misc import aggregate_per_sample
from ppmat.utils.misc import maybe_expand


class D3PMScheduler:
    """D3PM Scheduler

    Args:
        num_train_timesteps (int, optional): Number of training timesteps. Defaults to
            1000.
        beta_start (float, optional): Beta start. Defaults to 0.001.
        beta_end (float, optional): Beta end. Defaults to 0.1.
        beta_schedule (str, optional): Beta schedule. Defaults to "standard".
        scale (float, optional): Scale, used when `beta_schedule` is set to `standard`.
            Defaults to 1.0.
        dim (int, optional): Total dimension. Defaults to 101.
    """

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.001,
        beta_end: float = 0.1,
        beta_schedule: str = "standard",
        scale: float = 1.0,
        dim: int = 101,
    ):
        self.num_train_timesteps = num_train_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_schedule = beta_schedule
        self.scale = scale
        self.dim = dim

        if beta_schedule == "standard":
            steps = paddle.linspace(
                0, num_train_timesteps - 1, num_train_timesteps, dtype=paddle.float32
            )
            self.betas = 1 / (scale * num_train_timesteps - steps)
        else:
            raise NotImplementedError(
                f"{beta_schedule} does is not implemented for {self.__class__}"
            )

        self.alphas = paddle.concat(
            [paddle.to_tensor([1.0], dtype=paddle.float32), 1.0 - self.betas],
        )
        self.state = paddle.cumprod(x=self.alphas, dim=0)
        self.state[-1] = 0.0

    def to_discrete_time(self, t, N, T):
        return (t * (N - 1) / T).astype(dtype="int64")

    def get_qt_given_q0(
        self,
        q0,
        t,
        return_logits: bool = False,
        make_one_hot=False,
        epsilon=1e-20,
    ):
        if make_one_hot:
            assert q0.dtype in ["int32", "int64", paddle.int32, paddle.int64]
            q0 = paddle.eye(num_rows=self.dim)[q0]
        assert q0.dtype in ["float32", paddle.float32]
        assert len(tuple(q0.shape)) == 2
        p = self.state[t]
        non_mask_prob = p[:, None] * q0[:, :-1]
        mask_prob = 1 - non_mask_prob.sum(axis=-1)
        prob_at_time_t = (
            mask_prob[:, None] * paddle.eye(num_rows=self.dim)[self.dim - 1][None]
        )
        prob_at_time_t[:, :-1] = non_mask_prob
        prob_at_time_t = paddle.where(condition=t[:, None] == 0, x=q0, y=prob_at_time_t)
        if return_logits:
            return paddle.log(x=prob_at_time_t + epsilon)
        else:
            return prob_at_time_t

    def add_noise(
        self,
        original_samples: paddle.Tensor,
        timesteps: paddle.Tensor,
        batch_idx: paddle.Tensor = None,
    ) -> paddle.Tensor:
        # only support absorbing state
        # Q_t[i][j] = 1,                  when i = j = m
        #           = 1 - β_t,            when i = j ≠ m
        #           = β_t,                when j = m and i ≠ m

        # for example, for atom type, original_samples = [32, 2, 5, 6]
        original_samples = paddle.eye(num_rows=self.dim)[original_samples]

        timesteps = (
            maybe_expand(
                self.to_discrete_time(t=timesteps, N=self.num_train_timesteps, T=1.0),
                batch_idx,
            )
            + 1
        )

        p = self.state[timesteps]
        non_mask_prob = p[:, None] * original_samples[:, :-1]
        mask_prob = 1 - non_mask_prob.sum(axis=-1)
        prob_at_time_t = (
            mask_prob[:, None] * paddle.eye(num_rows=self.dim)[self.dim - 1][None]
        )
        prob_at_time_t[:, :-1] = non_mask_prob
        prob_at_time_t = paddle.where(
            condition=timesteps[:, None] == 0, x=original_samples, y=prob_at_time_t
        )

        logits = paddle.log(x=prob_at_time_t + 1e-20)
        noisy_samples = paddle.distribution.Categorical(logits=logits).sample()

        return noisy_samples

    def qt_reverse(
        self, qt_plus_1, t, return_logits=False, make_one_hot=False, epsilon=1e-20
    ):
        """Get q(x_{t+1} | x_t), for each possible value of x_t. Thus, the rows of the
        output do not sum to 1.

        Args:
          qt_plus_1: an array of floats specifying a distribution over q(x_{t+1} | x_0).
          t: t in q(x_{t+1} | x_t).
          return_logits: if True, return the output logits
          make_one_hot: if True, will convert q(x_{t+1}) to floats if needed.
          epsilon: a small number to normalize logits conversion with, if needed.

        Returns:
          q(x_{t+1} | x_t), shape [num_samples, num_classes].
        """
        if make_one_hot:
            assert qt_plus_1.dtype in ["int64", "int32", paddle.int32, paddle.int64]
            qt_plus_1 = paddle.eye(num_rows=self.dim)[qt_plus_1]
        assert qt_plus_1.dtype in ["float32", paddle.float32]
        beta = self.betas[t]
        non_mask_prob = (1 - beta)[:, None] * qt_plus_1[:, :-1] + beta[
            :, None
        ] * qt_plus_1[:, -1:]
        prob_at_time_t = (
            paddle.eye(num_rows=self.dim)[self.dim - 1][None] * qt_plus_1[:, -1:]
        )
        prob_at_time_t[:, :-1] = non_mask_prob
        if return_logits:
            return paddle.log(x=prob_at_time_t + epsilon)
        else:
            return prob_at_time_t

    def sample_and_compute_posterior_q(
        self,
        x_0,
        t,
        samples=None,
        transition_probs=None,
        return_logits=True,
        return_transition_probs=False,
        transition_probs_in_logits=True,
        make_one_hot=True,
        epsilon=1e-20,
        step_size=1,
    ):

        if make_one_hot:
            assert x_0.dtype in ["int64", "int32", paddle.int32, paddle.int64]
            x_0 = paddle.eye(num_rows=self.dim)[x_0].reshape(
                tuple(x_0.shape) + (self.dim,)
            )
        assert x_0.dtype in ["float32", paddle.float32]
        assert t.dtype in ["int64", "int32", paddle.int32, paddle.int64]
        prob_at_time_t = self.get_qt_given_q0(q0=x_0, t=t)
        prob_at_time_t_plus_one = self.get_qt_given_q0(q0=x_0, t=t + step_size)

        if samples is None and transition_probs is not None:
            raise ValueError("samples were not provided but transition_probs were.")
        if samples is None:
            logits = paddle.log(x=prob_at_time_t_plus_one + epsilon)
            samples = paddle.distribution.Categorical(logits=logits).sample()
        if transition_probs is None:
            if step_size > 1:
                transition_probs = paddle.eye(num_rows=self.dim)[samples]
                for i in range(step_size):
                    transition_probs = self.qt_reverse(
                        qt_plus_1=transition_probs,
                        make_one_hot=False,
                        t=t + step_size - 1 - i,
                    )
            else:
                transition_probs = self.qt_reverse(
                    qt_plus_1=samples, make_one_hot=True, t=t
                )
        if not transition_probs_in_logits and not return_logits:
            raise ValueError(
                "Cannot exclude transition probs from logits if return_logits is false."
            )
        if return_logits:
            posterior_logits = paddle.log(x=prob_at_time_t + epsilon)
            if transition_probs_in_logits:
                posterior_logits += paddle.log(x=transition_probs + epsilon)
            if return_transition_probs:
                return posterior_logits, samples, transition_probs
            else:
                return posterior_logits, samples
        else:
            raise NotImplementedError()
            # posterior = transition_probs * prob_at_time_t
            # denominator = paddle.sum(denominator, axis=-1, keepdim=True)

            # posterior = posterior / denominator
            # if return_transition_probs:
            #     return posterior, samples, transition_probs
            # else:
            #     return posterior, samples

    def p_forward(
        self,
        logits,
        x_t,
        t,
        predict_x0=True,
        return_x0=False,
        return_logits=False,
        special_case_x0=False,
        transition_probs=None,
        transition_probs_in_logits=True,
        maximum_likelihood=False,
        epsilon=1e-20,
        step_size=1,
    ):
        """Returns probabilities from the reverse process p(x_{t-1} | x_t).

        Args:
        logits: the logits for the model's predictions at time t.
        x_t: the current value of x_t to condition on.
        t: the timestep t.
        predict_x0: if True, assumes the model output corresponds to its prediction
            for p(x_0 | x_t). Otherwise assumes model predicts p(x_{t-1} | x_t).
        return_x0: if True, will return probs for x_0 as well as x_{t-1}.
        return_logits: if True, will return logits instead of probabilities.
        special_case_x0: if True, will directly predict x0 instead of using the
            forward process probabilities.
        transition_probs: if provided, q(x_{t+1} | x_t) probs to reuse.
        transition_probs_in_logits: if False, will ignore transition probs in logits
            (only allowed if return_logits is True). This is because this term is
            independent of theta.
        maximum_likelihood: if true, will draw the most likely x0 before applying
            the forward process.
        epsilon: a small number.
        step_size: step size to compute posterior from.

        Returns:
        probabilities for q(x_{t-1} | x_t) (and probabilities for x0 if predict_x0
        is True)
        """
        assert not (step_size > 1 and not predict_x0)

        probs = paddle.nn.functional.softmax(logits, axis=-1)
        if not predict_x0:
            retval = logits if return_logits else probs
            if return_x0:
                return retval, None
            else:
                return retval
        if maximum_likelihood:
            probs = probs.argmax(axis=-1)
        qt_probs, _ = self.sample_and_compute_posterior_q(
            x_0=probs,
            t=t - step_size,
            make_one_hot=maximum_likelihood,
            return_logits=return_logits,
            transition_probs_in_logits=transition_probs_in_logits,
            transition_probs=transition_probs,
            samples=x_t,
            epsilon=epsilon,
            step_size=step_size,
        )
        retval_x0 = logits if return_logits else probs
        retval = qt_probs
        mask = (t == step_size) & paddle.to_tensor(special_case_x0)
        retval = (
            mask[:, None].astype(retval_x0.dtype) * retval_x0
            + mask.logical_not()[:, None].astype(retval.dtype) * retval
        )
        if return_x0:
            return retval, retval_x0
        else:
            return retval

    def compute_kl_reverse_process(
        self,
        x_start,
        t,
        x_t_plus_1,
        logits,
        predict_x0: bool = True,
        log_space: bool = False,
        label_smoothing: float = 0.0,
        hybrid_lambda: float = 0.0,
        use_cached_transition: bool = True,
        target_mask: Optional[paddle.Tensor] = None,
        step_size: int = 1,
    ):
        """
        Computes KL divergence between reverse process and forward process.
        """
        assert x_start.dtype in ["int32", "int64", paddle.int32, paddle.int64]
        if step_size > 1 and not predict_x0:
            raise ValueError("cannot skip steps when not predicting x0.")
        q_t, x_t_plus_1, transition_probs = self.sample_and_compute_posterior_q(
            x_0=x_start,
            t=t,
            return_logits=log_space,
            return_transition_probs=True,
            step_size=step_size,
            samples=x_t_plus_1,
        )

        transition_probs = transition_probs if use_cached_transition else None
        p_t = self.p_forward(
            logits=logits,
            x_t=x_t_plus_1,
            t=t + step_size,
            predict_x0=predict_x0,
            return_x0=predict_x0 and hybrid_lambda > 0.0,
            return_logits=log_space,
            transition_probs=transition_probs,
            step_size=step_size,
        )
        hybrid_loss = paddle.to_tensor(data=0.0, place=x_start.place)
        if predict_x0 and hybrid_lambda > 0.0:
            p_t, p_0 = p_t
            if log_space:
                cross_entropy = paddle.nn.functional.cross_entropy(
                    input=p_0,
                    label=x_start,
                    label_smoothing=label_smoothing,
                    reduction="none",
                )
            else:
                cross_entropy = paddle.nn.functional.cross_entropy(
                    input=(p_0 + 1e-07).log(),
                    label=x_start,
                    label_smoothing=label_smoothing,
                    reduction="none",
                )
            hybrid_loss = hybrid_lambda * cross_entropy
        assert (
            not q_t.isnan().astype("bool").any()
            and not p_t.isnan().astype("bool").any()
        )
        if log_space:
            d1 = paddle.distribution.Categorical(logits=q_t)
            d2 = paddle.distribution.Categorical(logits=p_t)
            kl = paddle.distribution.kl_divergence(p=d1, q=d2)
            cross_entropy = paddle.nn.functional.cross_entropy(
                input=p_t,
                label=x_start,
                label_smoothing=label_smoothing,
                reduction="none",
            )
        else:
            d1 = paddle.distribution.Categorical(logits=(q_t + 1e-07).log())
            d2 = paddle.distribution.Categorical(logits=(p_t + 1e-07).log())
            kl = paddle.distribution.kl_divergence(p=d1, q=d2)
            cross_entropy = paddle.nn.functional.cross_entropy(
                input=(p_t + 1e-07).log(),
                label=x_start,
                label_smoothing=label_smoothing,
                reduction="none",
            )
        if target_mask is not None:
            kl = kl * target_mask
            cross_entropy = cross_entropy * target_mask
            hybrid_loss = hybrid_loss * target_mask
        mask = t == 0
        base_loss = (
            mask.astype(cross_entropy.dtype) * cross_entropy
            + mask.logical_not().astype(kl.dtype) * kl
        )
        loss = base_loss + hybrid_loss
        return loss, base_loss, cross_entropy

    def compute_loss(
        self,
        score_model_output: paddle.Tensor,
        t: paddle.Tensor,
        batch_idx,
        batch_size: int,
        x: paddle.Tensor,
        noisy_x: paddle.Tensor,
        reduce: Literal["sum", "mean"],
        d3pm_hybrid_lambda: float = 0.0,
    ) -> paddle.Tensor:
        t = maybe_expand(self.to_discrete_time(t, N=1000, T=1.0), batch_idx)
        loss, base_loss, cross_entropy = self.compute_kl_reverse_process(
            x.astype(dtype="int64"),
            t,
            logits=score_model_output,
            log_space=True,
            hybrid_lambda=d3pm_hybrid_lambda,
            x_t_plus_1=noisy_x.astype(dtype="int64"),
        )
        loss_per_structure = aggregate_per_sample(
            loss, batch_idx=batch_idx, reduce=reduce, batch_size=batch_size
        )
        base_loss_per_structure = aggregate_per_sample(
            base_loss, batch_idx=batch_idx, reduce=reduce, batch_size=batch_size
        )
        cross_entropy_per_structure = aggregate_per_sample(
            cross_entropy, batch_idx=batch_idx, reduce=reduce, batch_size=batch_size
        )
        return loss_per_structure, base_loss_per_structure, cross_entropy_per_structure

    def prior_sampling(
        self,
        shape: Union[list, Tuple],
    ) -> paddle.Tensor:
        """Generate one sample from the prior distribution, $p_T(x)$."""
        sample = paddle.full(shape=shape, fill_value=self.dim - 1, dtype="int64")
        return sample

    def step(
        self,
        x: paddle.Tensor,
        t: paddle.Tensor,
        batch_idx: paddle.Tensor,
        score: paddle.Tensor,
    ):
        """
        Takes the atom types at time t and returns the atom types at time t-1,
        sampled using the learned reverse atom diffusion model.

        Look at https://github.com/google-research/google-research/blob/master/d3pm/text/diffusion.py
        """
        t = self.to_discrete_time(t=t, N=self.num_train_timesteps, T=1.0)

        class_logits = score
        x_sample = paddle.distribution.Categorical(logits=class_logits).sample()

        class_probs = paddle.nn.functional.softmax(x=class_logits, axis=-1)
        # class_expected = paddle.argmax(x=class_probs, axis=-1)

        class_logits, _ = self.sample_and_compute_posterior_q(
            x_0=class_probs,
            t=t[batch_idx].to("int64"),
            make_one_hot=False,
            samples=x,
            return_logits=True,
        )
        x_sample = paddle.distribution.Categorical(logits=class_logits).sample()

        class_expected = paddle.argmax(
            x=paddle.nn.functional.softmax(
                x=class_logits.to(class_probs.dtype), axis=-1
            ),
            axis=-1,
        )

        return x_sample, class_expected

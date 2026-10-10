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

# Adapted from https://github.com/hspark1212/chemeleon

"""Diffusion schedulers and the D3PM transition machinery for atom types."""

import math

import numpy as np
import paddle

DEFAULT_DTYPE = paddle.get_default_dtype()


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = paddle.linspace(0, timesteps, steps)
    alphas_cumprod = paddle.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return paddle.clip(betas, 0.0001, 0.9999)


def linear_beta_schedule(timesteps, beta_start, beta_end):
    return paddle.linspace(beta_start, beta_end, timesteps)


def quadratic_beta_schedule(timesteps, beta_start, beta_end):
    return paddle.linspace(beta_start**0.5, beta_end**0.5, timesteps) ** 2


def sigmoid_beta_schedule(timesteps, beta_start, beta_end):
    betas = paddle.linspace(-6, 6, timesteps)
    return paddle.sigmoid(betas) * (beta_end - beta_start) + beta_start


def p_wrapped_normal(x, sigma, N=10, T=1.0):
    p_ = 0
    for i in range(-N, N + 1):
        p_ += paddle.exp(-((x + T * i) ** 2) / 2 / sigma**2)
    return p_


def d_log_p_wrapped_normal(x, sigma, N=10, T=1.0):
    p_ = 0
    for i in range(-N, N + 1):
        p_ += (
            (x + T * i) / sigma**2 * paddle.exp(-((x + T * i) ** 2) / 2 / sigma**2)
        )
    return p_ / p_wrapped_normal(x, sigma, N, T)


def sigma_norm(sigma, T=1.0, sn=10000):
    sigmas = sigma[None, :].repeat(sn, 1)
    x_sample = sigma * paddle.randn(sigmas.shape, dtype=sigma.dtype)
    x_sample = x_sample % T
    normal_ = d_log_p_wrapped_normal(x_sample, sigmas, T=T)
    return (normal_**2).mean(axis=0)


class BetaScheduler(paddle.nn.Layer):
    def __init__(self, timesteps, scheduler_mode, beta_start=0.0001, beta_end=0.02):
        super(BetaScheduler, self).__init__()
        self.timesteps = timesteps
        if scheduler_mode == "cosine":
            betas = cosine_beta_schedule(timesteps)
        elif scheduler_mode == "linear":
            betas = linear_beta_schedule(timesteps, beta_start, beta_end)
        elif scheduler_mode == "quadratic":
            betas = quadratic_beta_schedule(timesteps, beta_start, beta_end)
        elif scheduler_mode == "sigmoid":
            betas = sigmoid_beta_schedule(timesteps, beta_start, beta_end)
        else:
            raise ValueError(f"Invalid scheduler mode: {scheduler_mode}")

        betas = paddle.concat([paddle.zeros([1]), betas], axis=0)
        alphas = 1.0 - betas
        alphas_cumprod = paddle.cumprod(alphas, dim=0)

        # posterior mean
        posterior_mean_coeff1 = paddle.ones_like(betas)
        posterior_mean_coeff1[1:] = (
            betas[1:] * paddle.sqrt(alphas_cumprod[:-1]) / (1.0 - alphas_cumprod[1:])
        )  # coeff for x0 = beta_t * sqrt(alpha_cumprod(t-1)) / (1 - alpha_cumprod(t))
        posterior_mean_coeff2 = paddle.zeros_like(betas)
        posterior_mean_coeff2[1:] = (
            (1.0 - alphas_cumprod[:-1])
            * paddle.sqrt(alphas[1:])
            / (1.0 - alphas_cumprod[1:])
        )
        # coeff for xt = (1 - alpha_cumprod(t-1)) * sqrt(alpha(t))
        # / (1 - alpha_cumprod(t))
        # posterior variance
        sigmas = paddle.zeros_like(betas)
        sigmas[1:] = (
            betas[1:] * (1.0 - alphas_cumprod[:-1]) / (1.0 - alphas_cumprod[1:])
        )  # sigma_t = beta_t * (1 - alpha_cumprod(t-1)) / (1 - alpha_cumprod(t))
        sigmas = paddle.sqrt(sigmas)

        def register_buffer(name, val):
            return self.register_buffer(name, val.astype(DEFAULT_DTYPE))

        register_buffer("betas", betas)
        register_buffer("alphas", alphas)
        register_buffer("alphas_cumprod", alphas_cumprod)
        register_buffer("posterior_mean_coeff1", posterior_mean_coeff1)
        register_buffer("posterior_mean_coeff2", posterior_mean_coeff2)
        register_buffer("sigmas", sigmas)

    def uniform_sample_t(self, batch_size, device=None):
        # numpy RNG on purpose: seeded runs consume a fixed draw sequence
        ts = np.random.choice(np.arange(1, self.timesteps + 1), batch_size)
        t = paddle.to_tensor(ts)
        if device is not None:
            t = t.to(device)
        return t


class SigmaScheduler(paddle.nn.Layer):
    def __init__(self, timesteps, sigma_begin=0.01, sigma_end=1.0):
        super(SigmaScheduler, self).__init__()
        self.timesteps = timesteps
        self.sigma_begin = sigma_begin
        self.sigma_end = sigma_end
        sigmas = paddle.to_tensor(
            np.exp(np.linspace(np.log(sigma_begin), np.log(sigma_end), timesteps)),
            dtype="float32",
        )

        sigmas_norm_ = sigma_norm(sigmas)

        def register_buffer(name, val):
            return self.register_buffer(name, val.astype(DEFAULT_DTYPE))

        # the [0] pad scalars take the table dtype so the concatenated
        # buffers stay single-dtype under any default dtype; the tables
        # themselves are fp32-computed
        register_buffer(
            "sigmas",
            paddle.concat([paddle.zeros([1], dtype=sigmas.dtype), sigmas], axis=0),
        )
        register_buffer(
            "sigmas_norm",
            paddle.concat(
                [paddle.ones([1], dtype=sigmas_norm_.dtype), sigmas_norm_], axis=0
            ),
        )

    def uniform_sample_t(self, batch_size, device=None):
        ts = np.random.choice(np.arange(1, self.timesteps + 1), batch_size)
        t = paddle.to_tensor(ts)
        if device is not None:
            t = t.to(device)
        return t


def prob_mask_like(shape: tuple, prob: float, device=None) -> paddle.Tensor:
    """
    For classifier free guidance. Creates a boolean mask for given input
    shape and probability of `True`.

    :param shape: Shape of mask.
    :param prob: Probability of True. In interval [0., 1.].
    :param device: Optional device string; when omitted the mask follows the
        current device context.
    :return: The mask.
    """
    if prob == 1:
        return paddle.ones(shape, dtype=paddle.bool)
    elif prob == 0:
        return paddle.zeros(shape, dtype=paddle.bool)
    else:
        return paddle.zeros(shape).uniform_(0, 1) < prob


# D3PM for atom types
class D3PM(paddle.nn.Layer):
    """Discrete absorbing-state diffusion over atom types.

    Buffers:
      - q_one_step_mats [T+1, V, V]: one-step transition matrices Q_t
      - q_mats [T+1, V, V]: cumulative products Q_1 ... Q_t
      - q_one_step_transposed [T+1, V, V]: transposed one-step matrices,
        registered as a buffer so it moves with the layer across devices
    """

    def __init__(
        self,
        beta_scheduler: paddle.nn.Layer,
        num_timesteps: int,
        max_atoms: int,
        d3pm_hybrid_coeff: float,
    ):
        super().__init__()
        self.beta_scheduler = beta_scheduler
        self.num_timesteps = num_timesteps
        self.max_atoms = max_atoms
        self.hybrid_coeff = d3pm_hybrid_coeff
        self.eps = 1.0e-6

        # transition matrix for absorbing
        q_one_step_mats = paddle.stack(
            [
                self.get_absorbing_transition_mat(t)
                for t in range(0, self.num_timesteps + 1)
            ],
            axis=0,
        )
        self.register_buffer("q_one_step_mats", q_one_step_mats)

        # construct transition matrices for q(x_t | x_0)
        q_mat_t = self.q_one_step_mats[0]
        q_mats = [q_mat_t]
        for t in range(1, self.num_timesteps + 1):
            # Q_{1...t} = Q_{1 ... t-1} Q_t = Q_1 Q_2 ... Q_t
            q_mat_t = q_mat_t @ self.q_one_step_mats[t]
            q_mats.append(q_mat_t)
        q_mats = paddle.stack(q_mats, axis=0)
        self.register_buffer("q_mats", q_mats)

        assert tuple(self.q_mats.shape) == (
            self.num_timesteps + 1,
            self.max_atoms,
            self.max_atoms,
        )

        # registered as a buffer so it moves with the layer across devices
        self.register_buffer(
            "q_one_step_transposed", self.q_one_step_mats.transpose([0, 2, 1])
        )

    def get_absorbing_transition_mat(self, t: int):
        """Computes transition matrix for q(x_t|x_{t-1}).

        Args:
            t (int): timestep.
            max_atoms (int): maximum number of atoms (103 + 1 for dummy atom).
                Defaults to 104.

        Returns:
            Q_t: transition matrix. shape = (max_atoms, max_atoms)
        """
        # get beta at timestep t
        beta_t = self.beta_scheduler.betas[t]

        diag = paddle.full((self.max_atoms,), 1 - beta_t)
        mat = paddle.diag(diag, 0)
        # add beta_t at first row
        mat[:, 0] += beta_t
        return mat

    def at(
        self,
        a: paddle.Tensor,
        t_per_node: paddle.Tensor,
        x: paddle.Tensor,
    ):
        """Extract coefficients at specified timesteps t - 1 and conditioning data x.

        Args:
            a (paddle.Tensor): matrix of coefficients.
                [num_timesteps, max_atoms, max_atoms]
            t_per_node (paddle.Tensor): timesteps.[B_n]
            x (paddle.Tensor): atom_types. [B_n]

        Returns:
            a[t, x] (paddle.Tensor): coefficients at timesteps t and data x.
                [B_n, max_atoms]
        """
        a = a.to(x.place)
        bs = t_per_node.shape[0]
        t_per_node = t_per_node.reshape((bs, *([1] * (x.ndim - 1))))
        return a[t_per_node - 1, x, :]

    def q_sample(
        self,
        x_0: paddle.Tensor,
        t_per_node: paddle.Tensor,
        noise: paddle.Tensor,
    ):
        """Sample from q(x_t | x_0) (i.e. add noise to the data).
        q(x_t | x_0) = Categorical(x_t ; p = x_0 Q_{1...t})

        Args:
            x_0 (paddle.Tensor): clean atom types. [B_n]
            t_per_node (paddle.Tensor): Timesteps. [B_n]
            noise (paddle.Tensor): Noise. [B_n, max_atoms]

        Returns:
            paddle.Tensor: [B_n]
        """
        logits = paddle.log(self.at(self.q_mats, t_per_node, x_0) + self.eps)
        noise = paddle.clip(noise, self.eps, 1.0)
        gumbel_noise = -paddle.log(-paddle.log(noise))
        return paddle.argmax(logits + gumbel_noise, axis=-1)

    def q_posterior_logits(
        self,
        x_0: paddle.Tensor,
        x_t: paddle.Tensor,
        t_per_node: paddle.Tensor,
        is_x_0_one_hot: bool = False,
    ):
        """Compute logits for q(x_{t-1} | x_t, x_0)."""
        if is_x_0_one_hot:
            x_0_logits = x_0.clone()
        else:
            # one_hot returns fp32 regardless of the default dtype; cast to
            # the q_mats buffer dtype so the posterior keeps the buffer
            # precision. No-op under fp32-default builds.
            x_0_logits = paddle.log(
                paddle.nn.functional.one_hot(x_0, self.max_atoms).astype(
                    self.q_mats.dtype
                )
                + self.eps
            )

        assert tuple(x_0_logits.shape) == tuple(x_t.shape) + (
            self.max_atoms,
        ), f"x_0_logits.shape: {x_0_logits.shape}, x_t.shape: {x_t.shape}"

        fact1 = self.at(self.q_one_step_transposed, t_per_node, x_t)

        softmaxed = paddle.nn.functional.softmax(x_0_logits, axis=-1)
        # t - 2 is -1 when t == 1; the wrapped page would be discarded by the
        # where() below, so clamp to a valid non-negative page explicitly
        t_minus_2 = paddle.clip(t_per_node - 2, min=0)
        qmats2 = self.q_mats[t_minus_2]
        fact2 = paddle.einsum("b...c, bcd -> b...d", softmaxed, qmats2)

        out = paddle.log(fact1 + self.eps) + paddle.log(fact2 + self.eps)

        t_broadcast = t_per_node.reshape((t_per_node.shape[0], *([1] * x_t.ndim)))
        return paddle.where(t_broadcast == 1, x_0_logits, out)

    def categorical_kl_logits(self, logits1, logits2, eps=1.0e-6):
        """KL divergence between categorical distributions.

        Distributions parameterized by logits.

        Args:
            logits1: logits of the first distribution. Last dim is class dim.
            logits2: logits of the second distribution. Last dim is class dim.
            eps: float small number to avoid numerical issues.

        Returns:
            KL(C(logits1) || C(logits2)): shape: logits1.shape[:-1]
        """
        out = paddle.nn.functional.softmax(logits1 + eps, axis=-1) * (
            paddle.nn.functional.log_softmax(logits1 + eps, axis=-1)
            - paddle.nn.functional.log_softmax(logits2 + eps, axis=-1)
        )
        return out.sum(axis=-1).mean()

    def p_logits(
        self,
        pred_x_start_logits: paddle.Tensor,
        x_t_atom_types: paddle.Tensor,
        t_per_node: paddle.Tensor,
        noise: paddle.Tensor,
    ):
        pred_q_posterior_logits = self.q_posterior_logits(
            pred_x_start_logits, x_t_atom_types, t_per_node, is_x_0_one_hot=True
        )

        noise = paddle.clip(noise, min=self.eps, max=1.0)
        # if t == 1, use x_0_logits
        # the mask is cast to the noise dtype directly (a floating mask is
        # required for the multiply)
        nonzero_mask = (
            (t_per_node != 1)
            .astype(noise.dtype)
            .reshape([-1] + [1] * x_t_atom_types.ndim)
        )
        gumbel_noise = -paddle.log(-paddle.log(noise))
        sample = paddle.argmax(
            pred_q_posterior_logits + gumbel_noise * nonzero_mask, axis=-1
        )
        return sample

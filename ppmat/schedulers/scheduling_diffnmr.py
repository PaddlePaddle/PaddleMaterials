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

import math

import numpy as np
import paddle
import paddle.nn.functional as F

from ppmat.models.diffnmr.utils import diffgraphformer_utils


def sum_except_batch(x):
    x_reshaped = paddle.reshape(x, [x.shape[0], -1])
    return paddle.sum(x_reshaped, axis=-1)


def assert_correctly_masked(variable, node_mask):
    mask_int = node_mask.astype("int64")
    masked = variable * (1 - mask_int).astype(variable.dtype)
    if paddle.max(paddle.abs(masked)).item() >= 1e-4:
        raise ValueError("Variables not masked properly.")


def sample_gaussian(size):
    return paddle.randn(shape=size)


def sample_gaussian_with_mask(size, node_mask):
    x = paddle.randn(shape=size)
    x = x.astype(node_mask.dtype)
    x_masked = x * node_mask
    return x_masked


def clip_noise_schedule(alphas2, clip_value=0.001):
    """
    For a noise schedule given by alpha^2, this clips alpha_t / alpha_t-1.
    This may help improve stability during sampling.
    """
    alphas2 = np.concatenate([np.ones(1), alphas2], axis=0)
    alphas_step = alphas2[1:] / alphas2[:-1]

    alphas_step = np.clip(alphas_step, a_min=clip_value, a_max=1.0)
    alphas2 = np.cumprod(alphas_step, axis=0)

    return alphas2


def cosine_beta_schedule(timesteps, s=0.008, raise_to_power: float = 1):
    """
    Cosine schedule as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 2
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = np.clip(betas, a_min=0, a_max=0.999)
    alphas = 1.0 - betas
    alphas_cumprod = np.cumprod(alphas, axis=0)

    if raise_to_power != 1:
        alphas_cumprod = np.power(alphas_cumprod, raise_to_power)

    return alphas_cumprod


def cosine_beta_schedule_discrete(timesteps, s=0.008):
    """Cosine schedule as proposed in https://openreview.net/forum?id=-NEXDKk8gZ."""
    steps = timesteps + 2
    x = np.linspace(0, steps, steps)

    alphas_cumprod = np.cos(0.5 * np.pi * ((x / steps) + s) / (1 + s)) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    alphas = alphas_cumprod[1:] / alphas_cumprod[:-1]
    betas = 1 - alphas
    return betas.squeeze()


def custom_beta_schedule_discrete(timesteps, average_num_nodes=50, s=0.008):
    """
    Cosine schedule with modifications for a discrete setting.
    """
    steps = timesteps + 2
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(0.5 * np.pi * ((x / steps) + s) / (1 + s)) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    alphas = alphas_cumprod[1:] / alphas_cumprod[:-1]
    betas = 1 - alphas

    assert timesteps >= 100

    p = 4 / 5  # 1 - 1 / num_edge_classes
    num_edges = average_num_nodes * (average_num_nodes - 1) / 2

    # First 100 steps: only a few updates per graph
    updates_per_graph = 1.2
    beta_first = updates_per_graph / (p * num_edges)

    betas[betas < beta_first] = beta_first
    return np.array(betas)


def gaussian_KL(q_mu, q_sigma):
    """
    KL divergence between a normal distribution (q) and
    the standard normal distribution.
    """
    # 原: sum_except_batch((torch.log(1 / q_sigma) + 0.5*(q_sigma**2 + q_mu**2) - 0.5))
    inside = paddle.log(1.0 / q_sigma) + 0.5 * (q_sigma**2 + q_mu**2) - 0.5
    return sum_except_batch(inside)


def cdf_std_gaussian(x):
    return 0.5 * (1.0 + paddle.erf(x / math.sqrt(2)))


def SNR(gamma):
    """Computes signal to noise ratio (alpha^2/sigma^2) given gamma."""
    return paddle.exp(-gamma)


def inflate_batch_array(array, target_shape):
    """
    Inflates the batch array (array) with only a single axis (batch_size, ...)
    to match the target shape.
    """
    shape0 = array.shape[0]
    new_shape = [shape0] + [1] * (len(target_shape) - 1)
    return paddle.reshape(array, new_shape)


def sigma(gamma, target_shape):
    """Computes sigma given gamma."""
    sig = paddle.sqrt(F.sigmoid(gamma))
    return inflate_batch_array(sig, target_shape)


def alpha(gamma, target_shape):
    """Computes alpha given gamma."""
    sig = paddle.sqrt(F.sigmoid(-gamma))
    return inflate_batch_array(sig, target_shape)


def check_mask_correct(variables, node_mask):
    for var in variables:
        if var.numel() > 0:  # 说明张量非空
            assert_correctly_masked(var, node_mask)


def check_tensor_same_size(*args):
    for i, arg in enumerate(args):
        if i == 0:
            continue
        if args[0].shape != arg.shape:
            raise ValueError("Tensors have different shapes.")


def sigma_and_alpha_t_given_s(
    gamma_t: paddle.Tensor, gamma_s: paddle.Tensor, target_size: paddle.shape
):
    """
    Computes sigma_t_given_s and alpha_t_given_s for sampling.
    """
    part = paddle.softplus(gamma_s) - paddle.softplus(gamma_t)
    sigma2_t = -paddle.expm1(part)
    sigma2_t_given_s = inflate_batch_array(sigma2_t, target_size)

    log_alpha2_t = F.logsigmoid(-gamma_t)
    log_alpha2_s = F.logsigmoid(-gamma_s)
    log_alpha2_t_given_s = log_alpha2_t - log_alpha2_s

    alpha_t_given_s_ = paddle.exp(0.5 * log_alpha2_t_given_s)
    alpha_t_given_s_ = inflate_batch_array(alpha_t_given_s_, target_size)

    sigma_t_given_s = paddle.sqrt(sigma2_t_given_s)

    return sigma2_t_given_s, sigma_t_given_s, alpha_t_given_s_


def reverse_tensor(x):
    idx = paddle.arange(x.shape[0] - 1, -1, -1, dtype="int64")
    return paddle.index_select(x, index=idx, axis=0)


def sample_feature_noise(X_size, E_size, y_size, node_mask):
    """
    Standard normal noise for all features. Output size: X.size(), E.size(), y.size().
    """
    epsX = sample_gaussian(X_size)
    epsE = sample_gaussian(E_size)
    epsy = sample_gaussian(y_size)

    float_mask = node_mask.astype("float32")
    epsX = epsX.astype(float_mask.dtype)
    epsE = epsE.astype(float_mask.dtype)
    epsy = epsy.astype(float_mask.dtype)

    # Get upper triangular part of edge noise, without main diagonal
    upper_triangular_mask = paddle.zeros_like(epsE)

    row_idx, col_idx = np.triu_indices(n=epsE.shape[1], k=1)
    row_idx_t = paddle.to_tensor(row_idx, dtype="int64")
    col_idx_t = paddle.to_tensor(col_idx, dtype="int64")

    for b in range(epsE.shape[0]):
        upper_triangular_mask[b, row_idx_t, col_idx_t, :] = 1.0

    epsE = epsE * upper_triangular_mask
    epsE_T = paddle.transpose(epsE, perm=[0, 2, 1, 3])
    epsE = epsE + epsE_T

    # assert (epsE == torch.transpose(epsE, 1, 2)).all()
    # Paddle :
    eq_ = paddle.all(epsE == epsE_T)
    if not eq_.item():
        raise ValueError("epsE is not symmetric!")

    return diffgraphformer_utils.PlaceHolder(X=epsX, E=epsE, y=epsy).mask(node_mask)


def sample_normal(mu_X, mu_E, mu_y, sigma_, node_mask):
    """
    Samples from a Normal distribution.
    """
    eps = sample_feature_noise(mu_X.shape, mu_E.shape, mu_y.shape, node_mask)
    eps = eps.astype(mu_X.dtype)  # 如果需要与 mu_X 同 dtype

    X = mu_X + sigma_ * eps.X
    E = mu_E + paddle.unsqueeze(sigma_, 1) * eps.E
    y = mu_y + paddle.squeeze(sigma_, axis=1) * eps.y
    return diffgraphformer_utils.PlaceHolder(X=X, E=E, y=y)


def check_issues_norm_values(gamma_func, norm_val1, norm_val2, num_stdevs=8):
    """
    Check if 1 / norm_value is still larger than 10 * standard deviation.
    """
    zeros = paddle.zeros([1, 1], dtype="float32")
    gamma_0 = gamma_func(zeros)
    # sigma_0:
    sig0 = sigma(gamma_0, zeros.shape).item()
    max_norm_value = max(norm_val1, norm_val2)
    if sig0 * num_stdevs > 1.0 / max_norm_value:
        raise ValueError(
            f"Value for normalization {max_norm_value} too large "
            f"with sigma_0={sig0:.5f}."
        )


def sample_discrete_features(probX, probE, node_mask):
    """Sample features from multinomial distribution with given probabilities
        (probX, probE).

    Args:
        probX: node features with shape (bs, n, dx_out)
        probE: edge features with shape (bs, n, n, de_out)
        node_mask: node mask
    """
    bs, n, _ = probX.shape

    # Noise X
    # The masked rows should define probability distributions as well
    probX[~node_mask] = 1 / probX.shape[-1]
    # Flatten the probability tensor to sample with multinomial
    probX = probX.reshape([bs * n, -1])  # (bs * n, dx_out)
    # Sample X
    X_t = paddle.multinomial(probX, num_samples=1).reshape([bs, n])  # (bs, n)

    # Noise E
    # The masked rows should define probability distributions as well
    inverse_edge_mask = ~(node_mask.unsqueeze(1) * node_mask.unsqueeze(2)).unsqueeze(-1)
    diag_mask = paddle.eye(n).unsqueeze(0).expand([bs, -1, -1]).unsqueeze(-1)
    probE = paddle.where(
        inverse_edge_mask, paddle.full_like(probE, 1 / probE.shape[-1]), probE
    )
    probE = paddle.where(
        diag_mask.astype(paddle.bool),
        paddle.full_like(probE, 1 / probE.shape[-1]),
        probE,
    )
    probE = probE.reshape([bs * n * n, -1])  # (bs * n * n, de_out)
    # Sample E
    E_t = paddle.multinomial(probE, num_samples=1).reshape([bs, n, n])  # (bs, n, n)
    E_t = paddle.triu(E_t, diagonal=1)
    E_t = E_t + paddle.transpose(E_t, [0, 2, 1])

    # Create a placeholder for y, since it's not used in this function
    y = paddle.zeros([bs, 0], dtype=X_t.dtype)

    return diffgraphformer_utils.PlaceHolder(X=X_t, E=E_t, y=y)


def compute_posterior_distribution(M, M_t, Qt_M, Qsb_M, Qtb_M):
    """
    M, M_t: shape (bs, N, d) or (bs, N) and flattened
    compute xt @ Qt.T * x0 @ Qsb / x0 @ Qtb @ xt.T
    """
    # Flatten
    bs = M.shape[0]
    M_flat = paddle.reshape(M, [bs, -1, M.shape[-1]])  # e.g. (bs, N, d)
    M_t_flat = paddle.reshape(M_t, [bs, -1, M_t.shape[-1]]).astype("float32")

    Qt_M_T = paddle.transpose(Qt_M, perm=[0, 2, 1])  # (bs, d, d)

    left_term = paddle.matmul(M_t_flat, Qt_M_T)  # (bs, N, d)
    right_term = paddle.matmul(M_flat, Qsb_M)  # (bs, N, d)
    product = left_term * right_term  # (bs, N, d)

    denom = paddle.matmul(M_flat, Qtb_M)  # (bs, N, d)
    denom = paddle.sum(denom * M_t_flat, axis=-1)  # (bs, N)

    denom_ = paddle.unsqueeze(denom, axis=-1)  # (bs, N, 1)
    # avoid zero div
    zero_mask = denom_ == 0.0
    denom_ = paddle.where(zero_mask, paddle.ones_like(denom_), denom_)

    prob = product / denom_
    return prob


def compute_batched_over0_posterior_distribution_(X_t, Qt, Qsb, Qtb):
    """
    Compute xt @ Qt.T * x0 @ Qsb / x0 @ Qtb @ xt.T for each possible value of x0
    """
    X_t = X_t.astype("float32")
    Qt_T = paddle.transpose(Qt, perm=[0, 2, 1]).astype("float32")

    left_term = paddle.matmul(X_t, Qt_T)  # (bs, N, d_t-1)
    left_term = paddle.unsqueeze(left_term, axis=2)  # (bs, N, 1, d_t-1)

    right_term = paddle.unsqueeze(Qsb, axis=1)  # (bs, 1, d0, d_t-1)
    numerator = left_term * right_term  # (bs, N, d0, d_t-1)

    denominator = paddle.matmul(
        Qtb, paddle.transpose(X_t, perm=[0, 2, 1])
    )  # (bs, d0, N)
    denominator = paddle.transpose(denominator, perm=[0, 2, 1])  # (bs, N, d0)
    denominator = paddle.unsqueeze(denominator, axis=-1)  # (bs, N, d0, 1)

    zero_mask = denominator == 0.0
    denominator = paddle.where(zero_mask, paddle.ones_like(denominator), denominator)

    return numerator / denominator


def compute_batched_over0_posterior_distribution(X_t, Qt, Qsb, Qtb):
    """
    Flatten edge features to (bs, N, dt).
    Then compute the posterior distribution. (Same logic as the '_' version)
    """
    # Flatten
    X_t_f = X_t.flatten(start_axis=1, stop_axis=-2).astype("float32")  # (bs, N, dt)

    Qt_T = paddle.transpose(Qt, perm=[0, 2, 1])  # (bs, dt, d_t-1)
    left_term = paddle.matmul(X_t_f, Qt_T)  # (bs, N, d_t-1)
    left_term = paddle.unsqueeze(left_term, axis=2)  # (bs, N, 1, d_t-1)
    right_term = paddle.unsqueeze(Qsb, axis=1)  # (bs, 1, d0, d_t-1)
    numerator = left_term * right_term  # (bs, N, d0, d_t-1)

    X_t_transposed = paddle.transpose(X_t_f, perm=[0, 2, 1])  # (bs, dt, N)
    prod = paddle.matmul(Qtb, X_t_transposed)  # (bs, d0, N)
    prod = paddle.transpose(prod, perm=[0, 2, 1])  # (bs, N, d0)

    denominator = paddle.unsqueeze(prod, axis=-1)  # (bs, N, d0, 1)
    zero_mask = denominator == 0
    denominator = paddle.where(
        zero_mask, paddle.full_like(denominator, 1e-6), denominator
    )

    return numerator / denominator


def mask_distributions(true_X, true_E, pred_X, pred_E, node_mask):
    """
    Set masked rows to arbitrary distributions, so they don't contribute to loss.
    Then renormalize.
    """
    dtype_ = true_X.dtype

    row_X = paddle.zeros([true_X.shape[-1]], dtype=dtype_)
    row_X[0] = 1.0
    row_E = paddle.zeros([true_E.shape[-1]], dtype=dtype_)
    row_E[0] = 1.0

    n_ = node_mask.shape[1]
    diag_mask = paddle.eye(n_, dtype="int32").astype("bool")
    diag_mask_bs = diag_mask.unsqueeze(0).expand([node_mask.shape[0], n_, n_])

    mask_bool = node_mask.astype("bool")
    mask_not = paddle.logical_not(mask_bool)

    row_X_bc = row_X.unsqueeze(0).unsqueeze(0)  # shape (1,1,dx)
    row_X_bc = paddle.expand(
        row_X_bc, [mask_not.shape[0], mask_not.shape[1], row_X.shape[0]]
    )

    true_X = paddle.where(paddle.unsqueeze(mask_not, axis=-1), row_X_bc, true_X)
    pred_X = paddle.where(paddle.unsqueeze(mask_not, axis=-1), row_X_bc, pred_X)

    # Edge
    mask_2d = paddle.unsqueeze(mask_bool, axis=1) & paddle.unsqueeze(mask_bool, axis=2)
    inv_mask_2d = paddle.logical_not(mask_2d)
    # + diag => unify
    comb_mask = paddle.logical_or(inv_mask_2d, diag_mask_bs)

    row_E_bc = row_E.unsqueeze(0).unsqueeze(0).unsqueeze(0)  # shape (1,1,1,de)
    row_E_bc = paddle.expand(
        row_E_bc,
        [comb_mask.shape[0], comb_mask.shape[1], comb_mask.shape[2], row_E.shape[0]],
    )

    true_E = paddle.where(paddle.unsqueeze(comb_mask, axis=-1), row_E_bc, true_E)
    pred_E = paddle.where(paddle.unsqueeze(comb_mask, axis=-1), row_E_bc, pred_E)

    # + 1e-7
    eps_ = 1e-7
    true_X = true_X + eps_
    pred_X = pred_X + eps_
    true_E = true_E + eps_
    pred_E = pred_E + eps_

    # normalize
    sum_true_X = paddle.sum(true_X, axis=-1, keepdim=True)
    sum_pred_X = paddle.sum(pred_X, axis=-1, keepdim=True)
    sum_true_E = paddle.sum(true_E, axis=-1, keepdim=True)
    sum_pred_E = paddle.sum(pred_E, axis=-1, keepdim=True)

    true_X = true_X / sum_true_X
    pred_X = pred_X / sum_pred_X
    true_E = true_E / sum_true_E
    pred_E = pred_E / sum_pred_E

    return true_X, true_E, pred_X, pred_E


def posterior_distributions(X, E, y, X_t, E_t, y_t, Qt, Qsb, Qtb):
    """
    Compute posterior distribution for X, E.
    """
    prob_X = compute_posterior_distribution(
        M=X, M_t=X_t, Qt_M=Qt.X, Qsb_M=Qsb.X, Qtb_M=Qtb.X
    )  # (bs, n, dx)
    prob_E = compute_posterior_distribution(
        M=E, M_t=E_t, Qt_M=Qt.E, Qsb_M=Qsb.E, Qtb_M=Qtb.E
    )  # (bs, n*n, de)
    return diffgraphformer_utils.PlaceHolder(X=prob_X, E=prob_E, y=y_t)


def sample_discrete_feature_noise(limit_dist, node_mask):
    """
    Sample from the limit distribution of the diffusion process
    (multinomial with prob = limit_dist).
    """
    bs, n_max = node_mask.shape
    # x_limit => shape (bs, n_max, dx)
    x_limit = paddle.unsqueeze(limit_dist.X, axis=0)  # (1, dx)
    x_limit = paddle.unsqueeze(x_limit, axis=0)  # (1,1,dx)
    x_limit = paddle.expand(x_limit, [bs, n_max, x_limit.shape[-1]])  # (bs, n_max, dx)

    e_limit = paddle.unsqueeze(limit_dist.E, axis=0)  # (1, de)
    e_limit = paddle.unsqueeze(e_limit, axis=0)  # (1,1,de)
    e_limit = paddle.unsqueeze(e_limit, axis=0)  # (1,1,1,de)
    e_limit = paddle.expand(
        e_limit, [bs, n_max, n_max, e_limit.shape[-1]]
    )  # (bs, n_max, n_max, de)

    y_limit = paddle.unsqueeze(limit_dist.y, axis=0)  # (1, dy)
    if y_limit.shape[-1] == 0:
        y_limit = paddle.zeros([bs, 0], dtype=y_limit.dtype)
    else:
        y_limit = paddle.expand(y_limit, [bs, y_limit.shape[-1]])  # (bs, dy)

    # multinomial for X
    # flatten => (bs*n_max, dx)
    X_probs_flat = paddle.reshape(x_limit, [bs * n_max, -1])
    X_idx = paddle.multinomial(X_probs_flat, num_samples=1)
    X_idx = paddle.reshape(X_idx, [bs, n_max])  # (bs, n_max)

    # multinomial for E
    E_probs_flat = paddle.reshape(e_limit, [bs * n_max * n_max, -1])
    E_idx = paddle.multinomial(E_probs_flat, num_samples=1)
    E_idx = paddle.reshape(E_idx, [bs, n_max, n_max])

    U_y = paddle.zeros([bs, 0], dtype=X_idx.dtype)

    # one_hot
    X_onehot = F.one_hot(X_idx, num_classes=x_limit.shape[-1]).astype("float32")
    E_onehot = F.one_hot(E_idx, num_classes=e_limit.shape[-1]).astype("float32")

    # Get upper triangular part for E
    row_idx, col_idx = np.triu_indices(n=n_max, k=1)
    row_idx_t = paddle.to_tensor(row_idx, dtype="int64")
    col_idx_t = paddle.to_tensor(col_idx, dtype="int64")

    E_upper = paddle.zeros_like(E_onehot)
    for b in range(bs):
        E_upper[b, row_idx_t, col_idx_t] = E_onehot[b, row_idx_t, col_idx_t]

    E_sym = E_upper + paddle.transpose(E_upper, perm=[0, 2, 1, 3])
    # check symmetry
    eq_ = paddle.all(E_sym == paddle.transpose(E_sym, perm=[0, 2, 1, 3]))
    if not eq_.item():
        raise ValueError("Discrete feature noise E is not symmetric!")

    ph = diffgraphformer_utils.PlaceHolder(X=X_onehot, E=E_sym, y=U_y)
    return ph.mask(node_mask)


def _encode_spectrum_condition(model, condition):
    """Encode the four-branch NMR condition through the declared interface."""

    if model.flag_onlyH:
        embedding, _ = model.run_encoder(condition)
        return embedding, None

    embedding, (token_encoding, _) = model.run_encoder(condition)
    return embedding, token_encoding


@paddle.no_grad()
def step(
    model, s, t, X_t, E_t, y_t, node_mask, conditionVec, batch_X, batch_E, batch_y
):
    """
    sample from p(z_s | z_t) : take one step of reverse diffusion
    """
    beta_t = model.noise_schedule(t_normalized=t)
    alpha_s_bar = model.noise_schedule.get_alpha_bar(t_normalized=s)
    alpha_t_bar = model.noise_schedule.get_alpha_bar(t_normalized=t)

    # retrieve transitions matrix
    Qtb = model.transition_model.get_Qt_bar(alpha_t_bar)
    Qsb = model.transition_model.get_Qt_bar(alpha_s_bar)
    Qt = model.transition_model.get_Qt(beta_t)

    # prepare neural net input
    noisy_data = {
        "X_t": X_t,
        "E_t": E_t,
        "y_t": y_t,
        "t": t,
        "node_mask": node_mask,
    }
    extra_data = compute_extra_data(model, noisy_data)

    # input_X for decoder
    input_X = paddle.concat(
        [noisy_data["X_t"].astype("float32"), extra_data.X.astype(dtype="float32")],
        axis=2,
    )

    # input_E for decoder
    input_E = paddle.concat(
        [noisy_data["E_t"].astype("float32"), extra_data.E.astype(dtype="float32")],
        axis=3,
    )

    # partial input_y for decoder
    input_y = paddle.hstack(
        [noisy_data["y_t"].astype("float32"), extra_data.y.astype(dtype="float32")]
    )

    if getattr(model, "conditioning_mode", None) == "spectrum":
        embeddings_spectrum, spectrum_encoding = _encode_spectrum_condition(
            model, conditionVec
        )
        if model.connector_flag is True:
            # DiffPrior treats a missing token sequence as an empty optional
            # conditioning branch, which is the declared behavior for H-only
            # encoders.  Keep the connector batch aligned with the graph batch:
            # its public default generates multiple candidates, while a
            # reverse-diffusion step needs exactly one embedding per spectrum.
            embeddings_spectrum = model.connector.sample(
                embeddings_spectrum,
                spectrum_encodings=spectrum_encoding,
                num_samples_per_batch=1,
            )
        input_y = paddle.concat([input_y, embeddings_spectrum], axis=1).astype(
            "float32"
        )

        # 4. Decoder forward
        # Convention: pred.X and pred.E are logits with shapes [B, n, Cx] and
        # [B, n, n, Ce]
        pred = model.run_decoder(input_X, input_E, input_y, node_mask)
    else:
        # prepare the extra feature for encoder input without noisy
        batch_values = (
            diffgraphformer_utils.PlaceHolder(X=batch_X, E=batch_E, y=batch_y)
            .type_as(batch_X)
            .mask(node_mask)
        )
        extra_data_pure = compute_extra_data(
            model,
            {
                "X_t": batch_values.X,
                "E_t": batch_values.E,
                "y_t": batch_values.y,
                "node_mask": node_mask,
            },
            isPure=True,
        )
        # prepare the input data for encoder combining extra features
        input_X_pure = paddle.concat(
            [batch_values.X.astype("float32"), extra_data_pure.X], axis=2
        ).astype(dtype="float32")
        input_E_pure = paddle.concat(
            [batch_values.E.astype("float32"), extra_data_pure.E], axis=3
        ).astype(dtype="float32")
        input_y_pure = paddle.hstack(
            x=(batch_values.y.astype("float32"), extra_data_pure.y)
        ).astype(dtype="float32")
        # obtain the condition vector from output of encoder
        conditionVec = model.run_encoder(
            input_X_pure, input_E_pure, input_y_pure, node_mask
        )
        # complete input_y for decoder
        input_y = paddle.hstack(x=(input_y, conditionVec)).astype(dtype="float32")

        # forward of decoder with encoder output as condition vector of input of decoder
        pred = model.run_decoder(input_X, input_E, input_y, node_mask)

    pred_X = F.softmax(pred.X, axis=-1)
    pred_E = F.softmax(pred.E, axis=-1)

    # compute posterior distribution
    p_s_and_t_given_0_X = compute_batched_over0_posterior_distribution(
        X_t=X_t, Qt=Qt.X, Qsb=Qsb.X, Qtb=Qtb.X
    )
    p_s_and_t_given_0_E = compute_batched_over0_posterior_distribution(
        X_t=E_t, Qt=Qt.E, Qsb=Qsb.E, Qtb=Qtb.E
    )

    # compute node probability
    weighted_X = pred_X.unsqueeze(-1) * p_s_and_t_given_0_X
    unnormalized_prob_X = paddle.sum(weighted_X, axis=2)
    unnormalized_prob_X = paddle.where(
        paddle.sum(unnormalized_prob_X, axis=-1, keepdim=True) == 0,
        paddle.to_tensor(1e-5, dtype=unnormalized_prob_X.dtype),
        unnormalized_prob_X,
    )
    prob_X = unnormalized_prob_X / paddle.sum(
        unnormalized_prob_X, axis=-1, keepdim=True
    )

    # compute edge probability
    pred_E = pred_E.reshape([X_t.shape[0], -1, pred.E.shape[-1]])
    weighted_E = pred_E.unsqueeze(-1) * p_s_and_t_given_0_E
    unnormalized_prob_E = paddle.sum(weighted_E, axis=-2)
    unnormalized_prob_E = paddle.where(
        paddle.sum(unnormalized_prob_E, axis=-1, keepdim=True) == 0,
        paddle.to_tensor(1e-5, dtype=unnormalized_prob_E.dtype),
        unnormalized_prob_E,
    )
    prob_E = unnormalized_prob_E / paddle.sum(
        unnormalized_prob_E, axis=-1, keepdim=True
    )
    prob_E = prob_E.reshape([X_t.shape[0], X_t.shape[1], X_t.shape[1], -1])

    assert ((prob_X.sum(axis=-1) - 1).abs().max() < 1e-4).all()
    assert ((prob_E.sum(axis=-1) - 1).abs() < 1e-4).all()

    # sample from p(z_s | z_t)
    sampled_s = sample_discrete_features(prob_X, prob_E, node_mask)
    X_s = F.one_hot(sampled_s.X, num_classes=model.Xdim_output)
    E_s = F.one_hot(sampled_s.E, num_classes=model.Edim_output)

    assert paddle.all(E_s == paddle.transpose(E_s, perm=[0, 2, 1, 3]))
    assert (X_t.shape == X_s.shape) and (E_t.shape == E_s.shape)

    out_one_hot = diffgraphformer_utils.PlaceHolder(
        X=X_s, E=E_s, y=paddle.zeros([y_t.shape[0], 0])
    )
    out_discrete = diffgraphformer_utils.PlaceHolder(
        X=X_s, E=E_s, y=paddle.zeros([y_t.shape[0], 0])
    )

    return out_one_hot.mask(node_mask), out_discrete.mask(node_mask, collapse=True)


# -------------------------
# Noise & Q
# -------------------------
def apply_noise(model, X, E, y, node_mask, flag_use_formula=None):
    """
    Sample noise and apply it to the data.
    """
    t_int = paddle.randint(
        low=1, high=model.T + 1, shape=[X.shape[0], 1], dtype="int64"
    ).astype("float32")
    s_int = t_int - 1

    t_float = t_int / model.T  # nomarlize for stablizing training diffusion model
    s_float = s_int / model.T

    beta_t = model.noise_schedule(t_normalized=t_float)
    alpha_s_bar = model.noise_schedule.get_alpha_bar(t_normalized=s_float)
    alpha_t_bar = model.noise_schedule.get_alpha_bar(t_normalized=t_float)

    Qtb = model.transition_model.get_Qt_bar(alpha_t_bar)
    assert (abs(Qtb.X.sum(axis=2) - 1.0) < 1e-4).all(), Qtb.X.sum(axis=2) - 1
    assert (abs(Qtb.E.sum(axis=2) - 1.0) < 1e-4).all()

    probX = paddle.matmul(X, Qtb.X)  # (bs, n, dx_out)
    probE = paddle.matmul(E, Qtb.E.unsqueeze(1))  # (bs, n, n, de_out)

    sampled_t = sample_discrete_features(probX=probX, probE=probE, node_mask=node_mask)

    X_t = F.one_hot(sampled_t.X, num_classes=model.Xdim_output).astype("int64")
    if flag_use_formula is True:
        X_t = X
    E_t = F.one_hot(sampled_t.E, num_classes=model.Edim_output).astype("int64")
    assert (X.shape == X_t.shape) and (E.shape == E_t.shape)

    z_t = (
        diffgraphformer_utils.PlaceHolder(X=X_t, E=E_t, y=y)
        .type_as(X_t)
        .mask(node_mask)
    )

    noisy_data = {
        "t_int": t_int,
        "t": t_float,
        "beta_t": beta_t,
        "alpha_s_bar": alpha_s_bar,
        "alpha_t_bar": alpha_t_bar,
        "X_t": z_t.X,
        "E_t": z_t.E,
        "y_t": z_t.y,
        "node_mask": node_mask,
    }
    return noisy_data


def compute_extra_data(model, noisy_data, isPure=False):
    #  mix extra_features with domain_features and
    # noisy_data into X/E/y final inputs. domain_features
    extra_features = model.extra_features(noisy_data)
    extra_molecular_features = model.domain_features(noisy_data)

    extra_X = concat_without_empty(
        [extra_features.X, extra_molecular_features.X], axis=-1
    )
    extra_E = concat_without_empty(
        [extra_features.E, extra_molecular_features.E], axis=-1
    )
    extra_y = concat_without_empty(
        [extra_features.y, extra_molecular_features.y], axis=-1
    )

    if not isPure:
        t = noisy_data["t"]
        extra_y = concat_without_empty([extra_y, t], axis=1)

    return diffgraphformer_utils.PlaceHolder(X=extra_X, E=extra_E, y=extra_y)


def concat_without_empty(tensor_lst, axis=-1):
    new_lst = [t.astype("float32") for t in tensor_lst if 0 not in t.shape]
    if new_lst == []:
        return diffgraphformer_utils.return_empty(tensor_lst[0])
    return paddle.concat(new_lst, axis=axis)


# -------------------------
# KL prior
# -------------------------
def kl_prior(model, X, E, node_mask):
    """
    KL between q(zT|x) and prior p(zT)=Uniform(...)
    """
    bs = X.shape[0]
    ones = paddle.ones([bs, 1], dtype="float32")
    Ts = model.T * ones
    alpha_t_bar = model.noise_schedule.get_alpha_bar(t_int=Ts)  # (bs,1)

    Qtb = model.transition_model.get_Qt_bar(alpha_t_bar)
    probX = paddle.matmul(X, Qtb.X)  # (bs,n,dx_out)
    probE = paddle.matmul(E, Qtb.E.unsqueeze(1))  # (bs,n,n,de_out)

    # limit distribution
    limit_X = model.limit_dist.X.unsqueeze(0).unsqueeze(0)  # shape (1,1,dx_out)
    limit_X = paddle.expand(limit_X, [bs, X.shape[1], model.Xdim_output])

    limit_E = model.limit_dist.E.unsqueeze(0).unsqueeze(0).unsqueeze(0)
    limit_E = paddle.expand(limit_E, [bs, E.shape[1], E.shape[2], model.Edim_output])

    # mask
    limit_dist_X, limit_dist_E, probX, probE = mask_distributions(
        true_X=limit_X.clone(),
        true_E=limit_E.clone(),
        pred_X=probX,
        pred_E=probE,
        node_mask=node_mask,
    )

    kl_distance_X = F.kl_div(
        input=paddle.log(probX + 1e-10), label=limit_dist_X, reduction="none"
    )
    kl_distance_E = F.kl_div(
        input=paddle.log(probE + 1e-10), label=limit_dist_E, reduction="none"
    )
    klX_sum = sum_except_batch(kl_distance_X)
    klE_sum = sum_except_batch(kl_distance_E)
    return klX_sum + klE_sum


def compute_val_loss(
    model, pred, noisy_data, X, E, y, node_mask, condition, return_terms=False
):
    """
    Validation/Test VLB (NLL) with optional stateless return of decomposed terms.

    Args:
        model:             diffusion model with transition_model, node_dist, etc.
        pred:              namespace or object with .X/.E/.y logits.
        noisy_data:        dict produced by apply_noise (alpha_t_bar, beta_t, ...).
        X, E, y:           one-hot labels (same shapes as training).
        node_mask:         [B, N] boolean/int mask.
        condition:         reserved (as in original).
        return_terms:      if True, return per-sample vector terms dict instead of a
            scalar.

    Returns:
        If return_terms=False:
            - Scalar tensor (mean NLL) if stateless path;
            - Or model.(test|val)_nll(...) result (kept for backward-compat).
        If return_terms=True:
            - Dict[str, Tensor[B]] with keys:
              {"nll", "X_kl", "E_kl", "X_logp", "E_logp"}.
    """
    # 1.log p(N): number of nodes prior
    t = noisy_data["t"]
    N = paddle.sum(node_mask, axis=1).astype("int64")
    log_pN = model.node_dist.log_prob(N)

    # 2. KL(q(z_T|x), p(z_T)) => uniform prior
    kl_prior_ = kl_prior(model, X, E, node_mask)

    # 3. Stepwise diffusion loss
    if return_terms is True:
        (loss_all_t, xkl_vec, e_dl_vec) = compute_Lt(
            model, X, E, y, pred, noisy_data, node_mask, return_terms
        )
    else:
        loss_all_t = compute_Lt(
            model, X, E, y, pred, noisy_data, node_mask, return_terms
        )
        xkl_vec = e_dl_vec = None

    # 4. reconstruction loss
    prob0 = reconstruction_logp(model, t, X, E, node_mask, condition)
    loss_term_0_x = X * paddle.log(prob0.X + 1e-10)  # avoid log(0)
    loss_term_0_e = E * paddle.log(prob0.E + 1e-10)

    # Reduce to per-sample vectors
    x_logp_vec = _sum_over_non_batch_dims(loss_term_0_x)  # [B]
    e_logp_vec = _sum_over_non_batch_dims(loss_term_0_e)  # [B]
    rec_logp_vec = x_logp_vec + e_logp_vec  # [B]

    # combine
    # nlls = -log pN + KL_prior + stepwise_terms - recon_logp
    nlls = -log_pN + kl_prior_ + loss_all_t - rec_logp_vec

    return {
        "nll": nlls,  # [B]
        "X_logp": x_logp_vec,  # [B]
        "E_logp": e_logp_vec,  # [B]
        "X_kl": xkl_vec,  # [B]
        "E_kl": e_dl_vec,  # [B]
    }


def compute_Lt(model, X, E, y, pred, noisy_data, node_mask, return_terms: bool = False):
    """
    Step-wise diffusion term for VLB at validation/test.

    Returns
    -------
    if return_terms is False:
        loss_all_t : Tensor[B] == model.T * (x_kl + e_kl)
    else:
        (loss_all_t, x_kl, e_kl) : 3 Tensors[B]
           x_kl, e_kl are per-sample KL(P_true || P_pred) contributions (already scaled
            by model.T).
    """
    # 1. logits -> probabilities
    pred_probs_X = F.softmax(pred.X, axis=-1)
    pred_probs_E = F.softmax(pred.E, axis=-1)
    pred_probs_y = F.softmax(pred.y, axis=-1)

    # 2. schedules
    Qtb = model.transition_model.get_Qt_bar(noisy_data["alpha_t_bar"])
    Qsb = model.transition_model.get_Qt_bar(noisy_data["alpha_s_bar"])
    Qt = model.transition_model.get_Qt(noisy_data["beta_t"])

    # 3. true / predicted posterior distributions
    bs, n, _ = X.shape
    # compute true posterior distribution
    prob_true = posterior_distributions(
        X=X,
        E=E,
        y=y,
        X_t=noisy_data["X_t"],
        E_t=noisy_data["E_t"],
        y_t=noisy_data["y_t"],
        Qt=Qt,
        Qsb=Qsb,
        Qtb=Qtb,
    )
    prob_true.E = paddle.reshape(prob_true.E, [bs, n, n, -1])

    # compute predicted posterior distribution
    prob_pred = posterior_distributions(
        X=pred_probs_X,
        E=pred_probs_E,
        y=pred_probs_y,
        X_t=noisy_data["X_t"],
        E_t=noisy_data["E_t"],
        y_t=noisy_data["y_t"],
        Qt=Qt,
        Qsb=Qsb,
        Qtb=Qtb,
    )
    prob_pred.E = paddle.reshape(prob_pred.E, [bs, n, n, -1])

    # 4. mask invalid nodes/edges
    (prob_true_X, prob_true_E, prob_pred_X, prob_pred_E,) = mask_distributions(
        true_X=prob_true.X,
        true_E=prob_true.E,
        pred_X=prob_pred.X,
        pred_E=prob_pred.E,
        node_mask=node_mask,
    )

    # 5) KL(P_true || P_pred) = sum P_true * (log P_true - log P_pred)
    log_true_X = _safe_log(prob_true_X)
    log_pred_X = _safe_log(prob_pred_X)
    x_kl_per = prob_true_X * (log_true_X - log_pred_X)  # [B, N, Dx]
    x_kl_vec = paddle.sum(x_kl_per, axis=-1)  # [B, N]
    x_kl_vec = paddle.sum(x_kl_vec, axis=-1)  # [B]

    log_true_E = _safe_log(prob_true_E)
    log_pred_E = _safe_log(prob_pred_E)
    e_kl_per = prob_true_E * (log_true_E - log_pred_E)  # [B, N, N, De]
    e_kl_vec = paddle.sum(e_kl_per, axis=-1)  # [B, N, N]
    e_kl_vec = paddle.sum(e_kl_vec, axis=[1, 2])  # [B]

    # 6) scale by T and combine
    x_term = model.T * x_kl_vec  # [B]
    e_term = model.T * e_kl_vec  # [B]
    loss_all_t = x_term + e_term  # [B]

    if return_terms:
        return loss_all_t, x_term, e_term
    else:
        return loss_all_t


def reconstruction_logp(model, t, X, E, node_mask, condition_Spectrum):
    """
    L0: - log p(X,E|z0)
    sample randomly from X0, E0, then perform a forward pass
    """
    t_zeros = paddle.zeros_like(t)
    beta_0 = model.noise_schedule(t_zeros)
    Q0 = model.transition_model.get_Qt(beta_t=beta_0)

    probX0 = paddle.matmul(X, Q0.X)
    # E => broadcast
    probE0 = paddle.matmul(E, Q0.E.unsqueeze(1))

    sampled0 = sample_discrete_features(probX0, probE0, node_mask)  # TODO
    X0 = F.one_hot(sampled0.X, num_classes=model.Xdim_output)
    E0 = F.one_hot(sampled0.E, num_classes=model.Edim_output)
    y0 = sampled0.y
    assert (X.shape == X0.shape) and (E.shape == E0.shape)

    sampled_0 = diffgraphformer_utils.PlaceHolder(X=X0, E=E0, y=y0).mask(
        node_mask
    )  # TODO new add for step4

    # noisy_data
    noisy_data = {
        "X_t": sampled_0.X,
        "E_t": sampled_0.E,
        "y_t": sampled_0.y,
        "node_mask": node_mask,
        "t": paddle.zeros([X0.shape[0], 1]).astype(y0.dtype),
    }

    extra_data = compute_extra_data(model, noisy_data)

    # input_X
    input_X = paddle.concat(
        [noisy_data["X_t"].astype("float32"), extra_data.X], axis=2
    ).astype(dtype="float32")

    # input_E
    input_E = paddle.concat(
        [noisy_data["E_t"].astype("float32"), extra_data.E], axis=3
    ).astype(dtype="float32")

    # partial input_y for decoder
    input_y = paddle.hstack([noisy_data["y_t"].astype("float32"), extra_data.y]).astype(
        dtype="float32"
    )

    ###########################################################
    if getattr(model, "conditioning_mode", None) == "spectrum":
        embeddings_spectrum, _ = _encode_spectrum_condition(model, condition_Spectrum)
        input_y = paddle.concat([input_y, embeddings_spectrum], axis=1).astype(
            "float32"
        )

        # 4. Decoder forward
        # Convention: pred.X and pred.E are logits with shapes [B, n, Cx] and
        # [B, n, n, Ce]
        pred0 = model.run_decoder(input_X, input_E, input_y, node_mask)
    else:
        # prepare the extra feature for encoder input without noisy
        z_t = (
            diffgraphformer_utils.PlaceHolder(X=X0, E=E0, y=y0)
            .type_as(X)
            .mask(node_mask)
        )
        extra_data_pure = compute_extra_data(
            model,
            {"X_t": z_t.X, "E_t": z_t.E, "y_t": z_t.y, "node_mask": node_mask},
            isPure=True,
        )
        # prepare the input data for encoder combining extra features
        input_X_pure = paddle.concat(
            [z_t.X.astype("float32"), extra_data_pure.X], axis=2
        ).astype(dtype="float32")
        input_E_pure = paddle.concat(
            [z_t.E.astype("float32"), extra_data_pure.E], axis=3
        ).astype(dtype="float32")
        input_y_pure = paddle.hstack(
            x=(z_t.y.astype("float32"), extra_data_pure.y)
        ).astype(dtype="float32")
        # obtain the condition vector from output of encoder
        conditionVec = model.run_encoder(
            input_X_pure, input_E_pure, input_y_pure, node_mask
        )
        # complete input_y for decoder
        input_y = paddle.hstack(x=(input_y, conditionVec)).astype(dtype="float32")

        # forward of decoder with encoder output as condition vector of input of decoder
        pred0 = model.run_decoder(input_X, input_E, input_y, node_mask)
    ############################################################

    probX0 = F.softmax(pred0.X, axis=-1)
    probE0 = F.softmax(pred0.E, axis=-1)
    proby0 = F.softmax(pred0.y, axis=-1)

    ones_X = paddle.ones([model.Xdim_output], dtype=probX0.dtype)
    ones_E = paddle.ones([model.Edim_output], dtype=probE0.dtype)

    node_mask_3d = node_mask.unsqueeze(-1)
    probX0 = paddle.where(~node_mask_3d, ones_X, probX0)

    edge_mask = node_mask.unsqueeze(1) * node_mask.unsqueeze(2)
    edge_mask_4d = edge_mask.unsqueeze(-1)
    probE0 = paddle.where(~edge_mask_4d, ones_E, probE0)

    diag_mask = paddle.eye(probE0.shape[1], dtype="int64").astype("bool")
    diag_mask = diag_mask.unsqueeze(0).expand([probE0.shape[0], -1, -1])
    diag_mask_4d = diag_mask.unsqueeze(-1)
    probE0 = paddle.where(diag_mask_4d, ones_E, probE0)

    return diffgraphformer_utils.PlaceHolder(X=probX0, E=probE0, y=proby0)


# -----------------------
# molecule visualization/comparision
# -----------------------
def _safe_log(p: paddle.Tensor, eps: float = 1e-10) -> paddle.Tensor:
    # Avoid log(0)
    return paddle.log(paddle.clip(p, eps, 1.0))


def _sum_over_non_batch_dims(x: paddle.Tensor) -> paddle.Tensor:
    """Reduce tensor into [B] by summing over all non-batch dims."""
    if x is None:
        return None
    if x.ndim <= 1:
        return x
    axes = list(range(1, x.ndim))
    return paddle.sum(x, axis=axes)


class PredefinedNoiseSchedule(paddle.nn.Layer):
    """
    Predefined noise schedule. Essentially creates a lookup array for
        predefined (non-learned) noise schedules.
    """

    def __init__(self, noise_schedule, timesteps):
        super(PredefinedNoiseSchedule, self).__init__()
        self.timesteps = timesteps
        if noise_schedule == "cosine":
            alphas2 = cosine_beta_schedule(timesteps)
        elif noise_schedule == "custom":
            raise NotImplementedError()
        else:
            raise ValueError(noise_schedule)
        sigmas2 = 1 - alphas2
        log_alphas2 = np.log(alphas2)
        log_sigmas2 = np.log(sigmas2)
        log_alphas2_to_sigmas2 = log_alphas2 - log_sigmas2
        self.gamma = paddle.base.framework.EagerParamBase.from_tensor(
            tensor=paddle.to_tensor(data=-log_alphas2_to_sigmas2).astype(
                dtype="float32"
            ),
            trainable=False,
        )

    def forward(self, t):
        t_int = paddle.round(t * self.timesteps).astype(dtype="int64")
        return self.gamma[t_int]


class PredefinedNoiseScheduleDiscrete(paddle.nn.Layer):
    """
    Predefined noise schedule. Essentially creates a lookup array for
        predefined (non-learned) noise schedules.
    """

    def __init__(self, noise_schedule, timesteps):
        super(PredefinedNoiseScheduleDiscrete, self).__init__()
        self.timesteps = timesteps
        if noise_schedule == "cosine":
            betas = cosine_beta_schedule_discrete(timesteps)
        elif noise_schedule == "custom":
            betas = custom_beta_schedule_discrete(timesteps)
        else:
            raise NotImplementedError(noise_schedule)
        self.register_buffer(
            name="betas", tensor=paddle.to_tensor(data=betas).astype(dtype="float32")
        )
        self.alphas = 1 - paddle.clip(x=self.betas, min=0, max=0.9999)
        log_alpha = paddle.log(x=self.alphas)
        log_alpha_bar = paddle.cumsum(x=log_alpha, axis=0)
        self.alphas_bar = paddle.exp(x=log_alpha_bar)

    def forward(self, t_normalized=None, t_int=None):
        assert int(t_normalized is None) + int(t_int is None) == 1
        if t_int is None:
            t_int = paddle.round(t_normalized * self.timesteps)
        return self.betas[t_int.astype(dtype="int64")]

    def get_alpha_bar(self, t_normalized=None, t_int=None):
        assert int(t_normalized is None) + int(t_int is None) == 1
        if t_int is None:
            t_int = paddle.round(t_normalized * self.timesteps)
        return self.alphas_bar[t_int.astype(dtype="int64")]


class DiscreteUniformTransition:
    def __init__(self, x_classes: int, e_classes: int, y_classes: int):
        self.X_classes = x_classes
        self.E_classes = e_classes
        self.y_classes = y_classes
        self.u_x = paddle.ones(shape=[1, self.X_classes, self.X_classes])
        if self.X_classes > 0:
            self.u_x = self.u_x / self.X_classes
        self.u_e = paddle.ones(shape=[1, self.E_classes, self.E_classes])
        if self.E_classes > 0:
            self.u_e = self.u_e / self.E_classes
        self.u_y = paddle.ones(shape=[1, self.y_classes, self.y_classes])
        if self.y_classes > 0:
            self.u_y = self.u_y / self.y_classes

    def get_Qt(self, beta_t):
        """Returns one-step transition matrices for X and E, from step t - 1 to step t.
        Qt = (1 - beta_t) * I + beta_t / K

        beta_t: (bs)                         noise level between 0 and 1
        returns: qx (bs, dx, dx), qe (bs, de, de), qy (bs, dy, dy).
        """
        beta_t = beta_t.unsqueeze(axis=1)
        q_x = beta_t * self.u_x + (1 - beta_t) * paddle.eye(
            num_rows=self.X_classes
        ).unsqueeze(axis=0)
        q_e = beta_t * self.u_e + (1 - beta_t) * paddle.eye(
            num_rows=self.E_classes
        ).unsqueeze(axis=0)
        q_y = beta_t * self.u_y + (1 - beta_t) * paddle.eye(
            num_rows=self.y_classes
        ).unsqueeze(axis=0)
        return diffgraphformer_utils.PlaceHolder(X=q_x, E=q_e, y=q_y)

    def get_Qt_bar(self, alpha_bar_t):
        """Returns t-step transition matrices for X and E, from step 0 to step t.
        Qt = prod(1 - beta_t) * I + (1 - prod(1 - beta_t)) / K

        alpha_bar_t: (bs) Product of the (1 - beta_t) for each time step from 0 to t.
        returns: qx (bs, dx, dx), qe (bs, de, de), qy (bs, dy, dy).
        """
        alpha_bar_t = alpha_bar_t.unsqueeze(axis=1)
        q_x = (
            alpha_bar_t * paddle.eye(num_rows=self.X_classes).unsqueeze(axis=0)
            + (1 - alpha_bar_t) * self.u_x
        )
        q_e = (
            alpha_bar_t * paddle.eye(num_rows=self.E_classes).unsqueeze(axis=0)
            + (1 - alpha_bar_t) * self.u_e
        )
        q_y = (
            alpha_bar_t * paddle.eye(num_rows=self.y_classes).unsqueeze(axis=0)
            + (1 - alpha_bar_t) * self.u_y
        )
        return diffgraphformer_utils.PlaceHolder(X=q_x, E=q_e, y=q_y)


class MarginalUniformTransition:
    def __init__(self, x_marginals, e_marginals, y_classes):
        self.X_classes = len(x_marginals)
        self.E_classes = len(e_marginals)
        self.y_classes = y_classes
        self.x_marginals = x_marginals
        self.e_marginals = e_marginals
        self.u_x = (
            x_marginals.unsqueeze(axis=0)
            .expand(shape=[self.X_classes, -1])
            .unsqueeze(axis=0)
        )
        self.u_e = (
            e_marginals.unsqueeze(axis=0)
            .expand(shape=[self.E_classes, -1])
            .unsqueeze(axis=0)
        )
        self.u_y = paddle.ones(shape=[1, self.y_classes, self.y_classes])
        if self.y_classes > 0:
            self.u_y = self.u_y / self.y_classes

    def get_Qt(self, beta_t):
        """Returns one-step transition matrices for X and E, from step t - 1 to step t.
        Qt = (1 - beta_t) * I + beta_t / K

        beta_t: (bs)                         noise level between 0 and 1
        returns: qx (bs, dx, dx), qe (bs, de, de), qy (bs, dy, dy)."""
        beta_t = beta_t.unsqueeze(axis=1)
        q_x = (
            (
                beta_t * self.u_x
                + (1 - beta_t) * paddle.eye(num_rows=self.X_classes).unsqueeze(axis=0)
            )
            if self.X_classes != 0
            else diffgraphformer_utils.return_empty
        )
        q_e = (
            (
                beta_t * self.u_e
                + (1 - beta_t) * paddle.eye(num_rows=self.E_classes).unsqueeze(axis=0)
            )
            if self.E_classes != 0
            else diffgraphformer_utils.return_empty
        )
        q_y = (
            (
                beta_t * self.u_y
                + (1 - beta_t) * paddle.eye(num_rows=self.y_classes).unsqueeze(axis=0)
            )
            if self.y_classes != 0
            else diffgraphformer_utils.return_empty
        )
        return diffgraphformer_utils.PlaceHolder(X=q_x, E=q_e, y=q_y)

    def get_Qt_bar(self, alpha_bar_t):
        """Returns t-step transition matrices for X and E, from step 0 to step t.
        Qt = prod(1 - beta_t) * I + (1 - prod(1 - beta_t)) * K

        alpha_bar_t: (bs) Product of the (1 - beta_t) for each time step from 0 to t.
        returns: qx (bs, dx, dx), qe (bs, de, de), qy (bs, dy, dy).
        """
        alpha_bar_t = alpha_bar_t.unsqueeze(axis=1)
        q_x = (
            (
                alpha_bar_t * paddle.eye(num_rows=self.X_classes).unsqueeze(axis=0)
                + (1 - alpha_bar_t) * self.u_x
            )
            if self.X_classes != 0
            else diffgraphformer_utils.return_empty
        )
        q_e = (
            (
                alpha_bar_t * paddle.eye(num_rows=self.E_classes).unsqueeze(axis=0)
                + (1 - alpha_bar_t) * self.u_e
            )
            if self.E_classes != 0
            else diffgraphformer_utils.return_empty
        )
        q_y = (
            (
                alpha_bar_t * paddle.eye(num_rows=self.y_classes).unsqueeze(axis=0)
                + (1 - alpha_bar_t) * self.u_y
            )
            if self.y_classes != 0
            else diffgraphformer_utils.return_empty
        )
        return diffgraphformer_utils.PlaceHolder(X=q_x, E=q_e, y=q_y)


class AbsorbingStateTransition:
    def __init__(self, abs_state: int, x_classes: int, e_classes: int, y_classes: int):
        self.X_classes = x_classes
        self.E_classes = e_classes
        self.y_classes = y_classes
        self.u_x = paddle.zeros(shape=[1, self.X_classes, self.X_classes])
        self.u_x[:, :, abs_state] = 1
        self.u_e = paddle.zeros(shape=[1, self.E_classes, self.E_classes])
        self.u_e[:, :, abs_state] = 1
        self.u_y = paddle.zeros(shape=[1, self.y_classes, self.y_classes])
        self.u_e[:, :, abs_state] = 1

    def get_Qt(self, beta_t):
        """Returns two transition matrix for X and E"""
        beta_t = beta_t.unsqueeze(axis=1)
        q_x = beta_t * self.u_x + (1 - beta_t) * paddle.eye(
            num_rows=self.X_classes
        ).unsqueeze(axis=0)
        q_e = beta_t * self.u_e + (1 - beta_t) * paddle.eye(
            num_rows=self.E_classes
        ).unsqueeze(axis=0)
        q_y = beta_t * self.u_y + (1 - beta_t) * paddle.eye(
            num_rows=self.y_classes
        ).unsqueeze(axis=0)
        return q_x, q_e, q_y

    def get_Qt_bar(self, alpha_bar_t):
        """beta_t: (bs)
        Returns transition matrices for X and E"""
        alpha_bar_t = alpha_bar_t.unsqueeze(axis=1)
        q_x = (
            alpha_bar_t * paddle.eye(num_rows=self.X_classes).unsqueeze(axis=0)
            + (1 - alpha_bar_t) * self.u_x
        )
        q_e = (
            alpha_bar_t * paddle.eye(num_rows=self.E_classes).unsqueeze(axis=0)
            + (1 - alpha_bar_t) * self.u_e
        )
        q_y = (
            alpha_bar_t * paddle.eye(num_rows=self.y_classes).unsqueeze(axis=0)
            + (1 - alpha_bar_t) * self.u_y
        )
        return q_x, q_e, q_y

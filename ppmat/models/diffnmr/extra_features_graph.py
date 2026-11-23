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

import paddle

from ppmat.models.diffnmr.utils import diffgraphformer_utils


class DummyExtraFeatures:
    output_dims = {"X": 0, "E": 0, "y": 0}

    def __init__(self):
        """This class does not compute anything, just returns empty tensors."""

    def __call__(self, noisy_data):
        X = noisy_data["X_t"]
        E = noisy_data["E_t"]
        y = noisy_data["y_t"]

        empty_x = paddle.zeros(shape=X.shape[:-1] + [0], dtype=X.dtype)
        empty_e = paddle.zeros(shape=E.shape[:-1] + [0], dtype=E.dtype)
        empty_y = paddle.zeros(shape=[y.shape[0], 0], dtype=y.dtype)

        return diffgraphformer_utils.PlaceHolder(X=empty_x, E=empty_e, y=empty_y)


class ExtraFeatures:
    def __init__(self, extra_features_type, dataset_infos):
        self.max_n_nodes = dataset_infos.max_n_nodes
        self.ncycles = NodeCycleFeatures()
        self.features_type = extra_features_type
        if extra_features_type in ["eigenvalues", "all"]:
            self.eigenfeatures = EigenFeatures(mode=extra_features_type)
        self.output_dims = {
            "cycles": {"X": 3, "E": 0, "y": 5},
            "eigenvalues": {"X": 3, "E": 0, "y": 11},
            "all": {"X": 6, "E": 0, "y": 11},
        }[extra_features_type]

    def __call__(self, noisy_data):
        # n: (bs,1)
        mask_sum = paddle.sum(noisy_data["node_mask"], axis=1, keepdim=False)  # (bs,)
        n = paddle.unsqueeze(mask_sum, axis=1) / self.max_n_nodes  # (bs,1)

        # x_cycles, y_cycles: (bs, ?)
        x_cycles, y_cycles = self.ncycles(noisy_data)  # (bs, n_cycles)

        if self.features_type == "cycles":
            E = noisy_data["E_t"]
            extra_edge_attr = paddle.zeros(shape=E.shape[:-1] + [0], dtype=E.dtype)

            # 等效于 torch.hstack((n, y_cycles)) => concat along axis=1
            # 假设 n shape (bs,1), y_cycles shape (bs,k)
            # => result shape (bs, 1+k)
            y_stacked = paddle.concat([n, y_cycles], axis=1)

            return diffgraphformer_utils.PlaceHolder(
                X=x_cycles, E=extra_edge_attr, y=y_stacked
            )

        elif self.features_type == "eigenvalues":
            eigenfeatures = self.eigenfeatures(noisy_data)
            E = noisy_data["E_t"]
            extra_edge_attr = paddle.zeros(shape=E.shape[:-1] + [0], dtype=E.dtype)

            n_components, batched_eigenvalues = eigenfeatures  # (bs,1), (bs,10)

            # hstack => concat along axis=1
            y_stacked = paddle.concat(
                [n, y_cycles, n_components.astype(n.dtype), batched_eigenvalues], axis=1
            )

            return diffgraphformer_utils.PlaceHolder(
                X=x_cycles, E=extra_edge_attr, y=y_stacked
            )

        elif self.features_type == "all":
            eigenfeatures = self.eigenfeatures(noisy_data)
            E = noisy_data["E_t"]
            extra_edge_attr = paddle.zeros(shape=E.shape[:-1] + [0], dtype=E.dtype)

            (
                n_components,
                batched_eigenvalues,
                nonlcc_indicator,
                k_lowest_eigvec,
            ) = eigenfeatures
            # X = concat [x_cycles, nonlcc_indicator, k_lowest_eigvec] along last dim
            X_cat = paddle.concat(
                [x_cycles, nonlcc_indicator, k_lowest_eigvec], axis=-1
            )

            # y = hstack => concat along axis=1
            y_stacked = paddle.concat(
                [n, y_cycles, n_components.astype(n.dtype), batched_eigenvalues], axis=1
            )

            return diffgraphformer_utils.PlaceHolder(
                X=X_cat, E=extra_edge_attr, y=y_stacked
            )

        else:
            raise ValueError(f"Features type {self.features_type} not implemented")


class NodeCycleFeatures:
    def __init__(self):
        self.kcycles = KNodeCycles()

    def __call__(self, noisy_data):
        # adj_matrix: (bs, n, n), 取 E_t[...,1:] 并在最后一维 sum => shape (bs, n, n)
        E_t = noisy_data["E_t"]
        adj_matrix = paddle.sum(E_t[..., 1:], axis=-1).astype("float32")  # (bs, n, n)

        x_cycles, y_cycles = self.kcycles.k_cycles(
            adj_matrix=adj_matrix
        )  # (bs, n_cycles)

        # x_cycles 与 node_mask 对应位置相乘
        node_mask = paddle.unsqueeze(noisy_data["node_mask"], axis=-1)  # (bs, n, 1)
        x_cycles = x_cycles.astype(adj_matrix.dtype) * node_mask.astype(
            adj_matrix.dtype
        )

        # Avoid large values when the graph is dense
        x_cycles = x_cycles / 10
        y_cycles = y_cycles / 10

        # 类似 x_cycles[x_cycles > 1] = 1
        # Paddle 不支持直接 in-place boolean mask；需要先构造mask再赋值
        bool_mask_x = x_cycles > 1
        bool_mask_y = y_cycles > 1
        x_cycles = paddle.where(bool_mask_x, paddle.ones_like(x_cycles), x_cycles)
        y_cycles = paddle.where(bool_mask_y, paddle.ones_like(y_cycles), y_cycles)

        return x_cycles, y_cycles


class EigenFeatures:
    """
    Code taken from : https://github.com/Saro00/DGN/blob/master/models/pytorch/eigen_agg.py
    """

    def __init__(self, mode):
        """mode: 'eigenvalues' or 'all'"""
        self.mode = mode

    def __call__(self, noisy_data):
        E_t = noisy_data["E_t"]
        mask = noisy_data["node_mask"].astype("float32")
        A = paddle.sum(E_t[..., 1:], axis=-1).astype("float32")  # (bs, n, n)
        A = A * paddle.unsqueeze(mask, axis=1) * paddle.unsqueeze(mask, axis=2)

        L = compute_laplacian(A, normalize=False)

        # 添加正则化项以防止计算失败
        n_ = L.shape[-1]
        eps_eye = paddle.eye(n_, dtype=L.dtype) * 1e-6
        L = L + eps_eye
        # 强制对称化
        L = (L + paddle.transpose(L, perm=[0, 2, 1])) / 2

        # 构造对 mask 外节点的惩罚项
        mask_diag = paddle.eye(n_, dtype=L.dtype) * (2 * n_)
        mask_diag = paddle.unsqueeze(mask_diag, axis=0)  # (1, n, n)
        # (~mask) => paddle.logical_not
        # mask_bool = mask.astype("bool")
        mask_diag = (
            mask_diag
            * paddle.logical_not(paddle.unsqueeze(mask, 1)).astype(mask_diag.dtype)
            * paddle.logical_not(paddle.unsqueeze(mask, 2)).astype(mask_diag.dtype)
        )

        L = (
            L * paddle.unsqueeze(mask, axis=1) * paddle.unsqueeze(mask, axis=2)
            + mask_diag
        )

        if self.mode == "eigenvalues":
            # paddle.linalg.eigvalsh => (bs, n)
            eigvals = paddle.linalg.eigvalsh(L)  # (bs, n)
            sum_mask = paddle.sum(mask, axis=1, keepdim=True)
            eigvals = eigvals.astype(A.dtype) / sum_mask  # (bs,n)

            n_connected_comp, batch_eigenvalues = get_eigenvalues_features(eigvals)
            return (n_connected_comp.astype(A.dtype), batch_eigenvalues.astype(A.dtype))

        elif self.mode == "all":
            try:
                eigvals, eigvectors = paddle.linalg.eigh(L)  # (bs,n), (bs,n,n)
            except Exception as e:
                print(f"Warning: Eigen decomposition failed with linalg.eigh: {e}")
                # 回退到 SVD
                U, S, Vh = paddle.linalg.svd(L)
                eigvals = S
                eigvectors = paddle.transpose(Vh, perm=[0, 2, 1])
                print("Using SVD as fallback method.")

            sum_mask = paddle.sum(mask, axis=1, keepdim=True)
            eigvals = eigvals.astype(A.dtype) / sum_mask
            eigvectors = (
                eigvectors
                * paddle.unsqueeze(mask, axis=2)
                * paddle.unsqueeze(mask, axis=1)
            )

            # Retrieve eigenvalues features
            n_connected_comp, batch_eigenvalues = get_eigenvalues_features(eigvals)

            # Retrieve eigenvectors features
            nonlcc_indicator, k_lowest_eigvec = get_eigenvectors_features(
                vectors=eigvectors, node_mask=mask, n_connected=n_connected_comp
            )
            return (
                n_connected_comp,
                batch_eigenvalues,
                nonlcc_indicator,
                k_lowest_eigvec,
            )
        else:
            raise NotImplementedError(f"Mode {self.mode} is not implemented")


def compute_laplacian(adjacency, normalize: bool):
    """
    adjacency : batched adjacency matrix (bs, n, n)
    normalize: can be None, 'sym' or 'rw'
    """
    diag = paddle.sum(adjacency, axis=-1)  # (bs, n)
    n = diag.shape[-1]
    D = paddle.diag_embed(diag)  # (bs, n, n)
    combinatorial = D - adjacency  # (bs, n, n)

    if not normalize:
        return (combinatorial + paddle.transpose(combinatorial, perm=[0, 2, 1])) / 2

    diag0 = diag.clone()
    diag = paddle.where(diag == 0, paddle.to_tensor(1e-12, dtype=diag.dtype), diag)
    diag_norm = 1.0 / paddle.sqrt(diag)  # (bs, n)
    D_norm = paddle.diag_embed(diag_norm)  # (bs, n, n)
    eye_n = paddle.unsqueeze(paddle.eye(n, dtype=adjacency.dtype), axis=0)  # (1, n, n)

    L = eye_n - D_norm @ adjacency @ D_norm
    # For nodes where diag0 == 0, set their corresponding positions to 0
    zero_mask = (diag0 == 0).astype(L.dtype)  # (bs, n)
    # Need to broadcast to (bs, n, n), can first use unsqueeze
    zero_mask_2d = paddle.unsqueeze(zero_mask, axis=-1)  # (bs, n, 1)
    zero_mask_2d = paddle.matmul(
        zero_mask_2d, paddle.ones_like(zero_mask_2d).transpose([0, 2, 1])
    )
    # Set the corresponding position in L to 0
    L = paddle.where(zero_mask_2d > 0, paddle.zeros_like(L), L)
    return (L + paddle.transpose(L, perm=[0, 2, 1])) / 2


def get_eigenvalues_features(eigenvalues, k=5):
    """
    eigenvalues: (bs, n)
    k: num of non zero eigenvalues to keep
    """
    ev = eigenvalues
    bs, n = ev.shape
    # n_connected_components = (ev < 1e-5).sum(dim=-1)
    n_connected_components = paddle.sum(ev < 1e-5, axis=-1)  # (bs,)

    # TODO:Assertion: May need to handle errors here or add a check
    # assert (n_connected_components > 0).all(), "some assert..."

    to_extend = max(n_connected_components.numpy()) + k - n
    if to_extend > 0:
        fill_val = paddle.full(shape=[bs, to_extend], fill_value=2.0, dtype=ev.dtype)
        ev_extended = paddle.concat([ev, fill_val], axis=1)  # (bs, n+to_extend)
    else:
        ev_extended = ev

    # indices => shape (bs,k)
    #   range(k) + n_connected_components.unsqueeze(1)
    range_k = paddle.arange(k, dtype="int64")  # (k,)
    range_k = paddle.unsqueeze(range_k, axis=0)  # (1,k)
    indices = range_k + paddle.unsqueeze(
        n_connected_components.astype("int64"), axis=1
    )  # (bs,k)

    first_k_ev = batch_gather_2d(ev_extended, indices)

    n_connected_components = paddle.unsqueeze(n_connected_components, axis=-1)  # (bs,1)
    return n_connected_components, first_k_ev


def batch_gather_2d(data, index):
    bs, m = data.shape
    _, k = index.shape
    row_idx = paddle.arange(bs, dtype="int64")
    row_idx = paddle.unsqueeze(row_idx, axis=-1)  # (bs,1)
    row_idx = paddle.expand(row_idx, [bs, k])  # (bs,k)

    flat_indices = paddle.stack([row_idx.flatten(), index.flatten()], axis=1)

    gathered = paddle.gather_nd(data, flat_indices)  # (bs*k,)

    gathered = paddle.reshape(gathered, [bs, k])
    return gathered


def get_eigenvectors_features(vectors, node_mask, n_connected, k=2):
    """
    vectors (bs, n, n) : eigenvectors of Laplacian IN COLUMNS
    returns:
        not_lcc_indicator : indicator vectors of largest connected component (lcc) for
            each graph  -- (bs, n, 1)
        k_lowest_eigvec : k first eigenvectors for the largest connected component
            -- (bs, n, k)
    """
    bs, n = vectors.shape[0], vectors.shape[1]
    first_ev = paddle.round(vectors[:, :, 0] * 10**3) * node_mask
    random = paddle.randn(shape=[bs, n]) * (~node_mask.astype("bool")).astype("float32")
    first_ev = first_ev + random * 10**3
    most_common = paddle.mode(x=first_ev, axis=1)[0]  # TODO
    mask = ~(first_ev == most_common.unsqueeze(axis=1))
    not_lcc_indicator = (
        (mask * node_mask.astype("bool")).unsqueeze(axis=-1).astype(dtype="float32")
    )

    # Get the eigenvectors corresponding to the first nonzero eigenvalues
    to_extend = max(n_connected) + k - n
    if to_extend > 0:
        vectors = paddle.concat(
            x=(
                vectors,
                paddle.zeros(shape=[bs, n, to_extend]).astype(dtype=vectors.dtype),
            ),
            axis=2,
        )
    indices = paddle.arange(end=k).astype(dtype=vectors.dtype).astype(
        dtype="int64"
    ).unsqueeze(axis=0).unsqueeze(axis=0) + n_connected.unsqueeze(axis=2)
    indices = indices.expand(shape=[-1, n, -1])
    first_k_ev = paddle.take_along_axis(
        arr=vectors, axis=2, indices=indices, broadcast=False
    )
    first_k_ev = first_k_ev * node_mask.unsqueeze(axis=2)
    return not_lcc_indicator, first_k_ev


# ======================================
# 8. batch_trace, batch_diagonal
# ======================================
def batch_trace(X):
    """
    X: shape (bs, n, n)
    Return the trace for each sample  trace => shape (bs,)
    """
    diag = paddle.diagonal(X, axis1=-2, axis2=-1)  # (bs, n)
    return paddle.sum(diag, axis=-1)


def batch_diagonal(X):
    """
    X: shape (bs, n, n)
    Return its diagonal => (bs, n)
    """
    return paddle.diagonal(X, axis1=-2, axis2=-1)


# ======================================
# 9. KNodeCycles
# ======================================
class KNodeCycles:
    """Builds cycle counts for each node in a graph."""

    def __init__(self):
        super().__init__()

    def calculate_kpowers(self):
        self.k1_matrix = self.adj_matrix.astype("float32")
        self.d = paddle.sum(self.adj_matrix, axis=-1)  # (bs,n)
        self.k2_matrix = paddle.matmul(
            self.k1_matrix, self.adj_matrix.astype("float32")
        )
        self.k3_matrix = paddle.matmul(
            self.k2_matrix, self.adj_matrix.astype("float32")
        )
        self.k4_matrix = paddle.matmul(
            self.k3_matrix, self.adj_matrix.astype("float32")
        )
        self.k5_matrix = paddle.matmul(
            self.k4_matrix, self.adj_matrix.astype("float32")
        )
        self.k6_matrix = paddle.matmul(
            self.k5_matrix, self.adj_matrix.astype("float32")
        )

    def k3_cycle(self):
        c3 = batch_diagonal(self.k3_matrix)
        x3 = (c3 / 2.0).unsqueeze(-1).astype("float32")
        y3 = (paddle.sum(c3, axis=-1) / 6.0).unsqueeze(-1).astype("float32")
        return x3, y3

    def k4_cycle(self):
        diag_a4 = batch_diagonal(self.k4_matrix)  # (bs,n)
        c4 = (
            diag_a4
            - self.d * (self.d - 1)
            - paddle.sum(
                paddle.matmul(self.adj_matrix, paddle.unsqueeze(self.d, axis=-1)),
                axis=-1,
            )
        )
        x4 = (c4 / 2.0).unsqueeze(-1).astype("float32")
        y4 = (paddle.sum(c4, axis=-1) / 8.0).unsqueeze(-1).astype("float32")
        return x4, y4

    def k5_cycle(self):
        diag_a5 = batch_diagonal(self.k5_matrix)  # (bs,n)
        triangles = batch_diagonal(self.k3_matrix) / 2.0  # (bs,n)

        joint_cycles = self.k2_matrix * self.adj_matrix  # (bs,n,n)
        prod = 2 * paddle.matmul(joint_cycles, self.d.unsqueeze(-1)).squeeze(-1)
        prod2 = 2 * paddle.matmul(self.adj_matrix, triangles.unsqueeze(-1)).squeeze(-1)

        c5 = diag_a5 - prod - 4.0 * self.d * triangles - prod2 + 10.0 * triangles
        x5 = (c5 / 2.0).unsqueeze(-1).astype("float32")
        y5 = (paddle.sum(c5, axis=-1) / 10.0).unsqueeze(-1).astype("float32")
        return x5, y5

    def k6_cycle(self):
        term_1_t = batch_trace(self.k6_matrix)
        term_2_t = batch_trace(paddle.pow(self.k3_matrix, 2.0))
        term3_t = paddle.sum(
            self.adj_matrix * paddle.pow(self.k2_matrix, 2.0), axis=[-2, -1]
        )
        d_t4 = batch_diagonal(self.k2_matrix)
        a_4_t = batch_diagonal(self.k4_matrix)
        term_4_t = paddle.sum(d_t4 * a_4_t, axis=-1)
        term_5_t = batch_trace(self.k4_matrix)
        term_6_t = batch_trace(self.k3_matrix)
        term_7_t = paddle.sum(paddle.pow(batch_diagonal(self.k2_matrix), 3.0), axis=-1)
        term8_t = paddle.sum(self.k3_matrix, axis=[-2, -1])
        term9_t = paddle.sum(paddle.pow(batch_diagonal(self.k2_matrix), 2.0), axis=-1)
        term10_t = batch_trace(self.k2_matrix)

        c6_t = (
            term_1_t
            - 3.0 * term_2_t
            + 9.0 * term3_t
            - 6.0 * term_4_t
            + 6.0 * term_5_t
            - 4.0 * term_6_t
            + 4.0 * term_7_t
            + 3.0 * term8_t
            - 12.0 * term9_t
            + 4.0 * term10_t
        )
        y6 = (c6_t / 12.0).unsqueeze(-1).astype("float32")
        return None, y6

    def k_cycles(self, adj_matrix, verbose=False):
        """
        adj_matrix: (bs, n, n)
        return: (kcyclesx, kcyclesy)
        """
        self.adj_matrix = adj_matrix
        self.calculate_kpowers()

        k3x, k3y = self.k3_cycle()
        assert paddle.all(k3x >= -0.1)

        k4x, k4y = self.k4_cycle()
        assert paddle.all(k4x >= -0.1)

        k5x, k5y = self.k5_cycle()
        assert paddle.all(k5x >= -0.1), k5x

        _, k6y = self.k6_cycle()
        assert paddle.all(k6y >= -0.1)

        kcyclesx = paddle.concat([k3x, k4x, k5x], axis=-1)  # (bs, n, 3)
        kcyclesy = paddle.concat([k3y, k4y, k5y, k6y], axis=-1)  # (bs, ncycles?)
        return kcyclesx, kcyclesy


# ==============
# Some auxiliary functions
# ==============


def paddle_mode(tensor, axis=1):
    """
    approximate torch.mode(...).values function, returning the most frequently
    occurring element along the given axis.
    Note: Now, Paddle does not have a direct mode API, so here's a simplified approach:
        1. turn tensor into numpy
        2. call scipy.stats or numpy method to find mode
        3. return paddle.Tensor
    If your scenario has high performance requirements, you may implement a more
    efficient native statistical method for Paddle.
    """
    import numpy as np

    data_np = tensor.numpy()
    # Calculate the mode for each row
    # if you have scipy, then you can use scipy.stats.mode directly:
    # from scipy.stats import mode
    # m = mode(data_np, axis=axis, keepdims=False).mode
    # here is pure inplementation of numpy:
    bs = data_np.shape[0]
    modes = []
    for i in range(bs):
        vals, counts = np.unique(data_np[i], return_counts=True)
        max_count_idx = np.argmax(counts)
        modes.append(vals[max_count_idx])
    modes_np = np.array(modes).reshape([-1])  # shape (bs,)
    return paddle.to_tensor(modes_np, dtype=tensor.dtype)


def round_to_decimals(tensor, decimals=3):
    factor = 10**decimals
    return paddle.round(tensor * factor) / factor

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

import paddle
from pgl.math import segment_sum


class PlaceHolder:
    """
    Container for batched graph tensors with convenience utilities.

    Purpose
    -------
    Encapsulates (X, E, y) for a batch of graphs and provides:
      1) `type_as(x)`: align dtypes of X/E/y to `x.dtype`.
      2) `mask(node_mask, collapse=False)`: apply a node mask to X and E.

    Attributes
    ----------
    X : paddle.Tensor
        Node features or per-class distributions. Shape: (B, N, F).
    E : paddle.Tensor
        Edge features or per-class distributions. Shape: (B, N, N, D).
        Expected to be symmetric on node axes: E[:, i, j, :] == E[:, j, i, :].
    y : paddle.Tensor
        Target tensor; shape is task-dependent.

    Methods
    -------
    type_as(x: paddle.Tensor) -> PlaceHolder
        Cast X, E, y to `x.dtype` and return self (in-place).
    mask(node_mask, collapse: bool = False) -> PlaceHolder
        Apply node-wise masking and return self (in-place).
        - `node_mask`: shape (B, N), values in {0,1}/bool.
        - If `collapse == False`:
            * X *= x_mask where x_mask has shape (B, N, 1).
            * E *= e_mask1 * e_mask2, where e_mask1=(B,N,1,1), e_mask2=(B,1,N,1).
            * Verifies symmetry of E after masking and raises ValueError if violated.
        - If `collapse == True`:
            * X = argmax(X, axis=-1)  -> shape (B, N), integer labels.
            * E = argmax(E, axis=-1)  -> shape (B, N, N), integer labels.
            * Any masked node (0) is set to -1 in X; any edge touching a masked node is
                set to -1 in E.
            * Use -1 as a sentinel/padding id; change if your downstream code requires
                a different pad id.
    """

    def __init__(self, X, E, y):
        self.X = X
        self.E = E
        self.y = y

    def type_as(self, x: paddle.Tensor):
        self.X = self.X.astype(x.dtype)
        self.E = self.E.astype(x.dtype)
        self.y = self.y.astype(x.dtype)
        return self

    def mask(self, node_mask, collapse=False):
        # x_mask = node_mask.unsqueeze(-1)
        x_mask = paddle.unsqueeze(node_mask, axis=-1).astype(self.X.dtype)  # (bs, n, 1)
        e_mask1 = paddle.unsqueeze(x_mask, axis=2)  # (bs, n, 1, 1)
        e_mask2 = paddle.unsqueeze(x_mask, axis=1)  # (bs, 1, n, 1)

        if collapse:
            # self.X = torch.argmax(self.X, dim=-1)
            self.X = paddle.argmax(self.X, axis=-1)  # (bs,n)
            self.E = paddle.argmax(self.E, axis=-1)  # (bs,n,n)

            # self.X[node_mask == 0] = -1
            zero_mask = node_mask == 0
            self.X = paddle.where(
                zero_mask, paddle.full_like(self.X, fill_value=-1), self.X
            )

            # e_mask => (bs,n,n) shape (由 e_mask1 * e_mask2 => (bs,n,n,1)?)
            e_mask = paddle.squeeze(e_mask1 * e_mask2, axis=-1)  # (bs,n,n)
            self.E = paddle.where(
                e_mask == 0, paddle.full_like(self.E, fill_value=-1), self.E
            )
        else:
            # self.X = self.X * x_mask
            self.X = self.X * x_mask
            # self.E = self.E * e_mask1 * e_mask2
            self.E = self.E * e_mask1 * e_mask2

            E = self.E.astype("float32")
            if not paddle.allclose(E, paddle.transpose(E, perm=[0, 2, 1, 3])):
                raise ValueError("E is not symmetric after masking.")
        return self


def to_dense(x, edge_index, edge_attr, batch):
    """
    Convert sparse graph data to dense format (PaddlePaddle version) (Paddle version)
    Args:
        x (paddle.Tensor): node feature matrix, shape (N, F)
        edge_index (paddle.Tensor): edge index matrix, shape (2, E)
        edge_attr (paddle.Tensor): edge attribute matrix, shape (E, D)
        batch (paddle.Tensor): node-to-graph batch index vector, shape (N, )
    Returns:
        PlaceHolder: Contains the densified node feature matrix and adjacency matrix
    """
    X, node_mask = to_dense_batch(x=x, batch=batch)
    # remove self-loops
    edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)

    max_num_nodes = X.shape[1]
    E = to_dense_adj(
        edge_index=edge_index,
        batch=batch,
        edge_attr=edge_attr,
        max_num_nodes=max_num_nodes,
    )
    E = encode_no_edge(E)

    return PlaceHolder(X=X, E=E, y=None), node_mask


def to_dense_batch(x, batch, fill_value=0, max_num_nodes=None, batch_size=None):
    """Transfrom a batch of graphs to a dense node feature tensor and
       provide the mask  holing the positions of dummy nodes

    Args:
        x (paddle.tensor): The feature map of nodes
        batch (pgl.Graph): The graph holing the graph node id
        fill_value (bool): The value of dummy nodes. Default: 0.
        max_node_nodes: The dimension of nodes in dense batch. Default: None.
        batch_size (int, optional): The batch size. Default: None.

    Returns:

        out (paddle.tensor): Returns a dense node feature tensor
            (shape = [batch_size,max_num_nodes,-1])
        mask (paddle.tensor): Return a mask indicating the position of
            dummy nodes (shape = [batch_size, max_num_nodes])

    """
    if batch is None and max_num_nodes is None:
        mask = paddle.ones(shape=[1, x.shape[0]], dtype="bool")
        return paddle.unsqueeze(x, axis=0), mask

    if batch is None:
        batch = paddle.zeros(shape=[x.shape[0]], dtype="int64")

    if batch_size is None:
        batch_size = (batch.max().item()) + 1

    num_nodes = segment_sum(paddle.ones([x.shape[0]]), batch)
    cum_nodes = paddle.concat([paddle.zeros([1]), num_nodes.cumsum(0)]).astype(
        batch.dtype
    )

    if max_num_nodes is None:
        max_num_nodes = int(num_nodes.max())

    idx = paddle.arange(batch.shape[0], dtype=batch.dtype)
    idx = (idx - cum_nodes[batch]) + (batch * max_num_nodes)

    size = [batch_size * max_num_nodes] + list(x.shape)[1:]
    out = paddle.full(size, fill_value).astype(x.dtype)
    out = paddle.scatter(out, idx, x)
    out = out.reshape([batch_size, max_num_nodes] + list(x.shape)[1:])

    mask = paddle.zeros(batch_size * max_num_nodes, dtype=paddle.bool)
    mask[idx] = 1
    mask = mask.reshape([batch_size, max_num_nodes])

    return out, mask


def remove_self_loops(edge_index, edge_attr=None):
    mask = edge_index[0] != edge_index[1]
    edge_index = edge_index[:, mask]

    if edge_attr is not None:
        edge_attr = edge_attr[mask]

    return edge_index, edge_attr


def to_dense_adj(
    edge_index,
    batch=None,
    edge_attr=None,
    max_num_nodes=None,
    batch_size=None,
):
    if batch is None:
        max_index = int(edge_index.max()) + 1 if edge_index.numel() > 0 else 0
        batch = paddle.zeros(shape=[max_index], dtype="int64")

    if batch_size is None:
        batch_size = int(batch.max()) + 1 if batch.numel() > 0 else 1

    # ``batch`` still contains the node-to-graph assignment when a graph has no
    # edges.  Handle that case before indexing ``batch[edge_index[0]]`` below;
    # Paddle cannot index an empty tensor view in this path.  Keep a feature
    # channel even when ``edge_attr`` is omitted so the result remains
    # compatible with ``encode_no_edge`` and the dense graph contract.
    if edge_index.numel() == 0:
        if max_num_nodes is None:
            if batch.numel() == 0:
                max_num_nodes = 0
            else:
                num_nodes = segment_sum(paddle.ones([batch.shape[0]]), batch)
                max_num_nodes = int(num_nodes.max())
        feature_shape = list(edge_attr.shape[1:]) if edge_attr is not None else [1]
        if not feature_shape:
            feature_shape = [1]
        return paddle.zeros(
            [batch_size, max_num_nodes, max_num_nodes] + feature_shape,
            dtype="float32",
        )

    one = paddle.ones_like(batch, dtype=paddle.float32)
    num_nodes = segment_sum(one, batch)
    cum_nodes = paddle.concat([paddle.zeros([1]), num_nodes.cumsum(0)]).astype(
        edge_index.dtype
    )

    idx0 = batch[edge_index[0]].astype(edge_index.dtype)
    idx1 = edge_index[0] - cum_nodes[batch][edge_index[0]]
    idx2 = edge_index[1] - cum_nodes[batch][edge_index[1]]

    if max_num_nodes is None:
        max_num_nodes = int(num_nodes.max())
    elif (idx1.numel() > 0 and idx1.max() >= max_num_nodes) or (
        idx2.numel() > 0 and idx2.max() >= max_num_nodes
    ):
        mask = (idx1 < max_num_nodes) & (idx2 < max_num_nodes)
        idx0 = idx0[mask]
        idx1 = idx1[mask]
        idx2 = idx2[mask]
        edge_attr = None if edge_attr is None else edge_attr[mask]

    if edge_attr is None:
        edge_attr = paddle.ones(shape=[idx0.numel()], dtype=edge_index.dtype)

    size = [batch_size, max_num_nodes, max_num_nodes]
    size.extend(list(edge_attr.shape[1:]))
    flattened_size = batch_size * max_num_nodes * max_num_nodes

    idx = idx0 * max_num_nodes * max_num_nodes + idx1 * max_num_nodes + idx2
    adj_partial = segment_sum(edge_attr, idx)
    adj = paddle.zeros([flattened_size, edge_attr.shape[1]], dtype=paddle.float32)
    index = paddle.arange(idx.max() + 1)
    adj[index] = adj_partial
    adj = paddle.reshape(adj, size)

    return adj


def encode_no_edge(E):
    assert len(E.shape) == 4
    if E.shape[-1] == 0:
        return E
    no_edge = paddle.sum(E, axis=3) == 0
    first_elt = E[:, :, :, 0]
    first_elt = paddle.where(no_edge, paddle.ones_like(first_elt), first_elt)
    E[:, :, :, 0] = first_elt
    diag = paddle.eye(E.shape[1], dtype="int32").unsqueeze(0).tile([E.shape[0], 1, 1])
    diag = diag.astype("bool")
    E = paddle.where(diag.unsqueeze(-1), paddle.zeros_like(E), E)
    return E


def return_empty(x, shape=None):
    if shape is not None:
        return paddle.empty(shape, dtype="float32")
    return paddle.empty(x.shape, dtype="float32")


# ===========================
# test
# ===========================
if __name__ == "__main__":
    import paddle

    # create test data
    x = paddle.arange(15).reshape([5, 3])  # 5 nodes，every node has 3 dimension feature
    edge_index = paddle.to_tensor([[0, 1, 2], [1, 2, 0]], dtype="int64")
    edge_attr = paddle.ones([3, 2]) * 2  # 3 edge，every edge has 2 dimension feature
    batch = paddle.to_tensor([0, 0, 1, 1, 1], dtype="int64")

    # test to_dense function
    placeholder, node_mask = to_dense(x, edge_index, edge_attr, batch)
    print("X Shape:", placeholder.X.shape)
    print("E Shape:", placeholder.E.shape)
    print("Node Mask:", node_mask.shape)

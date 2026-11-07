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

import argparse
import hashlib
import json
import types
from pathlib import Path

import dgl
import numpy as np
import paddle
import torch
from omegaconf import OmegaConf

from ppmat.datasets.build_solvent import BuildBinaryMixture
from ppmat.datasets.build_solvent import BuildSolvent
from ppmat.datasets.collate_fn import DefaultCollator
from ppmat.models import build_graph_converter
from ppmat.models.gegnn import GEGNNBinary
from ppmat.vocab import build_vocab

parser = argparse.ArgumentParser(
    description="Compare the original PyTorch GE-GNN with Paddle"
)
parser.add_argument("--reference_model_path", required=True, type=Path)
parser.add_argument("--output_path", default="gegnn_alignment.json", type=Path)
args = parser.parse_args()
base_dir = Path(__file__).resolve().parents[1]
paddle.set_device("cpu")
torch.set_num_threads(1)
paddle.seed(2021)
source = args.reference_model_path.read_text()
upstream = types.ModuleType("upstream_gegnn")
exec(
    compile(
        source.replace(".cuda()", ".cpu()").replace('.to("cuda")', '.to("cpu")'),
        "upstream_model_GNN.py",
        "exec",
    ),
    upstream.__dict__,
)
config = OmegaConf.to_container(
    OmegaConf.load(
        base_dir / "property_prediction/configs/gegnn/gegnn_binary_activity.yaml"
    ),
    resolve=True,
)
builder = BuildSolvent(
    build_graph_converter(
        config["Predict"]["graph_converter"], vocab=build_vocab(config["Vocabulary"])
    )
)
pairs = [("CCO", "O", 0.25, [0.1, -0.05]), ("CC(=O)C", "c1ccncc1", 0.75, [0.2, 0.15])]
samples = [BuildBinaryMixture()(builder(a), builder(b), x, y) for a, b, x, y in pairs]
pd_data = DefaultCollator()(samples)


def torch_graph(g):
    edge = np.asarray(g.edges)
    graph = dgl.graph((edge[:, 0], edge[:, 1]), num_nodes=g.num_nodes)
    graph.ndata["h"] = torch.tensor(g.node_feat["h"])
    return graph


th_data = {
    "g1": dgl.batch([torch_graph(x["g1"]) for x in samples]),
    "g2": dgl.batch([torch_graph(x["g2"]) for x in samples]),
}
for key in ["inter_hb", "intra_hb1", "intra_hb2"]:
    th_data[key] = torch.tensor(np.concatenate([x[key] for x in samples]))
th_data["solv1_x"] = torch.tensor([x for a, b, x, y in pairs], dtype=torch.float32)
interaction = dgl.graph(
    ([0, 1, 2, 3, 0, 1, 2, 3], [2, 3, 0, 1, 0, 1, 2, 3]), num_nodes=4
)
reports = []
for hidden_dim in [8, 256]:
    pd_model = GEGNNBinary(
        hidden_dim=hidden_dim, edge_in_feats=1, mlp_activation="softplus"
    )
    th_model = upstream.gegnn_binary(
        74, hidden_dim, 1, mlp_activation="softplus", mpnn_activation="relu"
    )
    mapping = {}
    for name, tensor in th_model.state_dict().items():
        transpose = name.endswith(".weight")
        if name in ["conv1.weight", "conv2.weight"]:
            key = name.replace(".weight", ".linear.weight")
            transpose = False
        elif ".gru." in name:
            suffix = name.rsplit(".", 1)[1]
            base = name.rsplit(".", 1)[0]
            key = (
                base
                + "."
                + {
                    "weight_ih_l0": "input_linear.weight",
                    "weight_hh_l0": "hidden_linear.weight",
                    "bias_ih_l0": "input_linear.bias",
                    "bias_hh_l0": "hidden_linear.bias",
                }[suffix]
            )
            transpose = suffix.startswith("weight")
        else:
            key = name.replace(".edge_func.", ".edge_network.")
        array = pd_model.state_dict()[key].numpy()
        tensor.copy_(torch.from_numpy(array.T.copy() if transpose else array.copy()))
        mapping[name] = (key, transpose)
    assert len(mapping) == len(pd_model.state_dict())
    pd_opt = paddle.optimizer.Adam(
        learning_rate=0.001, parameters=pd_model.parameters()
    )
    th_opt = torch.optim.Adam(th_model.parameters(), lr=0.001)
    target = np.asarray([y for a, b, x, y in pairs], dtype=np.float32)
    steps = []
    for step in range(2):
        pd_opt.clear_grad()
        th_opt.zero_grad()
        pd_result = pd_model(pd_data)
        pd_pred = pd_result["pred_dict"]["gamma"]
        pd_loss = pd_result["loss_dict"]["loss"]
        th_pred = th_model(th_data, interaction)[0]
        th_loss = torch.nn.functional.mse_loss(th_pred, torch.tensor(target))
        diff = float(np.max(np.abs(pd_pred.numpy() - th_pred.detach().numpy())))
        loss_diff = abs(float(pd_loss) - float(th_loss.detach()))
        pd_loss.backward()
        th_loss.backward()
        grad_diff = max(
            float(
                np.max(
                    np.abs(
                        pd_model.state_dict()[key].grad.numpy()
                        - (
                            param.grad.detach().numpy().T
                            if trans
                            else param.grad.detach().numpy()
                        )
                    )
                )
            )
            for name, param in th_model.named_parameters()
            for key, trans in [mapping[name]]
        )
        pd_opt.step()
        th_opt.step()
        steps.append(
            {
                "step": step + 1,
                "prediction_max_abs_diff": diff,
                "loss_abs_diff": loss_diff,
                "gradient_max_abs_diff": grad_diff,
                "paddle_loss": float(pd_loss),
                "torch_loss": float(th_loss.detach()),
            }
        )
        assert diff < 1e-4 and loss_diff < 1e-4 and grad_diff < 1e-4
    reports.append({"hidden_dim": hidden_dim, "batch_size": 2, "adam_steps": steps})
report = {
    "paddle": paddle.__version__,
    "torch": torch.__version__,
    "dgl": dgl.__version__,
    "device": "CPU",
    "source_url": "https://git.rwth-aachen.de/avt-svt/public/GDI-NN/-/raw/main/model/model_GNN.py",
    "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
    "source_adaptation": "Device calls changed from CUDA to CPU only.",
    "results": reports,
}
args.output_path.write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))

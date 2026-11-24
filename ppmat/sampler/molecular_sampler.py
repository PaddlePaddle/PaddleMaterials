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

import copy
import inspect
import os
import os.path as osp
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

import numpy as np
import paddle
import pandas as pd
from omegaconf import OmegaConf

from ppmat.datasets import build_dataloader
from ppmat.datasets import build_dataset_infos
from ppmat.datasets import set_signal_handlers
from ppmat.metrics import build_metric
from ppmat.models import MODEL_REGISTRY
from ppmat.models import build_model
from ppmat.utils import download
from ppmat.utils import logger
from ppmat.utils import save_load
from ppmat.utils.execution import configure_execution_backend
from ppmat.utils.model_package import get_model_config_path
from ppmat.utils.model_package import resolve_model_package_dir
from ppmat.visualization import AnimationWriter
from ppmat.visualization import MoleculeVisualizer
from ppmat.visualization import molecular_graph_to_rdkit
from ppmat.vocab import build_vocab


class MolecularSampler:
    """Sample molecular-generation models through a common model contract.

    A molecular model used by this sampler must expose ``sample(data, **kwargs)``.
    Dataloader sampling additionally expects that method to return a mapping with a
    ``pred`` sequence. Models can optionally return ``true``, ``num_samples``,
    ``node_counts``, ``conditions``, ``atom_decoder``, ``node_onehot``, and
    ``edge_onehot``. This keeps model-specific generation inside the model while the
    sampler owns model loading, iteration, result aggregation, and metrics.

    Two initialization modes are supported:

    1. **Automatic Model Loading**
       Specify `model_name` and `weights_name` to automatically download
       and load pre-trained weights from the `MODEL_REGISTRY`.

    2. **Custom Model Loading**
       Provide explicit `config_path` and `checkpoint_path` to load
       custom-trained models from local files.

    Args:
        model_name (Optional[str], optional): Name of a predefined model package from
            ``MODEL_REGISTRY``. When specified, associated weights
            will be automatically downloaded. Defaults to None.

        weights_name (Optional[str], optional): Specific pre-trained weight identifier.
            Used only when `model_name` is provided. Valid options include:
            - 'best.pdparams' (highest validation performance)
            - 'latest.pdparams' (most recent training checkpoint)
            - Custom weight files ending with '.pdparams'
            Defaults to None.

        config_path (Optional[str], optional): Path to model configuration file (YAML)
            for custom models. Required when not using predefined `model_name`.
            Defaults to None.
        checkpoint_path (Optional[str], optional): Path to model checkpoint file
            (.pdparams) for custom models. Required when not using predefined
            `model_name`. Defaults to None.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        weights_name: Optional[str] = None,
        config_path: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
        config_overrides: Optional[List[str]] = None,
    ):
        package_config_dir = None
        if model_name is None:
            assert config_path is not None and checkpoint_path is not None, (
                "config_path and checkpoint_path must be provided when model_name is "
                "None."
            )

            logger.info(f"Loading model from {config_path} and {checkpoint_path}.")

            config_base_dir = os.path.dirname(os.path.abspath(config_path))
            checkpoint_dir = (
                checkpoint_path
                if checkpoint_path and os.path.isdir(checkpoint_path)
                else None
            )
            config = OmegaConf.load(config_path)
            if config_overrides:
                cli_config = OmegaConf.from_dotlist(config_overrides)
                config = OmegaConf.merge(config, cli_config)
            config = OmegaConf.to_container(config, resolve=True)
            self._resolve_package_paths(
                config,
                config_base_dir=config_base_dir,
                checkpoint_dir=checkpoint_dir,
            )
        else:
            logger.info(f"Loading registered model: {model_name}")
            extracted_path = download.get_weights_path_from_url(
                MODEL_REGISTRY[model_name]
            )
            package_config_dir = resolve_model_package_dir(model_name, extracted_path)
            config_path = get_model_config_path(model_name, package_config_dir)
            checkpoint_path = package_config_dir
            config = OmegaConf.load(config_path)
            if config_overrides:
                cli_config = OmegaConf.from_dotlist(config_overrides)
                config = OmegaConf.merge(config, cli_config)
            config = OmegaConf.to_container(config, resolve=True)
            self._resolve_package_paths(
                config,
                config_base_dir=package_config_dir,
                checkpoint_dir=None,
            )

        model_config = config.get("Model", None)
        assert model_config is not None, "Model config must be provided."
        self.vocab = build_vocab(config.get("Vocabulary"))

        set_signal_handlers()
        sample_config = config.get("Sampler")
        assert sample_config is not None, "Sampler config must be provided."
        sample_data_config = sample_config.get("data")
        sample_loader = (
            build_dataloader(sample_data_config, vocab=self.vocab)
            if sample_data_config is not None
            else None
        )

        dataset_infos = None
        train_config = config.get("Dataset", {}).get("train")
        if train_config is not None:
            # Dataset statistics are computed from the configured training split on
            # first use and reused from its cache afterwards.
            train_loader = build_dataloader(train_config, vocab=self.vocab)
            dataset_infos = build_dataset_infos(
                dataloaders={"train": train_loader},
                cfg=copy.deepcopy(config),
                vocab=self.vocab,
                recompute_statistics=False,
            )

        clip_config = config.get("CLIP")
        self.clip = (
            build_model(
                clip_config,
                vocab=self.vocab,
                dataset_infos=dataset_infos,
            )
            if clip_config is not None
            else None
        )

        self.dataset_infos = dataset_infos
        self.molecule_visualizer = MoleculeVisualizer()
        self.animation_writer = AnimationWriter()

        model_cfg = config["Model"]
        model = build_model(
            model_cfg,
            vocab=self.vocab,
            dataset_infos=dataset_infos,
            clip=self.clip,
        )

        self.pretrained_model_path = (
            checkpoint_path
            if checkpoint_path is not None
            else config.get("pretrained_model_path", None)
        )
        self.pretrained_weight_name = weights_name
        if self.pretrained_weight_name is None:
            self.pretrained_weight_name = config.get("pretrained_weight_name", None)
        if (
            self.pretrained_weight_name is None
            and self.pretrained_model_path is not None
            and os.path.isdir(self.pretrained_model_path)
        ):
            sampler_pretrained_path = config.get("Sampler", {}).get(
                "pretrained_model_path", None
            )
            if sampler_pretrained_path is not None:
                self.pretrained_weight_name = os.path.basename(sampler_pretrained_path)
        if self.pretrained_model_path is not None:
            save_load.load_pretrain(
                model, self.pretrained_model_path, self.pretrained_weight_name
            )

        self.model = model
        execution_config = config.get("Execution") or {}
        self.execution_backend = configure_execution_backend(
            self.model,
            execution_config.get("backend"),
            init_params=execution_config.get("__init_params__"),
            owner="MolecularSampler",
        )
        self.model.eval()
        self.config = config
        self.sample_config = sample_config

        self._sample_loader = sample_loader
        self.sample_batch_iters = sample_config.get("sample_batch_iters")
        self.visual_num = int(sample_config.get("visual_num", 0))
        self.chains_to_save = int(sample_config.get("chains_to_save", 0))
        self.model_sample_params = copy.deepcopy(
            sample_config.get("model_sample_params", {}) or {}
        )
        # Backward-compatible mapping for existing DiffNMR configs. New molecular
        # models should place model-owned arguments under ``model_sample_params``.
        legacy_model_params = {
            "keep_chain": sample_config.get("chains_to_save"),
            "number_chain_steps": sample_config.get("number_chain_steps"),
            "flag_useformula": sample_config.get("flag_use_formula"),
        }
        for key, value in legacy_model_params.items():
            if value is not None:
                self.model_sample_params.setdefault(key, value)
        self.metric_dict_sample = sample_config.get("out_dict", None)
        self.flag_retrieval_sampling = sample_config.get(
            "flag_retrieval_sampling", False
        )
        self.flag_retrieval_initialization = sample_config.get(
            "flag_retrieval_initialization", False
        )
        self.num_candidates = int(sample_config.get("num_candidates", 1))
        if self.num_candidates < 1:
            raise ValueError("Sampler.num_candidates must be at least 1.")

        # runtime info
        self.rank = (
            int(paddle.distributed.get_rank())
            if paddle.distributed.is_initialized()
            else 0
        )
        self.output_dir = self.config.get("Sampler", {}).get("output_dir", "./outputs")
        self._set_output_dir(self.output_dir)

        if self.flag_retrieval_sampling or self.flag_retrieval_initialization:
            self.molecular_vectors, self.smiles_list = self._init_retrieval_bank(
                self.sample_config,
            )
        else:
            self.molecular_vectors, self.smiles_list = None, None

        metrics = build_metric(config.get("Metric"))
        if isinstance(metrics, dict):
            self.metric_modules = metrics
        elif metrics is not None:
            self.metric_modules = {"default": metrics}
        else:
            self.metric_modules = {}
        for metric in self.metric_modules.values():
            if (
                hasattr(metric, "sample_metrics")
                and self.metric_dict_sample is not None
            ):
                selected_metrics = self.metric_dict_sample
                if isinstance(selected_metrics, Mapping):
                    selected_metrics = selected_metrics.keys()
                elif isinstance(selected_metrics, str):
                    selected_metrics = [selected_metrics]
                metric.sample_metrics = set(selected_metrics)
            if hasattr(metric, "bind"):
                metric.bind(
                    model=self.model,
                    dataset_infos=dataset_infos,
                    clip=self.clip,
                    num_candidate=self.num_candidates,
                )

    @staticmethod
    def _resolve_package_paths(
        config: Dict,
        config_base_dir: Optional[str],
        checkpoint_dir: Optional[str] = None,
    ):
        def resolve_path(path):
            if path is None or osp.isabs(path) or path.startswith("http"):
                return path

            if checkpoint_dir is not None:
                candidate = osp.join(checkpoint_dir, osp.basename(path))
                if osp.exists(candidate):
                    return candidate

            if config_base_dir is not None:
                if path.startswith("./checkpoints/") or path.startswith("checkpoints/"):
                    candidate = osp.join(
                        config_base_dir, "checkpoints", osp.basename(path)
                    )
                    if osp.exists(candidate):
                        return candidate
                candidate = osp.normpath(osp.join(config_base_dir, path))
                if osp.exists(candidate):
                    return candidate

            return path

        def visit(obj):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if key in {
                        "pretrained_path",
                        "pretrained_model_path",
                        "retrieval_database_path",
                        "path",
                        "datadir",
                    } and isinstance(value, str):
                        resolved_path = resolve_path(value)
                        obj[key] = resolved_path
                    else:
                        visit(value)
            elif isinstance(obj, list):
                for item in obj:
                    visit(item)

        visit(config)

        sampler_config = config.get("Sampler", {})
        retrieval_enabled = sampler_config.get(
            "flag_retrieval_sampling", False
        ) or sampler_config.get("flag_retrieval_initialization", False)
        retrieval_path = sampler_config.get("retrieval_database_path")
        if retrieval_enabled and (
            retrieval_path is None or not osp.isfile(retrieval_path)
        ):
            raise FileNotFoundError(
                "Retrieval sampling requires an existing "
                "Sampler.retrieval_database_path."
            )

    def _set_output_dir(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def _to_rdkit_molecule(self, graph):
        if self.dataset_infos is None:
            raise ValueError("Molecular visualization requires dataset information.")
        return molecular_graph_to_rdkit(
            graph[0],
            graph[1],
            atom_decoder=self.dataset_infos.atom_decoder,
            bond_decoder=self.dataset_infos.vocab["bond"],
        )

    def _save_molecular_visualizations(self, batch_id, result):
        graph_root = Path(self.output_dir) / "graph"
        true_samples = result.get("true", [])
        requested = min(self.visual_num, len(result["pred"]), len(true_samples))
        for index in range(requested):
            predicted = self._to_rdkit_molecule(result["pred"][index])
            reference = self._to_rdkit_molecule(true_samples[index])
            if predicted is None or reference is None:
                continue
            self.molecule_visualizer.save(
                self.molecule_visualizer.render(predicted),
                graph_root / f"batch_{batch_id}_predicted" / f"molecule_{index}.png",
            )
            self.molecule_visualizer.save(
                self.molecule_visualizer.render(reference),
                graph_root / f"batch_{batch_id}_true" / f"molecule_{index}.png",
            )

        chains = result.get("chains", [])[: self.chains_to_save]
        for index, (node_frames, edge_frames) in enumerate(chains):
            molecules = [
                self._to_rdkit_molecule([nodes, edges])
                for nodes, edges in zip(node_frames, edge_frames)
            ]
            molecules = [molecule for molecule in molecules if molecule is not None]
            if not molecules:
                continue
            frames = self.molecule_visualizer.render_trajectory(molecules)
            frame_dir = graph_root / "chain" / f"molecule_{batch_id}_{index}"
            for frame_index, frame in enumerate(frames):
                self.molecule_visualizer.save(
                    frame, frame_dir / f"frame_{frame_index}.png"
                )
            self.animation_writer.save_gif(
                frames,
                graph_root / "chain" / f"molecule_{batch_id}_{index}.gif",
            )
            grid = self.molecule_visualizer.render_grid(
                molecules, molecules_per_row=10, sub_image_size=(200, 200)
            )
            self.molecule_visualizer.save(
                grid, frame_dir / f"molecule_{batch_id}_{index}_grid.png"
            )

    @property
    def _core_model(self):
        if isinstance(self.model, paddle.DataParallel):
            return self.model._layers
        return self.model

    @paddle.no_grad()
    def sample(self, data, sample_params: Optional[Dict[str, Any]] = None):
        """Generate one molecular batch through ``model.sample``.

        Args:
            data: A model-specific input batch.
            sample_params: Optional model sampling arguments. When omitted,
                ``Sampler.model_sample_params`` is used.

        Returns:
            The model-specific value produced by ``model.sample``.
        """

        params = (
            copy.deepcopy(self.model_sample_params)
            if sample_params is None
            else copy.deepcopy(sample_params)
        )
        if not isinstance(params, dict):
            raise TypeError("sample_params must be a dict or None.")

        sample_fn = getattr(self._core_model, "sample", None)
        if not callable(sample_fn):
            raise TypeError(
                f"{type(self._core_model).__name__} does not implement sample()."
            )
        return sample_fn(data, **params)

    def _supports_sample_parameter(self, name: str) -> bool:
        signature = inspect.signature(self._core_model.sample)
        return name in signature.parameters or any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        )

    def compute_metric(
        self,
        save_path=None,
    ):
        if not self.metric_modules:
            raise ValueError("Metric config must be provided to compute metrics.")
        if save_path is not None:
            self._set_output_dir(save_path)
        return self.sample_by_dataloader(
            self.output_dir,
        )

    def sample_by_dataloader(
        self,
        save_path=None,
        data_loader=None,
    ):
        if save_path is not None:
            self._set_output_dir(save_path)
        if data_loader is None:
            data_loader = getattr(self, "_sample_loader", None)
        if data_loader is None:
            data_config = self.sample_config.get("data")
            if data_config is None:
                raise ValueError(
                    "Sampler.data is required for sample_by_dataloader(). "
                    "Use sample(data) when the model does not use a dataloader."
                )
            data_loader = build_dataloader(data_config, vocab=self.vocab)

        logger.info(f"Total iterations: {len(data_loader)}")
        logger.info("Start sampling process...")

        self.model.eval()
        epoch_id = 0

        data_length = len(data_loader)
        logger.message(f"Start to sample ... | Total Batches: {data_length}")
        start = time.time()

        metric_dict = self.sample_epoch(
            data_loader,
            epoch_id,
            keep_onehot=self.flag_retrieval_sampling,
            num_candidates=self.num_candidates,
        )

        if self.rank == 0 and self.metric_modules:
            msg = "Sample:"
            msg += f" | sample_metric cost: {time.time() - start:.5f}s"
            for k, v in metric_dict.items():
                if isinstance(v, paddle.Tensor):
                    v = v.item() if v.numel() == 1 else v.tolist()
                if self.metric_dict_sample is None or k in self.metric_dict_sample:
                    msg += (
                        f" | {k}(metric): {', '.join(f'{x:.5f}' for x in v)}"
                        if isinstance(v, (list, tuple))
                        else f" | {k}(metric): {v:.5f}"
                    )
            logger.info(msg)
        return metric_dict if self.metric_modules else self.last_samples

    @paddle.no_grad()
    def sample_epoch(
        self,
        dataloader: paddle.io.DataLoader,
        epoch_id: int,
        num_candidates: int = 1,
        keep_onehot: bool = False,
    ):
        """Sample one epoch through the common molecular-model interface."""

        self.model.eval()
        for metric in self.metric_modules.values():
            if hasattr(metric, "reset"):
                metric.reset()

        samples: Dict[str, object] = {
            "pred": [],
            "true": [],
            "n_all": 0,
            "node_mask_meta": [],
            "batch_condition": [],
            "dict": None,
        }
        if keep_onehot:
            samples["candidates"] = [[] for _ in range(num_candidates)]
            samples["candidates_X"] = [[] for _ in range(num_candidates)]
            samples["candidates_E"] = [[] for _ in range(num_candidates)]

        for iter_id, batch_data in enumerate(dataloader):
            batch_metadata = None
            for candidate_id in range(num_candidates):
                sample_kwargs = copy.deepcopy(self.model_sample_params)
                sample_context = {
                    "batch_id": iter_id,
                    "iter_idx": candidate_id,
                }
                if keep_onehot:
                    sample_context["return_onehot"] = True
                for key, value in sample_context.items():
                    if self._supports_sample_parameter(key):
                        sample_kwargs[key] = value
                if self.flag_retrieval_initialization:
                    sample_kwargs.update(
                        retrieval_initialization=True,
                        clip=self.clip,
                        molecular_vectors=self.molecular_vectors,
                        smiles_list=self.smiles_list,
                    )
                result = self.sample(batch_data, sample_params=sample_kwargs)
                if not isinstance(result, Mapping):
                    raise TypeError(
                        f"{type(self._core_model).__name__}.sample() must return a "
                        "mapping for dataloader sampling."
                    )
                if "pred" not in result:
                    raise KeyError("model.sample() result must contain 'pred'.")

                if keep_onehot:
                    if "node_onehot" not in result or "edge_onehot" not in result:
                        raise KeyError(
                            "Retrieval sampling requires 'node_onehot' and "
                            "'edge_onehot' in model.sample() output."
                        )
                    samples["candidates"][candidate_id].extend(result["pred"])
                    samples["candidates_X"][candidate_id].extend(result["node_onehot"])
                    samples["candidates_E"][candidate_id].extend(result["edge_onehot"])

                if candidate_id == 0:
                    self._save_molecular_visualizations(iter_id, result)
                    samples["pred"].extend(result["pred"])
                    true_samples = result.get("true")
                    if true_samples is not None:
                        samples["true"].extend(true_samples)
                    batch_metadata = result

            batch_condition = batch_metadata.get("conditions")
            if batch_condition is not None:
                if samples["batch_condition"]:
                    samples["batch_condition"] = [
                        paddle.concat([previous, current], axis=0)
                        for previous, current in zip(
                            samples["batch_condition"], batch_condition
                        )
                    ]
                else:
                    samples["batch_condition"] = batch_condition

            node_counts = batch_metadata.get("node_counts")
            if node_counts is not None:
                samples["node_mask_meta"].extend(node_counts)
            samples["n_all"] += int(
                batch_metadata.get("num_samples", len(batch_metadata["pred"]))
            )
            atom_decoder = batch_metadata.get("atom_decoder")
            if atom_decoder is not None:
                samples["dict"] = atom_decoder
            if (
                self.sample_batch_iters is not None
                and iter_id + 1 >= self.sample_batch_iters
            ):
                break

        metric_result = {
            "samples": samples,
            "epoch_id": epoch_id,
            "local_rank": self.rank,
            "output_dir": self.output_dir,
        }
        metric_dict = {}
        for metric in self.metric_modules.values():
            if hasattr(metric, "update_step"):
                metric.update_step(
                    result=metric_result,
                    batch=None,
                    stage="sample",
                )
            if hasattr(metric, "compute_epoch"):
                metric_dict.update(metric.compute_epoch(stage="sample") or {})
        self.last_samples = samples
        return metric_dict

    def _init_retrieval_bank(self, cfg):
        """
        load the molecular vector library for retrieval initialization/evaluation
        from configuration
        """
        if not cfg:
            return None, None
        path = cfg.get("retrieval_database_path", None)
        if path is None or not os.path.exists(path):
            logger.warning(f"[retrieval_bank] path missing or not found: {path}")
            return None, None

        ext = os.path.splitext(path)[1].lower()
        embs, smiles = None, None
        try:
            if ext == ".csv":
                data = pd.read_csv(path)
                data["molecularRep"] = data["molecularRep"].apply(
                    lambda x: np.fromstring(x.strip("[]"), sep=" ")
                )
                # to paddle tensor
                embs = paddle.to_tensor(
                    np.stack(data["molecularRep"].values), dtype="float32"
                )
                smiles = data["smiles"].tolist()
            else:
                raise ValueError(f"Unsupported retrieval bank ext: {ext}")
        except Exception as e:
            logger.warning(f"[retrieval_bank] load failed: {e}")
            return None, None

        return embs, smiles

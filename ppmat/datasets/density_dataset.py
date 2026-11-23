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


import json
import multiprocessing as mp
import os
import os.path as osp
import pickle
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from typing import Any
from typing import Dict

import numpy as np
import paddle
import paddle.distributed as dist

from ppmat.datasets.build_field import BuildField
from ppmat.datasets.build_grid import BuildGrid
from ppmat.datasets.grid_sampler import DensityGridSampler
from ppmat.models import build_graph_converter
from ppmat.utils import download
from ppmat.utils import logger
from ppmat.utils.io import write_cube
from ppmat.utils.misc import is_equal

_DENSITY_CACHE_FIELD_BUILDER = None
_DENSITY_CACHE_GRAPH_CONVERTER = None
_DENSITY_CACHE_FIELDS_PATH = None
_DENSITY_CACHE_GRAPHS_PATH = None


def _init_density_cache_worker(
    build_field_cfg,
    build_graph_cfg,
    vocab,
    fields_cache_path,
    graph_cache_path,
):
    """Initialize CPU-only converters once in each spawned cache worker."""
    global _DENSITY_CACHE_FIELD_BUILDER
    global _DENSITY_CACHE_GRAPH_CONVERTER
    global _DENSITY_CACHE_FIELDS_PATH
    global _DENSITY_CACHE_GRAPHS_PATH

    _DENSITY_CACHE_FIELD_BUILDER = BuildField(**build_field_cfg)
    _DENSITY_CACHE_GRAPH_CONVERTER = (
        build_graph_converter(build_graph_cfg, vocab=vocab)
        if build_graph_cfg is not None
        else None
    )
    _DENSITY_CACHE_FIELDS_PATH = fields_cache_path
    _DENSITY_CACHE_GRAPHS_PATH = graph_cache_path


def _build_density_cache_sample(index_and_source):
    """Build and serialize one field and its graph in a cache worker."""
    index, field_source = index_and_source
    field = _DENSITY_CACHE_FIELD_BUILDER(field_source)
    with open(osp.join(_DENSITY_CACHE_FIELDS_PATH, f"{index:010d}.pkl"), "wb") as f:
        pickle.dump(field, f)

    if _DENSITY_CACHE_GRAPH_CONVERTER is not None:
        graph = _DENSITY_CACHE_GRAPH_CONVERTER.from_structures([field.structure])[0]
        with open(osp.join(_DENSITY_CACHE_GRAPHS_PATH, f"{index:010d}.pkl"), "wb") as f:
            pickle.dump(graph, f)
    return index


class DensityDataset(paddle.io.Dataset):
    """Density Dataset Handler."""

    # Optional registry fields; can be overridden per config or subclass
    name: str | None = None
    url: str | None = None
    md5: str | None = None
    split_url: str | None = None
    split_md5: str | None = None

    def __init__(
        self,
        path: str | os.PathLike[str],
        split: str,
        vocab,
        build_graph_cfg: Dict[str, Any],
        build_field_cfg: Dict[str, Any] | None = None,
        transforms: Callable | None = None,
        grid_sampler_cfg: Dict[str, Any] | None = None,
        cache_path: str | os.PathLike[str] | None = None,
        cache_num_workers: int = 1,
        overwrite: bool = False,
    ) -> None:
        super().__init__()
        if not osp.exists(path):
            logger.message("The dataset is not found. Will download it now.")
            root_path = download.get_datasets_path_from_url(self.url, self.md5)
            path = osp.join(root_path, self.name, osp.basename(path))

        self.path = path
        self.split = split

        if build_field_cfg is None:
            build_field_cfg = {
                "format": "cube",
                "name": "density",
                "value_unit": "unknown",
                "num_cpus": 1,
            }
            logger.message(
                "The build_field_cfg is not set, will use the default "
                f"configs: {build_field_cfg}"
            )

        self.build_graph_cfg = build_graph_cfg
        self.build_field_cfg = build_field_cfg
        self.transforms = transforms
        if (
            isinstance(cache_num_workers, bool)
            or not isinstance(cache_num_workers, int)
            or cache_num_workers <= 0
        ):
            raise ValueError("cache_num_workers must be a positive integer.")
        self.cache_num_workers = cache_num_workers
        self.grid_sampler = (
            DensityGridSampler(**grid_sampler_cfg)
            if grid_sampler_cfg is not None
            else None
        )

        if cache_path is not None:
            self.cache_path = cache_path
        else:
            self.cache_path = osp.join(
                osp.split(path)[0] + "_cache",
                f"{osp.splitext(osp.basename(path))[0]}_{split}",
            )
        logger.info(f"Cache path: {self.cache_path}")

        # prepare vocab
        self.vocab = vocab
        atom_vocab = vocab["atom"]
        graph_vocab = {"atom": atom_vocab}

        self.overwrite = overwrite
        self.cache_exists = True if osp.exists(self.cache_path) else False
        self.row_data, self.num_samples = self.read_data(path)
        logger.info(f"Load {self.num_samples} samples from {path}")

        if self.cache_exists and not overwrite:
            logger.warning(
                "Cache enabled. If a cache file exists, it will be automatically "
                "read and current settings will be ignored. Please ensure that the "
                "settings used in match your current settings."
            )
            try:
                build_field_cfg_cache = self.load_from_cache(
                    osp.join(self.cache_path, "build_field_cfg.pkl")
                )
                if is_equal(build_field_cfg_cache, build_field_cfg):
                    logger.info(
                        "The cached build_field_cfg configuration matches "
                        "the current settings. Reusing previously generated "
                        "field and graph data to optimize performance."
                    )
                else:
                    logger.warning(
                        "build_field_cfg is different from build_field_cfg_cache. "
                        "Will rebuild the fields and graphs."
                    )
                    logger.warning(
                        "If you want to use the cached fields and graphs, please "
                        "ensure that the settings used in match your current settings."
                    )
                    overwrite = True
            except Exception as e:
                logger.warning(e)
                logger.warning(
                    "Failed to load build_field_cfg.pkl from cache. "
                    "Will rebuild the fields and graphs."
                )
                overwrite = True

            if build_graph_cfg is not None and not overwrite:
                try:
                    build_graph_cfg_cache = self.load_from_cache(
                        osp.join(self.cache_path, "build_graph_cfg.pkl")
                    )
                    graph_vocab_cache = self.load_from_cache(
                        osp.join(self.cache_path, "graph_vocab.pkl")
                    )
                    if is_equal(build_graph_cfg_cache, build_graph_cfg) and is_equal(
                        graph_vocab_cache, graph_vocab
                    ):
                        logger.info(
                            "The cached graph configuration and vocabulary match "
                            "the current settings. Reusing previously generated "
                            "graph data to optimize performance."
                        )
                    else:
                        logger.warning(
                            "Graph configuration or vocabulary differs from the "
                            "cache. Will rebuild the fields and graphs."
                        )
                        logger.warning(
                            "If you want to use the cached fields and graphs, "
                            "please ensure that the settings used in match your "
                            "current settings."
                        )
                        overwrite = True
                except Exception as e:
                    logger.warning(e)
                    logger.warning(
                        "Failed to load build_graph_cfg.pkl from cache. "
                        "Will rebuild the fields and graphs."
                    )
                    overwrite = True
        fields_cache_path = osp.join(self.cache_path, "fields")
        graph_cache_path = osp.join(self.cache_path, "graphs")
        if overwrite or not self.cache_exists:
            # convert fields and graphs
            # only rank 0 process do the conversion
            if dist.get_rank() == 0:
                # save build_field_cfg and build_graph_cfg to cache file
                os.makedirs(self.cache_path, exist_ok=True)
                self.save_to_cache(
                    osp.join(self.cache_path, "build_field_cfg.pkl"),
                    build_field_cfg,
                )
                self.save_to_cache(
                    osp.join(self.cache_path, "build_graph_cfg.pkl"),
                    build_graph_cfg,
                )
                self.save_to_cache(
                    osp.join(self.cache_path, "graph_vocab.pkl"),
                    graph_vocab,
                )
                # Save fields and graphs to cache. Spawned workers avoid
                # inheriting any CUDA state initialized before Dataset setup.
                os.makedirs(fields_cache_path, exist_ok=True)
                if build_graph_cfg is not None:
                    os.makedirs(graph_cache_path, exist_ok=True)
                if self.cache_num_workers == 1:
                    field_builder = BuildField(**build_field_cfg)
                    converter = (
                        build_graph_converter(build_graph_cfg, vocab=vocab)
                        if build_graph_cfg is not None
                        else None
                    )
                    for i, field_source in enumerate(self.row_data["field_source"]):
                        field = field_builder(field_source)
                        self.save_to_cache(
                            osp.join(fields_cache_path, f"{i:010d}.pkl"), field
                        )
                        if converter is not None:
                            graph = converter.from_structures([field.structure])[0]
                            self.save_to_cache(
                                osp.join(graph_cache_path, f"{i:010d}.pkl"), graph
                            )
                else:
                    worker_context = mp.get_context("spawn")
                    with ProcessPoolExecutor(
                        max_workers=self.cache_num_workers,
                        mp_context=worker_context,
                        initializer=_init_density_cache_worker,
                        initargs=(
                            build_field_cfg,
                            build_graph_cfg,
                            vocab,
                            fields_cache_path,
                            graph_cache_path,
                        ),
                    ) as executor:
                        for _ in executor.map(
                            _build_density_cache_sample,
                            enumerate(self.row_data["field_source"]),
                            chunksize=1,
                        ):
                            pass
                logger.info(f"Save {self.num_samples} fields to {fields_cache_path}")
                if build_graph_cfg is not None:
                    logger.info(f"Save {self.num_samples} graphs to {graph_cache_path}")

            # sync all processes
            if dist.is_initialized():
                dist.barrier()
        self.fields = [
            osp.join(fields_cache_path, f"{i:010d}.pkl")
            for i in range(self.num_samples)
        ]
        if build_graph_cfg is not None:
            self.graphs = [
                osp.join(graph_cache_path, f"{i:010d}.pkl")
                for i in range(self.num_samples)
            ]
        else:
            self.graphs = None

        missing_fields = [f for f in self.fields if not osp.exists(f)]
        assert not missing_fields, (
            f"Missing {len(missing_fields)} field cache file(s) under "
            f"{fields_cache_path}, e.g. {missing_fields[0]}"
        )
        if self.graphs is not None:
            missing_graphs = [g for g in self.graphs if not osp.exists(g)]
            assert not missing_graphs, (
                f"Missing {len(missing_graphs)} graph cache file(s) under "
                f"{graph_cache_path}, e.g. {missing_graphs[0]}"
            )

    def read_data(self, path: str):
        """Read the requested split index and resolve its field sources.

        Args:
            path (str): Path to the data.
        """
        with open(path) as file_obj:
            split_data = json.load(file_obj)
        entries = list(split_data[self.split])
        file_names = [str(entry) for entry in entries]
        field_source = [
            osp.join(osp.dirname(path), file_name) for file_name in file_names
        ]
        data = {
            "id": entries,
            "file_name": file_names,
            "field_source": field_source,
        }
        num_samples = len(entries)
        return data, num_samples

    def save_to_cache(self, cache_path: str, data: Any):
        with open(cache_path, "wb") as f:
            pickle.dump(data, f)

    def load_from_cache(self, cache_path: str):
        if osp.exists(cache_path):
            with open(cache_path, "rb") as f:
                data = pickle.load(f)
            return data
        else:
            raise FileNotFoundError(f"No such file or directory: {cache_path}")

    @staticmethod
    def _read_property_data(field):
        return {
            "density": np.asarray(field.flat, dtype=np.float32),
        }

    def __getitem__(self, idx: int):
        """Get item at index idx."""
        data = {}

        if self.graphs is not None:
            graph = self.graphs[idx]
            if isinstance(graph, str):
                graph = self.load_from_cache(graph)
            data["graph"] = graph

        field = self.fields[idx]
        if isinstance(field, str):
            field = self.load_from_cache(field)
        grid = field.grid
        data.update(self._read_property_data(field))
        data["grid_coord"] = np.asarray(field.coordinates(), dtype=np.float32)
        data["info"] = {
            "shape": list(grid.shape),
            "cell": np.asarray(grid.cell_vectors, dtype=np.float32),
            "origin": np.asarray(grid.origin, dtype=np.float32),
            "coordinate_unit": grid.length_unit,
            "density_unit": grid.value_unit,
            "file_name": self.row_data["file_name"][idx],
        }
        data["id"] = self.row_data["id"][idx]
        if self.grid_sampler is not None:
            identity = (
                data["info"].get("source_split", self.split),
                data["id"],
                data["info"].get("file_name"),
            )
            data = self.grid_sampler(data, identity)
        data = self.transforms(data) if self.transforms is not None else data
        return data

    def __len__(self):
        return self.num_samples


MD17_ATOMIC_NUMBERS: Dict[str, np.ndarray] = {
    "benzene": np.array([6, 6, 6, 6, 6, 6, 1, 1, 1, 1, 1, 1], dtype="int64"),
    "ethanol": np.array([6, 6, 8, 1, 1, 1, 1, 1, 1], dtype="int64"),
    "phenol": np.array([6, 6, 6, 6, 6, 6, 8, 1, 1, 1, 1, 1, 1], dtype="int64"),
    "resorcinol": np.array([6, 6, 6, 6, 6, 6, 8, 1, 8, 1, 1, 1, 1, 1], dtype="int64"),
    "ethane": np.array([6, 6, 1, 1, 1, 1, 1, 1], dtype="int64"),
    "malonaldehyde": np.array([8, 6, 6, 6, 8, 1, 1, 1, 1], dtype="int64"),
}


class MD17DensityDataset(DensityDataset):
    """MD17 small-molecule electron-density dataset.

    A thin :class:`DensityDataset` subclass. The MD17 release stores each field
    as half-space packed FFT coefficients plus Cartesian structures in two
    ``.npy`` arrays under ``<molecule>_<train|test>/``. ``read_data`` inverts the
    FFT and materializes one CUBE file per sample (only on first use); the base
    class then caches the decoded field and radius graph exactly like
    QM9/MP/OMol25.

    The source release only provides train and test directories; ``validation``
    reads the same source samples as ``test``.

    Args:
        path: Source directory ``<molecule>_<train|test>`` (e.g.
            ``./data/data_md/ethanol/ethanol_train``). Downloaded from ``url``
            when missing; the molecule and physical split are inferred from the
            directory name.
        split: ``train``, ``validation``, or ``test``.
        n_grid: Cube grid resolution per axis (even, > 1).
        grid_size: Physical box size (bohr) for the grid.
        vocab: Registered vocabularies; ``atom.atomic_number_to_id`` maps the
            molecule's atomic numbers to embedding ids.
        build_graph_cfg: Registered graph converter configuration.
        build_field_cfg: Only ``name``/``num_cpus`` are honored; MD17 always reads
            its own materialized CUBE files (``format='cube'``).
        cache_path: Optional cache directory. Defaults beside ``path`` and
            includes the grid geometry.
        overwrite: Force rebuilding cache even if it exists.
        transforms: Optional callable applied to each sample.
        grid_sampler_cfg: Optional :class:`DensityGridSampler` configuration.
    """

    name = "md17_es"
    url = "https://paddle-org.bj.bcebos.com/paddlematerials/datasets/MD17_ES/md17_es.tar.gz"
    md5 = None

    def __init__(
        self,
        path: str | os.PathLike[str] = "./data/data_md/ethanol/ethanol_train",
        split: str = "train",
        n_grid: int = 50,
        grid_size: float = 20.0,
        *,
        vocab,
        build_graph_cfg: Dict[str, Any],
        build_field_cfg: Dict[str, Any] | None = None,
        cache_path: str | os.PathLike[str] | None = None,
        overwrite: bool = False,
        transforms: Callable | None = None,
        grid_sampler_cfg: Dict[str, Any] | None = None,
    ) -> None:
        if split not in {"train", "validation", "test"}:
            raise ValueError("split must be one of ['train', 'validation', 'test']")
        if isinstance(n_grid, bool) or not isinstance(n_grid, (int, np.integer)):
            raise TypeError("n_grid must be an integer.")
        if n_grid < 2 or n_grid % 2 != 0:
            raise ValueError("n_grid must be an even integer greater than 1.")
        if not np.isfinite(grid_size) or grid_size <= 0:
            raise ValueError("grid_size must be a positive finite number.")
        if not isinstance(build_graph_cfg, dict):
            raise TypeError("build_graph_cfg must be a dictionary.")
        if transforms is not None and not callable(transforms):
            raise TypeError("transforms must be callable or None.")

        n_grid = int(n_grid)
        grid_size = float(grid_size)
        grid_step = grid_size / n_grid

        # MD17 always reads its own materialized CUBE files; honor only the
        # consumer-controlled keys from build_field_cfg.
        cube_field_cfg = {"format": "cube", "name": "density", "num_cpus": 1}
        if build_field_cfg is not None:
            if not isinstance(build_field_cfg, dict):
                raise TypeError("build_field_cfg must be a dictionary or None.")
            cube_field_cfg.update(build_field_cfg)
        cube_field_cfg["format"] = "cube"

        # Resolve the source directory (MD17 lays out <molecule>/<molecule>_<split>).
        path = osp.abspath(osp.normpath(osp.expanduser(os.fspath(path))))
        requested_tail = path.split(osp.sep)[-2:]
        if not osp.exists(path):
            logger.message(
                f"Dataset path {path} not found. Downloading {self.name} "
                f"from {self.url}."
            )
            root_path = download.get_datasets_path_from_url(self.url, self.md5)
            path = osp.abspath(osp.join(root_path, *requested_tail))
        if not osp.isdir(path):
            raise NotADirectoryError(f"Dataset path {path} is not a directory.")

        source_dir = osp.basename(path)
        try:
            mol_name, source_split = source_dir.rsplit("_", 1)
        except ValueError as error:
            raise ValueError(
                "MD17 path must end in '<molecule>_<train|test>', "
                f"but got {source_dir!r}."
            ) from error
        if source_split not in {"train", "test"}:
            raise ValueError(
                "MD17 path must identify a train or test source directory, "
                f"but got {source_dir!r}."
            )
        expected_source_split = "test" if split == "validation" else split
        if source_split != expected_source_split:
            raise ValueError(
                f"Dataset split {split!r} requires an MD17 "
                f"{expected_source_split!r} source path, but got {source_dir!r}."
            )
        if mol_name not in MD17_ATOMIC_NUMBERS:
            raise ValueError(
                f"Unsupported molecule {mol_name}. "
                f"Options: {list(MD17_ATOMIC_NUMBERS)}"
            )

        atom_number_to_id = vocab["atom"]["atomic_number_to_id"]
        atom_numbers = MD17_ATOMIC_NUMBERS[mol_name]
        try:
            atom_types = np.asarray(
                [atom_number_to_id[int(z)] for z in atom_numbers], dtype=np.int64
            )
        except KeyError as error:
            raise KeyError(
                f"Atomic number {error} for molecule {mol_name} is missing from "
                "the atom vocabulary."
            ) from error

        # Attributes consumed by read_data; set before the base initializer runs it.
        self.n_grid = n_grid
        self.grid_size = grid_size
        self.mol_name = mol_name
        self.source_split = source_split
        self._atom_numbers = np.asarray(atom_numbers, dtype=np.int64)
        self._atom_types = atom_types
        self._grid_data = BuildGrid(format="array", coordinate_unit="bohr")(
            {
                "shape": (n_grid, n_grid, n_grid),
                "voxel_vectors": np.eye(3, dtype=np.float32) * grid_step,
                "origin": np.full(3, grid_step, dtype=np.float32),
            }
        )
        self._cube_dir = f"{path}_n{n_grid}_g{grid_size:g}_cubes"

        if cache_path is None:
            cache_path = f"{path}_n{n_grid}_g{grid_size:g}_cache"

        DensityDataset.__init__(
            self,
            path=path,
            split=split,
            vocab=vocab,
            build_graph_cfg=build_graph_cfg,
            build_field_cfg=cube_field_cfg,
            transforms=transforms,
            grid_sampler_cfg=grid_sampler_cfg,
            cache_path=cache_path,
            overwrite=overwrite,
        )

    def read_data(self, path: str | os.PathLike[str]):
        """Materialize MD17 FFT densities into per-sample CUBE files.

        Returns the base-class contract ``{"id", "file_name", "field_source"}``
        where each ``field_source`` is a CUBE path that
        ``BuildField(format="cube")`` reads single-argument. FFT inversion runs
        only when the CUBE files are absent (first use); later runs list the
        existing CUBE files without re-decoding.
        """
        structures_path = osp.join(path, "structures.npy")
        density_path = osp.join(path, "dft_densities.npy")
        if not osp.exists(structures_path) or not osp.exists(density_path):
            raise FileNotFoundError(
                f"Cannot locate expected files under {path}. "
                "Expected structures.npy and dft_densities.npy."
            )
        structures = np.load(structures_path, mmap_mode="r")
        densities_fft = np.load(density_path, mmap_mode="r")
        num_samples = int(structures.shape[0])
        shape = (self.n_grid, self.n_grid, self.n_grid)

        cube_paths = [
            osp.join(self._cube_dir, f"{i:06d}.cube") for i in range(num_samples)
        ]
        rank = dist.get_rank() if dist.is_initialized() else 0
        if not all(osp.exists(cube) for cube in cube_paths):
            if rank == 0:
                os.makedirs(self._cube_dir, exist_ok=True)
                atom_numbers = self._atom_numbers
                logger.message(
                    f"Materializing {num_samples} MD17 density CUBE files under "
                    f"{self._cube_dir} (one-time, n_grid={self.n_grid}) ..."
                )
                for i in range(num_samples):
                    real_space = MD17DensityDataset.invert_fft(
                        np.asarray(densities_fft[i], dtype=np.float32), shape
                    )
                    atom_coord = np.asarray(structures[i], dtype=np.float32)
                    write_cube(
                        cube_paths[i],
                        atom_numbers,
                        atom_coord,
                        real_space,
                        self._grid_data,
                    )
        if dist.is_initialized():
            dist.barrier()

        file_names = [
            f"{self.mol_name}_{self.source_split}_{i:06d}" for i in range(num_samples)
        ]
        data = {
            "id": list(range(num_samples)),
            "file_name": file_names,
            "field_source": cube_paths,
        }
        return data, num_samples

    @staticmethod
    def invert_fft(fft_coeff, shape):
        """Invert half-space packed FFT coefficients into a real-space field.

        The MD17 release stores each field as real coefficients of a half-space
        packed transform; every axis is folded back before the inverse transform.
        ``shape`` must be a cubic grid with an even edge length.
        """
        shape = tuple(int(size) for size in shape)
        if len(shape) != 3 or len(set(shape)) != 1 or shape[0] % 2 != 0:
            raise ValueError(
                "fft coefficients require a cubic grid with an even edge "
                f"length, but got shape {shape}."
            )
        num_values = int(np.prod(shape))
        values = np.asarray(fft_coeff, dtype=np.float32).reshape(-1)
        if values.size != num_values:
            raise ValueError(
                f"fft coefficients must hold {num_values} values for grid "
                f"{shape}, but got {values.size}."
            )
        half = shape[0] // 2
        data = values.astype(np.complex64).reshape(1, *shape, order="C")
        for axis in (1, 2, 3):
            front = [slice(None)] * 4
            back = [slice(None)] * 4
            mirror = [slice(None)] * 4
            front[axis] = slice(None, half)
            back[axis] = slice(half, None)
            mirror[axis] = slice(1, half + 1)
            data[tuple(front)] = (data[tuple(front)] - data[tuple(back)] * 1.0j) / 2
            data[tuple(back)] = np.flip(data[tuple(mirror)], axis=axis).conj()
            data = np.fft.ifft(data, axis=axis).astype(np.complex64)
        return np.ascontiguousarray(
            np.flip(data.real.reshape(num_values, order="C"), axis=-1)
        )


class QM9DensityDataset(DensityDataset):
    """QM9 electron-density release used by field prediction models.

    Args:
        path: Path to the split manifest. The published dataset is downloaded
            automatically when this path is missing.
        split: One of ``train``, ``validation``, or ``test``.
        vocab: Vocabularies used to encode atom types.
        **kwargs: Cache and transform options accepted by
            :class:`DensityDataset`.
    """

    name = "qm9_es"
    url = "https://paddle-org.bj.bcebos.com/paddlematerials/datasets/QM9_ES/qm9_es.tar"
    # BCE does not currently publish a content MD5 for this archive.
    md5 = None
    split_url = "https://paddle-org.bj.bcebos.com/paddlematerials/datasets/QM9_ES/qm9_data_split.json"
    split_md5 = "a7547fe43cc1b36348bbd4c8d1a18b2a"
    split_filename = "qm9_data_split.json"

    def read_data(self, path: str | os.PathLike[str]):
        with open(path) as file_obj:
            split_data = json.load(file_obj)
        entries = list(split_data[self.split])
        file_names = [f"{int(entry) + 1:06d}.CHGCAR.lz4" for entry in entries]
        data = {
            "id": entries,
            "file_name": file_names,
            "field_source": [
                osp.join(osp.dirname(path), file_name) for file_name in file_names
            ],
        }
        return data, len(entries)

    def __init__(
        self,
        path: str | os.PathLike[str] = "./data/qm9_data_split.json",
        split: str = "train",
        *,
        vocab,
        build_field_cfg: Dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        if build_field_cfg is None:
            build_field_cfg = {
                "format": "chgcar",
                "name": "density",
                "value_unit": "electron/angstrom^3",
                "num_cpus": 1,
            }
        super().__init__(
            path=path,
            split=split,
            vocab=vocab,
            build_field_cfg=build_field_cfg,
            **kwargs,
        )


class MPCubicDensityDataset(DensityDataset):
    """Materials Project cubic electron-density release.

    Args:
        path: Path to the split manifest. The published dataset is downloaded
            automatically when this path is missing.
        split: One of ``train``, ``validation``, or ``test``.
        vocab: Vocabularies used to encode atom types.
        **kwargs: Cache and transform options accepted by
            :class:`DensityDataset`.
    """

    name = "mp_es_cubic"
    url = "https://paddle-org.bj.bcebos.com/paddlematerials/datasets/MP_ES/mp_es.tar"
    # BCE does not currently publish a content MD5 for this archive.
    md5 = None
    split_url = "https://paddle-org.bj.bcebos.com/paddlematerials/datasets/MP_ES/crystal_data_split.json"
    split_md5 = "3f19bd6bce7d4f10ace1f80f202b1aa5"
    split_filename = "crystal_data_split.json"

    def read_data(self, path: str | os.PathLike[str]):
        with open(path) as file_obj:
            split_data = json.load(file_obj)
        entries = list(split_data[self.split])
        file_names = [f"{entry}.json.xz" for entry in entries]
        data = {
            "id": entries,
            "file_name": file_names,
            "field_source": [
                osp.join(osp.dirname(path), file_name) for file_name in file_names
            ],
        }
        return data, len(entries)

    def __init__(
        self,
        path: str | os.PathLike[str] = "./data/crystal_data_split.json",
        split: str = "train",
        *,
        vocab,
        build_field_cfg: Dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        if build_field_cfg is None:
            build_field_cfg = {
                "format": "json",
                "name": "density",
                "value_unit": "electron/angstrom^3",
                "num_cpus": 1,
            }
        super().__init__(
            path=path,
            split=split,
            vocab=vocab,
            build_field_cfg=build_field_cfg,
            **kwargs,
        )


class OMol25MC5kDensityDataset(DensityDataset):
    """OMol25 metal-complex electron-density release with 4,929 samples.

    Args:
        path: Path to the split manifest. The published dataset is downloaded
            automatically when this path is missing.
        split: One of ``train``, ``validation``, or ``test``.
        vocab: Vocabularies used to encode atom types.
        **kwargs: Cache and transform options accepted by
            :class:`DensityDataset`.
    """

    name = "dataset_OMol25_MC_5k"
    url = "https://paddle-org.bj.bcebos.com/paddlematerials/datasets/OMol25_ES/MC_5k/omol25_mc_5k.tar"
    # BCE does not currently publish a content MD5 for this archive.
    md5 = None
    split_md5 = "4fbc298ab48e34ee44c112567b69a927"
    split_filename = "omol25_data_split.json"

    def read_data(self, path: str | os.PathLike[str]):
        with open(path) as file_obj:
            split_data = json.load(file_obj)
        entries = list(split_data[self.split])
        file_names = [f"{int(entry):06d}.cube.lz4" for entry in entries]
        data = {
            "id": entries,
            "file_name": file_names,
            "field_source": [
                osp.join(osp.dirname(path), file_name) for file_name in file_names
            ],
        }
        return data, len(entries)

    def __init__(
        self,
        path: str | os.PathLike[str] = "./data/omol25_data_split.json",
        split: str = "train",
        *,
        vocab,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            path=path,
            split=split,
            vocab=vocab,
            **kwargs,
        )


class OMol25MC5kTrimmedDensityDataset(OMol25MC5kDensityDataset):
    """Filtered OMol25 MC 5k split used by the InfGCN configuration.

    The raw archive is shared with :class:`OMol25MC5kDensityDataset`; only the
    split manifest differs.
    """

    name = "dataset_OMol25_MC_5k"
    split_filename = "omol25_mc_5k_trimmed_split.json"
    split_url = "https://paddle-org.bj.bcebos.com/paddlematerials/datasets/OMol25/omol25_mc_5k_trimmed_split.json"
    split_md5 = "03f7c71cf9ed448ee476c6ce47042bea"

    def __init__(
        self,
        path: str | os.PathLike[str] = ("./data/omol25_mc_5k_trimmed_split.json"),
        split: str = "train",
        *,
        vocab,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            path=path,
            split=split,
            vocab=vocab,
            **kwargs,
        )

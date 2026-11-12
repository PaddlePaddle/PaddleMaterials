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


from __future__ import annotations

import math
import os
import os.path as osp
import pickle
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional

import numpy as np
import paddle.distributed as dist
from ase import Atoms
from paddle.io import Dataset
from pymatgen.io.ase import AseAtomsAdaptor

from ppmat.datasets.build_structure import BuildStructure
from ppmat.datasets.cache_fingerprint import cache_matches_data
from ppmat.datasets.cache_fingerprint import compute_data_fingerprint
from ppmat.datasets.cache_fingerprint import mark_cache_rebuilding
from ppmat.datasets.cache_fingerprint import save_data_fingerprint
from ppmat.datasets.custom_data_type import ConcatData
from ppmat.models import build_graph_converter
from ppmat.utils import download
from ppmat.utils import logger
from ppmat.utils.misc import is_equal


class TmqmDataset(Dataset):
    """tmQM Dataset Handler.

    The tmQM dataset is a comprehensive quantum chemistry database
    for transition metal complexes, containing multiple complementary
    data files: .xyz files provide molecular geometry structures optimized
    at the GFN2-xTB level; .csv files include SMILES molecular representations
    as well as quantum chemical properties calculated at the TPSSh/def2SVP level,
    including electronic energy, dispersion energy, dipole moment, metal charge,
    HOMO/LUMO energy gap and energy, and polarizability; .q files provide the
    natural atomic charge distribution calculated at the TPSSh/def2SVP level;
    .BO files contain Wiberg bond orders and atomic valence indices calculated
    at the GFN2-xTB level (with the exception of polarizability, which is
    calculated separately at the GFN2-xTB level).

    For model utility, we provide the option: "Whether to use charge and chemical
    bond data" If you want to use it, the settings are as follows:
    use use_atomic_charge: True
    use_chemical_bonding: True

    Args:
    path (str): File path to the dataset file.

    Electronic_E_key (Optional[str], optional):
        Electronic energy of the system in atomic units.
        Defaults to "Electronic_E".

    Dispersion_E_key (Optional[str], optional):
        Dispersion correction energy in atomic units.
        Defaults to "Dispersion_E".

    Dipole_M_key (Optional[str], optional):
        Dipole moment vector in atomic units.
        Defaults to "Dipole_M".

    Metal_q_key (Optional[str], optional):
        Partial charge on the metal atom in atomic units.
        Defaults to "Metal_q".

    HL_Gap_key (Optional[str], optional):
        HOMO-LUMO gap energy in electron volts (eV).
        Defaults to "HL_Gap".

    HOMO_Energy_key (Optional[str], optional):
        Highest Occupied Molecular Orbital energy in electron volts (eV).
        Defaults to "HOMO_Energy".

    LUMO_Energy_key (Optional[str], optional):
        Lowest Unoccupied Molecular Orbital energy in electron volts (eV).
        Defaults to "LUMO_Energy".

    Polarizability_key (Optional[str], optional):
        Isotropic polarizability in atomic units.
        Defaults to "Polarizability".

    SMILES_key (Optional[str], optional):
        Simplified Molecular Input Line Entry System representation
        of the molecule. Defaults to "SMILES".

    build_structure_cfg (Dict, optional):
        The configs for building the structure.Defaults to None.

    build_graph_cfg (Dict, optional):
        The configs for building the graph.Defaults to None.

    transforms (Optional[Callable], optional):
        The preprocess transforms for each sample.Defaults to None.

    cache_path (Optional[str], optional):
        If a cache_path is set, structures and graph will be read directly
        from this path; if the cache does not exist, the converted structures
        and graph will be saved to this path. Defaults to None.

    overwrite (bool, optional):
        Overwrite the existing cache file at the given path if it already exists.
            Defaults to False.
    filter_unvalid (bool, optional):
        Whether to filter out invalid samples. Defaults to True.
    use_atomic_charge (bool, optional):
        Whether to use atomic charge information. Defaults to True.
    use_chemical_bonding (bool, optional):
        Whether to use chemical bonding information. Defaults to True.
    """

    name = "tmqm_train_108k"
    url = "https://paddle-org.bj.bcebos.com/paddlematerials/datasets/tmQM/tmQM.zip"
    md5 = "292f42fbabcd19ba08e9a2878d7f5995"

    # Optional attachment information
    url_charge = "https://paddle-org.bj.bcebos.com/paddlematerials/datasets/tmQM/tmQM_X.q.gz"  # noqa
    md5_charge = "501c42f8479c740ffbe0b0ebe25c2ab8"
    url_bond = "https://paddle-org.bj.bcebos.com/paddlematerials/datasets/tmQM/tmQM_X.BO.gz"  # noqa
    md5_bond = "b8892367c50942b45332e0550be2ba9f"

    def __init__(
        self,
        path: str,
        path_charge: str = None,
        path_bond: str = None,
        electronic_e_key: Optional[str] = None,
        dispersion_e_key: Optional[str] = None,
        dipole_m_key: Optional[str] = None,
        metal_q_key: Optional[str] = None,
        hl_gap_key: Optional[str] = None,
        homo_energy_key: Optional[str] = None,
        lumo_energy_key: Optional[str] = None,
        polarizability_key: Optional[str] = None,
        smiles_key: Optional[str] = None,
        build_structure_cfg: Dict = None,
        build_graph_cfg: Dict = None,
        transforms: Optional[Callable] = None,
        cache_path: Optional[str] = None,
        cache_path_charge: Optional[str] = None,
        cache_path_bond: Optional[str] = None,
        overwrite: bool = False,
        filter_unvalid: bool = True,
        # Whether to use additional information for training
        use_atomic_charge: bool = True,
        use_chemical_bonding: bool = True,
        **kwargs,  # for compatibility
    ):
        super().__init__()

        self.use_atomic_charge = use_atomic_charge
        self.use_chemical_bonding = use_chemical_bonding

        if not osp.exists(path):
            logger.message("The dataset is not found. Will download it now.")
            root_path = download.get_datasets_path_from_url(self.url, self.md5)
            if root_path.endswith("/tmQM/tmQM"):
                root_path = root_path[:-4]
            # /home/aistudio/.paddlemat/datasets/tmQM/tmQM.xyz
            path = osp.join(root_path, osp.basename(path))

        if use_atomic_charge and path_charge is not None:
            logger.message("Use atomic charge dataset. Will download it now.")
            root_path_charge = download.get_datasets_path_from_url(
                self.url_charge, self.md5_charge
            )
            path_charge = osp.join(root_path_charge, "tmQM_X.q")

        if use_chemical_bonding and path_bond is not None:
            logger.message("Use chemical bonding dataset. Will download it now.")
            root_path_bond = download.get_datasets_path_from_url(
                self.url_bond, self.md5_bond
            )
            path_bond = osp.join(root_path_bond, "tmQM_X.BO")

        # Add decompression logic for gz
        if use_atomic_charge and path_charge is not None:
            logger.message("Use atomic charge dataset. Will download it now.")
            root_path_charge = download.get_datasets_path_from_url(
                self.url_charge, self.md5_charge
            )

            gz_file = root_path_charge
            decompressed_file = osp.join(osp.dirname(gz_file), "tmQM_X.q")

            if gz_file.endswith(".gz") and not osp.exists(decompressed_file):
                logger.message(f"Decompressing {gz_file}...")
                import gzip

                with gzip.open(gz_file, "rb") as f_in:
                    with open(decompressed_file, "wb") as f_out:
                        f_out.write(f_in.read())

            path_charge = decompressed_file

        if use_chemical_bonding and path_bond is not None:
            logger.message("Use chemical bonding dataset. Will download it now.")
            root_path_bond = download.get_datasets_path_from_url(
                self.url_bond, self.md5_bond
            )

            gz_file = root_path_bond
            decompressed_file = osp.join(osp.dirname(gz_file), "tmQM_X.BO")

            if gz_file.endswith(".gz") and not osp.exists(decompressed_file):
                logger.message(f"Decompressing {gz_file}...")
                import gzip

                with gzip.open(gz_file, "rb") as f_in:
                    with open(decompressed_file, "wb") as f_out:
                        f_out.write(f_in.read())

            path_bond = decompressed_file

        self.path = path
        self.path_charge = path_charge
        self.path_bond = path_bond
        self.electronic_e_key = electronic_e_key
        self.dispersion_e_key = dispersion_e_key
        self.dipole_m_key = dipole_m_key
        self.metal_q_key = metal_q_key
        self.hl_gap_key = hl_gap_key
        self.homo_energy_key = homo_energy_key
        self.lumo_energy_key = lumo_energy_key
        self.polarizability_key = polarizability_key
        self.smiles_key = smiles_key

        self.property_names = []
        if electronic_e_key is not None:
            self.property_names.append(electronic_e_key)
        if dispersion_e_key is not None:
            self.property_names.append(dispersion_e_key)
        if dipole_m_key is not None:
            self.property_names.append(dipole_m_key)
        if metal_q_key is not None:
            self.property_names.append(metal_q_key)
        if hl_gap_key is not None:
            self.property_names.append(hl_gap_key)
        if homo_energy_key is not None:
            self.property_names.append(homo_energy_key)
        if lumo_energy_key is not None:
            self.property_names.append(lumo_energy_key)
        if polarizability_key is not None:
            self.property_names.append(polarizability_key)
        if smiles_key is not None:
            self.property_names.append(smiles_key)

        if build_structure_cfg is None:
            build_structure_cfg = {
                "format": "ase_atoms",
                "primitive": False,
                "niggli": False,
                "num_cpus": 1,
            }
            logger.message(
                "The build_structure_cfg is not set, will use the default "
                f"configs: {build_structure_cfg}"
            )

        self.build_structure_cfg = build_structure_cfg
        self.build_graph_cfg = build_graph_cfg
        self.transforms = transforms

        if cache_path is not None:
            self.cache_path = cache_path
        else:
            # for example:
            # path = ./data/tmqm_train_108k/tmQM.xyz
            # cache_path= ./data/tmqm_train_108k_cache/tmQM
            self.cache_path = osp.join(
                osp.split(path)[0] + "_cache", osp.splitext(osp.basename(path))[0]
            )
        logger.info(f"Cache path: {self.cache_path}")

        if use_atomic_charge and path_charge is not None:
            if cache_path_charge is not None:
                self.cache_path_charge = cache_path_charge
            else:
                self.cache_path_charge = osp.join(
                    osp.split(path)[0] + "_cache",
                    osp.splitext(osp.basename(path_charge))[0],
                )
            logger.info(f"Cache path of charge: {self.cache_path_charge}")

        if use_chemical_bonding and path_bond is not None:
            if cache_path_bond is not None:
                self.cache_path_bond = cache_path_bond
            else:
                self.cache_path_bond = osp.join(
                    osp.split(path)[0] + "_cache",
                    osp.splitext(osp.basename(path_bond))[0],
                )
            logger.info(f"Cache path of bond: {self.cache_path_bond}")

        self.overwrite = overwrite
        self.filter_unvalid = filter_unvalid

        # Initializing the cache of the path has a flag
        self.cache_exists = True if osp.exists(self.cache_path) else False

        # Read master data
        self.row_data, self.num_samples = self.read_data(path)
        logger.info(f"Load {self.num_samples} samples from {path}")
        self.property_data = self.read_property_data(self.row_data)

        # Read supplemental data
        self.charge_data = None
        self.bond_data = None

        if self.use_atomic_charge and path_charge is not None:
            self.cache_exists_charge = osp.exists(self.cache_path_charge)
            self.charge_data = self.read_charge_data(path_charge)
            logger.info(f"Load charge data from {path_charge}")

        if self.use_chemical_bonding and path_bond is not None:
            self.cache_exists_bond = osp.exists(self.cache_path_bond)
            self.bond_data = self.read_bond_data(path_bond)
            logger.info(f"Load bond data from {path_bond}")

        structure_cache_path = osp.join(self.cache_path, "structures")
        graph_cache_path = osp.join(self.cache_path, "graphs")

        # Check the cache configuration
        overwrite_orig = self._check_cache_config(
            self.cache_path, build_structure_cfg, build_graph_cfg, overwrite
        )

        if self.cache_exists and not overwrite_orig:
            missing_files = self._check_cache_integrity(
                structure_cache_path, graph_cache_path
            )
            if missing_files:
                logger.warning("Found missing cache files, will regenerate cache")
                overwrite_orig = True

        # The build settings alone do not identify the data: also check that the
        # cache was built from the current input before reusing it.
        data_fingerprint = compute_data_fingerprint(
            self.row_data, extra=(self.charge_data, self.bond_data)
        )
        if self.cache_exists and not overwrite_orig:
            if not cache_matches_data(
                self.cache_path,
                data_fingerprint,
                ["structures"] + (["graphs"] if build_graph_cfg is not None else []),
            ):
                overwrite_orig = True
        if overwrite_orig or not self.cache_exists:

            if dist.get_rank() == 0:

                os.makedirs(self.cache_path, exist_ok=True)
                mark_cache_rebuilding(self.cache_path)
                self.save_to_cache(
                    osp.join(self.cache_path, "build_structure_cfg.pkl"),
                    build_structure_cfg,
                )
                self.save_to_cache(
                    osp.join(self.cache_path, "build_graph_cfg.pkl"), build_graph_cfg
                )

                structures = BuildStructure(**build_structure_cfg)(self.row_data)

                if len(structures) > 0:
                    sample_structure = structures[0]
                    logger.info(f"Structure object type: {type(sample_structure)}")
                    logger.info(f"Structure object attributes: {dir(sample_structure)}")
                # If you use supplementary data, incorporate it into the structure
                if self.use_atomic_charge and self.charge_data is not None:
                    structures = self._integrate_charge_data(
                        structures, self.charge_data
                    )  # noqa

                if self.use_chemical_bonding and self.bond_data is not None:
                    structures = self._integrate_bond_data(structures, self.bond_data)

                # Save the structure to the cache file
                os.makedirs(structure_cache_path, exist_ok=True)
                for i in range(self.num_samples):
                    self.save_to_cache(
                        osp.join(structure_cache_path, f"{i:010d}.pkl"),
                        structures[i],
                    )
                logger.info(
                    f"Save {self.num_samples} structures to {structure_cache_path}"
                )

                if build_graph_cfg is not None:
                    converter = build_graph_converter(build_graph_cfg)
                    graphs = converter(structures)
                    # Save the diagram to a cache file
                    os.makedirs(graph_cache_path, exist_ok=True)
                    for i in range(self.num_samples):
                        self.save_to_cache(
                            osp.join(graph_cache_path, f"{i:010d}.pkl"), graphs[i]
                        )
                    logger.info(f"Save {self.num_samples} graphs to {graph_cache_path}")

            # Synchronize all processes
            if dist.is_initialized():
                dist.barrier()
        if dist.get_rank() == 0:
            save_data_fingerprint(self.cache_path, data_fingerprint)

        self.structures = [
            osp.join(structure_cache_path, f"{i:010d}.pkl")
            for i in range(self.num_samples)
        ]

        if build_graph_cfg is not None:
            self.graphs = [
                osp.join(graph_cache_path, f"{i:010d}.pkl")
                for i in range(self.num_samples)
            ]
        else:
            self.graphs = None

        assert (
            len(self.structures) == self.num_samples
        ), "The number of structures must be equal to the number of samples."
        assert (
            self.graphs is None or len(self.graphs) == self.num_samples
        ), "The number of graphs must be equal to the number of samples."

        # Filter invalid samples based on attribute data
        if filter_unvalid:
            self.filter_unvalid_by_property()

    def _check_cache_config(
        self, cache_path, build_structure_cfg, build_graph_cfg, overwrite
    ):

        local_overwrite = overwrite
        cache_exists = osp.exists(cache_path)

        if cache_exists and not local_overwrite:
            logger.warning(
                f"Cache enabled for {cache_path}. If a cache file exists, "
                "it will be automatically read and current settings will be ignored. "
                "Please ensure that the settings used in match your current settings."
            )
            try:
                build_structure_cfg_cache = self.load_from_cache(
                    osp.join(cache_path, "build_structure_cfg.pkl")
                )
                if is_equal(build_structure_cfg_cache, build_structure_cfg):
                    logger.info(
                        "The cached build_structure_cfg configuration matches "
                        "the current settings. Reusing previously generated"
                        " structural data to optimize performance."
                    )
                else:
                    logger.warning(
                        "build_structure_cfg is different from "
                        "build_structure_cfg_cache. Will rebuild the structures and "
                        "graphs."
                    )
                    local_overwrite = True
            except Exception as e:
                logger.warning(e)
                logger.warning(
                    "Failed to load builded_structure_cfg.pkl from cache. "
                    "Will rebuild the structures and graphs(if need)."
                )
                local_overwrite = True

            if build_graph_cfg is not None and not local_overwrite:
                try:
                    build_graph_cfg_cache = self.load_from_cache(
                        osp.join(cache_path, "build_graph_cfg.pkl")
                    )
                    if is_equal(build_graph_cfg_cache, build_graph_cfg):
                        logger.info(
                            "The cached build_graph_cfg configuration "
                            "matches the current settings."
                        )
                    else:
                        logger.warning(
                            "build_graph_cfg is different from build_graph_cfg_cache"
                            ". Will rebuild the graphs."
                        )
                        local_overwrite = True
                except Exception as e:
                    logger.warning(e)
                    logger.warning(
                        "Failed to load builded_graph_cfg.pkl from cache. "
                        "Will rebuild the graphs."
                    )
                    local_overwrite = True

        return local_overwrite

    def read_charge_data(self, path_charge: str):
        # Read atomic charge properties
        charge_data = []
        try:
            with open(path_charge, "r") as f:
                current_sample_charges = []
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    if line.startswith("CSD_code ="):

                        if current_sample_charges:
                            charge_data.append(current_sample_charges)
                            current_sample_charges = []
                        continue

                    if line.startswith("Total charge") or not line:
                        continue

                    # Analyze atomic charge row: Element symbol + charge value
                    parts = line.split()
                    if len(parts) >= 2:
                        try:

                            charge = float(parts[1])
                            current_sample_charges.append(charge)
                        except ValueError:

                            continue

                if current_sample_charges:
                    charge_data.append(current_sample_charges)

            logger.info(f"Loaded charge data for {len(charge_data)} samples")

            # Verify data consistency
            if len(charge_data) != self.num_samples:
                logger.warning(
                    f"Charge data samples ({len(charge_data)}) don't match "
                    f"main data samples ({self.num_samples})"
                )

        except Exception as e:
            logger.warning(f"Error reading charge data from {path_charge}: {e}")
            return None

        return charge_data

    def read_bond_data(self, path_bond: str):
        # Read key-level data
        bond_data = []
        try:
            with open(path_bond, "r") as f:
                current_sample_bonds = []
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    if line.startswith("CSD_code ="):

                        if current_sample_bonds:
                            bond_data.append(current_sample_bonds)
                            current_sample_bonds = []
                        continue

                    parts = line.split()
                    if len(parts) >= 4:
                        try:
                            atom_idx = int(parts[0]) - 1
                            element = parts[1]
                            total_bond_order = float(parts[2])

                            neighbors = []
                            i = 3
                            while i < len(parts):
                                if i + 2 < len(parts):
                                    neighbor_element = parts[i]
                                    neighbor_idx = int(parts[i + 1]) - 1
                                    bond_order = float(parts[i + 2])

                                    neighbors.append(
                                        {
                                            "element": neighbor_element,
                                            "index": neighbor_idx,
                                            "bond_order": bond_order,
                                        }
                                    )
                                    i += 3
                                else:
                                    break

                            current_sample_bonds.append(
                                {
                                    "atom_index": atom_idx,
                                    "element": element,
                                    "total_bond_order": total_bond_order,
                                    "neighbors": neighbors,
                                }
                            )
                        except (ValueError, IndexError) as e:
                            logger.warning(f"Error parsing bond: {line}, error: {e}")
                            continue

                if current_sample_bonds:
                    bond_data.append(current_sample_bonds)

            logger.info(f"Loaded bond data for {len(bond_data)} samples")

            if len(bond_data) != self.num_samples:
                logger.warning(
                    f"Bond data samples ({len(bond_data)}) don't match "
                    f"main data samples ({self.num_samples})"
                )

        except Exception as e:
            logger.warning(f"Error reading bond data from {path_bond}: {e}")
            return None

        return bond_data

    def _integrate_charge_data(self, structures, charge_data):

        if charge_data is None or len(structures) != len(charge_data):
            logger.warning("Mismatch between structures and charge data length")
            return structures

        for i, (structure, charges) in enumerate(zip(structures, charge_data)):

            if len(structure) != len(charges):
                logger.warning(
                    f"Sample {i}: Number of atoms in structure ({len(structure)}) "
                    f"doesn't match number of charges ({len(charges)})"
                )
                continue

            if hasattr(structure, "site_properties"):
                structure.add_site_property("charge", charges)

            elif hasattr(structure, "arrays"):
                structure.arrays["charge"] = np.array(charges, dtype="float32")
            else:
                structure.info["atomic_charges"] = charges

        logger.info("Successfully integrated charge data into structures")
        return structures

    def _integrate_bond_data(self, structures, bond_data):

        if bond_data is None or len(structures) != len(bond_data):
            logger.warning("Mismatch between structures and bond data length")
            return structures

        for i, (structure, bonds) in enumerate(zip(structures, bond_data)):
            if len(structure) != len(bonds):
                logger.warning(f"Sample {i}: Atom count mismatch")
                continue

            n_atoms = len(structure)
            bond_order_matrix = np.zeros((n_atoms, n_atoms), dtype="float32")

            for bond_entry in bonds:
                atom_i = bond_entry["atom_index"]
                for neighbor in bond_entry["neighbors"]:
                    atom_j = neighbor["index"]
                    bond_order = neighbor["bond_order"]
                    bond_order_matrix[atom_i, atom_j] = bond_order
                    bond_order_matrix[atom_j, atom_i] = bond_order

            try:

                structure.bond_order_matrix = bond_order_matrix
                structure.bond_data = bonds
            except AttributeError:

                if hasattr(structure, "info"):
                    structure.info["bond_order_matrix"] = bond_order_matrix
                    structure.info["bond_data"] = bonds
                else:
                    logger.warning(f"Cannot add bond data: {type(structure)}")
                    continue

        logger.info("Successfully integrated bond data into structures")
        return structures

    def get_structure_array(self, structure):
        atom_types = np.array([site.specie.Z for site in structure])

        lattice = np.eye(3, dtype="float32")
        lengths = np.array([1.0, 1.0, 1.0], dtype="float32").reshape(1, 3)
        angles = np.array([90.0, 90.0, 90.0], dtype="float32").reshape(1, 3)

        structure_array = {
            "frac_coords": ConcatData(structure.frac_coords.astype("float32")),
            "cart_coords": ConcatData(structure.cart_coords.astype("float32")),
            "atom_types": ConcatData(atom_types),
            "lattice": ConcatData(lattice.reshape(1, 3, 3)),
            "lengths": ConcatData(lengths),
            "angles": ConcatData(angles),
            "num_atoms": ConcatData(np.array([tuple(atom_types.shape)[0]])),
        }

        if self.use_atomic_charge:
            if (
                hasattr(structure, "site_properties")
                and "charge" in structure.site_properties
            ):  # naqo
                charges = np.array(structure.site_properties["charge"], dtype="float32")
                structure_array["atomic_charges"] = ConcatData(charges)
            elif hasattr(structure, "atomic_charges"):
                charges = np.array(structure.atomic_charges, dtype="float32")
                structure_array["atomic_charges"] = ConcatData(charges)

        if self.use_chemical_bonding:
            if hasattr(structure, "bond_order_matrix"):

                bond_matrix = structure.bond_order_matrix
                structure_array["bond_orders"] = bond_matrix
            elif (
                hasattr(structure, "site_properties")
                and "bond_orders" in structure.site_properties
            ):  # naqo

                bond_orders = np.array(
                    structure.site_properties["bond_orders"], dtype="float32"
                )
                structure_array["atomic_bond_orders"] = ConcatData(bond_orders)

            if hasattr(structure, "bond_data"):
                structure_array["bond_data"] = structure.bond_data

        return structure_array

    def _parse_comment_line(self, comment: str):
        # Parsing attributes in comment lines
        properties = {}
        if not comment:
            return properties

        try:
            parts = comment.split()
            for part in parts:
                if "=" in part:
                    key, value = part.split("=", 1)
                    if value.lower() == "nan":
                        properties[key] = float("nan")
                    else:
                        try:
                            if "." in value:
                                properties[key] = float(value)
                            else:
                                properties[key] = int(value)
                        except ValueError:
                            properties[key] = value
        except Exception as e:
            logger.warning(f"Comment line parsing error: {e}")

        return properties

    def _add_virtual_lattice(self, atoms):
        # Add an effective virtual lattice to molecular data
        import numpy as np

        if hasattr(atoms, "cell") and atoms.cell.rank == 3:

            if np.linalg.det(atoms.cell) > 1e-6:
                return

        coords = atoms.positions
        if len(coords) > 0:
            min_coords = coords.min(axis=0)
            max_coords = coords.max(axis=0)
            size = max_coords - min_coords

            padding = max(10.0, np.max(size) * 0.5)
            cell_size = size + padding

            atoms.set_cell(np.diag(cell_size))

            center = (min_coords + max_coords) / 2
            lattice_center = atoms.cell.lengths() / 2
            atoms.positions += lattice_center - center
        else:

            atoms.set_cell(np.eye(3) * 20.0)

        atoms.set_pbc([False, False, False])

    def _manual_xyz_parser(self, path: str):

        atoms_list = []

        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()

        structures = [s.strip() for s in content.split("\n\n") if s.strip()]

        for struct_idx, struct_content in enumerate(structures):
            lines = [
                line.strip() for line in struct_content.split("\n") if line.strip()
            ]

            if len(lines) < 2:
                logger.warning(
                    f"Skip structure {struct_idx}: Insufficient number of lines"
                )
                continue

            try:
                natoms_line = lines[0]
                try:
                    natoms = int(natoms_line)
                except ValueError:
                    import re

                    numbers = re.findall(r"\d+", natoms_line)
                    if numbers:
                        natoms = int(numbers[0])
                        logger.warning(f"{struct_idx}: Inferring the number of atoms")
                    else:
                        logger.warning(
                            f"{struct_idx}: The number of atoms cannot be resolved"
                        )
                        continue

                comment_line = lines[1] if len(lines) > 1 else ""

                # Resolve atomic coordinates
                symbols = []
                positions = []
                atom_lines = (
                    lines[2 : 2 + natoms] if len(lines) >= 2 + natoms else lines[2:]
                )

                for atom_line in atom_lines:
                    parts = atom_line.split()
                    if len(parts) >= 4:
                        symbol = parts[0]
                        try:
                            coords = [float(x) for x in parts[1:4]]
                            symbols.append(symbol)
                            positions.append(coords)
                        except ValueError:
                            logger.warning(
                                f"{struct_idx}: Coordinates cannot be resolved"
                            )

                if len(symbols) != natoms:
                    logger.warning(
                        f"{struct_idx}: expect {natoms} atoms, actual {len(symbols)} "
                    )
                    if len(symbols) == 0:
                        continue

                from ase import Atoms

                atoms = Atoms(symbols=symbols, positions=positions)

                self._add_virtual_lattice(atoms)

                properties = self._parse_comment_line(comment_line)
                atoms.info.update(properties)

                atoms_list.append(atoms)

            except Exception as e:
                logger.warning(f"Structure {struct_idx} Parsing error: {e}")
                continue

        if not atoms_list:
            raise ValueError("Didn't find an effective structure")

        return atoms_list

    def read_data(self, path: str, format: str = None):
        logger.info("Parse to read xyz files...")

        try:
            atoms_list = self._manual_xyz_parser(path)
            logger.info(f"Successful reading {len(atoms_list)} structure")
            return atoms_list, len(atoms_list)
        except Exception as e:
            logger.error(f"Parsing failed: {e}")
            raise

    def atoms_to_structure(self, atoms: Atoms):
        return AseAtomsAdaptor().get_structure(atoms)

    def filter_unvalid_by_property(self):
        for property_name in self.property_names:
            data = self.property_data[property_name]
            reserve_idx = []
            for i, data_item in enumerate(data):
                if isinstance(data_item, str) or (
                    data_item is not None and not math.isnan(data_item)
                ):
                    reserve_idx.append(i)
            for key in self.property_data.keys():
                self.property_data[key] = [
                    self.property_data[key][i] for i in reserve_idx
                ]

            self.row_data = [self.row_data[i] for i in reserve_idx]
            self.structures = [self.structures[i] for i in reserve_idx]
            if self.graphs is not None:
                self.graphs = [self.graphs[i] for i in reserve_idx]
            logger.warning(
                f"Filter out {len(reserve_idx)} samples with valid properties: "
                f"{property_name}"
            )
        self.num_samples = len(self.row_data)
        logger.warning(f"Remaining {self.num_samples} samples after filtering.")

    def read_property_data(self, data: List[Atoms]):
        """Read the property data from the given data and property names.

        Args:
            data (List[Atoms]): List of ASE Atoms objects.
        """
        property_data = {}

        if self.electronic_e_key is not None:
            property_data[self.electronic_e_key] = [
                data[i].info.get("Electronic_E", np.nan)
                for i in range(self.num_samples)
            ]

        if self.dispersion_e_key is not None:
            property_data[self.dispersion_e_key] = [
                data[i].info.get("Dispersion_E", np.nan)
                for i in range(self.num_samples)
            ]

        if self.dipole_m_key is not None:
            property_data[self.dipole_m_key] = [
                data[i].info.get("Dipole_M", np.nan) for i in range(self.num_samples)
            ]

        if self.metal_q_key is not None:
            property_data[self.metal_q_key] = [
                data[i].info.get("Metal_q", np.nan) for i in range(self.num_samples)
            ]

        if self.hl_gap_key is not None:
            property_data[self.hl_gap_key] = [
                data[i].info.get("HL_Gap", np.nan) for i in range(self.num_samples)
            ]

        if self.homo_energy_key is not None:
            property_data[self.homo_energy_key] = [
                data[i].info.get("HOMO_Energy", np.nan) for i in range(self.num_samples)
            ]

        if self.lumo_energy_key is not None:
            property_data[self.lumo_energy_key] = [
                data[i].info.get("LUMO_Energy", np.nan) for i in range(self.num_samples)
            ]

        if self.polarizability_key is not None:
            property_data[self.polarizability_key] = [
                data[i].info.get("Polarizability", np.nan)
                for i in range(self.num_samples)
            ]

        if self.smiles_key is not None:
            property_data[self.smiles_key] = [
                data[i].info.get("SMILES", None) for i in range(self.num_samples)
            ]

        return property_data

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

    def _check_cache_integrity(self, structure_cache_path, graph_cache_path):

        missing_files = []

        for i in range(min(100, self.num_samples)):
            cache_file = osp.join(structure_cache_path, f"{i:010d}.pkl")
            if not osp.exists(cache_file):
                missing_files.append(cache_file)
                break

        if self.build_graph_cfg is not None:
            for i in range(min(100, self.num_samples)):
                cache_file = osp.join(graph_cache_path, f"{i:010d}.pkl")
                if not osp.exists(cache_file):
                    missing_files.append(cache_file)
                    break

        return missing_files

    def __getitem__(self, idx: int):
        """Get item at index idx."""
        data = {}
        # get graph
        if self.graphs is not None:
            graph = self.graphs[idx]
            if isinstance(graph, str):

                if osp.exists(graph):
                    graph = self.load_from_cache(graph)
                else:

                    logger.warning(f"Graph cache file {graph} not found")

                    structure = self.structures[idx]
                    if isinstance(structure, str) and osp.exists(structure):
                        structure = self.load_from_cache(structure)
                    if self.build_graph_cfg is not None and structure is not None:
                        converter = build_graph_converter(self.build_graph_cfg)
                        graph = converter([structure])[0]
                    else:
                        graph = None
            data["graph"] = graph
        else:
            structure = self.structures[idx]
            if isinstance(structure, str):
                structure = self.load_from_cache(structure)
            data["structure_array"] = self.get_structure_array(structure)

        for property_name in self.property_names:

            if property_name in self.property_data:
                # SMILES is a string type
                if property_name == self.smiles_key:
                    data[property_name] = self.property_data[property_name][idx]
                else:
                    # Other numeric attributes are converted to numpy arrays
                    data[property_name] = np.array(
                        [self.property_data[property_name][idx]]
                    ).astype("float32")
            else:
                raise KeyError(f"Property {property_name} not found.")
        # Use indexes as IDs
        data["id"] = idx
        data = self.transforms(data) if self.transforms is not None else data

        return data

    def __len__(self):
        return self.num_samples

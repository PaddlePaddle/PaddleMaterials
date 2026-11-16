import itertools
import warnings
from collections import Counter

import numpy as np
import paddle
import smact
from matminer.featurizers.composition.composite import ElementProperty
from matminer.featurizers.site.fingerprint import CrystalNNFingerprint
from pymatgen.core import Element
from pymatgen.core.composition import Composition
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure
from scipy.linalg import polar
from smact.screening import pauling_test

from ppmat.utils.crystal import lattices_to_params_shape_numpy

# Ignore warnings
warnings.filterwarnings("ignore", category=UserWarning)


# Warning: the smact package version is 2.5.5,
# different version may cause slight differences in accuracy.


CrystalNNFP = CrystalNNFingerprint.from_preset("ops")
CompFP = ElementProperty.from_preset("magpie")


def smact_validity(comp, count, use_pauling_test=True, include_alloys=True):
    elem_symbols = tuple([Element.from_Z(elem).symbol for elem in comp])
    space = smact.element_dictionary(elem_symbols)
    smact_elems = [e[1] for e in space.items()]
    electronegs = [e.pauling_eneg for e in smact_elems]
    ox_combos = [e.oxidation_states for e in smact_elems]
    if len(set(elem_symbols)) == 1:
        return True
    if include_alloys:
        is_metal_list = [(elem_s in smact.metals) for elem_s in elem_symbols]
        if all(is_metal_list):
            return True
    threshold = np.max(count)
    oxn = 1
    for oxc in ox_combos:
        oxn *= len(oxc)
    if oxn > 10000000.0:
        return False
    for ox_states in itertools.product(*ox_combos):
        stoichs = [(c,) for c in count]
        cn_e, cn_r = smact.neutral_ratios(
            ox_states, stoichs=stoichs, threshold=threshold
        )
        if cn_e:
            if use_pauling_test:
                try:
                    electroneg_OK = pauling_test(ox_states, electronegs)
                except TypeError:
                    electroneg_OK = True
            else:
                electroneg_OK = True
            if electroneg_OK:
                return True
    return False


def structure_validity(crystal, cutoff=0.5):
    dist_mat = crystal.distance_matrix
    dist_mat = dist_mat + np.diag(np.ones(tuple(dist_mat.shape)[0]) * (cutoff + 10.0))
    if dist_mat.min() < cutoff or crystal.volume < 0.1:
        return False
    else:
        return True


class Crystal(object):
    def __init__(self, crys_array_dict):
        if isinstance(crys_array_dict["frac_coords"], paddle.Tensor):
            self.frac_coords = crys_array_dict["frac_coords"].cpu().numpy()
        else:
            self.frac_coords = np.array(crys_array_dict["frac_coords"])
        if isinstance(crys_array_dict["atom_types"], paddle.Tensor):
            self.atom_types = crys_array_dict["atom_types"].cpu().numpy()
        else:
            self.atom_types = np.array(crys_array_dict["atom_types"])

        if "lengths" in crys_array_dict and "angles" in crys_array_dict:
            if isinstance(crys_array_dict["lengths"], paddle.Tensor):
                self.lengths = crys_array_dict["lengths"].cpu().numpy()
            else:
                self.lengths = np.array(crys_array_dict["lengths"])
            if isinstance(crys_array_dict["angles"], paddle.Tensor):
                self.angles = crys_array_dict["angles"].cpu().numpy()
            else:
                self.angles = np.array(crys_array_dict["angles"])
        else:
            if isinstance(crys_array_dict["lattice"], paddle.Tensor):
                lattice = crys_array_dict["lattice"].cpu().numpy()
            else:
                lattice = np.array([crys_array_dict["lattice"]])
            self.lengths, self.angles = lattices_to_params_shape_numpy(lattice)
            self.lengths, self.angles = self.lengths[0], self.angles[0]
        self.dict = {
            "frac_coords": self.frac_coords,
            "atom_types": self.atom_types,
            "lengths": self.lengths,
            "angles": self.angles,
        }
        if len(tuple(self.atom_types.shape)) > 1:
            self.dict["atom_types"] = np.argmax(self.atom_types, axis=-1) + 1
            self.atom_types = np.argmax(self.atom_types, axis=-1) + 1
        self.get_structure()
        self.get_composition()
        self.get_validity()
        self.get_fingerprints()

    def get_structure(self):
        if min(self.lengths.tolist()) < 0:
            self.constructed = False
            self.invalid_reason = "non_positive_lattice"
        if (
            np.isnan(self.lengths).any()
            or np.isnan(self.angles).any()
            or np.isnan(self.frac_coords).any()
        ):
            self.constructed = False
            self.invalid_reason = "nan_value"
        else:
            try:
                self.structure = Structure(
                    lattice=Lattice.from_parameters(
                        *(self.lengths.tolist() + self.angles.tolist())
                    ),
                    species=self.atom_types,
                    coords=self.frac_coords,
                    coords_are_cartesian=False,
                )
                self.constructed = True
                if self.structure.volume < 0.1:
                    self.constructed = False
                    self.invalid_reason = "unrealistically_small_lattice"
            except Exception:
                self.constructed = False
                self.invalid_reason = "construction_raises_exception"

    def get_composition(self):
        elem_counter = Counter(self.atom_types)
        composition = [
            (elem, elem_counter[elem]) for elem in sorted(elem_counter.keys())
        ]
        elems, counts = list(zip(*composition))
        counts = np.array(counts)
        counts = counts / np.gcd.reduce(counts)
        self.elems = elems
        self.comps = tuple(counts.astype("int").tolist())

    def get_validity(self):
        self.comp_valid = smact_validity(self.elems, self.comps)
        if self.constructed:
            self.struct_valid = structure_validity(self.structure)
        else:
            self.struct_valid = False
        self.valid = self.comp_valid and self.struct_valid

    def get_fingerprints(self):
        elem_counter = Counter(self.atom_types)
        comp = Composition(elem_counter)
        self.comp_fp = CompFP.featurize(comp)
        try:
            site_fps = [
                CrystalNNFP.featurize(self.structure, i)
                for i in range(len(self.structure))
            ]
        except Exception:
            self.valid = False
            self.comp_fp = None
            self.struct_fp = None
            return
        self.struct_fp = np.array(site_fps).mean(axis=0)


def get_crys_from_cif(cif, polar_decompose=False):
    structure = Structure.from_str(cif, fmt="cif")
    lattice = structure.lattice

    atom_types = np.array([site.specie.Z for site in structure])

    if polar_decompose:
        lattice_m = lattice.matrix
        _, lattice_m = polar(lattice.matrix)
        lengths, angles = lattices_to_params_shape_numpy(lattice_m)
        crys_array_dict = {
            "frac_coords": structure.frac_coords,
            "atom_types": atom_types,
            "lengths": lengths,
            "angles": angles,
        }
    else:
        crys_array_dict = {
            "frac_coords": structure.frac_coords,
            "atom_types": atom_types,
            "lengths": np.array(lattice.abc),
            "angles": np.array(lattice.angles),
        }
    return Crystal(crys_array_dict)

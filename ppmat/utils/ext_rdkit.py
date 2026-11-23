from __future__ import annotations

import re

import numpy as np
import paddle
from rdkit import Chem

from ppmat.utils import logger

allowed_bonds = {
    "H": 1,
    "C": 4,
    "N": 3,
    "O": 2,
    "F": 1,
    "B": 3,
    "Al": 3,
    "Si": 4,
    "P": [3, 5],
    "S": 4,
    "Cl": 1,
    "As": 3,
    "Br": 1,
    "I": 1,
    "Hg": [1, 2],
    "Bi": [3, 5],
    "Se": [2, 4, 6],
}
BOND_TYPES = {
    "SINGLE": Chem.rdchem.BondType.SINGLE,
    "DOUBLE": Chem.rdchem.BondType.DOUBLE,
    "TRIPLE": Chem.rdchem.BondType.TRIPLE,
    "AROMATIC": Chem.rdchem.BondType.AROMATIC,
}
bond_dict = [None, *BOND_TYPES.values()]
ATOM_VALENCY = {6: 4, 7: 3, 8: 2, 9: 1, 15: 3, 16: 2, 17: 1, 35: 1, 53: 1}


def bond_types_from_vocab(bond_vocab=None):
    if bond_vocab is None:
        return {
            index: bond_type for index, bond_type in enumerate(BOND_TYPES.values(), 1)
        }
    return {
        index: BOND_TYPES[token]
        for token, index in bond_vocab["token_to_id"].items()
        if token in BOND_TYPES
    }


def iter_bonds(adjacency_matrix, bond_vocab=None):
    if hasattr(adjacency_matrix, "numpy"):
        adjacency_matrix = adjacency_matrix.numpy()
    adjacency_matrix = np.asarray(adjacency_matrix)
    bond_types = bond_types_from_vocab(bond_vocab)
    for source in range(adjacency_matrix.shape[0]):
        for target in range(source + 1, adjacency_matrix.shape[1]):
            bond_type = bond_types.get(int(adjacency_matrix[source, target]))
            if bond_type is None:
                bond_type = bond_types.get(int(adjacency_matrix[target, source]))
            if bond_type is not None:
                yield source, target, bond_type


def mol_from_graphs(atom_decoder, node_list, adjacency_matrix, bond_decoder=None):
    """Convert a standard discrete molecular graph to an RDKit molecule."""

    if hasattr(node_list, "numpy"):
        node_list = node_list.numpy()
    node_list = np.asarray(node_list).reshape(-1)

    molecule = Chem.RWMol()
    node_to_idx = {}
    for index, node in enumerate(node_list):
        node = int(node)
        if node == -1:
            continue
        node_to_idx[index] = molecule.AddAtom(Chem.Atom(atom_decoder[node]))

    for source, target, bond_type in iter_bonds(adjacency_matrix, bond_decoder):
        if source not in node_to_idx or target not in node_to_idx:
            continue
        try:
            molecule.AddBond(node_to_idx[source], node_to_idx[target], bond_type)
        except Exception:
            return None

    try:
        return molecule.GetMol()
    except Exception:
        return None


class BasicMolecularMetrics(object):
    """
    Generate and evaluate the sturcte of molecules
    """

    def __init__(self, dataset_info, train_smiles=None):
        self.atom_decoder = dataset_info.atom_decoder
        self.dataset_info = dataset_info
        self.bond_decoder = dataset_info.vocab["bond"]
        # Retrieve dataset smiles only for qm9 currently
        self.dataset_smiles_list = train_smiles

    def compute_validity(self, generated):
        """
        generated: list of couples (positions, atom_types) for generated molecules
        """
        valid = []
        num_components = []
        all_smiles = []
        for graph in generated:
            atom_types, edge_types = graph
            try:
                mol = mol_from_graphs(
                    self.atom_decoder,
                    atom_types,
                    edge_types,
                    bond_decoder=self.bond_decoder,
                )
                mol_frags = Chem.rdmolops.GetMolFrags(
                    mol, asMols=True, sanitizeFrags=True
                )
                num_components.append(len(mol_frags))
                largest_mol = max(mol_frags, default=mol, key=lambda m: m.GetNumAtoms())
                smiles = mol2smiles(largest_mol)
                if smiles is not None:
                    valid.append(smiles)
                    all_smiles.append(smiles)
            except Exception as e:
                logger.debug(f"Error in GetMolFrags: {e}")
                all_smiles.append(None)
            else:
                if smiles is None:
                    all_smiles.append(None)
        return valid, len(valid) / len(generated), np.array(num_components), all_smiles

    def compute_uniqueness(self, valid):
        """valid: list of SMILES strings."""
        return list(set(valid)), len(set(valid)) / len(valid)

    def compute_novelty(self, unique):
        num_novel = 0
        novel = []
        if self.dataset_smiles_list is None:
            print("Dataset smiles is None, novelty computation skipped")
            return 1, 1
        for smiles in unique:
            if smiles not in self.dataset_smiles_list:
                novel.append(smiles)
                num_novel += 1
        return novel, num_novel / len(unique)

    def compute_relaxed_validity(self, generated):
        valid = []
        for graph in generated:
            atom_types, edge_types = graph
            atom_types = paddle.to_tensor(atom_types)
            edge_types = paddle.to_tensor(edge_types)
            try:
                mol = build_molecule_with_partial_charges(
                    atom_types,
                    edge_types,
                    self.atom_decoder,
                    bond_decoder=self.bond_decoder,
                )
                smiles = mol2smiles(mol)
                if smiles is not None:
                    mol_frags = Chem.rdmolops.GetMolFrags(
                        mol, asMols=True, sanitizeFrags=True
                    )
                    largest_mol = max(
                        mol_frags, default=mol, key=lambda m: m.GetNumAtoms()
                    )
                    smiles = mol2smiles(largest_mol)
                    if smiles is not None:
                        valid.append(smiles)
            except Exception as error:
                logger.debug(f"Error in relaxed molecule construction: {error}")
        return valid, len(valid) / len(generated)

    def evaluate(self, generated):
        """generated: list of pairs (positions: n x 3, atom_types: n [int])
        the positions and atom types should already be masked."""
        _, validity, num_components, all_smiles = self.compute_validity(generated)
        nc_mu = num_components.mean() if len(num_components) > 0 else 0
        nc_min = num_components.min() if len(num_components) > 0 else 0
        nc_max = num_components.max() if len(num_components) > 0 else 0

        relaxed_valid, relaxed_validity = self.compute_relaxed_validity(generated)

        if relaxed_validity > 0:
            unique, uniqueness = self.compute_uniqueness(relaxed_valid)
            if self.dataset_smiles_list is not None:
                _, novelty = self.compute_novelty(unique)
            else:
                novelty = -1.0
        else:
            novelty = -1.0
            uniqueness = 0.0
            unique = []
        return (
            [validity, relaxed_validity, uniqueness, novelty],
            unique,
            dict(nc_min=nc_min, nc_max=nc_max, nc_mu=nc_mu),
            all_smiles,
        )


def mol2smiles(mol):
    if mol is None:
        return None
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return None
    return Chem.MolToSmiles(mol)


def build_molecule(
    atom_types, edge_types, atom_decoder, verbose=False, bond_decoder=None
):
    if verbose:
        print("building new molecule")
    mol = Chem.RWMol()
    for atom in atom_types:
        atom_id = int(atom.item())
        a = Chem.Atom(atom_decoder[atom_id])
        mol.AddAtom(a)
        if verbose:
            print("Atom added: ", atom_id, atom_decoder[atom_id])
    for source, target, bond_type in iter_bonds(edge_types, bond_decoder):
        mol.AddBond(source, target, bond_type)
        if verbose:
            print("bond added:", source, target, bond_type)
    return mol


def build_molecule_with_partial_charges(
    atom_types, edge_types, atom_decoder, verbose=False, bond_decoder=None
):
    if verbose:
        print("\nbuilding new molecule")
    mol = Chem.RWMol()
    for atom in atom_types:
        atom_id = int(atom.item())
        a = Chem.Atom(atom_decoder[atom_id])
        mol.AddAtom(a)
        if verbose:
            print("Atom added: ", atom_id, atom_decoder[atom_id])
    for source, target, bond_type in iter_bonds(edge_types, bond_decoder):
        mol.AddBond(source, target, bond_type)
        if verbose:
            print("bond added:", source, target, bond_type)
        flag, atomid_valence = check_valency(mol)
        if verbose:
            print("flag, valence", flag, atomid_valence)
        if flag:
            continue
        assert len(atomid_valence) == 2
        idx, valence = atomid_valence
        atomic_number = mol.GetAtomWithIdx(idx).GetAtomicNum()
        if atomic_number in (7, 8, 16) and valence - ATOM_VALENCY[atomic_number] == 1:
            mol.GetAtomWithIdx(idx).SetFormalCharge(1)
    return mol


def check_valency(mol):
    try:
        Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_PROPERTIES)
        return True, None
    except ValueError as e:
        e = str(e)
        p = e.find("#")
        e_sub = e[p:]
        atomid_valence = list(map(int, re.findall("\\d+", e_sub)))
        return False, atomid_valence


def correct_mol(m):
    mol = m
    no_correct = False
    flag, _ = check_valency(mol)
    if flag:
        no_correct = True
    while True:
        flag, atomid_valence = check_valency(mol)
        if flag:
            break
        else:
            assert len(atomid_valence) == 2
            idx = atomid_valence[0]
            queue = []
            check_idx = 0
            for b in mol.GetAtomWithIdx(idx).GetBonds():
                type = int(b.GetBondType())
                queue.append((b.GetIdx(), type, b.GetBeginAtomIdx(), b.GetEndAtomIdx()))
                if type == 12:
                    check_idx += 1
            queue.sort(key=lambda tup: tup[1], reverse=True)
            if queue[-1][1] == 12:
                return None, no_correct
            elif len(queue) > 0:
                start = queue[check_idx][2]
                end = queue[check_idx][3]
                t = queue[check_idx][1] - 1
                mol.RemoveBond(start, end)
                if t >= 1:
                    mol.AddBond(start, end, bond_dict[t])
    return mol, no_correct


def valid_mol_can_with_seg(m, largest_connected_comp=True):
    if m is None:
        return None
    sm = Chem.MolToSmiles(m, isomericSmiles=True)
    if largest_connected_comp and "." in sm:
        vsm = [(s, len(s)) for s in sm.split(".")]
        vsm.sort(key=lambda tup: tup[1], reverse=True)
        mol = Chem.MolFromSmiles(vsm[0][0])
    else:
        mol = Chem.MolFromSmiles(sm)
    return mol


def check_stability(
    atom_types,
    edge_types,
    dataset_info,
    debug=False,
    atom_decoder=None,
    bond_decoder=None,
):
    if atom_decoder is None:
        atom_decoder = dataset_info.atom_decoder
    if bond_decoder is None:
        bond_decoder = dataset_info.vocab["bond"]
    bond_orders = {
        Chem.rdchem.BondType.SINGLE: 1.0,
        Chem.rdchem.BondType.DOUBLE: 2.0,
        Chem.rdchem.BondType.TRIPLE: 3.0,
        Chem.rdchem.BondType.AROMATIC: 1.5,
    }
    n_bonds = np.zeros(len(atom_types), dtype=np.float64)
    for source, target, bond_type in iter_bonds(edge_types, bond_decoder):
        n_bonds[source] += bond_orders[bond_type]
        n_bonds[target] += bond_orders[bond_type]
    n_stable_bonds = 0
    for atom_type, atom_n_bond in zip(atom_types, n_bonds):
        atom_type = int(atom_type.item())
        possible_bonds = allowed_bonds[atom_decoder[atom_type]]
        if type(possible_bonds) == int:
            is_stable = bool(np.isclose(possible_bonds, atom_n_bond))
        else:
            is_stable = any(np.isclose(value, atom_n_bond) for value in possible_bonds)
        if not is_stable and debug:
            logger.info(
                "Invalid bonds for molecule %s with %.3f bonds"
                % (atom_decoder[atom_type], atom_n_bond)
            )
        n_stable_bonds += int(is_stable)
    molecule_stable = n_stable_bonds == len(atom_types)
    return molecule_stable, n_stable_bonds, len(atom_types)


def compute_molecular_metrics(molecule_list, train_smiles, dataset_info):
    """molecule_list: (dict)"""
    if not dataset_info.remove_h:
        logger.info("Analyzing molecule stability...")
        molecule_stable = 0
        nr_stable_bonds = 0
        n_atoms = 0
        n_molecules = len(molecule_list)
        for i, mol in enumerate(molecule_list):
            atom_types, edge_types = mol
            try:
                validity_results = check_stability(atom_types, edge_types, dataset_info)
            except (IndexError, KeyError, TypeError, ValueError):
                continue
            molecule_stable += int(validity_results[0])
            nr_stable_bonds += int(validity_results[1])
            n_atoms += int(validity_results[2])
        fraction_mol_stable = molecule_stable / float(n_molecules)
        fraction_atm_stable = nr_stable_bonds / float(max(n_atoms, 1))
        validity_dict = {
            "mol_stable": fraction_mol_stable,
            "atm_stable": fraction_atm_stable,
        }
    else:
        validity_dict = {"mol_stable": -1, "atm_stable": -1}
    metrics = BasicMolecularMetrics(dataset_info, train_smiles)
    rdkit_metrics = metrics.evaluate(molecule_list)
    all_smiles = rdkit_metrics[-1]
    return validity_dict, rdkit_metrics, all_smiles


if __name__ == "__main__":
    smiles_mol = "C1CCC1"
    print("Smiles mol %s" % smiles_mol)
    chem_mol = Chem.MolFromSmiles(smiles_mol)
    block_mol = Chem.MolToMolBlock(chem_mol)
    print("Block mol:")
    print(block_mol)
use_rdkit = True

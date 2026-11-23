from typing import Any, Dict, List, Union

from ase.data import atomic_numbers, chemical_symbols

from ppmat.models.sevennet import _keys as KEY
from ppmat.models.sevennet.nn.node_embedding import get_type_mapper_from_specie
from ppmat.models.common.e3nn.o3 import FullTensorProduct
from ppmat.models.common.e3nn.o3 import Irreps


def chemical_species_preprocess(
    input_chem: List[str], universal: bool = False
) -> Dict[str, Any]:
    """Create SevenNet's deterministic atomic-number-to-type mapping."""
    if universal:
        return {
            KEY.CHEMICAL_SPECIES: chemical_symbols,
            KEY.CHEMICAL_SPECIES_BY_ATOMIC_NUMBER: list(range(len(chemical_symbols))),
            KEY.NUM_SPECIES: len(chemical_symbols),
            KEY.TYPE_MAP: {z: z for z in range(len(chemical_symbols))},
        }
    species = sorted({element.strip() for element in input_chem if element.strip()})
    if not species:
        raise ValueError("chemical_species must contain at least one element")
    unknown = [element for element in species if element not in atomic_numbers]
    if unknown:
        raise ValueError(f"Unknown chemical species: {unknown}")
    return {
        KEY.CHEMICAL_SPECIES: species,
        KEY.CHEMICAL_SPECIES_BY_ATOMIC_NUMBER: [atomic_numbers[element] for element in species],
        KEY.NUM_SPECIES: len(species),
        KEY.TYPE_MAP: get_type_mapper_from_specie(species),
    }


def infer_irreps_out(
    irreps_x: Irreps,
    irreps_operand: Irreps,
    drop_l: Union[bool, int] = False,
    parity_mode: str = "full",
    fix_multiplicity: Union[bool, int] = False,
) -> Irreps:
    """Infer output irreps used by SevenNet interaction blocks."""

    if parity_mode not in ["full", "even", "sph"]:
        raise ValueError(
            f"Unsupported parity_mode: {parity_mode}"
        )

    irreps_out = FullTensorProduct(
        irreps_x,
        irreps_operand,
    ).irreps_out.simplify()

    new_irreps_elem = []

    for mul, (l, p) in irreps_out:
        elem = (mul, (l, p))

        if drop_l is not False and l > drop_l:
            continue

        if parity_mode == "even" and p == -1:
            continue

        if parity_mode == "sph" and p != (-1) ** l:
            continue

        if fix_multiplicity:
            elem = (
                fix_multiplicity,
                (l, p),
            )

        new_irreps_elem.append(elem)

    return Irreps(new_irreps_elem)

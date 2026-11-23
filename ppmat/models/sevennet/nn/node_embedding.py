from typing import Dict, List, Optional

import paddle
from ppmat.models.sevennet import _keys as KEY
from ase.symbols import symbols2numbers
from ppmat.models.common.e3nn.util.jit import compile_mode
from ppmat.models.sevennet._const import AtomGraphDataType


@compile_mode("script")
class OnehotEmbedding(paddle.nn.Layer):
    """
    x : tensor of shape (N, 1)
    x_after : tensor of shape (N, num_classes)
    It overwrite data_key_x
    and saves input to data_key_save and output to data_key_additional
    I know this is strange but it is for compatibility with previous version
    and to specie wise shift scale work
    ex) [0 1 1 0] -> [[1, 0] [0, 1] [0, 1] [1, 0]] (num_classes = 2)
    """

    def __init__(
        self,
        num_classes: int,
        data_key_x: str = KEY.NODE_FEATURE,
        data_key_out: Optional[str] = None,
        data_key_save: Optional[str] = None,
        data_key_additional: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.key_x = data_key_x
        if data_key_out is None:
            self.key_output = data_key_x
        else:
            self.key_output = data_key_out
        self.key_save = data_key_save
        self.key_additional_output = data_key_additional

    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:
        inp = data[self.key_x]
        embd = paddle.nn.functional.one_hot(inp, self.num_classes)
        embd = paddle.cast(embd, dtype="float32")
        data[self.key_output] = embd
        if self.key_additional_output is not None:
            data[self.key_additional_output] = embd
        if self.key_save is not None:
            data[self.key_save] = inp
        return data


def get_type_mapper_from_specie(specie_list: List[str]) -> Dict[int, int]:
    """
    from ['Hf', 'O']
    return {72: 0, 8: 1}
    """
    specie_list = sorted(specie_list)
    type_map = {}
    unique_counter = 0
    for specie in specie_list:
        atomic_num = symbols2numbers(specie)[0]
        if atomic_num in type_map:
            continue
        type_map[atomic_num] = unique_counter
        unique_counter += 1
    return type_map


def one_hot_atom_embedding(
    atomic_numbers: List[int], type_map: Dict[int, int]
) -> paddle.Tensor:
    """
    atomic numbers from ase.get_atomic_numbers
    type_map from get_type_mapper_from_specie()
    """
    num_classes = len(type_map)
    try:
        type_numbers = paddle.LongTensor([type_map[num] for num in atomic_numbers])
    except KeyError as e:
        raise ValueError(f"Atomic number {e.args[0]} is not expected")
    embd = paddle.nn.functional.one_hot(type_numbers, num_classes)
    embd = embd.to(paddle.get_default_dtype())
    return embd

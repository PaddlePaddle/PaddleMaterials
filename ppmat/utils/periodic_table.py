# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Periodic table position data for the 2D atomic representation.

The mapping assigns every atomic number a ``(period, group)`` position on the
periodic table. Lanthanides and actinides are embedded inside the period 6/7
main group block with fractional group offsets (3.01-3.14) so that every
element keeps a distinct 2D position. Atom number 0 is the invalid-atom
placeholder mapped to ``(0, 0)``.
"""

MAX_PERIOD = 7
MAX_GROUP = 18

ATOMIC_NUMBER_TO_POSITION = {
    0: (0, 0),
    1: (1, 1),
    2: (1, 18),
    3: (2, 1),
    4: (2, 2),
    5: (2, 13),
    6: (2, 14),
    7: (2, 15),
    8: (2, 16),
    9: (2, 17),
    10: (2, 18),
    11: (3, 1),
    12: (3, 2),
    13: (3, 13),
    14: (3, 14),
    15: (3, 15),
    16: (3, 16),
    17: (3, 17),
    18: (3, 18),
    19: (4, 1),
    20: (4, 2),
    21: (4, 3),
    22: (4, 4),
    23: (4, 5),
    24: (4, 6),
    25: (4, 7),
    26: (4, 8),
    27: (4, 9),
    28: (4, 10),
    29: (4, 11),
    30: (4, 12),
    31: (4, 13),
    32: (4, 14),
    33: (4, 15),
    34: (4, 16),
    35: (4, 17),
    36: (4, 18),
    37: (5, 1),
    38: (5, 2),
    39: (5, 3),
    40: (5, 4),
    41: (5, 5),
    42: (5, 6),
    43: (5, 7),
    44: (5, 8),
    45: (5, 9),
    46: (5, 10),
    47: (5, 11),
    48: (5, 12),
    49: (5, 13),
    50: (5, 14),
    51: (5, 15),
    52: (5, 16),
    53: (5, 17),
    54: (5, 18),
    55: (6, 1),
    56: (6, 2),
    57: (6, 3),
    58: (6, 3.01),
    59: (6, 3.02),
    60: (6, 3.03),
    61: (6, 3.04),
    62: (6, 3.05),
    63: (6, 3.06),
    64: (6, 3.07),
    65: (6, 3.08),
    66: (6, 3.09),
    67: (6, 3.10),
    68: (6, 3.11),
    69: (6, 3.12),
    70: (6, 3.13),
    71: (6, 3.14),
    72: (6, 4),
    73: (6, 5),
    74: (6, 6),
    75: (6, 7),
    76: (6, 8),
    77: (6, 9),
    78: (6, 10),
    79: (6, 11),
    80: (6, 12),
    81: (6, 13),
    82: (6, 14),
    83: (6, 15),
    84: (6, 16),
    85: (6, 17),
    86: (6, 18),
    87: (7, 1),
    88: (7, 2),
    89: (7, 3),
    90: (7, 3.01),
    91: (7, 3.02),
    92: (7, 3.03),
    93: (7, 3.04),
    94: (7, 3.05),
    95: (7, 3.06),
    96: (7, 3.07),
    97: (7, 3.08),
    98: (7, 3.09),
    99: (7, 3.10),
    100: (7, 3.11),
    101: (7, 3.12),
    102: (7, 3.13),
    103: (7, 3.14),
    104: (7, 4),
    105: (7, 5),
    106: (7, 6),
    107: (7, 7),
    108: (7, 8),
    109: (7, 9),
    110: (7, 10),
    111: (7, 11),
    112: (7, 12),
    113: (7, 13),
    114: (7, 14),
    115: (7, 15),
    116: (7, 16),
    117: (7, 17),
    118: (7, 18),
}


def normalize_row(row):
    """Normalize a period number to the [-1, 1] range."""
    return (row / MAX_PERIOD) * 2 - 1


def normalize_column(column):
    """Normalize a group number to the [-1, 1] range."""
    return (column / MAX_GROUP) * 2 - 1

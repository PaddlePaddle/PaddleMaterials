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

from typing import Any
from typing import Dict
from typing import List
from typing import Sequence
from typing import Union

import numpy as np
from p_tqdm import p_map


class BuildSpectrumNMR:
    """
    Convert tokenized NMR JSON into fixed-size numeric arrays for 1H and 13C.

    Input format (per sample, e.g. from CSV "tokenized_input" JSON):
        {
          "1HNMR": [
            [chem_shift, peak_width_token, split_token, "nH", [J1, J2, ...]],
            ...
          ],
          "13CNMR": [c_shift1, c_shift2, ...]
        }

    Output (dict of NumPy arrays and counts):
        {
          "H_nmr":      float32 [seq_len_H1, 4 + j_len],
            # [δ, peakwidth_id, split_id, integral, J*]
          "num_H_peak": int,
          "C_nmr":      float32 [seq_len_C13],
            # δ (ppm), padded/truncated
          "num_C_peak": int,
        }

    Notes:
        - Unknown tokens map to the ``<unk>`` entry of their vocabulary.
        - Peaks beyond the sequence length are truncated; missing slots are zero-padded.
    """

    def __init__(
        self,
        vocab: Dict[str, Dict[str, Any]],
        seq_len_H1: int,
        seq_len_C13: int,
        *,
        j_len: int = 6,
        dtype: str = "float32",
        num_cpus: int = 1,
    ) -> None:
        self.vocab_peakwidth = vocab["peakwidth"]["token_to_id"]
        self.vocab_split = vocab["split"]["token_to_id"]
        self.vocab_integral = vocab["integral"]["token_to_id"]
        self.seq_len_H1 = int(seq_len_H1)
        self.seq_len_C13 = int(seq_len_C13)
        self.j_len = int(j_len)
        self.dtype = np.dtype(dtype)
        self.num_cpus = int(num_cpus)

    @staticmethod
    def _parse_integral(h_str: Union[str, int, float]) -> str:
        # Accept "3H" or 3 and normalize to the vocabulary token.
        if h_str is None:
            return "<unk>"
        value = str(h_str).upper().replace("H", "").strip()
        try:
            return f"{int(float(value))}H"
        except (TypeError, ValueError, OverflowError):
            return "<unk>"

    @staticmethod
    def build_one(
        nmrdata: Dict[str, Any],
        vocab_peakwidth: Dict[str, int],
        vocab_split: Dict[str, int],
        vocab_integral: Dict[str, int],
        seq_len_H1: int,
        seq_len_C13: int,
        j_len: int,
        dtype: np.dtype,
    ) -> Dict[str, Any]:
        # ----- 1H NMR -----
        Hnmr = nmrdata.get("1HNMR", []) or []
        num_h = len(Hnmr)

        # Allocate [seq_len_H1, 4 + j_len]: [δ, peakwidth_id, split_id, integral,
        # J1..Jj_len]
        H_arr = np.zeros((seq_len_H1, 4 + j_len), dtype=dtype)

        # Fill rows up to seq_len_H1
        limit_h = min(seq_len_H1, num_h)
        for i in range(limit_h):
            peak = Hnmr[i]
            # Expected: [chem_shift (float), peakwidth_token (str), split_token (str),
            # "nH", [J...]]
            chem_shift = float(peak[0])
            peakwidth_tok = str(peak[1])
            split_tok = str(peak[2])
            integral_str = peak[3]
            j_list = (
                peak[4] if len(peak) > 4 and isinstance(peak[4], (list, tuple)) else []
            )

            peakwidth_id = vocab_peakwidth.get(peakwidth_tok, vocab_peakwidth["<unk>"])
            split_id = vocab_split.get(split_tok, vocab_split["<unk>"])
            integral_token = BuildSpectrumNMR._parse_integral(integral_str)
            integral_id = vocab_integral.get(integral_token, vocab_integral["<unk>"])

            row = [chem_shift, float(peakwidth_id), float(split_id), float(integral_id)]
            if len(j_list) >= j_len:
                row += [float(x) for x in j_list[:j_len]]
            else:
                row += [float(x) for x in j_list] + [0.0] * (j_len - len(j_list))

            H_arr[i, : len(row)] = np.asarray(row, dtype=dtype)

        # ----- 13C NMR -----
        Cnmr = nmrdata.get("13CNMR", []) or []
        num_c = len(Cnmr)
        C_arr = np.zeros((seq_len_C13,), dtype=dtype)
        if num_c > 0:
            C_vals = np.asarray([float(x) for x in Cnmr[:seq_len_C13]], dtype=dtype)
            C_arr[: len(C_vals)] = C_vals

        return {
            "H_nmr": H_arr,  # [seq_len_H1, 4 + j_len] float32
            "num_H_peak": int(num_h),
            "C_nmr": C_arr,  # [seq_len_C13] float32
            "num_C_peak": int(num_c),
        }

    def __call__(
        self, nmr_list: Union[Sequence[Dict[str, Any]], Dict[str, Any]]
    ) -> Union[List[Dict[str, Any]], Dict[str, Any]]:
        """Vectorized/batched conversion with p_tqdm.p_map (or single sample)."""
        if isinstance(nmr_list, (list, tuple)):
            if len(nmr_list) == 0:
                return []
            return p_map(
                BuildSpectrumNMR.build_one,
                nmr_list,
                [self.vocab_peakwidth] * len(nmr_list),
                [self.vocab_split] * len(nmr_list),
                [self.vocab_integral] * len(nmr_list),
                [self.seq_len_H1] * len(nmr_list),
                [self.seq_len_C13] * len(nmr_list),
                [self.j_len] * len(nmr_list),
                [self.dtype] * len(nmr_list),
                num_cpus=self.num_cpus,
                desc="Building spectrums",
                dynamic_ncols=True,
                mininterval=0.2,
            )
        # single sample
        return BuildSpectrumNMR.build_one(
            nmr_list,
            self.vocab_peakwidth,
            self.vocab_split,
            self.vocab_integral,
            self.seq_len_H1,
            self.seq_len_C13,
            self.j_len,
            self.dtype,
        )

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

import os

from pymatgen.io.cif import CifWriter

from ppmat.models.matinvent.rewards.calculators import Calculator


class DFTCalc(Calculator):
    """Write structures as CIFs for an external DFT run and read values back.

    Results belong in the exported batch directory, one float per structure.
    """

    def calc(self, structures, label="tmp"):
        cif_dir, out_path = self.results_path(structures, label)
        for i, struc in enumerate(structures):
            CifWriter(struc).write_file(os.path.join(cif_dir, f"{i}.cif"))
        if not os.path.isfile(out_path):
            raise FileNotFoundError(
                f"DFT results are missing: {out_path}. CIF files for the external DFT "
                f"run were written to {cif_dir}; write one value per structure to "
                f"{out_path} and rerun."
            )
        return self.read_results(structures, out_path)

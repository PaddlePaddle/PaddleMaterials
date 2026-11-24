# DeepH inference

DeepH predicts Hamiltonian blocks from processed structures containing
`lat.dat`, `site_positions.dat`, `element.dat`, `orbital_types.dat`,
and `rc.npz`. Hamiltonian labels (`rh.npz`) are not required.

```python
from omegaconf import OmegaConf
from ppmat.predictor.deeph import DeepHPredictor

task = OmegaConf.to_container(
    OmegaConf.load("ppmatSim/configs/Task/deeph_dft_inference.yaml"),
    resolve=True,
)
predictor = DeepHPredictor(
    config_path="electronic_structure/configs/deeph/deeph_graphene.yaml",
    checkpoint_path="data/deeph/graphene/best.pdparams",
    device="gpu:0",
)
predictor.load_inference_model(interface_type="deeph_dft")
predictor.predict_hamiltonian(
    processed_dir="data/deeph/graphene/processed/1000",
    output_path="rh_pred.npz",
    orbital=task["orbital"],
    species=task["species"],
    num_l=task["num_l"],
)
```

The output contains periodic atom-pair keys `[Rx, Ry, Rz, atom_i, atom_j]`
and 13-by-13 Hamiltonian blocks for the graphene checkpoint.

## OpenMX calculation

Configure the OpenMX overlap executable and basis data before running:

Use the overlap-output build described by the
[overlap-only OpenMX instructions](https://github.com/mzjb/overlap-only-OpenMX#usage).
A regular OpenMX executable does not provide the required `overlaps_*.h5`
outputs. The calculator config selects `C6.0-s2p2d1` and `C_PBE19`;
the corresponding PAO and VPS files must be available in `DFT_DATA19`.

```bash
export OPENMX_DFT_DATA_PATH=/path/to/DFT_DATA19

python ppmatSim/main.py --config-name deeph_openmx \
  Model.config_path=electronic_structure/configs/deeph/deeph_graphene.yaml \
  Model.checkpoint_path=data/deeph/graphene/best.pdparams \
  Calculator.command='mpirun -np 4 /path/to/openmx_overlap' \
  System.file_path=/path/to/inference_cifs
```

Inference parameters, including the orbital mapping, are configured in
[the DeepH task](../Task/deeph_dft_inference.yaml). External calculator
parameters are configured in
[the OpenMX calculator](../Calculator/deeph_openmx.yaml).

Each structure produces OpenMX output, processed geometry and `rh_pred.npz`
under `deeph_dft_results/<sample_id>/`. The graphene checkpoint uses
the carbon `C6.0-s2p2d1` basis.

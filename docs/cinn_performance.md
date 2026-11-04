# CINN performance matrix

[中文版](cinn_performance_ch.md)

## Conclusion

Under the tested workloads, all 65 registered weights with a CINN path
completed in eager and CINN modes. The median warm speedup across all weights
was 1.11x, ranging from 0.51x to 4.36x.

- Families where every registered weight was faster, with median speedup:
  SphereNet MD17 3.24x, MatterSim 2.85x, SFIN 2.23x, CHGNet 2.07x, DimeNet++
  1.87x, SphereNet QM9 1.47x, DiffCSP 1.32x, and DiffNMR 1.11x.
- Families where every registered weight was slower, with median speedup:
  ComFormer 0.83x, MatterGen 0.71x, and MEGNet 0.63x.

## Method

- Environment: Paddle 3.3.1, NVIDIA A100-SXM4-40GB GPUs, and an Intel Xeon Gold
  6148 host.
- Workloads: repository prediction and sampling examples. Structure generators
  sampled eight atoms for two denoising steps; DiffNMR ran complete reverse
  diffusion.
- Timing: model loading, checkpoint loading, and input conversion were excluded.
  GPU synchronization bracketed each call. Warm results are the median of ten
  identically seeded calls; the first CINN call is reported separately.
- Execution: each weight ran in a separate process, with tests running
  concurrently.

| Metric | Definition |
| --- | --- |
| First CINN | Compilation plus the first execution |
| Warm | Median of ten executions after the first call |
| Speedup | `eager warm / CINN warm`; values above 1 are faster |
| Compile estimate | `first CINN (s) - CINN warm (ms) / 1000` |
| Break-even | `ceil(compile estimate (s) * 1000 / (eager warm - CINN warm) (ms))`; `never` when warm CINN is not faster |

Compile estimates and break-even counts use unrounded timings; displayed times
are rounded.

## Family results

Family values are medians; brackets show the speedup range across registered
weights in the family.

| Family | Weights | Faster | First CINN median (s) | Eager warm (ms) | CINN warm (ms) | Median speedup [range] |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CHGNet | 1 | 1 | 143.703 | 39.511 | 19.048 | 2.07x [2.07x, 2.07x] |
| ComFormer | 8 | 0 | 54.159 | 8.832 | 10.408 | 0.83x [0.63x, 0.94x] |
| DiffCSP | 1 | 1 | 22.063 | 42.967 | 32.509 | 1.32x [1.32x, 1.32x] |
| DiffNMR | 1 | 1 | 132.210 | 19,624.969 | 17,736.939 | 1.11x [1.11x, 1.11x] |
| DimeNet++ | 4 | 4 | 181.085 | 22.395 | 12.052 | 1.87x [1.59x, 2.01x] |
| MatterGen | 16 | 0 | 266.045 | 368.771 | 512.713 | 0.71x [0.65x, 0.89x] |
| MatterSim | 2 | 2 | 163.861 | 39.482 | 14.154 | 2.85x [2.53x, 3.16x] |
| MEGNet | 8 | 0 | 49.942 | 12.442 | 20.326 | 0.63x [0.51x, 0.75x] |
| SFIN | 4 | 4 | 23.689 | 38.419 | 17.536 | 2.23x [1.76x, 2.66x] |
| SphereNet MD17 | 8 | 8 | 241.368 | 56.755 | 17.303 | 3.24x [2.24x, 4.36x] |
| SphereNet QM9 | 12 | 12 | 135.946 | 21.272 | 14.755 | 1.47x [1.29x, 1.67x] |

## Per-weight results

| Registered weight | First CINN (s) | Compile estimate (s) | Eager warm (ms) | CINN warm (ms) | Speedup | Break-even calls |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `chgnet_mptrj` | 143.703 | 143.684 | 39.511 | 19.048 | 2.07x | 7,022 |
| `comformer_jarvis_alex_pbe_2d_all_e_form` | 54.088 | 54.074 | 8.462 | 13.435 | 0.63x | never |
| `comformer_jarvis_dft_2d_e_form` | 54.793 | 54.782 | 8.965 | 11.510 | 0.78x | never |
| `comformer_jarvis_dft_3d_e_form` | 54.067 | 54.056 | 8.403 | 11.188 | 0.75x | never |
| `comformer_mp2018_train_60k_G` | 52.843 | 52.834 | 8.853 | 9.425 | 0.94x | never |
| `comformer_mp2018_train_60k_K` | 52.949 | 52.939 | 8.895 | 9.542 | 0.93x | never |
| `comformer_mp2018_train_60k_band_gap` | 54.231 | 54.220 | 8.810 | 10.750 | 0.82x | never |
| `comformer_mp2018_train_60k_e_form` | 54.615 | 54.605 | 8.884 | 10.004 | 0.89x | never |
| `comformer_mp2024_train_130k_e_form` | 55.211 | 55.201 | 8.468 | 10.065 | 0.84x | never |
| `diffcsp_mp20` | 22.063 | 22.031 | 42.967 | 32.509 | 1.32x | 2,107 |
| `diffnmr_msdnmr_nless15` | 132.210 | 114.473 | 19,624.969 | 17,736.939 | 1.11x | 61 |
| `dimenetpp_mp2018_train_60k_G` | 180.093 | 180.079 | 21.759 | 13.703 | 1.59x | 22,352 |
| `dimenetpp_mp2018_train_60k_K` | 171.299 | 171.287 | 22.774 | 11.985 | 1.90x | 15,877 |
| `dimenetpp_mp2018_train_60k_band_gap` | 182.077 | 182.065 | 22.285 | 12.119 | 1.84x | 17,909 |
| `dimenetpp_mp2018_train_60k_e_form` | 182.780 | 182.768 | 22.505 | 11.172 | 2.01x | 16,126 |
| `mattergen_alex_mp20` | 270.002 | 269.720 | 185.655 | 282.416 | 0.66x | never |
| `mattergen_alex_mp20_chemical_system` | 254.266 | 253.801 | 366.483 | 464.109 | 0.79x | never |
| `mattergen_alex_mp20_chemical_system_energy_above_hull` | 280.498 | 280.009 | 347.134 | 488.762 | 0.71x | never |
| `mattergen_alex_mp20_dft_band_gap` | 296.511 | 295.984 | 470.514 | 527.441 | 0.89x | never |
| `mattergen_alex_mp20_dft_mag_density` | 269.997 | 269.470 | 371.652 | 527.079 | 0.71x | never |
| `mattergen_alex_mp20_dft_mag_density_hhi_score` | 268.470 | 267.927 | 383.777 | 543.615 | 0.71x | never |
| `mattergen_alex_mp20_ml_bulk_modulus` | 263.676 | 263.146 | 371.146 | 530.166 | 0.70x | never |
| `mattergen_alex_mp20_space_group` | 283.368 | 282.870 | 345.509 | 498.348 | 0.69x | never |
| `mattergen_ml2ddb` | 258.583 | 258.348 | 201.411 | 235.057 | 0.86x | never |
| `mattergen_ml2ddb_chemical_system` | 254.858 | 254.285 | 398.299 | 573.297 | 0.69x | never |
| `mattergen_ml2ddb_space_group` | 265.844 | 265.301 | 353.056 | 542.639 | 0.65x | never |
| `mattergen_mp20` | 256.684 | 256.426 | 186.634 | 258.112 | 0.72x | never |
| `mattergen_mp20_chemical_system` | 266.245 | 265.774 | 371.506 | 471.607 | 0.79x | never |
| `mattergen_mp20_dft_band_gap` | 262.027 | 261.484 | 371.059 | 543.468 | 0.68x | never |
| `mattergen_mp20_dft_bulk_modulus` | 266.645 | 266.164 | 373.397 | 481.629 | 0.78x | never |
| `mattergen_mp20_dft_mag_density` | 252.039 | 251.503 | 357.569 | 535.265 | 0.67x | never |
| `mattersim_1M` | 154.510 | 154.498 | 36.779 | 11.630 | 3.16x | 6,144 |
| `mattersim_5M` | 173.213 | 173.196 | 42.185 | 16.677 | 2.53x | 6,790 |
| `megnet_jarvis_alex_pbe_2d_all_e_form` | 48.958 | 48.938 | 12.186 | 19.689 | 0.62x | never |
| `megnet_jarvis_dft_2d_e_form` | 48.735 | 48.716 | 11.966 | 18.777 | 0.64x | never |
| `megnet_jarvis_dft_3d_e_form` | 49.736 | 49.711 | 12.364 | 24.403 | 0.51x | never |
| `megnet_mp2018_train_60k_G` | 52.215 | 52.194 | 14.147 | 20.818 | 0.68x | never |
| `megnet_mp2018_train_60k_K` | 49.096 | 49.074 | 15.032 | 21.569 | 0.70x | never |
| `megnet_mp2018_train_60k_band_gap` | 52.039 | 52.016 | 12.520 | 22.826 | 0.55x | never |
| `megnet_mp2018_train_60k_e_form` | 50.148 | 50.128 | 11.917 | 19.834 | 0.60x | never |
| `megnet_mp2024_train_130k_e_form` | 51.522 | 51.502 | 14.767 | 19.694 | 0.75x | never |
| `sfin_bf_detect` | 23.778 | 23.761 | 38.683 | 16.694 | 2.32x | 1,081 |
| `sfin_bf_enhance` | 23.862 | 23.845 | 38.154 | 17.798 | 2.14x | 1,172 |
| `sfin_haadf_detect` | 23.600 | 23.583 | 45.907 | 17.274 | 2.66x | 824 |
| `sfin_haadf_enhance` | 22.830 | 22.808 | 37.582 | 21.294 | 1.76x | 1,401 |
| `spherenet_md17_aspirin` | 261.710 | 261.695 | 57.038 | 15.136 | 3.77x | 6,246 |
| `spherenet_md17_benzene_old` | 237.398 | 237.377 | 59.966 | 21.351 | 2.81x | 6,148 |
| `spherenet_md17_ethanol` | 245.666 | 245.648 | 56.473 | 18.842 | 3.00x | 6,528 |
| `spherenet_md17_malonaldehyde` | 245.590 | 245.571 | 57.761 | 19.217 | 3.01x | 6,372 |
| `spherenet_md17_naphthalene` | 240.196 | 240.180 | 59.384 | 15.322 | 3.88x | 5,452 |
| `spherenet_md17_salicylic` | 242.541 | 242.518 | 51.359 | 22.921 | 2.24x | 8,528 |
| `spherenet_md17_toluene` | 237.127 | 237.112 | 54.688 | 15.763 | 3.47x | 6,092 |
| `spherenet_md17_uracil` | 237.772 | 237.759 | 56.281 | 12.912 | 4.36x | 5,483 |
| `spherenet_qm9_Cv` | 133.758 | 133.743 | 20.434 | 14.703 | 1.39x | 23,338 |
| `spherenet_qm9_G` | 136.702 | 136.687 | 20.460 | 15.556 | 1.32x | 27,873 |
| `spherenet_qm9_H` | 131.905 | 131.890 | 23.071 | 14.744 | 1.56x | 15,841 |
| `spherenet_qm9_U` | 133.085 | 133.071 | 21.841 | 14.200 | 1.54x | 17,416 |
| `spherenet_qm9_U0` | 137.800 | 137.785 | 21.817 | 14.766 | 1.48x | 19,541 |
| `spherenet_qm9_alpha` | 140.015 | 140.000 | 22.533 | 14.971 | 1.51x | 18,515 |
| `spherenet_qm9_gap` | 135.691 | 135.675 | 20.394 | 15.852 | 1.29x | 29,867 |
| `spherenet_qm9_homo` | 138.437 | 138.422 | 21.446 | 14.431 | 1.49x | 19,732 |
| `spherenet_qm9_lumo` | 136.201 | 136.187 | 20.624 | 14.188 | 1.45x | 21,159 |
| `spherenet_qm9_mu` | 138.283 | 138.268 | 24.879 | 14.929 | 1.67x | 13,896 |
| `spherenet_qm9_r2` | 88.877 | 88.863 | 19.404 | 13.825 | 1.40x | 15,929 |
| `spherenet_qm9_zpve` | 134.760 | 134.745 | 21.098 | 14.799 | 1.43x | 21,391 |

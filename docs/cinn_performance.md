# CINN performance matrix

[中文版](cinn_performance_ch.md)

## Conclusion

Under the tested workloads, all 79 registered weights with a CINN path
completed in eager and CINN modes. The median warm speedup across all weights
was 1.41x, ranging from 0.57x to 4.15x.

- Families where every registered weight was faster, with median speedup:
  MatterSim 3.82x, SphereNet MD17 2.93x, SFIN 2.32x, CHGNet 2.09x, DimeNet++
  MP2018 2.01x, DimeNet++ QM9 1.93x, SphereNet QM9 1.41x, DiffCSP 1.17x, and
  DiffNMR 1.09x.
- Families where every registered weight was slower, with median speedup:
  ComFormer 0.84x, MatterGen 0.68x, and MEGNet 0.63x.

## Method

- Environment: Paddle 3.3.1, NVIDIA A100-SXM4-40GB GPUs, and an Intel Xeon Gold
  6148 host.
- Workloads: repository prediction and sampling examples. Structure generators
  sampled eight atoms for two denoising steps; DiffNMR ran complete reverse
  diffusion.
- Timing: model loading, checkpoint loading, and input conversion were excluded.
  GPU synchronization bracketed each call. Warm results are the median of ten
  identically seeded calls; the first CINN call is reported separately.
- Execution: each weight ran in a separate process. Three passes ran on three
  GPUs, one pass per GPU; each pass was bound to the CPU cores of its GPU's NUMA
  node and tested the weights one at a time. Each reported time is the median of
  the three passes.

| Metric | Definition |
| --- | --- |
| First CINN | Compilation plus the first execution |
| Warm | Median of ten executions after the first call |
| Speedup | `eager warm / CINN warm`; values above 1 are faster |
| Compile estimate | `first CINN (s) - CINN warm (ms) / 1000` |
| Break-even | `ceil(compile estimate (s) * 1000 / (eager warm - CINN warm) (ms))`; `never` when warm CINN is not faster |

Compile estimates and break-even counts are computed from the displayed timings.

## Family results

Family values are medians; brackets show the speedup range across registered
weights in the family.

| Family | Weights | Faster | First CINN median (s) | Eager warm (ms) | CINN warm (ms) | Median speedup [range] |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CHGNet | 1 | 1 | 137.296 | 26.336 | 12.621 | 2.09x [2.09x, 2.09x] |
| ComFormer | 8 | 0 | 52.733 | 8.311 | 9.835 | 0.84x [0.71x, 0.91x] |
| DiffCSP | 2 | 2 | 21.728 | 39.480 | 33.615 | 1.17x [1.16x, 1.18x] |
| DiffNMR | 1 | 1 | 126.865 | 19,685.083 | 18,033.541 | 1.09x [1.09x, 1.09x] |
| DimeNet++ MP2018 | 4 | 4 | 174.935 | 22.745 | 11.255 | 2.01x [1.89x, 2.16x] |
| DimeNet++ QM9 | 12 | 12 | 175.312 | 21.207 | 10.843 | 1.93x [1.71x, 2.06x] |
| MatterGen | 17 | 0 | 258.981 | 340.562 | 503.302 | 0.68x [0.63x, 0.76x] |
| MatterSim | 2 | 2 | 149.109 | 36.900 | 9.617 | 3.82x [3.50x, 4.15x] |
| MEGNet | 8 | 0 | 53.760 | 12.113 | 19.769 | 0.63x [0.57x, 0.65x] |
| SFIN | 4 | 4 | 26.193 | 38.263 | 16.195 | 2.32x [2.01x, 2.52x] |
| SphereNet MD17 | 8 | 8 | 260.129 | 47.980 | 16.946 | 2.93x [2.39x, 3.18x] |
| SphereNet QM9 | 12 | 12 | 143.700 | 21.154 | 14.883 | 1.41x [1.33x, 1.46x] |

## Per-weight results

| Registered weight | First CINN (s) | Compile estimate (s) | Eager warm (ms) | CINN warm (ms) | Speedup | Break-even calls |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `chgnet_mptrj` | 137.296 | 137.283 | 26.336 | 12.621 | 2.09x | 10,010 |
| `comformer_jarvis_alex_pbe_2d_all_e_form` | 53.234 | 53.224 | 8.030 | 9.895 | 0.81x | never |
| `comformer_jarvis_dft_2d_e_form` | 53.752 | 53.742 | 7.984 | 9.792 | 0.81x | never |
| `comformer_jarvis_dft_3d_e_form` | 53.904 | 53.892 | 8.260 | 11.610 | 0.71x | never |
| `comformer_mp2018_train_60k_G` | 50.869 | 50.860 | 8.528 | 9.364 | 0.91x | never |
| `comformer_mp2018_train_60k_K` | 52.233 | 52.223 | 8.366 | 9.627 | 0.87x | never |
| `comformer_mp2018_train_60k_band_gap` | 53.459 | 53.449 | 8.149 | 10.051 | 0.81x | never |
| `comformer_mp2018_train_60k_e_form` | 51.646 | 51.636 | 8.491 | 9.879 | 0.86x | never |
| `comformer_mp2024_train_130k_e_form` | 51.825 | 51.815 | 8.361 | 9.578 | 0.87x | never |
| `diffcsp_mp20` | 21.572 | 21.539 | 38.599 | 33.149 | 1.16x | 3,953 |
| `diffcsp_mp20_ab_initio` | 21.883 | 21.849 | 40.362 | 34.082 | 1.18x | 3,480 |
| `diffnmr_msdnmr_nless15` | 126.865 | 108.831 | 19,685.083 | 18,033.541 | 1.09x | 66 |
| `dimenetpp_mp2018_train_60k_G` | 173.587 | 173.576 | 20.954 | 11.063 | 1.89x | 17,549 |
| `dimenetpp_mp2018_train_60k_K` | 174.487 | 174.476 | 21.633 | 11.331 | 1.91x | 16,937 |
| `dimenetpp_mp2018_train_60k_band_gap` | 176.005 | 175.994 | 24.346 | 11.251 | 2.16x | 13,440 |
| `dimenetpp_mp2018_train_60k_e_form` | 175.383 | 175.372 | 23.857 | 11.259 | 2.12x | 13,921 |
| `dimenetpp_qm9_Cv` | 184.914 | 184.902 | 20.878 | 12.006 | 1.74x | 20,842 |
| `dimenetpp_qm9_G` | 183.871 | 183.860 | 21.867 | 10.797 | 2.02x | 16,609 |
| `dimenetpp_qm9_H` | 182.481 | 182.470 | 21.412 | 11.126 | 1.93x | 17,740 |
| `dimenetpp_qm9_U` | 181.177 | 181.164 | 21.531 | 12.563 | 1.71x | 20,202 |
| `dimenetpp_qm9_U0` | 175.980 | 175.969 | 20.820 | 10.840 | 1.92x | 17,633 |
| `dimenetpp_qm9_alpha` | 174.686 | 174.675 | 21.398 | 10.690 | 2.00x | 16,313 |
| `dimenetpp_qm9_gap` | 175.422 | 175.411 | 20.922 | 10.805 | 1.94x | 17,339 |
| `dimenetpp_qm9_homo` | 174.717 | 174.706 | 20.895 | 10.826 | 1.93x | 17,351 |
| `dimenetpp_qm9_lumo` | 175.203 | 175.192 | 20.276 | 11.105 | 1.83x | 19,103 |
| `dimenetpp_qm9_mu` | 174.907 | 174.896 | 21.016 | 10.846 | 1.94x | 17,198 |
| `dimenetpp_qm9_r2` | 174.449 | 174.438 | 21.489 | 11.036 | 1.95x | 16,688 |
| `dimenetpp_qm9_zpve` | 174.874 | 174.863 | 21.795 | 10.560 | 2.06x | 15,565 |
| `mattergen_alex_mp20` | 256.315 | 256.081 | 172.278 | 234.290 | 0.73x | never |
| `mattergen_alex_mp20_chemical_system` | 264.755 | 264.298 | 343.203 | 456.753 | 0.75x | never |
| `mattergen_alex_mp20_chemical_system_energy_above_hull` | 276.666 | 276.182 | 345.945 | 483.861 | 0.71x | never |
| `mattergen_alex_mp20_dft_band_gap` | 253.012 | 252.499 | 328.696 | 512.786 | 0.64x | never |
| `mattergen_alex_mp20_dft_mag_density` | 253.310 | 252.786 | 339.267 | 523.796 | 0.65x | never |
| `mattergen_alex_mp20_dft_mag_density_hhi_score` | 256.601 | 256.099 | 341.793 | 501.565 | 0.68x | never |
| `mattergen_alex_mp20_ml_bulk_modulus` | 254.362 | 253.841 | 337.521 | 520.516 | 0.65x | never |
| `mattergen_alex_mp20_space_group` | 281.360 | 280.833 | 340.562 | 527.493 | 0.65x | never |
| `mattergen_ml2ddb` | 248.953 | 248.723 | 175.187 | 230.254 | 0.76x | never |
| `mattergen_ml2ddb_chemical_system` | 264.622 | 264.105 | 338.964 | 516.656 | 0.66x | never |
| `mattergen_ml2ddb_space_group` | 254.549 | 253.985 | 353.574 | 564.080 | 0.63x | never |
| `mattergen_mp20` | 253.121 | 252.873 | 176.741 | 248.288 | 0.71x | never |
| `mattergen_mp20_chemical_system` | 262.961 | 262.466 | 348.049 | 494.905 | 0.70x | never |
| `mattergen_mp20_csp` | 268.722 | 268.459 | 179.104 | 263.406 | 0.68x | never |
| `mattergen_mp20_dft_band_gap` | 258.981 | 258.478 | 365.091 | 503.302 | 0.72x | never |
| `mattergen_mp20_dft_bulk_modulus` | 259.948 | 259.398 | 351.841 | 549.921 | 0.64x | never |
| `mattergen_mp20_dft_mag_density` | 261.062 | 260.529 | 351.732 | 533.397 | 0.66x | never |
| `mattersim_1M` | 141.675 | 141.666 | 32.414 | 9.253 | 3.50x | 6,117 |
| `mattersim_5M` | 156.544 | 156.534 | 41.386 | 9.982 | 4.15x | 4,985 |
| `megnet_jarvis_alex_pbe_2d_all_e_form` | 47.983 | 47.964 | 11.926 | 18.902 | 0.63x | never |
| `megnet_jarvis_dft_2d_e_form` | 49.512 | 49.493 | 12.289 | 18.981 | 0.65x | never |
| `megnet_jarvis_dft_3d_e_form` | 49.362 | 49.343 | 12.011 | 18.549 | 0.65x | never |
| `megnet_mp2018_train_60k_G` | 53.957 | 53.935 | 13.716 | 21.750 | 0.63x | never |
| `megnet_mp2018_train_60k_K` | 54.010 | 53.988 | 13.596 | 21.878 | 0.62x | never |
| `megnet_mp2018_train_60k_band_gap` | 54.819 | 54.798 | 11.968 | 20.904 | 0.57x | never |
| `megnet_mp2018_train_60k_e_form` | 54.508 | 54.488 | 11.857 | 20.410 | 0.58x | never |
| `megnet_mp2024_train_130k_e_form` | 53.562 | 53.543 | 12.216 | 19.127 | 0.64x | never |
| `sfin_bf_detect` | 25.757 | 25.738 | 39.045 | 19.379 | 2.02x | 1,309 |
| `sfin_bf_enhance` | 26.325 | 26.308 | 37.119 | 16.899 | 2.20x | 1,302 |
| `sfin_haadf_detect` | 26.115 | 26.100 | 38.595 | 15.326 | 2.52x | 1,122 |
| `sfin_haadf_enhance` | 26.272 | 26.257 | 37.931 | 15.491 | 2.45x | 1,171 |
| `spherenet_md17_aspirin` | 259.083 | 259.067 | 47.250 | 15.508 | 3.05x | 8,162 |
| `spherenet_md17_benzene_old` | 255.978 | 255.962 | 48.587 | 15.567 | 3.12x | 7,752 |
| `spherenet_md17_ethanol` | 260.034 | 260.017 | 49.112 | 16.827 | 2.92x | 8,054 |
| `spherenet_md17_malonaldehyde` | 260.490 | 260.470 | 47.662 | 19.892 | 2.40x | 9,380 |
| `spherenet_md17_naphthalene` | 260.223 | 260.208 | 48.298 | 15.202 | 3.18x | 7,863 |
| `spherenet_md17_salicylic` | 257.560 | 257.542 | 53.548 | 18.251 | 2.93x | 7,297 |
| `spherenet_md17_toluene` | 261.926 | 261.907 | 44.640 | 18.647 | 2.39x | 10,077 |
| `spherenet_md17_uracil` | 261.918 | 261.901 | 47.446 | 17.066 | 2.78x | 8,621 |
| `spherenet_qm9_Cv` | 143.733 | 143.718 | 21.131 | 14.631 | 1.44x | 22,111 |
| `spherenet_qm9_G` | 143.666 | 143.651 | 20.882 | 14.841 | 1.41x | 23,780 |
| `spherenet_qm9_H` | 144.182 | 144.167 | 21.233 | 15.162 | 1.40x | 23,747 |
| `spherenet_qm9_U` | 142.687 | 142.672 | 20.866 | 14.664 | 1.42x | 23,005 |
| `spherenet_qm9_U0` | 144.034 | 144.020 | 21.200 | 14.491 | 1.46x | 21,467 |
| `spherenet_qm9_alpha` | 146.084 | 146.069 | 21.294 | 14.959 | 1.42x | 23,058 |
| `spherenet_qm9_gap` | 142.761 | 142.746 | 20.481 | 15.411 | 1.33x | 28,155 |
| `spherenet_qm9_homo` | 142.876 | 142.861 | 21.309 | 14.793 | 1.44x | 21,925 |
| `spherenet_qm9_lumo` | 144.167 | 144.152 | 21.176 | 15.019 | 1.41x | 23,413 |
| `spherenet_qm9_mu` | 142.856 | 142.841 | 20.986 | 14.925 | 1.41x | 23,568 |
| `spherenet_qm9_r2` | 94.262 | 94.248 | 19.075 | 13.871 | 1.38x | 18,111 |
| `spherenet_qm9_zpve` | 143.778 | 143.763 | 21.282 | 15.099 | 1.41x | 23,252 |

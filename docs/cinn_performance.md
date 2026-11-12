# CINN performance matrix

[中文版](cinn_performance_ch.md)

## Conclusion

Under the tested workloads, all 79 registered weights with a CINN path
completed in eager and CINN modes. The median warm speedup across all weights
was 2.59x, ranging from 0.92x to 5.14x.

- Families where every registered weight was faster, with median speedup:
  DimeNet++ QM9 4.78x, DimeNet++ MP2018 4.61x, MatterSim 3.51x, SphereNet MD17
  3.04x, SphereNet QM9 2.82x, ComFormer 2.59x, SFIN 2.43x, DiffCSP 2.40x,
  DiffNMR 2.14x, CHGNet 2.08x, and MEGNet 1.70x.
- Families with both faster and slower weights, with median speedup: MatterGen
  1.19x (14/17 faster).

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

Compile estimates and break-even counts use unrounded timings; displayed times
are rounded.

## Family results

Family values are medians; brackets show the speedup range across registered
weights in the family.

| Family | Weights | Faster | First CINN median (s) | Eager warm (ms) | CINN warm (ms) | Median speedup [range] |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CHGNet | 1 | 1 | 135.170 | 26.087 | 12.532 | 2.08x [2.08x, 2.08x] |
| ComFormer | 8 | 8 | 55.149 | 7.969 | 3.080 | 2.59x [2.35x, 2.66x] |
| DiffCSP | 2 | 2 | 25.289 | 38.124 | 15.923 | 2.40x [2.33x, 2.47x] |
| DiffNMR | 1 | 1 | 159.365 | 19,071.246 | 8,924.011 | 2.14x [2.14x, 2.14x] |
| DimeNet++ MP2018 | 4 | 4 | 81.018 | 20.567 | 4.453 | 4.61x [4.13x, 5.14x] |
| DimeNet++ QM9 | 12 | 12 | 79.850 | 19.590 | 4.089 | 4.78x [3.56x, 5.09x] |
| MatterGen | 17 | 14 | 344.952 | 312.138 | 259.120 | 1.19x [0.92x, 1.62x] |
| MatterSim | 2 | 2 | 148.518 | 37.067 | 10.524 | 3.51x [3.43x, 3.60x] |
| MEGNet | 8 | 8 | 73.413 | 11.810 | 7.197 | 1.70x [1.46x, 1.72x] |
| SFIN | 4 | 4 | 25.638 | 38.205 | 15.634 | 2.43x [2.40x, 2.59x] |
| SphereNet MD17 | 8 | 8 | 247.740 | 44.202 | 14.320 | 3.04x [2.57x, 4.19x] |
| SphereNet QM9 | 12 | 12 | 121.554 | 19.745 | 6.889 | 2.82x [2.47x, 2.99x] |

## Per-weight results

| Registered weight | First CINN (s) | Compile estimate (s) | Eager warm (ms) | CINN warm (ms) | Speedup | Break-even calls |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `chgnet_mptrj` | 135.170 | 135.158 | 26.087 | 12.532 | 2.08x | 9,972 |
| `comformer_jarvis_alex_pbe_2d_all_e_form` | 55.331 | 55.328 | 7.925 | 3.054 | 2.59x | 11,359 |
| `comformer_jarvis_dft_2d_e_form` | 55.514 | 55.511 | 8.036 | 3.106 | 2.59x | 11,262 |
| `comformer_jarvis_dft_3d_e_form` | 55.095 | 55.092 | 7.994 | 3.021 | 2.65x | 11,079 |
| `comformer_mp2018_train_60k_G` | 52.626 | 52.623 | 7.884 | 3.361 | 2.35x | 11,634 |
| `comformer_mp2018_train_60k_K` | 52.491 | 52.487 | 7.969 | 3.290 | 2.42x | 11,218 |
| `comformer_mp2018_train_60k_band_gap` | 55.204 | 55.201 | 7.985 | 3.049 | 2.62x | 11,186 |
| `comformer_mp2018_train_60k_e_form` | 55.246 | 55.243 | 7.969 | 2.995 | 2.66x | 11,106 |
| `comformer_mp2024_train_130k_e_form` | 54.778 | 54.775 | 7.784 | 3.271 | 2.38x | 12,139 |
| `diffcsp_mp20` | 22.060 | 22.045 | 36.795 | 14.901 | 2.47x | 1,007 |
| `diffcsp_mp20_ab_initio` | 28.518 | 28.501 | 39.453 | 16.945 | 2.33x | 1,267 |
| `diffnmr_msdnmr_nless15` | 159.365 | 150.441 | 19,071.246 | 8,924.011 | 2.14x | 15 |
| `dimenetpp_mp2018_train_60k_G` | 80.509 | 80.505 | 20.920 | 4.416 | 4.74x | 4,878 |
| `dimenetpp_mp2018_train_60k_K` | 80.651 | 80.646 | 22.259 | 4.330 | 5.14x | 4,499 |
| `dimenetpp_mp2018_train_60k_band_gap` | 81.447 | 81.442 | 20.214 | 4.895 | 4.13x | 5,317 |
| `dimenetpp_mp2018_train_60k_e_form` | 81.385 | 81.381 | 20.115 | 4.491 | 4.48x | 5,209 |
| `dimenetpp_qm9_Cv` | 80.948 | 80.943 | 19.291 | 4.378 | 4.41x | 5,428 |
| `dimenetpp_qm9_G` | 78.906 | 78.902 | 20.648 | 4.054 | 5.09x | 4,756 |
| `dimenetpp_qm9_H` | 77.438 | 77.434 | 19.405 | 4.047 | 4.80x | 5,042 |
| `dimenetpp_qm9_U` | 77.814 | 77.809 | 19.449 | 5.458 | 3.56x | 5,562 |
| `dimenetpp_qm9_U0` | 79.948 | 79.944 | 19.939 | 4.382 | 4.55x | 5,139 |
| `dimenetpp_qm9_alpha` | 80.296 | 80.291 | 19.061 | 4.716 | 4.04x | 5,598 |
| `dimenetpp_qm9_gap` | 80.700 | 80.696 | 19.730 | 4.087 | 4.83x | 5,159 |
| `dimenetpp_qm9_homo` | 81.502 | 81.498 | 18.989 | 3.986 | 4.76x | 5,433 |
| `dimenetpp_qm9_lumo` | 80.958 | 80.954 | 20.404 | 4.091 | 4.99x | 4,963 |
| `dimenetpp_qm9_mu` | 79.752 | 79.747 | 19.406 | 4.890 | 3.97x | 5,494 |
| `dimenetpp_qm9_r2` | 78.661 | 78.657 | 19.921 | 4.015 | 4.96x | 4,946 |
| `dimenetpp_qm9_zpve` | 78.126 | 78.122 | 19.816 | 4.018 | 4.93x | 4,946 |
| `mattergen_alex_mp20` | 230.151 | 230.036 | 172.557 | 114.723 | 1.50x | 3,978 |
| `mattergen_alex_mp20_chemical_system` | 288.607 | 288.289 | 330.220 | 318.707 | 1.04x | 25,040 |
| `mattergen_alex_mp20_chemical_system_energy_above_hull` | 288.085 | 287.782 | 314.020 | 302.816 | 1.04x | 25,687 |
| `mattergen_alex_mp20_dft_band_gap` | 360.722 | 360.432 | 320.963 | 290.088 | 1.11x | 11,675 |
| `mattergen_alex_mp20_dft_mag_density` | 360.858 | 360.577 | 335.224 | 281.203 | 1.19x | 6,675 |
| `mattergen_alex_mp20_dft_mag_density_hhi_score` | 368.237 | 368.017 | 310.922 | 219.721 | 1.42x | 4,036 |
| `mattergen_alex_mp20_ml_bulk_modulus` | 359.882 | 359.613 | 299.475 | 268.653 | 1.11x | 11,668 |
| `mattergen_alex_mp20_space_group` | 264.998 | 264.776 | 328.361 | 222.398 | 1.48x | 2,499 |
| `mattergen_ml2ddb` | 332.360 | 332.237 | 199.439 | 123.041 | 1.62x | 4,349 |
| `mattergen_ml2ddb_chemical_system` | 276.026 | 275.703 | 296.373 | 322.426 | 0.92x | never |
| `mattergen_ml2ddb_space_group` | 360.008 | 359.676 | 344.641 | 331.837 | 1.04x | 28,090 |
| `mattergen_mp20` | 344.522 | 344.399 | 172.206 | 122.738 | 1.40x | 6,962 |
| `mattergen_mp20_chemical_system` | 296.203 | 295.881 | 308.778 | 322.060 | 0.96x | never |
| `mattergen_mp20_csp` | 344.952 | 344.756 | 186.600 | 195.774 | 0.95x | never |
| `mattergen_mp20_dft_band_gap` | 361.192 | 360.944 | 316.114 | 247.392 | 1.28x | 5,253 |
| `mattergen_mp20_dft_bulk_modulus` | 361.598 | 361.350 | 312.138 | 248.780 | 1.25x | 5,704 |
| `mattergen_mp20_dft_mag_density` | 350.550 | 350.291 | 315.443 | 259.120 | 1.22x | 6,220 |
| `mattersim_1M` | 142.105 | 142.095 | 33.114 | 9.659 | 3.43x | 6,059 |
| `mattersim_5M` | 154.932 | 154.920 | 41.020 | 11.389 | 3.60x | 5,229 |
| `megnet_jarvis_alex_pbe_2d_all_e_form` | 72.649 | 72.642 | 11.586 | 6.853 | 1.69x | 15,346 |
| `megnet_jarvis_dft_2d_e_form` | 73.419 | 73.412 | 11.764 | 6.920 | 1.70x | 15,155 |
| `megnet_jarvis_dft_3d_e_form` | 73.232 | 73.225 | 11.855 | 6.936 | 1.71x | 14,889 |
| `megnet_mp2018_train_60k_G` | 73.527 | 73.519 | 13.362 | 7.750 | 1.72x | 13,101 |
| `megnet_mp2018_train_60k_K` | 72.990 | 72.983 | 13.585 | 7.902 | 1.72x | 12,842 |
| `megnet_mp2018_train_60k_band_gap` | 73.785 | 73.778 | 11.561 | 7.213 | 1.60x | 16,970 |
| `megnet_mp2018_train_60k_e_form` | 73.727 | 73.718 | 11.884 | 8.164 | 1.46x | 19,815 |
| `megnet_mp2024_train_130k_e_form` | 73.407 | 73.400 | 11.658 | 7.181 | 1.62x | 16,395 |
| `sfin_bf_detect` | 25.647 | 25.631 | 38.244 | 15.785 | 2.42x | 1,142 |
| `sfin_bf_enhance` | 26.189 | 26.173 | 38.165 | 15.919 | 2.40x | 1,177 |
| `sfin_haadf_detect` | 25.308 | 25.292 | 37.681 | 15.483 | 2.43x | 1,140 |
| `sfin_haadf_enhance` | 25.629 | 25.614 | 39.420 | 15.192 | 2.59x | 1,058 |
| `spherenet_md17_aspirin` | 249.762 | 249.745 | 44.350 | 17.189 | 2.58x | 9,195 |
| `spherenet_md17_benzene_old` | 250.148 | 250.135 | 45.457 | 12.068 | 3.77x | 7,492 |
| `spherenet_md17_ethanol` | 245.667 | 245.651 | 44.533 | 15.682 | 2.84x | 8,515 |
| `spherenet_md17_malonaldehyde` | 252.529 | 252.514 | 44.054 | 14.932 | 2.95x | 8,671 |
| `spherenet_md17_naphthalene` | 247.338 | 247.324 | 42.775 | 13.708 | 3.12x | 8,509 |
| `spherenet_md17_salicylic` | 248.142 | 248.125 | 42.901 | 16.670 | 2.57x | 9,460 |
| `spherenet_md17_toluene` | 241.188 | 241.177 | 42.795 | 10.209 | 4.19x | 7,402 |
| `spherenet_md17_uracil` | 241.622 | 241.609 | 45.331 | 13.029 | 3.48x | 7,480 |
| `spherenet_qm9_Cv` | 121.943 | 121.935 | 20.059 | 7.513 | 2.67x | 9,720 |
| `spherenet_qm9_G` | 121.457 | 121.450 | 19.475 | 6.778 | 2.87x | 9,565 |
| `spherenet_qm9_H` | 122.981 | 122.974 | 20.067 | 6.758 | 2.97x | 9,240 |
| `spherenet_qm9_U` | 121.367 | 121.360 | 19.522 | 6.833 | 2.86x | 9,565 |
| `spherenet_qm9_U0` | 120.193 | 120.185 | 19.755 | 8.007 | 2.47x | 10,231 |
| `spherenet_qm9_alpha` | 120.728 | 120.721 | 20.237 | 6.764 | 2.99x | 8,961 |
| `spherenet_qm9_gap` | 120.033 | 120.026 | 19.460 | 7.189 | 2.71x | 9,782 |
| `spherenet_qm9_homo` | 122.154 | 122.147 | 19.869 | 7.529 | 2.64x | 9,899 |
| `spherenet_qm9_lumo` | 122.565 | 122.558 | 19.442 | 7.040 | 2.76x | 9,882 |
| `spherenet_qm9_mu` | 121.651 | 121.644 | 19.736 | 6.868 | 2.87x | 9,454 |
| `spherenet_qm9_r2` | 68.212 | 68.205 | 17.950 | 6.452 | 2.78x | 5,932 |
| `spherenet_qm9_zpve` | 124.597 | 124.590 | 20.266 | 6.909 | 2.93x | 9,328 |

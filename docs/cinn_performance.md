# CINN performance matrix

[中文版](cinn_performance_ch.md)

## Conclusion

Under the tested workloads, all 79 registered weights with a CINN path
completed in eager and CINN modes. The median warm speedup across all weights
was 2.50x, ranging from 0.96x to 5.15x.

- Families where every registered weight was faster, with median speedup:
  DimeNet++ MP2018 4.79x, DimeNet++ QM9 4.38x, MatterSim 3.53x, SphereNet MD17
  3.13x, SphereNet QM9 2.87x, SFIN 2.45x, ComFormer 2.43x, DiffCSP 2.12x,
  DiffNMR 2.06x, CHGNet 1.96x, and MEGNet 1.66x.
- Families with both faster and slower weights, with median speedup: MatterGen
  1.19x (13/17 faster).

## Method

- Measured model sources: PR #332 at `f5ec186`, remeasured on 16–17 November
  2026 (Asia/Shanghai). The remote working tree's 197 Python files under `ppmat/` were
  verified against that revision, as was the benchmark harness. The snapshot's
  Git metadata records `dd87607` plus working-tree changes; the file comparison
  establishes the measured model revision.
- Environment: Paddle 3.3.1, NVIDIA A100-SXM4-40GB GPUs, and an Intel Xeon Gold
  6148 host. The node was shared; CPU affinity did not provide exclusive CPU use.
- Workloads: repository prediction and sampling examples. Structure generators
  sampled eight atoms for two denoising steps; DiffNMR ran complete reverse
  diffusion (500 steps).
- Timing: model loading, checkpoint loading, and input conversion were excluded.
  GPU synchronization bracketed each call. The first CINN call is reported
  separately; warm results are the median of the ten calls immediately after
  it. The random seeds are reset for each call. There is no separate
  warm-up-stability gate in this protocol.
- Execution: each weight ran in a separate process. Three passes ran on three
  GPUs, one pass per GPU; each pass was bound to the CPU cores of its GPU's NUMA
  node and tested the weights one at a time. Each reported time is the median of
  the three passes. Each per-weight speedup is the ratio of the separately
  computed three-pass eager and CINN medians; family speedups are medians of
  those per-weight ratios.

| Metric | Definition |
| --- | --- |
| First CINN | First call, including initial compilation and execution |
| Warm | Median of the ten calls immediately after the first call |
| Speedup | `eager warm / CINN warm`; values above 1 are faster |
| First-call extra-cost estimate | `first CINN (s) - CINN warm (ms) / 1000` |
| Estimated break-even calls | `ceil((first CINN - CINN warm) / (eager warm - CINN warm))`, using consistent units; `never` when warm CINN is not faster |

The break-even estimate counts total model calls and uses the extra cost of the
first call. It compares `F + (N - 1)C` with `NE`, where `F` is the first CINN
call, `C` is CINN warm, and `E` is eager warm; the extra cost of the first eager
call is not included. Subsequent-call samples retain any compilation work that
occurs within those calls. The estimate does not separately accumulate later
compilation costs. Estimates use unrounded timings; displayed times
are rounded. Faster counts and break-even eligibility use unrounded values;
a displayed 1.00x can lie on either side of parity.

### Attempt history

The initial 474 eager/CINN process attempts produced 473 successful timing
records. In pass 1, CINN for
`mattergen_alex_mp20_dft_mag_density_hhi_score` exited with SIGSEGV (139)
during a later call, in CINN symbolic-shape inference after its first call had
completed. The same weight passed in the other two initial passes. After all
three passes finished, only this failed slot was retried, on the same GPU with
the same code and harness, and passed. The tables therefore contain 474 valid
records from 475 process attempts; the original failed log is preserved. The
trace does not establish what triggered the additional compilation.

For example, MatterGen CSP's subsequent-ten medians were 0.151–0.174 s
across passes, while the longest calls were 111.582–125.573 s and the sums
of all ten calls were 112.944–127.137 s. These long calls remain in the raw
samples. The reported median is not a cumulative-cost measure.

## Family results

Family values are medians; brackets show the speedup range across registered
weights in the family.

| Family | Weights | Faster | First CINN median (s) | Eager warm (ms) | CINN warm (ms) | Median speedup [range] |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CHGNet | 1 | 1 | 134.162 | 25.842 | 13.153 | 1.96x [1.96x, 1.96x] |
| ComFormer | 8 | 8 | 53.464 | 7.430 | 3.055 | 2.43x [2.18x, 2.50x] |
| DiffCSP | 2 | 2 | 23.638 | 34.483 | 16.303 | 2.12x [2.10x, 2.13x] |
| DiffNMR | 1 | 1 | 163.490 | 19,127.407 | 9,279.350 | 2.06x [2.06x, 2.06x] |
| DimeNet++ MP2018 | 4 | 4 | 77.639 | 20.899 | 4.357 | 4.79x [4.45x, 5.15x] |
| DimeNet++ QM9 | 12 | 12 | 75.339 | 19.024 | 4.329 | 4.38x [3.79x, 4.77x] |
| MatterGen | 17 | 13 | 348.575 | 326.272 | 274.224 | 1.19x [0.96x, 1.57x] |
| MatterSim | 2 | 2 | 157.544 | 34.751 | 9.845 | 3.53x [3.52x, 3.54x] |
| MEGNet | 8 | 8 | 70.291 | 11.555 | 6.717 | 1.66x [1.62x, 1.77x] |
| SFIN | 4 | 4 | 25.566 | 38.312 | 15.670 | 2.45x [1.94x, 2.50x] |
| SphereNet MD17 | 8 | 8 | 251.388 | 44.110 | 14.344 | 3.13x [2.40x, 3.70x] |
| SphereNet QM9 | 12 | 12 | 122.902 | 19.755 | 6.849 | 2.87x [2.63x, 3.21x] |

## Per-weight results

| Registered weight | First CINN (s) | First-call extra cost (s) | Eager warm (ms) | CINN warm (ms) | Speedup | Estimated break-even calls |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `chgnet_mptrj` | 134.162 | 134.149 | 25.842 | 13.153 | 1.96x | 10,573 |
| `comformer_jarvis_alex_pbe_2d_all_e_form` | 53.230 | 53.227 | 7.370 | 3.142 | 2.35x | 12,590 |
| `comformer_jarvis_dft_2d_e_form` | 53.734 | 53.731 | 7.358 | 3.041 | 2.42x | 12,449 |
| `comformer_jarvis_dft_3d_e_form` | 53.698 | 53.695 | 7.452 | 2.981 | 2.50x | 12,010 |
| `comformer_mp2018_train_60k_G` | 52.814 | 52.811 | 7.396 | 3.269 | 2.26x | 12,796 |
| `comformer_mp2018_train_60k_K` | 51.731 | 51.728 | 7.527 | 3.451 | 2.18x | 12,690 |
| `comformer_mp2018_train_60k_band_gap` | 53.101 | 53.098 | 7.409 | 3.019 | 2.45x | 12,095 |
| `comformer_mp2018_train_60k_e_form` | 54.260 | 54.257 | 7.608 | 3.042 | 2.50x | 11,882 |
| `comformer_mp2024_train_130k_e_form` | 54.089 | 54.085 | 7.472 | 3.069 | 2.43x | 12,286 |
| `diffcsp_mp20` | 20.976 | 20.960 | 32.991 | 15.466 | 2.13x | 1,197 |
| `diffcsp_mp20_ab_initio` | 26.300 | 26.283 | 35.975 | 17.141 | 2.10x | 1,396 |
| `diffnmr_msdnmr_nless15` | 163.490 | 154.211 | 19,127.407 | 9,279.350 | 2.06x | 16 |
| `dimenetpp_mp2018_train_60k_G` | 77.094 | 77.090 | 19.291 | 4.335 | 4.45x | 5,155 |
| `dimenetpp_mp2018_train_60k_K` | 78.203 | 78.199 | 19.627 | 4.354 | 4.51x | 5,120 |
| `dimenetpp_mp2018_train_60k_band_gap` | 77.705 | 77.700 | 22.171 | 4.366 | 5.08x | 4,364 |
| `dimenetpp_mp2018_train_60k_e_form` | 77.573 | 77.569 | 22.441 | 4.360 | 5.15x | 4,291 |
| `dimenetpp_qm9_Cv` | 75.602 | 75.598 | 19.022 | 4.273 | 4.45x | 5,126 |
| `dimenetpp_qm9_G` | 75.144 | 75.139 | 19.026 | 4.798 | 3.97x | 5,282 |
| `dimenetpp_qm9_H` | 76.038 | 76.034 | 18.656 | 4.467 | 4.18x | 5,359 |
| `dimenetpp_qm9_U` | 73.927 | 73.922 | 18.717 | 4.896 | 3.82x | 5,349 |
| `dimenetpp_qm9_U0` | 76.348 | 76.344 | 19.629 | 4.190 | 4.69x | 4,945 |
| `dimenetpp_qm9_alpha` | 74.610 | 74.605 | 18.817 | 4.963 | 3.79x | 5,386 |
| `dimenetpp_qm9_gap` | 75.241 | 75.236 | 18.916 | 4.384 | 4.31x | 5,178 |
| `dimenetpp_qm9_homo` | 75.419 | 75.415 | 19.022 | 4.211 | 4.52x | 5,092 |
| `dimenetpp_qm9_lumo` | 75.258 | 75.254 | 19.381 | 4.214 | 4.60x | 4,962 |
| `dimenetpp_qm9_mu` | 75.582 | 75.578 | 19.937 | 4.184 | 4.77x | 4,798 |
| `dimenetpp_qm9_r2` | 75.071 | 75.067 | 19.318 | 4.152 | 4.65x | 4,950 |
| `dimenetpp_qm9_zpve` | 75.859 | 75.854 | 19.204 | 4.898 | 3.92x | 5,303 |
| `mattergen_alex_mp20` | 338.525 | 338.384 | 165.197 | 141.504 | 1.17x | 14,282 |
| `mattergen_alex_mp20_chemical_system` | 298.418 | 298.078 | 326.272 | 339.898 | 0.96x | never |
| `mattergen_alex_mp20_chemical_system_energy_above_hull` | 286.127 | 285.789 | 336.399 | 337.867 | 1.00x | never |
| `mattergen_alex_mp20_dft_band_gap` | 369.134 | 368.844 | 352.789 | 289.786 | 1.22x | 5,855 |
| `mattergen_alex_mp20_dft_mag_density` | 372.261 | 371.986 | 342.179 | 275.031 | 1.24x | 5,540 |
| `mattergen_alex_mp20_dft_mag_density_hhi_score` | 369.039 | 368.815 | 328.087 | 223.820 | 1.47x | 3,538 |
| `mattergen_alex_mp20_ml_bulk_modulus` | 361.645 | 361.358 | 342.073 | 286.654 | 1.19x | 6,521 |
| `mattergen_alex_mp20_space_group` | 272.753 | 272.534 | 343.790 | 219.119 | 1.57x | 2,187 |
| `mattergen_ml2ddb` | 341.259 | 341.141 | 161.957 | 118.196 | 1.37x | 7,796 |
| `mattergen_ml2ddb_chemical_system` | 276.401 | 276.087 | 312.834 | 313.953 | 1.00x | never |
| `mattergen_ml2ddb_space_group` | 363.190 | 362.888 | 350.261 | 301.931 | 1.16x | 7,509 |
| `mattergen_mp20` | 348.575 | 348.442 | 180.682 | 132.346 | 1.37x | 7,209 |
| `mattergen_mp20_chemical_system` | 297.668 | 297.346 | 314.303 | 321.837 | 0.98x | never |
| `mattergen_mp20_csp` | 341.529 | 341.375 | 177.939 | 153.973 | 1.16x | 14,245 |
| `mattergen_mp20_dft_band_gap` | 366.559 | 366.293 | 318.311 | 265.692 | 1.20x | 6,962 |
| `mattergen_mp20_dft_bulk_modulus` | 367.520 | 367.283 | 327.714 | 236.860 | 1.38x | 4,043 |
| `mattergen_mp20_dft_mag_density` | 362.261 | 361.987 | 315.356 | 274.224 | 1.15x | 8,801 |
| `mattersim_1M` | 149.221 | 149.212 | 31.375 | 8.856 | 3.54x | 6,626 |
| `mattersim_5M` | 165.868 | 165.857 | 38.126 | 10.834 | 3.52x | 6,078 |
| `megnet_jarvis_alex_pbe_2d_all_e_form` | 69.601 | 69.594 | 11.737 | 6.695 | 1.75x | 13,804 |
| `megnet_jarvis_dft_2d_e_form` | 69.996 | 69.989 | 10.983 | 6.650 | 1.65x | 16,153 |
| `megnet_jarvis_dft_3d_e_form` | 70.319 | 70.313 | 11.195 | 6.744 | 1.66x | 15,795 |
| `megnet_mp2018_train_60k_G` | 70.309 | 70.301 | 12.908 | 7.814 | 1.65x | 13,801 |
| `megnet_mp2018_train_60k_K` | 69.740 | 69.732 | 12.737 | 7.873 | 1.62x | 14,335 |
| `megnet_mp2018_train_60k_band_gap` | 70.273 | 70.267 | 11.495 | 6.641 | 1.73x | 14,478 |
| `megnet_mp2018_train_60k_e_form` | 70.430 | 70.423 | 11.615 | 6.561 | 1.77x | 13,935 |
| `megnet_mp2024_train_130k_e_form` | 70.454 | 70.447 | 11.200 | 6.739 | 1.66x | 15,793 |
| `sfin_bf_detect` | 25.612 | 25.596 | 38.292 | 15.674 | 2.44x | 1,132 |
| `sfin_bf_enhance` | 25.865 | 25.846 | 37.926 | 19.521 | 1.94x | 1,405 |
| `sfin_haadf_detect` | 25.373 | 25.358 | 38.475 | 15.666 | 2.46x | 1,112 |
| `sfin_haadf_enhance` | 25.519 | 25.504 | 38.333 | 15.347 | 2.50x | 1,110 |
| `spherenet_md17_aspirin` | 251.601 | 251.586 | 48.385 | 15.118 | 3.20x | 7,563 |
| `spherenet_md17_benzene_old` | 250.319 | 250.304 | 50.101 | 14.174 | 3.53x | 6,968 |
| `spherenet_md17_ethanol` | 247.947 | 247.933 | 48.941 | 13.213 | 3.70x | 6,940 |
| `spherenet_md17_malonaldehyde` | 253.856 | 253.838 | 43.493 | 18.151 | 2.40x | 10,017 |
| `spherenet_md17_naphthalene` | 251.782 | 251.768 | 44.249 | 14.449 | 3.06x | 8,449 |
| `spherenet_md17_salicylic` | 251.175 | 251.163 | 43.972 | 12.514 | 3.51x | 7,985 |
| `spherenet_md17_toluene` | 253.191 | 253.175 | 42.458 | 15.853 | 2.68x | 9,517 |
| `spherenet_md17_uracil` | 246.768 | 246.754 | 41.925 | 14.238 | 2.94x | 8,913 |
| `spherenet_qm9_Cv` | 121.391 | 121.384 | 20.054 | 6.770 | 2.96x | 9,138 |
| `spherenet_qm9_G` | 122.783 | 122.776 | 19.728 | 6.923 | 2.85x | 9,588 |
| `spherenet_qm9_H` | 123.147 | 123.140 | 19.838 | 6.816 | 2.91x | 9,457 |
| `spherenet_qm9_U` | 123.489 | 123.482 | 19.766 | 6.961 | 2.84x | 9,643 |
| `spherenet_qm9_U0` | 122.973 | 122.965 | 20.265 | 7.718 | 2.63x | 9,801 |
| `spherenet_qm9_alpha` | 122.832 | 122.825 | 20.154 | 6.857 | 2.94x | 9,238 |
| `spherenet_qm9_gap` | 123.911 | 123.904 | 21.881 | 6.808 | 3.21x | 8,221 |
| `spherenet_qm9_homo` | 122.285 | 122.279 | 19.456 | 6.872 | 2.83x | 9,718 |
| `spherenet_qm9_lumo` | 123.271 | 123.264 | 19.744 | 6.840 | 2.89x | 9,553 |
| `spherenet_qm9_mu` | 121.930 | 121.924 | 19.597 | 6.664 | 2.94x | 9,428 |
| `spherenet_qm9_r2` | 66.340 | 66.334 | 17.649 | 6.519 | 2.71x | 5,960 |
| `spherenet_qm9_zpve` | 125.330 | 125.323 | 19.496 | 6.921 | 2.82x | 9,967 |

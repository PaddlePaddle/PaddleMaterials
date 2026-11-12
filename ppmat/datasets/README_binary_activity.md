# binaryGamma dataset

The [source dataset](https://git.rwth-aachen.de/avt-svt/public/GDI-NN) contains binary solvent mixtures and logarithmic activity coefficients. Missing source CSVs and solvent metadata are downloaded by the shared dataset utilities with MD5 verification.

Create the default training and validation files from the repository root:

```bash
python -m ppmat.datasets.split_binary_activity_data --mode=comp_inter --fold=0 --seed=2021 --output_dir=data/binary_activity/fold_0
```

The comp-inter protocol stratifies five folds by `tpsa_binary_avg`. Each fold contains 224,000 training rows and 56,000 validation rows. `split.json` records the source SHA-256, seed and exact row indices. Existing outputs are never overwritten. For a different fold, select another output directory and update both dataset paths in the model configuration. `--mode=system_extra` instead holds out complete solvent pairs.

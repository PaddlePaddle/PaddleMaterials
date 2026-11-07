# GE-GNN

[Thermodynamics-consistent graph neural networks](https://doi.org/10.1039/D4SC04554H)

## Abstract

GE-GNN predicts the dimensionless excess Gibbs free energy of binary solvent mixtures. Differentiation with respect to the mole fraction gives logarithmic activity coefficients that satisfy the Gibbs-Duhem relation:

```math
\ln \gamma_1 = G^E + (1 - x_1) \frac{dG^E}{dx_1}, \qquad
\ln \gamma_2 = G^E - x_1 \frac{dG^E}{dx_1}.
```

## Model

![GE-GNN architecture](gegnn.svg)

The two solvents share a molecular GCN encoder and exchange hydrogen-bond information through an interaction MPNN. A shared composition-dependent MLP and mean pooling produce the excess Gibbs energy; automatic differentiation produces the two log activity coefficients.

## Results

| Model Name | Dataset | Target | MAE (Val) | Config | Checkpoint |
| --- | --- | --- | --- | --- | --- |
| gegnn_binary_activity | binaryGamma | $\ln \gamma_1$, $\ln \gamma_2$ | Not evaluated | [gegnn_binary_activity.yaml](gegnn_binary_activity.yaml) | [checkpoint](https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/property_prediction/ge-gnn/gegnn_binary_activity.zip) |

The linked checkpoint was trained on the full dataset before the batch correction; it is provided for inference compatibility, not held-out evaluation.

## Training

The configuration uses the shared loader's seeded 80/10/10 random row split (`Trainer.seed=2021`). This is an example split, not the paper's cross-validation protocol. Use the same seed and unchanged input CSV when evaluating checkpoints trained with this configuration.

```bash
python property_prediction/train.py -c property_prediction/configs/gegnn/gegnn_binary_activity.yaml
```

## Validation

Use the checkpoint directory from the training run (including its timestamp and seed). GE-GNN requires `Trainer.eval_with_no_grad=False` for composition derivatives.

```bash
python property_prediction/train.py \
    -c property_prediction/configs/gegnn/gegnn_binary_activity.yaml \
    Global.do_train=False \
    Global.do_eval=True \
    Global.do_test=False \
    Trainer.pretrained_model_path=output/YOUR_TRAINING_RUN/checkpoints
```

## Prediction

The input CSV contains `smiles1`, `smiles2`, and `x1` (the first solvent's mole fraction). Predictions are saved as `ln_gamma1` and `ln_gamma2` in `results/result.csv`, alongside the input columns.

Use the current repository configuration with the published checkpoint ZIP. The shared loader automatically downloads the archive and selects `checkpoints/best.pdparams`; the archive's older YAML lacks the current vocabulary and graph settings.

```bash
python property_prediction/predict.py --config_path=property_prediction/configs/gegnn/gegnn_binary_activity.yaml --checkpoint_path=https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/property_prediction/ge-gnn/gegnn_binary_activity.zip --input_format=mixture_csv --input_path=property_prediction/example_data/binary_mixtures.csv
```

For a newly trained checkpoint:

```bash
python property_prediction/predict.py --config_path=property_prediction/configs/gegnn/gegnn_binary_activity.yaml --checkpoint_path=output/YOUR_TRAINING_RUN/checkpoints/best.pdparams --input_format=mixture_csv --input_path=property_prediction/example_data/binary_mixtures.csv
```

## Citation

```bibtex
@article{rittig2024thermodynamics,
  title={Thermodynamics-consistent graph neural networks},
  author={Rittig, Jan G. and Mitsos, Alexander},
  journal={Chemical Science},
  volume={15},
  pages={18504--18512},
  year={2024},
  doi={10.1039/D4SC04554H}
}
```

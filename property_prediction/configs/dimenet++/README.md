# DimeNet++

[Fast and Uncertainty-Aware Directional Message Passing for Non-Equilibrium Molecules](https://arxiv.org/abs/2011.14115)

## Abstract

Many important tasks in chemistry revolve around molecules during reactions. This requires predictions far from the equilibrium, while most recent work in machine learning for molecules has been focused on equilibrium or near-equilibrium states. In this paper we aim to extend this scope in three ways. First, we propose the DimeNet++ model, which is 8x faster and 10% more accurate than the original DimeNet on the QM9 benchmark of equilibrium molecules. Second, we validate DimeNet++ on highly reactive molecules by developing the challenging COLL dataset, which contains distorted configurations of small molecules during collisions. Finally, we investigate ensembling and mean-variance estimation for uncertainty quantification with the goal of accelerating the exploration of the vast space of non-equilibrium structures. Our DimeNet++ implementation as well as the COLL dataset are available online.


![DimeNet++](../../docs/DimeNet++.png)

## Datasets:

The primary datasets employed in the evaluation of ComFormer include the Materials Project (MP). The dataset provides the ground-truth labels derived from Density Functional Theory (DFT) calculations, which serve as the target for the supervised learning process. The preprocessing and partitioning of these datasets are critical for ensuring the generalizability of the model.

The MP2018.6.1 dataset represents a foundational benchmark for the field, encompassing a curated set of inorganic crystals with calculated thermodynamic and electronic properties.

- MP2018.6.1:

    The original dataset can download from [here](https://figshare.com/ndownloader/files/15087992). Following the methodology outlined in the Comformer paper, we randomly partitioned the dataset into subsets, with the specific sample sizes for each subset detailed in the table below.

    |                                   Dataset                                    | Train |  Val  | Test  | Properties |
    | :--------------------------------------------------------------------------: | :---: | :---: | :---: | :---------: |
    | [mp2018_train_60k](https://paddle-org.bj.bcebos.com/paddlematerial/datasets/mp2018/mp2018_train_60k.zip) | 60000 | 5000  | 4239  | Formation Energy, Band Gap, Bulk Modulus(K), Shear Modulus(G) |

## Model

DimeNet++ is a directional message passing neural network designed for accurate and efficient prediction of molecular energies and forces, particularly for non-equilibrium molecular configurations. It builds upon the original DimeNet architecture by preserving its physically motivated directional message passing mechanism while substantially improving computational efficiency and predictive accuracy through architectural refinements.

### Graph representation and embeddings

A molecular system is represented as a graph ( G = (V, E) ), where nodes ( i \in V ) correspond to atoms and directed edges ( j \to i \in E ) correspond to interatomic interactions within a cutoff radius. Unlike conventional atom-centered GNNs, DimeNet++ associates learnable embeddings with **directed edges**, enabling explicit encoding of geometric directionality.

Each directed edge ( j \to i ) is characterized by the interatomic distance ( d_{ji} ), which is expanded using a radial basis function (RBF):

```math
\mathbf{e}^{\text{RBF}}_{ji} = \text{RBF}(d_{ji})
```

To capture angular information, DimeNet++ further considers atom triplets ( k \to j \to i ), where the bond angle ( \alpha_{kji} ) together with distance ( d_{kj} ) is expanded using a spherical basis function (SBF):

```math
\mathbf{a}^{\text{SBF}}_{kji} = \text{SBF}(d_{kj}, \alpha_{kji})
```

These basis representations enable a joint encoding of distances and angles, which is essential for modeling anisotropic interactions and directional bonding effects.

### Directional message passing

The core of DimeNet++ is its **directional message passing** scheme, where messages are propagated along directed edges rather than between atoms. Each directed edge ( j \to i ) is associated with a message embedding ( \mathbf{m}_{ji}^{(l)} ) at layer ( l ).

The update of a message embedding consists of two steps: interaction aggregation and message update. First, messages incoming to atom ( j ) from its neighbors ( k \neq i ) are aggregated through an interaction function:

```math
\mathbf{z}_{ji}^{(l)} =
\sum_{k \in \mathcal{N}(j) \setminus \{i\}}
f_{\text{int}}\!\left(
\mathbf{m}_{kj}^{(l)},
\mathbf{e}^{\text{RBF}}_{ji},
\mathbf{a}^{\text{SBF}}_{kji}
\right)
```

The aggregated interaction is then combined with the current message embedding to produce the updated message:

```math
\mathbf{m}_{ji}^{(l+1)} =
f_{\text{update}}\!\left(
\mathbf{m}_{ji}^{(l)}, \mathbf{z}_{ji}^{(l)}
\right)
```

Here, ( f_{\text{int}} ) and ( f_{\text{update}} ) are learnable neural network modules.

### Efficient interaction modeling in DimeNet++

In the original DimeNet, the interaction function ( f_{\text{int}} ) relied on a bilinear transformation between message embeddings and basis representations, which incurred significant computational cost due to the large number of edge triplets. DimeNet++ replaces this bilinear layer with a more efficient **Hadamard (element-wise) product**, while maintaining expressiveness by introducing multilayer perceptrons (MLPs) applied to the basis functions:

```math
f_{\text{int}}(\mathbf{m}, \mathbf{e}, \mathbf{a})
=
\left( \text{MLP}_{\text{RBF}}(\mathbf{e})
\odot
\text{MLP}_{\text{SBF}}(\mathbf{a}) \right)
\odot
\mathbf{m}
```

This modification significantly reduces computational complexity while preserving or improving predictive accuracy.

### Embedding hierarchy and residual connections

To further improve efficiency, DimeNet++ introduces an **embedding hierarchy** through down-projection and up-projection layers. Message embeddings are projected to a lower-dimensional space during the costly interaction steps and projected back afterward:

```math
\mathbf{m}^{\downarrow} = \mathbf{W}_{\downarrow} \mathbf{m},
\quad
\mathbf{m}^{\uparrow} = \mathbf{W}_{\uparrow} \mathbf{m}^{\downarrow}
```

Residual connections are applied throughout the network to stabilize training and facilitate deeper architectures:

```math
\mathbf{m}_{ji}^{(l+1)} =
\mathbf{m}_{ji}^{(l)} + \Delta \mathbf{m}_{ji}^{(l)}
```

Empirically, DimeNet++ achieves comparable or better performance using fewer interaction layers than the original DimeNet.

### Atomic representations and output

Although message passing is performed on directed edges, atomic representations are obtained by aggregating incoming messages for each atom:

```math
\mathbf{t}_i^{(l)} =
\sum_{j \in \mathcal{N}(i)}
\mathbf{W}_{\text{out}} \mathbf{m}_{ji}^{(l)}
```

The final atomic embeddings are passed through an output network to predict atomic energy contributions. The total molecular energy is computed as a sum over atoms:

```math
E = \sum_i E_i
```

Atomic forces are obtained by differentiating the predicted energy with respect to atomic positions, ensuring full energy–force consistency:

```math
\mathbf{F}_i = - \frac{\partial E}{\partial \mathbf{x}_i}
```

### Model stacking and training characteristics

DimeNet++ stacks multiple directional message passing layers to enable information propagation from local to longer-range interactions. The model is trained using the Adam optimizer, and mixed precision is avoided due to numerical stability issues arising from the high accuracy requirements of energy and force predictions.

Through its architectural improvements, DimeNet++ achieves up to an **8× speedup** over DimeNet while improving prediction accuracy, making it well suited for large-scale simulations of reactive and non-equilibrium molecular systems.


## Results

<table>
    <head>
        <tr>
            <th  nowrap="nowrap">Model Name</th>
            <th  nowrap="nowrap">Dataset</th>
            <th  nowrap="nowrap">Property</th>
            <th  nowrap="nowrap">MAE(Val / Test dataset)</th>
            <th  nowrap="nowrap">GPUs</th>
            <th  nowrap="nowrap">Training time</th>
            <th  nowrap="nowrap">Config</th>
            <th  nowrap="nowrap">Checkpoint | Log</th>
        </tr>
    </head>
    <body>
        <tr>
            <td  nowrap="nowrap">dimenetpp_mp2018_train_60k_e_form</td>
            <td  nowrap="nowrap">mp2018_train_60k</td>
            <td  nowrap="nowrap">Form. Energy(eV/atom)</td>
            <td  nowrap="nowrap">0.023471 / 0.023361</td>
            <td  nowrap="nowrap">4</td>
            <td  nowrap="nowrap">20 hours 0 min</td>
            <td  nowrap="nowrap"><a href="dimenet++_mp2018_train_60k_e_form.yaml">dimenet++_mp2018_train_60k_e_form</a></td>
            <td  nowrap="nowrap"><a href="https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/property_prediction/dimenet++/dimenetpp_mp2018_train_60k_e_form.zip">checkpoint | log</a></td>
        </tr>  
        <tr>
            <td  nowrap="nowrap">dimenetpp_mp2018_train_60k_band_gap</td>
            <td  nowrap="nowrap">mp2018_train_60k</td>
            <td  nowrap="nowrap">Band Gap(eV)</td>
            <td  nowrap="nowrap">0.241686 / 0.245299</td>
            <td  nowrap="nowrap">1 (batch 128 = 4 × 32)</td>
            <td  nowrap="nowrap">29 hours 51 min</td>
            <td  nowrap="nowrap"><a href="dimenet++_mp2018_train_60k_band_gap.yaml">dimenet++_mp2018_train_60k_band_gap</a></td>
            <td  nowrap="nowrap"><a href="https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/property_prediction/dimenet++/dimenetpp_mp2018_train_60k_band_gap.zip">checkpoint | log</a></td>
        </tr>  
        <tr>
            <td  nowrap="nowrap">dimenetpp_mp2018_train_60k_K</td>
            <td  nowrap="nowrap">mp2018_train_60k</td>
            <td  nowrap="nowrap">Bulk modulus(GPa)</td>
            <td  nowrap="nowrap">6.259347 / 5.580545</td>
            <td  nowrap="nowrap">4</td>
            <td  nowrap="nowrap">1 hour 56 min</td>
            <td  nowrap="nowrap"><a href="dimenet++_mp2018_train_60k_K.yaml">dimenet++_mp2018_train_60k_k</a></td>
            <td  nowrap="nowrap"><a href="https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/property_prediction/dimenet++/dimenetpp_mp2018_train_60k_K.zip">checkpoint | log</a></td>
        </tr>
        <tr>
            <td  nowrap="nowrap">dimenetpp_mp2018_train_60k_G</td>
            <td  nowrap="nowrap">mp2018_train_60k</td>
            <td  nowrap="nowrap">Shear modulus(GPa)</td>
            <td  nowrap="nowrap">7.223798 / 5.725558</td>
            <td  nowrap="nowrap">4</td>
            <td  nowrap="nowrap">1 hour 53 min</td>
            <td  nowrap="nowrap"><a href="dimenet++_mp2018_train_60k_G.yaml">dimenet++_mp2018_train_60k_G</a></td>
            <td  nowrap="nowrap"><a href="https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/property_prediction/dimenet++/dimenetpp_mp2018_train_60k_G.zip">checkpoint | log</a></td>
        </tr>
    </body>
</table>

### Training
```bash
# formation energy per atom
# multi-gpu training, we use 4 gpus here
python -m paddle.distributed.launch --gpus="0,1,2,3" property_prediction/train.py -c property_prediction/configs/dimenet++/dimenet++_mp2018_train_60k_e_form.yaml
# single-gpu training
python property_prediction/train.py -c property_prediction/configs/dimenet++/dimenet++_mp2018_train_60k_e_form.yaml

# band gap
# multi-gpu training, we use 4 gpus here
python -m paddle.distributed.launch --gpus="0,1,2,3" property_prediction/train.py -c property_prediction/configs/dimenet++/dimenet++_mp2018_train_60k_band_gap.yaml
# single-gpu training
python property_prediction/train.py -c property_prediction/configs/dimenet++/dimenet++_mp2018_train_60k_band_gap.yaml

# bulk modulus
# multi-gpu training, we use 4 gpus here
python -m paddle.distributed.launch --gpus="0,1,2,3" property_prediction/train.py -c property_prediction/configs/dimenet++/dimenet++_mp2018_train_60k_K.yaml
# single-gpu training
python property_prediction/train.py -c property_prediction/configs/dimenet++/dimenet++_mp2018_train_60k_K.yaml

# shear modulus
# multi-gpu training, we use 4 gpus here
python -m paddle.distributed.launch --gpus="0,1,2,3" property_prediction/train.py -c property_prediction/configs/dimenet++/dimenet++_mp2018_train_60k_G.yaml
# single-gpu training
python property_prediction/train.py -c property_prediction/configs/dimenet++/dimenet++_mp2018_train_60k_G.yaml
```

### Validation
```bash
# Adjust program behavior on-the-fly using command-line parameters – this provides a convenient way to customize settings without modifying the configuration file directly.
# such as: --Global.do_eval=True

# formation energy per atom
python property_prediction/train.py -c property_prediction/configs/dimenet++/dimenet++_mp2018_train_60k_e_form.yaml Global.do_eval=True Global.do_train=False Global.do_test=False Trainer.pretrained_model_path='your model path(*.pdparams)'

# band gap
python property_prediction/train.py -c property_prediction/configs/dimenet++/dimenet++_mp2018_train_60k_band_gap.yaml Global.do_eval=True Global.do_train=False Global.do_test=False Trainer.pretrained_model_path='your model path(*.pdparams)'

# bulk modulus
python property_prediction/train.py -c property_prediction/configs/dimenet++/dimenet++_mp2018_train_60k_K.yaml Global.do_eval=True Global.do_train=False Global.do_test=False Trainer.pretrained_model_path='your model path(*.pdparams)'

# shear modulus
python property_prediction/train.py -c property_prediction/configs/dimenet++/dimenet++_mp2018_train_60k_G.yaml Global.do_eval=True Global.do_train=False Global.do_test=False Trainer.pretrained_model_path='your model path(*.pdparams)'
```

### Testing
```bash
# This command is used to evaluate the model's performance on the test dataset.

# formation energy per atom
python property_prediction/train.py -c property_prediction/configs/dimenet++/dimenet++_mp2018_train_60k_e_form.yaml Global.do_test=True Global.do_train=False Global.do_eval=False Trainer.pretrained_model_path='your model path(*.pdparams)'

# band gap
python property_prediction/train.py -c property_prediction/configs/dimenet++/dimenet++_mp2018_train_60k_band_gap.yaml Global.do_test=True Global.do_train=False Global.do_eval=False Trainer.pretrained_model_path='your model path(*.pdparams)'

# bulk modulus
python property_prediction/train.py -c property_prediction/configs/dimenet++/dimenet++_mp2018_train_60k_K.yaml Global.do_test=True Global.do_train=False Global.do_eval=False Trainer.pretrained_model_path='your model path(*.pdparams)'

# shear modulus
python property_prediction/train.py -c property_prediction/configs/dimenet++/dimenet++_mp2018_train_60k_G.yaml Global.do_test=True Global.do_train=False Global.do_eval=False Trainer.pretrained_model_path='your model path(*.pdparams)'
```

### Prediction

You can replace the `--model_name` parameter at  `Mode 1` with other model names from the `results` table.

```bash
# This command is used to predict the properties of new crystal structures using a trained model.
# Note: The model_name and weights_name parameters are used to specify the pre-trained model and its corresponding weights. The cif_file_path parameter is used to specify the path to the CIF files for which properties need to be predicted.
# The prediction results will be saved in a CSV file specified by the save_path parameter. Default save_path is 'result.csv'.

# formation energy per atom

# Mode 1: Leverage a pre-trained machine learning model for crystal formation energy prediction. The implementation includes automated model download functionality, eliminating the need for manual configuration.
python property_prediction/predict.py --model_name='dimenetpp_mp2018_train_60k_e_form' --input_format=cif --input_path='./property_prediction/example_data/cifs/'

# Mode2: Use a custom configuration file and checkpoint for crystal formation energy prediction. This approach allows for more flexibility and customization.
python property_prediction/predict.py --config_path='property_prediction/configs/dimenet++/dimenetpp_mp2018_train_60k_e_form.yaml' --checkpoint_path='you_checkpoint_path.pdparams' --input_format=cif --input_path='./property_prediction/example_data/cifs/'


# band gap

# Mode 1: Leverage a pre-trained machine learning model for crystal band gap prediction. The implementation includes automated model download functionality, eliminating the need for manual configuration.
python property_prediction/predict.py --model_name='dimenetpp_mp2018_train_60k_band_gap' --input_format=cif --input_path='./property_prediction/example_data/cifs/'

# Mode2: Use a custom configuration file and checkpoint for crystal band gap prediction. This approach allows for more flexibility and customization.
python property_prediction/predict.py --config_path='property_prediction/configs/dimenetpp/dimenetpp_mp2018_train_60k_band_gap.yaml' --checkpoint_path='you_checkpoint_path.pdparams' --input_format=cif --input_path='./property_prediction/example_data/cifs/'

# bulk modulus

# Mode 1: Leverage a pre-trained machine learning model for crystal bulk modulus prediction. The implementation includes automated model download functionality, eliminating the need for manual configuration.
python property_prediction/predict.py --model_name='dimenetpp_mp2018_train_60k_K' --input_format=cif --input_path='./property_prediction/example_data/cifs/'

# Mode2: Use a custom configuration file and checkpoint for crystal bulk modulus prediction. This approach allows for more flexibility and customization.
python property_prediction/predict.py --config_path='property_prediction/configs/dimenetpp/dimenetpp_mp2018_train_60k_K.yaml' --checkpoint_path='you_checkpoint_path.pdparams' --input_format=cif --input_path='./property_prediction/example_data/cifs/'


# shear modulus

# Mode 1: Leverage a pre-trained machine learning model for crystal shear modulus prediction. The implementation includes automated model download functionality, eliminating the need for manual configuration.
python property_prediction/predict.py --model_name='dimenetpp_mp2018_train_60k_G' --input_format=cif --input_path='./property_prediction/example_data/cifs/'

# Mode2: Use a custom configuration file and checkpoint for crystal shear modulus prediction. This approach allows for more flexibility and customization.
python property_prediction/predict.py --config_path='property_prediction/configs/dimenetpp/dimenet++_mp2018_train_60k_G.yaml' --checkpoint_path='you_checkpoint_path.pdparams' --input_format=cif --input_path='./property_prediction/example_data/cifs/'
```


## Citation
```
@article{gasteiger2020fast,
  title={Fast and Uncertainty-Aware Directional Message Passing for Non-Equilibrium Molecules},
  author={Gasteiger, Johannes and Giri, Shankari and Margraf, Johannes T. and Günnemann, Stephan},
  journal={arXiv preprint arXiv:2011.14115},
  year={2020}
}
```

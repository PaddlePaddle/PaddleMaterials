# CHGNet

[CHGNet: Pretrained universal neural network potential for charge-informed atomistic modeling](https://www.nature.com/articles/s42256-023-00716-3)

## Abstract

The simulation of large-scale systems with complex electron interactions remains one of the greatest challenges for the atomistic modeling of materials. Although classical force fields often fail to describe the coupling between electronic states and ionic rearrangements, the more accurate ab-initio molecular dynamics suffers from computational complexity that prevents long-time and large-scale simulations, which are essential to study many technologically relevant phenomena, such as reactions, ion migrations, phase transformations, and degradation. In this work, we present the Crystal Hamiltonian Graph neural Network (CHGNet) as a novel machine-learning interatomic potential (MLIP), using a graph-neural-network-based force field to model a universal potential energy surface. CHGNet is pretrained on the energies, forces, stresses, and magnetic moments from the Materials Project Trajectory Dataset, which consists of over 10 years of density functional theory static and relaxation trajectories of ∼ 1.5 million inorganic structures. The explicit inclusion of magnetic moments enables CHGNet to learn and accurately represent the orbital occupancy of electrons, enhancing its capability to describe both atomic and electronic degrees of freedom. We demonstrate several applications of CHGNet in solid-state materials, including charge-informed molecular dynamics in LixMnO2, the finite temperature phase diagram for LixFePO4 and Li diffusion in garnet conductors. We critically analyze the significance of including charge information for capturing appropriate chemistry, and we provide new insights into ionic systems with additional electronic degrees of freedom that can not be observed by previous MLIPs.

![CHGNet Overview](../../docs/chgnet.png)

## Datasets:

CHGNet is trained and evaluated on large-scale atomistic datasets covering both crystalline bulk materials and surface reaction systems. These datasets provide high-fidelity quantum-mechanical labels, including energies, forces, stresses, and electronic properties, enabling the construction of a charge-aware universal interatomic potential.

The MPtrj dataset is used for CHGNet pretraining and bulk material modeling. The OC20 S2EF dataset is used to evaluate model generalization to surface reaction systems. All dataset splits are fixed and reproducible. Reported MAE values in the Results section follow the evaluation protocol of the original CHGNet paper.

- MPtrj_2022.9_full:

    The Materials Project Trajectory Dataset (MPtrj_2022.9) is the primary pretraining dataset for CHGNet. The original dataset can download from [here](https://figshare.com/articles/dataset/Materials_Project_Trjectory_MPtrj_Dataset/23713842).

    This dataset contains long-term accumulated density functional theory (DFT) static and relaxation trajectories from the Materials Project (2022.9 release), covering a wide range of inorganic crystalline compounds.

    - 145,923 unique compounds
    - 1,580,395 crystal structures

    Corresponding labels:
    - 1,580,395 total energies
    - 49,295,660 atomic forces
    - 14,223,555 stresses
    - 7,944,833 magnetic moments

    All calculations are performed at the GGA / GGA+U level of theory. A strict filtering and deduplication protocol is applied to remove incompatible calculations and redundant structures, ensuring data consistency and quality.

    Following the CHGNet paper, the dataset is randomly partitioned based on mp-id, such that structures from the same compound do not appear across different splits.

    |                                   Dataset                                    | Train |  Val  | Test  |
    | :--------------------------------------------------------------------------: | :---: | :---: | :---: |
    | [MPtrj_2022.9_full](https://paddle-org.bj.bcebos.com/paddlematerial/datasets/mptrj/MPtrj_2022.9_full.zip) | 116738 | 14592  | 14593  |

    This dataset enables CHGNet to learn a unified potential energy surface across diverse chemistries, crystal symmetries, and magnetic configurations.

- OC20 S2EF

    The Open Catalyst 2020 (OC20) Structure-to-Energy-and-Force (S2EF) dataset is a large-scale benchmark for evaluating interatomic potentials in surface chemistry and catalysis.

    OC20 S2EF focuses on predicting energies and atomic forces for adsorbate–surface systems, featuring:

    - Large structural diversity
    - Challenging out-of-equilibrium configurations
    - Strong relevance to catalytic reaction modeling

    We evaluate the CHGNet architecture on the OC20 S2EF dataset to assess its transferability beyond bulk crystalline systems. For more information and the download link, please visit [here](https://paddle-org.bj.bcebos.com/paddlematerials/datasets/OC20/s2ef_train_2M/0000.parquet).

    | Dataset        | Train     | Val      | Test     |
    | :------------ | :-------- | :------- | :------- |
    | oc20_s2ef_2M  | 2,000,000 | 100,000  | 200,000  |


## Models

Given atomic coordinates and lattice vectors, CHGNet constructs three coupled graphs within a cutoff radius:

* **Atom graph**: nodes represent atoms with element-dependent features
* **Bond graph**: edges encode pairwise interactions based on interatomic distances
* **Angle graph**: captures three-body interactions through bond angles

Interatomic distances are expanded using radial basis functions:
$$
e_{ij,n} =
\sqrt{\frac{2}{r_c}}
\frac{\sin\left(\frac{n\pi r_{ij}}{r_c}\right)}{r_{ij}}.
$$

Angular information is encoded using Fourier basis functions of bond angles.

### Energy and Forces

The total energy is obtained by summing atomic energy contributions:
$$
E_{\text{tot}} = \sum_i E_i.
$$

Atomic forces are computed as energy gradients with respect to atomic positions:
$$
\mathbf{F}_i = -\frac{\partial E_{\text{tot}}}{\partial \mathbf{r}_i}.
$$

Stresses are derived consistently from the energy–strain relation:
$$
\boldsymbol{\sigma} = \frac{1}{V} \frac{\partial E_{\text{tot}}}{\partial \boldsymbol{\varepsilon}}.
$$

CHGNet provides a unified, charge-aware interatomic potential capable of modeling complex crystalline materials, including systems with magnetism and charge transfer. It is suitable for structure relaxation, molecular dynamics, and materials property prediction, offering strong transferability across diverse inorganic systems.


## Results

<table>
    <head>
        <tr>
            <th  nowrap="nowrap">Model Name</th>
            <th  nowrap="nowrap">Dataset</th>
            <th  nowrap="nowrap">Energy MAE(meV/atom)</th>
            <th  nowrap="nowrap">Force MAE(meV/A)</th>
            <th  nowrap="nowrap">Stress MAE(GPa)</th>
            <th  nowrap="nowrap">Magmom MAE(μB)</th>
            <th  nowrap="nowrap">GPUs</th>
            <th  nowrap="nowrap">Training time</th>
            <th  nowrap="nowrap">Config</th>
            <th  nowrap="nowrap">Checkpoint | Log</th>
        </tr>
    </head>
    <body>
        <tr>
            <td  nowrap="nowrap">chgnet_mptrj</td>
            <td  nowrap="nowrap">MPtrj_2022.9_full</td>
            <td  nowrap="nowrap">30</td>
            <td  nowrap="nowrap">77</td>
            <td  nowrap="nowrap">4.348</td>
            <td  nowrap="nowrap">0.032</td>
            <td  nowrap="nowrap"> ~ </td>
            <td  nowrap="nowrap"> ~ </td>
            <td  nowrap="nowrap"><a href="chgnet_mptrj.yaml">chgnet_mptrj</a></td>
            <td  nowrap="nowrap"><a href="https://paddle-org.bj.bcebos.com/paddlematerial/checkpoints/interatomic_potentials/chgnet/chgnet_mptrj.zip">checkpoint | log</a></td>
        </tr>  
        <tr>
            <td  nowrap="nowrap">chgnet_oc20_s2ef_energy</td>
            <td  nowrap="nowrap">oc20_s2ef</td>
            <td  nowrap="nowrap"> - </td>
            <td  nowrap="nowrap"> - </td>
            <td  nowrap="nowrap"> - </td>
            <td  nowrap="nowrap"> - </td>
            <td  nowrap="nowrap"> ~ </td>
            <td  nowrap="nowrap"> ~ </td>
            <td  nowrap="nowrap"><a href="chgnet_oc20_s2ef_energy.yaml">chgnet_oc20_s2ef_energy</a></td>
            <td  nowrap="nowrap"><a href="0">checkpoint | log</a></td>
        </tr>  
        <tr>
            <td  nowrap="nowrap">chgnet_oc20_s2ef_forces</td>
            <td  nowrap="nowrap">oc20_s2ef</td>
            <td  nowrap="nowrap"> - </td>
            <td  nowrap="nowrap"> - </td>
            <td  nowrap="nowrap"> - </td>
            <td  nowrap="nowrap"> - </td>
            <td  nowrap="nowrap"> ~ </td>
            <td  nowrap="nowrap"> ~ </td>
            <td  nowrap="nowrap"><a href="chgnet_oc20_s2ef_forces.yaml">chgnet_oc20_s2ef_forces</a></td>
            <td  nowrap="nowrap"><a href="0">checkpoint | log</a></td>
        </tr>  
    </body>
</table>

**Note**: The model weights were directly adapted from the [CHGNet](https://github.com/CederGroupHub/chgnet) repository. Since the original paper did not disclose its randomly split test set, we repartitioned the test data according to the proportions described in the paper. However, due to differences in random seeds, the data partitioning could not be fully replicated, limiting the referential value of evaluation results obtained with our test set. To ensure result comparability, the MAE metrics listed in the table are directly cited from the original [paper's](https://www.nature.com/articles/s42256-023-00716-3) experimental results.

### Training

```bash
# multi-gpu training
python -m paddle.distributed.launch --gpus="0,1,2,3" interatomic_potentials/train.py -c interatomic_potentials/configs/chgnet/chgnet_mptrj.yaml
# single-gpu training
python interatomic_potentials/train.py -c interatomic_potentials/configs/chgnet/chgnet_mptrj.yaml
```

### Validation
```bash
# Adjust program behavior on-the-fly using command-line parameters – this provides a convenient way to customize settings without modifying the configuration file directly.
# such as: --Global.do_eval=True

python interatomic_potentials/train.py -c interatomic_potentials/configs/chgnet/chgnet_mptrj.yaml Global.do_eval=True Global.do_train=False Global.do_test=False Trainer.pretrained_model_path='your checkpoint path(*.pdparams)'

```


### Testing
```bash
# This command is used to evaluate the model's performance on the test dataset.

python interatomic_potentials/train.py -c interatomic_potentials/configs/chgnet/chgnet_mptrj.yaml Global.do_test=True Global.do_train=False Global.do_eval=False Trainer.pretrained_model_path='your checkpoint path(*.pdparams)'

```

### Prediction

```bash
# This command is used to predict the properties of new crystal structures using a trained model.
# Note: The model_name and weights_name parameters are used to specify the pre-trained model and its corresponding weights. The cif_file_path parameter is used to specify the path to the CIF files for which properties need to be predicted.
# The prediction results will be saved in a CSV file specified by the save_path parameter. Default save_path is 'result.csv'.


# Mode 1: Leverage a pre-trained machine learning model for crystal shear moduli prediction. The implementation includes automated model download functionality, eliminating the need for manual configuration.
python interatomic_potentials/predict.py --model_name='chgnet_mptrj' --input_format=cif --input_path='./interatomic_potentials/example_data/cifs/'

# Mode2: Use a custom configuration file and checkpoint for crystal shear moduli prediction. This approach allows for more flexibility and customization.
python interatomic_potentials/predict.py --config_path='interatomic_potentials/configs/chgnet/chgnet_mptrj.yaml' --checkpoint_path="your checkpoint path(*.pdparams)"
```


## CINN execution

Forces and stress are differentiated inside the compiled boundary, and SOT capture can split
`paddle.grad` from its input. When running this model with `Execution.backend=cinn`, set
`Execution.__init_params__.full_graph=true` so that the whole boundary is captured as one graph:

```yaml
Execution:
  backend: cinn
  __init_params__:
    full_graph: true
```

Without `full_graph`, CINN fails for force or stress predictions with an error about an input that
does not appear in the backward graph. Energy-only predictions can use the default SOT capture.

## Citation
```
@article{deng2023chgnet,
  title={CHGNet as a pretrained universal neural network potential for charge-informed atomistic modelling},
  author={Deng, Bowen and Zhong, Peichen and Jun, KyuJung and Riebesell, Janosh and Han, Kevin and Bartel, Christopher J and Ceder, Gerbrand},
  journal={Nature Machine Intelligence},
  volume={5},
  number={9},
  pages={1031--1041},
  year={2023},
  publisher={Nature Publishing Group UK London}
}
```

# MegNet

[Graph Networks as a Universal Machine Learning Framework for Molecules and Crystals](https://arxiv.org/abs/1812.05055)

## Abstract

Graph networks are a new machine learning (ML) paradigm that supports both relational reasoning and combinatorial generalization. Here, we develop universal MatErials Graph Network (MEGNet) models for accurate property prediction in both molecules and crystals. We demonstrate that the MEGNet models outperform prior ML models such as the SchNet in 11 out of 13 properties of the QM9 molecule data set. Similarly, we show that MEGNet models trained on ∼60,000 crystals in the Materials Project substantially outperform prior ML models in the prediction of the formation energies, band gaps and elastic modulus of crystals, achieving better than DFT accuracy over a much larger data set. We present two new strategies to address data limitations common in materials science and chemistry. First, we demonstrate a physically-intuitive approach to unify four separate molecular MEGNet models for the internal energy at 0 K and room temperature, enthalpy and Gibbs free energy into a single free energy MEGNet model by incorporating the temperature, pressure and entropy as global state inputs. Second, we show that the learned element embeddings in MEGNet models encode periodic chemical trends and can be transfer-learned from a property model trained on a larger data set (formation energies) to improve property models with smaller amounts of data (band gaps and elastic modulus).


![MegNet Overview](../../docs/megnet.png)

## Datasets:

The primary datasets employed in the evaluation of ComFormer include the Materials Project (MP), JARVIS-DFT, and the Alexandria Material Project. These datasets provide the ground-truth labels derived from Density Functional Theory (DFT) calculations, which serve as the target for the supervised learning process. The preprocessing and partitioning of these datasets are critical for ensuring the generalizability of the model.

The MP2018.6.1 dataset represents a foundational benchmark for the field, encompassing a curated set of inorganic crystals with calculated thermodynamic and electronic properties. The MP2024 subset significantly expands the scale of the training data, allowing for the observation of model saturation and the benefits of large-scale pre-training. JARVIS-DFT datasets are particularly valued for their high-precision calculations and the inclusion of diverse properties like the energy above the convex hull ($E_{hull}$), which is a critical indicator of material stability.

- MP2018.6.1:

    The original dataset can download from [here](https://figshare.com/ndownloader/files/15087992). Following the methodology outlined in the Comformer paper, we randomly partitioned the dataset into subsets, with the specific sample sizes for each subset detailed in the table below.

    |                                   Dataset                                    | Train |  Val  | Test  | Properties |
    | :--------------------------------------------------------------------------: | :---: | :---: | :---: | :---------: |
    | [mp2018_train_60k](https://paddle-org.bj.bcebos.com/paddlematerial/datasets/mp2018/mp2018_train_60k.zip) | 60000 | 5000  | 4239  | Formation Energy, Band Gap, Bulk Modulus(K), Shear Modulus(G) |

- MP2024

    |                                   Dataset                                    | Train |  Val  | Test  | Properties |
    | :--------------------------------------------------------------------------: | :---: | :---: | :---: | :---------: |
    | [mp2024_train_130k](https://paddle-org.bj.bcebos.com/paddlematerial/datasets/mp2024/mp2024_train_130k.zip) | 130000 | 10000  | 15361  | Formation Energy, Band Gap, Bulk Modulus(K), Shear Modulus(G) |

- Jarvis

    The original dataset can download from [here](https://github.com/usnistgov/jarvis).
    | Dataset | Count |  Properties |
    | :----: | :---: | :---------: |
    | dft_2d | 1109 | Formation Energy, Band Gap, Bulk Modulus(K), Shear Modulus(G), $E_{hull}$, et al. |
    | dft_3d | 75993| Formation Energy, Band Gap, Bulk Modulus(K), Shear Modulus(G), $E_{hull}$, et al. |
    | cfid_3d | 55723 | Formation Energy, Band Gap, Bulk Modulus(K), Shear Modulus(G), $E_{hull}$, et al. |
    | dft_3d_2021 | 55723 | Formation Energy, Band Gap, Bulk Modulus(K), Shear Modulus(G), $E_{hull}$, et al. |

- Alexandria Material Project

    | Dataset | Count | Properties |
    | :---: | :---: | :---------: |
    | pbe_2d | 100000 | Formation Energy, et al.  |

- Matbench

    The Matbench benchmark dataset for materials property prediction. The original dataset can be downloaded from [here](https://paddle-org.bj.bcebos.com/paddlematerial/datasets/matbench/matbench.zip/).

    | Dataset | Property | Count |
    | :---: | :---: | :---: |
    | mp_e_form | Formation Energy (eV/atom) | 132752 |
    | mp_gap | Band Gap (eV) | 106113 |
    | G |  Shear Modulus (GPa) | 10987 |
    | K |  Bulk Modulus (GPa) | 10987 |

- OMol25:

    The OMol25 dataset is widely used for benchmarking molecular modeling methods that predict quantum chemical properties (such as internal energy, HOMO-LUMO gap, and dipole moment) given molecular structures. We conducted experiments based on the CHGNet model on this dataset.
    For more information and the download link, please visit [here](https://paddle-org.bj.bcebos.com/paddlematerials/datasets/OMol25/train_4M.tar.gz).

    | Dataset | Count |
    | :---: | :---: |
    | OMol25 | 4000000 |

## Model
MEGNet represents material systems as graphs composed of three fundamental elements: nodes ((V)), edges ((E)), and a global state ((u)). Nodes represent atoms and encode attributes such as atomic number, hybridization state, chirality, ring size, and aromaticity. Edges represent chemical bonds or neighborhood relationships between atoms, incorporating bond types, interatomic distances, and topological features. The global state vector stores system-level information, such as molecular weight, temperature, pressure, and entropy, enabling unified modeling of state-dependent properties such as free energy. For crystalline systems, edges are defined based on a cutoff radius to determine atomic neighborhoods, and spatial distances are expanded using Gaussian basis functions, whereas molecular systems can include richer node and edge attributes.

MEGNet achieves information fusion among atoms, edges, and the global state through three successive update steps.

### Edge update

The updated attribute of each edge is determined by its own attributes, the attributes of the atoms it connects, and the global state:

```math
e_k' = \phi_e \left( v_{s_k} \oplus v_{r_k} \oplus e_k \oplus u \right)
```

where (\phi_e) denotes the edge update function, (\oplus) represents vector concatenation, and (s_k) and (r_k) are the indices of the two atoms connected by edge (k).

### Node update

The updated attribute of each node is determined by its own attributes, the updated attributes of all connected edges, and the global state. First, a local aggregation over incident edges is performed:

```math
\bar{v}_i^{\,e} = \frac{1}{N_i^e} \sum_{k=1}^{N_i^e} \left\{ e_k' \right\}_{r_k = i}
```

The aggregated edge information is then combined with the node’s original attributes and the global state to update the node features:

```math
v_i' = \phi_v \left( \bar{v}_i^{\,e} \oplus v_i \oplus u \right)
```

where (N_i^e) denotes the number of edges connected to atom (i), and (\phi_v) is the node update function. This step is equivalent to performing a local convolution on each atom, allowing its features to incorporate the influence of neighboring atoms.

### Global state update

The updated global state is determined by its original attributes as well as all updated atom and edge attributes in the graph. First, global aggregations over edges and nodes are computed:

```math
\bar{u}^{\,e} = \frac{1}{N^e} \sum_{k=1}^{N^e} e_k'
```

```math
\bar{u}^{\,v} = \frac{1}{N^v} \sum_{i=1}^{N^v} v_i'
```

The global state is then updated as:

```math
u' = \phi_u \left( \bar{u}^{\,e} \oplus \bar{u}^{\,v} \oplus u \right)
```

where (N^e) and (N^v) denote the total numbers of edges and nodes, respectively. By stacking multiple MEGNet modules, information among atoms, edges, and the global state can be propagated over multiple rounds, enabling the modeling of interactions ranging from local neighborhoods to long-range effects.

In terms of model architecture, two fully connected (Dense) layers are added before each MEGNet module for input preprocessing to enhance model flexibility. Residual connections are employed within the modules to support deep training and reduce the risk of overfitting. A Dense layer combined with a message-passing module constitutes one block, and multiple such blocks can be stacked to form a deep model. In the final readout stage, the set2set operation is used to encode sets of atomic and edge attributes into fixed-length vectors, which are then concatenated with the global state vector and passed through a multilayer perceptron to produce the final output. This design enables the prediction of material properties such as energy, band gap, or elastic modulus.

MEGNet is trained using the Adam optimizer with an initial learning rate of 0.001, which is dynamically reduced to 0.0001 during training. Most models converge within 1,000 epochs, while free-energy-related models typically require 2,000 to 4,000 epochs.


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
            <td  nowrap="nowrap">megnet_mp2018_train_60k_e_form</td>
            <td  nowrap="nowrap">mp2018_train_60k</td>
            <td  nowrap="nowrap">Form. Energy(meV/atom)</td>
            <td  nowrap="nowrap">28.3 / 26.5</td>
            <td  nowrap="nowrap">1</td>
            <td  nowrap="nowrap">~15 hours</td>
            <td  nowrap="nowrap"><a href="megnet_mp2018_train_60k_e_form.yaml">megnet_mp2018_train_60k_e_form</a></td>
            <td  nowrap="nowrap"><a href="https://paddle-org.bj.bcebos.com/paddlematerial/checkpoints/property_prediction/megnet/megnet_mp2018_train_60k_e_form.zip">checkpoint | log</a></td>
        </tr>
        <tr>
            <td  nowrap="nowrap">megnet_mp2018_train_60k_band_gap</td>
            <td  nowrap="nowrap">mp2018_train_60k</td>
            <td  nowrap="nowrap">band gap</td>
            <td  nowrap="nowrap"> 0.2962 / 0.2934</td>
            <td  nowrap="nowrap">1</td>
            <td  nowrap="nowrap">~20 hours</td>
            <td  nowrap="nowrap"><a href="megnet_mp2018_train_60k_band_gap.yaml">megnet_mp2018_train_60k_band_gap</a></td>
            <td  nowrap="nowrap"><a href="https://paddle-org.bj.bcebos.com/paddlematerial/checkpoints/property_prediction/megnet/megnet_mp2018_train_60k_band_gap.zip">checkpoint | log</a></td>
        </tr>
        <tr>
            <td  nowrap="nowrap">megnet_mp2018_train_60k_G</td>
            <td  nowrap="nowrap">mp2018_train_60k</td>
            <td  nowrap="nowrap">G</td>
            <td  nowrap="nowrap">0.0836 / 0.0962</td>
            <td  nowrap="nowrap">1</td>
            <td  nowrap="nowrap">~1.5 hours</td>
            <td  nowrap="nowrap"><a href="megnet_mp2018_train_60k_G.yaml">megnet_mp2018_train_60k_G</a></td>
            <td  nowrap="nowrap"><a href="https://paddle-org.bj.bcebos.com/paddlematerial/checkpoints/property_prediction/megnet/megnet_mp2018_train_60k_G.zip">checkpoint | log</a></td>
        </tr>
        <tr>
            <td  nowrap="nowrap">megnet_mp2018_train_60k_K</td>
            <td  nowrap="nowrap">mp2018_train_60k</td>
            <td  nowrap="nowrap">K</td>
            <td  nowrap="nowrap">0.0512 / 0.0585</td>
            <td  nowrap="nowrap">1</td>
            <td  nowrap="nowrap">~1.5 hours</td>
            <td  nowrap="nowrap"><a href="megnet_mp2018_train_60k_K.yaml">megnet_mp2018_train_60k_K</a></td>
            <td  nowrap="nowrap"><a href="https://paddle-org.bj.bcebos.com/paddlematerial/checkpoints/property_prediction/megnet/megnet_mp2018_train_60k_K.zip">checkpoint | log</a></td>
        </tr>
        <tr>
            <td  nowrap="nowrap">megnet_mp2024_train_130k_e_form</td>
            <td  nowrap="nowrap">mp2024_train_130k</td>
            <td  nowrap="nowrap">Form. Energy(meV/atom)</td>
            <td  nowrap="nowrap">40.7 / 41.0</td>
            <td  nowrap="nowrap">1</td>
            <td  nowrap="nowrap">~48 hours</td>
            <td  nowrap="nowrap"><a href="megnet_mp2024_train_130k_e_form.yaml">megnet_mp2024_train_130k_e_form</a></td>
            <td  nowrap="nowrap"><a href="https://paddle-org.bj.bcebos.com/paddlematerial/checkpoints/property_prediction/megnet/megnet_mp2024_train_130k_e_form.zip">checkpoint | log</a></td>
        </tr>
        <tr>
            <td  nowrap="nowrap">megnet_jarvis_dft_2d_e_form</td>
            <td  nowrap="nowrap">Jarvis_dft_2d</td>
            <td  nowrap="nowrap">Form. Energy(meV/atom)</td>
            <td  nowrap="nowrap">313.910 / 286.372 </td>
            <td  nowrap="nowrap">1</td>
            <td  nowrap="nowrap">~0.25 hours</td>
            <td  nowrap="nowrap"><a href="megnet_jarvis_dft_2d_e_form.yaml">megnet_jarvis_dft_2d_e_form</a></td>
            <td  nowrap="nowrap"><a href="https://paddle-org.bj.bcebos.com/paddlematerial/checkpoints/property_prediction/megnet/megnet_jarvis_dft_2d_e_form.zip">checkpoint | log</a></td>
        </tr>
        <tr>
            <td  nowrap="nowrap">megnet_jarvis_dft_3d_e_form</td>
            <td  nowrap="nowrap">Jarvis_dft_3d</td>
            <td  nowrap="nowrap">Form. Energy(meV/atom)</td>
            <td  nowrap="nowrap"> 50.728 / 49.318 </td>
            <td  nowrap="nowrap">1</td>
            <td  nowrap="nowrap">~20 hours</td>
            <td  nowrap="nowrap"><a href="megnet_jarvis_dft_3d_e_form.yaml">megnet_jarvis_dft_3d_e_form</a></td>
            <td  nowrap="nowrap"><a href="https://paddle-org.bj.bcebos.com/paddlematerial/checkpoints/property_prediction/megnet/megnet_jarvis_dft_3d_e_form.zip">checkpoint | log</a></td>
        </tr>
        <tr>
            <td  nowrap="nowrap">megnet_jarvis_cfid_3d_e_form</td>
            <td  nowrap="nowrap">Jarvis_cfid_3d</td>
            <td  nowrap="nowrap">Form. Energy(meV/atom)</td>
            <td  nowrap="nowrap"> 0.056092 / 0.057279 </td>
            <td  nowrap="nowrap">1</td>
            <td  nowrap="nowrap">~18 hours</td>
            <td  nowrap="nowrap"><a href="megnet_jarvis_cfid_3d_e_form.yaml">megnet_jarvis_cfid_3d_e_form</a></td>
            <td  nowrap="nowrap"><a href="https://paddle-org.bj.bcebos.com/paddlematerial/checkpoints/property_prediction/megnet/megnet_jarvis_cfid_3d_formation_energy_peratom_t_20250807_092757_s_42.zip">checkpoint | log</a></td>
        </tr>
        <tr>
            <td  nowrap="nowrap">megnet_jarvis_cfid_3d_band_gap</td>
            <td  nowrap="nowrap">Jarvis_cfid_3d</td>
            <td  nowrap="nowrap">Band Gap(eV)</td>
            <td  nowrap="nowrap"> 0.172418 / 0.162828 </td>
            <td  nowrap="nowrap">1</td>
            <td  nowrap="nowrap">~12.5 hours</td>
            <td  nowrap="nowrap"><a href="megnet_jarvis_cfid_3d_band_gap.yaml">megnet_jarvis_cfid_3d_band_gap</a></td>
            <td  nowrap="nowrap"><a href="https://paddle-org.bj.bcebos.com/paddlematerial/checkpoints/property_prediction/megnet/megnet_jarvis_cfid_3d_optb88vdw_bandgap_t_20250807_095235_s_42.zip">checkpoint | log</a></td>
        </tr>
        <tr>
            <td  nowrap="nowrap">megnet_jarvis_cfid_3d_shear_modulus</td>
            <td  nowrap="nowrap">Jarvis_cfid_3d</td>
            <td  nowrap="nowrap">Shear Modulus(G)</td>
            <td  nowrap="nowrap"> 0.121244 / 0.117699 </td>
            <td  nowrap="nowrap">1</td>
            <td  nowrap="nowrap">~4.5 hours</td>
            <td  nowrap="nowrap"><a href="megnet_jarvis_cfid_3d_shear_modulus.yaml">megnet_jarvis_cfid_3d_shear_modulus</a></td>
            <td  nowrap="nowrap"><a href="https://paddle-org.bj.bcebos.com/paddlematerial/checkpoints/property_prediction/megnet/megnet_jarvis_cfid_3d_shear_modulus_gv_t_20250808_082145_s_42.zip">checkpoint | log</a></td>
        </tr>
        <tr>
            <td  nowrap="nowrap">megnet_jarvis_cfid_3d_bulk_modulus</td>
            <td  nowrap="nowrap">Jarvis_cfid_3d</td>
            <td  nowrap="nowrap">Bulk Modulus(K)</td>
            <td  nowrap="nowrap"> 0.138926 / 0.141083 </td>
            <td  nowrap="nowrap">1</td>
            <td  nowrap="nowrap">~4.5 hours</td>
            <td  nowrap="nowrap"><a href="megnet_jarvis_cfid_3d_bulk_modulus.yaml">megnet_jarvis_cfid_3d_bulk_modulus</a></td>
            <td  nowrap="nowrap"><a href="https://paddle-org.bj.bcebos.com/paddlematerial/checkpoints/property_prediction/megnet/megnet_jarvis_cfid_3d_bulk_modulus_kv_t_20250808_081712_s_42.zip">checkpoint | log</a></td>
        </tr>
            <tr>
            <td  nowrap="nowrap">megnet_jarvis_dft_3d_2021_e_form</td>
            <td  nowrap="nowrap">Jarvis_dft_3d_2021</td>
            <td  nowrap="nowrap">Form. Energy(meV/atom)</td>
            <td  nowrap="nowrap"> 0.048386 / 0.049537 </td>
            <td  nowrap="nowrap">1</td>
            <td  nowrap="nowrap">~12.5 hours</td>
            <td  nowrap="nowrap"><a href="megnet_jarvis_dft_3d_2021_e_form.yaml">megnet_jarvis_dft_3d_e_form</a></td>
            <td  nowrap="nowrap"><a href="https://paddle-org.bj.bcebos.com/paddlematerial/checkpoints/property_prediction/megnet/megnet_jarvis_dft_3d_2021_formation_energy_peratom_t_20250811_114527_s_42.zip">checkpoint | log</a></td>
        </tr>
        <tr>
            <td  nowrap="nowrap">megnet_jarvis_alex_pbe_2d_all_e_form</td>
            <td  nowrap="nowrap">Alex_pbe_2d_all</td>
            <td  nowrap="nowrap">Form. Energy(meV/atom)</td>
            <td  nowrap="nowrap"> 62.708 / 62.972 </td>
            <td  nowrap="nowrap">1</td>
            <td  nowrap="nowrap">~34 hours</td>
            <td  nowrap="nowrap"><a href="megnet_jarvis_alex_pbe_2d_all_e_form.yaml">megnet_jarvis_alex_pbe_2d_all_e_form</a></td>
            <td  nowrap="nowrap"><a href="https://paddle-org.bj.bcebos.com/paddlematerial/checkpoints/property_prediction/megnet/megnet_jarvis_alex_pbe_2d_all_e_form.zip">checkpoint | log</a></td>
        </tr>
        <tr>
            <td  nowrap="nowrap">megnet_jarvis_dft_2d_bandgap</td>
            <td  nowrap="nowrap">Jarvis_dft_2d_2020</td>
            <td  nowrap="nowrap">Band Gap(eV)</td>
            <td  nowrap="nowrap"> - </td>
            <td  nowrap="nowrap">1</td>
            <td  nowrap="nowrap"> - </td>
            <td  nowrap="nowrap"><a href="megnet_jarvis_dft_2d_bandgap.yaml">megnet_jarvis_dft_2d_bandgap</a></td>
            <td  nowrap="nowrap"><a href="-">checkpoint | log</a></td>
        </tr>
        <tr>
            <td  nowrap="nowrap">megnet_matbench_e_form</td>
            <td  nowrap="nowrap">Matbench</td>
            <td  nowrap="nowrap">Form. Energy(eV/atom)</td>
            <td  nowrap="nowrap"> 2.084808/ 2.072724 </td>
            <td  nowrap="nowrap">1</td>
            <td  nowrap="nowrap">~40 hours</td>
            <td  nowrap="nowrap"><a href="megnet_matbench_e_form.yaml">megnet_matbench_e_form</a></td>
            <td  nowrap="nowrap"><a href="https://paddle-org.bj.bcebos.com/paddlematerial/checkpoints/property_prediction/megnet/megnet_matbench_e_form_t_20250731_093333_s_42.zip">checkpoint | log</a></td>
        </tr>
        <tr>
            <td  nowrap="nowrap">megnet_matbench_band_gap</td>
            <td  nowrap="nowrap">Matbench</td>
            <td  nowrap="nowrap">Band Gap(eV)</td>
            <td  nowrap="nowrap"> 0.225403 / 0.226996 </td>
            <td  nowrap="nowrap">1</td>
            <td  nowrap="nowrap">~8 hours</td>
            <td  nowrap="nowrap"><a href="megnet_matbench_band_gap.yaml">megnet_matbench_band_gap</a></td>
            <td  nowrap="nowrap"><a href="https://paddle-org.bj.bcebos.com/paddlematerial/checkpoints/property_prediction/megnet/megnet_matbench_band_gap_t_20250731_041639_s_42.zip">checkpoint | log</a></td>
        </tr>
        <tr>
            <td  nowrap="nowrap">megnet_matbench_shear_modulus</td>
            <td  nowrap="nowrap">Matbench</td>
            <td  nowrap="nowrap">Shear Modulus (G)</td>
            <td  nowrap="nowrap"> 0.098680 / 0.093513 </td>
            <td  nowrap="nowrap">1</td>
            <td  nowrap="nowrap">~4 hours</td>
            <td  nowrap="nowrap"><a href="megnet_matbench_shear_modulus.yaml">megnet_matbench_shear_modulus</a></td>
            <td  nowrap="nowrap"><a href="https://paddle-org.bj.bcebos.com/paddlematerial/checkpoints/property_prediction/megnet/megnet_matbench_shear_modulus_t_20250731_041740_s_42.zip">checkpoint | log</a></td>
        </tr>
        <tr>
            <td  nowrap="nowrap">megnet_matbench_bulk_modulus</td>
            <td  nowrap="nowrap">Matbench</td>
            <td  nowrap="nowrap">Bulk Modulus (K)</td>
            <td  nowrap="nowrap"> 0.080528 / 0.077150 </td>
            <td  nowrap="nowrap">1</td>
            <td  nowrap="nowrap">~4 hours</td>
            <td  nowrap="nowrap"><a href="megnet_matbench_bulk_modulus.yaml">megnet_matbench_bulk_modulus</a></td>
            <td  nowrap="nowrap"><a href="https://paddle-org.bj.bcebos.com/paddlematerial/checkpoints/property_prediction/megnet/megnet_matbench_bulk_modulus_t_20250731_041800_s_42.zip">checkpoint | log</a></td>
        </tr>
        <tr>
            <td  nowrap="nowrap">megnet_tmqm_train_108k_dipole_m</td>
            <td  nowrap="nowrap">tmQM_108k</td>
            <td  nowrap="nowrap">Dipole Moment (Debye)</td>
            <td  nowrap="nowrap"> - </td>
            <td  nowrap="nowrap">1</td>
            <td  nowrap="nowrap"> - </td>
            <td  nowrap="nowrap"><a href="megnet_tmqm_train_108k_dipole_m.yaml">megnet_tmqm_train_108k_dipole_m</a></td>
            <td  nowrap="nowrap"><a href="None">checkpoint | log</a></td>
        </tr>
        <tr>
            <td  nowrap="nowrap">megnet_tmqm_train_108k_dispersion_e</td>
            <td  nowrap="nowrap">tmQM_108k</td>
            <td  nowrap="nowrap">Dispersion Energy (Hartree)</td>
            <td  nowrap="nowrap"> - </td>
            <td  nowrap="nowrap">1</td>
            <td  nowrap="nowrap"> - </td>
            <td  nowrap="nowrap"><a href="megnet_tmqm_train_108k_dispersion_e.yaml">megnet_tmqm_train_108k_dispersion_e</a></td>
            <td  nowrap="nowrap"><a href="None">checkpoint | log</a></td>
        </tr>
        <tr>
            <td  nowrap="nowrap">megnet_tmqm_train_108k_electronic_e</td>
            <td  nowrap="nowrap">tmQM_108k</td>
            <td  nowrap="nowrap">Electronic Energy (Hartree)</td>
            <td  nowrap="nowrap"> - </td>
            <td  nowrap="nowrap">1</td>
            <td  nowrap="nowrap"> - </td>
            <td  nowrap="nowrap"><a href="megnet_tmqm_train_108k_electronic_e.yaml">megnet_tmqm_train_108k_electronic_e</a></td>
            <td  nowrap="nowrap"><a href="None">checkpoint | log</a></td>
        </tr>
        <tr>
            <td  nowrap="nowrap">megnet_tmqm_train_108k_hl_gap</td>
            <td  nowrap="nowrap">tmQM_108k</td>
            <td  nowrap="nowrap">HOMO-LUMO Gap (eV)</td>
            <td  nowrap="nowrap"> - </td>
            <td  nowrap="nowrap">1</td>
            <td  nowrap="nowrap"> - </td>
            <td  nowrap="nowrap"><a href="megnet_tmqm_train_108k_hl_gap.yaml">megnet_tmqm_train_108k_hl_gap</a></td>
            <td  nowrap="nowrap"><a href="None">checkpoint | log</a></td>
        </tr>
        <tr>
            <td  nowrap="nowrap">megnet_tmqm_train_108k_homo_energy</td>
            <td  nowrap="nowrap">tmQM_108k</td>
            <td  nowrap="nowrap">HOMO Energy (eV)</td>
            <td  nowrap="nowrap"> - </td>
            <td  nowrap="nowrap">1</td>
            <td  nowrap="nowrap"> - </td>
            <td  nowrap="nowrap"><a href="megnet_tmqm_train_108k_homo_energy.yaml">megnet_tmqm_train_108k_homo_energy</a></td>
            <td  nowrap="nowrap"><a href="None">checkpoint | log</a></td>
        </tr>
        <tr>
            <td  nowrap="nowrap">megnet_tmqm_train_108k_lumo_energy</td>
            <td  nowrap="nowrap">tmQM_108k</td>
            <td  nowrap="nowrap">LUMO Energy (eV)</td>
            <td  nowrap="nowrap"> - </td>
            <td  nowrap="nowrap">1</td>
            <td  nowrap="nowrap"> - </td>
            <td  nowrap="nowrap"><a href="megnet_tmqm_train_108k_lumo_energy.yaml">megnet_tmqm_train_108k_lumo_energy</a></td>
            <td  nowrap="nowrap"><a href="None">checkpoint | log</a></td>
        </tr>
        <tr>
            <td  nowrap="nowrap">megnet_tmqm_train_108k_metal_q</td>
            <td  nowrap="nowrap">tmQM_108k</td>
            <td  nowrap="nowrap">Metal Charge (e)</td>
            <td  nowrap="nowrap"> - </td>
            <td  nowrap="nowrap">1</td>
            <td  nowrap="nowrap"> - </td>
            <td  nowrap="nowrap"><a href="megnet_tmqm_train_108k_metal_q.yaml">megnet_tmqm_train_108k_metal_q</a></td>
            <td  nowrap="nowrap"><a href="None">checkpoint | log</a></td>
        </tr>
        <tr>
            <td  nowrap="nowrap">megnet_tmqm_train_108k_polarizability</td>
            <td  nowrap="nowrap">tmQM_108k</td>
            <td  nowrap="nowrap">Polarizability (Bohr³)</td>
            <td  nowrap="nowrap"> - </td>
            <td  nowrap="nowrap">1</td>
            <td  nowrap="nowrap"> - </td>
            <td  nowrap="nowrap"><a href="megnet_tmqm_train_108k_polarizability.yaml">megnet_tmqm_train_108k_polarizability</a></td>
            <td  nowrap="nowrap"><a href="None">checkpoint | log</a></td>
        </tr>
        <tr>
            <td  nowrap="nowrap">megnet_omol25_dipole</td>
            <td  nowrap="nowrap">OMol25</td>
            <td  nowrap="nowrap"> - </td>
            <td  nowrap="nowrap"> - </td>
            <td  nowrap="nowrap"> - </td>
            <td  nowrap="nowrap"> - </td>
            <td  nowrap="nowrap"> ~ </td>
            <td  nowrap="nowrap"> ~ </td>
            <td  nowrap="nowrap"><a href="megnet_omol25_dipole.yaml">megnet_omol25_dipole</a></td>
            <td  nowrap="nowrap"><a href="-">checkpoint | log</a></td>
        </tr>  
        <tr>
            <td  nowrap="nowrap">megnet_omol25_gap</td>
            <td  nowrap="nowrap">OMol25</td>
            <td  nowrap="nowrap"> - </td>
            <td  nowrap="nowrap"> - </td>
            <td  nowrap="nowrap"> - </td>
            <td  nowrap="nowrap"> - </td>
            <td  nowrap="nowrap"> ~ </td>
            <td  nowrap="nowrap"> ~ </td>
            <td  nowrap="nowrap"><a href="megnet_omol25_gap.yaml">megnet_omol25_gap</a></td>
            <td  nowrap="nowrap"><a href="-">checkpoint | log</a></td>
        </tr>
        <tr>
            <td  nowrap="nowrap">megnet_omol25_homo</td>
            <td  nowrap="nowrap">OMol25</td>
            <td  nowrap="nowrap"> - </td>
            <td  nowrap="nowrap"> - </td>
            <td  nowrap="nowrap"> - </td>
            <td  nowrap="nowrap"> - </td>
            <td  nowrap="nowrap"> ~ </td>
            <td  nowrap="nowrap"> ~ </td>
            <td  nowrap="nowrap"><a href="megnet_omol25_homo.yaml">megnet_omol25_homo</a></td>
            <td  nowrap="nowrap"><a href="-">checkpoint | log</a></td>
        </tr>
        <tr>
            <td  nowrap="nowrap">megnet_omol25_lumo</td>
            <td  nowrap="nowrap">OMol25</td>
            <td  nowrap="nowrap"> - </td>
            <td  nowrap="nowrap"> - </td>
            <td  nowrap="nowrap"> - </td>
            <td  nowrap="nowrap"> - </td>
            <td  nowrap="nowrap"> ~ </td>
            <td  nowrap="nowrap"> ~ </td>
            <td  nowrap="nowrap"><a href="megnet_omol25_lumo.yaml">megnet_omol25_lumo</a></td>
            <td  nowrap="nowrap"><a href="-">checkpoint | log</a></td>
        </tr>
        <tr>
            <td  nowrap="nowrap">megnet_omol25_u0</td>
            <td  nowrap="nowrap">OMol25</td>
            <td  nowrap="nowrap"> - </td>
            <td  nowrap="nowrap"> - </td>
            <td  nowrap="nowrap"> - </td>
            <td  nowrap="nowrap"> - </td>
            <td  nowrap="nowrap"> ~ </td>
            <td  nowrap="nowrap"> ~ </td>
            <td  nowrap="nowrap"><a href="megnet_omol25_u0.yaml">megnet_omol25_u0</a></td>
            <td  nowrap="nowrap"><a href="-">checkpoint | log</a></td>
        </tr>
    </body>
</table>

### Training
```bash
# formation energy per atom
# multi-gpu training, we use 4 gpus here
python -m paddle.distributed.launch --gpus="0,1,2,3" property_prediction/train.py -c property_prediction/configs/megnet/megnet_mp2018_train_60k_e_form.yaml
# single-gpu training
python property_prediction/train.py -c property_prediction/configs/megnet/megnet_mp2018_train_60k_e_form.yaml

```

### Validation
```bash
# Run model evaluation on the validation dataset.
# Adjust program behavior on-the-fly using command-line parameters – this provides a convenient way to customize settings without modifying the configuration file directly.
# Trainer.pretrained_model_path specifies the path to the saved model checkpoint to be loaded.
# such as: --Global.do_eval=True

# formation energy per atom
python property_prediction/train.py \
    -c property_prediction/configs/megnet/megnet_mp2018_train_60k_e_form.yaml \
    Global.do_train=False \
    Global.do_eval=True \
    Global.do_test=False \
    Trainer.pretrained_model_path=output/megnet_mp2018_train_60k_e_form/checkpoints
```

### Testing
```bash
# This command is used to evaluate the model's performance on the test dataset.

# formation energy per atom
python property_prediction/train.py \
    -c property_prediction/configs/megnet/megnet_mp2018_train_60k_e_form.yaml \
    Global.do_train=False \
    Global.do_test=True \
    Global.do_eval=False \
    Trainer.pretrained_model_path=output/megnet_mp2018_train_60k_e_form/checkpoints

```

### Prediction

You can replace the `--model_name` parameter at  `Mode 1` with other model names from the `results` table.

```bash
# This command is used to predict the properties of new crystal structures using a trained model.
# Note: The model_name and weights_name parameters are used to specify the pre-trained model and its corresponding weights. The cif_file_path parameter is used to specify the path to the CIF files for which properties need to be predicted.
# The prediction results will be saved in a CSV file specified by the save_path parameter. Default save_path is 'result.csv'.

# formation energy per atom

# Mode 1: Leverage a pre-trained machine learning model for crystal formation energy prediction. The implementation includes automated model download functionality, eliminating the need for manual configuration.
python property_prediction/predict.py \
    --model_name='megnet_mp2018_train_60k_e_form' \
    --input_format=cif --input_path='./property_prediction/example_data/cifs/'

# Mode2: Use a custom configuration file and checkpoint for crystal formation energy prediction. This approach allows for more flexibility and customization.
python property_prediction/predict.py \
    --config_path='property_prediction/configs/megnet/megnet_mp2018_train_60k_e_form.yaml' \
    --checkpoint_path='you_checkpoint_path.pdparams' \
    --input_format=cif --input_path='./property_prediction/example_data/cifs/'

```


## Citation
```
@article{chen2019graph,
  title={Graph networks as a universal machine learning framework for molecules and crystals},
  author={Chen, Chi and Ye, Weike and Zuo, Yunxing and Zheng, Chen and Ong, Shyue Ping},
  journal={Chemistry of Materials},
  volume={31},
  number={9},
  pages={3564--3572},
  year={2019},
  publisher={ACS Publications}
}
```

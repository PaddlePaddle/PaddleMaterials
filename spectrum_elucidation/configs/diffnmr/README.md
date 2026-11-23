# DiffNMR

[DiffNMR: Diffusion Models for Nuclear Magnetic Resonance Spectra Elucidation](https://arxiv.org/abs/2507.08854)

## Abstract

Nuclear Magnetic Resonance (NMR) spectroscopy is a central characterization method for molecular structure elucidation, yet interpreting NMR spectra to deduce molecular structures remains challenging due to the complexity of spectral data and the vastness of the chemical space. In this work, we introduce DiffNMR, a novel end-to-end framework that leverages a conditional discrete diffusion model for de novo molecular structure elucidation from NMR spectra. DiffNMR refines molecular graphs iteratively through a diffusion-based generative process, ensuring global consistency and mitigating error accumulation inherent in autoregressive methods. The framework integrates a two-stage pretraining strategy that aligns spectral and molecular representations via diffusion autoencoder (Diff-AE) and contrastive learning, the incorporation of retrieval initialization and similarity filtering during inference, and a specialized NMR encoder with radial basis function (RBF) encoding for chemical shifts, preserving continuity and chemical correlation. Experimental results demonstrate that DiffNMR achieves competitive performance for NMR-based structure elucidation, offering an efficient and robust solution for automated molecular analysis.

![DiffNMR Overview](../../docs/diffnmr_overview.png)

## Datasets:

- MSD-NMR:

    MSD-NMR Multimodal-Spectroscopic-Dataset (MSD-NMR) is a comprehensive dataset for molecular structure elucidation from NMR spectra. It contains 121,509 spectra, each corresponding to a molecular structure with up to 15 heavy atoms. Up to 574,799 spectra with up to 35 heavy atoms. The dataset is divided into training, validation, and test sets.

    | Dataset | train | val | test | total |
    |:--------|------:|----:|-----:|------:|
    | [MSD-NMR](https://paddle-org.bj.bcebos.com/paddlematerial/datasets/msd/msd_nmr.zip) |  |  |  |  |
    | n<15    | 109,358 | 6,076  | 6,075  | 121,509 |
    | n<20    | 235,512 | 13,085 | 13,084 | 261,681 |
    | n<25    | 351,273 | 19,516 | 19,515 | 390,304 |
    | n<35    | 517,319 | 28,741 | 28,739 | 574,799 |

## Data Preparation

To set up the DiffNMR environment, please follow these steps:

1. Download the required files:
   - Vocabulary list: [vocab.tar.gz](https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/spectrum_elucidation/diffnmr/vocab.tar.gz)
   - Retrieval database: [retrival_database.zip](https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/spectrum_elucidation/diffnmr/retrival_database.zip)

2. Place the downloaded files in the `spectrum_elucidation` directory

3. Decompress the files using the following commands:
   ```bash
   tar -xvzf vocab.tar.gz
   unzip retrival_database.zip
   ```

## Results

<table>
    <head>
        <tr>
            <th  nowrap="nowrap">Model</th>
            <th  nowrap="nowrap">Dataset</th>
            <th  nowrap="nowrap">Loss</th>
            <th  nowrap="nowrap">Negative log likelihood</th>
            <th  nowrap="nowrap">GPUs</th>
            <th  nowrap="nowrap">Training time</th>
            <th  nowrap="nowrap">Config</th>
            <th  nowrap="nowrap">Checkpoint | Log</th>
        </tr>
    </head>
    <body>
        <tr>
            <td  nowrap="nowrap">diffnmr_diffgraphfromer_msdnmr_nless15</td>
            <td  nowrap="nowrap">msdnmr_nless15</td>
            <td  nowrap="nowrap">1.946618</td>
            <td  nowrap="nowrap">66.028621</td>
            <td  nowrap="nowrap">4</td>
            <td  nowrap="nowrap">~34.15 hours</td>
            <td  nowrap="nowrap"><a href="DiffNMR_DiffGraphFormer.yaml">DiffNMR_DiffGraphFormer</a></td>
            <td  nowrap="nowrap"><a href="">checkpoint | log</a></td>
        </tr>  
    </body>
    <body>
        <tr>
            <td  nowrap="nowrap">diffnmr_nmrnet_msdnmr_nless15</td>
            <td  nowrap="nowrap">msdnmr_nless15</td>
            <td  nowrap="nowrap">3.217951</td>
            <td  nowrap="nowrap">-</td>
            <td  nowrap="nowrap">4</td>
            <td  nowrap="nowrap">~6.5 hours</td>
            <td  nowrap="nowrap"><a href="DiffNMR_DiffGraphFormer.yaml">DiffNMR_NMRNet</a></td>
            <td  nowrap="nowrap"><a href="">checkpoint | log</a></td>
        </tr>  
    </body>
    <body>
        <tr>
            <td  nowrap="nowrap">diffnmr_msdnmr_nless15</td>
            <td  nowrap="nowrap">msdnmr_nless15</td>
            <td  nowrap="nowrap">1.946618</td>
            <td  nowrap="nowrap">66.028621</td>
            <td  nowrap="nowrap">4</td>
            <td  nowrap="nowrap">~30.24 hours</td>
            <td  nowrap="nowrap"><a href="DiffNMR_DiffGraphFormer.yaml">DiffNMR</a></td>
            <td  nowrap="nowrap"><a href="">checkpoint | log</a></td>
        </tr>  
    </body>
</table>

Note: please refer to the following pretrained weights:
- [DiffNMR_DiffGraphFormer_nless15_best.pdparams](https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/spectrum_elucidation/diffnmr/DiffNMR_DiffGraphFormer_nless15_best.pdparams)
- [DiffNMR_DiffGraphFormer_nless15_init.pdparams](https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/spectrum_elucidation/diffnmr/DiffNMR_DiffGraphFormer_nless15_init.pdparams)
- [DiffNMR_NMRNet_nless15_best.pdparams](https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/spectrum_elucidation/diffnmr/DiffNMR_NMRNet_nless15_best.pdparams)
- [DiffNMR_NMRNet_nless15_init.pdparams](https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/spectrum_elucidation/diffnmr/DiffNMR_NMRNet_nless15_init.pdparams)
- [DiffNMR_NMRNet_nless15_init_v2.pdparams](https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/spectrum_elucidation/diffnmr/DiffNMR_NMRNet_nless15_init_v2.pdparams)
- [DiffNMR_nless15_best.pdparams](https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/spectrum_elucidation/diffnmr/DiffNMR_nless15_best.pdparams)
- [DiffNMR_nless15_onlyH_best.pdparams](https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/spectrum_elucidation/diffnmr/DiffNMR_nless15_onlyH_best.pdparams)

### Training
```bash
## 2 stage pretraining
### stage 1: pretrain Diff-AE of Molecular Encoder and Molecular Decoder
# multi-gpu training, we use 4 gpus here
python -m paddle.distributed.launch --gpus="0,1,2,3" spectrum_elucidation/train.py -c spectrum_elucidation/configs/diffnmr/DiffNMR_DiffGraphFormer.yaml
# single-gpu training
python spectrum_elucidation/train.py -c spectrum_elucidation/configs/diffnmr/DiffNMR_DiffGraphFormer.yaml
### stage 2: pretrain NMR Spectrum Encoder NMRNet by CLIP
python -m paddle.distributed.launch --gpus="0,1,2,3" spectrum_elucidation/train.py -c spectrum_elucidation/configs/diffnmr/DiffNMR_NMRNet.yaml
# single-gpu training
python spectrum_elucidation/train.py -c spectrum_elucidation/configs/diffnmr/DiffNMR_NMRNet.yaml
## fine-tuning
# multi-gpu training, we use 4 gpus here
python -m paddle.distributed.launch --gpus="0,1,2,3" spectrum_elucidation/train.py -c spectrum_elucidation/configs/diffnmr/DiffNMR.yaml
# single-gpu training
python spectrum_elucidation/train.py -c spectrum_elucidation/configs/diffnmr/DiffNMR.yaml
```

### Validation
```bash
# Adjust program behavior on-the-fly using command-line parameters – this provides a convenient way to customize settings without modifying the configuration file directly.
# such as: --Global.do_eval=True
## 2 stage pretraining
### stage 1: pretrain Diff-AE of Molecular Encoder and Molecular Decoder
python spectrum_elucidation/train.py -c spectrum_elucidation/configs/diffnmr/DiffNMR_DiffGraphFormer.yaml Global.do_eval=True Global.do_train=False Global.do_test=False Trainer.pretrained_model_path='your model path(*.pdparams)'
### stage 2: pretrain NMR Spectrum Encoder NMRNet by CLIP
python spectrum_elucidation/train.py -c spectrum_elucidation/configs/diffnmr/DiffNMR_NMRNet.yaml Global.do_eval=True Global.do_train=False Global.do_test=False Trainer.pretrained_model_path='your model path(*.pdparams)'
## fine-tuning
python spectrum_elucidation/train.py -c spectrum_elucidation/configs/diffnmr/DiffNMR.yaml Global.do_eval=True Global.do_train=False Global.do_test=False Trainer.pretrained_model_path='your model path(*.pdparams)'
```

### Testing
```bash
# This command is used to evaluate the model's performance on the test dataset.
## 2 stage pretraining
### stage 1: pretrain Diff-AE of Molecular Encoder and Molecular Decoder
python spectrum_elucidation/train.py -c spectrum_elucidation/configs/diffnmr/DiffNMR_DiffGraphFormer.yaml Global.do_eval=False Global.do_train=False Global.do_test=True Trainer.pretrained_model_path='your model path(*.pdparams)'
### stage 2: pretrain NMR Spectrum Encoder NMRNet by CLIP
python spectrum_elucidation/train.py -c spectrum_elucidation/configs/diffnmr/DiffNMR_NMRNet.yaml Global.do_eval=False Global.do_train=False Global.do_test=True Trainer.pretrained_model_path='your model path(*.pdparams)'
## fine-tuning
python spectrum_elucidation/train.py -c spectrum_elucidation/configs/diffnmr/DiffNMR.yaml Global.do_eval=False Global.do_train=False Global.do_test=True Trainer.pretrained_model_path='your model path(*.pdparams)'
```

### Sample
```bash
# This command is used to predict the  crystal structure using a trained model.
# Note: The model_name and weights_name parameters are used to specify the pre-trained model and its corresponding weights.
# The prediction results will be saved in the folder specified by the `output_dir` parameter, with the default set to `result`.

# Mode 1: Use a custom configuration file and checkpoint for crystal structure prediction. This approach allows for more flexibility and customization.
python spectrum_elucidation/sample.py --config_path='spectrum_elucidation/configs/diffnmr/DiffNMR.yaml' --weights_name='DiffNMR_nless15_best.pdparams' --output_path='result_diffnmr_nless15/' --checkpoint_path="pretrained"

```

## Citation
```
@article{yang2025diffnmr,
  title={DiffNMR: Diffusion Models for Nuclear Magnetic Resonance Spectra Elucidation},
  author= {Yang, Qingsong and Wu, Binglan and Liu, Xuwei and Chen, Bo and Li, Wei and Long, Gen and Chen, Xin and Xiao, Mingjun},
  journal={arXiv preprint arXiv:2507.08854},
  year={2025}
}
```

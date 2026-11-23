# SE-Spectrum Enhancement

## 1.Introduction

Spectrum Enhancement (SE) targets reconstruction and denoising of noisy
spectral or microscopy observations. In materials characterization workflows,
low-dose or low-signal acquisitions often reduce image quality and make
atomic-scale structure analysis harder. SE models learn to recover cleaner
signals from paired noisy and reference data, improving downstream inspection
of crystal structures, defects, and local material morphology.

The current PaddleMaterials SE workflow focuses on STEM image enhancement.
Given noisy HAADF or BF STEM inputs, the model predicts a configured target
image (`gt_enhance` or `gt_detect`) and supports training, evaluation, and
prediction with the common PaddleMaterials trainer/predictor style.

## 2.Models Matrix

| **Supported Functions**             | **[SFIN](./configs/sfin/README.md)** |
| ----------------------------------- | ------------------------------------ |
| **Support Data Types**              |                                      |
| &emsp;STEM images                   | ✅                                   |
| &emsp;HAADF / BF inputs             | ✅                                   |
| **Spectrum Enhancement**            |                                      |
| &emsp;Image denoising/enhancement   | ✅                                   |
| &emsp;Detection target restoration  | ✅                                   |
| **ML Capabilities · Training**      |                                      |
| &emsp;Single-GPU                    | ✅                                   |
| &emsp;Distributed Train             | -                                    |
| &emsp;Mixed Precision               | -                                    |
| &emsp;Fine-tuning                   | ✅                                   |
| **ML Capabilities · Predict**       |                                      |
| &emsp;Standard inference            | ✅                                   |
| &emsp;Distributed inference         | -                                    |
| **Dataset**                         |                                      |
| &emsp;HAADF STEM                    | ✅                                   |
| &emsp;BF STEM                       | ✅                                   |

**Legend:** ✅ Verified · 🧪 Implemented, pending validation · 🚧 In development · `-` Not supported · 🌟 Original Work

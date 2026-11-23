# MLES-Machine Learning Electronic Structure

## 1.Introduction

Machine Learning Electronic Structure (MLES) is an emerging paradigm in computational chemistry and materials science that leverages machine learning to accelerate or even replace traditional ab initio electronic structure methods. It aims to retain quantum accuracy while drastically reducing computational costs. Current research in MLES can be broadly categorized into several directions: Neural Quantum States, Graph-Based Electronic Structure Models, ML Hamiltonians, Neural XC, SCF Accelerators etc. MLES has demonstrated strong potential in predicting material properties, guiding molecular design, and understanding catalytic mechanisms, making it an increasingly important tool in computational materials science and quantum chemistry.

## 2.Models Matrix

| **Supported Functions**                      | **[InfGCN](./configs/infgcn/README.md)** |
| -------------------------------------------- | :--------: |
| **Forward Prediction · Materials Properties**|            |
| Electron density                             |      ✅    |
| **ML Capabilities · Training**               |            |
| Single-GPU                                   |      ✅    |
| Distributed training                         |      ✅    |
| Mixed precision (AMP)                        |      —     |
| Fine-tuning                                  |      ✅    |
| Uncertainty / Active Learning                |      —     |
| Dynamic→Static graphs                        |      —     |
| Compiler (CINN) opt.                         |      —     |
| **ML Capabilities · Predict**                |            |
| Distillation / Pruning                       |      —     |
| Standard inference                           |      ✅    |
| Distributed inference                        |      —     |
| Compiler-level inference                     |      —     |
| **Datasets**                                 |            |
| **Materials Project**                        |            |
| MP_EC                                        |      ✅    |
| MD17_EC                                      |      ✅    |
| QM9_EC                                       |      ✅    |
| OMol25_EC                                    |      ✅    |

**Legend:** ✅ Verified · 🧪 Implemented, pending validation · 🚧 In development · `-` Not supported · 🌟 Original Work
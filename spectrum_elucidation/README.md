# SE-Spectrum Elucidation

## 1.Introduction

Spectrum Elucidation（SE） focuses on automatically (or semi‑automatically) deducing molecular scaffolds, functional groups, and 3‑D conformations of organic materials from multimodal spectral data—typically 1D/2D NMR, IR, Raman, and MS. The workflow combines experimental spectroscopy, cheminformatics, and machine learning. Efficient spectrum elucidation dramatically shortens the discovery cycle for organic semiconductors, optoelectronic materials, and functional polymers while reducing synthesis‑and‑test costs.

## 2.Model Matrix

| **Supported Functions**                      | **🌟[DiffNMR](./configs/diffnmr/README.md)** | **[AtomSegNet](./configs/atomsegnet/README.md)** |
| -------------------------------------------- | :------------------------------------------: | :----------------------------------------------: |
| **Support Material Types**                   |                                              |                                                  |
| Organic Materials                            |                      ✅                      |                                                  |
| Inorganic Materials                          |                      -                       |                       ✅                         |
| **Inverse Elucidate Molecules**              |                                              |                                                  |
| NMR to Molecular Structure                   |                      ✅                      |                       -                          |
| **Inverse Elucidate Crystalline**            |                                              |                       -                          |
| STEM to Crystatl Structures                  |                      -                       |                       -                          |
| XRD to Crystatl Structures                   |                      -                       |                       -                          |
| Atom segmentation                            |                      -                       |                       -                          |
| **ML Capabilities · Training**               |                                              |                       -                          |
| Single-GPU                                   |                      ✅                      |                       -                          |
| Distributed training                         |                      ✅                      |                       -                          |
| Mixed precision (AMP)                        |                      —                       |                       -                          |
| Fine-tuning                                  |                      ✅                      |                       -                          |
| Uncertainty / Active Learning                |                      —                       |                       -                          |
| Dynamic→Static graphs                        |                      —                       |                       -                          |
| Compiler (CINN) opt.                         |                      —                       |                       -                          |
| **ML Capabilities · Predict**                |                                              |                                                  |
| Distillation / Pruning                       |                      —                       |                       -                          |
| Standard inference                           |                      ✅                      |                       -                          |
| Distributed inference                        |                      —                       |                       -                          |
| Compiler-level inference                     |                      —                       |                       -                          |
| Retrival initilization                       |                      ✅                      |                       -                          |
| Similarity filter                            |                      ✅                      |                       -                          |
| Formula included                             |                      ✅                      |                       -                          |
| **Datasets**                                 |                                              |                                                  |
| **Multimodal Spectroscopic**                 |                                              |                                                  |
| NMR(Nuclear Magnetic Resonance)              |                      ✅                      |                       -                          |
| <small>n<15<small>                           |                      ✅                      |                       -                          |
| <small>n<20<small>                           |                      ✅                      |                       -                          |
| <small>n<25<small>                           |                      ✅                      |                       -                          |
| <small>n<35<small>                           |                      ✅                      |                       -                          |
| IR(InfraRed)                                 |                      -                       |                       -                          |
| MS(Mass Spectrum)                            |                      -                       |                       -                          |
| **TEMImageNet**                              |                      -                       |                       -                          |


**Legend:** ✅ Verified · 🧪 Implemented, pending validation · 🚧 In development · `-` Not supported · 🌟 Original Work
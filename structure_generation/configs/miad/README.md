# MiAD

[Mirage Atom Diffusion for De Novo Crystal Generation](https://arxiv.org/abs/2511.14426)

## Abstract

MiAD introduces mirage infusion, which lets a diffusion model change the number of atoms during crystal generation. Atoms can switch between real elements and a non-existent mirage state. The model jointly diffuses lattice matrices, fractional coordinates and atom types, using DiffCSP's CSPNet as its denoising backbone.

![MiAD Method Scheme](../../docs/miad_method_scheme.png)

---

## Model Description

### Overview

MiAD jointly diffuses lattice matrices, fractional coordinates and atom types.
Type 0 represents a mirage atom, allowing the final atom count to change during
generation.

### Method

- **Lattices:** DDPM with a clipped cosine noise schedule and no sample clipping.
- **Coordinates:** wrapped Gaussian diffusion with predictor-corrector sampling.
- **Atom types:** D3PM with uniform transitions and a cosine schedule.
- **Denoiser:** DiffCSP's CSPNet with periodic Fourier features.

Training pads crystals with mirage atoms to 25 candidate sites and excludes
mirage atoms from the coordinate loss. Sampling starts with the same capacity
and removes type-0 atoms from the output.

---

## Dataset Description

| Dataset | Train | Val | Test |
| --- | --- | --- | --- |
| [MP-20](https://paddle-org.bj.bcebos.com/paddlematerial/datasets/mp_20/mp_20.zip) | 27136 | 9047 | 9046 |

---

## Results

| Model | Dataset | S.U.N. (%) | Stable (%) | Unique (%) | Novel (%) | RMSD (Å) | Config | Checkpoint |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| miad_mp20 | MP-20 | 13.90 | 19.60 | 99.40 | 76.60 | 0.229 | [miad_mp20.yaml](miad_mp20.yaml) | [registry](../../../ppmat/models/__init__.py) |

Results use the converted official epoch-8000 checkpoint and 1000 structures
sampled from 25 candidates over 1000 diffusion steps. Relaxation uses CHGNet
(MPtrj), FIRE (fmax 0.05 eV/Å, up to 1500 steps), and the 2023-02-07 MP hull.
All samples enter the stability/S.U.N. denominators; 997 relaxations converged.
RMSD is the mean per-structure root mean square of same-index atomic
displacements, using the minimum image under the relaxed cell.

---

## Command

### Training

```bash
python structure_generation/train.py -c structure_generation/configs/miad/miad_mp20.yaml Trainer.pretrained_model_path=null
```

Remove `Trainer.pretrained_model_path=null` to fine-tune the pretrained model.

### Validation

```bash
python structure_generation/train.py -c structure_generation/configs/miad/miad_mp20.yaml Global.do_eval=True Global.do_train=False Global.do_test=False
```

### Testing

```bash
python structure_generation/train.py -c structure_generation/configs/miad/miad_mp20.yaml Global.do_eval=False Global.do_train=False Global.do_test=True
```

### Sample

```bash
python structure_generation/sample.py --model_name=miad_mp20 --mode=by_dataloader
```

The default generates 16 structures with 25 candidate sites each. Mirage
filtering can reduce the final atom count. `compute_metric` reports uniqueness
and novelty; stability and S.U.N. require externally supplied energies above hull.

---

## Citation

```
@article{okhotin2025miad,
  title={MiAD: Mirage Atom Diffusion for De Novo Crystal Generation},
  author={Okhotin, Andrey and Nakhodnov, Maksim and Kazeev, Nikita and Lazarev, Mikhail and Ustyuzhanin, Andrey E and Vetrov, Dmitry},
  journal={arXiv preprint arXiv:2511.14426},
  year={2025},
  url={https://arxiv.org/abs/2511.14426}
}
```

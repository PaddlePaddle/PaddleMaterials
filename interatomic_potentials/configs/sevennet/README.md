# SevenNet

SevenNet predicts total energy, atomic forces, and stress for periodic crystal
structures. This PaddleMaterials port uses the shared MLIP trainer and
PotentialPredictor. Data preparation uses the existing MPTrjDataset
structure/graph cache and the shared FindPointsInSpheres PGL converter.
SevenNet uses the shared e3nn layers, CHGNet's polynomial cutoff, and
PaddleMaterials loss and scatter functions. Force training uses Paddle's
decomposed einsum implementation for second-order gradients. The shared
Gate sorts irreps with the e3nn API, and Extract assembles its output with
slice and concat operations that support force-loss backpropagation.

The pretrained package contains weights converted from upstream SevenNet-0.

## Environment

Run these commands from the repository root after following the
[PaddleMaterials installation guide](../../../Install.md). The verification
runs used a single GPU, Python 3.10, and PaddlePaddle GPU 3.3.0.

## Data

For larger training runs, use the PaddleMaterials
[MPtrj_2022.9_full archive](https://paddle-org.bj.bcebos.com/paddlematerial/datasets/mptrj/MPtrj_2022.9_full.zip).
Place the extracted train.json under data/MPtrj_2022.9_full/.
The dataset loader can also download the archive when the file is absent.
The configured labels are per-atom energy, forces, and stress.
The validation data archive supplied with the PR contains the 15-frame
training sample and eight structures with paired upstream predictions.

## Training

Extract sevennet-validation-data.zip in the repository root to reproduce the
one-epoch MPtrj sample run:

~~~bash
FLAGS_new_einsum=0 python -m interatomic_potentials.train \
  --config interatomic_potentials/configs/sevennet/sevennet_mptrj.yaml \
  Trainer.max_epochs=1 Trainer.save_freq=1 \
  Trainer.output_dir=./output/sevennet_mptrj_sample \
  Dataset.train.dataset.__init_params__.path=./sevennet-validation-data/mptrj_train_sample.json \
  Dataset.train.dataset.__init_params__.cache_path=./output/sevennet_mptrj_sample_pgl_cache \
  Dataset.train.sampler.__init_params__.drop_last=false
~~~

The MPtrj recipe uses three convolution layers, a 4.5 Å cutoff, Adam
with a learning rate of 0.001, and a batch size of 6. To launch full-dataset
training, set the same einsum flag and use the YAML without the overrides after
preparing train.json. On PowerShell, set `$env:FLAGS_new_einsum = '0'`
before running the training command.
The sample run does not establish full-dataset training accuracy.
For separate dataset splits or custom paths, set a distinct cache_path.
Caches from the earlier SevenNet graph representation must be regenerated.

## Pretrained prediction

Extract sevennet0.zip in the repository root. Use the configuration in
that archive with its matching checkpoint:

~~~bash
python -m interatomic_potentials.predict \
  --config_path sevennet0/sevennet0.yaml \
  --checkpoint_path sevennet0/checkpoints/latest.pdparams \
  --input_format cif --input_path /path/to/structure.cif
~~~

The outputs use eV for energy, eV/Å for force, and eV/Å³ for Voigt stress.
Use the YAML saved by the trainer when loading a locally trained checkpoint.

## Evaluation and results

Run the integration tests from the repository root:

~~~bash
python -m pytest -q test/models/sevennet test/test_scatter.py
~~~

Extract both review archives in the repository root to run pretrained parity:

~~~bash
SEVENNET_REVIEW_ASSETS=. python -m pytest -q -s test/models/sevennet -k pretrained
~~~

| Run | Dataset | Result | Artifact in the review submission |
| --- | --- | --- | --- |
| MPtrj sample | 15 training frames | 1 epoch, 3 optimizer steps | sevennet-mptrj-sample-train.log |
| SevenNet-0 conversion | 8 structures | Maximum absolute differences: 7.63e-06 eV, 9.54e-06 eV/Å, 3.13e-07 eV/Å³ | sevennet0.zip and sevennet0-parity.log |

The parity comparison checks the converted checkpoint against upstream
predictions. It reports neither test-set MAE nor from-scratch replication
of the upstream training results.

## References

- [SevenNet source and upstream weights](https://github.com/MDIL-SNU/SevenNet)
- [SevenNet-0 pretrained model documentation](https://sevennet.readthedocs.io/en/latest/user_guide/pretrained.html#sevennet-0)
- [MPtrj dataset](https://figshare.com/articles/dataset/Materials_Project_Trjectory_MPtrj_Dataset/23713842)

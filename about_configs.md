# About Configs 🧩

PaddleMaterials implements full lifecycle management for model training, covering core stages like training, fine-tuning, and prediction. It includes standardized datasets and build-in pre-trained model libraries, supporting one-click prediction. Training workflows are parameterized through structured configuration files, allowing end-to-end model training with simple parameter adjustments.

<table>
    <thead>
        <tr>
            <th>Field Name</th>
            <th>Description</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>Global</td>
            <td>System-level parameters for centralized management of public configurations and cross-module shared settings.</td>
        </tr>
        <tr>
            <td>Trainer</td>
            <td>Defines core training parameters including epoch count, checkpoint saving policies, and distributed training configurations.</td>
        </tr>
        <tr>
            <td>Model</td>
            <td>Neural network architecture definition module with initialization parameters and loss function configurations.</td>
        </tr>
        <tr>
            <td>Dataset</td>
            <td>Standardized data loading with integrated preprocessing, batching, and multi-process reading mechanisms.</td>
        </tr>
        <tr>
            <td>Metric</td>
            <td>Evaluation metric functions for performance assessment during training and testing.</td>
        </tr>
        <tr>
            <td>Optimizer</td>
            <td>Optimizer configuration interface supporting learning rate scheduling, weight decay, and gradient clipping parameters.</td>
        </tr>
        <tr>
            <td>Predict</td>
            <td>Configuration parameters for prediction workflows.</td>
        </tr>
    </tbody>
</table>

Next, we demonstrate the configuration structure using MegNet training on the mp2018.6.1 dataset. The complete configuration file is available at [megnet_mp2018_train_60k_e_form.yaml](./property_prediction/configs/megnet/megnet_mp2018_train_60k_e_form.yaml). This configuration enables training of the MegNet model on mp2018.6.1 for formation energy, with the trained model capable of predicting formation energy for input structures.

## 1. Global Configuration
```yaml
Global:
# For mp2018 dataset, property names include:
# "formation_energy_per_atom", "band_gap", "G", "K"
label_names: ["formation_energy_per_atom"]
do_train: True
do_eval: False
do_test: False

graph_converter:
    __class_name__: FindPointsInSpheres
    __init_params__:
        cutoff: 4.0
        num_cpus: 10
```

<table>
    <thead>
        <tr>
            <th>Field Name</th>
            <th>Type</th>
            <th>Description</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>label_names</td>
            <td>List[str]</td>
            <td>Defines model training targets (must match dataset column names exactly). This example enables only formation energy prediction.</td>
        </tr>
        <tr>
            <td>do_train</td>
            <td>Bool</td>
            <td>Enables/disables training loop execution.</td>
        </tr>
        <tr>
            <td>do_eval</td>
            <td>Bool</td>
            <td>Enables/disables standalone evaluation process (independent of periodic validation during training).</td>
        </tr>
        <tr>
            <td>do_test</td>
            <td>Bool</td>
            <td>Enables/disables inference testing (disabled by default).</td>
        </tr>
        <tr>
            <td>graph_converter</td>
            <td>Class Config</td>
            <td>Material structure to graph conversion configuration for data loading and prediction stages.</td>
        </tr>
    </tbody>
</table>

PaddleMaterials uses `__class_name__` and `__init_params__` for flexible class instantiation without hardcoding, enabling different graph construction methods through configuration changes.

## 2. Trainer Configuration

The Trainer section initializes a `BaseTrainer` object controlling training, evaluation, and testing workflows:

```yaml
Trainer:
  max_epochs: 2000
  seed: 42
  output_dir: ./output/megnet_mp2018_train_60k_e_form
  save_freq: 100
  log_freq: 20
  start_eval_epoch: 1
  eval_freq: 1
  pretrained_model_path: null
  pretrained_weight_name: null
  resume_from_checkpoint: null
  use_amp: False
  amp_level: 'O1'
  eval_with_no_grad: True
  gradient_accumulation_steps: 1
  best_metric_indicator: 'eval_metric'
  name_for_best_metric: "formation_energy_per_atom"
  greater_is_better: False
  compute_metric_during_train: True
  metric_strategy_during_eval: 'epoch'
  use_visualdl: False
  use_wandb: False
  use_tensorboard: False
```

<table>
    <thead>
        <tr>
            <th>Field Name</th>
            <th>Type</th>
            <th>Description</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>max_epochs</td>
            <td>int</td>
            <td>Maximum training epochs.</td>
        </tr>
        <tr>
            <td>seed</td>
            <td>int</td>
            <td>Random seed for reproducibility (controls numpy/paddle/random libraries).</td>
        </tr>
        <tr>
            <td>output_dir</td>
            <td>str</td>
            <td>Output directory for model weights and logs.</td>
        </tr>
        <tr>
            <td>save_freq</td>
            <td>int</td>
            <td>Checkpoint saving interval (epochs). Set to 0 for final epoch-only saving.</td>
        </tr>
        <tr>
            <td>log_freq</td>
            <td>int</td>
            <td>Training log interval (steps).</td>
        </tr>
        <tr>
            <td>start_eval_epoch</td>
            <td>int</td>
            <td>Epoch to begin evaluation (avoids early-stage fluctuations).</td>
        </tr>
        <tr>
            <td>eval_freq</td>
            <td>int</td>
            <td>Evaluation interval (epochs). Set to 0 to disable periodic validation.</td>
        </tr>
        <tr>
            <td>pretrained_model_path</td>
            <td>str/None</td>
            <td>Pre-trained model path (None = no pre-training).</td>
        </tr>
        <tr>
            <td>pretrained_weight_name</td>
            <td>str/None</td>
            <td>When using the built-in model, specify the exact weight file name (e.g., latest.pdparams).</td>
        </tr>
        <tr>
            <td>resume_from_checkpoint</td>
            <td>str/None</td>
            <td>Checkpoint path for training resumption (requires optimizer state and training metadata).</td>
        </tr>
        <tr>
            <td>use_amp</td>
            <td>bool</td>
            <td>Enables automatic mixed precision training.</td>
        </tr>
        <tr>
            <td>amp_level</td>
            <td>str</td>
            <td>Mixed precision mode ('O1'=partial FP32, 'O2'=FP16 optimization).</td>
        </tr>
        <tr>
            <td>eval_with_no_grad</td>
            <td>bool</td>
            <td>Disables gradient computation during evaluation (set to False for models with higher-order derivatives).</td>
        </tr>
        <tr>
            <td>gradient_accumulation_steps</td>
            <td>int</td>
            <td>Gradient accumulation steps for large batch simulation.</td>
        </tr>
        <tr>
            <td>best_metric_indicator</td>
            <td>str</td>
            <td>Metric for best model selection (train/eval loss/metric).</td>
        </tr>
        <tr>
            <td>name_for_best_metric</td>
            <td>str</td>
            <td>Specific metric name (must match Metric configuration).</td>
        </tr>
        <tr>
            <td>greater_is_better</td>
            <td>bool</td>
            <td>Metric optimization direction (False = lower is better).</td>
        </tr>
        <tr>
            <td>compute_metric_during_train</td>
            <td>bool</td>
            <td>Enables training set metric computation.</td>
        </tr>
        <tr>
            <td>metric_strategy_during_eval</td>
            <td>str</td>
            <td>Evaluation strategy (an "epoch" refers to calculations performed after completing a full pass through the entire dataset, whereas a "step" denotes incremental calculations processed with each individual batch.).</td>
        </tr>
        <tr>
            <td>use_visualdl/wandb/tensorboard</td>
            <td>bool</td>
            <td>Enables specific training logging tools.</td>
        </tr>
    </tbody>
</table>

## 3. Model Configuration

Defines model architecture and hyperparameters. Example for MEGNetPlus:

```yaml
Model:
  __class_name__: MEGNetPlus
  __init_params__:
    dim_node_embedding: 16
    dim_edge_embedding: 100
    dim_state_embedding: 2
    nblocks: 3
    nlayers_set2set: 1
    niters_set2set: 2
    bond_expansion_cfg:
      rbf_type: "Gaussian"
      initial: 0.0
      final: 5.0
      num_centers: 100
      width: 0.5
    property_name: ${Global.label_names}
    data_mean: -1.6519
    data_std: 1.0694
```

<table>
    <thead>
        <tr>
            <th>Field Name</th>
            <th>Type</th>
            <th>Description</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>__class_name__</td>
            <td>str</td>
            <td>Model class name.</td>
        </tr>
        <tr>
            <td>__init_params__</td>
            <td>dict</td>
            <td>Initialization parameters (e.g., node embedding dimension).</td>
        </tr>
    </tbody>
</table>

## 4. Metric Configuration

Defines evaluation metrics. Example:

```yaml
Metric:
  formation_energy_per_atom:
    __class_name__: paddle.nn.L1Loss
    __init_params__: {}
```

Specifies metrics for specific properties (e.g., MAE for formation energy).

<table>
    <thead>
        <tr>
            <th>Field Name</th>
            <th>Type</th>
            <th>Description</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>__class_name__</td>
            <td>str</td>
            <td>Metric class name (supports PaddlePaddle APIs).</td>
        </tr>
        <tr>
            <td>__init_params__</td>
            <td>dict</td>
            <td>Initialization parameters (empty dict if none).</td>
        </tr>
    </tbody>
</table>

## 5. Optimizer Configuration

Defines optimizer and learning rate parameters. Example:

```yaml
Optimizer:
  __class_name__: Adam
  __init_params__:
    beta1: 0.9
    beta2: 0.999
    lr:
      __class_name__: Cosine
      __init_params__:
        learning_rate: 0.001
        eta_min: 0.0001
        by_epoch: True
```

<table>
    <thead>
        <tr>
            <th>Field Name</th>
            <th>Type</th>
            <th>Description</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>__class_name__</td>
            <td>str</td>
            <td>Optimizer class name (e.g., Adam).</td>
        </tr>
        <tr>
            <td>__init_params__</td>
            <td>dict</td>
            <td>Optimizer parameters (e.g., beta1/beta2 for Adam).</td>
        </tr>
        <tr>
            <td>lr.__class_name__</td>
            <td>str</td>
            <td>Learning rate scheduler class name (e.g., Cosine).</td>
        </tr>
        <tr>
            <td>lr.__init_params__</td>
            <td>dict</td>
            <td>Scheduler parameters (e.g., initial/min learning rates).</td>
        </tr>
        <tr>
            <td>clip_norm</td>
            <td>float</td>
            <td>Clips the gradient of each parameter tensor to the given L2 norm; see `paddle.nn.ClipGradByNorm`.</td>
        </tr>
        <tr>
            <td>clip_norm_global</td>
            <td>float</td>
            <td>Scales all gradients together when their global L2 norm exceeds the value, keeping the gradient direction; see `paddle.nn.ClipGradByGlobalNorm`.</td>
        </tr>
        <tr>
            <td>clip_value</td>
            <td>float</td>
            <td>Clips every gradient element to the range [-value, value]; see `paddle.nn.ClipGradByValue`.</td>
        </tr>
    </tbody>
</table>

At most one of the three keys is applied, checked in the order `clip_norm`, `clip_norm_global`, `clip_value`; when none is present, gradients are not clipped. `clip_norm` and `clip_norm_global` bound the overall step size and preserve the gradient direction, which suits models whose gradients are dominated by a few large components, such as the force and band-gap models trained with Adam. `clip_value` bounds each element independently and changes the gradient direction, so it suits the per-element noise-prediction objectives used by the generative models. Under gradient accumulation the trainer sums the gradients of the accumulated steps without averaging, so a `clip_value` threshold has to be scaled by `gradient_accumulation_steps` to keep the per-step constraint the same; `clip_norm` and `clip_norm_global` are applied to the summed gradient and need no such scaling.

## 6. Dataset Configuration

Defines dataset classes and parameters. Example:

```yaml
Dataset:
  train:
    dataset:
      __class_name__: MP2018Dataset
      __init_params__:
        path: "./data/mp2018_train_60k/mp.2018.6.1_train.json"
        property_names: ${Global.label_names}
        build_structure_cfg:
          format: cif_str
          num_cpus: 10
        build_graph_cfg: ${Global.graph_converter}
        cache_path: "./data/mp2018_train_60k_cache_find_points_in_spheres_cutoff_4/mp.2018.6.1_train"
      num_workers: 4
      use_shared_memory: False
    sampler:
      __class_name__: BatchSampler
      __init_params__:
        shuffle: True
        drop_last: True
        batch_size: 128
  val:
    # Similar structure to train with validation-specific parameters
  test:
    # Similar structure to train with test-specific parameters
```

<table>
    <thead>
        <tr>
            <th>Field Name</th>
            <th>Type</th>
            <th>Description</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>train.dataset.__class_name__</td>
            <td>str</td>
            <td>Dataset class name (e.g., MP2018Dataset).</td>
        </tr>
        <tr>
            <td>train.dataset.__init_params__.path</td>
            <td>str</td>
            <td>Data file path.</td>
        </tr>
        <tr>
            <td>train.dataset.__init_params__.property_names</td>
            <td>str</td>
            <td>Target properties (references Global labels).</td>
        </tr>
        <tr>
            <td>train.dataset.__init_params__.build_structure_cfg</td>
            <td>dict</td>
            <td>Material structure construction parameters.</td>
        </tr>
        <tr>
            <td>train.sampler.__init_params__.batch_size</td>
            <td>int</td>
            <td>Training batch size (per GPU).</td>
        </tr>
    </tbody>
</table>

## 7. Predict Configuration

Defines prediction parameters. Example:

```yaml
Predict:
  graph_converter: ${Global.graph_converter}
  eval_with_no_grad: True
```

References global graph converter and disables gradient computation during prediction (set to False for models with higher-order derivatives).

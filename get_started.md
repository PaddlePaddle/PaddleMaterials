# Get Started  ⚡

PaddleMaterials provides multiple pre-trained models and standard datasets for material property prediction, material structure generation, and interatomic potentials tasks. This document demonstrates how to perform common tasks using these existing models and standard datasets.

Training workflows are parameterized through structured configuration files, allowing end-to-end model training with simple parameter adjustments. You can refer to the [PaddleMaterials Configuration](./about_configs.md) section for detailed configuration information.

We have provided commands for training, evaluation, testing, and inference in each model's README file. You can also refer directly to these README files to complete corresponding tasks.

## 1. Inference with Existing Model

You can perform inference using either built-in models or local models.

### 1.1 Inference with Built-in Model

PaddleMaterials offers multiple built-in models that can be directly used for inference. Taking the `megnet_mp2018_train_60k_e_form` model as an example (a MEGNet model trained on the MP2018 dataset for material formation energy prediction), use the following command for inference:
```bash
python property_prediction/predict.py --model_name='megnet_mp2018_train_60k_e_form' --weights_name='best.pdparams' --input_format=cif --input_path='./property_prediction/example_data/cifs/' --output_path='results/'
```

<table>
    <thead>
        <tr>
            <th>Parameter</th>
            <th>Description</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>--model_name</td>
            <td>Name of the built-in model</td>
        </tr>
        <tr>
            <td>--weights_name</td>
            <td>Weights file name</td>
        </tr>
        <tr>
            <td>--input_format</td>
            <td>Input file format, such as cif or xyz</td>
        </tr>
        <tr>
            <td>--input_path</td>
            <td>Path to CIF files for prediction</td>
        </tr>
        <tr>
            <td>--output_path</td>
            <td>Path to save prediction results</td>
        </tr>
    </tbody>
</table>

### 1.2 Inference with Local Model

In addition to built-in models, you can also use your own locally trained models for inference. Taking the `megnet_mp2018_train_60k_e_form` model as an example (assuming you've trained it locally), use the following command:
```bash
python property_prediction/predict.py --config_path='property_prediction/configs/megnet/megnet_mp2018_train_60k_e_form.yaml' --checkpoint_path='you_checkpoint_path.pdparams' --input_format=cif --input_path='./property_prediction/example_data/cifs/' --output_path='results/'
```

<table>
    <thead>
        <tr>
            <th>Parameter</th>
            <th>Description</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>--config_path</td>
            <td>Configuration file path</td>
        </tr>
        <tr>
            <td>--checkpoint_path</td>
            <td>Model weights file path</td>
        </tr>
        <tr>
            <td>--input_format</td>
            <td>Input file format, such as cif or xyz</td>
        </tr>
        <tr>
            <td>--input_path</td>
            <td>Path to CIF files for prediction</td>
        </tr>
        <tr>
            <td>--output_path</td>
            <td>Path to save prediction results</td>
        </tr>
    </tbody>
</table>

## 2. Test Existing Models on Standard Datasets

To test the `megnet_mp2018_train_60k_e_form` model (assuming you've trained it locally) on the MP2018 test set, use:
```bash
python property_prediction/train.py -c property_prediction/configs/megnet/megnet_mp2018_train_60k_e_form.yaml Global.do_test=True Global.do_train=False Global.do_eval=False Trainer.pretrained_model_path='your_checkpoint_path(*.pdparams)' Trainer.output_dir='your_output_dir'
```

<table>
    <thead>
        <tr>
            <th>Parameter</th>
            <th>Description</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>-c</td>
            <td>Configuration file path</td>
        </tr>
        <tr>
            <td>Global.do_train</td>
            <td>Set to False for testing</td>
        </tr>
        <tr>
            <td>Global.do_eval</td>
            <td>Whether to evaluate on validation set</td>
        </tr>
        <tr>
            <td>Global.do_test</td>
            <td>Whether to evaluate on test set</td>
        </tr>
        <tr>
            <td>Trainer.pretrained_model_path</td>
            <td>Your model weights path</td>
        </tr>
        <tr>
            <td>Trainer.output_dir</td>
            <td>Output directory for log files</td>
        </tr>
    </tbody>
</table>

## 3. Train Predefined Models on Standard Datasets

You can train models using PaddleMaterials's standard datasets and predefined configurations. For the `megnet_mp2018_train_60k_e_form` model:
```bash
# Single-GPU training for formation energy per atom
python property_prediction/train.py -c property_prediction/configs/megnet/megnet_mp2018_train_60k_e_form.yaml
```

This command uses the `-c` parameter to specify the model configuration file. Training will be performed on the MP2018 training set, with logs saved to `Trainer.output_dir` by default (you can modify this path in the configuration file).

PaddleMaterials also supports multi-GPU training using `paddle.distributed.launch`:
```bash
# Multi-GPU training with 4 GPUs
python -m paddle.distributed.launch --gpus="0,1,2,3" property_prediction/train.py -c property_prediction/configs/megnet/megnet_mp2018_train_60k_e_form.yaml
```

The `--gpus` parameter specifies the GPU IDs and quantity to use.

## 4. Train with Customized Datasets

PaddleMaterials supports training with custom datasets. If your dataset format matches the standard format, you can directly use the provided configurations by modifying the dataset paths:

```yaml
...
Dataset:
  train:
    dataset:
      __class_name__: MP2018Dataset
      __init_params__:
        path: "your_train_data.json"
...
  val:
    dataset:
      __class_name__: MP2018Dataset
      __init_params__:
        path: "your_val_data.json"
...
  test:
    dataset:
      __class_name__: MP2018Dataset
      __init_params__:
        path: "your_test_data.json"
```

For datasets with different formats, you can either:
1. Create a custom dataset class, import it in `ppmat/datasets/__init__.py`, and modify the configuration
2. Convert your dataset to PaddleMaterials's supported format (recommended for convenience)

## 5. Train with Customized Models and Standard Datasets

1. Implement your custom model class (inheriting from `nn.Layer`) and import it in `ppmat/models/__init__.py`
   > Your model must implement `__init__` and `forward` methods. The `forward` method should return a dictionary containing model outputs and losses.

2. Copy the configuration file of the standard dataset you want to use (e.g., `megnet_mp2018_train_60k_e_form.yaml` for MP2018)

3. Modify the `Model` section in the configuration to use your custom model:
    ```yaml
    Model:
      __class_name__: your_model_class_name
      __init_params__:
          your_model_parameters
    ```

4. Adjust other hyperparameters (learning rate, batch size, etc.) as needed

5. Start training with the modified configuration file

## 6. Finetuning Models

PaddleMaterials supports model finetuning. Follow these steps using standard configurations (only need to modify pretrained model path):

1. Prepare your custom dataset (refer to Section 4)
2. Copy the original model configuration file (e.g., `megnet_mp2018_train_60k_e_form.yaml`)
3. Modify dataset paths in the copied configuration to point to your custom data
4. Configure pretrained model parameters:
   - For local models: Set `Trainer.pretrained_model_path` to your local path
   - For built-in models:
     - Set `Trainer.pretrained_model_path` to the built-in model URL
     - Set `Trainer.pretrained_weight_name` to the weights file name (e.g., `latest.pdparams`)
5. Adjust training parameters (learning rate, batch size, log directory, etc.)
6. Execute training with the updated configuration
   > The message `Finish loading pretrained model from: xxx.pdparams` indicates successful model loading

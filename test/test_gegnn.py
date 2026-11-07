import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import paddle
import pandas as pd
from omegaconf import OmegaConf
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

from ppmat.datasets.build_solvent import BuildBinaryMixture
from ppmat.datasets.build_solvent import BuildSolvent
from ppmat.datasets.collate_fn import DefaultCollator
from ppmat.datasets.gegnn_dataset import BinaryActivityDataset
from ppmat.models import build_graph_converter
from ppmat.models import build_model
from ppmat.models.gegnn import GEGNNBinary
from ppmat.predictor.property_predictor import PropertyPredictor
from ppmat.vocab import build_vocab

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def solvent_builder():
    config = OmegaConf.to_container(
        OmegaConf.load(
            os.path.join(
                BASE_DIR,
                "property_prediction",
                "configs",
                "gegnn",
                "gegnn_binary_activity.yaml",
            )
        ),
        resolve=True,
    )
    return BuildSolvent(
        build_graph_converter(
            config["Predict"]["graph_converter"],
            vocab=build_vocab(config["Vocabulary"]),
        )
    )


class TestGEGNNBinary(unittest.TestCase):
    @staticmethod
    def _synthetic_sample(composition, gamma=None):
        builder = solvent_builder()
        return BuildBinaryMixture()(builder("CCO"), builder("O"), composition, gamma)

    @classmethod
    def _synthetic_batch(cls):
        samples = [
            cls._synthetic_sample(composition, gamma)
            for composition, gamma in ((0.25, [0.10, -0.05]), (0.75, [0.20, 0.15]))
        ]
        return DefaultCollator()(samples)

    @staticmethod
    def _new_model():
        return GEGNNBinary(in_dim=74, hidden_dim=8, n_classes=1)

    def test_reference_and_legacy_edge_dimensions(self):
        batch = self._synthetic_batch()
        for edge_features in (1, 3):
            model = GEGNNBinary(in_dim=74, hidden_dim=8, edge_in_feats=edge_features)
            self.assertEqual(model.predict(batch)["gamma"].shape, [2, 2])
        with self.assertRaisesRegex(ValueError, "edge_in_feats"):
            GEGNNBinary(edge_in_feats=2)
        with self.assertRaisesRegex(ValueError, "dropout"):
            GEGNNBinary(mlp_dropout_rate=0.1)

    def test_checkpoint_edge_dimension_mismatch_is_not_silently_ignored(self):
        reference = self._new_model()
        legacy = GEGNNBinary(in_dim=74, hidden_dim=8, edge_in_feats=3)
        with self.assertRaisesRegex(ValueError, "Checkpoint edge feature dimension"):
            reference.set_state_dict(legacy.state_dict())

    def test_instantiation(self):
        self.assertIsNotNone(self._new_model())

    def test_excess_gibbs_formula(self):
        excess_gibbs_energy = paddle.to_tensor([[0.5], [1.0]], dtype="float32")
        x1 = paddle.to_tensor([[0.25], [0.75]], dtype="float32")
        derivative = paddle.to_tensor([[2.0], [-0.5]], dtype="float32")

        gamma = GEGNNBinary.activity_coefficients(excess_gibbs_energy, x1, derivative)

        np.testing.assert_allclose(
            gamma.numpy(),
            np.array([[2.0, 0.0], [0.875, 1.375]], dtype=np.float32),
            rtol=0.0,
            atol=1e-6,
        )

    def test_label_free_collation_for_prediction(self):
        batch = DefaultCollator()([self._synthetic_sample(0.5)])

        self.assertNotIn("gamma", batch)
        prediction = self._new_model().predict(batch)
        self.assertEqual(prediction["gamma"].shape, [1, 2])

    def test_forward_loss_and_predict_on_synthetic_batch(self):
        batch = self._synthetic_batch()
        model = self._new_model()

        output = model(batch)
        gamma = output["pred_dict"]["gamma"]
        loss = output["loss_dict"]["loss"]
        model.eval()
        prediction = model.predict(batch)

        self.assertEqual(gamma.shape, [2, 2])
        self.assertIn("supervised_loss", output["loss_dict"])
        self.assertNotIn("gibbs_duhem_loss", output["loss_dict"])
        self.assertTrue(np.isfinite(loss.numpy()).all())
        self.assertIn("excess_gibbs_energy", output["pred_dict"])
        self.assertIn("d_excess_gibbs_energy_dx1", output["pred_dict"])
        self.assertIn("gamma", prediction)
        self.assertEqual(prediction["gamma"].shape, [2, 2])
        self.assertTrue(np.isfinite(prediction["gamma"].numpy()).all())

    def test_optimized_prediction_matches_full_autograd(self):
        batch = self._synthetic_batch()
        model = self._new_model()
        model.eval()

        reference = model._predict_with_derivative(batch, create_graph=True)
        optimized = model._predict_head_gradient(batch)
        for expected, actual in zip(reference, optimized):
            np.testing.assert_allclose(
                actual.numpy(), expected.numpy(), rtol=3e-5, atol=3e-6
            )

    def test_two_step_adam_training_is_deterministic(self):
        batch = self._synthetic_batch()
        paddle.seed(2025)
        first_model = self._new_model()
        paddle.seed(2025)
        second_model = self._new_model()
        first_optimizer = paddle.optimizer.Adam(
            learning_rate=1e-3, parameters=first_model.parameters()
        )
        second_optimizer = paddle.optimizer.Adam(
            learning_rate=1e-3, parameters=second_model.parameters()
        )

        for _ in range(2):
            first_result = first_model(batch)
            second_result = second_model(batch)
            np.testing.assert_allclose(
                first_result["pred_dict"]["gamma"].numpy(),
                second_result["pred_dict"]["gamma"].numpy(),
                rtol=0.0,
                atol=1e-4,
            )

            first_loss = first_result["loss_dict"]["loss"]
            first_loss.backward()
            self.assertTrue(
                any(
                    parameter.grad is not None for parameter in first_model.parameters()
                )
            )
            first_optimizer.step()
            first_optimizer.clear_grad()

            second_loss = second_result["loss_dict"]["loss"]
            second_loss.backward()
            second_optimizer.step()
            second_optimizer.clear_grad()

            np.testing.assert_allclose(
                first_loss.numpy(), second_loss.numpy(), rtol=0.0, atol=1e-6
            )
            for first_parameter, second_parameter in zip(
                first_model.parameters(), second_model.parameters()
            ):
                np.testing.assert_allclose(
                    first_parameter.numpy(),
                    second_parameter.numpy(),
                    rtol=0.0,
                    atol=1e-6,
                )

    def test_predictions_are_independent_of_batch_companions(self):
        for hidden_dim in (8, 256):
            paddle.seed(2021)
            model = GEGNNBinary(hidden_dim=hidden_dim, mlp_activation="softplus")
            model.eval()
            builder = solvent_builder()
            mixture = BuildBinaryMixture()

            def predict(inputs):
                samples = [mixture(builder(a), builder(b), x) for a, b, x in inputs]
                return model.predict(DefaultCollator()(samples))["gamma"].numpy()

            target = ("CCO", "O", 0.35)
            companion = ("CC(=O)C", "c1ccncc1", 0.6)
            replacement = ("CCN", "CC(=O)O", 0.2)
            alone = predict([target])[0]
            for inputs, index in [
                ([target, companion], 0),
                ([target, replacement], 0),
                ([companion, target], 1),
            ]:
                np.testing.assert_allclose(
                    predict(inputs)[index],
                    alone,
                    rtol=0.0,
                    atol=1e-6 if hidden_dim == 8 else 1e-4,
                )

    def test_build_model_from_config(self):
        config_path = os.path.join(
            BASE_DIR,
            "property_prediction",
            "configs",
            "gegnn",
            "gegnn_binary_activity.yaml",
        )
        config = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
        self.assertIsInstance(build_model(config["Model"]), GEGNNBinary)

    def test_gegnn_config_contract(self):
        config_path = os.path.join(
            BASE_DIR,
            "property_prediction",
            "configs",
            "gegnn",
            "gegnn_binary_activity.yaml",
        )
        config = OmegaConf.load(config_path)
        init_params = config.Model["__init_params__"]

        self.assertEqual(config.Model["__class_name__"], "GEGNNBinary")
        self.assertNotIn("Loss", config)
        self.assertEqual(init_params["in_dim"], 74)
        scheduler = config.Optimizer["__init_params__"]["lr"]
        self.assertEqual(scheduler["__class_name__"], "ReduceOnPlateau")
        scheduler_params = scheduler["__init_params__"]
        self.assertEqual(scheduler_params["learning_rate"], 1e-3)
        self.assertEqual(scheduler_params["factor"], 0.8)
        self.assertEqual(scheduler_params["patience"], 3)
        self.assertEqual(scheduler_params["min_lr"], 1e-7)
        self.assertEqual(scheduler_params["indicator"], "train_loss")
        self.assertEqual(scheduler_params["indicator_name"], "loss")
        self.assertNotIn("pinn_lambda", init_params)
        self.assertNotIn("finite_difference_step", init_params)
        self.assertFalse(config.Trainer["eval_with_no_grad"])
        self.assertEqual(
            OmegaConf.to_container(config, resolve=False)["Dataset"]["train"][
                "dataset"
            ]["__init_params__"]["path"],
            "./data/binary_activity/fold_0/train.csv",
        )
        self.assertIn("val", config.Dataset)
        self.assertEqual(config.Model["__init_params__"].edge_in_feats, 1)
        self.assertEqual(config.Trainer.best_metric_indicator, "eval_metric")
        self.assertNotIn("collate_params", config.Dataset["train"]["loader"])


class TestAtomFeaturization(unittest.TestCase):
    def test_molecular_graph_converter(self):
        graph = solvent_builder()("CCO")["graph"]
        self.assertIsNotNone(graph)
        self.assertEqual(graph.num_nodes, 3)
        self.assertIn("h", graph.node_feat)
        self.assertEqual(graph.node_feat["h"].shape[1], 74)

    def test_reference_feature_values(self):
        builder = solvent_builder()
        vocab = builder.graph_converter.vocab

        def one_hot(value, tokens):
            return [int(value == token) for token in tokens]

        for smiles in [
            "CCO",
            "O",
            "N",
            "c1ccncc1",
            "[NH4+]",
            "[CH3]",
            "P(F)(F)(F)(F)F",
        ]:
            with self.subTest(smiles=smiles):
                molecule = Chem.MolFromSmiles(smiles)
                expected = []
                for atom in molecule.GetAtoms():
                    expected.append(
                        one_hot(atom.GetSymbol(), vocab["atom"]["tokens"])
                        + one_hot(atom.GetDegree(), list(range(11)))
                        + one_hot(atom.GetImplicitValence(), list(range(7)))
                        + [atom.GetFormalCharge(), atom.GetNumRadicalElectrons()]
                        + one_hot(
                            atom.GetHybridization(),
                            [
                                Chem.HybridizationType.SP,
                                Chem.HybridizationType.SP2,
                                Chem.HybridizationType.SP3,
                                Chem.HybridizationType.SP3D,
                                Chem.HybridizationType.SP3D2,
                            ],
                        )
                        + [int(atom.GetIsAromatic())]
                        + one_hot(atom.GetTotalNumHs(), list(range(5)))
                    )
                entry = builder(smiles)
                np.testing.assert_array_equal(
                    entry["graph"].node_feat["h"], np.asarray(expected, dtype="float32")
                )
                self.assertEqual(entry["hba"], rdMolDescriptors.CalcNumHBA(molecule))
                self.assertEqual(entry["hbd"], rdMolDescriptors.CalcNumHBD(molecule))

    def test_default_converter_feature_key(self):
        builder = solvent_builder()
        graph = build_graph_converter(
            {
                "__class_name__": "MolecularGraphConverter",
                "__init_params__": {"vocab": builder.graph_converter.vocab},
            }
        )(Chem.MolFromSmiles("CCO"))
        self.assertIn("feat", graph.node_feat)
        self.assertNotIn("h", graph.node_feat)


class TestBinaryActivityPipeline(unittest.TestCase):
    def test_predictor_uses_shared_mixture_pipeline(self):
        builder = solvent_builder()
        predictor = object.__new__(PropertyPredictor)
        predictor._run_model = lambda data: data
        data = predictor.from_mixture(builder("CCO"), builder("O"), 0.5)
        self.assertEqual(data["x1"].shape, (1, 1))
        self.assertEqual(data["g1"].num_graph, 1)
        self.assertNotIn("gamma", data)

    def test_interaction_batch_keeps_samples_separate(self):
        batch = TestGEGNNBinary._synthetic_batch()
        batch["inter_hb"] = np.asarray([[10], [20]], dtype="float32")
        batch["intra_hb1"] = np.asarray([[1], [2]], dtype="float32")
        batch["intra_hb2"] = np.asarray([[3], [4]], dtype="float32")
        hg1 = paddle.to_tensor([[1.0, 2.0], [3.0, 4.0]])
        hg2 = paddle.to_tensor([[5.0, 6.0], [7.0, 8.0]])

        def capture(graph, node_features, edge_features):
            np.testing.assert_array_equal(
                graph.edges.numpy(),
                [[0, 1], [1, 0], [0, 0], [1, 1], [2, 3], [3, 2], [2, 2], [3, 3]],
            )
            np.testing.assert_array_equal(
                node_features.numpy(), [[1, 2], [5, 6], [3, 4], [7, 8]]
            )
            np.testing.assert_array_equal(
                edge_features.numpy(),
                [
                    [10, 1, 3],
                    [10, 3, 1],
                    [10, 1, 1],
                    [10, 3, 3],
                    [20, 2, 4],
                    [20, 4, 2],
                    [20, 2, 2],
                    [20, 4, 4],
                ],
            )
            return node_features

        model = SimpleNamespace(
            edge_in_feats=3,
            conv1=lambda graph, features: features,
            conv2=lambda graph, features: features,
            _as_column=GEGNNBinary._as_column,
            global_conv1=capture,
        )
        with patch("ppmat.models.gegnn._segment_mean", side_effect=[hg1, hg2]):
            output = GEGNNBinary._component_features(model, batch, 2)
        np.testing.assert_array_equal(output.numpy(), paddle.concat([hg1, hg2]).numpy())

    def test_mixture_labels_and_invalid_composition(self):
        builder = solvent_builder()
        first, second = builder("CCO"), builder("O")
        sample = BuildBinaryMixture()(first, second, 0.25, [0.1, -0.2])
        np.testing.assert_array_equal(
            sample["gamma"], np.asarray([0.1, -0.2], dtype="float32")
        )
        self.assertEqual(sample["x1"].shape, (1,))
        self.assertNotIn("gamma", BuildBinaryMixture()(first, second, 0.25))
        for composition in [-0.1, 1.1, float("nan")]:
            with self.assertRaises(ValueError):
                BuildBinaryMixture()(first, second, composition)

    def test_dataset_cache_reuse_and_invalidation(self):
        config = OmegaConf.to_container(
            OmegaConf.load(
                os.path.join(
                    BASE_DIR,
                    "property_prediction",
                    "configs",
                    "gegnn",
                    "gegnn_binary_activity.yaml",
                )
            ),
            resolve=True,
        )
        params = config["Dataset"]["train"]["dataset"]["__init_params__"]
        with tempfile.TemporaryDirectory() as root:
            params["path"] = os.path.join(root, "train.csv")
            params["cache_path"] = os.path.join(root, "cache")
            params["solvent_list_path"] = os.path.join(root, "solvent_list.csv")
            pd.DataFrame(
                {
                    "solv1": [1],
                    "solv2": [2],
                    "solv1_x": [0.25],
                    "solv1_gamma": [0.1],
                    "solv2_gamma": [-0.2],
                }
            ).to_csv(params["path"], index=False)
            pd.DataFrame({"solvent_id": [1, 2], "smiles_can": ["CCO", "O"]}).to_csv(
                os.path.join(root, "solvent_list.csv"), index=False
            )
            first = BinaryActivityDataset(**params)
            original_nodes = first[0]["g1"].num_nodes
            with patch(
                "ppmat.datasets.gegnn_dataset.BuildSolvent",
                side_effect=AssertionError("cache should be reused"),
            ):
                cached = BinaryActivityDataset(**params)
            np.testing.assert_array_equal(
                first[0]["g1"].node_feat["h"], cached[0]["g1"].node_feat["h"]
            )
            params["build_molecule_cfg"] = {"format": "smiles", "add_hs": True}
            rebuilt = BinaryActivityDataset(**params)
            self.assertGreater(rebuilt[0]["g1"].num_nodes, original_nodes)
            os.remove(rebuilt.graphs[0])
            repaired = BinaryActivityDataset(**params)
            self.assertTrue(os.path.exists(repaired.graphs[0]))


class TestMixtureCSVPrediction(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.TemporaryDirectory()
        self.addCleanup(self.root.cleanup)
        self.config_path = os.path.join(self.root.name, "gegnn.yaml")
        self.checkpoint_path = os.path.join(self.root.name, "best.pdparams")
        self.csv_path = os.path.join(self.root.name, "mixtures.csv")
        config = OmegaConf.load(
            os.path.join(
                BASE_DIR, "property_prediction/configs/gegnn/gegnn_binary_activity.yaml"
            )
        )
        config.Model["__init_params__"].hidden_dim = 8
        OmegaConf.save(config, self.config_path)
        model = build_model(OmegaConf.to_container(config, resolve=True)["Model"])
        paddle.save(model.state_dict(), self.checkpoint_path)
        self.predictor = PropertyPredictor(
            config_path=self.config_path,
            checkpoint_path=self.checkpoint_path,
            device="cpu",
        )

    def test_csv_predictions_match_shared_pipeline(self):
        records = pd.DataFrame(
            {"smiles1": ["CCO", "CCO"], "smiles2": ["O", "CC(=O)C"], "x1": [0.5, 0.25]}
        )
        records.to_csv(self.csv_path, index=False)
        output_path = os.path.join(self.root.name, "result.csv")
        results = self.predictor.from_mixture_csv(
            self.csv_path, output_path, batch_size=2
        )
        self.assertEqual(
            list(results.columns), list(records.columns) + ["ln_gamma1", "ln_gamma2"]
        )
        builder = solvent_builder()
        for index, row in records.iterrows():
            expected = (
                self.predictor.from_mixture(
                    builder(row.smiles1), builder(row.smiles2), row.x1
                )["gamma"]
                .numpy()
                .reshape(2)
            )
            np.testing.assert_allclose(
                results.loc[index, ["ln_gamma1", "ln_gamma2"]].to_numpy(dtype=float),
                expected,
                rtol=0.0,
                atol=1e-6,
            )
        np.testing.assert_allclose(
            pd.read_csv(output_path)[["ln_gamma1", "ln_gamma2"]],
            results[["ln_gamma1", "ln_gamma2"]],
            rtol=1e-6,
        )

    def test_csv_prediction_is_invariant_to_batch_size(self):
        pd.DataFrame(
            {
                "smiles1": ["CCO", "CCN", "O"],
                "smiles2": ["O", "CC(=O)C", "c1ccncc1"],
                "x1": [0.5, 0.2, 0.8],
            }
        ).to_csv(self.csv_path, index=False)
        single = self.predictor.from_mixture_csv(self.csv_path, batch_size=1)
        batch = self.predictor.from_mixture_csv(self.csv_path, batch_size=2)
        np.testing.assert_allclose(
            single[["ln_gamma1", "ln_gamma2"]],
            batch[["ln_gamma1", "ln_gamma2"]],
            rtol=0,
            atol=1e-6,
        )
        with self.assertRaisesRegex(ValueError, "batch_size"):
            self.predictor.from_mixture_csv(self.csv_path, batch_size=0)

    def test_csv_rejects_missing_columns_and_values(self):
        pd.DataFrame({"smiles1": ["O"]}).to_csv(self.csv_path, index=False)
        with self.assertRaisesRegex(ValueError, "missing columns"):
            self.predictor.from_mixture_csv(self.csv_path)
        pd.DataFrame({"smiles1": ["O"], "smiles2": [None], "x1": [0.5]}).to_csv(
            self.csv_path, index=False
        )
        with self.assertRaisesRegex(ValueError, "missing SMILES"):
            self.predictor.from_mixture_csv(self.csv_path)

    def test_csv_rejects_invalid_composition(self):
        pd.DataFrame({"smiles1": ["CCO"], "smiles2": ["O"], "x1": [1.5]}).to_csv(
            self.csv_path, index=False
        )
        with self.assertRaisesRegex(ValueError, "Mole fraction"):
            self.predictor.from_mixture_csv(self.csv_path)

    def test_cli_loads_checkpoint_directory(self):
        pd.DataFrame({"smiles1": ["CCO"], "smiles2": ["O"], "x1": [0.5]}).to_csv(
            self.csv_path, index=False
        )
        output_path = os.path.join(self.root.name, "cli_results")
        checkpoint_dir = os.path.join(self.root.name, "package", "checkpoints")
        os.makedirs(checkpoint_dir)
        shutil.copyfile(
            self.checkpoint_path, os.path.join(checkpoint_dir, "best.pdparams")
        )
        completed = subprocess.run(
            [
                sys.executable,
                "property_prediction/predict.py",
                "--config_path",
                self.config_path,
                "--checkpoint_path",
                os.path.join(self.root.name, "package"),
                "--input_format",
                "mixture_csv",
                "--input_path",
                self.csv_path,
                "--output_path",
                output_path,
                "--device",
                "cpu",
            ],
            cwd=BASE_DIR,
            env={**os.environ, "PYTHONPATH": BASE_DIR},
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        results = pd.read_csv(os.path.join(output_path, "result.csv"))
        self.assertEqual(len(results), 1)
        self.assertTrue(np.isfinite(results[["ln_gamma1", "ln_gamma2"]]).all().all())


if __name__ == "__main__":
    unittest.main()
